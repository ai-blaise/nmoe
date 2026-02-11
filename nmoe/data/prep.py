"""
Data preprocessing pipeline for nmoe.

Supports:
- Local execution (DirectRunner)
- Apache Beam for distributed execution (DataflowRunner, etc.)

Pipeline stages:
1. Source: Stream documents from HuggingFace/JSONL/Arrow
2. Normalize: NFC normalization, clean text
3. Tokenize: Parallel tokenization with tiktoken
4. Shard: Deterministic doc→shard assignment
5. Pack: Write .npy + .idx shards
6. Manifest: Write dataset manifest
"""
from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Callable, Optional, Tuple

import numpy as np

from .sources import DataSource, Document, HuggingFaceSource, create_source
from .sinks import ShardedWriter, ManifestInfo
from .transforms import normalize_text, tokenize

log = logging.getLogger(__name__)


@dataclass
class PrepConfig:
    """Configuration for data preprocessing pipeline."""
    # Output
    output_dir: Path
    dataset_name: str
    version: str = "v1"

    # Tokenizer
    tokenizer: str = "o200k_harmony"
    vocab_size: int = 201088  # o200k_harmony
    eos_token_id: int = 199999  # o200k_harmony

    # Sharding
    num_shards: int = 1024
    tokens_per_shard: int = 500_000_000  # ~2GB for uint32

    # Processing
    num_workers: int = 4
    batch_size: int = 1000
    dtype: type = np.uint32

    # Filtering
    min_tokens: int = 10
    max_tokens: int = 1_000_000

    # Early stopping
    max_tokens_total: int | None = None  # Stop after this many tokens (None = no limit)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


def _get_shard_id(doc_id: str, num_shards: int) -> int:
    """Deterministic shard assignment via MD5 hash."""
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % num_shards


def _tokenize_document(
    doc: Document,
    tokenizer: str,
    min_tokens: int,
    max_tokens: int,
) -> Tuple[str, List[int]] | None:
    """Normalize and tokenize a document.

    Returns (doc_id, tokens) or None if filtered out.
    """
    # Normalize text
    text = normalize_text(doc.text)
    if not text:
        return None

    # Tokenize
    tokens = tokenize(text, tokenizer)

    # Filter by length
    if len(tokens) < min_tokens:
        return None
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    return (doc.doc_id, tokens)


def _tokenize_batch(
    docs: List[Document],
    tokenizer: str,
    min_tokens: int,
    max_tokens: int,
) -> List[Tuple[str, List[int]]]:
    """Tokenize a batch of documents."""
    results = []
    for doc in docs:
        result = _tokenize_document(doc, tokenizer, min_tokens, max_tokens)
        if result is not None:
            results.append(result)
    return results


class PrepPipeline:
    """Local data preprocessing pipeline.

    For distributed execution, use PrepBeamPipeline instead.

    Example:
        source = HuggingFaceSource("HuggingFaceFW/fineweb-edu", split="train")
        config = PrepConfig(
            output_dir="/data/fineweb_edu",
            dataset_name="fineweb_edu",
        )
        pipeline = PrepPipeline(source, config)
        manifest = pipeline.run()
    """

    def __init__(self, source: DataSource, config: PrepConfig):
        self.source = source
        self.config = config

    def run(self, progress_callback: Callable[[int, int], None] | None = None) -> ManifestInfo:
        """Run the preprocessing pipeline.

        Args:
            progress_callback: Optional callback(docs_processed, tokens_processed)

        Returns:
            ManifestInfo with dataset metadata
        """
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Starting preprocessing: {self.source.name} → {cfg.output_dir}")
        log.info(f"Tokenizer: {cfg.tokenizer}, Shards: {cfg.num_shards}, Workers: {cfg.num_workers}")

        # Initialize per-shard writers
        shard_writers: Dict[int, ShardedWriter] = {}

        def get_writer(shard_id: int) -> ShardedWriter:
            if shard_id not in shard_writers:
                shard_writers[shard_id] = ShardedWriter(
                    output_dir=cfg.output_dir / f"shard_{shard_id:04d}",
                    dataset=cfg.dataset_name,
                    version=cfg.version,
                    eos_token_id=cfg.eos_token_id,
                    vocab_size=cfg.vocab_size,
                    tokenizer=cfg.tokenizer,
                    dtype=cfg.dtype,
                    tokens_per_shard=cfg.tokens_per_shard,
                    source_info={"source": self.source.name},
                )
            return shard_writers[shard_id]

        # Process documents
        docs_processed = 0
        tokens_processed = 0
        batch: List[Document] = []
        reached_limit = False

        def process_batch(batch: List[Document]) -> bool:
            """Process batch, return True if should stop (token limit reached)."""
            nonlocal docs_processed, tokens_processed

            # Tokenize batch (could parallelize further)
            results = _tokenize_batch(
                batch,
                cfg.tokenizer,
                cfg.min_tokens,
                cfg.max_tokens,
            )

            # Distribute to shards
            for doc_id, tokens in results:
                shard_id = _get_shard_id(doc_id, cfg.num_shards)
                writer = get_writer(shard_id)
                writer.add_document(tokens)
                tokens_processed += len(tokens) + 1  # +1 for EOS

                # Check early stopping
                if cfg.max_tokens_total and tokens_processed >= cfg.max_tokens_total:
                    docs_processed += 1
                    if progress_callback:
                        progress_callback(docs_processed, tokens_processed)
                    return True  # Stop

            docs_processed += len(results)

            if progress_callback:
                progress_callback(docs_processed, tokens_processed)
            return False

        # Stream and batch documents
        for doc in self.source:
            batch.append(doc)
            if len(batch) >= cfg.batch_size:
                if process_batch(batch):
                    reached_limit = True
                    break
                batch = []

        # Process remaining batch (unless we hit limit)
        if batch and not reached_limit:
            process_batch(batch)

        if reached_limit:
            log.info(f"Reached token limit ({cfg.max_tokens_total:,}), stopping early")

        log.info(f"Processed {docs_processed:,} documents, {tokens_processed:,} tokens")

        # Finalize all shard writers and collect shards with correct paths
        all_shards = []
        total_tokens = 0
        total_docs = 0
        created_at = ""
        for shard_id in sorted(shard_writers.keys()):
            manifest = shard_writers[shard_id].finalize()
            if not created_at:
                created_at = manifest.created_at
            total_tokens += manifest.total_tokens
            total_docs += manifest.total_documents
            # Prepend subdirectory to shard paths
            shard_dir = f"shard_{shard_id:04d}"
            for s in manifest.shards:
                s.path = f"{shard_dir}/{s.path}"
                s.index_path = f"{shard_dir}/{s.index_path}"
                all_shards.append(s)

        # Create combined manifest
        combined_manifest = ManifestInfo(
            dataset=cfg.dataset_name,
            version=cfg.version,
            tokenizer=cfg.tokenizer,
            vocab_size=cfg.vocab_size,
            eos_token_id=cfg.eos_token_id,
            dtype=str(np.dtype(cfg.dtype)),
            created_at=created_at,
            total_tokens=total_tokens,
            total_documents=total_docs,
            num_shards=len(all_shards),
            shards=all_shards,
            source_info={"source": self.source.name, "num_shard_dirs": len(shard_writers)},
        )

        # Write combined manifest
        import json
        manifest_path = cfg.output_dir / "manifest.json"
        tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(combined_manifest.to_dict(), f, indent=2)
        tmp_path.replace(manifest_path)

        log.info(f"Wrote manifest: {manifest_path}")
        return combined_manifest


class ParallelPrepPipeline:
    """Parallel data preprocessing pipeline using multiprocessing.

    Tokenizes documents in parallel across workers, then writes to shards.

    Example:
        source = HuggingFaceSource("HuggingFaceFW/fineweb-edu", split="train")
        config = PrepConfig(output_dir="/data/fineweb_edu", dataset_name="fineweb_edu")
        pipeline = ParallelPrepPipeline(source, config)
        manifest = pipeline.run()
    """

    def __init__(self, source: DataSource, config: PrepConfig):
        self.source = source
        self.config = config

    def run(self, progress_callback: Callable[[int, int], None] | None = None) -> ManifestInfo:
        """Run the preprocessing pipeline with parallel tokenization."""
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Starting parallel preprocessing: {self.source.name} → {cfg.output_dir}")
        log.info(f"Tokenizer: {cfg.tokenizer}, Workers: {cfg.num_workers}")

        # Collect all tokenized documents grouped by shard
        shard_docs: Dict[int, List[Tuple[str, List[int]]]] = {i: [] for i in range(cfg.num_shards)}

        docs_processed = 0
        tokens_processed = 0

        # Use ProcessPoolExecutor for CPU-bound tokenization
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
            batch: List[Document] = []

            def submit_batch(batch: List[Document]):
                return executor.submit(
                    _tokenize_batch,
                    batch,
                    cfg.tokenizer,
                    cfg.min_tokens,
                    cfg.max_tokens,
                )

            futures = []
            for doc in self.source:
                batch.append(doc)
                if len(batch) >= cfg.batch_size:
                    futures.append(submit_batch(batch))
                    batch = []

            if batch:
                futures.append(submit_batch(batch))

            # Collect results
            for future in futures:
                results = future.result()
                for doc_id, tokens in results:
                    shard_id = _get_shard_id(doc_id, cfg.num_shards)
                    shard_docs[shard_id].append((doc_id, tokens))
                    tokens_processed += len(tokens)
                docs_processed += len(results)

                if progress_callback:
                    progress_callback(docs_processed, tokens_processed)

        log.info(f"Tokenized {docs_processed:,} documents, {tokens_processed:,} tokens")

        # Write shards (could parallelize this too)
        all_shard_infos = []
        for shard_id, docs in shard_docs.items():
            if not docs:
                continue

            shard_dir = cfg.output_dir / f"shard_{shard_id:04d}"
            with ShardedWriter(
                output_dir=shard_dir,
                dataset=cfg.dataset_name,
                version=cfg.version,
                eos_token_id=cfg.eos_token_id,
                vocab_size=cfg.vocab_size,
                tokenizer=cfg.tokenizer,
                dtype=cfg.dtype,
                tokens_per_shard=cfg.tokens_per_shard,
            ) as writer:
                for doc_id, tokens in docs:
                    writer.add_document(tokens)

        # Create combined manifest
        import json
        from datetime import datetime

        # Re-read all shard manifests
        total_tokens = 0
        total_docs = 0
        all_shards = []
        for shard_dir in sorted(cfg.output_dir.glob("shard_*")):
            manifest_file = shard_dir / "manifest.json"
            if not manifest_file.exists():
                continue
            with open(manifest_file) as f:
                m = json.load(f)
            total_tokens += m["total_tokens"]
            total_docs += m["total_documents"]
            for s in m["shards"]:
                s["path"] = f"{shard_dir.name}/{s['path']}"
                s["index_path"] = f"{shard_dir.name}/{s['index_path']}"
                all_shards.append(s)

        combined_manifest = {
            "dataset": cfg.dataset_name,
            "version": cfg.version,
            "tokenizer": cfg.tokenizer,
            "vocab_size": cfg.vocab_size,
            "eos_token_id": cfg.eos_token_id,
            "dtype": str(np.dtype(cfg.dtype)),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_tokens": total_tokens,
            "total_documents": total_docs,
            "num_shards": len(all_shards),
            "shards": all_shards,
            "source_info": {"source": self.source.name},
        }

        manifest_path = cfg.output_dir / "manifest.json"
        tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(combined_manifest, f, indent=2)
        tmp_path.replace(manifest_path)

        log.info(f"Wrote manifest: {manifest_path}")
        return ManifestInfo.from_dict(combined_manifest)


try:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions

    BEAM_AVAILABLE = True

    class TokenizeDoFn(beam.DoFn):
        """Beam DoFn for tokenization."""

        def __init__(self, tokenizer: str, min_tokens: int, max_tokens: int):
            self.tokenizer = tokenizer
            self.min_tokens = min_tokens
            self.max_tokens = max_tokens

        def process(self, element: Dict[str, Any]):
            doc_id = element.get("id", element.get("doc_id", ""))
            text = element.get("text", "")

            # Normalize
            text = normalize_text(text)
            if not text:
                return

            # Tokenize
            tokens = tokenize(text, self.tokenizer)

            # Filter
            if len(tokens) < self.min_tokens:
                return
            if len(tokens) > self.max_tokens:
                tokens = tokens[:self.max_tokens]

            yield {"doc_id": doc_id, "tokens": tokens}

    class AssignShardDoFn(beam.DoFn):
        """Beam DoFn to assign shard IDs."""

        def __init__(self, num_shards: int):
            self.num_shards = num_shards

        def process(self, element: Dict[str, Any]):
            doc_id = element["doc_id"]
            shard_id = _get_shard_id(doc_id, self.num_shards)
            yield (shard_id, element)

    class WriteShardDoFn(beam.DoFn):
        """Beam DoFn to write shards."""

        def __init__(
            self,
            output_dir: str,
            dataset: str,
            version: str,
            eos_token_id: int,
            vocab_size: int,
            tokenizer: str,
        ):
            self.output_dir = output_dir
            self.dataset = dataset
            self.version = version
            self.eos_token_id = eos_token_id
            self.vocab_size = vocab_size
            self.tokenizer = tokenizer

        def process(self, element: Tuple[int, List[Dict[str, Any]]]):
            shard_id, docs = element

            shard_dir = Path(self.output_dir) / f"shard_{shard_id:04d}"
            with ShardedWriter(
                output_dir=shard_dir,
                dataset=self.dataset,
                version=self.version,
                eos_token_id=self.eos_token_id,
                vocab_size=self.vocab_size,
                tokenizer=self.tokenizer,
            ) as writer:
                for doc in docs:
                    writer.add_document(doc["tokens"])

            yield shard_id

    def run_beam_pipeline(
        source_config: Dict[str, Any],
        prep_config: PrepConfig,
        pipeline_options: PipelineOptions | None = None,
    ) -> None:
        """Run preprocessing as Apache Beam pipeline.

        Args:
            source_config: Config for create_source()
            prep_config: PrepConfig for output settings
            pipeline_options: Beam pipeline options (runner, etc.)
        """
        cfg = prep_config

        with beam.Pipeline(options=pipeline_options) as p:
            # Read from HuggingFace
            source = create_source(**source_config)

            docs = (
                p
                | "CreateDocs" >> beam.Create(list(source))
                | "ToDict" >> beam.Map(lambda d: {"doc_id": d.doc_id, "text": d.text})
            )

            # Tokenize
            tokenized = docs | "Tokenize" >> beam.ParDo(
                TokenizeDoFn(cfg.tokenizer, cfg.min_tokens, cfg.max_tokens)
            )

            # Assign shards and group
            sharded = (
                tokenized
                | "AssignShard" >> beam.ParDo(AssignShardDoFn(cfg.num_shards))
                | "GroupByShard" >> beam.GroupByKey()
            )

            # Write shards
            _ = sharded | "WriteShards" >> beam.ParDo(
                WriteShardDoFn(
                    str(cfg.output_dir),
                    cfg.dataset_name,
                    cfg.version,
                    cfg.eos_token_id,
                    cfg.vocab_size,
                    cfg.tokenizer,
                )
            )

except ImportError:
    BEAM_AVAILABLE = False
