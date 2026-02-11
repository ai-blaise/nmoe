"""
Shard writers for nmoe data preprocessing.

Writes tokenized documents to:
- .npy shards (uint32 token arrays)
- .idx index files (document boundaries)
- manifest.json (dataset metadata)

Follows OLMo-core patterns with atomic writes and SHA256 checksums.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from .index import IndexWriter, index_path_for_shard


@dataclass
class ShardInfo:
    """Metadata for a single shard."""
    path: str
    index_path: str
    num_tokens: int
    num_documents: int
    checksum: str  # SHA256 of .npy file


@dataclass
class ManifestInfo:
    """Dataset manifest with full inventory."""
    dataset: str
    version: str
    tokenizer: str
    vocab_size: int
    eos_token_id: int
    dtype: str
    created_at: str
    total_tokens: int
    total_documents: int
    num_shards: int
    shards: List[ShardInfo]
    source_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "dtype": self.dtype,
            "created_at": self.created_at,
            "total_tokens": self.total_tokens,
            "total_documents": self.total_documents,
            "num_shards": self.num_shards,
            "shards": [
                {
                    "path": s.path,
                    "index_path": s.index_path,
                    "num_tokens": s.num_tokens,
                    "num_documents": s.num_documents,
                    "checksum": s.checksum,
                }
                for s in self.shards
            ],
            "source_info": self.source_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestInfo":
        shards = [
            ShardInfo(
                path=s["path"],
                index_path=s["index_path"],
                num_tokens=s["num_tokens"],
                num_documents=s["num_documents"],
                checksum=s["checksum"],
            )
            for s in data["shards"]
        ]
        return cls(
            dataset=data["dataset"],
            version=data["version"],
            tokenizer=data["tokenizer"],
            vocab_size=data["vocab_size"],
            eos_token_id=data["eos_token_id"],
            dtype=data["dtype"],
            created_at=data["created_at"],
            total_tokens=data["total_tokens"],
            total_documents=data["total_documents"],
            num_shards=data["num_shards"],
            shards=shards,
            source_info=data.get("source_info", {}),
        )


def _sha256_file(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_shard_name(dataset: str, version: str, shard_idx: int) -> str:
    """Generate canonical shard filename."""
    return f"{dataset}-{version}-shard-{shard_idx:06d}.npy"


class ShardWriter:
    """Write documents to a single shard with index.

    True streaming: writes tokens to disk immediately on add_document().
    Only index entries are kept in memory (8 bytes per doc).

    Usage:
        writer = ShardWriter(output_dir, "fineweb", "v1", shard_idx=0, eos_token_id=100257)
        for tokens in tokenized_docs:
            writer.add_document(tokens)
        info = writer.finalize()
    """

    def __init__(
        self,
        output_dir: Path,
        dataset: str,
        version: str,
        shard_idx: int,
        eos_token_id: int,
        dtype: type = np.uint32,
        max_tokens: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset
        self.version = version
        self.shard_idx = shard_idx
        self.eos_token_id = eos_token_id
        self.dtype = np.dtype(dtype)
        self.max_tokens = max_tokens

        self._shard_name = _make_shard_name(dataset, version, shard_idx)
        self._shard_path = self.output_dir / self._shard_name
        self._index_path = index_path_for_shard(self._shard_path)

        # Stream to a temp .bin file, convert to .npy at finalize
        self._tmp_bin_path = self._shard_path.with_suffix(".bin.tmp")
        self._bin_file = open(self._tmp_bin_path, "wb")
        self._current_offset = 0
        self._num_documents = 0
        self._index_writer = IndexWriter(self._index_path)
        self._hasher = hashlib.sha256()

    @property
    def num_tokens(self) -> int:
        return self._current_offset

    @property
    def num_documents(self) -> int:
        return self._num_documents

    def can_fit(self, num_tokens: int) -> bool:
        """Check if shard can fit more tokens."""
        if self.max_tokens is None:
            return True
        return self._current_offset + num_tokens <= self.max_tokens

    def add_document(self, tokens: List[int] | np.ndarray, append_eos: bool = True) -> bool:
        """Add a tokenized document. Writes immediately to disk.

        Args:
            tokens: Token array
            append_eos: Whether to append EOS token (default: True for backward compat)

        Returns True if added, False if shard is full.
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=self.dtype)
        else:
            tokens = tokens.astype(self.dtype)

        # Optionally append EOS
        if append_eos:
            eos_arr = np.array([self.eos_token_id], dtype=self.dtype)
            doc_len = len(tokens) + 1
        else:
            eos_arr = None
            doc_len = len(tokens)

        if not self.can_fit(doc_len):
            return False

        # Record document boundary
        start_idx = self._current_offset
        end_idx = start_idx + doc_len
        self._index_writer.add_document(start_idx, end_idx)

        # Write immediately to disk
        token_bytes = tokens.tobytes()
        self._bin_file.write(token_bytes)
        self._hasher.update(token_bytes)

        if append_eos:
            eos_bytes = eos_arr.tobytes()
            self._bin_file.write(eos_bytes)
            self._hasher.update(eos_bytes)

        self._current_offset = end_idx
        self._num_documents += 1
        return True

    def finalize(self) -> ShardInfo:
        """Finalize shard: close bin file, convert to .npy, write index."""
        self._bin_file.close()

        if self._num_documents == 0:
            # Clean up empty shard
            self._tmp_bin_path.unlink(missing_ok=True)
            raise ValueError("Cannot finalize empty shard")

        # Convert raw binary to .npy format
        # Read back and save as proper numpy array
        raw_data = np.fromfile(self._tmp_bin_path, dtype=self.dtype)
        np.save(self._shard_path.with_suffix(""), raw_data)  # np.save adds .npy

        # Clean up temp file
        self._tmp_bin_path.unlink()

        # Write .idx
        self._index_writer.finalize()

        # Compute checksum of final .npy file
        checksum = _sha256_file(self._shard_path)

        return ShardInfo(
            path=self._shard_name,
            index_path=self._index_path.name,
            num_tokens=self._current_offset,
            num_documents=self._num_documents,
            checksum=checksum,
        )

    def __del__(self):
        """Cleanup if not properly finalized."""
        if hasattr(self, "_bin_file") and not self._bin_file.closed:
            self._bin_file.close()
        if hasattr(self, "_tmp_bin_path") and self._tmp_bin_path.exists():
            self._tmp_bin_path.unlink(missing_ok=True)


class ShardedWriter:
    """Write documents across multiple shards with automatic rotation.

    Usage:
        with ShardedWriter(output_dir, "fineweb", "v1", ...) as writer:
            for tokens in tokenized_docs:
                writer.add_document(tokens)
        manifest = writer.manifest
    """

    def __init__(
        self,
        output_dir: Path,
        dataset: str,
        version: str,
        eos_token_id: int,
        vocab_size: int,
        tokenizer: str,
        dtype: type = np.uint32,
        tokens_per_shard: int = 500_000_000,  # ~2GB per shard for uint32
        source_info: Dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset
        self.version = version
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.tokens_per_shard = tokens_per_shard
        self.source_info = source_info or {}

        self._shard_infos: List[ShardInfo] = []
        self._current_shard: ShardWriter | None = None
        self._shard_idx = 0
        self._total_tokens = 0
        self._total_docs = 0

    def _new_shard(self) -> ShardWriter:
        """Create a new shard writer."""
        writer = ShardWriter(
            output_dir=self.output_dir,
            dataset=self.dataset,
            version=self.version,
            shard_idx=self._shard_idx,
            eos_token_id=self.eos_token_id,
            dtype=self.dtype,
            max_tokens=self.tokens_per_shard,
        )
        self._shard_idx += 1
        return writer

    def _rotate_shard(self) -> None:
        """Finalize current shard and start a new one."""
        if self._current_shard and self._current_shard.num_documents > 0:
            info = self._current_shard.finalize()
            self._shard_infos.append(info)
            self._total_tokens += info.num_tokens
            self._total_docs += info.num_documents
        self._current_shard = self._new_shard()

    def add_document(self, tokens: List[int] | np.ndarray, append_eos: bool = True) -> None:
        """Add a tokenized document, rotating shards as needed."""
        if self._current_shard is None:
            self._current_shard = self._new_shard()

        # Try to add to current shard
        if not self._current_shard.add_document(tokens, append_eos=append_eos):
            # Shard is full, rotate
            self._rotate_shard()
            # Add to new shard (should always succeed)
            if not self._current_shard.add_document(tokens, append_eos=append_eos):
                raise ValueError(f"Document too large for shard: {len(tokens)} tokens")

    def finalize(self) -> ManifestInfo:
        """Finalize all shards and write manifest."""
        # Finalize current shard
        if self._current_shard and self._current_shard.num_documents > 0:
            info = self._current_shard.finalize()
            self._shard_infos.append(info)
            self._total_tokens += info.num_tokens
            self._total_docs += info.num_documents

        # Create manifest
        manifest = ManifestInfo(
            dataset=self.dataset,
            version=self.version,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            eos_token_id=self.eos_token_id,
            dtype=str(np.dtype(self.dtype)),
            created_at=datetime.utcnow().isoformat() + "Z",
            total_tokens=self._total_tokens,
            total_documents=self._total_docs,
            num_shards=len(self._shard_infos),
            shards=self._shard_infos,
            source_info=self.source_info,
        )

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        tmp_path.replace(manifest_path)

        return manifest

    def __enter__(self) -> "ShardedWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.finalize()


def load_manifest(path: Path) -> ManifestInfo:
    """Load manifest from JSON file."""
    with open(path, "r") as f:
        return ManifestInfo.from_dict(json.load(f))


def verify_manifest(manifest_path: Path, check_checksums: bool = True) -> Tuple[bool, List[str]]:
    """Verify manifest integrity.

    Returns (is_valid, error_messages).
    """
    errors = []
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent

    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        return False, [f"Failed to load manifest: {e}"]

    # Check all shards exist
    for shard in manifest.shards:
        shard_path = base_dir / shard.path
        idx_path = base_dir / shard.index_path

        if not shard_path.exists():
            errors.append(f"Missing shard: {shard.path}")
            continue

        if not idx_path.exists():
            errors.append(f"Missing index: {shard.index_path}")

        if check_checksums:
            actual_checksum = _sha256_file(shard_path)
            if actual_checksum != shard.checksum:
                errors.append(f"Checksum mismatch for {shard.path}: expected {shard.checksum}, got {actual_checksum}")

    return len(errors) == 0, errors
