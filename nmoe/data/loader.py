"""
Data loaders for nmoe training.

Primary: DeterministicLoader (Global Stream, Local Slice with SWRR)

build_loader() is the main entry point.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .dataset import Cursor, NumpyFSLDataset
from .mixture import MixturePlan, StagePlan, SourcePlan, resolve_plan, populate_paths


@dataclass
class _SourceState:
    dataset: NumpyFSLDataset
    cursor: Cursor = field(default_factory=Cursor)
    emitted_sequences: int = 0


class DeterministicLoader:
    def __init__(
        self,
        *,
        plan: MixturePlan,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        tokens_per_update: int,
        device: str = "cuda",
        prefetch_depth: int = 0,
        warmup_bytes: int = 512 * 1024 * 1024,
    ):
        self.plan = plan
        self.dp_world_size = int(dp_world_size)
        self.dp_rank = int(dp_rank)
        self.seq_len = int(seq_len)
        self.tpu = int(tokens_per_update)
        self.device = device
        self.prefetch_depth = max(0, int(prefetch_depth))

        # Build stage/source state; paths are empty in initial plan — caller should fill when available
        self._stages: List[StagePlan] = plan.stages
        self._acc: List[int] = []  # filled per stage
        self._w_sum: int = 0
        self._stage_index: int = 0
        self._global_seq_idx: int = 0
        self._src_state: Dict[str, _SourceState] = {}

        self._init_stage(0)

        # Compute per-step slicing (stable across steps)
        self._seqs_per_step = max(1, self.tpu // self.seq_len)
        q, r = divmod(self._seqs_per_step, self.dp_world_size)
        self._mine = q + (1 if self.dp_rank < r else 0)
        self._start_off = self.dp_rank * q + min(self.dp_rank, r)

        # Warmup: touch a slice of each dataset
        for ss in self._src_state.values():
            if ss.dataset is not None:
                try:
                    ss.dataset.warmup(bytes_to_read=warmup_bytes)
                except Exception:
                    pass

        # Prefetcher (optional)
        self._q: "queue.Queue[torch.Tensor]" = queue.Queue(maxsize=(self.prefetch_depth * max(1, self._mine)))
        self._stop_evt = threading.Event()
        self._producer: Optional[threading.Thread] = None
        if self.prefetch_depth > 0:
            self._producer = threading.Thread(target=self._run_producer, name="loader-producer", daemon=True)
            self._producer.start()

    def _init_stage(self, idx: int) -> None:
        self._stage_index = idx
        st = self._stages[idx]
        self._acc = [0 for _ in st.sources]
        self._w_sum = sum(sp.weight_fp for sp in st.sources) or 1
        self._src_state.clear()
        # Initialize datasets with any available paths (may be empty; caller should set later)
        for sp in st.sources:
            paths = sp.paths
            # If paths are missing, create an empty dataset that will raise later when used
            ds = NumpyFSLDataset(paths) if paths else None  # type: ignore
            self._src_state[sp.id] = _SourceState(dataset=ds, cursor=Cursor(), emitted_sequences=0)  # type: ignore

    def _advance_stage_if_done(self):
        st = self._stages[self._stage_index]
        done = all(self._src_state[sp.id].emitted_sequences >= sp.quota_sequences for sp in st.sources)
        if done and self._stage_index + 1 < len(self._stages):
            self._init_stage(self._stage_index + 1)

    def _swr_next_source_idx(self) -> int:
        st = self._stages[self._stage_index]
        # add weights
        for i, sp in enumerate(st.sources):
            self._acc[i] += sp.weight_fp
        # select argmax
        k = int(np.argmax(self._acc))
        # subtract total
        self._acc[k] -= self._w_sum
        return k

    def _select_next_non_exhausted(self) -> int:
        """Return next source index by SWRR, skipping exhausted sources."""
        st = self._stages[self._stage_index]
        for _ in range(len(st.sources) * 2):
            idx = self._swr_next_source_idx()
            sp = st.sources[idx]
            ss = self._src_state[sp.id]
            if ss.emitted_sequences < sp.quota_sequences:
                return idx
        raise StopIteration

    def _emit_from_source_idx(self, idx: int) -> torch.Tensor:
        st = self._stages[self._stage_index]
        sp = st.sources[idx]
        ss = self._src_state[sp.id]
        if ss.dataset is None:
            raise RuntimeError(f"dataset paths for source '{sp.id}' are not set")
        arr, new_cursor = ss.dataset.next_window(ss.cursor, self.seq_len + 1)
        ss.cursor = new_cursor
        ss.emitted_sequences += 1
        self._global_seq_idx += 1
        return torch.from_numpy(arr.astype(np.int64))

    def _advance_from_source_idx(self, idx: int) -> None:
        st = self._stages[self._stage_index]
        sp = st.sources[idx]
        ss = self._src_state[sp.id]
        if ss.dataset is None:
            raise RuntimeError(f"dataset paths for source '{sp.id}' are not set")
        ss.cursor = ss.dataset.advance(ss.cursor, self.seq_len + 1)
        ss.emitted_sequences += 1
        self._global_seq_idx += 1

    def _emit_one_sequence(self) -> torch.Tensor:
        self._advance_stage_if_done()
        st = self._stages[self._stage_index]
        # choose source by SWRR, but skip exhausted
        for _ in range(len(st.sources) * 2):
            idx = self._swr_next_source_idx()
            sp = st.sources[idx]
            ss = self._src_state[sp.id]
            if ss.emitted_sequences < sp.quota_sequences:
                return self._emit_from_source_idx(idx)
        raise StopIteration

    def _advance_only(self) -> None:
        """Advance the global stream by one sequence without materializing tensors."""
        self._advance_stage_if_done()
        st = self._stages[self._stage_index]
        for _ in range(len(st.sources) * 2):
            idx = self._swr_next_source_idx()
            sp = st.sources[idx]
            ss = self._src_state[sp.id]
            if ss.emitted_sequences < sp.quota_sequences:
                self._advance_from_source_idx(idx)
                return
        raise StopIteration

    def _run_producer(self) -> None:
        try:
            while not self._stop_evt.is_set():
                # Keep at most prefetch_depth steps worth of our sequences in the queue
                if self._q.qsize() >= max(1, self._mine) * self.prefetch_depth:
                    time.sleep(0.001)
                    continue
                # Skip sequences for earlier ranks
                for _ in range(self._start_off):
                    self._advance_only()
                # Produce my sequences
                for _ in range(self._mine):
                    t = self._emit_one_sequence()
                    self._q.put(t, timeout=5)
                # Advance to end of global step so next iteration starts cleanly
                # at the next global-batch boundary. This keeps per-rank loaders
                # aligned with the world-size invariant Global Stream.
                remaining = self._seqs_per_step - (self._start_off + self._mine)
                for _ in range(max(0, remaining)):
                    self._advance_only()
        except StopIteration:
            pass
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Data prefetch thread failed")

    def state_dict(self) -> Dict:
        st = self._stages[self._stage_index]
        return {
            "version": 1,
            "current_stage": st.stage_id,
            "stage_index": self._stage_index,
            "global_sequence_index": self._global_seq_idx,
            "accumulators": list(self._acc),
            "emitted_sequences": {sp.id: self._src_state[sp.id].emitted_sequences for sp in st.sources},
            "cursors": {sp.id: self._src_state[sp.id].cursor.__dict__ for sp in st.sources},
        }

    def load_state_dict(self, state: Dict) -> None:
        # basic restoration; assume same plan
        self._stage_index = int(state.get("stage_index", 0))
        self._global_seq_idx = int(state.get("global_sequence_index", 0))
        self._acc = [int(v) for v in state.get("accumulators", self._acc)]
        st = self._stages[self._stage_index]
        em = state.get("emitted_sequences", {})
        cs = state.get("cursors", {})
        for sp in st.sources:
            if sp.id in em:
                self._src_state[sp.id].emitted_sequences = int(em[sp.id])
            if sp.id in cs:
                d = cs[sp.id]
                self._src_state[sp.id].cursor = Cursor(
                    file_idx=int(d.get("file_idx", 0)),
                    pos_in_file=int(d.get("pos_in_file", 0)),
                    wrap_count=int(d.get("wrap_count", 0)),
                )

    @torch.no_grad()
    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs: List[torch.Tensor] = []
        if self.prefetch_depth > 0:
            # Consume directly from queue (blocks until available)
            for _ in range(self._mine):
                t = self._q.get()
                seqs.append(t)
        else:
            # Step-level deterministic selection that allows mid-step stage transitions.
            start = self._start_off
            end = self._start_off + self._mine
            for i in range(self._seqs_per_step):
                # Handle possible stage completion before each global selection
                self._advance_stage_if_done()
                idx = self._select_next_non_exhausted()
                if start <= i < end:
                    t = self._emit_from_source_idx(idx)
                    seqs.append(t)
                else:
                    self._advance_from_source_idx(idx)
        if not seqs:
            raise StopIteration
        batch = torch.stack(seqs)
        if self.device == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda'):
            batch = batch.pin_memory()
        batch = batch.to(self.device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    def close(self) -> None:
        if self._producer is not None:
            self._stop_evt.set()
            self._producer.join(timeout=1.0)


# =============================================================================
# build_loader: Main entry point
# =============================================================================


def _tokens_available(paths: list[str], print_fn=print) -> int:
    """Count available tokens in shard files."""
    total = 0
    for p in paths:
        try:
            if p.endswith('.npy'):
                # Use numpy to read .npy header for shape without loading data
                arr = np.load(p, mmap_mode='r')
                total += len(arr)
            else:
                size_bytes = os.path.getsize(p)
                total += size_bytes // 4  # uint32 = 4 bytes
        except FileNotFoundError:
            print_fn(f"[preflight] warning: file not found: {p}")
        except PermissionError:
            print_fn(f"[preflight] warning: permission denied: {p}")
    return total


def _preflight_check(plan: MixturePlan, seq_len: int, print_fn=print) -> None:
    """Verify quotas vs available tokens. Raises RuntimeError on failure.

    Uses proportional token allocation (matching prep-mixture) rather than
    sequence-based quotas to avoid rounding mismatches.
    """
    for st in plan.stages:
        # Compute total tokens for this stage and proportional needs per source
        # This matches how prep-mixture allocates tokens
        stage_tokens = int(st.total_tokens_b * 1_000_000_000)
        total_target = sum(sp.target_tokens for sp in st.sources)

        for sp in st.sources:
            # Proportional allocation matching prep-mixture calculation
            need = int(stage_tokens * sp.target_tokens / total_target) if total_target > 0 else 0
            have = _tokens_available(sp.paths, print_fn)
            if have < need:
                raise RuntimeError(
                    f"Preflight failed for source '{sp.id}' in stage '{st.stage_id}': "
                    f"need {need:,} tokens, have {have:,}. Check data_root or adjust flow scale."
                )


def build_loader(
    cfg,
    rank: int,
    world_size: int,
    split: str = "train",
    print_fn=print,
) -> tuple[DeterministicLoader, MixturePlan]:
    """Build data loader from config.

    Args:
        cfg: Config object with data settings (flow_mode required)
        rank: Current process rank
        world_size: Total number of processes
        split: Data split to load ("train" or "valid")
        print_fn: Function for logging (default: print)

    Returns:
        (loader, plan): DeterministicLoader instance and MixturePlan
    """
    # Fast path: stream directly from a directory of shards (no flow TOMLs)
    if getattr(cfg, 'data_path', None) and not getattr(cfg, 'flow_mode', None):
        from glob import glob
        from pathlib import Path
        shard_glob = str(Path(cfg.data_path) / '**/*.npy')
        shards = sorted(set(glob(shard_glob, recursive=True)))
        if rank == 0:
            print_fn(f"Streaming from data_path: {cfg.data_path}  shards={len(shards)}")
        if not shards:
            raise RuntimeError(f"No .npy shards found under {cfg.data_path}")

        # Compute tokens needed and sanity check against shard inventory (without loading into memory)
        tokens_per_step = cfg.batch_size * cfg.seq_len
        total_needed = cfg.steps * tokens_per_step
        # Estimate available tokens cheaply via file sizes (uint32 = 4 bytes)
        try:
            import os
            have = sum(os.path.getsize(p) for p in shards) // 4
        except Exception:
            have = 0
        if rank == 0:
            print_fn(f"Tokens needed: {total_needed:,}  available (approx): {have:,}")
        if have < total_needed:
            if rank == 0:
                print_fn("[preflight] warning: available tokens < needed; training will wrap.")

        # Synthesize a minimal MixturePlan (single stage/source) without materializing tokens
        sp = SourcePlan(id='data_path', weight_fp=1, quota_sequences=total_needed // cfg.seq_len,
                        target_tokens=total_needed, paths=shards)
        stage = StagePlan(stage_id='pretrain', total_tokens_b=total_needed / 1_000_000_000, sources=[sp])
        plan = MixturePlan(plan_id='data_path', plan_hash='0', mixture_id='data_path',
                           flow_mode='direct', sample_temperature=1.0, seq_len=cfg.seq_len, stages=[stage])

        # Build deterministic loader directly
        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=world_size,
            dp_rank=rank,
            seq_len=cfg.seq_len,
            tokens_per_update=cfg.batch_size * cfg.seq_len,
            device='cuda',
            # Prefetch breaks exact-resume unless queue contents are checkpointed.
            prefetch_depth=0,
        )
        return loader, plan

    if not cfg.flow_mode:
        raise ValueError(
            "flow_mode is required. Set flow_mode, mixture_toml, and flow_profiles_toml in your config. "
            "Example: flow_mode = 'dev', mixture_toml = 'configs/mixtures/olmo3_1025.toml', "
            "flow_profiles_toml = 'configs/flow_profiles.toml'"
        )

    print_fn(f'Using deterministic loader (Global Stream, Local Slice) [{split}].')
    mix_path = Path(cfg.mixture_toml) if cfg.mixture_toml else None
    flow_path = Path(cfg.flow_profiles_toml) if cfg.flow_profiles_toml else None
    if not mix_path or not flow_path:
        raise ValueError('flow_mode is set but mixture_toml/flow_profiles_toml are missing')

    plan = resolve_plan(
        mixture_toml=mix_path,
        flow_profiles_toml=flow_path,
        flow_section=f"flow.{cfg.flow_mode}",
        seq_len=cfg.seq_len,
        active_params_b=getattr(cfg, 'active_params_b', None),
    )

    # Compute flow token budget
    flow_tokens_b = sum(st.total_tokens_b for st in plan.stages)
    flow_tokens = int(flow_tokens_b * 1_000_000_000)
    tokens_per_step = cfg.batch_size * cfg.seq_len

    # Store flow info on cfg for banner/logging
    cfg._flow_tokens_b = flow_tokens_b
    cfg._flow_stage = plan.stages[0].stage_id if plan.stages else "pretrain"

    # Derive steps from flow if not explicitly set
    if cfg.steps <= 0:
        cfg.steps = flow_tokens // tokens_per_step
        if rank == 0:
            print_fn(f"Steps derived from flow: {cfg.steps:,} ({flow_tokens_b:.2f}B tokens / {tokens_per_step:,} tokens/step)")
    else:
        # Check if explicit steps exceeds flow budget
        training_tokens = cfg.steps * tokens_per_step
        if training_tokens > flow_tokens and rank == 0:
            epochs = training_tokens / flow_tokens
            print_fn(f"Warning: training will wrap data {epochs:.1f}x (steps={cfg.steps:,} needs {training_tokens/1e9:.2f}B tokens, flow has {flow_tokens_b:.2f}B)")

    # Populate source file paths (flow-scoped root)
    dataset_root = Path(cfg.data_root) / "flows" / cfg.flow_mode
    populate_paths(
        plan,
        dataset_root=dataset_root,
        split=split,
    )

    # Log stage info (rank 0 only)
    if rank == 0:
        for st in plan.stages:
            assigned = sum(len(sp.paths) for sp in st.sources)
            print_fn(f"Stage {st.stage_id}: sources={len(st.sources)} files={assigned}")

    # Preflight check
    _preflight_check(plan, cfg.seq_len, print_fn)

    # Use batch_size * seq_len as tokens_per_update so each .next() returns batch_size sequences
    # This matches what train.py expects: inputs.shape[0] == batch_size
    effective_tpu = cfg.batch_size * cfg.seq_len
    loader = DeterministicLoader(
        plan=plan,
        dp_world_size=world_size,
        dp_rank=rank,
        seq_len=cfg.seq_len,
        tokens_per_update=effective_tpu,
        device='cuda',
        # Prefetch breaks exact-resume unless queue contents are checkpointed.
        prefetch_depth=0,
    )

    return loader, plan
