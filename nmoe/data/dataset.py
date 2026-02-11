"""
Minimal memmap FSL dataset + cursor utilities.

We keep it simple: each source is a concatenation of .npy (uint32) shards.
Cursor addresses a (file_idx, pos) pair and we can emit fixed-length windows
of tokens (seq_len). Crossing file boundaries is supported.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Cursor:
    file_idx: int = 0
    pos_in_file: int = 0
    wrap_count: int = 0


class NumpyFSLDataset:
    def __init__(self, shard_paths: List[str]):
        if not shard_paths:
            raise ValueError("no shard paths")
        self.paths = [str(p) for p in shard_paths]
        self.arrs = [np.load(p, mmap_mode="r") if p.endswith(".npy") else np.memmap(p, dtype=np.uint32, mode="r") for p in self.paths]
        self.lens = [int(len(a)) for a in self.arrs]

    def total_tokens(self) -> int:
        return sum(self.lens)

    def next_window(self, cursor: Cursor, length: int) -> Tuple[np.ndarray, Cursor]:
        """
        Return a contiguous window of `length` tokens (may cross shard boundaries).
        """
        remaining = length
        parts: List[np.ndarray] = []
        fidx = cursor.file_idx
        pos = cursor.pos_in_file
        wrap = cursor.wrap_count

        while remaining > 0:
            if fidx >= len(self.arrs):
                # wrap
                fidx = 0
                pos = 0
                wrap += 1
            arr = self.arrs[fidx]
            n = min(remaining, max(0, self.lens[fidx] - pos))
            if n > 0:
                parts.append(arr[pos : pos + n])
                pos += n
                remaining -= n
            else:
                # move to next file
                fidx += 1
                pos = 0

        out = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return out, Cursor(file_idx=fidx, pos_in_file=pos, wrap_count=wrap)

    def advance(self, cursor: Cursor, length: int) -> Cursor:
        """Advance the cursor by `length` tokens without reading."""
        remaining = length
        fidx = cursor.file_idx
        pos = cursor.pos_in_file
        wrap = cursor.wrap_count
        while remaining > 0:
            if fidx >= len(self.arrs):
                fidx = 0
                pos = 0
                wrap += 1
            take = min(remaining, max(0, self.lens[fidx] - pos))
            if take > 0:
                pos += take
                remaining -= take
            else:
                fidx += 1
                pos = 0
        return Cursor(file_idx=fidx, pos_in_file=pos, wrap_count=wrap)

    def warmup(self, bytes_to_read: int = 512 * 1024 * 1024) -> None:
        """Sequentially read a portion of each shard to warm the page cache."""
        chunk = 4 * 1024 * 1024 // 4  # 4MB worth of uint32 tokens
        tokens_to_read = bytes_to_read // 4
        for arr in self.arrs:
            n = min(tokens_to_read, len(arr))
            off = 0
            while off < n:
                _ = arr[off : off + min(chunk, n - off)]
                off += chunk
