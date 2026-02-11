"""
Deduplication utilities: exact and near-duplicate detection.

Implements:
- Exact hash (content) dedup.
- Line‑level and paragraph‑level within-document dedup.
- MinHash/LSH near‑duplicate detection suitable for streaming "keep-first".

Design goals:
- Deterministic, dependency-light MinHash with seeded permutations.
- Character-shingle default (k=13) to match common web dedup practice.
- LSH banding to avoid O(N^2) comparisons; approximate Jaccard via signature match rate.

Notes:
- The MinHash/LSH here targets correctness and simplicity. For trillion‑scale,
  swapping to a specialized library (e.g., datasketch, RAPIDS) is advised.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

from .transforms import normalize_text


def exact_hash(text: str) -> str:
    """Compute stable SHA-256 content hash for exact deduplication.

    Args:
        text: Input text to hash.

    Returns:
        Hex string of SHA-256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_exact(texts: Iterable[str]) -> Tuple[List[str], Set[str]]:
    """Deduplicate texts using exact hash matching.

    Args:
        texts: Iterable of text strings to deduplicate.

    Returns:
        Tuple of (unique texts, set of seen hashes).
    """
    seen: Set[str] = set()
    out: List[str] = []
    for t in texts:
        h = exact_hash(t)
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out, seen


def dedup_line_level(text: str) -> str:
    """Remove repeated lines within a single document.

    Args:
        text: Input document text.

    Returns:
        Text with duplicate lines removed (first occurrence kept).
    """
    lines = text.splitlines()
    seen: Set[str] = set()
    out: List[str] = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return "\n".join(out)


def dedup_paragraph_level(text: str) -> str:
    """Remove repeated paragraphs within a single document.

    Args:
        text: Input document text with paragraphs separated by blank lines.

    Returns:
        Text with duplicate paragraphs removed (first occurrence kept).
    """
    paras = [p.strip() for p in text.split("\n\n")]
    seen: Set[str] = set()
    out: List[str] = []
    for p in paras:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out)


# ========================
# MinHash / LSH primitives
# ========================

def _shingles(text: str, k: int) -> Iterator[int]:
    """Yield 64-bit hashes for all character shingles of length k.

    Uses blake2b seeded with a fixed prefix for determinism.
    """
    if k <= 0:
        return
    t = normalize_text(text)
    if len(t) < k:
        return
    # To keep fast and deterministic, hash shingles via blake2b to 64-bit ints
    for i in range(len(t) - k + 1):
        s = t[i : i + k]
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
        yield int.from_bytes(h.digest(), "little", signed=False)


@dataclass
class _Perms:
    a: np.ndarray  # [P]
    b: np.ndarray  # [P]
    p: int         # large prime modulus


def _make_perms(num_perm: int, seed: int = 42) -> _Perms:
    rng = np.random.default_rng(seed)
    # 61-bit prime (2^61-1 is prime); use smaller to avoid overflow in Python ints → cast to uint64
    prime = (1 << 61) - 1
    a = rng.integers(1, prime - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, prime - 1, size=num_perm, dtype=np.uint64)
    return _Perms(a=a, b=b, p=prime)


def minhash_signature(
    text: str,
    *,
    shingle: int = 13,
    num_perm: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Compute MinHash signature for text using character shingles.

    Returns: np.ndarray shape [num_perm] dtype=uint64
    """
    perms = _make_perms(num_perm, seed)
    sig = np.full((num_perm,), (1 << 64) - 1, dtype=np.uint64)
    # Iterate hashed shingles and update signature
    for x in _shingles(text, shingle):
        x64 = np.uint64(x)
        # h_i(x) = (a_i * x + b_i) mod p, then cast to uint64
        vals = (perms.a * x64 + perms.b) % np.uint64(perms.p)
        sig = np.minimum(sig, vals)
    return sig


def jaccard_from_signature(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Approximate Jaccard similarity from two MinHash signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have same shape")
    if sig_a.size == 0:
        return 0.0
    return float(np.mean(sig_a == sig_b))


def _derive_bands_rows(num_perm: int, bands: Optional[int], rows: Optional[int]) -> Tuple[int, int]:
    if bands and rows:
        if bands * rows != num_perm:
            raise ValueError("bands * rows must equal num_perm")
        return bands, rows
    # Heuristic: prefer rows=4 when divisible, else rows=3, else fall back to rows=2
    for r in (4, 3, 2):
        if num_perm % r == 0:
            return num_perm // r, r
    return num_perm, 1


class LSHIndex:
    """Simple LSH banding index for MinHash signatures.

    Not optimized for huge corpora but good for streaming keep‑first logic.
    """

    def __init__(self, num_perm: int, *, bands: Optional[int] = None, rows: Optional[int] = None):
        b, r = _derive_bands_rows(num_perm, bands, rows)
        self.num_perm = int(num_perm)
        self.bands = int(b)
        self.rows = int(r)
        self._tables: List[Dict[Tuple[int, ...], List[int]]] = [dict() for _ in range(self.bands)]
        self._sigs: List[np.ndarray] = []

    def _band_tuple(self, sig: np.ndarray, b: int) -> Tuple[int, ...]:
        start = b * self.rows
        end = start + self.rows
        view = sig[start:end].tolist()
        return tuple(int(v) for v in view)

    def add(self, sig: np.ndarray) -> int:
        idx = len(self._sigs)
        self._sigs.append(sig)
        for b in range(self.bands):
            key = self._band_tuple(sig, b)
            tb = self._tables[b]
            if key not in tb:
                tb[key] = [idx]
            else:
                tb[key].append(idx)
        return idx

    def candidates(self, sig: np.ndarray) -> Set[int]:
        out: Set[int] = set()
        for b in range(self.bands):
            key = self._band_tuple(sig, b)
            ids = self._tables[b].get(key)
            if ids:
                out.update(ids)
        return out

    def get(self, idx: int) -> np.ndarray:
        return self._sigs[idx]


def stream_near_dedup(
    texts: Iterable[str],
    *,
    shingle: int = 13,
    num_perm: int = 128,
    jaccard_threshold: float = 0.82,
    bands: Optional[int] = None,
    rows: Optional[int] = None,
    seed: int = 42,
) -> Iterator[str]:
    """Stream near‑deduplicated texts (keep‑first) using MinHash LSH.

    For each incoming text, compute signature → query LSH → verify with
    signature Jaccard approximation → keep if below threshold; otherwise drop.
    """
    idx = LSHIndex(num_perm, bands=bands, rows=rows)
    for t in texts:
        sig = minhash_signature(t, shingle=shingle, num_perm=num_perm, seed=seed)
        dup = False
        for cand in idx.candidates(sig):
            sim = jaccard_from_signature(sig, idx.get(cand))
            if sim >= jaccard_threshold:
                dup = True
                break
        if not dup:
            idx.add(sig)
            yield t
