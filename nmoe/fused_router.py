"""Fused Router + TopK + Dispatch kernel.

Task 5.2.5: Fuse router scoring, TopK selection, and dispatch metadata
generation into a single Triton kernel launch.

Benefits:
- Reduces kernel launch overhead from 3 to 1
- Better memory locality (intermediate tensors stay in L2)
- Particularly beneficial for small batches

Architecture:
- Phase 1: Compute router scores via tiled matmul (hidden @ router_weight)
- Phase 2: Apply sigmoid activation for routing probabilities
- Phase 3: TopK selection using iterative argmax with masking
- Phase 4: Atomic per-expert token counting for dispatch metadata

Backward:
- Fused CUDA kernel (router_bwd.cu) replaces ~18 PyTorch kernel launches
  with a single sparse-formulation kernel + transpose/cast helper.

Note: For large hidden dimensions (D > 1024) and small expert counts (E < 128),
the unfused path using cuBLAS may be faster. The fused kernel shines when:
- Batch sizes are small (kernel launch overhead dominates)
- Expert count is moderate (fits in L2 cache)
- Hidden dimension is not too large

Fused Bias Update (Performance Optimization):
- Eliminates 8 kernels + 1 CPU sync per layer (464 kernels + 58 stalls per step)
- Replaces: bincount, cast, normalize, sync, sign, mean, bias_update, clamp
- With: 2 fused Triton kernels (bincount atomics + single-block bias update)
"""

import os
import ctypes
import stat
from contextlib import nullcontext
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in ("1", "true", "True")


_NVTX_ENABLED = (
    _env_flag("NMOE_NVTX", "0")
    and torch.cuda.is_available()
    and hasattr(torch.cuda, "nvtx")
    and hasattr(torch.cuda.nvtx, "range_push")
)
_ROUTER_BWD_ALLOW_STANDALONE = _env_flag("NMOE_ROUTER_BWD_ALLOW_STANDALONE", "0")
_ROUTER_STREAM_CACHE_LIMIT = max(1, int(os.getenv("NMOE_ROUTER_STREAM_CACHE_LIMIT", "8")))
_EXPECTED_RDEP_ABI = 1
_FUSED_ROUTER_MAX_K = 16
_NULL_NVTX_CTX = nullcontext()


def _nvtx(name: str):
    """Emit an NVTX range visible in Nsight Systems / nvprof."""
    if _NVTX_ENABLED:
        return torch.cuda.nvtx.range(name)
    return _NULL_NVTX_CTX


# ---------------------------------------------------------------------------
# Fused Bias Update Kernels
# ---------------------------------------------------------------------------
# These kernels fuse the load tracking + bias update pipeline into 2 launches:
#   Kernel 1: Parallel bincount via atomics (over T*K expert assignments)
#   Kernel 2: Single-block normalize + sign + mean + update + clamp (over E experts)
#
# This eliminates:
#   - torch.bincount (1 kernel)
#   - cast to float32 (1 kernel)
#   - normalize (1 kernel)
#   - torch.cuda.synchronize() (CPU STALL!)
#   - torch.sign (2 kernels for sub + sign)
#   - bias update (3 kernels for sub, mul, sub)
#   - clamp (1 kernel)
# Total: 8 kernels + 1 CPU sync -> 2 kernels, no sync
# ---------------------------------------------------------------------------


@triton.jit
def _bincount_kernel(
    expert_ids_ptr,  # [T * K] int32 - flattened expert assignments
    counts_ptr,      # [E] float32 - output counts (must be zeroed before launch)
    TK,              # Total number of expert assignments (T * K)
    BLOCK: tl.constexpr,
):
    """Parallel bincount via atomic adds.

    Each program processes BLOCK expert_ids and atomically increments counts.
    This is the first kernel in the fused bias update pipeline.

    Args:
        expert_ids_ptr: Pointer to flattened expert IDs [T*K]
        counts_ptr: Pointer to count accumulator [E] (must be pre-zeroed)
        TK: Total number of expert assignments
        BLOCK: Number of elements to process per program
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TK

    # Load expert IDs (with bounds checking)
    eids = tl.load(expert_ids_ptr + offs, mask=mask, other=0)

    # Atomic increment - use float for later normalization
    # tl.atomic_add expects matching types, so we increment by 1.0
    tl.atomic_add(counts_ptr + eids, tl.where(mask, 1.0, 0.0))


@triton.jit
def _update_bias_fused_kernel(
    counts_ptr,      # [E] float32 - expert counts from bincount
    bias_ptr,        # [E] float32 - router bias (in-place update)
    TK,              # float - total assignments (T * K) for normalization
    E,               # int - number of experts
    gamma,           # float - bias learning rate
    BLOCK_E: tl.constexpr,
):
    """Fused bias update: normalize + sign + mean + update + clamp.

    Single thread block processes all E experts. Since E is small (64-256),
    this fits in one warp/block and avoids inter-block communication.

    Operations (all fused in registers):
        1. load_frac = counts / TK  (normalize to get load fraction)
        2. expected = 1.0 / E
        3. s = sign(load_frac - expected)
        4. s_mean = mean(s)  (reduction within block)
        5. bias -= gamma * (s - s_mean)
        6. bias = clamp(bias, -16, 16)

    Args:
        counts_ptr: Expert counts from _bincount_kernel
        bias_ptr: Router bias buffer (updated in-place)
        TK: Total assignments for normalization (float)
        E: Number of experts
        gamma: Bias update learning rate
        BLOCK_E: Block size (must be >= E, power of 2)
    """
    tid = tl.arange(0, BLOCK_E)
    mask = tid < E

    # Step 1: Load counts and normalize to get load fraction
    counts = tl.load(counts_ptr + tid, mask=mask, other=0).to(tl.float32)
    load_frac = counts / TK

    # Step 2: Expected load per expert (uniform distribution)
    expected = 1.0 / E

    # Step 3: Sign of deviation from expected
    # s = 1 if load > expected, -1 if load < expected, 0 if equal
    diff = load_frac - expected
    s = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

    # Step 4: Compute mean of s (reduction)
    # Only sum over valid experts (mask out padding)
    s_sum = tl.sum(tl.where(mask, s, 0.0), axis=0)
    s_mean = s_sum / E

    # Step 5: Compute bias update delta
    delta = gamma * (s - s_mean)

    # Step 6: Load old bias, apply update, clamp to [-16, 16]
    old_bias = tl.load(bias_ptr + tid, mask=mask, other=0.0)
    new_bias = old_bias - delta

    # Clamp to prevent extreme bias values
    new_bias = tl.minimum(tl.maximum(new_bias, -16.0), 16.0)

    # Store updated bias
    tl.store(bias_ptr + tid, new_bias, mask=mask)


@triton.jit
def _update_bias_normalized_kernel(
    loads_ptr,       # [E] float32 - normalized loads (sum to 1.0)
    bias_ptr,        # [E] float32 - router bias (in-place update)
    E,               # int - number of experts
    gamma,           # float - bias learning rate
    BLOCK_E: tl.constexpr,
):
    """Fused bias update for pre-normalized loads.

    Use this when loads are already normalized (sum to 1.0), e.g., after
    Transformer.forward() which normalizes across EP ranks.

    Operations:
        1. expected = 1.0 / E
        2. s = sign(loads - expected)
        3. s_mean = mean(s)
        4. bias -= gamma * (s - s_mean)
        5. bias = clamp(bias, -16, 16)

    Args:
        loads_ptr: Pre-normalized expert loads (sum to 1.0)
        bias_ptr: Router bias buffer (updated in-place)
        E: Number of experts
        gamma: Bias update learning rate
        BLOCK_E: Block size (must be >= E, power of 2)
    """
    tid = tl.arange(0, BLOCK_E)
    mask = tid < E

    # Step 1: Load pre-normalized loads
    loads = tl.load(loads_ptr + tid, mask=mask, other=0.0)

    # Step 2: Expected load per expert
    expected = 1.0 / E

    # Step 3: Sign of deviation
    diff = loads - expected
    s = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

    # Step 4: Mean of s
    s_sum = tl.sum(tl.where(mask, s, 0.0), axis=0)
    s_mean = s_sum / E

    # Step 5: Bias update
    delta = gamma * (s - s_mean)
    old_bias = tl.load(bias_ptr + tid, mask=mask, other=0.0)
    new_bias = old_bias - delta

    # Step 6: Clamp
    new_bias = tl.minimum(tl.maximum(new_bias, -16.0), 16.0)
    tl.store(bias_ptr + tid, new_bias, mask=mask)
    # Reset counts in-place for next launch (eliminate separate counts.zero_ pass).
    tl.store(counts_ptr + tid, 0.0, mask=mask)


def fused_update_bias_from_expert_ids(
    expert_ids: torch.Tensor,
    bias: torch.Tensor,
    gamma: float = 0.001,
) -> None:
    """Fused bias update directly from expert_ids tensor.

    This function replaces the naive pipeline:
        loads = torch.bincount(expert_ids.flatten(), minlength=E).float()
        loads /= (T * K)
        torch.cuda.synchronize()  # <-- CPU STALL eliminated!
        expected = 1.0 / E
        s = torch.sign(loads - expected)
        bias -= gamma * (s - s.mean())
        bias.clamp_(-16, 16)

    With 2 fused Triton kernels that run entirely on GPU with no sync.

    Args:
        expert_ids: [T, K] int32 tensor of selected expert IDs
        bias: [E] float32 tensor of router bias (updated in-place)
        gamma: Learning rate for bias update (default 0.001)

    Note:
        - E (number of experts) is inferred from bias.shape[0]
        - T*K is computed from expert_ids.numel()
        - bias must be float32 and contiguous
        - expert_ids must be int32 and contiguous
    """
    with _nvtx("fused_router/update_bias_from_expert_ids"):
        if not (expert_ids.is_cuda and bias.is_cuda):
            raise RuntimeError("fused_update_bias_from_expert_ids requires CUDA tensors")
        if bias.dtype != torch.float32:
            raise RuntimeError(f"Bias must be float32, got {bias.dtype}")
        if expert_ids.dtype != torch.int32:
            raise RuntimeError(f"expert_ids must be int32, got {expert_ids.dtype}")

        if not expert_ids.is_contiguous():
            raise RuntimeError("expert_ids must be contiguous on production no-fallback path.")
        if not bias.is_contiguous():
            raise RuntimeError("bias must be contiguous on production no-fallback path.")

        E = bias.shape[0]
        TK = expert_ids.numel()
        if TK == 0:
            return

        # Reuse temporary counts scratch; fused update kernel resets it in-place.
        stream_id = int(torch.cuda.current_stream(bias.device).cuda_stream)
        counts = _get_bias_counts_scratch(bias.device, E, stream_id)

        # Kernel 1: Parallel bincount
        BLOCK = 1024
        grid = ((TK + BLOCK - 1) // BLOCK,)
        _bincount_kernel[grid](
            expert_ids.view(-1),
            counts,
            TK,
            BLOCK=BLOCK,
        )

        # Kernel 2: Fused bias update (single block)
        BLOCK_E = _get_block_e(E, max_block=1024)

        _update_bias_fused_kernel[(1,)](
            counts,
            bias,
            float(TK),
            E,
            gamma,
            BLOCK_E=BLOCK_E,
        )


def fused_update_bias_from_counts(
    expert_counts: torch.Tensor,
    bias: torch.Tensor,
    gamma: float = 0.001,
    *,
    total: Optional[int] = None,
) -> None:
    """Fused bias update from pre-computed expert counts.

    Use this when expert_counts are already available (e.g., from fused router).
    Runs a single Triton kernel for the bias update logic.

    Args:
        expert_counts: [E] int32 or float32 tensor of expert counts
        bias: [E] float32 tensor of router bias (updated in-place)
        gamma: Learning rate for bias update (default 0.001)
        total: Optional pre-computed sum of counts (T*K). If not provided,
               assumes counts are already normalized (sum to 1.0) and uses
               the normalized kernel variant.

    Note:
        When total is not provided, this function assumes expert_counts is
        pre-normalized (sums to 1.0), which is the case after
        Transformer.forward() normalizes loads across EP ranks.
    """
    with _nvtx("fused_router/update_bias_from_counts"):
        if not (expert_counts.is_cuda and bias.is_cuda):
            raise RuntimeError("fused_update_bias_from_counts requires CUDA tensors")
        if bias.dtype != torch.float32:
            raise RuntimeError(f"Bias must be float32, got {bias.dtype}")

        if not expert_counts.is_contiguous():
            raise RuntimeError("expert_counts must be contiguous on production no-fallback path.")
        if not bias.is_contiguous():
            raise RuntimeError("bias must be contiguous on production no-fallback path.")

        E = bias.shape[0]

        BLOCK_E = _get_block_e(E, max_block=1024)

        if total is None:
            raise RuntimeError(
                "fused_update_bias_from_counts requires explicit `total` on production no-fallback path."
            )
        counts = expert_counts
        # Unnormalized counts with known total - use standard kernel
        _update_bias_fused_kernel[(1,)](
            counts,
            bias,
            float(total),
            E,
            gamma,
            BLOCK_E=BLOCK_E,
        )


# ---------------------------------------------------------------------------
# Load the fused router backward CUDA extension (router_bwd.cu).
# We load via ctypes to avoid coupling with the Makefile build system --
# the .so can be produced by any build path that compiles router_bwd.cu.
# This is a hard requirement: if the library is missing, the backward pass
# cannot run.
# ---------------------------------------------------------------------------

_router_bwd_lib: Optional[ctypes.CDLL] = None
_router_bwd_sig_initialized: bool = False
_router_bwd_has_bf16_fused: bool = False
_router_bwd_has_bf16_hidden_only: bool = False
_router_bwd_scratch_fp32: dict[tuple[int, int, int, int], torch.Tensor] = {}
_bias_counts_scratch: dict[tuple[int, int, int], torch.Tensor] = {}
_expert_counts_dummy_i32: dict[tuple[int, int], torch.Tensor] = {}
_router_launch_config_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
_router_block_e_cache: dict[tuple[int, int], int] = {}
_router_static_contract_cache: set[tuple[int, int]] = set()


def _validate_router_contract_once(E: int, K: int) -> None:
    """Validate static fused-router limits once per (E, K) pair."""
    key = (int(E), int(K))
    if key in _router_static_contract_cache:
        return
    if K <= 0:
        raise RuntimeError(f"topk must be >= 1, got K={K}.")
    if K > E:
        raise RuntimeError(f"topk must be <= n_experts, got K={K}, E={E}.")
    if K > _FUSED_ROUTER_MAX_K:
        raise RuntimeError(
            f"Fused router backward supports topk <= {_FUSED_ROUTER_MAX_K}, got K={K}. "
            "Adjust n_activated_experts or rebuild CUDA kernel constants."
        )
    if E > 256:
        raise RuntimeError(
            f"Fused router forward currently supports n_experts <= 256 in one kernel launch, got E={E}."
        )
    _router_static_contract_cache.add(key)


def _get_block_e(E: int, *, max_block: int) -> int:
    """Return cached BLOCK_E selection for Triton single-block reductions."""
    key = (int(E), int(max_block))
    block = _router_block_e_cache.get(key)
    if block is None:
        block = triton.next_power_of_2(int(E))
        block = max(block, 32)
        block = min(block, int(max_block))
        _router_block_e_cache[key] = int(block)
    return int(block)


def _get_router_launch_config(E: int, D: int) -> tuple[int, int, int, int]:
    """Return cached Triton launch config for fused router forward."""
    key = (int(E), int(D))
    cached = _router_launch_config_cache.get(key)
    if cached is not None:
        return cached

    block_e = triton.next_power_of_2(int(E))
    block_e = max(block_e, 32)
    block_e = min(block_e, 256)

    if D >= 8192:
        block_d = 128
    elif D >= 4096:
        block_d = 64
    else:
        block_d = 32

    num_warps = 4 if block_e <= 64 else 8
    if D >= 4096:
        num_warps = max(num_warps, 8)
    num_stages = 3 if D >= 4096 else 2

    config = (int(block_e), int(block_d), int(num_warps), int(num_stages))
    _router_launch_config_cache[key] = config
    return config


def _prune_stream_cache(cache: dict, dev_idx: int, limit: int) -> None:
    dev_keys = [k for k in cache.keys() if isinstance(k, tuple) and len(k) >= 1 and k[0] == dev_idx]
    while len(dev_keys) > limit:
        cache.pop(dev_keys[0], None)
        dev_keys.pop(0)


def _get_router_bwd_scratch_fp32(
    device: torch.device,
    E: int,
    D: int,
    stream_id: int,
) -> torch.Tensor:
    """Get reusable FP32 accumulation scratch [E, D] for router backward."""
    dev_idx = device.index if device.index is not None else -1
    key = (dev_idx, int(E), int(D), int(stream_id))
    buf = _router_bwd_scratch_fp32.get(key)
    if buf is None or buf.device != device or buf.shape != (E, D):
        buf = torch.empty(E, D, dtype=torch.float32, device=device)
        _router_bwd_scratch_fp32[key] = buf
        _prune_stream_cache(_router_bwd_scratch_fp32, dev_idx, _ROUTER_STREAM_CACHE_LIMIT)
    buf.record_stream(torch.cuda.current_stream(device))
    return buf


def _get_bias_counts_scratch(
    device: torch.device,
    E: int,
    stream_id: int,
) -> torch.Tensor:
    """Get reusable FP32 counts scratch [E] for fused bias update."""
    dev_idx = device.index if device.index is not None else -1
    key = (dev_idx, int(E), int(stream_id))
    buf = _bias_counts_scratch.get(key)
    if buf is None or buf.device != device or int(buf.numel()) != int(E):
        # Initialize once; fused update kernel clears counts after each use.
        buf = torch.zeros(E, dtype=torch.float32, device=device)
        _bias_counts_scratch[key] = buf
        _prune_stream_cache(_bias_counts_scratch, dev_idx, _ROUTER_STREAM_CACHE_LIMIT)
    buf.record_stream(torch.cuda.current_stream(device))
    return buf


def _get_expert_counts_dummy_i32(device: torch.device, E: int) -> torch.Tensor:
    """Get reusable dummy int32 counts buffer [E] for COMPUTE_COUNTS=False launches."""
    dev_idx = device.index if device.index is not None else -1
    key = (dev_idx, int(E))
    buf = _expert_counts_dummy_i32.get(key)
    if buf is None or buf.device != device or int(buf.numel()) != int(E):
        buf = torch.empty(E, dtype=torch.int32, device=device)
        _expert_counts_dummy_i32[key] = buf
    buf.record_stream(torch.cuda.current_stream(device))
    return buf


def _resolve_primary_rdep_path() -> str:
    from nmoe.csrc import rdep as _rdep_mod  # type: ignore

    path = getattr(_rdep_mod, "__file__", None)
    if not (isinstance(path, str) and path):
        raise RuntimeError("nmoe.csrc.rdep did not expose a valid extension path.")
    abi_fn = getattr(_rdep_mod, "abi_version", None)
    if not callable(abi_fn):
        raise RuntimeError("nmoe.csrc.rdep missing abi_version(); rebuild extension.")
    abi = int(abi_fn())
    if abi != _EXPECTED_RDEP_ABI:
        raise RuntimeError(
            f"nmoe.csrc.rdep ABI mismatch: expected {_EXPECTED_RDEP_ABI}, got {abi}. Rebuild extension."
        )
    real = os.path.realpath(path)
    mode = os.stat(real).st_mode
    if (mode & stat.S_IWGRP) or (mode & stat.S_IWOTH):
        raise RuntimeError(f"Refusing writable extension path: {real}")
    return real


def _init_router_bwd_signatures(lib: ctypes.CDLL) -> None:
    """Initialize ctypes signatures once to avoid per-call Python overhead."""
    global _router_bwd_has_bf16_fused, _router_bwd_has_bf16_hidden_only, _router_bwd_sig_initialized
    if _router_bwd_sig_initialized:
        return

    _router_bwd_has_bf16_fused = hasattr(lib, "fused_router_backward_bf16_fused")
    if not _router_bwd_has_bf16_fused:
        raise AttributeError(
            "fused_router_backward_bf16_fused not found in loaded rdep extension "
            "(router backward compatibility fallback is disabled)."
        )
    lib.fused_router_backward_bf16_fused.restype = ctypes.c_int
    lib.fused_router_backward_bf16_fused.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _router_bwd_has_bf16_hidden_only = hasattr(lib, "fused_router_backward_bf16_hidden_only")
    if _router_bwd_has_bf16_hidden_only:
        lib.fused_router_backward_bf16_hidden_only.restype = ctypes.c_int
        lib.fused_router_backward_bf16_hidden_only.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
    _router_bwd_sig_initialized = True


def _load_router_bwd() -> ctypes.CDLL:
    """Load and return the shared library containing fused_router_backward.

    Raises RuntimeError if the library cannot be found or loaded.  The result
    is cached so the dlopen only happens once.
    """
    global _router_bwd_lib

    if _router_bwd_lib is not None:
        return _router_bwd_lib

    _candidates: list[str] = []
    try:
        _candidates.append(_resolve_primary_rdep_path())
    except Exception as e:
        raise RuntimeError(
            "Failed to resolve primary nmoe.csrc.rdep extension for fused router backward."
        ) from e
    if _ROUTER_BWD_ALLOW_STANDALONE:
        raise RuntimeError(
            "NMOE_ROUTER_BWD_ALLOW_STANDALONE=1 is forbidden in production. "
            "Only the loaded nmoe.csrc.rdep extension may provide fused router backward kernels."
        )
    errors: list[str] = []
    for path in _candidates:
        if os.path.isfile(path):
            try:
                mode = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_LOCAL", 0)
                lib = ctypes.CDLL(path, mode=mode) if mode else ctypes.CDLL(path)
                _init_router_bwd_signatures(lib)
                _router_bwd_lib = lib
                return lib
            except (OSError, AttributeError) as exc:
                errors.append(f"  {path}: {exc}")

    msg = (
        "Failed to load fused router backward CUDA library. "
        "The fused CUDA kernel is required; there is no fallback.\n"
        "Searched paths:\n"
        + "\n".join(f"  {p}" for p in _candidates)
    )
    if errors:
        msg += "\nErrors:\n" + "\n".join(errors)
    raise RuntimeError(msg)


def _call_fused_router_backward(
    hidden: torch.Tensor,           # [T, D] BF16
    router_weight: torch.Tensor,    # [D, E] BF16
    expert_ids: torch.Tensor,       # [T, K] int32
    pre_probs: torch.Tensor,        # [T, K] BF16/FP32
    gates: torch.Tensor,            # [T, K] BF16/FP32
    grad_gates: torch.Tensor,       # [T, K] BF16/FP32
    route_scale: float,
    need_hidden_grad: bool = False,
    need_wgrad: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Call fused CUDA backward kernel; returns (grad_router_weight, grad_hidden?)."""
    with _nvtx("fused_router/call_fused_router_backward"):
        return _call_fused_router_backward_impl(
            hidden,
            router_weight,
            expert_ids,
            pre_probs,
            gates,
            grad_gates,
            route_scale,
            need_hidden_grad,
            need_wgrad,
        )


def _call_fused_router_backward_impl(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    expert_ids: torch.Tensor,
    pre_probs: torch.Tensor,
    gates: torch.Tensor,
    grad_gates: torch.Tensor,
    route_scale: float,
    need_hidden_grad: bool,
    need_wgrad: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Inner implementation of fused CUDA backward kernel call."""
    lib = _load_router_bwd()

    T, D = hidden.shape
    E = router_weight.shape[1]
    K = expert_ids.shape[1]
    stream = torch.cuda.current_stream(hidden.device).cuda_stream

    grad_rw_fp32: Optional[torch.Tensor] = None
    grad_rw_bf16: Optional[torch.Tensor] = None
    if need_wgrad:
        # Reuse FP32 accumulation scratch [E, D].
        grad_rw_fp32 = _get_router_bwd_scratch_fp32(hidden.device, E, D, int(stream))
        # Allocate output [D, E] BF16 for transposed result.
        grad_rw_bf16 = torch.empty(D, E, dtype=torch.bfloat16, device=hidden.device)
    grad_hidden = torch.empty_like(hidden) if need_hidden_grad else None

    if (
        pre_probs.dtype != torch.bfloat16
        or gates.dtype != torch.bfloat16
        or grad_gates.dtype != torch.bfloat16
    ):
        raise RuntimeError(
            "Router backward BF16 fast-path is required (FP32 fallback disabled). "
            f"Got pre_probs.dtype={pre_probs.dtype}, gates.dtype={gates.dtype}, "
            f"grad_gates.dtype={grad_gates.dtype}."
        )
    if need_hidden_grad and not need_wgrad:
        if not _router_bwd_has_bf16_hidden_only:
            raise RuntimeError(
                "Router backward hidden-only BF16 entrypoint is required when router weight grad is disabled. "
                "Rebuild nmoe/csrc extension with fused_router_backward_bf16_hidden_only."
            )
        err = lib.fused_router_backward_bf16_hidden_only(
            ctypes.c_void_p(hidden.data_ptr()),
            ctypes.c_void_p(router_weight.data_ptr()),
            ctypes.c_void_p(expert_ids.data_ptr()),
            ctypes.c_void_p(pre_probs.data_ptr()),
            ctypes.c_void_p(gates.data_ptr()),
            ctypes.c_void_p(grad_gates.data_ptr()),
            ctypes.c_void_p(grad_hidden.data_ptr()),
            ctypes.c_float(route_scale),
            ctypes.c_int(T), ctypes.c_int(E), ctypes.c_int(K), ctypes.c_int(D),
            ctypes.c_void_p(stream),
        )
        if err != 0:
            raise RuntimeError(f"fused_router_backward_bf16_hidden_only returned CUDA error {err}")
        return None, grad_hidden

    if not _router_bwd_has_bf16_fused:
        raise RuntimeError(
            "Router backward fused BF16 entrypoint is required. "
            "Rebuild nmoe/csrc extension with fused_router_backward_bf16_fused."
        )

    err = lib.fused_router_backward_bf16_fused(
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(router_weight.data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_void_p(pre_probs.data_ptr()),
        ctypes.c_void_p(gates.data_ptr()),
        ctypes.c_void_p(grad_gates.data_ptr()),
        ctypes.c_void_p(grad_rw_fp32.data_ptr()) if grad_rw_fp32 is not None else ctypes.c_void_p(0),
        ctypes.c_void_p(grad_rw_bf16.data_ptr()) if grad_rw_bf16 is not None else ctypes.c_void_p(0),
        ctypes.c_void_p(grad_hidden.data_ptr()) if grad_hidden is not None else ctypes.c_void_p(0),
        ctypes.c_float(route_scale),
        ctypes.c_int(T), ctypes.c_int(E), ctypes.c_int(K), ctypes.c_int(D),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"fused_router_backward_bf16_fused returned CUDA error {err}")

    return grad_rw_bf16, grad_hidden


@triton.jit
def _fused_router_kernel(
    # Input pointers
    hidden_ptr,           # [T, D] - input hidden states
    router_weight_ptr,    # [D, E] - router weight matrix (column-major for coalescing)
    bias_ptr,             # [E] - router bias for expert selection
    # Output pointers
    expert_ids_ptr,       # [T, K] - selected expert IDs
    pre_probs_ptr,        # [T, K] - selected pre-normalized probs (sigmoid scores)
    gates_ptr,            # [T, K] - gating weights
    expert_counts_ptr,    # [E] - count per expert
    # Dimensions
    T,                    # Total tokens
    D: tl.constexpr,      # Hidden dimension
    E,                    # Number of experts
    K: tl.constexpr,      # TopK (must be constexpr for static_range)
    # Strides
    stride_h_t,           # hidden stride for token dimension
    stride_h_d,           # hidden stride for hidden dimension
    stride_w_d,           # router weight stride for D
    stride_w_e,           # router weight stride for E
    # Config
    route_scale,          # Scaling factor for router logits
    COMPUTE_COUNTS: tl.constexpr,
    # Block sizes
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Optimized fused kernel - one token per program.

    Each program handles one token and computes:
    1. Router scores: hidden[t, :] @ router_weight[:, :] -> [E]
    2. TopK selection with iterative argmax
    3. Dispatch index computation via atomics
    """
    pid_t = tl.program_id(0)

    if pid_t >= T:
        return

    # Initialize scores accumulator
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    scores = tl.zeros((BLOCK_E,), dtype=tl.float32)

    # Tiled accumulation over D dimension.
    # This reduces scalar load/mul loop overhead versus per-dim iteration.
    for d0 in tl.static_range(0, D, BLOCK_D):
        d_offs = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        h_vals = tl.load(
            hidden_ptr + pid_t * stride_h_t + d_offs * stride_h_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        w_vals = tl.load(
            router_weight_ptr + d_offs[:, None] * stride_w_d + e_offs[None, :] * stride_w_e,
            mask=d_mask[:, None] & e_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores += tl.sum(w_vals * h_vals[:, None], axis=0)

    # Apply route scaling
    scores = scores * route_scale

    # Sigmoid activation (DeepSeek-style routing)
    probs = tl.sigmoid(scores)

    # Load bias and add for selection
    bias = tl.load(bias_ptr + e_offs, mask=e_mask, other=0.0)
    selection_scores = probs + bias

    # TopK selection with iterative argmax.
    # Accumulate gate_sum in registers during selection and write only pre_probs
    # first. Normalized gates are written in the second pass from pre_probs.
    # This removes an extra gates store+load round-trip.
    gate_sum = tl.zeros((), dtype=tl.float32)
    for k in tl.static_range(K):
        # Find maximum among unmasked experts
        masked_scores = tl.where(e_mask, selection_scores, -float('inf'))
        max_idx = tl.argmax(masked_scores, axis=0)
        max_score = tl.max(masked_scores, axis=0)
        is_sel = e_offs == max_idx

        # Get original probability for gating (not biased score):
        # selection_score = prob + bias => prob = max_score - bias[max_idx]
        bias_sel = tl.load(bias_ptr + max_idx).to(tl.float32)
        gate_val = max_score - bias_sel

        # Store expert ID and pre-normalized gate.
        tl.store(expert_ids_ptr + pid_t * K + k, max_idx.to(tl.int32))
        tl.store(pre_probs_ptr + pid_t * K + k, gate_val)
        if COMPUTE_COUNTS:
            tl.atomic_add(expert_counts_ptr + max_idx, 1)

        # Accumulate gate sum in register
        gate_sum += gate_val

        # Mask out selected expert for next iteration
        selection_scores = tl.where(is_sel, -float('inf'), selection_scores)

    # Normalize gates using the register-accumulated sum (single pass)
    gate_sum = tl.maximum(gate_sum, 1e-12)

    for k in tl.static_range(K):
        gate_val = tl.load(pre_probs_ptr + pid_t * K + k).to(tl.float32)
        tl.store(gates_ptr + pid_t * K + k, gate_val / gate_sum)


class _FusedRouterFunction(torch.autograd.Function):
    """Custom autograd function for fused router with backward support.

    Forward: Uses fused Triton kernel for speed.
    Backward: Uses fused CUDA kernel (router_bwd.cu) exploiting TopK sparsity.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        router_weight: torch.Tensor,
        bias: torch.Tensor,
        topk: int,
        route_scale: float,
        dtype: torch.dtype,
        need_counts: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using fused Triton kernel."""
        with _nvtx("fused_router/forward"):
            T, D = hidden.shape
            E = router_weight.shape[1]
            K = topk

            # Ensure inputs are contiguous
            if not hidden.is_contiguous():
                raise RuntimeError("hidden must be contiguous on production no-fallback path.")
            _validate_router_contract_once(E, K)

            # Allocate outputs
            expert_ids = torch.empty(T, K, dtype=torch.int32, device=hidden.device)
            pre_probs = torch.empty(T, K, dtype=dtype, device=hidden.device)
            gates = torch.empty(T, K, dtype=dtype, device=hidden.device)
            if need_counts:
                expert_counts = torch.zeros(E, dtype=torch.int32, device=hidden.device)
            else:
                expert_counts = _get_expert_counts_dummy_i32(hidden.device, E)

            # Choose block sizes
            BLOCK_E, BLOCK_D, num_warps, num_stages = _get_router_launch_config(E, D)

            # Launch fused kernel
            grid = (T,)
            _fused_router_kernel[grid](
                hidden,
                router_weight,
                bias,
                expert_ids,
                pre_probs,
                gates,
                expert_counts,
                T=T,
                D=D,
                E=E,
                K=K,
                stride_h_t=hidden.stride(0),
                stride_h_d=hidden.stride(1),
                stride_w_d=router_weight.stride(0),
                stride_w_e=router_weight.stride(1),
                route_scale=route_scale,
                COMPUTE_COUNTS=need_counts,
                BLOCK_E=BLOCK_E,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # Save for backward
            ctx.save_for_backward(hidden, router_weight, expert_ids, pre_probs, gates)
            ctx.route_scale = route_scale

            return expert_ids, gates, expert_counts

    @staticmethod
    def backward(ctx, grad_expert_ids, grad_gates, grad_expert_counts):
        """Backward pass computing gradient w.r.t router_weight.

        The gradient flows through:
        gates = softmax(topk(sigmoid(hidden @ router_weight * scale + bias)))

        Uses a fused CUDA kernel (router_bwd.cu) that exploits the TopK
        sparsity to avoid materializing the dense [T, E] intermediates.
        """
        with _nvtx("fused_router/backward"):
            hidden, router_weight, expert_ids, pre_probs, gates = ctx.saved_tensors
            route_scale = ctx.route_scale

            need_hidden = bool(ctx.needs_input_grad[0])
            need_w = bool(ctx.needs_input_grad[1])
            grad_hidden = None
            grad_router_weight = None

            if grad_gates is not None and (need_hidden or need_w):
                if not grad_gates.is_contiguous():
                    raise RuntimeError("grad_gates must be contiguous on production no-fallback path.")
                grad_router_weight, grad_hidden = _call_fused_router_backward(
                    hidden,
                    router_weight,
                    expert_ids,
                    pre_probs,
                    gates,
                    grad_gates,
                    route_scale,
                    need_hidden_grad=need_hidden,
                    need_wgrad=need_w,
                )

            # Return gradients for all inputs (None for non-differentiable ones)
            return grad_hidden, grad_router_weight, None, None, None, None, None


class FusedRouterTopKDispatch(torch.nn.Module):
    """Fused Router + TopK + Dispatch module with backward support for training.

    Combines router scoring, TopK expert selection, and dispatch metadata
    generation into a single operation. This reduces kernel launch overhead
    from 3 separate operations to 1.

    The fused kernel:
    1. Computes router logits: hidden @ router_weight
    2. Applies sigmoid activation for routing probabilities
    3. Adds bias for expert load balancing
    4. Performs TopK selection via iterative argmax
    5. Normalizes gates to sum to 1
    6. Computes per-expert routed-token counts via atomics

    TRAINING SUPPORT:
    - Uses custom autograd function with backward pass
    - Gradients flow through router_weight for training
    - Forward uses fused Triton kernel, backward uses fused CUDA kernel

    Args:
        hidden_dim: Hidden dimension size
        n_experts: Number of experts
        topk: Number of experts to select per token
        route_scale: Scaling factor for router logits (default: 1.0)
        dtype: Data type for parameters (default: torch.bfloat16)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_experts: int,
        topk: int,
        route_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if dtype != torch.bfloat16:
            raise RuntimeError(
                f"FusedRouterTopKDispatch requires dtype=torch.bfloat16 for fused BF16 backward; got {dtype}."
            )
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.topk = topk
        self.route_scale = route_scale
        self.dtype = dtype
        _validate_router_contract_once(n_experts, topk)

        # Router weight matrix [D, E]
        self.router_weight = torch.nn.Parameter(
            torch.empty(hidden_dim, n_experts, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.router_weight)

        # Bias for load balancing (as in Router class)
        self.register_buffer(
            "bias",
            torch.zeros(n_experts, dtype=torch.float32)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        need_counts: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with backward support for training.

        Args:
            hidden: [T, D] or [B, S, D] input hidden states

        Returns:
            expert_ids: [T, K] selected expert IDs (int32)
            gates: [T, K] gating weights (normalized, dtype)
            expert_counts: [E] number of tokens per expert (int32)
        """
        with _nvtx("fused_router/module_forward"):
            # Flatten to [T, D]
            if hidden.dtype != torch.bfloat16:
                raise RuntimeError(f"Fused router requires BF16 hidden input; got {hidden.dtype}.")
            if hidden.dim() == 3:
                hidden = hidden.view(-1, hidden.size(-1))

            T, D = hidden.shape

            if D != self.hidden_dim:
                raise RuntimeError(
                    f"Hidden dim mismatch: got {D}, expected {self.hidden_dim}"
                )

            # Use custom autograd function for backward support
            return _FusedRouterFunction.apply(
                hidden,
                self.router_weight,
                self.bias,
                self.topk,
                self.route_scale,
                self.dtype,
                need_counts,
            )

    @torch.no_grad()
    def update_bias(
        self,
        expert_loads: torch.Tensor,
        gamma: float = 0.001,
        *,
        expert_ids: Optional[torch.Tensor] = None,
    ):
        """Update bias for load balancing using fused Triton kernels.

        This method uses fused GPU kernels to eliminate CPU synchronization
        and reduce kernel launch overhead from 8 kernels + 1 sync to 2 kernels.

        Args:
            expert_loads: [E] expert counts (unnormalized int32 or normalized float32).
                         Used when expert_ids is not provided.
            gamma: Learning rate for bias update (default 0.001)
            expert_ids: Optional [T, K] int32 tensor of selected expert IDs.
                       When provided, computes counts internally via fused bincount.
                       This is the most efficient path - 2 kernel launches total.

        Note:
            - NO torch.cuda.synchronize() - runs entirely on GPU
            - When expert_ids is provided: uses fused bincount + update (2 kernels)
            - When expert_loads is provided: uses fused update only (1 kernel)
        """
        with _nvtx("fused_router/update_bias"):
            if expert_ids is None:
                raise RuntimeError(
                    "FusedRouterTopKDispatch.update_bias requires expert_ids on production no-fallback path."
                )
            # Most efficient path: fused bincount + bias update
            # 2 Triton kernel launches, no CPU sync
            fused_update_bias_from_expert_ids(expert_ids, self.bias, gamma)
