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
- Phase 4: Atomic dispatch index computation per expert

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
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


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
    counts = tl.load(counts_ptr + tid, mask=mask, other=0.0)
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
    assert expert_ids.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    assert bias.dtype == torch.float32, "Bias must be float32"
    assert expert_ids.dtype == torch.int32, "expert_ids must be int32"

    expert_ids = expert_ids.contiguous()
    bias = bias.contiguous()

    E = bias.shape[0]
    TK = expert_ids.numel()

    # Allocate temporary counts buffer (zeroed)
    counts = torch.zeros(E, dtype=torch.float32, device=bias.device)

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
    BLOCK_E = triton.next_power_of_2(E)
    BLOCK_E = max(BLOCK_E, 32)
    BLOCK_E = min(BLOCK_E, 1024)

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
    assert expert_counts.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    assert bias.dtype == torch.float32, "Bias must be float32"

    expert_counts = expert_counts.contiguous()
    bias = bias.contiguous()

    E = bias.shape[0]

    # Convert counts to float32 if needed
    if expert_counts.dtype != torch.float32:
        counts = expert_counts.float()
    else:
        counts = expert_counts

    BLOCK_E = triton.next_power_of_2(E)
    BLOCK_E = max(BLOCK_E, 32)
    BLOCK_E = min(BLOCK_E, 1024)

    if total is not None:
        # Unnormalized counts with known total - use standard kernel
        _update_bias_fused_kernel[(1,)](
            counts,
            bias,
            float(total),
            E,
            gamma,
            BLOCK_E=BLOCK_E,
        )
    else:
        # Pre-normalized loads (sum to 1.0) - use normalized kernel
        _update_bias_normalized_kernel[(1,)](
            counts,
            bias,
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


def _load_router_bwd() -> ctypes.CDLL:
    """Load and return the shared library containing fused_router_backward.

    Raises RuntimeError if the library cannot be found or loaded.  The result
    is cached so the dlopen only happens once.
    """
    global _router_bwd_lib

    if _router_bwd_lib is not None:
        return _router_bwd_lib

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_this_dir, "csrc", "rdep.cpython-313-x86_64-linux-gnu.so"),
        os.path.join(_this_dir, "csrc", "rdep.cpython-314-x86_64-linux-gnu.so"),
        os.path.join(_this_dir, "csrc", "router_bwd.so"),
        os.path.join(_this_dir, "csrc", "librouter_bwd.so"),
    ]

    errors: list[str] = []
    for path in _candidates:
        if os.path.isfile(path):
            try:
                lib = ctypes.CDLL(path)
                _ = lib.fused_router_backward
                _ = lib.fused_router_bwd_transpose
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
    gates_f32: torch.Tensor,        # [T, K] FP32
    grad_gates_f32: torch.Tensor,   # [T, K] FP32
    route_scale: float,
) -> torch.Tensor:
    """Call the fused CUDA backward kernel; returns grad_router_weight [D, E] BF16."""
    lib = _load_router_bwd()

    T, D = hidden.shape
    E = router_weight.shape[1]
    K = expert_ids.shape[1]

    # Allocate FP32 accumulation buffer [E, D] -- must be zeroed.
    grad_rw_fp32 = torch.zeros(E, D, dtype=torch.float32, device=hidden.device)

    # Allocate output [D, E] BF16 for transposed result.
    grad_rw_bf16 = torch.empty(D, E, dtype=torch.bfloat16, device=hidden.device)

    # Get raw data pointers.
    stream = torch.cuda.current_stream(hidden.device).cuda_stream

    lib.fused_router_backward.restype = ctypes.c_int
    lib.fused_router_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]

    err = lib.fused_router_backward(
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(router_weight.data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_void_p(gates_f32.data_ptr()),
        ctypes.c_void_p(grad_gates_f32.data_ptr()),
        ctypes.c_void_p(grad_rw_fp32.data_ptr()),
        ctypes.c_void_p(0),  # grad_hidden = NULL (not needed)
        ctypes.c_float(route_scale),
        ctypes.c_int(T), ctypes.c_int(E), ctypes.c_int(K), ctypes.c_int(D),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"fused_router_backward returned CUDA error {err}")

    # Transpose + cast [E, D] FP32 -> [D, E] BF16.
    lib.fused_router_bwd_transpose.restype = ctypes.c_int
    lib.fused_router_bwd_transpose.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]

    err = lib.fused_router_bwd_transpose(
        ctypes.c_void_p(grad_rw_fp32.data_ptr()),
        ctypes.c_void_p(grad_rw_bf16.data_ptr()),
        ctypes.c_int(E), ctypes.c_int(D),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"fused_router_bwd_transpose returned CUDA error {err}")

    return grad_rw_bf16


@triton.jit
def _fused_router_kernel(
    # Input pointers
    hidden_ptr,           # [T, D] - input hidden states
    router_weight_ptr,    # [D, E] - router weight matrix (column-major for coalescing)
    bias_ptr,             # [E] - router bias for expert selection
    # Output pointers
    expert_ids_ptr,       # [T, K] - selected expert IDs
    gates_ptr,            # [T, K] - gating weights
    dispatch_indices_ptr, # [T, K] - dispatch indices
    expert_counts_ptr,    # [E] - count per expert
    # Dimensions
    T,                    # Total tokens
    D,                    # Hidden dimension
    E,                    # Number of experts
    K: tl.constexpr,      # TopK (must be constexpr for static_range)
    # Strides
    stride_h_t,           # hidden stride for token dimension
    stride_h_d,           # hidden stride for hidden dimension
    stride_w_d,           # router weight stride for D
    stride_w_e,           # router weight stride for E
    # Config
    route_scale,          # Scaling factor for router logits
    # Block sizes
    BLOCK_E: tl.constexpr,
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

    # Simple accumulation over D dimension (no blocking for simplicity)
    for d_idx in range(D):
        # Load hidden value
        h_val = tl.load(hidden_ptr + pid_t * stride_h_t + d_idx * stride_h_d).to(tl.float32)

        # Load weights for all experts at this D position
        w_vals = tl.load(
            router_weight_ptr + d_idx * stride_w_d + e_offs * stride_w_e,
            mask=e_mask,
            other=0.0
        ).to(tl.float32)

        # Scalar-vector multiply and accumulate
        scores += h_val * w_vals

    # Apply route scaling
    scores = scores * route_scale

    # Sigmoid activation (DeepSeek-style routing)
    probs = tl.sigmoid(scores)

    # Load bias and add for selection
    bias = tl.load(bias_ptr + e_offs, mask=e_mask, other=0.0)
    selection_scores = probs + bias

    # TopK selection with iterative argmax.
    # Accumulate gate_sum in registers during selection to avoid a second
    # pass over global memory for normalization (saves K loads + K stores).
    gate_sum = tl.zeros((), dtype=tl.float32)
    for k in tl.static_range(K):
        # Find maximum among unmasked experts
        max_idx = tl.argmax(tl.where(e_mask, selection_scores, -float('inf')), axis=0)

        # Get original probability for gating (not biased score)
        gate_val = tl.sum(tl.where(e_offs == max_idx, probs, tl.zeros_like(probs)), axis=0)

        # Store expert ID and unnormalized gate
        tl.store(expert_ids_ptr + pid_t * K + k, max_idx.to(tl.int32))
        tl.store(gates_ptr + pid_t * K + k, gate_val.to(tl.float16))

        # Accumulate gate sum in register (no global memory round-trip)
        gate_sum += gate_val

        # Mask out selected expert for next iteration
        selection_scores = tl.where(e_offs == max_idx, -float('inf'), selection_scores)

    # Normalize gates using the register-accumulated sum (single pass)
    gate_sum = tl.maximum(gate_sum, 1e-12)

    for k in tl.static_range(K):
        gate_val = tl.load(gates_ptr + pid_t * K + k).to(tl.float32)
        tl.store(gates_ptr + pid_t * K + k, (gate_val / gate_sum).to(tl.float16))

    # Compute dispatch indices via atomics
    for k in tl.static_range(K):
        expert_id = tl.load(expert_ids_ptr + pid_t * K + k)
        offset = tl.atomic_add(expert_counts_ptr + expert_id, 1)
        tl.store(dispatch_indices_ptr + pid_t * K + k, offset)


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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using fused Triton kernel."""
        T, D = hidden.shape
        E = router_weight.shape[1]
        K = topk

        # Ensure inputs are contiguous
        hidden = hidden.contiguous()

        # Allocate outputs
        expert_ids = torch.empty(T, K, dtype=torch.int32, device=hidden.device)
        gates = torch.empty(T, K, dtype=torch.float16, device=hidden.device)
        dispatch_indices = torch.empty(T, K, dtype=torch.int32, device=hidden.device)
        expert_counts = torch.zeros(E, dtype=torch.int32, device=hidden.device)

        # Choose block sizes
        BLOCK_E = triton.next_power_of_2(E)
        BLOCK_E = max(BLOCK_E, 32)
        BLOCK_E = min(BLOCK_E, 256)

        # Launch fused kernel
        grid = (T,)
        _fused_router_kernel[grid](
            hidden,
            router_weight,
            bias,
            expert_ids,
            gates,
            dispatch_indices,
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
            BLOCK_E=BLOCK_E,
        )

        # Save for backward
        ctx.save_for_backward(hidden, router_weight, expert_ids, gates)
        ctx.route_scale = route_scale

        return expert_ids, gates.to(dtype), dispatch_indices, expert_counts

    @staticmethod
    def backward(ctx, grad_expert_ids, grad_gates, grad_dispatch_indices, grad_expert_counts):
        """Backward pass computing gradient w.r.t router_weight.

        The gradient flows through:
        gates = softmax(topk(sigmoid(hidden @ router_weight * scale + bias)))

        Uses a fused CUDA kernel (router_bwd.cu) that exploits the TopK
        sparsity to avoid materializing the dense [T, E] intermediates.
        """
        hidden, router_weight, expert_ids, gates = ctx.saved_tensors
        route_scale = ctx.route_scale

        grad_router_weight = None

        if router_weight.requires_grad and grad_gates is not None:
            gates_f32 = gates.float()
            grad_gates_f32 = grad_gates.float()

            grad_router_weight = _call_fused_router_backward(
                hidden.contiguous(),
                router_weight.contiguous(),
                expert_ids.contiguous(),
                gates_f32.contiguous(),
                grad_gates_f32.contiguous(),
                route_scale,
            )

        # Return gradients for all inputs (None for non-differentiable ones)
        return None, grad_router_weight, None, None, None, None


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
    6. Computes per-expert dispatch indices via atomics

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
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.topk = topk
        self.route_scale = route_scale
        self.dtype = dtype

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with backward support for training.

        Args:
            hidden: [T, D] or [B, S, D] input hidden states

        Returns:
            expert_ids: [T, K] selected expert IDs (int32)
            gates: [T, K] gating weights (normalized, dtype)
            dispatch_indices: [T, K] per-expert offsets (int32)
            expert_counts: [E] number of tokens per expert (int32)
        """
        # Flatten to [T, D]
        original_shape = hidden.shape
        if hidden.dim() == 3:
            hidden = hidden.view(-1, hidden.size(-1))

        T, D = hidden.shape

        assert D == self.hidden_dim, f"Hidden dim mismatch: got {D}, expected {self.hidden_dim}"

        # Use custom autograd function for backward support
        return _FusedRouterFunction.apply(
            hidden,
            self.router_weight,
            self.bias,
            self.topk,
            self.route_scale,
            self.dtype,
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
        if expert_ids is not None:
            # Most efficient path: fused bincount + bias update
            # 2 Triton kernel launches, no CPU sync
            fused_update_bias_from_expert_ids(expert_ids, self.bias, gamma)
        else:
            # Fallback: expert counts already computed, just do fused update
            # 1 Triton kernel launch, no CPU sync
            fused_update_bias_from_counts(expert_loads, self.bias, gamma)

