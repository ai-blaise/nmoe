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

Note: For large hidden dimensions (D > 1024) and small expert counts (E < 128),
the unfused path using cuBLAS may be faster. The fused kernel shines when:
- Batch sizes are small (kernel launch overhead dominates)
- Expert count is moderate (fits in L2 cache)
- Hidden dimension is not too large
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


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

    # TopK selection with iterative argmax
    for k in tl.static_range(K):
        # Find maximum
        max_idx = tl.argmax(tl.where(e_mask, selection_scores, -float('inf')), axis=0)

        # Get original probability for gating (not biased score)
        gate_val = tl.sum(tl.where(e_offs == max_idx, probs, tl.zeros_like(probs)), axis=0)

        # Store expert ID and gate
        tl.store(expert_ids_ptr + pid_t * K + k, max_idx.to(tl.int32))
        tl.store(gates_ptr + pid_t * K + k, gate_val.to(tl.float16))

        # Mask out selected expert
        selection_scores = tl.where(e_offs == max_idx, -float('inf'), selection_scores)

    # Normalize gates
    gate_sum = tl.zeros((), dtype=tl.float32)
    for k in tl.static_range(K):
        gate_sum += tl.load(gates_ptr + pid_t * K + k).to(tl.float32)

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

    Forward: Uses fused Triton kernel for speed
    Backward: Computes gradient w.r.t. router_weight using PyTorch ops

    The backward pass computes:
    d_router_weight = hidden.T @ d_gates_expanded
    where d_gates_expanded broadcasts the gate gradients back through topk selection.
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
        ctx.topk = topk
        ctx.route_scale = route_scale
        ctx.E = E

        return expert_ids, gates.to(dtype), dispatch_indices, expert_counts

    @staticmethod
    def backward(ctx, grad_expert_ids, grad_gates, grad_dispatch_indices, grad_expert_counts):
        """Backward pass computing gradient w.r.t router_weight.

        The gradient flows through:
        gates = softmax(topk(sigmoid(hidden @ router_weight * scale + bias)))

        We compute d_router_weight by:
        1. Expanding grad_gates back to full [T, E] shape
        2. Computing d_logits = grad_gates_full * sigmoid_grad * scale
        3. d_router_weight = hidden.T @ d_logits
        """
        hidden, router_weight, expert_ids, gates = ctx.saved_tensors
        topk = ctx.topk
        route_scale = ctx.route_scale
        E = ctx.E
        T = hidden.shape[0]

        grad_router_weight = None

        if router_weight.requires_grad and grad_gates is not None:
            # Recompute logits and probs for gradient
            logits = hidden.float() @ router_weight.float()  # [T, E]
            if route_scale != 1.0:
                logits = logits * route_scale
            probs = torch.sigmoid(logits)  # [T, E]

            # Sigmoid gradient: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            sigmoid_grad = probs * (1.0 - probs)  # [T, E]

            # Expand grad_gates [T, K] back to [T, E] using scatter
            grad_gates_full = torch.zeros(T, E, dtype=torch.float32, device=hidden.device)
            expert_ids_long = expert_ids.long()  # [T, K]
            grad_gates_float = grad_gates.float()  # [T, K]

            # ---- Correct Jacobian through the topk renormalization ----
            # Forward: x_i = sigmoid(logit_i) for selected experts, s_i = x_i / S
            # where S = sum_j x_j over the K selected experts.
            #
            # Jacobian of the normalization s = x / S:
            #   ds_i / dx_j = (delta_ij - s_i) / S       (for i,j in topk)
            #
            # VJP:  dx_j = sum_i grad_s_i * ds_i/dx_j
            #            = (grad_s_j - dot(grad_s, s)) / S
            #
            # Here grad_s is grad_gates [T, K] and s is gates [T, K] (normalized).

            # Gather the unnormalized sigmoid outputs for the selected experts
            gates_unnorm = torch.gather(probs, 1, expert_ids_long)  # [T, K]
            gate_sum = gates_unnorm.sum(dim=-1, keepdim=True).clamp(min=1e-12)  # [T, 1]

            gates_float = gates.float()  # [T, K] -- normalized gates (s)

            # VJP through the sum-normalization
            # dot(grad_gates, gates) per token
            dot_gs = (grad_gates_float * gates_float).sum(dim=-1, keepdim=True)  # [T, 1]
            # grad w.r.t. unnormalized gates x_j
            grad_gates_unnorm = (grad_gates_float - dot_gs) / gate_sum  # [T, K]

            # Scatter the corrected gradients back to [T, E]
            grad_gates_full.scatter_add_(1, expert_ids_long, grad_gates_unnorm)

            # Chain rule through sigmoid: d_logit = d_x * sigmoid'(logit) * route_scale
            d_logits = grad_gates_full * sigmoid_grad * route_scale  # [T, E]

            # Compute gradient w.r.t router_weight: [D, E] = [D, T] @ [T, E]
            grad_router_weight = hidden.float().T @ d_logits  # [D, E]
            grad_router_weight = grad_router_weight.to(router_weight.dtype)

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
    - Forward uses fused Triton kernel, backward uses PyTorch for gradient computation

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
    def update_bias(self, expert_loads: torch.Tensor, gamma: float = 0.001):
        """Update bias for load balancing.

        Args:
            expert_loads: [E] normalized load per expert (must be on same device as bias)
            gamma: Learning rate for bias update
        """
        # P3.9: Ensure expert_loads computation is complete before reading values
        # This prevents race conditions when loads are computed asynchronously
        if expert_loads.is_cuda:
            torch.cuda.current_stream(expert_loads.device).synchronize()
        expected = 1.0 / self.n_experts
        s = torch.sign(expert_loads - expected)
        self.bias -= gamma * (s - s.mean())
        self.bias.clamp_(-16.0, 16.0)





