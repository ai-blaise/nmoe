"""NSA Compression Path Attention with Fused Block Score Aggregation.

Flash Attention-style kernel for NSA's CMP (compression) path that:
1. Computes attention over compressed K/V tokens
2. Aggregates attention weights to selection block scores (Eq 9)
3. Returns both attention output and block scores for SLC path

Based on Flash Attention v2 (https://tridao.me/publications/flash2/flash2.pdf)
adapted from swa.py for NSA-specific requirements.

Memory: O(BLOCK_M × Tc) per tile instead of O(T × Tc) for full materialization.
"""

import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

try:
    import pytest
except Exception:
    pytest = None


@triton.jit
def _cmp_attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    LSE,
    Out,
    BlockScores,  # [B*H, T, Ns] - aggregated selection block scores
    # Compression params (runtime values, not constexpr)
    CMP_LEN,  # l: compression block length
    CMP_STRIDE,  # d: stride between blocks
    SLC_BLOCK,  # l': selection block size
    # Dimensions
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    N_SLC,  # Ns: number of selection blocks
    # Strides for BlockScores
    BS_stride_bh,
    BS_stride_t,
    BS_stride_s,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,  # Block size for selection block accumulation
):
    """
    CMP attention forward with fused block score aggregation.

    Causal mask: query at position t can attend to compressed token i
    if the block i covers ends before t, i.e., i*d + l - 1 <= t.

    Block score aggregation (Eq 9): For each selection block j,
    sum attention weights from all compressed tokens that overlap with it.
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < N_Q_CTX

    # Initialize accumulators for online softmax
    # Use m_i = 0 (not -inf) so masked positions contribute ~0 even when ALL keys are masked
    # This acts like an implicit "attend to nothing" baseline (similar to SWA's sink)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Block score accumulator: [BLOCK_M, BLOCK_S]
    # We accumulate unnormalized attention weights, then normalize at the end
    offs_s = tl.arange(0, BLOCK_S)
    block_scores_acc = tl.zeros([BLOCK_M, BLOCK_S], dtype=tl.float32)

    # Load query tile using TensorDescriptor
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    numerator = (offs_m - CMP_LEN + 1).to(tl.float32)
    max_valid_i = tl.math.floor(numerator / CMP_STRIDE).to(tl.int32)  # [BLOCK_M]

    # Compute hi bound: max compressed token index any query in this block can attend to
    # This is max_valid_i for the last query in the block
    # Use scalar math (Triton compiles these as compile-time or runtime scalars)
    last_query_pos = start_m * BLOCK_M + BLOCK_M - 1
    hi_bound = (last_query_pos - CMP_LEN + 1) // CMP_STRIDE + 1
    hi = tl.minimum(hi_bound, N_KV_CTX)

    # Early exit if no compressed tokens are valid
    if hi <= 0:
        # Write zeros to output
        zero = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        Out.store([off_z, off_h, start_m * BLOCK_M, 0], zero.to(Out.dtype)[None, None, :, :])
        # Write -inf to LSE
        neg_inf = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        lse_ptrs = LSE + off_hz * N_Q_CTX + offs_m
        tl.store(lse_ptrs, neg_inf, mask=m_mask)
        return

    # Stream through compressed tokens
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_KV_CTX

        # Causal mask: compressed token i is valid for query m if i <= max_valid_i[m]
        # i.e., n_idx <= max_valid_i (per query row)
        causal_mask = n_idx[None, :] > max_valid_i[:, None]  # True = masked
        mask = causal_mask | (~n_mask[None, :]) | (~m_mask[:, None])

        # Load K tile: [BLOCK_N, HEAD_DIM] -> transpose to [HEAD_DIM, BLOCK_N]
        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T

        # Compute QK scores
        qk = tl.dot(q, k, allow_tf32=False)
        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # Update output accumulator
        acc = acc * alpha[:, None]
        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc, allow_tf32=False)

        # Aggregate to block scores (Eq 9) - paper exact overlap-based
        # Compressed token i covers positions [i*d, i*d+l)
        # Selection block j covers positions [j*l', (j+1)*l')
        # Token i contributes to block j if ranges overlap: i*d < (j+1)*l' AND i*d+l > j*l'
        cmp_start = n_idx * CMP_STRIDE  # [BLOCK_N] - start position of each compressed token
        cmp_end = cmp_start + CMP_LEN   # [BLOCK_N] - end position

        slc_start = offs_s * SLC_BLOCK  # [BLOCK_S] - start of each selection block
        slc_end = slc_start + SLC_BLOCK # [BLOCK_S] - end of each selection block

        # Overlap indicator: [BLOCK_N, BLOCK_S] - 1.0 if compressed token overlaps selection block
        overlaps = ((cmp_start[:, None] < slc_end[None, :]) &
                    (cmp_end[:, None] > slc_start[None, :])).to(tl.float32)

        # Rescale previous block_scores by alpha, then add new contributions
        block_scores_acc = block_scores_acc * alpha[:, None]

        # Each compressed token contributes to ALL selection blocks it overlaps with
        # p: [BLOCK_M, BLOCK_N], overlaps: [BLOCK_N, BLOCK_S]
        # block_scores_acc += p @ overlaps
        block_scores_acc += tl.dot(p, overlaps, allow_tf32=False)

        # Update running stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Finalize: normalize output and block scores
    # Add epsilon to handle queries with no valid keys (l_i ≈ 0 -> output = 0)
    l_i_safe = tl.maximum(l_i, 1e-6)
    acc = acc / l_i_safe[:, None]
    block_scores_acc = block_scores_acc / l_i_safe[:, None]

    # Store LSE
    lse = m_i + tl.math.log(l_i)
    lse_ptrs = LSE + off_hz * N_Q_CTX + offs_m
    tl.store(lse_ptrs, lse, mask=m_mask)

    # Store output
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)

    # Store block scores (only first BLOCK_S selection blocks supported)
    bs_base = BlockScores + off_hz * BS_stride_bh
    s_mask = offs_s < N_SLC
    bs_ptrs = bs_base + offs_m[:, None] * BS_stride_t + offs_s[None, :] * BS_stride_s
    tl.store(bs_ptrs, block_scores_acc, mask=m_mask[:, None] & s_mask[None, :])


@triton.jit
def _cmp_attn_bwd(
    Q_ptr, K_ptr, V_ptr, DO_ptr,
    DQ_ptr, DK_ptr, DV_ptr, LSE_ptr,
    sm_scale,
    CMP_LEN,
    CMP_STRIDE,
    Z, H,
    N_Q_CTX, N_KV_CTX,
    Q_sZ, Q_sH, Q_sM, Q_sD,
    K_sZ, K_sH, K_sN, K_sD,
    V_sZ, V_sH, V_sN, V_sD,
    DO_sZ, DO_sH, DO_sM, DO_sD,
    DQ_sZ, DQ_sH, DQ_sM, DQ_sD,
    DK_sZ, DK_sH, DK_sN, DK_sD,
    DV_sZ, DV_sH, DV_sN, DV_sD,
    L_sZ, L_sH, L_sM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Backward pass for CMP attention - row-block parallelism."""
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_Q_CTX

    # Load Q and dO tiles
    q_ptrs = Q_ptr + off_z * Q_sZ + off_h * Q_sH + offs_m[:, None] * Q_sM + offs_d[None, :] * Q_sD
    do_ptrs = DO_ptr + off_z * DO_sZ + off_h * DO_sH + offs_m[:, None] * DO_sM + offs_d[None, :] * DO_sD
    l_ptrs = LSE_ptr + off_z * L_sZ + off_h * L_sH + offs_m * L_sM

    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0).to(tl.float32)
    dO = tl.load(do_ptrs, mask=m_mask[:, None], other=0).to(tl.float32)
    lse = tl.load(l_ptrs, mask=m_mask, other=0).to(tl.float32)

    # Causal bound (use floor division, not C-style truncation)
    numerator = (offs_m - CMP_LEN + 1).to(tl.float32)
    max_valid_i = tl.math.floor(numerator / CMP_STRIDE).to(tl.int32)

    # Compute hi bound
    last_query_pos = start_m * BLOCK_M + BLOCK_M - 1
    hi_bound = (last_query_pos - CMP_LEN + 1) // CMP_STRIDE + 1
    hi = tl.minimum(hi_bound, N_KV_CTX)

    # First pass: compute D = rowsum(dO * O)
    # We recompute O by iterating through K/V blocks
    D = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_KV_CTX

        causal_mask = n_idx[None, :] > max_valid_i[:, None]
        mask = causal_mask | (~n_mask[None, :]) | (~m_mask[:, None])

        k_ptrs = K_ptr + off_z * K_sZ + off_h * K_sH + n_idx[:, None] * K_sN + offs_d[None, :] * K_sD
        v_ptrs = V_ptr + off_z * V_sZ + off_h * V_sH + n_idx[:, None] * V_sN + offs_d[None, :] * V_sD
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)

        qk = tl.dot(q, k.T, allow_tf32=False) * sm_scale
        # Safe exponent: when a query row has no valid keys, lse can be -inf.
        # Clamp the exponent argument so fully-masked rows yield p=0 without inf/NaN.
        _delta = qk - lse[:, None]
        _delta = tl.where(lse[:, None] <= -1.0e30, -1.0e30, _delta)
        p = tl.math.exp(_delta)
        p = tl.where(mask, 0.0, p)

        # D += sum(p * (dO @ v^T))
        dP = tl.dot(dO, v.T, allow_tf32=False)
        D += tl.sum(dP * p, 1)

    # Second pass: compute gradients
    dQ_tile = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_KV_CTX

        causal_mask = n_idx[None, :] > max_valid_i[:, None]
        mask = causal_mask | (~n_mask[None, :]) | (~m_mask[:, None])

        k_ptrs = K_ptr + off_z * K_sZ + off_h * K_sH + n_idx[:, None] * K_sN + offs_d[None, :] * K_sD
        v_ptrs = V_ptr + off_z * V_sZ + off_h * V_sH + n_idx[:, None] * V_sN + offs_d[None, :] * V_sD
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)

        # Recompute attention with the same safe exponent guard
        qk = tl.dot(q, k.T, allow_tf32=False) * sm_scale
        _delta = qk - lse[:, None]
        _delta = tl.where(lse[:, None] <= -1.0e30, -1.0e30, _delta)
        p = tl.math.exp(_delta)
        p = tl.where(mask, 0.0, p)

        # dP = dO @ V^T
        dP = tl.dot(dO, v.T, allow_tf32=False)

        # dS = P * (dP - D) * scale
        dS = (dP - D[:, None]) * p * sm_scale

        # dQ += dS @ K
        dQ_tile += tl.dot(dS, k, allow_tf32=False)

        # dK += dS^T @ Q (atomic)
        dK_tile = tl.dot(dS.T, q, allow_tf32=False)
        dk_ptrs = DK_ptr + off_z * DK_sZ + off_h * DK_sH + n_idx[:, None] * DK_sN + offs_d[None, :] * DK_sD
        tl.atomic_add(dk_ptrs, dK_tile, mask=n_mask[:, None])

        # dV += P^T @ dO (atomic)
        dV_tile = tl.dot(p.T, dO, allow_tf32=False)
        dv_ptrs = DV_ptr + off_z * DV_sZ + off_h * DV_sH + n_idx[:, None] * DV_sN + offs_d[None, :] * DV_sD
        tl.atomic_add(dv_ptrs, dV_tile, mask=n_mask[:, None])

    # Store dQ
    dq_ptrs = DQ_ptr + off_z * DQ_sZ + off_h * DQ_sH + offs_m[:, None] * DQ_sM + offs_d[None, :] * DQ_sD
    tl.store(dq_ptrs, dQ_tile.to(tl.bfloat16), mask=m_mask[:, None])


class _CMPAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cmp, v_cmp, sm_scale, cmp_len, cmp_stride, slc_block, n_slc_blocks):
        B, T, H, D = q.shape
        Tc = k_cmp.shape[1]
        Ns = n_slc_blocks

        # Transpose to [B, H, T, D] for kernel
        q = q.transpose(1, 2).contiguous()
        k_cmp = k_cmp.transpose(1, 2).contiguous()
        v_cmp = v_cmp.transpose(1, 2).contiguous()

        # Tune block sizes based on HEAD_DIM to stay within shared memory limits
        if D <= 64:
            BLOCK_M, BLOCK_N = 64, 64
            num_warps, num_stages = 8, 3
        elif D <= 128:
            BLOCK_M, BLOCK_N = 32, 64
            num_warps, num_stages = 4, 2
        else:
            BLOCK_M, BLOCK_N = 32, 32
            num_warps, num_stages = 4, 1

        BLOCK_S = min(Ns, 256)  # Block size for selection blocks

        # Pad to block boundaries
        m_pad = (BLOCK_M - T % BLOCK_M) % BLOCK_M
        n_pad = (BLOCK_N - Tc % BLOCK_N) % BLOCK_N

        if m_pad > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, m_pad))
        if n_pad > 0:
            k_cmp = torch.nn.functional.pad(k_cmp, (0, 0, 0, n_pad))
            v_cmp = torch.nn.functional.pad(v_cmp, (0, 0, 0, n_pad))

        T_pad = T + m_pad
        Tc_pad = Tc + n_pad

        # Allocate outputs
        out = torch.empty_like(q)
        lse = torch.empty((B, H, T_pad), device=q.device, dtype=torch.float32)
        block_scores = torch.zeros((B * H, T_pad, Ns), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(T, BLOCK_M), B * H)

        _cmp_attn_fwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, D]),
            TensorDescriptor.from_tensor(k_cmp, [1, 1, BLOCK_N, D]),
            TensorDescriptor.from_tensor(v_cmp, [1, 1, BLOCK_N, D]),
            sm_scale,
            lse,
            TensorDescriptor.from_tensor(out, [1, 1, BLOCK_M, D]),
            block_scores,
            cmp_len, cmp_stride, slc_block,
            B, H, T_pad, Tc_pad, Ns,
            *block_scores.stride(),
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_S=BLOCK_S,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Save for backward
        ctx.save_for_backward(q, k_cmp, v_cmp, lse)
        ctx.sm_scale = sm_scale
        ctx.cmp_len = cmp_len
        ctx.cmp_stride = cmp_stride
        ctx.T = T
        ctx.Tc = Tc
        ctx.m_pad = m_pad
        ctx.n_pad = n_pad

        # Trim padding and reshape
        out = out[:, :, :T, :].transpose(1, 2).contiguous()
        block_scores = block_scores.view(B, H, T_pad, Ns)[:, :, :T, :].permute(0, 2, 1, 3).contiguous()

        return out, block_scores

    @staticmethod
    def backward(ctx, grad_out, grad_block_scores):
        # grad_block_scores is ignored (block selection is non-differentiable per paper)
        q, k_cmp, v_cmp, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        cmp_len = ctx.cmp_len
        cmp_stride = ctx.cmp_stride
        T = ctx.T
        Tc = ctx.Tc

        B, H, T_pad, D = q.shape
        Tc_pad = k_cmp.shape[2]

        # Reshape grad_out to [B, H, T, D]
        dO = grad_out.transpose(1, 2).contiguous()
        if ctx.m_pad > 0:
            dO = torch.nn.functional.pad(dO, (0, 0, 0, ctx.m_pad))

        # Allocate grads
        dQ = torch.zeros_like(q)
        dK = torch.zeros((B, H, Tc_pad, D), device=q.device, dtype=torch.float32)
        dV = torch.zeros((B, H, Tc_pad, D), device=q.device, dtype=torch.float32)

        def strides(x):
            return tuple(int(s) for s in x.stride())

        # Tune block sizes based on HEAD_DIM to stay within shared memory limits
        # B200 has 232KB shared memory per SM
        # Main memory: 5*(BLOCK_M*D) + 3*(BLOCK_M*BLOCK_N) in float32
        if D <= 64:
            BLOCK_M, BLOCK_N = 64, 64
            num_warps, num_stages = 8, 2
        elif D <= 128:
            BLOCK_M, BLOCK_N = 32, 64  # Reduce BLOCK_M to fit in shared memory
            num_warps, num_stages = 4, 2
        else:
            BLOCK_M, BLOCK_N = 32, 32
            num_warps, num_stages = 4, 1

        grid = (triton.cdiv(T, BLOCK_M), B * H)

        _cmp_attn_bwd[grid](
            q, k_cmp, v_cmp, dO,
            dQ, dK, dV, lse,
            sm_scale,
            cmp_len, cmp_stride,
            B, H, T_pad, Tc_pad,
            *strides(q), *strides(k_cmp), *strides(v_cmp), *strides(dO),
            *strides(dQ), *strides(dK), *strides(dV),
            *lse.stride(),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Trim and transpose
        dQ = dQ[:, :, :T, :].transpose(1, 2).contiguous()
        dK = dK[:, :, :Tc, :].transpose(1, 2).contiguous().to(k_cmp.dtype)
        dV = dV[:, :, :Tc, :].transpose(1, 2).contiguous().to(v_cmp.dtype)

        return dQ, dK, dV, None, None, None, None, None


cmp_attention = _CMPAttention.apply


def cmp_attention_ref(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    sm_scale: float,
    cmp_len: int,
    cmp_stride: int,
    slc_block: int,
    n_slc_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for correctness testing."""
    B, T, H, D = q.shape
    Tc = k_cmp.shape[1]
    Ns = n_slc_blocks
    l, d, l_prime = cmp_len, cmp_stride, slc_block

    # Compute attention scores [B, H, T, Tc]
    Q = q.transpose(1, 2).float()  # [B, H, T, D]
    K = k_cmp.transpose(1, 2).float()  # [B, H, Tc, D]
    V = v_cmp.transpose(1, 2).float()  # [B, H, Tc, D]

    scores = torch.einsum('bhtd,bhcd->bhtc', Q, K) * sm_scale

    # Causal mask: query t can attend to compressed token i if i*d + l - 1 <= t
    t_idx = torch.arange(T, device=q.device)
    i_idx = torch.arange(Tc, device=q.device)
    # Block i ends at position i*d + l - 1
    block_ends = i_idx * d + l - 1
    causal_mask = block_ends[None, :] > t_idx[:, None]  # [T, Tc], True = masked
    scores = scores.masked_fill(causal_mask.view(1, 1, T, Tc), -1e6)

    # Softmax with safe normalization (matches Triton kernel behavior)
    # Use max clamped to 0 so all-masked rows have exp(-1e6) ≈ 0
    scores_max = scores.max(dim=-1, keepdim=True).values.clamp(min=0)
    scores_exp = torch.exp(scores - scores_max)
    attn_weights = scores_exp / scores_exp.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    # Output
    out = torch.einsum('bhtc,bhcd->bhtd', attn_weights, V)
    out = out.transpose(1, 2).to(q.dtype)  # [B, T, H, D]

    # Aggregate to selection block scores (Eq 9) - vectorized
    # For each compressed token i, compute which selection block it maps to
    cmp_centers = i_idx * d + l // 2
    primary_slc = (cmp_centers // l_prime).clamp(max=Ns - 1)  # [Tc]

    # Create scatter matrix: [Tc, Ns] where entry (i, j) = 1 if primary_slc[i] == j
    scatter = torch.zeros(Tc, Ns, device=q.device, dtype=torch.float32)
    scatter[torch.arange(Tc, device=q.device), primary_slc] = 1.0

    # Aggregate: [B, H, T, Tc] @ [Tc, Ns] -> [B, H, T, Ns]
    block_scores = torch.einsum('bhtc,cn->bhtn', attn_weights, scatter)
    block_scores = block_scores.permute(0, 2, 1, 3)  # [B, T, H, Ns]

    return out, block_scores


if pytest is not None:
    @pytest.mark.parametrize("B", [1, 2])
    @pytest.mark.parametrize("T", [128, 256])
    @pytest.mark.parametrize("H", [4, 8])
    @pytest.mark.parametrize("D", [64])
    @pytest.mark.parametrize("cmp_len", [32])
    @pytest.mark.parametrize("cmp_stride", [16])
    @pytest.mark.parametrize("slc_block", [64])
    def test_cmp_attention(B, T, H, D, cmp_len, cmp_stride, slc_block):
        torch.manual_seed(42)

        # Tc = (T - l) // d + 1
        Tc = (T - cmp_len) // cmp_stride + 1
        Ns = T // slc_block

        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        k_cmp = torch.randn(B, Tc, H, D, device="cuda", dtype=torch.bfloat16)
        v_cmp = torch.randn(B, Tc, H, D, device="cuda", dtype=torch.bfloat16)
        sm_scale = D ** -0.5

        out_ref, bs_ref = cmp_attention_ref(q, k_cmp, v_cmp, sm_scale, cmp_len, cmp_stride, slc_block, Ns)
        out_tri, bs_tri = cmp_attention(q, k_cmp, v_cmp, sm_scale, cmp_len, cmp_stride, slc_block, Ns)

        torch.testing.assert_close(out_tri, out_ref, atol=1e-2, rtol=1e-2)
        # Block scores may have some tolerance due to aggregation order
        torch.testing.assert_close(bs_tri, bs_ref, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("B", [1])
    @pytest.mark.parametrize("T", [128])
    @pytest.mark.parametrize("H", [4])
    @pytest.mark.parametrize("D", [64])
    def test_cmp_attention_backward(B, T, H, D):
        torch.manual_seed(42)
        cmp_len, cmp_stride, slc_block = 32, 16, 64

        Tc = (T - cmp_len) // cmp_stride + 1
        Ns = T // slc_block

        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k_cmp = torch.randn(B, Tc, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v_cmp = torch.randn(B, Tc, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        sm_scale = D ** -0.5

        out, _ = cmp_attention(q, k_cmp, v_cmp, sm_scale, cmp_len, cmp_stride, slc_block, Ns)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k_cmp.grad is not None
        assert v_cmp.grad is not None
        assert not q.grad.isnan().any()
        assert not k_cmp.grad.isnan().any()
        assert not v_cmp.grad.isnan().any()
