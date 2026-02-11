"""
gpt-oss batch inference for B200.

Minimal design: Transformer + generate that returns (logits, tokens).
All sampling logic (temperature, diverse, beam) belongs in the caller.
"""
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import torch
import triton
import triton.language as tl
import xxhash
import numpy as np
from safetensors import safe_open

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.topk import topk
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import (
    PrecisionConfig, FlexCtx, FnSpecs, FusedActivation,
    RoutingData, GatherIndx, ScatterIndx, matmul_ogs,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import convert_layout, wrap_torch_tensor, FP4, make_ragged_tensor_metadata
from triton_kernels.tensor_details.layout import StridedLayout


# =============================================================================
# Sequence & Block Management
# =============================================================================

class SeqStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


@dataclass
class Sequence:
    tokens: list[int]
    max_tokens: int = 256
    seq_id: int = field(default_factory=lambda: Sequence._next_id())
    status: SeqStatus = SeqStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    _id_counter: int = 0

    @classmethod
    def _next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __len__(self): return len(self.tokens)
    def append(self, tok): self.tokens.append(tok)
    @property
    def prompt_len(self): return getattr(self, '_prompt_len', len(self.tokens))


class BlockManager:
    """Paged KV cache with prefix caching."""
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.ref_count = [0] * num_blocks
        self.block_hash = [-1] * num_blocks
        self.block_tokens = [[] for _ in range(num_blocks)]
        self.hash_to_block = {}
        self.free = deque(range(num_blocks))

    def _hash(self, tokens, prefix=-1):
        h = xxhash.xxh64()
        if prefix != -1: h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(tokens, dtype=np.int32).tobytes())
        return h.intdigest()

    def num_blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free) >= self.num_blocks_for(len(seq)) - len(seq.block_table)

    def allocate(self, seq: Sequence):
        n_blocks = self.num_blocks_for(len(seq))
        prefix_hash = -1
        for i in range(len(seq.block_table), n_blocks):
            start, end = i * self.block_size, min((i + 1) * self.block_size, len(seq))
            block_tokens = seq.tokens[start:end]
            is_full = len(block_tokens) == self.block_size
            h = self._hash(block_tokens, prefix_hash) if is_full else -1
            cached_id = self.hash_to_block.get(h, -1) if h != -1 else -1

            if cached_id != -1 and self.block_tokens[cached_id] == block_tokens:
                self.ref_count[cached_id] += 1
                seq.block_table.append(cached_id)
                seq.num_cached_tokens = end
                prefix_hash = h
            else:
                if not self.free: raise RuntimeError("No free blocks")
                block_id = self.free.popleft()
                self.ref_count[block_id] = 1
                self.block_hash[block_id] = h
                self.block_tokens[block_id] = block_tokens
                if h != -1: self.hash_to_block[h] = block_id
                seq.block_table.append(block_id)
                prefix_hash = h if is_full else -1

    def free_seq(self, seq: Sequence):
        for bid in seq.block_table:
            self.ref_count[bid] -= 1
            if self.ref_count[bid] == 0:
                h = self.block_hash[bid]
                if h != -1 and self.hash_to_block.get(h) == bid:
                    del self.hash_to_block[h]
                self.free.append(bid)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def append_slot(self, seq: Sequence) -> int:
        pos = len(seq) - 1
        block_idx = pos // self.block_size
        if block_idx >= len(seq.block_table):
            if not self.free: raise RuntimeError("No free blocks")
            block_id = self.free.popleft()
            self.ref_count[block_id] = 1
            self.block_hash[block_id] = -1
            self.block_tokens[block_id] = []
            seq.block_table.append(block_id)
        return seq.block_table[block_idx] * self.block_size + (pos % self.block_size)


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def _store_kv_kernel(
    K, V, K_cache, V_cache, Slot_mapping,
    stride_kn, stride_kh, stride_kd, stride_cn, stride_ch, stride_cd,
    N, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N: return
    slot = tl.load(Slot_mapping + pid)
    if slot < 0: return
    offs_h, offs_d = tl.arange(0, NUM_KV_HEADS), tl.arange(0, HEAD_DIM)
    k_ptrs = K + pid * stride_kn + offs_h[:, None] * stride_kh + offs_d[None, :] * stride_kd
    v_ptrs = V + pid * stride_kn + offs_h[:, None] * stride_kh + offs_d[None, :] * stride_kd
    cache_k = K_cache + slot * stride_cn + offs_h[:, None] * stride_ch + offs_d[None, :] * stride_cd
    cache_v = V_cache + slot * stride_cn + offs_h[:, None] * stride_ch + offs_d[None, :] * stride_cd
    tl.store(cache_k, tl.load(k_ptrs))
    tl.store(cache_v, tl.load(v_ptrs))


def store_kv_cache(k, v, k_cache, v_cache, slot_mapping):
    N, num_kv_heads, head_dim = k.shape
    kc, vc = k_cache.view(-1, num_kv_heads, head_dim), v_cache.view(-1, num_kv_heads, head_dim)
    _store_kv_kernel[(N,)](k, v, kc, vc, slot_mapping,
        k.stride(0), k.stride(1), k.stride(2), kc.stride(0), kc.stride(1), kc.stride(2),
        N, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim)


@triton.jit
def _prefill_attn_kernel(
    Q, K, V, Sinks, Out, sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    N_CTX: tl.constexpr, KV_HEAD_RATIO: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BANDWIDTH: tl.constexpr,
):
    batch_idx, head_idx, q_block_idx = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head_idx = head_idx // KV_HEAD_RATIO
    sink = tl.load(Sinks + head_idx).to(tl.float32)
    offs_m, offs_d = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M), tl.arange(0, HEAD_DIM)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    lo = tl.maximum(0, q_block_idx * BLOCK_M - BANDWIDTH) if BANDWIDTH > 0 else 0
    hi = tl.minimum((q_block_idx + 1) * BLOCK_M, N_CTX)

    for start_n in range(lo, hi, BLOCK_N):
        k_pos = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_pos < N_CTX
        k_ptrs = K + batch_idx * stride_kb + kv_head_idx * stride_kh + k_pos[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale
        qk = qk + tl.where(k_pos[None, :] > offs_m[:, None], -1.0e6, 0.0)
        if BANDWIDTH > 0:
            qk = qk + tl.where(k_pos[None, :] < (offs_m[:, None] - BANDWIDTH + 1), -1.0e6, 0.0)
        qk = qk + tl.where(~k_mask[None, :], -1.0e6, 0.0)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        alpha = tl.math.exp(m_i - m_ij)
        acc, l_i = acc * alpha[:, None], l_i * alpha

        v_ptrs = V + batch_idx * stride_vb + kv_head_idx * stride_vh + k_pos[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=False)
        l_i, m_i = l_i + tl.sum(p, 1), m_ij

    acc = acc / (l_i + tl.math.exp(sink - m_i))[:, None]
    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def attention_prefill(q, k, v, sinks, sm_scale, bandwidth=0):
    batch, num_heads, seq_len, head_dim = q.shape
    BLOCK_M, BLOCK_N = 64, 64
    seq_pad = ((seq_len + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if seq_pad > seq_len:
        q = torch.nn.functional.pad(q, (0, 0, 0, seq_pad - seq_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, seq_pad - seq_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, seq_pad - seq_len))
    o = torch.empty_like(q)
    _prefill_attn_kernel[(batch, num_heads, seq_pad // BLOCK_M)](
        q, k, v, sinks, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_CTX=seq_len, KV_HEAD_RATIO=num_heads // k.shape[1], HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BANDWIDTH=bandwidth)
    return o[:, :, :seq_len, :]


@triton.jit
def _decode_attn_kernel(
    Q, K_cache, V_cache, Block_tables, Context_lens, Sinks, Out, sm_scale,
    stride_qb, stride_qh, stride_qd, stride_cs, stride_ch, stride_cd,
    stride_btb, stride_bts, stride_ob, stride_oh, stride_od,
    BLOCK_SIZE: tl.constexpr, KV_HEAD_RATIO: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr, BANDWIDTH: tl.constexpr,
):
    batch_idx, head_idx = tl.program_id(0), tl.program_id(1)
    kv_head_idx = head_idx // KV_HEAD_RATIO
    context_len = tl.load(Context_lens + batch_idx)
    sink = tl.load(Sinks + head_idx).to(tl.float32)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd)

    m_i, l_i = sink, 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    q_pos = context_len - 1
    kv_start = tl.maximum(0, q_pos - BANDWIDTH + 1) if BANDWIDTH > 0 else 0

    for start_n in range(kv_start, context_len, BLOCK_N):
        k_pos = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_pos < context_len
        block_idx, block_offset = k_pos // BLOCK_SIZE, k_pos % BLOCK_SIZE
        block_ids = tl.load(Block_tables + batch_idx * stride_btb + block_idx * stride_bts, mask=k_mask, other=0)
        slots = block_ids * BLOCK_SIZE + block_offset

        k = tl.load(K_cache + slots[:, None] * stride_cs + kv_head_idx * stride_ch + offs_d[None, :] * stride_cd, mask=k_mask[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(k_pos > q_pos, -1.0e6, qk)
        if BANDWIDTH > 0: qk = tl.where(k_pos < (q_pos - BANDWIDTH + 1), -1.0e6, qk)
        qk = tl.where(~k_mask, -1.0e6, qk)

        m_ij = tl.maximum(m_i, tl.max(qk))
        p = tl.math.exp(qk - m_ij)
        alpha = tl.math.exp(m_i - m_ij)
        acc, l_i = acc * alpha, l_i * alpha + tl.sum(p)

        v = tl.load(V_cache + slots[:, None] * stride_cs + kv_head_idx * stride_ch + offs_d[None, :] * stride_cd, mask=k_mask[:, None], other=0.0).to(tl.float32)
        acc, m_i = acc + tl.sum(p[:, None] * v, axis=0), m_ij

    acc = acc / (l_i + tl.math.exp(sink - m_i))
    tl.store(Out + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od, acc.to(Out.dtype.element_ty))


def attention_decode(q, k_cache, v_cache, block_tables, context_lens, sinks, sm_scale, block_size, bandwidth=0):
    batch, num_heads, head_dim = q.shape
    o = torch.empty_like(q)
    kc, vc = k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]), v_cache.view(-1, v_cache.shape[2], v_cache.shape[3])
    _decode_attn_kernel[(batch, num_heads)](
        q, kc, vc, block_tables, context_lens, sinks, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), kc.stride(0), kc.stride(1), kc.stride(2),
        block_tables.stride(0), block_tables.stride(1), o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SIZE=block_size, KV_HEAD_RATIO=num_heads // k_cache.shape[2], HEAD_DIM=head_dim, BLOCK_N=64, BANDWIDTH=bandwidth)
    return o


# =============================================================================
# MoE
# =============================================================================

def routing(logits, n_expts_act):
    sparse = topk(logits, n_expts_act)
    dispatch, combine = sparse.mask_metadata.col_sorted_indx, sparse.mask_metadata.row_sorted_indx
    ragged = make_ragged_tensor_metadata(sparse.mask_metadata.col_sum, dispatch.shape[0])
    return (RoutingData(sparse.vals.flatten()[combine], ragged.slice_sizes, logits.shape[-1], n_expts_act, ragged),
            GatherIndx(combine, dispatch), ScatterIndx(dispatch, combine))


def quantize_mx4(w):
    w, s = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
    return convert_layout(wrap_torch_tensor(w, dtype=FP4), StridedLayout), convert_layout(wrap_torch_tensor(s), StridedLayout)


def moe(x, wg, w1, w1_mx, w2, w2_mx, bg, b1, b2, n_active=4, n_experts=128, swiglu_limit=7.0):
    if x.numel() == 0: return x
    pc1 = PrecisionConfig(weight_scale=w1_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    pc2 = PrecisionConfig(weight_scale=w2_mx, flex_ctx=FlexCtx(rhs_data=InFlexData()))
    logits = matmul_ogs(x, wg, bg, precision_config=PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData())))
    rdata, gather, scatter = routing(logits, n_active)
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.702, swiglu_limit), 2)
    x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather, precision_config=pc1, fused_activation=act)
    return matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter, precision_config=pc2, gammas=rdata.gate_scal)


# =============================================================================
# Model
# =============================================================================

@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


FP4_LUT = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class Checkpoint:
    def __init__(self, path: str, device: torch.device):
        self.dev = f"{device.type}:{device.index}" if device.index else device.type
        self.files = {}
        for f in os.listdir(path):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(path, f), framework="pt", device=self.dev) as sf:
                    for k in sf.keys(): self.files[k] = os.path.join(path, f)

    def get(self, name: str):
        if name.endswith("_weight") and "mlp" in name:
            base = name.replace("_weight", "")
            return self._mxfp4(f"{base}_weight.blocks", f"{base}_weight.scales")
        with safe_open(self.files[name], framework="pt", device=self.dev) as f:
            return f.get_tensor(name)

    def _mxfp4(self, bn, sn):
        with safe_open(self.files[bn], framework="pt", device=self.dev) as f: blocks = f.get_tensor(bn)
        with safe_open(self.files[sn], framework="pt", device=self.dev) as f: scales = f.get_tensor(sn).int() - 127
        lut = torch.tensor(FP4_LUT, dtype=torch.bfloat16, device=blocks.device)
        *pf, G, B = blocks.shape
        out = torch.empty(math.prod(pf) * G, B * 2, dtype=torch.bfloat16, device=blocks.device)
        blocks, scales = blocks.reshape(-1, B), scales.reshape(-1, 1)
        out[:, 0::2], out[:, 1::2] = lut[(blocks & 0x0F).long()], lut[(blocks >> 4).long()]
        return torch.ldexp(out, scales).view(*pf, G * B * 2)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5, device=None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.scale).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base, init_ctx=4096, max_ctx=131072, scale=1.0, alpha=1.0, beta=32.0, device=None):
        super().__init__()
        freq = base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        if scale > 1.0:
            d = dim / 2
            lo = d * math.log(init_ctx / (beta * 2 * math.pi)) / math.log(base)
            hi = d * math.log(init_ctx / (alpha * 2 * math.pi)) / math.log(base)
            mask = 1 - ((torch.arange(d, device=device) - lo) / (hi - lo)).clamp(0, 1)
            inv = (1 / (scale * freq)) * (1 - mask) + (1 / freq) * mask
            conc = 0.1 * math.log(scale) + 1.0
        else:
            inv, conc = 1 / freq, 1.0
        freqs = torch.einsum("i,j->ij", torch.arange(max_ctx, dtype=torch.float32, device=device), inv)
        self.register_buffer("cos", (freqs.cos() * conc).to(torch.bfloat16))
        self.register_buffer("sin", (freqs.sin() * conc).to(torch.bfloat16))

    def forward(self, q, k, positions):
        cos, sin = self.cos[positions], self.sin[positions]
        while cos.dim() < q.dim(): cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        def rotate(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotate(q), rotate(k)


class Attention(torch.nn.Module):
    def __init__(self, cfg, layer_idx=0, device=None):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.sm_scale = 1 / math.sqrt(cfg.head_dim)
        self.bandwidth = cfg.sliding_window if layer_idx % 2 == 0 else 0
        self.norm = RMSNorm(cfg.hidden_size, device=device)
        self.qkv = torch.nn.Linear(cfg.hidden_size, cfg.head_dim * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads),
                                   device=device, dtype=torch.bfloat16)
        self.out = torch.nn.Linear(cfg.head_dim * cfg.num_attention_heads, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        self.sinks = torch.nn.Parameter(torch.empty(cfg.num_attention_heads, device=device, dtype=torch.bfloat16))
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.initial_context_length, scale=cfg.rope_scaling_factor,
                                    alpha=cfg.rope_ntk_alpha, beta=cfg.rope_ntk_beta, device=device)
        self.k_cache = self.v_cache = None

    def forward(self, x, positions, slot_mapping=None, is_prefill=True, block_tables=None, context_lens=None, block_size=16):
        if is_prefill:
            batch, seq_len, _ = x.shape
            qkv = self.qkv(self.norm(x))
            q, k, v = qkv.split([self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim], dim=-1)
            q, k, v = q.view(batch, seq_len, self.n_heads, self.head_dim), k.view(batch, seq_len, self.n_kv, self.head_dim), v.view(batch, seq_len, self.n_kv, self.head_dim)
            q, k = self.rope(q, k, positions)
            if self.k_cache is not None and slot_mapping is not None:
                store_kv_cache(k.reshape(-1, self.n_kv, self.head_dim).clone(), v.reshape(-1, self.n_kv, self.head_dim).clone(), self.k_cache, self.v_cache, slot_mapping.view(-1))
            o = attention_prefill(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.sinks, self.sm_scale, self.bandwidth)
            o = o.transpose(1, 2).reshape(batch, seq_len, self.n_heads * self.head_dim)
        else:
            batch = x.shape[0]
            qkv = self.qkv(self.norm(x))
            q, k, v = qkv.split([self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim], dim=-1)
            q, k, v = q.view(batch, self.n_heads, self.head_dim), k.view(batch, self.n_kv, self.head_dim), v.view(batch, self.n_kv, self.head_dim)
            q, k = self.rope(q, k, positions)
            if self.k_cache is not None and slot_mapping is not None:
                store_kv_cache(k.contiguous(), v.contiguous(), self.k_cache, self.v_cache, slot_mapping)
            o = attention_decode(q, self.k_cache, self.v_cache, block_tables, context_lens, self.sinks, self.sm_scale, block_size, self.bandwidth)
            o = o.view(batch, self.n_heads * self.head_dim)
        return x + self.out(o)


class MLP(torch.nn.Module):
    def __init__(self, cfg, layer_idx=0, device=None):
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_size, device=device)
        self.gate = torch.nn.ParameterDict({
            "weight": torch.nn.Parameter(torch.empty(cfg.hidden_size, cfg.num_experts, device=device, dtype=torch.bfloat16)),
            "bias": torch.nn.Parameter(torch.empty(cfg.num_experts, device=device, dtype=torch.bfloat16)),
        })
        self.w1, self.w1_mx = quantize_mx4(torch.empty(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size * 2, device=device, dtype=torch.bfloat16))
        self.w1_data = torch.nn.Parameter(self.w1.storage.data, requires_grad=False)
        self.b1 = torch.nn.Parameter(torch.empty(cfg.num_experts, cfg.intermediate_size * 2, device=device, dtype=torch.bfloat16))
        self.w2, self.w2_mx = quantize_mx4(torch.empty(cfg.num_experts, cfg.intermediate_size, cfg.hidden_size, device=device, dtype=torch.bfloat16))
        self.w2_data = torch.nn.Parameter(self.w2.storage.data, requires_grad=False)
        self.b2 = torch.nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size, device=device, dtype=torch.bfloat16))
        self.n_experts, self.n_active, self.limit = cfg.num_experts, cfg.experts_per_token, cfg.swiglu_limit

    def forward(self, x):
        shape = x.shape
        t = moe(self.norm(x.view(-1, shape[-1])), self.gate["weight"], self.w1, self.w1_mx, self.w2, self.w2_mx,
                self.gate["bias"].float(), self.b1.float(), self.b2.float(), self.n_active, self.n_experts, self.limit)
        return x + t.view(shape)


class Transformer(torch.nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.config = cfg
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(Attention(cfg, i, device), MLP(cfg, i, device)) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, device=device)
        self.unembed = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids,
        positions,
        *,
        return_hidden_states: bool = False,
        up_to_layer: int | None = None,
        from_layer: int | None = None,
        keep_kv: bool = False,
        no_logits: bool = False,
        **kw,
    ):
        x = self.embed(input_ids)
        L = len(self.blocks)
        start = 0 if from_layer is None else max(0, int(from_layer))
        stop = L if up_to_layer is None else max(0, min(L, int(up_to_layer)))
        if from_layer is not None and up_to_layer is not None:
            start = max(0, min(L, int(from_layer)))
            stop = max(start, min(L, int(up_to_layer)))

        captured = {} if return_hidden_states else None
        for i in range(start, stop):
            attn, mlp = self.blocks[i]
            x = mlp(attn(x, positions=positions, **kw))
            if captured is not None:
                if i == 17:
                    captured[18] = x.detach()
                if i == 23:
                    captured[24] = x.detach()

        # Compute logits only if requested. When training HYDRA heads we only need hidden
        # states; skipping unembedding avoids a large [B,T,V] allocation.
        if captured is not None:
            if no_logits:
                return None, captured
            logits = self.unembed(self.norm(x)).float()
            return logits, captured
        logits = self.unembed(self.norm(x)).float()
        return logits

    @staticmethod
    def from_checkpoint(path, config=None, device="cuda"):
        device = torch.device(device) if isinstance(device, str) else device
        if not config:
            with open(os.path.join(path, "config.json")) as f:
                hf_config = json.load(f)
                # Rename HuggingFace fields to match ModelConfig
                if 'num_local_experts' in hf_config:
                    hf_config['num_experts'] = hf_config.pop('num_local_experts')
                if 'num_experts_per_tok' in hf_config:
                    hf_config['experts_per_token'] = hf_config.pop('num_experts_per_tok')
                # Flatten rope_scaling dict
                if 'rope_scaling' in hf_config:
                    rs = hf_config.pop('rope_scaling')
                    hf_config['rope_scaling_factor'] = rs.get('factor', 32.0)
                    hf_config['rope_ntk_alpha'] = rs.get('beta_slow', 1.0)
                    hf_config['rope_ntk_beta'] = rs.get('beta_fast', 32.0)
                # Filter to only ModelConfig fields
                valid_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
                config_dict = {k: v for k, v in hf_config.items() if k in valid_fields}
                config = ModelConfig(**config_dict)
        model = Transformer(config, device).eval()
        ckpt = Checkpoint(path, device)
        for name, param in model.named_parameters():
            torch.cuda.empty_cache()
            parts = name.split(".")
            # Map model param names to HuggingFace checkpoint names
            if parts[0] == "embed":
                ckpt_name = "model.embed_tokens.weight"
            elif parts[0] == "unembed":
                ckpt_name = "lm_head.weight"
            elif parts[0] == "norm":
                ckpt_name = "model.norm.weight"
            elif parts[0] == "blocks":
                layer_idx = parts[1]
                is_attn = (parts[2] == "0")
                rest = parts[3:]
                if is_attn:
                    if rest[0] == "norm":
                        ckpt_name = f"model.layers.{layer_idx}.input_layernorm.weight"
                    elif rest[0] == "qkv":
                        # QKV is combined in model but separate in checkpoint - handle specially
                        q_w = ckpt.get(f"model.layers.{layer_idx}.self_attn.q_proj.{rest[1]}")
                        k_w = ckpt.get(f"model.layers.{layer_idx}.self_attn.k_proj.{rest[1]}")
                        v_w = ckpt.get(f"model.layers.{layer_idx}.self_attn.v_proj.{rest[1]}")
                        param.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
                        continue
                    elif rest[0] == "out":
                        ckpt_name = f"model.layers.{layer_idx}.self_attn.o_proj.{rest[1]}"
                    elif rest[0] == "sinks":
                        ckpt_name = f"model.layers.{layer_idx}.self_attn.sinks"
                    else:
                        raise ValueError(f"Unknown attn param: {name}")
                else:  # MLP
                    if rest[0] == "norm":
                        ckpt_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                    elif rest[0] == "gate":
                        ckpt_name = f"model.layers.{layer_idx}.mlp.router.{rest[1]}"
                    elif rest[0] == "w1_data":
                        # Load quantized weights directly (blocks + scales)
                        blocks = ckpt.get(f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_blocks")
                        scales = ckpt.get(f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_scales")
                        # Reconstruct from mxfp4 format
                        lut = torch.tensor(FP4_LUT, dtype=torch.bfloat16, device=blocks.device)
                        *pf, G, B = blocks.shape
                        out = torch.empty(math.prod(pf) * G, B * 2, dtype=torch.bfloat16, device=blocks.device)
                        blocks_flat, scales_flat = blocks.reshape(-1, B), (scales.int() - 127).reshape(-1, 1)
                        out[:, 0::2], out[:, 1::2] = lut[(blocks_flat & 0x0F).long()], lut[(blocks_flat >> 4).long()]
                        data = torch.ldexp(out, scales_flat).view(*pf, G * B * 2)
                        t, s = quantize_mx4(data.mT.contiguous())
                        setattr(model.blocks[int(layer_idx)][1], "w1_mx", s)
                        param.data.copy_(t.storage.data)
                        continue
                    elif rest[0] == "b1":
                        ckpt_name = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_bias"
                    elif rest[0] == "w2_data":
                        # Load quantized weights directly (blocks + scales)
                        blocks = ckpt.get(f"model.layers.{layer_idx}.mlp.experts.down_proj_blocks")
                        scales = ckpt.get(f"model.layers.{layer_idx}.mlp.experts.down_proj_scales")
                        # Reconstruct from mxfp4 format
                        lut = torch.tensor(FP4_LUT, dtype=torch.bfloat16, device=blocks.device)
                        *pf, G, B = blocks.shape
                        out = torch.empty(math.prod(pf) * G, B * 2, dtype=torch.bfloat16, device=blocks.device)
                        blocks_flat, scales_flat = blocks.reshape(-1, B), (scales.int() - 127).reshape(-1, 1)
                        out[:, 0::2], out[:, 1::2] = lut[(blocks_flat & 0x0F).long()], lut[(blocks_flat >> 4).long()]
                        data = torch.ldexp(out, scales_flat).view(*pf, G * B * 2)
                        t, s = quantize_mx4(data.mT.contiguous())
                        setattr(model.blocks[int(layer_idx)][1], "w2_mx", s)
                        param.data.copy_(t.storage.data)
                        continue
                    elif rest[0] == "b2":
                        ckpt_name = f"model.layers.{layer_idx}.mlp.experts.down_proj_bias"
                    else:
                        raise ValueError(f"Unknown mlp param: {name}")
            else:
                raise ValueError(f"Unknown param: {name}")

            data = ckpt.get(ckpt_name)
            if "gate.weight" in name and data.ndim == 2:
                param.data.copy_(data.mT.contiguous())
            else:
                param.data.copy_(data)
        torch.cuda.empty_cache()
        return model


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        m = mask.to(hidden.dtype)
        s = (hidden * m.unsqueeze(-1)).sum(dim=1)
        d = m.sum(dim=1).clamp(min=1)
        return s / d.unsqueeze(-1)
    return hidden.mean(dim=1)


# =============================================================================
# Batched Generator
# =============================================================================

class BatchedGenerator:
    """Continuous batching inference with paged KV cache.

    Usage:
        gen = BatchedGenerator(checkpoint, max_seq_len=512, max_batch=32)
        seq_id = gen.add(prompt_tokens)

        while not gen.idle:
            logits, seq_ids = gen.step()
            for i, sid in enumerate(seq_ids):
                token = sample(logits[i])
                gen.update(sid, token, finished=(token in stop_tokens))
    """

    def __init__(self, checkpoint: str, max_seq_len: int = 512, max_batch: int = 32, device: torch.device = None):
        self.device = device or torch.device("cuda")
        self.max_seq_len = max_seq_len
        self.max_batch = max_batch
        self.block_size = 16

        # Load model
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)
        cfg = self.model.config

        # Size KV cache: max_batch sequences * max_seq_len tokens each
        torch.cuda.empty_cache()
        blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size
        num_blocks = max_batch * blocks_per_seq

        self.kv_cache = torch.zeros(
            2, cfg.num_hidden_layers, num_blocks, self.block_size,
            cfg.num_key_value_heads, cfg.head_dim,
            dtype=torch.bfloat16, device=self.device
        )
        for i, (attn, _) in enumerate(self.model.blocks):
            attn.k_cache, attn.v_cache = self.kv_cache[0, i], self.kv_cache[1, i]

        self.block_mgr = BlockManager(num_blocks, self.block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, tokens: list[int], max_tokens: int = None) -> int:
        """Add a request. Returns sequence ID."""
        max_tokens = max_tokens or (self.max_seq_len - len(tokens))
        seq = Sequence(list(tokens), max_tokens)
        seq._prompt_len = len(tokens)
        self.waiting.append(seq)
        return seq.seq_id

    @property
    def idle(self) -> bool:
        return not self.waiting and not self.running

    def _schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule sequences for next step. Returns (seqs, is_prefill)."""
        # First: try to admit waiting sequences (prefill)
        to_prefill = []
        while self.waiting and len(self.running) + len(to_prefill) < self.max_batch:
            seq = self.waiting[0]
            if not self.block_mgr.can_allocate(seq):
                break  # No space, wait for running seqs to finish
            self.waiting.popleft()
            self.block_mgr.allocate(seq)
            seq.status = SeqStatus.RUNNING
            to_prefill.append(seq)

        if to_prefill:
            self.running.extend(to_prefill)
            return to_prefill, True

        # Second: decode running sequences
        if not self.running:
            return [], False

        # All running sequences should be able to append (we sized the cache correctly)
        return list(self.running), False

    @torch.inference_mode()
    def step(self) -> tuple[torch.Tensor, list[int]] | None:
        """Run one step. Returns (logits, seq_ids) or None if idle."""
        seqs, is_prefill = self._schedule()
        if not seqs:
            return None

        seq_ids = [s.seq_id for s in seqs]
        batch = len(seqs)

        if is_prefill:
            max_len = max(len(s) for s in seqs)
            input_ids = torch.zeros(batch, max_len, dtype=torch.long, device=self.device)
            positions = torch.zeros(batch, max_len, dtype=torch.long, device=self.device)
            slot_mapping = torch.full((batch, max_len), -1, dtype=torch.int32, device=self.device)

            for i, seq in enumerate(seqs):
                start, end = seq.num_cached_tokens, len(seq)
                toks = seq.tokens[start:end]
                input_ids[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=self.device)
                positions[i, :len(toks)] = torch.arange(start, end, dtype=torch.long, device=self.device)
                for j, pos in enumerate(range(start, end)):
                    block_idx = pos // self.block_size
                    slot_mapping[i, j] = seq.block_table[block_idx] * self.block_size + (pos % self.block_size)

            logits = self.model(input_ids, positions, slot_mapping=slot_mapping, is_prefill=True, block_size=self.block_size)
            lens = [len(s) - s.num_cached_tokens for s in seqs]
            logits = torch.stack([logits[i, lens[i] - 1] for i in range(batch)])
        else:
            input_ids = torch.tensor([s.tokens[-1] for s in seqs], dtype=torch.long, device=self.device)
            positions = torch.tensor([len(s) - 1 for s in seqs], dtype=torch.long, device=self.device)
            slot_mapping = torch.tensor([self.block_mgr.append_slot(s) for s in seqs], dtype=torch.int32, device=self.device)
            context_lens = torch.tensor([len(s) for s in seqs], dtype=torch.int32, device=self.device)
            max_blocks = max(len(s.block_table) for s in seqs)
            block_tables = torch.tensor(
                [s.block_table + [0] * (max_blocks - len(s.block_table)) for s in seqs],
                dtype=torch.int32, device=self.device
            )
            logits = self.model(
                input_ids, positions, slot_mapping=slot_mapping, is_prefill=False,
                block_tables=block_tables, context_lens=context_lens, block_size=self.block_size
            )

        return logits, seq_ids

    def update(self, seq_id: int, token: int, finished: bool = False):
        """Update sequence with sampled token."""
        for i, seq in enumerate(self.running):
            if seq.seq_id == seq_id:
                if finished or len(seq) >= seq._prompt_len + seq.max_tokens:
                    seq.status = SeqStatus.FINISHED
                    self.block_mgr.free_seq(seq)
                    self.running.pop(i)
                else:
                    seq.append(token)
                return
        raise ValueError(f"Sequence {seq_id} not found")
