**Research Task: Find all techniques to reduce optimizer state memory for NVFP4 MoE training on memory-constrained GPUs**

## Context

We are training DeepSeek V3.2 REAP-345B, a Mixture-of-Experts model, using SFT (supervised fine-tuning) on 8x NVIDIA B200 GPUs (178.35 GiB HBM each). The model has:

- **Architecture**: 61 transformer layers, 58 of which contain MoE blocks. Hidden dim H=7168, FFN intermediate dim Dff=2048. 128 total experts, partitioned across 8 GPUs via expert parallelism (EP=8), so 16 local experts per GPU.
- **Weight format**: Expert weights stored in NVFP4 (E2M1 4-bit float, from NVIDIA's compressed_tensors format). Each MoE layer has 3 weight matrices per expert: W1 [2048, 7168], W3 [2048, 7168], W2 [7168, 2048]. These are the "primary" weights — no BF16 master copy exists.
- **Optimizer**: ECO (Error-Compensating Optimizer) from arXiv:2601.22101v1 (Algorithm 3). AdamW-style with FP8 first moment (m) and FP8 second moment (v), per-row scaling, stochastic rounding, and error feedback to compensate for weight quantization.
- **Forward**: NVFP4 weights are converted to blockscaled MMA format (E8M0 per-block-32 scales) for CUTLASS grouped GEMM.
- **Backward**: NVFP4 weights are transiently dequantized to BF16 via a CUDA kernel (ct_nvfp4_to_bf16) for wgrad computation, then freed immediately. Optimizer update is fused inside backward (per-layer, not end-of-step).

## The Problem: GPU Memory Budget

Current per-GPU memory breakdown:
- Dense/Attention BF16 parameters: ~30 GiB
- NVFP4 expert weight buffers (packed + scale + global_scale): ~23 GiB
- Blockscaled weight cache (MMA-swizzled, rebuilt each forward): ~23 GiB
- **FP8 optimizer states (m + v): ~76 GiB** ← THIS IS THE PROBLEM
- Total static: ~152 GiB
- Remaining for activations/gradients/temporaries: ~26 GiB

The ~26 GiB headroom is insufficient for backward pass temporaries. Each MoE layer's backward requires transiently materializing one BF16 expert weight (~1.3 GiB), a FP32 gradient (~0.9 GiB), plus activation tensors (Xe_pad, H1, H3, dA, dH1, dH3, dYe_pad, dX_pad). We are hitting OOM even with aggressive staggering (never holding two BF16 weights simultaneously, freeing each tensor immediately after use).

## FP8 Optimizer State Size Breakdown

Per expert weight matrix (e.g., W1 at [7168, 2048] in optimizer layout):
- m_data: [16, 7168, 2048] uint8 (float8_e5m2) = 224 MiB
- m_scale: [16 × 7168] float32 (per-row scale) = 0.44 MiB
- v_data: [16, 7168, 2048] uint8 (float8_e4m3fn) = 224 MiB
- v_scale: [16 × 7168] float32 (per-row scale) = 0.44 MiB

Per weight: ~448 MiB for m+v
× 3 weights per layer (W1, W3, W2)
× 58 MoE layers
= **76.2 GiB total for optimizer states**

## What We Already Know About

1. **Adafactor factored v** (Shazeer & Stern, 2018): Replace full v[M,N] with v_row[M] + v_col[N], reconstruct via outer product on-the-fly. Would save ~38 GiB (v goes from 38 GiB → negligible). Well-proven at scale (T5, PaLM).

2. **4-bit optimizer states**: Papers like "4-bit optimizers" (Dettmers et al.) that quantize m and v to 4-bit with block-wise quantization.

3. **GaLore** (Zhao et al., 2024): Low-rank gradient projection to reduce optimizer state to rank-r subspace.

## What We Need You To Find

Search arxiv and recent ML literature (2023-2026) for ALL techniques that could reduce optimizer state memory, specifically for:

1. **Techniques compatible with MoE architectures** — many experts, per-expert optimizer states
2. **Techniques compatible with quantized (NVFP4/FP8) primary weights** — no FP32 master copy exists
3. **Techniques compatible with fused backward optimizer updates** — optimizer step happens inside backward, per-layer, not end-of-step
4. **Techniques that preserve or improve convergence** for fine-tuning (SFT), not just pretraining

Specific questions:
- Are there papers that combine ECO-style error feedback with factored optimizer states?
- Are there papers on MoE-specific optimizer memory reduction (e.g., sharing states across experts)?
- Are there papers on "lazy" or "on-demand" optimizer states (only materializing states for active experts)?
- What about diagonal or Kronecker-factored second moments beyond Adafactor?
- SM3, CAME, Sophia, Lion — do any of these eliminate or drastically reduce stored state?
- Any papers on offloading optimizer states to CPU with async prefetch for MoE (since only 1 expert's state is needed at a time)?
- NVIDIA-specific papers on memory-efficient training for Blackwell/Hopper architectures?
- Any papers on sharing first/second moment statistics across the gate/up/down projections (W1/W3/W2) of the same MoE layer?

For each technique found, provide:
- Paper title, authors, arxiv ID
- Memory savings (quantified if possible for our dimensions: 16 experts × [7168, 2048] per weight × 58 layers)
- Convergence impact (if studied)
- Compatibility notes with our setup (NVFP4 primary weights, fused backward, MoE)
- Implementation complexity (trivial / moderate / significant)

Rank all techniques by VRAM freed (descending) for our specific setup.
