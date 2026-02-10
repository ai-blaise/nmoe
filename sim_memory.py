"""Memory simulation for DeepSeek V3.2 REAP-345B NVFP4 training on B200 GPUs.

Computes exact per-GPU memory breakdown for EP=8 (intra-node), DP=N (inter-node).
No GPU required — pure arithmetic from model config.
"""

def fmt(gib: float) -> str:
    if gib < 0.01:
        return f"{gib * 1024:.1f} MiB"
    return f"{gib:.2f} GiB"

def main():
    # =========================================================================
    # Model config (from dsv3_reap_sft_1node.toml)
    # =========================================================================
    vocab_size = 129280
    dim = 7168              # H
    inter_dim = 18432       # Dense FFN intermediate
    moe_inter_dim = 2048    # Dff (MoE FFN intermediate)
    n_layers = 61
    n_dense_layers = 3      # Layers 0-2 are dense MLP
    n_moe_layers = n_layers - n_dense_layers  # 58
    n_heads = 128
    n_routed_experts = 128
    n_shared_experts = 1
    n_activated_experts = 8
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_position_embeddings = 163840

    # Parallelism
    ep_size = 8
    n_local_experts = n_routed_experts // ep_size  # 16

    # Training
    batch_size = 8
    seq_len = 512
    group_size = 16  # NVFP4 scale group size

    # GPU
    gpu_hbm_gib = 178.35  # B200

    BF16 = 2  # bytes
    FP32 = 4
    UINT8 = 1
    FP8 = 1

    def params_to_gib(n_params: int, bytes_per: int) -> float:
        return n_params * bytes_per / (1024**3)

    print("=" * 80)
    print("DeepSeek V3.2 REAP-345B — Per-GPU Memory Simulation")
    print(f"EP={ep_size}, n_local_experts={n_local_experts}, GPU HBM={gpu_hbm_gib} GiB")
    print("=" * 80)

    # =========================================================================
    # 1. DENSE PARAMETERS (replicated on every GPU, BF16)
    # =========================================================================
    print("\n--- 1. DENSE PARAMETERS (BF16, replicated) ---")

    # Embedding
    emb_params = vocab_size * dim
    emb_gib = params_to_gib(emb_params, BF16)
    print(f"  Embedding [{vocab_size}, {dim}]: {emb_params:,} params = {fmt(emb_gib)}")

    # LM Head (not weight-tied)
    lm_head_params = dim * vocab_size
    lm_head_gib = params_to_gib(lm_head_params, BF16)
    print(f"  LM Head [{dim}, {vocab_size}]: {lm_head_params:,} params = {fmt(lm_head_gib)}")

    # RMSNorm (all layers + final)
    norm_params = (n_layers * 2 + 1) * dim  # attn_norm + ffn_norm per layer + final
    norm_gib = params_to_gib(norm_params, FP32)  # RMSNorm is FP32
    print(f"  RMSNorm ({n_layers}×2 + 1) × [{dim}]: {norm_params:,} params = {fmt(norm_gib)}")

    # MLA Attention (per layer, all 61 layers)
    # wq_a: [dim, q_lora_rank] = [7168, 1536]
    # wq_b: [q_lora_rank, n_heads * (qk_nope_head_dim + qk_rope_head_dim)] = [1536, 128*192] = [1536, 24576]
    # wkv_a: [dim, kv_lora_rank + qk_rope_head_dim] = [7168, 576]
    # wkv_b: [kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)] = [512, 128*256] = [512, 32768]
    # wo: [n_heads * v_head_dim, dim] = [16384, 7168]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    wq_a_params = dim * q_lora_rank
    wq_b_params = q_lora_rank * (n_heads * qk_head_dim)
    wkv_a_params = dim * (kv_lora_rank + qk_rope_head_dim)
    wkv_b_params = kv_lora_rank * (n_heads * (qk_nope_head_dim + v_head_dim))
    wo_params = (n_heads * v_head_dim) * dim
    # q_norm, kv_norm (RMSNorm, small)
    q_norm_params = q_lora_rank  # actually it's the lora dim
    kv_norm_params = kv_lora_rank

    mla_per_layer = wq_a_params + wq_b_params + wkv_a_params + wkv_b_params + wo_params
    mla_total_params = mla_per_layer * n_layers
    mla_gib = params_to_gib(mla_total_params, BF16)
    print(f"  MLA Attention ({n_layers} layers):")
    print(f"    wq_a [{dim},{q_lora_rank}]: {wq_a_params:,}")
    print(f"    wq_b [{q_lora_rank},{n_heads * qk_head_dim}]: {wq_b_params:,}")
    print(f"    wkv_a [{dim},{kv_lora_rank + qk_rope_head_dim}]: {wkv_a_params:,}")
    print(f"    wkv_b [{kv_lora_rank},{n_heads * (qk_nope_head_dim + v_head_dim)}]: {wkv_b_params:,}")
    print(f"    wo [{n_heads * v_head_dim},{dim}]: {wo_params:,}")
    print(f"    Per layer: {mla_per_layer:,} params")
    print(f"    Total ({n_layers} layers): {mla_total_params:,} params = {fmt(mla_gib)}")

    # Dense MLP (layers 0-2)
    dense_mlp_per_layer = dim * inter_dim * 3  # w1 + w3 + w2
    dense_mlp_total = dense_mlp_per_layer * n_dense_layers
    dense_mlp_gib = params_to_gib(dense_mlp_total, BF16)
    print(f"  Dense MLP ({n_dense_layers} layers) [{dim}→{inter_dim}→{dim}]:")
    print(f"    Per layer: {dense_mlp_per_layer:,} params (w1+w3+w2)")
    print(f"    Total: {dense_mlp_total:,} params = {fmt(dense_mlp_gib)}")

    # Shared experts (1 per MoE layer, 58 layers)
    shared_per_layer = dim * moe_inter_dim * 3 * n_shared_experts  # w1+w3+w2
    shared_total = shared_per_layer * n_moe_layers
    shared_gib = params_to_gib(shared_total, BF16)
    print(f"  Shared Experts ({n_moe_layers} layers × {n_shared_experts}) [{dim}→{moe_inter_dim}→{dim}]:")
    print(f"    Per layer: {shared_per_layer:,} params")
    print(f"    Total: {shared_total:,} params = {fmt(shared_gib)}")

    # Router gates (58 layers)
    router_per_layer = n_routed_experts * dim  # [128, 7168]
    router_total = router_per_layer * n_moe_layers
    router_gib = params_to_gib(router_total, BF16)
    print(f"  Router Gates ({n_moe_layers} layers) [{n_routed_experts},{dim}]:")
    print(f"    Total: {router_total:,} params = {fmt(router_gib)}")

    # Total dense
    dense_total_params = (emb_params + lm_head_params + mla_total_params +
                          dense_mlp_total + shared_total + router_total)
    dense_total_gib = (emb_gib + lm_head_gib + norm_gib + mla_gib +
                       dense_mlp_gib + shared_gib + router_gib)
    print(f"\n  DENSE TOTAL: {dense_total_params:,} params = {fmt(dense_total_gib)}")

    # =========================================================================
    # 2. EXPERT NVFP4 BUFFERS (sharded by EP)
    # =========================================================================
    print(f"\n--- 2. EXPERT NVFP4 BUFFERS ({n_local_experts} local experts × {n_moe_layers} layers) ---")

    # Per expert weight: packed [out_dim, in_dim//2] uint8 + scale [out_dim, in_dim//group_size] fp8 + gs [1] fp32
    # W1: out_dim=2048, in_dim=7168
    # W3: out_dim=2048, in_dim=7168
    # W2: out_dim=7168, in_dim=2048

    def nvfp4_bytes(E, out_dim, in_dim):
        packed = E * out_dim * (in_dim // 2) * UINT8
        scale = E * out_dim * (in_dim // group_size) * FP8
        gs = E * FP32
        return packed, scale, gs

    w1_packed, w1_scale, w1_gs = nvfp4_bytes(n_local_experts, moe_inter_dim, dim)
    w3_packed, w3_scale, w3_gs = nvfp4_bytes(n_local_experts, moe_inter_dim, dim)
    w2_packed, w2_scale, w2_gs = nvfp4_bytes(n_local_experts, dim, moe_inter_dim)

    nvfp4_per_layer = (w1_packed + w1_scale + w1_gs +
                       w3_packed + w3_scale + w3_gs +
                       w2_packed + w2_scale + w2_gs)
    nvfp4_total = nvfp4_per_layer * n_moe_layers
    nvfp4_gib = nvfp4_total / (1024**3)

    print(f"  W1 packed [{n_local_experts},{moe_inter_dim},{dim//2}] uint8: {fmt(w1_packed / 1024**3)}")
    print(f"  W1 scale  [{n_local_experts},{moe_inter_dim},{dim//group_size}] fp8: {fmt(w1_scale / 1024**3)}")
    print(f"  W3: same as W1")
    print(f"  W2 packed [{n_local_experts},{dim},{moe_inter_dim//2}] uint8: {fmt(w2_packed / 1024**3)}")
    print(f"  W2 scale  [{n_local_experts},{dim},{moe_inter_dim//group_size}] fp8: {fmt(w2_scale / 1024**3)}")
    print(f"  Per layer: {fmt(nvfp4_per_layer / 1024**3)}")
    print(f"\n  NVFP4 TOTAL ({n_moe_layers} layers): {fmt(nvfp4_gib)}")

    # =========================================================================
    # 3. BLOCKSCALED WEIGHT CACHE
    # =========================================================================
    print(f"\n--- 3. BLOCKSCALED WEIGHT CACHE ---")

    # W13 interleaved: [E, 2*Dff, in_dim//2] uint8 + [E, 2*Dff, ceil(in_dim/32)] uint8
    # W2: [E, dim, moe_inter_dim//2] uint8 + [E, dim, ceil(moe_inter_dim/32)] uint8
    # Note: pack_factor=2 for fp8 profile means in_dim//2, and sf_k = in_dim//32
    w13_q_bytes = n_local_experts * (2 * moe_inter_dim) * (dim // 2) * UINT8
    sf_k = dim // 32
    sf_k_pad = ((sf_k + 3) // 4) * 4
    w13_sf_bytes = n_local_experts * (2 * moe_inter_dim) * sf_k_pad * UINT8
    w2_q_bytes = n_local_experts * dim * (moe_inter_dim // 2) * UINT8
    w2_sf_k = moe_inter_dim // 32
    w2_sf_k_pad = ((w2_sf_k + 3) // 4) * 4
    w2_sf_bytes = n_local_experts * dim * w2_sf_k_pad * UINT8

    cache_per_layer = w13_q_bytes + w13_sf_bytes + w2_q_bytes + w2_sf_bytes
    cache_all_layers = cache_per_layer * n_moe_layers
    cache_per_layer_gib = cache_per_layer / (1024**3)
    cache_all_gib = cache_all_layers / (1024**3)

    print(f"  W13_q [{n_local_experts},{2*moe_inter_dim},{dim//2}] uint8: {fmt(w13_q_bytes / 1024**3)}")
    print(f"  W13_sf [{n_local_experts},{2*moe_inter_dim},{sf_k_pad}] uint8: {fmt(w13_sf_bytes / 1024**3)}")
    print(f"  W2_q [{n_local_experts},{dim},{moe_inter_dim//2}] uint8: {fmt(w2_q_bytes / 1024**3)}")
    print(f"  W2_sf [{n_local_experts},{dim},{w2_sf_k_pad}] uint8: {fmt(w2_sf_bytes / 1024**3)}")
    print(f"  Per layer cache: {fmt(cache_per_layer_gib)}")
    print(f"  If ALL 58 layers cached simultaneously: {fmt(cache_all_gib)}")
    print(f"  If rebuilt per-layer (lazy, 1 at a time): {fmt(cache_per_layer_gib)}")

    # =========================================================================
    # 4. ECO OPTIMIZER STATES (FP8 m + v, persistent after step 1)
    # =========================================================================
    print(f"\n--- 4. ECO OPTIMIZER STATES (FP8, persistent after first backward) ---")

    # Per weight: exp_avg [E, in_dim, out_dim] fp8 + scale [E, in_dim, 1] fp32
    #           + exp_avg_sq [E, in_dim, out_dim] fp8 + scale [E, in_dim, 1] fp32
    # Note: in nmoe layout, shape = (E, in_dim, out_dim) — transposed from HF

    def eco_state_bytes(E, in_dim, out_dim):
        m_data = E * in_dim * out_dim * FP8
        m_scale = E * in_dim * 1 * FP32
        v_data = E * in_dim * out_dim * FP8
        v_scale = E * in_dim * 1 * FP32
        return m_data, m_scale, v_data, v_scale

    # W1: nmoe layout [E=16, in_dim=7168, out_dim=2048]
    w1_m, w1_ms, w1_v, w1_vs = eco_state_bytes(n_local_experts, dim, moe_inter_dim)
    # W3: same dims as W1
    w3_m, w3_ms, w3_v, w3_vs = eco_state_bytes(n_local_experts, dim, moe_inter_dim)
    # W2: nmoe layout [E=16, in_dim=2048, out_dim=7168]
    w2_m, w2_ms, w2_v, w2_vs = eco_state_bytes(n_local_experts, moe_inter_dim, dim)

    eco_per_layer = (w1_m + w1_ms + w1_v + w1_vs +
                     w3_m + w3_ms + w3_v + w3_vs +
                     w2_m + w2_ms + w2_v + w2_vs)
    eco_total = eco_per_layer * n_moe_layers
    eco_gib = eco_total / (1024**3)

    eco_m_total = (w1_m + w3_m + w2_m) * n_moe_layers
    eco_v_total = (w1_v + w3_v + w2_v) * n_moe_layers
    eco_scales_total = (w1_ms + w1_vs + w3_ms + w3_vs + w2_ms + w2_vs) * n_moe_layers

    print(f"  W1 exp_avg [{n_local_experts},{dim},{moe_inter_dim}] fp8: {fmt(w1_m / 1024**3)}")
    print(f"  W1 exp_avg_scale [{n_local_experts},{dim},1] fp32: {fmt(w1_ms / 1024**3)}")
    print(f"  W1 exp_avg_sq [{n_local_experts},{dim},{moe_inter_dim}] fp8: {fmt(w1_v / 1024**3)}")
    print(f"  W1 exp_avg_sq_scale [{n_local_experts},{dim},1] fp32: {fmt(w1_vs / 1024**3)}")
    print(f"  W2 exp_avg [{n_local_experts},{moe_inter_dim},{dim}] fp8: {fmt(w2_m / 1024**3)}")
    print(f"  Per layer: {fmt(eco_per_layer / 1024**3)}")
    print(f"  m total (all layers): {fmt(eco_m_total / 1024**3)}")
    print(f"  v total (all layers): {fmt(eco_v_total / 1024**3)}")
    print(f"  scales total: {fmt(eco_scales_total / 1024**3)}")
    print(f"\n  ECO STATES TOTAL ({n_moe_layers} layers × 3 weights): {fmt(eco_gib)}")

    # With factored v (Adafactor):
    # v_row [E, in_dim, 1] fp32 + v_col [E, 1, out_dim] fp32
    def factored_v_bytes(E, in_dim, out_dim):
        return E * in_dim * FP32 + E * out_dim * FP32

    fv_w1 = factored_v_bytes(n_local_experts, dim, moe_inter_dim)
    fv_w3 = factored_v_bytes(n_local_experts, dim, moe_inter_dim)
    fv_w2 = factored_v_bytes(n_local_experts, moe_inter_dim, dim)
    fv_total = (fv_w1 + fv_w3 + fv_w2) * n_moe_layers
    print(f"\n  [WHAT-IF] Factored v (Adafactor):")
    print(f"    v_row + v_col per W1: {fmt(fv_w1 / 1024**3)}")
    print(f"    Total factored v: {fmt(fv_total / 1024**3)}")
    print(f"    Savings: {fmt((eco_v_total - fv_total) / 1024**3)}")

    # =========================================================================
    # 5. ZeRO-2 DENSE OPTIMIZER STATES (FP32, sharded by DP)
    # =========================================================================
    for dp_size in [1, 8, 16]:
        print(f"\n--- 5. ZeRO-2 DENSE OPTIMIZER (FP32, DP={dp_size}) ---")

        # ZeRO-2 stores: FP32 master copy + FP32 m + FP32 v, sharded across DP ranks
        # Dense params that need optimizer state:
        dense_opt_params = dense_total_params  # All dense params
        zero2_per_rank = dense_opt_params * 3 * FP32 / dp_size  # 3 = (param + m + v)
        zero2_gib = zero2_per_rank / (1024**3)
        print(f"  Dense params: {dense_total_params:,}")
        print(f"  FP32 × 3 (param + m + v): {fmt(params_to_gib(dense_total_params * 3, FP32))}")
        print(f"  Sharded by DP={dp_size}: {fmt(zero2_gib)} per GPU")

    # =========================================================================
    # 6. BACKWARD TRANSIENT TENSORS (per-layer peak)
    # =========================================================================
    print(f"\n--- 6. BACKWARD TRANSIENT TENSORS (per-layer peak) ---")

    T = batch_size * seq_len  # tokens
    # max_pad estimate (with align=128)
    M_recv_est = T * n_activated_experts  # worst case: all tokens routed here
    max_pad_est = M_recv_est + n_local_experts * 127  # padding

    xe_pad = max_pad_est * dim * BF16
    dye_pad = max_pad_est * dim * BF16
    h1 = max_pad_est * moe_inter_dim * BF16
    h3 = max_pad_est * moe_inter_dim * BF16
    da = max_pad_est * moe_inter_dim * BF16
    dh1 = max_pad_est * moe_inter_dim * BF16
    dh3 = max_pad_est * moe_inter_dim * BF16
    a_act = max_pad_est * moe_inter_dim * BF16  # SwiGLU activation
    dx_pad = max_pad_est * dim * BF16

    # BF16 weight dequant (ONE at a time with stagger)
    bf16_weight = n_local_experts * dim * moe_inter_dim * BF16

    # FP32 gradient (for ECO update)
    fp32_grad = n_local_experts * dim * moe_inter_dim * FP32

    # dW tensor (BF16, for wgrad)
    dw_bf16 = n_local_experts * dim * moe_inter_dim * BF16

    transient_peak = (xe_pad + dye_pad + h1 + h3 + da + dh1 + dh3 + a_act +
                      dx_pad + bf16_weight + fp32_grad + dw_bf16)
    transient_gib = transient_peak / (1024**3)

    print(f"  T={T}, max_pad≈{max_pad_est}, E_local={n_local_experts}")
    print(f"  Xe_pad [{max_pad_est},{dim}] bf16: {fmt(xe_pad / 1024**3)}")
    print(f"  dYe_pad [{max_pad_est},{dim}] bf16: {fmt(dye_pad / 1024**3)}")
    print(f"  H1 [{max_pad_est},{moe_inter_dim}] bf16: {fmt(h1 / 1024**3)}")
    print(f"  H3, dA, A, dH1, dH3: each {fmt(h1 / 1024**3)}")
    print(f"  dX_pad [{max_pad_est},{dim}] bf16: {fmt(dx_pad / 1024**3)}")
    print(f"  BF16 weight (1 at a time) [{n_local_experts},{dim},{moe_inter_dim}]: {fmt(bf16_weight / 1024**3)}")
    print(f"  FP32 grad [{n_local_experts},{dim},{moe_inter_dim}]: {fmt(fp32_grad / 1024**3)}")
    print(f"  dW bf16 [{n_local_experts},{dim},{moe_inter_dim}]: {fmt(dw_bf16 / 1024**3)}")
    print(f"\n  BACKWARD TRANSIENT PEAK (one layer): {fmt(transient_gib)}")

    # =========================================================================
    # 7. SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Per-GPU Memory Breakdown")
    print("=" * 80)

    # Scenario: EP=8, DP=16 (16 nodes), cache all 58 layers
    for dp_size, cache_mode in [(1, "all"), (1, "lazy"), (16, "all"), (16, "lazy")]:
        zero2_gib = dense_total_params * 3 * FP32 / dp_size / (1024**3)
        cache_gib_used = cache_all_gib if cache_mode == "all" else cache_per_layer_gib

        total_static = dense_total_gib + nvfp4_gib + eco_gib + zero2_gib + cache_gib_used
        headroom = gpu_hbm_gib - total_static
        fits_backward = headroom > transient_gib

        print(f"\n  Scenario: EP={ep_size}, DP={dp_size}, cache={cache_mode}")
        print(f"  {'Component':<40} {'GiB':>10}")
        print(f"  {'-'*40} {'-'*10}")
        print(f"  {'Dense BF16 (replicated)':<40} {dense_total_gib:>10.2f}")
        print(f"  {'Expert NVFP4 buffers':<40} {nvfp4_gib:>10.2f}")
        print(f"  {'Blockscaled cache (' + cache_mode + ')':<40} {cache_gib_used:>10.2f}")
        print(f"  {'ECO states (m+v, FP8)':<40} {eco_gib:>10.2f}")
        print(f"  {'ZeRO-2 dense (FP32, DP=' + str(dp_size) + ')':<40} {zero2_gib:>10.2f}")
        print(f"  {'-'*40} {'-'*10}")
        print(f"  {'TOTAL STATIC':<40} {total_static:>10.2f}")
        print(f"  {'GPU HBM':<40} {gpu_hbm_gib:>10.2f}")
        print(f"  {'HEADROOM':<40} {headroom:>10.2f}")
        print(f"  {'Backward transient need':<40} {transient_gib:>10.2f}")
        print(f"  {'FITS? ':<40} {'YES' if fits_backward else 'NO <<<':>10}")

    # =========================================================================
    # 8. WHAT-IF: Factored v
    # =========================================================================
    print("\n" + "=" * 80)
    print("WHAT-IF: Factored v (Adafactor)")
    print("=" * 80)

    eco_factored_gib = (eco_m_total + eco_scales_total + fv_total) / (1024**3)
    savings = eco_gib - eco_factored_gib

    for dp_size, cache_mode in [(1, "all"), (16, "all")]:
        zero2_gib = dense_total_params * 3 * FP32 / dp_size / (1024**3)
        cache_gib_used = cache_all_gib if cache_mode == "all" else cache_per_layer_gib

        total_static = dense_total_gib + nvfp4_gib + eco_factored_gib + zero2_gib + cache_gib_used
        headroom = gpu_hbm_gib - total_static

        print(f"\n  Scenario: EP={ep_size}, DP={dp_size}, cache={cache_mode}, factored v")
        print(f"  ECO states (factored v): {fmt(eco_factored_gib)} (saved {fmt(savings)})")
        print(f"  TOTAL STATIC: {fmt(total_static)}")
        print(f"  HEADROOM: {fmt(headroom)}")
        print(f"  Backward need: {fmt(transient_gib)}")
        print(f"  FITS? {'YES' if headroom > transient_gib else 'NO'}")

    # =========================================================================
    # 9. RAW NUMBERS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RAW PARAMETER COUNTS")
    print("=" * 80)

    expert_params_per_gpu = n_local_experts * (dim * moe_inter_dim * 3) * n_moe_layers
    expert_params_total = n_routed_experts * (dim * moe_inter_dim * 3) * n_moe_layers
    total_params = dense_total_params + expert_params_total

    print(f"  Dense parameters: {dense_total_params:,} ({dense_total_params/1e9:.2f}B)")
    print(f"  Expert params per GPU: {expert_params_per_gpu:,} ({expert_params_per_gpu/1e9:.2f}B)")
    print(f"  Expert params TOTAL (all {n_routed_experts} experts): {expert_params_total:,} ({expert_params_total/1e9:.2f}B)")
    print(f"  TOTAL MODEL PARAMS: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Model in BF16 (total): {fmt(params_to_gib(total_params, BF16))}")
    print(f"  Model in NVFP4 (experts) + BF16 (dense): {fmt(dense_total_gib + nvfp4_gib * ep_size)}")


if __name__ == "__main__":
    main()
