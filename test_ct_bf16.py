"""Isolated test for ct_nvfp4_to_bf16 CUDA kernel on B200."""
import torch
import sys
sys.path.insert(0, '/home/nourdine/nmoe/nmoe/csrc')
import rdep

def test_ct_nvfp4_to_bf16():
    """Test with DeepSeek V3.2 dimensions: E=16, Dff=2048, H=7168, group_size=16."""
    device = torch.device('cuda:0')
    stream = torch.cuda.current_stream(device)

    # W1 dimensions: [E=16, out_dim=2048, in_dim=7168]
    E = 16
    M_per_expert = 2048  # Dff (out_dim)
    K = 7168             # H (in_dim)
    group_size = 16
    total_M = E * M_per_expert

    print(f"Testing ct_nvfp4_to_bf16: E={E}, M={M_per_expert}, K={K}, total_M={total_M}")
    print(f"  total elements = {total_M * K:,}")
    print(f"  grid size = {(total_M * K + 255) // 256}")

    # Create fake NVFP4 data
    packed = torch.randint(0, 256, (total_M, K // 2), dtype=torch.uint8, device=device)
    n_groups = K // group_size
    scale = torch.randint(0, 256, (total_M, n_groups), dtype=torch.uint8, device=device)
    gs = torch.ones(E, dtype=torch.float32, device=device)

    # Test 1: Non-transposed
    print("\n  Test 1: Non-transposed...")
    out = torch.empty(total_M, K, dtype=torch.bfloat16, device=device)
    rdep.ct_nvfp4_to_bf16(
        packed.data_ptr(), scale.data_ptr(), gs.data_ptr(),
        out.data_ptr(),
        total_M, K, group_size,
        1,  # gs_stride
        M_per_expert,  # expert_rows
        0,  # transpose
        stream,
    )
    torch.cuda.synchronize()
    print(f"  OK. out shape={out.shape}, out[0,:5]={out[0,:5]}")

    # Test 2: Transposed
    print("\n  Test 2: Transposed...")
    out_t = torch.empty(K, total_M, dtype=torch.bfloat16, device=device)
    rdep.ct_nvfp4_to_bf16(
        packed.data_ptr(), scale.data_ptr(), gs.data_ptr(),
        out_t.data_ptr(),
        total_M, K, group_size,
        1,  # gs_stride
        M_per_expert,  # expert_rows
        1,  # transpose
        stream,
    )
    torch.cuda.synchronize()
    print(f"  OK. out_t shape={out_t.shape}, out_t[0,:5]={out_t[0,:5]}")

    # Test 3: W2 dimensions [E=16, out_dim=7168, in_dim=2048]
    M2 = 7168
    K2 = 2048
    total_M2 = E * M2
    print(f"\n  Test 3: W2 dims total_M={total_M2}, K={K2}...")
    packed2 = torch.randint(0, 256, (total_M2, K2 // 2), dtype=torch.uint8, device=device)
    n_groups2 = K2 // group_size
    scale2 = torch.randint(0, 256, (total_M2, n_groups2), dtype=torch.uint8, device=device)
    out2 = torch.empty(K2, total_M2, dtype=torch.bfloat16, device=device)
    rdep.ct_nvfp4_to_bf16(
        packed2.data_ptr(), scale2.data_ptr(), gs.data_ptr(),
        out2.data_ptr(),
        total_M2, K2, group_size,
        1,  # gs_stride
        M2,  # expert_rows
        1,  # transpose
        stream,
    )
    torch.cuda.synchronize()
    print(f"  OK. out2 shape={out2.shape}")

    print("\nAll ct_nvfp4_to_bf16 tests PASSED.")


def test_eco_adam():
    """Test eco_adam kernel with DeepSeek V3.2 dimensions."""
    device = torch.device('cuda:0')
    stream = torch.cuda.current_stream(device)

    E = 16
    out_dim = 2048   # Dff
    in_dim = 7168    # H
    group_size = 16

    print(f"\nTesting eco_adam: E={E}, out_dim={out_dim}, in_dim={in_dim}")

    W_packed = torch.randint(0, 256, (E, out_dim, in_dim // 2), dtype=torch.uint8, device=device)
    W_scale = torch.randint(1, 200, (E, out_dim, in_dim // group_size), dtype=torch.uint8, device=device)
    W_gs = torch.ones(E, dtype=torch.float32, device=device)

    # m/v are in nmoe layout [E, in_dim, out_dim]
    m_data = torch.randint(0, 256, (E, in_dim, out_dim), dtype=torch.uint8, device=device)
    m_scale = torch.ones(E * in_dim, dtype=torch.float32, device=device)
    v_data = torch.randint(0, 256, (E, in_dim, out_dim), dtype=torch.uint8, device=device)
    v_scale = torch.ones(E * in_dim, dtype=torch.float32, device=device)

    # grad is FP32 in nmoe layout [E, in_dim, out_dim]
    grad = torch.randn(E, in_dim, out_dim, dtype=torch.float32, device=device) * 0.01

    if not hasattr(rdep, 'eco_adam_nvfp4_update'):
        print("  eco_adam_nvfp4_update not found in rdep, skipping")
        return

    rdep.eco_adam_nvfp4_update(
        W_packed.data_ptr(), W_scale.data_ptr(), W_gs.data_ptr(),
        m_data.data_ptr(), m_scale.data_ptr(), v_data.data_ptr(), v_scale.data_ptr(),
        grad.data_ptr(),
        E, out_dim, in_dim, group_size,
        1e-4,   # lr
        0.9,    # beta1
        0.999,  # beta2
        0.01,   # weight_decay
        1e-8,   # eps
        1e-4,   # step_size
        1.0,    # inv_bc2_sqrt
        0.5,    # eco_alpha
        1,      # stochastic_rounding
        1,      # error_feedback
        12345, 67890,  # prng seeds
        stream,
    )
    torch.cuda.synchronize()
    print("  eco_adam OK.")

    # Now test the fp8 row scale recomputation
    if hasattr(rdep, 'fp8_recompute_row_scale'):
        rdep.fp8_recompute_row_scale(
            m_data.data_ptr(), m_scale.data_ptr(),
            E, in_dim, out_dim,
            0,  # is_e5m2=false (E4M3 for m)
            stream,
        )
        torch.cuda.synchronize()
        print("  fp8_recompute_row_scale (m) OK.")

        rdep.fp8_recompute_row_scale(
            v_data.data_ptr(), v_scale.data_ptr(),
            E, in_dim, out_dim,
            0,  # is_e5m2=false (E4M3 for v)
            stream,
        )
        torch.cuda.synchronize()
        print("  fp8_recompute_row_scale (v) OK.")
    else:
        print("  fp8_recompute_row_scale not found, skipping")


if __name__ == '__main__':
    test_ct_nvfp4_to_bf16()
    test_eco_adam()
    print("\n=== ALL TESTS PASSED ===")
