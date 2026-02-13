#!/usr/bin/env python3
"""Minimal reproducer script for MoE backward pass CUDA launch failure.

This script exercises the backward pass of _MoEBlockscaledFused autograd Function
when fused_eco is active, without requiring distributed training (torchrun).

Usage:
    python test_backward_repro.py

    # With compute-sanitizer to detect CUDA errors:
    compute-sanitizer --tool memcheck python test_backward_repro.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

# Set environment for single-GPU testing
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import torch
import torch.nn as nn

print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. This script requires a GPU.")
    sys.exit(1)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
print(f"[INFO] Using device: {torch.cuda.get_device_name(device)}")
print(f"[INFO] Device capability: {torch.cuda.get_device_capability(device)}")

# Check for SM100 (Blackwell / B200)
cap = torch.cuda.get_device_capability(device)
if cap != (10, 0):
    print(f"[WARN] This script targets SM100 (B200). Got capability={cap}.")
    print("[WARN] Blockscaled GEMM kernels may not work on other architectures.")


# DeepSeek V3.2 MoE dimensions
E = 16          # Number of local experts
Dff = 2048      # MoE intermediate dimension
H = 7168        # Hidden dimension
group_size = 16 # NVFP4 group size
K = 8           # Top-k experts per token
batch_size = 4
seq_len = 128
T = batch_size * seq_len  # Total tokens

print(f"\n[INFO] Model dimensions:")
print(f"  E (local experts): {E}")
print(f"  Dff (moe_inter_dim): {Dff}")
print(f"  H (hidden_dim): {H}")
print(f"  group_size: {group_size}")
print(f"  K (top-k): {K}")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")
print(f"  T (total tokens): {T}")

print("\n[STEP 1] Creating fake NVFP4 weight buffers...")

# NVFP4 compressed_tensors format:
#   packed: uint8, 2 E2M1 nibbles per byte
#   scale: float8_e4m3fn (stored as uint8), one scale per group_size elements
#   global_scale: float32, one per expert

# W1/W3: [E, Dff, H/2] uint8 packed, [E, Dff, H/group_size] scale, [E] gs
# Note: HF layout is [out_features, in_features/2] -> [Dff, H/2]
W1_packed = torch.randint(0, 256, (E, Dff, H // 2), dtype=torch.uint8, device=device)
W1_scale = torch.randint(0, 256, (E, Dff, H // group_size), dtype=torch.uint8, device=device)
W1_gs = torch.ones(E, dtype=torch.float32, device=device)

W3_packed = torch.randint(0, 256, (E, Dff, H // 2), dtype=torch.uint8, device=device)
W3_scale = torch.randint(0, 256, (E, Dff, H // group_size), dtype=torch.uint8, device=device)
W3_gs = torch.ones(E, dtype=torch.float32, device=device)

# W2: [E, H, Dff/2] uint8 packed, [E, H, Dff/group_size] scale, [E] gs
# Note: HF layout is [out_features, in_features/2] -> [H, Dff/2]
W2_packed = torch.randint(0, 256, (E, H, Dff // 2), dtype=torch.uint8, device=device)
W2_scale = torch.randint(0, 256, (E, H, Dff // group_size), dtype=torch.uint8, device=device)
W2_gs = torch.ones(E, dtype=torch.float32, device=device)

# Convert scale to float8_e4m3fn view if supported
if hasattr(torch, 'float8_e4m3fn'):
    W1_scale = W1_scale.view(torch.float8_e4m3fn)
    W3_scale = W3_scale.view(torch.float8_e4m3fn)
    W2_scale = W2_scale.view(torch.float8_e4m3fn)
    print("  Using float8_e4m3fn for scales")
else:
    print("  Note: float8_e4m3fn not available, using uint8")

print(f"  W1_packed: {tuple(W1_packed.shape)} {W1_packed.dtype}")
print(f"  W1_scale: {tuple(W1_scale.shape)} {W1_scale.dtype}")
print(f"  W1_gs: {tuple(W1_gs.shape)} {W1_gs.dtype}")
print(f"  W2_packed: {tuple(W2_packed.shape)} {W2_packed.dtype}")
print(f"  W2_scale: {tuple(W2_scale.shape)} {W2_scale.dtype}")
print(f"  W2_gs: {tuple(W2_gs.shape)} {W2_gs.dtype}")

print("\n[STEP 2] Creating mock MoE module...")


@dataclass
class MockConfig:
    """Minimal config for FusedBackwardECO initialization."""
    lr_expert: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eco_stochastic_rounding: bool = True
    eco_error_feedback: bool = True


class MockMoE(nn.Module):
    """Mock MoE module with NVFP4 primary buffers."""

    def __init__(self, E: int, H: int, Dff: int, group_size: int = 16):
        super().__init__()
        self.n_local = E
        self.dim = H
        self.moe_inter_dim = Dff
        self._dtype = 'nvfp4'
        self._nvfp4_primary = True
        self._nvfp4_group_size = group_size
        self._W_cache = None
        self._fused_eco = None

        # BF16 parameters (will be populated transiently during backward)
        # In NVFP4 primary mode, these hold transient dequantized values
        self.W1 = nn.Parameter(torch.empty(E, H, Dff, dtype=torch.bfloat16, device=device))
        self.W3 = nn.Parameter(torch.empty(E, H, Dff, dtype=torch.bfloat16, device=device))
        self.W2 = nn.Parameter(torch.empty(E, Dff, H, dtype=torch.bfloat16, device=device))

        # NVFP4 compressed_tensors buffers
        self.register_buffer('_W1_packed', None, persistent=False)
        self.register_buffer('_W1_scale', None, persistent=False)
        self.register_buffer('_W1_gs', None, persistent=False)
        self.register_buffer('_W3_packed', None, persistent=False)
        self.register_buffer('_W3_scale', None, persistent=False)
        self.register_buffer('_W3_gs', None, persistent=False)
        self.register_buffer('_W2_packed', None, persistent=False)
        self.register_buffer('_W2_scale', None, persistent=False)
        self.register_buffer('_W2_gs', None, persistent=False)

    def set_nvfp4_buffers(
        self,
        W1_packed, W1_scale, W1_gs,
        W3_packed, W3_scale, W3_gs,
        W2_packed, W2_scale, W2_gs,
        group_size: int = 16,
    ):
        """Set NVFP4 buffers as primary weights."""
        self._W1_packed = W1_packed
        self._W1_scale = W1_scale
        self._W1_gs = W1_gs
        self._W3_packed = W3_packed
        self._W3_scale = W3_scale
        self._W3_gs = W3_gs
        self._W2_packed = W2_packed
        self._W2_scale = W2_scale
        self._W2_gs = W2_gs
        self._nvfp4_group_size = group_size
        self._nvfp4_primary = True

    def refresh_weight_cache(self):
        """Build blockscaled weight cache from NVFP4 buffers."""
        from nmoe.blockscaled.grouped import quantize_weights_from_nvfp4
        self._W_cache = quantize_weights_from_nvfp4(
            self._W1_packed, self._W1_scale, self._W1_gs,
            self._W3_packed, self._W3_scale, self._W3_gs,
            self._W2_packed, self._W2_scale, self._W2_gs,
            group_size=self._nvfp4_group_size,
            profile=self._dtype,
        )


# Create mock MoE module
moe = MockMoE(E, H, Dff, group_size)
moe.set_nvfp4_buffers(
    W1_packed, W1_scale, W1_gs,
    W3_packed, W3_scale, W3_gs,
    W2_packed, W2_scale, W2_gs,
    group_size=group_size,
)
print(f"  Created MockMoE with _nvfp4_primary={moe._nvfp4_primary}")

print("\n[STEP 3] Building blockscaled weight cache...")
try:
    moe.refresh_weight_cache()
    print(f"  W_cache.profile: {moe._W_cache.profile}")
    print(f"  W_cache.E: {moe._W_cache.E}")
    print(f"  W_cache.H: {moe._W_cache.H}")
    print(f"  W_cache.Dff: {moe._W_cache.Dff}")
    print(f"  W13_q shape: {tuple(moe._W_cache.W13_q.shape)}")
    print(f"  W2_q shape: {tuple(moe._W_cache.W2_q.shape)}")
except Exception as e:
    print(f"[ERROR] Failed to build weight cache: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 4] Creating FusedBackwardECO controller...")


class MockModel(nn.Module):
    """Mock model structure to hold MoE modules for FusedBackwardECO."""

    def __init__(self, moe_modules: list):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i, moe in enumerate(moe_modules):
            block = nn.Module()
            block.ffn = moe
            self.blocks.append(block)


# Create mock model with our MoE module
mock_model = MockModel([moe])
cfg = MockConfig()

try:
    from nmoe.eco import FusedBackwardECO
    fused_eco = FusedBackwardECO(mock_model, cfg)
    print(f"  Created FusedBackwardECO with {len(fused_eco._moes)} MoE modules")
except Exception as e:
    print(f"[ERROR] Failed to create FusedBackwardECO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 5] Attaching FusedBackwardECO to model (disabling gradients on expert params)...")
try:
    fused_eco.attach(mock_model)
    print(f"  W1 requires_grad: {moe.W1.requires_grad}")
    print(f"  W1 data shape: {tuple(moe.W1.data.shape) if moe.W1.data.numel() > 0 else '(freed)'}")
    print(f"  moe._fused_eco is set: {moe._fused_eco is not None}")
except Exception as e:
    print(f"[ERROR] Failed to attach FusedBackwardECO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 6] Creating Rdep dispatcher...")
capacity = T * K * 2  # Some headroom
try:
    from nmoe.rdep import Rdep
    rdep = Rdep(
        dim=H,
        n_local=E,
        topk=K,
        profile='nvfp4',
        capacity=capacity,
    )
    print(f"  Rdep mode: {rdep._mode}")
    print(f"  Rdep capacity: {rdep.capacity}")
except Exception as e:
    print(f"[ERROR] Failed to create Rdep: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 7] Creating fake input/routing data...")

# Input activations: [T, H] bfloat16
# Requires grad to trigger backward propagation
x = torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)

# Expert IDs: [T, K] int32
# Each token selects K experts from E local experts
eid = torch.randint(0, E, (T, K), dtype=torch.int32, device=device)

# Routing gates: [T, K] bfloat16
# Normalized weights (sum to 1 per token)
# Requires grad so backward computes dGates
gates = torch.rand(T, K, dtype=torch.float32, device=device)
gates = gates / gates.sum(dim=-1, keepdim=True)
gates = gates.to(torch.bfloat16).requires_grad_(True)

print(f"  x: {tuple(x.shape)} {x.dtype}")
print(f"  eid: {tuple(eid.shape)} {eid.dtype}")
print(f"  gates: {tuple(gates.shape)} {gates.dtype}")

print("\n[STEP 8] Running forward pass through _MoEBlockscaledFused.apply()...")

# Prepare for backward step
fused_eco.pre_backward(step=0)

try:
    from nmoe.moe import _MoEBlockscaledFused

    # Forward pass
    # Note: W1, W3, W2 are empty placeholders when fused_eco is active
    # The backward will dequant from NVFP4 buffers on-the-fly
    out = _MoEBlockscaledFused.apply(
        rdep, x, eid, gates,
        moe.W1, moe.W3, moe.W2,
        moe._W_cache,
        fused_eco,
        moe,  # moe_ref
    )
    print(f"  Output shape: {tuple(out.shape)} {out.dtype}")
    print(f"  Output mean: {out.float().mean().item():.6f}")
    print(f"  Output std: {out.float().std().item():.6f}")
except Exception as e:
    print(f"[ERROR] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 9] Running backward pass (this triggers the CUDA code path)...")

try:
    # Create a fake loss to trigger backward
    loss = out.sum()
    print(f"  Loss value: {loss.item():.6f}")

    # Synchronize before backward to ensure forward is complete
    torch.cuda.synchronize()
    print("  CUDA synchronized before backward")

    # Backward pass - this triggers the fused_eco code path:
    # 1. dequant_nvfp4_to_bf16_transient() for W1, W3, W2
    # 2. grouped_mm for activation/gradient computation
    # 3. bf16_wgrad kernels for weight gradients
    # 4. fused_eco.fused_update() for ECO Adam + NVFP4 requantization
    loss.backward()

    # Synchronize after backward
    torch.cuda.synchronize()
    print("  Backward pass completed successfully!")

except Exception as e:
    print(f"[ERROR] Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 10] Post-backward cleanup...")
try:
    fused_eco.post_backward()
    print(f"  prev_global_norm: {fused_eco._prev_global_norm:.6f}")
except Exception as e:
    print(f"[ERROR] post_backward failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[STEP 11] Verifying NVFP4 buffers were updated...")
print(f"  W1_packed sum: {moe._W1_packed.sum().item()}")
print(f"  W3_packed sum: {moe._W3_packed.sum().item()}")
print(f"  W2_packed sum: {moe._W2_packed.sum().item()}")

print("\n[SUCCESS] All steps completed without CUDA errors!")
print("\nTo detect subtle CUDA errors, run with compute-sanitizer:")
print("  compute-sanitizer --tool memcheck python test_backward_repro.py")
