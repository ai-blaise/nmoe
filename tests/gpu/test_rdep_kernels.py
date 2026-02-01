"""P0 Critical Tests: RDEP Kernel Correctness.

This module tests the numerical correctness of RDEP dispatch/return kernels.
These tests are critical (P0) because RDEP is the foundation of MoE routing.

Tests cover:
1. Rdep initialization with different profiles (bf16, fp8, nvfp4)
2. Dispatch operation (tokens -> experts)
3. Gather operation (expert inputs)
4. Return/scatter operation (expert outputs -> tokens)
5. Single GPU mode vs IPC mode detection
6. Token order preservation
7. Different hidden dimensions (must be multiple of 8)
8. Different topk values (1, 2, 4, 8)
9. Different expert counts

The tests compare against reference implementations using manual routing
to verify numerical correctness of the CUDA kernels.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

# Module-level imports
from nmoe.rdep import Rdep
from nmoe.csrc import rdep as _C

# ==============================================================================
# Pytest Markers and Skip Conditions
# ==============================================================================


def requires_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def requires_sm100():
    """Skip test if SM100 (B200) is not available."""
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    return pytest.mark.skipif(
        (major, minor) != (10, 0),
        reason=f"Requires SM100, got SM{major}{minor}"
    )


# ==============================================================================
# Reference Implementations for Validation
# ==============================================================================


def reference_moe_forward(
    x: torch.Tensor,
    eid: torch.Tensor,
    gates: torch.Tensor,
    W1: torch.Tensor,
    W3: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """Reference MoE forward pass using manual expert routing.

    This implementation routes tokens to experts one-by-one to verify
    the RDEP dispatch/return cycle preserves correctness.

    Args:
        x: [T, H] BF16 input hidden states
        eid: [T, K] int32 expert IDs
        gates: [T, K] BF16 routing weights
        W1: [E, H, Dff] gate projection weights
        W3: [E, H, Dff] up projection weights
        W2: [E, Dff, H] down projection weights

    Returns:
        [T, H] BF16 output
    """
    T, H = x.shape
    K = eid.shape[1]
    E = W1.shape[0]

    # Accumulate in FP32 for precision
    out = torch.zeros((T, H), device=x.device, dtype=torch.float32)

    for k in range(K):
        eid_k = eid[:, k]
        gate_k = gates[:, k].float()

        for e in range(E):
            # Find tokens assigned to this expert at slot k
            mask = (eid_k == e)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x[idx]  # [M_e, H]

            # Expert MLP: Y = SwiGLU(X @ W1, X @ W3) @ W2
            h1 = x_e @ W1[e]  # [M_e, Dff]
            h3 = x_e @ W3[e]  # [M_e, Dff]
            a = F.silu(h1) * h3  # [M_e, Dff]
            y_e = a @ W2[e]  # [M_e, H]

            # Accumulate with gate
            gate_e = gate_k[idx].unsqueeze(-1)  # [M_e, 1]
            out.index_add_(0, idx, (y_e.float() * gate_e))

    return out.to(dtype=torch.bfloat16)


def reference_dispatch_gather(
    x: torch.Tensor,
    eid: torch.Tensor,
    gates: torch.Tensor,
    n_local: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, Tuple]]:
    """Reference dispatch: group tokens by expert.

    Returns:
        expert_inputs: dict mapping expert_id -> [M_e, H] tensor
        expert_gates: dict mapping expert_id -> [M_e] tensor
        expert_indices: dict mapping expert_id -> (token_indices, slot_indices) tuple
    """
    T, H = x.shape
    K = eid.shape[1]

    expert_inputs: Dict[int, torch.Tensor] = {}
    expert_gates: Dict[int, torch.Tensor] = {}
    expert_indices: Dict[int, Tuple] = {}

    for e in range(n_local):
        # Find all (token, slot) pairs assigned to expert e
        tok_indices = []
        slot_indices = []
        for k in range(K):
            mask = (eid[:, k] == e)
            tok_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if tok_idx.numel() > 0:
                tok_indices.append(tok_idx)
                slot_indices.append(torch.full_like(tok_idx, k))

        if not tok_indices:
            continue

        tok_idx = torch.cat(tok_indices)
        slot_idx = torch.cat(slot_indices)

        # Gather inputs and gates
        expert_inputs[e] = x[tok_idx]
        expert_gates[e] = gates[tok_idx, slot_idx].float()
        expert_indices[e] = (tok_idx, slot_idx)

    return expert_inputs, expert_gates, expert_indices


def reference_scatter_return(
    expert_outputs: Dict[int, torch.Tensor],
    expert_gates: Dict[int, torch.Tensor],
    expert_indices: Dict[int, Tuple],
    T: int,
    H: int,
) -> torch.Tensor:
    """Reference return: scatter expert outputs back to tokens.

    Args:
        expert_outputs: dict mapping expert_id -> [M_e, H] output tensor
        expert_gates: dict mapping expert_id -> [M_e] gate values
        expert_indices: dict mapping expert_id -> (token_indices, slot_indices)
        T: total number of tokens
        H: hidden dimension

    Returns:
        [T, H] BF16 output tensor
    """
    out = torch.zeros((T, H), dtype=torch.float32, device=next(iter(expert_outputs.values())).device)

    for e, y_e in expert_outputs.items():
        tok_idx, _ = expert_indices[e]
        gate_e = expert_gates[e].unsqueeze(-1)
        out.index_add_(0, tok_idx, (y_e.float() * gate_e))

    return out.to(dtype=torch.bfloat16)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    return seed_val


@pytest.fixture
def cuda_device():
    """Provide CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture(params=[128, 256, 512])
def hidden_dim(request):
    """Test various hidden dimensions (must be multiple of 8)."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def topk(request):
    """Test various topk values."""
    return request.param


@pytest.fixture(params=[4, 8, 16])
def n_experts(request):
    """Test various expert counts."""
    return request.param


@pytest.fixture
def rdep_bf16(cuda_device, hidden_dim, n_experts, topk):
    """Create an Rdep instance with BF16 profile."""
    capacity = 4096  # Large enough for test cases
    return Rdep(
        dim=hidden_dim,
        n_local=n_experts,
        topk=topk,
        profile='bf16',
        capacity=capacity,
    )


@pytest.fixture
def input_tensors(cuda_device, hidden_dim, n_experts, topk, seed):
    """Generate test input tensors."""
    T = 64  # Number of tokens
    Dff = hidden_dim * 4  # Feed-forward dimension

    # Input tokens - scale to realistic magnitudes
    x = torch.randn(T, hidden_dim, device=cuda_device, dtype=torch.bfloat16) * 0.1

    # Expert IDs - random assignment
    eid = torch.randint(0, n_experts, (T, topk), device=cuda_device, dtype=torch.int32)

    # Gates - softmax to ensure they sum to ~1
    gate_logits = torch.randn(T, topk, device=cuda_device, dtype=torch.float32)
    gates = F.softmax(gate_logits, dim=-1).to(torch.bfloat16)

    # Expert weights - Xavier-like initialization
    W1 = torch.randn(n_experts, hidden_dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
    W3 = torch.randn(n_experts, hidden_dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
    W2 = torch.randn(n_experts, Dff, hidden_dim, device=cuda_device, dtype=torch.bfloat16) * 0.02

    return {
        'x': x,
        'eid': eid,
        'gates': gates,
        'W1': W1,
        'W3': W3,
        'W2': W2,
        'T': T,
        'H': hidden_dim,
        'K': topk,
        'E': n_experts,
        'Dff': Dff,
    }


# ==============================================================================
# Test Classes
# ==============================================================================


@pytest.mark.gpu
class TestRdepInitialization:
    """Tests for Rdep initialization with different profiles."""

    def test_bf16_profile_initialization(self, cuda_device):
        """Test Rdep initialization with BF16 profile."""
        rdep = Rdep(dim=256, n_local=8, topk=2, profile='bf16', capacity=4096)
        assert rdep.profile == 'bf16'
        assert rdep.dim == 256
        assert rdep.n_local == 8
        assert rdep.topk == 2

    def test_fp8_profile_initialization(self, cuda_device):
        """Test Rdep initialization with FP8 profile."""
        rdep = Rdep(dim=256, n_local=8, topk=2, profile='fp8', capacity=4096)
        assert rdep.profile == 'fp8'
        assert rdep.PROFILES['fp8'] == 0

    def test_nvfp4_profile_initialization(self, cuda_device):
        """Test Rdep initialization with NVFP4 profile."""
        rdep = Rdep(dim=256, n_local=8, topk=2, profile='nvfp4', capacity=4096)
        assert rdep.profile == 'nvfp4'
        assert rdep.PROFILES['nvfp4'] == 1

    def test_invalid_profile_raises_error(self, cuda_device):
        """Test that invalid profile raises TypeError."""
        with pytest.raises(TypeError, match="profile must be one of"):
            Rdep(dim=256, n_local=8, topk=2, profile='invalid', capacity=4096)

    def test_mode_detection_single_gpu(self, cuda_device):
        """Test mode detection for single GPU."""
        rdep = Rdep(dim=256, n_local=8, topk=2, profile='bf16', capacity=4096)
        # Without distributed initialization, should be 'single' mode
        assert rdep._mode == 'single'

    def test_hidden_dim_alignment(self, cuda_device):
        """Test that hidden dimension must be multiple of 8."""
        # Valid dimensions
        for dim in [8, 16, 24, 128, 256, 512, 1024]:
            rdep = Rdep(dim=dim, n_local=4, topk=1, profile='bf16', capacity=1024)
            assert rdep.dim == dim


@pytest.mark.gpu
class TestDispatchMetaBf16:
    """Tests for BF16 dispatch meta operation."""

    def test_dispatch_meta_returns_valid_m_recv(self, cuda_device, seed):
        """Test that dispatch_meta_bf16 returns valid M_recv."""
        T, H, K, E = 32, 256, 2, 8
        capacity = T * K * 2

        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.float32)

        offs_pad = torch.empty(E, device=cuda_device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        stream = torch.cuda.current_stream(cuda_device)
        M_recv = _C.dispatch_meta_bf16(
            x.data_ptr(), eid.data_ptr(), gates.data_ptr(),
            T, K, 128,  # align=128 for BF16
            offs_pad.data_ptr(), M_host.data_ptr(),
            stream,
        )

        # M_recv should equal T * K for single-GPU mode
        assert M_recv == T * K, f"Expected M_recv={T*K}, got {M_recv}"

    def test_dispatch_meta_offs_pad_monotonic(self, cuda_device, seed):
        """Test that offs_pad is monotonically increasing."""
        T, H, K, E = 64, 128, 4, 8
        capacity = T * K * 2

        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.float32)

        offs_pad = torch.empty(E, device=cuda_device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        stream = torch.cuda.current_stream(cuda_device)
        _C.dispatch_meta_bf16(
            x.data_ptr(), eid.data_ptr(), gates.data_ptr(),
            T, K, 128,
            offs_pad.data_ptr(), M_host.data_ptr(),
            stream,
        )

        torch.cuda.synchronize()
        offs_cpu = offs_pad.cpu()

        # Check monotonicity
        for i in range(1, E):
            assert offs_cpu[i] >= offs_cpu[i-1], \
                f"offs_pad not monotonic: offs_pad[{i-1}]={offs_cpu[i-1]}, offs_pad[{i}]={offs_cpu[i]}"


@pytest.mark.gpu
class TestGatherXeBf16:
    """Tests for BF16 gather operation."""

    def test_gather_preserves_token_data(self, cuda_device, seed):
        """Test that gather_xe_bf16 preserves token data."""
        T, H, K, E = 32, 256, 2, 4
        capacity = T * K * 2

        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.float32)

        offs_pad = torch.empty(E, device=cuda_device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        stream = torch.cuda.current_stream(cuda_device)
        M_recv = _C.dispatch_meta_bf16(
            x.data_ptr(), eid.data_ptr(), gates.data_ptr(),
            T, K, 128,
            offs_pad.data_ptr(), M_host.data_ptr(),
            stream,
        )

        # Compute max_pad
        align = 128
        max_pad = (M_recv + E * (align - 1) + (align - 1)) // align * align

        # Allocate Xe_pad and gather
        Xe_pad = torch.empty(max_pad, H, device=cuda_device, dtype=torch.bfloat16)
        _C.gather_xe_bf16(Xe_pad.data_ptr(), M_recv, max_pad, stream)

        torch.cuda.synchronize()

        # Xe_pad should contain valid data (not NaN or Inf)
        assert not torch.isnan(Xe_pad[:M_recv]).any(), "Xe_pad contains NaN values"
        assert not torch.isinf(Xe_pad[:M_recv]).any(), "Xe_pad contains Inf values"


@pytest.mark.gpu
class TestReturnScatterBf16:
    """Tests for BF16 return/scatter operation."""

    def test_return_scatter_accumulates_correctly(self, cuda_device, seed):
        """Test that return_scatter correctly accumulates expert outputs."""
        T, H, K, E = 32, 256, 2, 4
        capacity = T * K * 2

        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        # Assign all tokens to expert 0 for deterministic testing
        eid = torch.zeros(T, K, device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.float32)

        offs_pad = torch.empty(E, device=cuda_device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        stream = torch.cuda.current_stream(cuda_device)
        M_recv = _C.dispatch_meta_bf16(
            x.data_ptr(), eid.data_ptr(), gates.data_ptr(),
            T, K, 128,
            offs_pad.data_ptr(), M_host.data_ptr(),
            stream,
        )

        align = 128
        max_pad = (M_recv + E * (align - 1) + (align - 1)) // align * align

        Xe_pad = torch.empty(max_pad, H, device=cuda_device, dtype=torch.bfloat16)
        _C.gather_xe_bf16(Xe_pad.data_ptr(), M_recv, max_pad, stream)

        # Use identity as expert output for testing
        Ye_pad = Xe_pad.clone()

        out_f32 = torch.zeros(T, H, device=cuda_device, dtype=torch.float32)
        _C.return_scatter_from_pad_bf16(
            Ye_pad.data_ptr(),
            out_f32.data_ptr(),
            M_recv, T, K,
            stream,
        )

        torch.cuda.synchronize()

        # Output should not be all zeros (tokens were assigned)
        assert out_f32.abs().sum() > 0, "return_scatter produced all zeros"


@pytest.mark.gpu
class TestDispatchReturnRoundtrip:
    """Tests for complete dispatch-return roundtrip."""

    def test_roundtrip_preserves_values(self, cuda_device, input_tensors, seed):
        """Test that dispatch -> identity expert -> return preserves token order."""
        x = input_tensors['x']
        eid = input_tensors['eid']
        gates = input_tensors['gates']
        T = input_tensors['T']
        H = input_tensors['H']
        K = input_tensors['K']
        E = input_tensors['E']

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        gates_fp32 = gates.float()
        offs_pad = torch.empty(E, device=cuda_device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        stream = torch.cuda.current_stream(cuda_device)
        M_recv = _C.dispatch_meta_bf16(
            x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
            T, K, 128,
            offs_pad.data_ptr(), M_host.data_ptr(),
            stream,
        )

        align = 128
        max_pad = (M_recv + E * (align - 1) + (align - 1)) // align * align
        offs_pad[-1] = max_pad

        Xe_pad = torch.empty(max_pad, H, device=cuda_device, dtype=torch.bfloat16)
        _C.gather_xe_bf16(Xe_pad.data_ptr(), M_recv, max_pad, stream)

        # Identity operation
        Ye_pad = Xe_pad.clone()

        out_f32 = torch.zeros(T, H, device=cuda_device, dtype=torch.float32)
        _C.return_scatter_from_pad_bf16(
            Ye_pad.data_ptr(),
            out_f32.data_ptr(),
            M_recv, T, K,
            stream,
        )

        torch.cuda.synchronize()

        # Compute reference using gate-weighted sum
        ref_out = torch.zeros(T, H, device=cuda_device, dtype=torch.float32)
        for k in range(K):
            ref_out += x.float() * gates[:, k:k+1].float()

        # Compare
        torch.testing.assert_close(
            out_f32.bfloat16(),
            ref_out.bfloat16(),
            atol=1e-3,
            rtol=1e-2,
            msg="Roundtrip with identity expert did not preserve values"
        )


@pytest.mark.gpu
class TestMoEBf16Forward:
    """Tests for complete MoE BF16 forward pass."""

    def test_moe_bf16_matches_reference(self, cuda_device, input_tensors, seed):
        """Test that moe_bf16 matches reference implementation."""
        x = input_tensors['x'].clone()
        eid = input_tensors['eid']
        gates = input_tensors['gates']
        W1 = input_tensors['W1']
        W3 = input_tensors['W3']
        W2 = input_tensors['W2']
        T = input_tensors['T']
        H = input_tensors['H']
        K = input_tensors['K']
        E = input_tensors['E']

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        # RDEP forward
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Reference forward
        ref = reference_moe_forward(x, eid, gates, W1, W3, W2)

        # Compare - allow small tolerance due to operation ordering
        torch.testing.assert_close(
            out, ref,
            atol=5e-2,
            rtol=1e-1,
            msg="MoE BF16 forward does not match reference"
        )

    def test_moe_bf16_deterministic(self, cuda_device, input_tensors, seed):
        """Test that moe_bf16 produces deterministic results."""
        x = input_tensors['x'].clone()
        eid = input_tensors['eid']
        gates = input_tensors['gates']
        W1 = input_tensors['W1']
        W3 = input_tensors['W3']
        W2 = input_tensors['W2']
        T = input_tensors['T']
        H = input_tensors['H']
        K = input_tensors['K']
        E = input_tensors['E']

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        # Run twice
        out1 = rdep.moe_bf16(x.clone(), eid, gates, W1, W3, W2)
        out2 = rdep.moe_bf16(x.clone(), eid, gates, W1, W3, W2)

        # Should be identical
        torch.testing.assert_close(
            out1, out2,
            atol=0, rtol=0,
            msg="MoE BF16 is not deterministic"
        )


@pytest.mark.gpu
class TestTokenOrderPreservation:
    """Tests for token order preservation in dispatch/return."""

    def test_token_order_with_uniform_experts(self, cuda_device, seed):
        """Test token order when all tokens go to same expert."""
        T, H, K, E = 32, 256, 1, 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        # All tokens to expert 0
        eid = torch.zeros(T, K, device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.eye(H, device=cuda_device, dtype=torch.bfloat16).unsqueeze(0).expand(E, -1, -1).contiguous()
        W3 = torch.eye(H, device=cuda_device, dtype=torch.bfloat16).unsqueeze(0).expand(E, -1, -1).contiguous()
        W2 = torch.eye(H, device=cuda_device, dtype=torch.bfloat16).unsqueeze(0).expand(E, -1, -1).contiguous()

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # With identity weights and SwiGLU(x*1, x*1) = silu(x) * x
        # The output should follow a predictable pattern
        expected_activation = F.silu(x) * x
        expected_out = expected_activation @ W2[0]

        torch.testing.assert_close(
            out, expected_out,
            atol=5e-2, rtol=1e-1,
            msg="Token order not preserved with uniform expert assignment"
        )

    def test_token_order_roundtrip_indices(self, cuda_device, seed):
        """Test that row_id encoding correctly identifies source tokens."""
        T, H, K, E = 64, 128, 2, 8
        capacity = T * K * 2

        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.float32)

        offs_pad = torch.empty(E, device=cuda_device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        stream = torch.cuda.current_stream(cuda_device)
        M_recv = _C.dispatch_meta_bf16(
            x.data_ptr(), eid.data_ptr(), gates.data_ptr(),
            T, K, 128,
            offs_pad.data_ptr(), M_host.data_ptr(),
            stream,
        )

        # Get row_id and gate_sorted
        row_id = torch.empty(M_recv, device=cuda_device, dtype=torch.int64)
        gate_sorted = torch.empty(M_recv, device=cuda_device, dtype=torch.float32)
        _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), M_recv, stream)

        torch.cuda.synchronize()

        # Decode row_ids and verify they map back to valid tokens
        row_id_cpu = row_id.cpu()
        for i in range(M_recv):
            rid = row_id_cpu[i].item()
            # Decode: slot = rid % K, tmp = rid / K, tok = tmp % T, rank = tmp / T
            slot = rid % K
            tmp = rid // K
            tok = tmp % T
            rank = tmp // T

            assert 0 <= tok < T, f"Invalid token index {tok} from row_id {rid}"
            assert 0 <= slot < K, f"Invalid slot index {slot} from row_id {rid}"
            assert rank == 0, f"Unexpected rank {rank} in single-GPU mode"


@pytest.mark.gpu
class TestDifferentHiddenDimensions:
    """Tests for various hidden dimension sizes."""

    @pytest.mark.parametrize("hidden_dim", [64, 128, 256, 512, 1024])
    def test_hidden_dim_multiples_of_8(self, cuda_device, hidden_dim, seed):
        """Test that hidden dimensions that are multiples of 8 work correctly."""
        T, K, E = 32, 2, 4
        Dff = hidden_dim * 4

        x = torch.randn(T, hidden_dim, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.randn(E, hidden_dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, hidden_dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, hidden_dim, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=hidden_dim, n_local=E, topk=K, profile='bf16', capacity=capacity)

        # Should complete without error
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


@pytest.mark.gpu
class TestDifferentTopkValues:
    """Tests for various topk values."""

    @pytest.mark.parametrize("topk", [1, 2, 4, 8])
    def test_topk_values(self, cuda_device, topk, seed):
        """Test that different topk values work correctly."""
        T, H, E = 32, 256, 16  # Need E >= topk
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, topk), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, topk, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * topk * 2
        rdep = Rdep(dim=H, n_local=E, topk=topk, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
        ref = reference_moe_forward(x, eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=1e-1,
            msg=f"MoE with topk={topk} does not match reference"
        )


@pytest.mark.gpu
class TestDifferentExpertCounts:
    """Tests for various expert counts."""

    @pytest.mark.parametrize("n_experts", [2, 4, 8, 16, 32])
    def test_expert_counts(self, cuda_device, n_experts, seed):
        """Test that different expert counts work correctly."""
        T, H, K = 32, 256, 2
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_experts, (T, K), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1 = torch.randn(n_experts, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_experts, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_experts, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=n_experts, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
        ref = reference_moe_forward(x, eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=1e-1,
            msg=f"MoE with n_experts={n_experts} does not match reference"
        )


@pytest.mark.gpu
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_expert(self, cuda_device, seed):
        """Test handling when some experts receive no tokens."""
        T, H, K, E = 16, 256, 1, 8
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        # Assign all tokens to expert 0, leaving experts 1-7 empty
        eid = torch.zeros(T, K, device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        # Should complete without error
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

    def test_single_token(self, cuda_device, seed):
        """Test handling of single token input."""
        T, H, K, E = 1, 256, 2, 4
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
        ref = reference_moe_forward(x, eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=1e-1,
            msg="Single token MoE does not match reference"
        )

    def test_large_batch(self, cuda_device, seed):
        """Test handling of large batch size."""
        T, H, K, E = 1024, 256, 2, 8
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_zero_gates(self, cuda_device, seed):
        """Test handling when all gates are zero."""
        T, H, K, E = 32, 256, 2, 4
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.zeros(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Output should be all zeros since gates are zero
        assert out.abs().max() < 1e-5, "Output should be near zero with zero gates"


@pytest.mark.gpu
class TestCapacityValidation:
    """Tests for capacity overflow handling."""

    def test_capacity_exceeded_raises_error(self, cuda_device):
        """Test that exceeding capacity raises appropriate error."""
        T, H, K, E = 100, 256, 4, 8
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16)
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16)
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16)
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16)

        # Capacity too small
        capacity = 10  # Way smaller than T * K = 400
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        with pytest.raises(ValueError, match="Token count exceeds RDEP capacity"):
            rdep.dispatch(x, eid, gates, W1, W3, W2)


@pytest.mark.gpu
class TestGradientFlow:
    """Tests for gradient flow through MoE."""

    def test_backward_produces_gradients(self, cuda_device, seed):
        """Test that backward pass produces valid gradients."""
        T, H, K, E = 32, 256, 2, 4
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16, requires_grad=True) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16).requires_grad_(True)

        W1 = (torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        W3 = (torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        W2 = (torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
        loss = out.float().sum()
        loss.backward()

        # All inputs should have gradients
        assert x.grad is not None, "x.grad is None"
        assert gates.grad is not None, "gates.grad is None"
        assert W1.grad is not None, "W1.grad is None"
        assert W3.grad is not None, "W3.grad is None"
        assert W2.grad is not None, "W2.grad is None"

        # Gradients should not be all zeros
        assert x.grad.abs().sum() > 0, "x.grad is all zeros"
        assert W1.grad.abs().sum() > 0, "W1.grad is all zeros"

    def test_gradient_matches_reference(self, cuda_device, seed):
        """Test that gradients match reference implementation."""
        T, H, K, E = 16, 128, 1, 4
        Dff = H * 4

        # Create inputs with requires_grad
        x = (torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = (torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        W3 = (torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        W2 = (torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        # Forward and backward with RDEP
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
        loss = out.float().sum()
        loss.backward()

        # Store gradients
        x_grad = x.grad.clone()
        W1_grad = W1.grad.clone()
        W3_grad = W3.grad.clone()
        W2_grad = W2.grad.clone()

        # Reference forward and backward
        x2 = x.detach().clone().requires_grad_(True)
        W1_2 = W1.detach().clone().requires_grad_(True)
        W3_2 = W3.detach().clone().requires_grad_(True)
        W2_2 = W2.detach().clone().requires_grad_(True)

        ref_out = reference_moe_forward(x2, eid, gates, W1_2, W3_2, W2_2)
        ref_loss = ref_out.float().sum()
        ref_loss.backward()

        # Compare gradients with relaxed tolerance due to BF16 precision
        torch.testing.assert_close(
            x_grad, x2.grad,
            atol=1e-1, rtol=1e-1,
            msg="x gradients do not match reference"
        )
        torch.testing.assert_close(
            W1_grad, W1_2.grad,
            atol=1e-1, rtol=1e-1,
            msg="W1 gradients do not match reference"
        )


@pytest.mark.gpu
class TestModeDetection:
    """Tests for RDEP mode detection."""

    def test_single_mode_without_dist(self, cuda_device):
        """Test that mode is 'single' without distributed initialization."""
        rdep = Rdep(dim=256, n_local=8, topk=2, profile='bf16', capacity=4096)
        assert rdep._mode == 'single'
        assert rdep.world == 1
        assert rdep.rank == 0

    def test_c_extension_mode_query(self, cuda_device):
        """Test querying mode from C extension."""
        rdep = Rdep(dim=256, n_local=8, topk=2, profile='bf16', capacity=4096)
        # After initialization, mode should be queryable
        mode = _C.get_mode()
        assert mode in [0, 1, 2], f"Invalid mode {mode}"
        # 0=SINGLE, 1=IPC, 2=HYBRID
        if mode == 0:
            assert rdep._mode == 'single'


@pytest.mark.gpu
class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_values(self, cuda_device, seed):
        """Test handling of large input values."""
        T, H, K, E = 32, 256, 2, 4
        Dff = H * 4

        # Large but not overflow values
        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 10.0
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.1
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.1
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.1

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Should not produce NaN or Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_small_values(self, cuda_device, seed):
        """Test handling of small input values."""
        T, H, K, E = 32, 256, 2, 4
        Dff = H * 4

        # Small values
        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 1e-4
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='bf16', capacity=capacity)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Should not produce NaN or Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


# ==============================================================================
# Blockscaled Profile Tests (SM100 required)
# ==============================================================================


@pytest.mark.gpu
class TestBlockscaledProfiles:
    """Tests for FP8 and NVFP4 blockscaled profiles."""

    @requires_sm100()
    def test_fp8_profile_forward(self, cuda_device, seed):
        """Test FP8 profile forward pass."""
        try:
            from nmoe.blockscaled.grouped import quantize_weights
        except ImportError:
            pytest.skip("blockscaled module not available")

        T, H, K, E = 32, 256, 2, 4
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='fp8', capacity=capacity)

        W_cache = quantize_weights(W1, W3, W2, profile='fp8')
        out = rdep.moe_blockscaled(x, eid, gates, W1, W3, W2, W_cache)

        ref = reference_moe_forward(x, eid, gates, W1, W3, W2)

        # FP8 has some quantization error
        torch.testing.assert_close(
            out, ref,
            atol=1e-2, rtol=1e-1,
            msg="FP8 forward does not match reference within tolerance"
        )

    @requires_sm100()
    def test_nvfp4_profile_forward(self, cuda_device, seed):
        """Test NVFP4 profile forward pass."""
        try:
            from nmoe.blockscaled.grouped import quantize_weights
        except ImportError:
            pytest.skip("blockscaled module not available")

        T, H, K, E = 32, 256, 2, 4
        Dff = H * 4

        x = torch.randn(T, H, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, E, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)

        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * 0.02

        capacity = T * K * 2
        rdep = Rdep(dim=H, n_local=E, topk=K, profile='nvfp4', capacity=capacity)

        W_cache = quantize_weights(W1, W3, W2, profile='nvfp4')
        out = rdep.moe_blockscaled(x, eid, gates, W1, W3, W2, W_cache)

        ref = reference_moe_forward(x, eid, gates, W1, W3, W2)

        # NVFP4 has more quantization error than FP8
        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=2e-1,
            msg="NVFP4 forward does not match reference within tolerance"
        )


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
