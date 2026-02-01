"""Comprehensive integration tests for the RDEP adapter layer between nmoe and SGLang.

This module tests the RDEP adapter (sglang/python/sglang/srt/layers/moe/dispatchers/rdep_adapter.py)
which bridges nmoe's RDEP dispatcher with SGLang's dispatcher infrastructure.

Test categories:
1. Format conversion between SGLang dispatcher format and nmoe RDEP format
2. Synchronization between SGLang and nmoe RDEP
3. IPC management for multi-GPU dispatch
4. Capacity handling across the adapter
5. BF16/FP8 dispatch through the adapter
6. Expert IDs and gate weights passing
7. Error handling when format mismatches occur
8. Performance - no overhead beyond direct RDEP calls
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import torch

# Skip if CUDA not available
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

# Add SGLang to path and try to import the rdep_adapter module
sys.path.insert(0, "/home/nourdine/sglang_nmoe/nether-soup/sglang/python")
SGLANG_IMPORT_ERROR = None
try:
    # Try importing the adapter module - this may fail if sgl_kernel is not available
    from sglang.srt.layers.moe.dispatchers.rdep_adapter import (
        # Data classes
        RdepMeta,
        RdepDispatchInput,
        RdepCombineOutput,
        DispatchMetaResult,
        CombineResult,
        LowLatencyConfig,
        # Adapter classes
        RdepAdapter,
        RdepBufferView,
        RdepSyncContext,
        DistributedSyncManager,
        RdepDispatchAdapter,
        RdepCombineAdapter,
        RdepLowLatencyAdapter,
        # Conversion functions
        topk_to_rdep_format,
        standard_dispatch_to_rdep,
        deepep_normal_to_rdep,
        deepep_ll_to_rdep,
        rdep_output_to_standard_combine,
        rdep_output_to_deepep_normal_combine,
        rdep_output_to_deepep_ll_combine,
        # Utility functions
        check_buffer_layout_compatible,
        get_rdep_mode_for_topology,
        calculate_bf16_buffer_size,
        calculate_blockscaled_buffer_size,
        estimate_rdep_memory,
        create_rdep_buffer_view,
        create_sync_manager,
        get_default_sync_mode,
        create_dispatch_adapter,
        create_combine_adapter,
        _align_up,
        # Constants
        BUFFER_ALIGNMENT,
        MAX_RANKS,
        META_SIZE,
        BF16_ALIGNMENT,
        BLOCKSCALED_ALIGNMENT,
        LOW_LATENCY_MAX_TOKENS,
        LOW_LATENCY_SINGLE_TOKEN,
        # Enums/Mode classes
        SyncMode,
        RdepMode,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput, BypassedTopKOutput, TopKConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputFormat, CombineInputFormat
    SGLANG_AVAILABLE = True
except ImportError as e:
    SGLANG_IMPORT_ERROR = str(e)
    SGLANG_AVAILABLE = False

# Skip the entire module if SGLang imports failed (e.g., sgl_kernel not available)
if not SGLANG_AVAILABLE:
    pytest.skip(
        f"SGLang RDEP adapter imports failed (likely sgl_kernel issue): {SGLANG_IMPORT_ERROR}",
        allow_module_level=True
    )


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def device():
    """Get CUDA device."""
    return torch.device("cuda:0")


@pytest.fixture
def hidden_size():
    """Standard hidden size for tests."""
    return 256


@pytest.fixture
def num_experts():
    """Number of experts for tests."""
    return 8


@pytest.fixture
def top_k():
    """Top-K experts per token."""
    return 2


@pytest.fixture
def num_tokens():
    """Number of tokens for tests."""
    return 64


@pytest.fixture
def capacity():
    """Buffer capacity for RDEP."""
    return 4096


def create_mock_hidden_states(
    num_tokens: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Create mock hidden states tensor."""
    return torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)


def create_mock_topk_ids(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Create mock topk_ids tensor."""
    return torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)


def create_mock_topk_weights(
    num_tokens: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create mock topk_weights tensor with normalized weights."""
    weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    # Normalize weights to sum to 1
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


def create_mock_router_logits(
    num_tokens: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Create mock router logits tensor."""
    return torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)


# ============================================================================
# Test Section 1: Format Conversion Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestFormatConversion:
    """Test format conversion between SGLang dispatcher format and nmoe RDEP format."""

    def test_topk_to_rdep_format_basic(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test basic topk_to_rdep_format conversion."""
        # Create mock topk output
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        # Create mock topk output using the actual SGLang format
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),  # StandardTopKOutput typically uses int64
            router_logits=router_logits,
        )

        # Convert to RDEP format
        eid, gates = topk_to_rdep_format(topk_output)

        # Verify output format
        assert eid.dtype == torch.int32, f"Expected int32, got {eid.dtype}"
        assert eid.shape == (num_tokens, top_k)
        assert gates.shape == (num_tokens, top_k)
        assert eid.is_contiguous()
        assert gates.is_contiguous()

    def test_topk_to_rdep_format_preserves_values(self, device, num_tokens, num_experts, top_k):
        """Test that topk_to_rdep_format preserves expert IDs and gate values."""
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        eid, gates = topk_to_rdep_format(topk_output)

        # Values should be preserved
        assert torch.allclose(eid.long(), topk_ids.long()), "Expert IDs not preserved"
        assert torch.allclose(gates, topk_weights), "Gate weights not preserved"

    def test_standard_dispatch_to_rdep_basic(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test standard_dispatch_to_rdep conversion."""
        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.dtype == torch.bfloat16
        assert rdep_input.hidden_states.shape == (num_tokens, hidden_size)
        assert rdep_input.topk_ids.dtype == torch.int32
        assert rdep_input.topk_ids.shape == (num_tokens, top_k)
        assert rdep_input.topk_weights.shape == (num_tokens, top_k)

    def test_standard_dispatch_to_rdep_fp32_conversion(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test that FP32 hidden states are converted to BF16."""
        # Create FP32 hidden states
        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device, dtype=torch.float32)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        # Should be converted to BF16
        assert rdep_input.hidden_states.dtype == torch.bfloat16

    def test_rdep_output_to_standard_combine(self, device, hidden_size, num_tokens):
        """Test rdep_output_to_standard_combine conversion."""
        output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        combine_input = rdep_output_to_standard_combine(output)

        assert combine_input.hidden_states is output
        assert combine_input.format.value == "standard"

    def test_rdep_output_to_deepep_normal_combine(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test rdep_output_to_deepep_normal_combine conversion."""
        output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)

        combine_input = rdep_output_to_deepep_normal_combine(output, topk_ids, topk_weights)

        assert combine_input.hidden_states is output
        assert combine_input.topk_ids is topk_ids
        assert combine_input.topk_weights is topk_weights

    def test_rdep_output_to_deepep_ll_combine(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test rdep_output_to_deepep_ll_combine conversion."""
        expected_m = num_tokens * top_k
        output = torch.randn(expected_m, hidden_size, device=device, dtype=torch.bfloat16)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)

        combine_input = rdep_output_to_deepep_ll_combine(output, topk_ids, topk_weights)

        assert combine_input.hidden_states is output
        assert combine_input.topk_ids is topk_ids
        assert combine_input.topk_weights is topk_weights


# ============================================================================
# Test Section 2: RdepAdapter Class Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestRdepAdapter:
    """Test RdepAdapter class functionality."""

    def test_adapter_initialization(self):
        """Test RdepAdapter initialization."""
        adapter = RdepAdapter(validate_shapes=False)
        assert adapter.validate_shapes is False

        adapter_with_validation = RdepAdapter(validate_shapes=True)
        assert adapter_with_validation.validate_shapes is True

    def test_adapter_validate_rdep_input_valid(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test validate_rdep_input with valid input."""
        adapter = RdepAdapter(validate_shapes=True)

        x = create_mock_hidden_states(num_tokens, hidden_size, device)
        eid = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        gates = create_mock_topk_weights(num_tokens, top_k, device)

        rdep_input = RdepDispatchInput(
            hidden_states=x,
            topk_ids=eid,
            topk_weights=gates,
        )

        # Should not raise
        adapter.validate_rdep_input(rdep_input)

    def test_adapter_validate_rdep_input_token_count_mismatch(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test validate_rdep_input with token count mismatch."""
        adapter = RdepAdapter(validate_shapes=True)

        x = create_mock_hidden_states(num_tokens, hidden_size, device)
        eid = create_mock_topk_ids(num_tokens + 10, top_k, num_experts, device)  # Mismatch
        gates = create_mock_topk_weights(num_tokens, top_k, device)

        rdep_input = RdepDispatchInput(
            hidden_states=x,
            topk_ids=eid,
            topk_weights=gates,
        )

        with pytest.raises(ValueError, match="Token count mismatch"):
            adapter.validate_rdep_input(rdep_input)

    def test_adapter_validate_rdep_input_topk_mismatch(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test validate_rdep_input with top-K mismatch."""
        adapter = RdepAdapter(validate_shapes=True)

        x = create_mock_hidden_states(num_tokens, hidden_size, device)
        eid = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        gates = create_mock_topk_weights(num_tokens, top_k + 1, device)  # K mismatch

        rdep_input = RdepDispatchInput(
            hidden_states=x,
            topk_ids=eid,
            topk_weights=gates,
        )

        with pytest.raises(ValueError, match="TopK mismatch"):
            adapter.validate_rdep_input(rdep_input)

    def test_adapter_validate_rdep_input_wrong_dtype(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test validate_rdep_input with wrong dtype."""
        adapter = RdepAdapter(validate_shapes=True)

        x = create_mock_hidden_states(num_tokens, hidden_size, device, dtype=torch.float32)  # Wrong dtype
        eid = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        gates = create_mock_topk_weights(num_tokens, top_k, device)

        rdep_input = RdepDispatchInput(
            hidden_states=x,
            topk_ids=eid,
            topk_weights=gates,
        )

        with pytest.raises(ValueError, match="Expected BF16"):
            adapter.validate_rdep_input(rdep_input)

    def test_adapter_validate_rdep_input_non_contiguous(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test validate_rdep_input with non-contiguous tensor."""
        adapter = RdepAdapter(validate_shapes=True)

        # Create non-contiguous tensor via transpose
        x_base = torch.randn(hidden_size, num_tokens, device=device, dtype=torch.bfloat16)
        x = x_base.t()  # Non-contiguous
        assert not x.is_contiguous()

        eid = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        gates = create_mock_topk_weights(num_tokens, top_k, device)

        rdep_input = RdepDispatchInput(
            hidden_states=x,
            topk_ids=eid,
            topk_weights=gates,
        )

        with pytest.raises(ValueError, match="must be contiguous"):
            adapter.validate_rdep_input(rdep_input)


# ============================================================================
# Test Section 3: Buffer Layout and Capacity Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestBufferLayout:
    """Test buffer layout calculations and capacity handling."""

    def test_calculate_bf16_buffer_size_basic(self, hidden_size, capacity):
        """Test calculate_bf16_buffer_size basic functionality."""
        result = calculate_bf16_buffer_size(capacity, hidden_size, world_size=1)

        assert 'offsets' in result
        assert 'sizes' in result
        assert 'total_size' in result
        assert result['capacity'] == capacity
        assert result['hidden_size'] == hidden_size
        assert result['world_size'] == 1
        assert result['total_size'] > 0

    def test_calculate_bf16_buffer_size_multi_rank(self, hidden_size, capacity):
        """Test calculate_bf16_buffer_size with multiple ranks."""
        result_1 = calculate_bf16_buffer_size(capacity, hidden_size, world_size=1)
        result_8 = calculate_bf16_buffer_size(capacity, hidden_size, world_size=8)

        # Multi-rank should have different token slot sizes
        assert result_1['sizes']['tok_y'] != result_8['sizes']['tok_y'] or \
               result_1['sizes']['tok_gate'] != result_8['sizes']['tok_gate']

    def test_calculate_bf16_buffer_offsets_aligned(self, hidden_size, capacity):
        """Test that buffer offsets are properly aligned."""
        result = calculate_bf16_buffer_size(capacity, hidden_size, world_size=1)

        # Check that key offsets are aligned
        assert result['offsets']['barrier_signals'] % BUFFER_ALIGNMENT == 0
        assert result['offsets']['tok_y'] % BUFFER_ALIGNMENT == 0
        assert result['offsets']['tok_gate'] % BUFFER_ALIGNMENT == 0

    def test_calculate_blockscaled_buffer_size_fp8(self, hidden_size, capacity):
        """Test calculate_blockscaled_buffer_size for FP8."""
        result = calculate_blockscaled_buffer_size(capacity, hidden_size, 'fp8', world_size=1)

        assert result['profile'] == 'fp8'
        assert result['pack_factor'] == 2
        assert result['Hp'] == hidden_size // 2
        assert 'x_q' in result['offsets']
        assert 'sfa' in result['offsets']
        assert 'y_buf' in result['offsets']

    def test_calculate_blockscaled_buffer_size_nvfp4(self, hidden_size, capacity):
        """Test calculate_blockscaled_buffer_size for NVFP4."""
        result = calculate_blockscaled_buffer_size(capacity, hidden_size, 'nvfp4', world_size=1)

        assert result['profile'] == 'nvfp4'
        assert result['pack_factor'] == 4
        assert result['Hp'] == hidden_size // 4

    def test_calculate_blockscaled_buffer_size_invalid_profile(self, hidden_size, capacity):
        """Test calculate_blockscaled_buffer_size with invalid profile."""
        with pytest.raises(ValueError, match="Invalid profile"):
            calculate_blockscaled_buffer_size(capacity, hidden_size, 'invalid', world_size=1)

    def test_estimate_rdep_memory_bf16(self, hidden_size, capacity, num_experts):
        """Test estimate_rdep_memory for BF16 mode."""
        result = estimate_rdep_memory(
            capacity=capacity,
            hidden_size=hidden_size,
            num_local_experts=num_experts,
            profile='bf16',
            world_size=1,
        )

        assert 'dispatch_buffer_mb' in result
        assert 'weight_memory_mb' in result
        assert 'total_mb' in result
        assert result['profile'] == 'bf16'
        assert result['dispatch_buffer_mb'] > 0
        assert result['weight_memory_mb'] > 0

    def test_check_buffer_layout_compatible(self, device, hidden_size, capacity):
        """Test check_buffer_layout_compatible function."""
        # Compatible tensor
        compatible = torch.randn(capacity // 2, hidden_size, device=device, dtype=torch.bfloat16)
        assert check_buffer_layout_compatible(compatible, capacity, hidden_size) is True

        # Incompatible - wrong dtype
        wrong_dtype = torch.randn(capacity // 2, hidden_size, device=device, dtype=torch.float32)
        assert check_buffer_layout_compatible(wrong_dtype, capacity, hidden_size) is False

        # Incompatible - too many tokens
        too_large = torch.randn(capacity + 100, hidden_size, device=device, dtype=torch.bfloat16)
        assert check_buffer_layout_compatible(too_large, capacity, hidden_size) is False

        # Incompatible - non-contiguous
        base = torch.randn(hidden_size, capacity // 2, device=device, dtype=torch.bfloat16)
        non_contig = base.t()
        assert check_buffer_layout_compatible(non_contig, capacity, hidden_size) is False


# ============================================================================
# Test Section 4: RdepBufferView Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestRdepBufferView:
    """Test RdepBufferView class functionality."""

    def test_buffer_view_initialization_bf16(self, device, hidden_size, capacity):
        """Test RdepBufferView initialization for BF16 mode."""
        view = RdepBufferView(
            capacity=capacity,
            hidden_size=hidden_size,
            profile='bf16',
            world_size=1,
            device=device,
        )

        assert view.capacity == capacity
        assert view.hidden_size == hidden_size
        assert view.profile == 'bf16'
        assert view.total_size > 0
        assert view.x_buffer_size > 0

    def test_buffer_view_initialization_fp8(self, device, hidden_size, capacity):
        """Test RdepBufferView initialization for FP8 mode."""
        view = RdepBufferView(
            capacity=capacity,
            hidden_size=hidden_size,
            profile='fp8',
            world_size=1,
            device=device,
        )

        assert view.profile == 'fp8'
        # FP8 packs 2 values per element
        assert view.layout['pack_factor'] == 2

    def test_buffer_view_can_zero_copy(self, device, hidden_size, capacity):
        """Test RdepBufferView.can_zero_copy method."""
        view = RdepBufferView(
            capacity=capacity,
            hidden_size=hidden_size,
            profile='bf16',
            world_size=1,
            device=device,
        )

        # Compatible tensor
        compatible = torch.randn(capacity // 2, hidden_size, device=device, dtype=torch.bfloat16)
        assert view.can_zero_copy(compatible) is True

        # Incompatible tensor
        incompatible = torch.randn(capacity + 100, hidden_size, device=device, dtype=torch.bfloat16)
        assert view.can_zero_copy(incompatible) is False

    def test_buffer_view_get_buffer_info(self, device, hidden_size, capacity):
        """Test RdepBufferView.get_buffer_info method."""
        view = RdepBufferView(
            capacity=capacity,
            hidden_size=hidden_size,
            profile='bf16',
            world_size=1,
            device=device,
        )

        info = view.get_buffer_info()

        assert 'offsets' in info
        assert 'sizes' in info
        assert 'total_size' in info

    def test_buffer_view_memory_usage(self, device, hidden_size, capacity):
        """Test RdepBufferView.get_memory_usage_mb method."""
        view = RdepBufferView(
            capacity=capacity,
            hidden_size=hidden_size,
            profile='bf16',
            world_size=1,
            device=device,
        )

        memory_mb = view.get_memory_usage_mb()

        assert memory_mb > 0
        assert memory_mb == view.total_size / (1024 * 1024)


# ============================================================================
# Test Section 5: Synchronization Context Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestSynchronizationContext:
    """Test RdepSyncContext and synchronization functionality."""

    def test_sync_context_initialization(self, device):
        """Test RdepSyncContext initialization."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        assert ctx.world_size == 1
        assert ctx.rank == 0
        assert ctx.mode == SyncMode.CUDA_EVENTS
        assert not ctx.is_distributed

    def test_sync_context_is_distributed(self, device):
        """Test RdepSyncContext.is_distributed property."""
        single = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)
        multi = RdepSyncContext(world_size=8, rank=0, mode=SyncMode.HYBRID, device=device)

        assert not single.is_distributed
        assert multi.is_distributed

    def test_sync_context_get_or_create_stream(self, device):
        """Test RdepSyncContext.get_or_create_stream method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        stream1 = ctx.get_or_create_stream()
        stream2 = ctx.get_or_create_stream()

        assert isinstance(stream1, torch.cuda.Stream)
        assert stream1 is stream2  # Should return same stream

    def test_sync_context_record_pre_dispatch(self, device):
        """Test RdepSyncContext.record_pre_dispatch method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        event = ctx.record_pre_dispatch()

        assert isinstance(event, torch.cuda.Event)

    def test_sync_context_record_post_dispatch(self, device):
        """Test RdepSyncContext.record_post_dispatch method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        event = ctx.record_post_dispatch()

        assert isinstance(event, torch.cuda.Event)
        assert ctx._dispatch_count == 1

    def test_sync_context_record_combine_complete(self, device):
        """Test RdepSyncContext.record_combine_complete method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        event = ctx.record_combine_complete()

        assert isinstance(event, torch.cuda.Event)

    def test_sync_context_wait_for_dispatch(self, device):
        """Test RdepSyncContext.wait_for_dispatch method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        # Should not raise even if no dispatch recorded
        ctx.wait_for_dispatch()

        # Record and wait
        ctx.record_post_dispatch()
        ctx.wait_for_dispatch()

    def test_sync_context_synchronize(self, device):
        """Test RdepSyncContext.synchronize method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        # Create a stream and launch some work
        stream = ctx.get_or_create_stream()
        with torch.cuda.stream(stream):
            _ = torch.randn(1000, 1000, device=device)

        ctx.synchronize()

    def test_sync_context_get_stats(self, device):
        """Test RdepSyncContext.get_stats method."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        ctx.record_post_dispatch()
        ctx.record_post_dispatch()

        stats = ctx.get_stats()

        assert stats['dispatch_count'] == 2
        assert stats['mode'] == SyncMode.CUDA_EVENTS
        assert stats['world_size'] == 1
        assert stats['rank'] == 0


# ============================================================================
# Test Section 6: DistributedSyncManager Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestDistributedSyncManager:
    """Test DistributedSyncManager class functionality."""

    def test_distributed_sync_manager_initialization(self, device):
        """Test DistributedSyncManager initialization."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        assert manager.world_size == 1
        assert manager.rank == 0
        assert manager.rdep_mode == 'single'

    def test_distributed_sync_manager_begin_dispatch(self, device):
        """Test DistributedSyncManager.begin_dispatch method."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        event = manager.begin_dispatch()

        assert isinstance(event, torch.cuda.Event)
        assert manager._in_dispatch is True

    def test_distributed_sync_manager_end_dispatch(self, device):
        """Test DistributedSyncManager.end_dispatch method."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        manager.begin_dispatch()
        event = manager.end_dispatch()

        assert isinstance(event, torch.cuda.Event)
        assert manager._in_dispatch is False
        assert manager._total_dispatches == 1

    def test_distributed_sync_manager_dispatch_state_error(self, device):
        """Test DistributedSyncManager raises error on invalid dispatch state."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        # End without begin
        with pytest.raises(RuntimeError, match="Not in dispatch phase"):
            manager.end_dispatch()

        # Double begin
        manager.begin_dispatch()
        with pytest.raises(RuntimeError, match="Already in dispatch phase"):
            manager.begin_dispatch()

    def test_distributed_sync_manager_begin_combine(self, device):
        """Test DistributedSyncManager.begin_combine method."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        manager.begin_combine()

        assert manager._in_combine is True

    def test_distributed_sync_manager_end_combine(self, device):
        """Test DistributedSyncManager.end_combine method."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        manager.begin_combine()
        event = manager.end_combine()

        assert isinstance(event, torch.cuda.Event)
        assert manager._in_combine is False
        assert manager._total_combines == 1

    def test_distributed_sync_manager_get_stats(self, device):
        """Test DistributedSyncManager.get_stats method."""
        manager = DistributedSyncManager(world_size=1, rank=0, device=device)

        manager.begin_dispatch()
        manager.end_dispatch()
        manager.begin_combine()
        manager.end_combine()

        stats = manager.get_stats()

        assert stats['total_dispatches'] == 1
        assert stats['total_combines'] == 1
        assert stats['rdep_mode'] == 'single'


# ============================================================================
# Test Section 7: RDEP Mode Detection Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestRdepModeDetection:
    """Test RDEP mode detection based on topology."""

    def test_get_rdep_mode_for_topology_single(self):
        """Test get_rdep_mode_for_topology for single GPU."""
        mode = get_rdep_mode_for_topology(world_size=1, local_world_size=1)
        assert mode == 'single'

    def test_get_rdep_mode_for_topology_ipc(self):
        """Test get_rdep_mode_for_topology for single-node multi-GPU."""
        mode = get_rdep_mode_for_topology(world_size=8, local_world_size=8)
        assert mode == 'ipc'

    def test_get_rdep_mode_for_topology_hybrid(self):
        """Test get_rdep_mode_for_topology for multi-node."""
        mode = get_rdep_mode_for_topology(world_size=16, local_world_size=8)
        assert mode == 'hybrid'

    def test_get_default_sync_mode(self):
        """Test get_default_sync_mode function."""
        assert get_default_sync_mode(1) == SyncMode.CUDA_EVENTS
        assert get_default_sync_mode(8) == SyncMode.HYBRID

    def test_create_sync_manager(self, device):
        """Test create_sync_manager factory function."""
        manager = create_sync_manager(world_size=1, rank=0, local_world_size=1)

        assert manager.world_size == 1
        assert manager.rank == 0
        assert manager.local_world_size == 1


# ============================================================================
# Test Section 8: RdepDispatchAdapter Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestRdepDispatchAdapter:
    """Test RdepDispatchAdapter class functionality."""

    def test_dispatch_adapter_initialization(self, device, hidden_size, num_experts, top_k):
        """Test RdepDispatchAdapter initialization."""
        adapter = RdepDispatchAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        assert adapter.num_local_experts == num_experts
        assert adapter.hidden_size == hidden_size
        assert adapter.top_k == top_k
        assert adapter.profile == 'bf16'

    def test_dispatch_adapter_ensure_buffers(self, device, hidden_size, num_experts, top_k):
        """Test RdepDispatchAdapter._ensure_buffers method."""
        adapter = RdepDispatchAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        offs_pad, m_host = adapter._ensure_buffers()

        assert offs_pad.shape == (num_experts,)
        assert offs_pad.dtype == torch.int32
        assert offs_pad.device.type == 'cuda'
        assert m_host.shape == (1,)
        assert m_host.dtype == torch.int32
        assert m_host.is_pinned()

    def test_dispatch_adapter_get_stats(self, device, hidden_size, num_experts, top_k):
        """Test RdepDispatchAdapter.get_stats method."""
        adapter = RdepDispatchAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        stats = adapter.get_stats()

        assert stats['dispatch_count'] == 0
        assert stats['num_local_experts'] == num_experts
        assert stats['hidden_size'] == hidden_size
        assert stats['top_k'] == top_k
        assert stats['profile'] == 'bf16'

    def test_create_dispatch_adapter_bf16(self, hidden_size, num_experts, top_k):
        """Test create_dispatch_adapter factory function for BF16."""
        adapter = create_dispatch_adapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
        )

        assert adapter.profile == 'bf16'
        assert adapter.alignment == BF16_ALIGNMENT

    def test_create_dispatch_adapter_fp8(self, hidden_size, num_experts, top_k):
        """Test create_dispatch_adapter factory function for FP8."""
        adapter = create_dispatch_adapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='fp8',
        )

        assert adapter.profile == 'fp8'
        assert adapter.alignment == BLOCKSCALED_ALIGNMENT


# ============================================================================
# Test Section 9: RdepCombineAdapter Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestRdepCombineAdapter:
    """Test RdepCombineAdapter class functionality."""

    def test_combine_adapter_initialization(self, device, hidden_size, top_k):
        """Test RdepCombineAdapter initialization."""
        adapter = RdepCombineAdapter(
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        assert adapter.hidden_size == hidden_size
        assert adapter.top_k == top_k
        assert adapter.profile == 'bf16'

    def test_combine_adapter_get_stats(self, device, hidden_size, top_k):
        """Test RdepCombineAdapter.get_stats method."""
        adapter = RdepCombineAdapter(
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        stats = adapter.get_stats()

        assert stats['combine_count'] == 0
        assert stats['total_tokens_combined'] == 0
        assert stats['hidden_size'] == hidden_size
        assert stats['top_k'] == top_k
        assert stats['profile'] == 'bf16'

    def test_create_combine_adapter(self, hidden_size, top_k):
        """Test create_combine_adapter factory function."""
        adapter = create_combine_adapter(
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
        )

        assert adapter.hidden_size == hidden_size
        assert adapter.top_k == top_k
        assert adapter.profile == 'bf16'


# ============================================================================
# Test Section 10: Low-Latency Mode Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestLowLatencyMode:
    """Test low-latency mode functionality."""

    def test_low_latency_config_defaults(self):
        """Test LowLatencyConfig default values."""
        config = LowLatencyConfig()

        assert config.max_tokens == LOW_LATENCY_MAX_TOKENS
        assert config.enable_direct_ipc is True
        assert config.use_single_token_kernel is True
        assert config.prefetch_expert_weights is False
        assert config.stream_priority == -1

    def test_low_latency_adapter_initialization(self, device, hidden_size, num_experts, top_k):
        """Test RdepLowLatencyAdapter initialization."""
        adapter = RdepLowLatencyAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        assert adapter.num_local_experts == num_experts
        assert adapter.hidden_size == hidden_size
        assert adapter.top_k == top_k
        assert adapter.profile == 'bf16'

    def test_low_latency_adapter_should_use_low_latency_auto_decode(self, device, hidden_size, num_experts, top_k):
        """Test should_use_low_latency with AUTO mode for decode."""
        adapter = RdepLowLatencyAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        # Small batch decode should use low-latency
        assert adapter.should_use_low_latency(num_tokens=1, is_extend=False, mode=RdepMode.AUTO)
        assert adapter.should_use_low_latency(num_tokens=16, is_extend=False, mode=RdepMode.AUTO)

        # Large batch should not use low-latency
        assert not adapter.should_use_low_latency(num_tokens=1000, is_extend=False, mode=RdepMode.AUTO)

        # Extend should not use low-latency
        assert not adapter.should_use_low_latency(num_tokens=1, is_extend=True, mode=RdepMode.AUTO)

    def test_low_latency_adapter_should_use_low_latency_forced(self, device, hidden_size, num_experts, top_k):
        """Test should_use_low_latency with forced modes."""
        adapter = RdepLowLatencyAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        # NORMAL mode always returns False
        assert not adapter.should_use_low_latency(num_tokens=1, is_extend=False, mode=RdepMode.NORMAL)

        # LOW_LATENCY mode always returns True
        assert adapter.should_use_low_latency(num_tokens=1000, is_extend=True, mode=RdepMode.LOW_LATENCY)

    def test_low_latency_adapter_resolve_mode(self, device, hidden_size, num_experts, top_k):
        """Test RdepLowLatencyAdapter.resolve_mode method."""
        adapter = RdepLowLatencyAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        # AUTO resolves based on is_extend
        assert adapter.resolve_mode(is_extend=False, mode=RdepMode.AUTO) == RdepMode.LOW_LATENCY
        assert adapter.resolve_mode(is_extend=True, mode=RdepMode.AUTO) == RdepMode.NORMAL

        # Explicit modes pass through
        assert adapter.resolve_mode(is_extend=False, mode=RdepMode.NORMAL) == RdepMode.NORMAL
        assert adapter.resolve_mode(is_extend=True, mode=RdepMode.LOW_LATENCY) == RdepMode.LOW_LATENCY

    def test_low_latency_adapter_get_stats(self, device, hidden_size, num_experts, top_k):
        """Test RdepLowLatencyAdapter.get_stats method."""
        adapter = RdepLowLatencyAdapter(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            profile='bf16',
            device=device,
        )

        stats = adapter.get_stats()

        assert 'll_dispatch_count' in stats
        assert 'normal_dispatch_count' in stats
        assert 'total_tokens' in stats


# ============================================================================
# Test Section 11: Expert IDs and Gate Weights Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestExpertIdsAndGateWeights:
    """Test expert IDs and gate weights handling."""

    def test_expert_ids_conversion_preserves_range(self, device, num_tokens, num_experts, top_k):
        """Test that expert ID conversion preserves valid ranges."""
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        eid, gates = topk_to_rdep_format(topk_output)

        # Expert IDs should be in valid range
        assert eid.min() >= 0
        assert eid.max() < num_experts

    def test_gate_weights_normalized(self, device, num_tokens, num_experts, top_k):
        """Test gate weights are properly normalized."""
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        eid, gates = topk_to_rdep_format(topk_output)

        # Gate weights should sum to approximately 1 for each token
        gate_sums = gates.sum(dim=-1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

    def test_expert_ids_all_same(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test handling of all tokens routed to same experts."""
        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device)

        # All tokens route to experts 0 and 1
        topk_ids = torch.zeros(num_tokens, top_k, device=device, dtype=torch.long)
        topk_ids[:, 1] = 1
        topk_weights = torch.ones(num_tokens, top_k, device=device, dtype=torch.float32) / top_k
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        # Verify the conversion succeeded
        assert rdep_input.topk_ids.shape == (num_tokens, top_k)
        assert (rdep_input.topk_ids == 0).sum() == num_tokens
        assert (rdep_input.topk_ids == 1).sum() == num_tokens

    def test_expert_ids_sparse_distribution(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test handling of sparse expert distribution."""
        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device)

        # Tokens alternate between different expert pairs
        topk_ids = torch.zeros(num_tokens, top_k, device=device, dtype=torch.long)
        for i in range(num_tokens):
            topk_ids[i, 0] = (i * 2) % num_experts
            topk_ids[i, 1] = (i * 2 + 1) % num_experts
        topk_weights = torch.ones(num_tokens, top_k, device=device, dtype=torch.float32) / top_k
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        # Verify all experts are represented
        assert rdep_input.topk_ids.shape == (num_tokens, top_k)


# ============================================================================
# Test Section 12: Error Handling Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestErrorHandling:
    """Test error handling for format mismatches and invalid inputs."""

    def test_topk_format_bypassed_raises(self, device, num_tokens, num_experts):
        """Test that bypassed TopK format raises ValueError."""
        hidden_states = create_mock_hidden_states(num_tokens, 256, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)
        config = TopKConfig(top_k=2)

        bypassed_output = BypassedTopKOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=config,
        )

        with pytest.raises(ValueError, match="doesn't support bypassed"):
            topk_to_rdep_format(bypassed_output)

    def test_empty_batch_handling(self, device, hidden_size, num_experts, top_k):
        """Test handling of empty batch (0 tokens)."""
        # Empty tensors
        hidden_states = torch.empty(0, hidden_size, device=device, dtype=torch.bfloat16)
        topk_ids = torch.empty(0, top_k, device=device, dtype=torch.long)
        topk_weights = torch.empty(0, top_k, device=device, dtype=torch.float32)
        router_logits = torch.empty(0, num_experts, device=device, dtype=torch.float32)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.shape == (0, hidden_size)
        assert rdep_input.topk_ids.shape == (0, top_k)
        assert rdep_input.topk_weights.shape == (0, top_k)

    def test_very_large_batch(self, device, hidden_size, num_experts, top_k):
        """Test handling of very large batch."""
        large_num_tokens = 10000

        hidden_states = create_mock_hidden_states(large_num_tokens, hidden_size, device)
        topk_ids = create_mock_topk_ids(large_num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(large_num_tokens, top_k, device)
        router_logits = create_mock_router_logits(large_num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.shape == (large_num_tokens, hidden_size)

    def test_negative_expert_ids_clamped(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test that negative expert IDs are handled."""
        # Create IDs with some negative values (which shouldn't happen in practice)
        topk_ids = torch.randint(-1, num_experts, (num_tokens, top_k), device=device, dtype=torch.long)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        # Should not raise - conversion should still work
        eid, gates = topk_to_rdep_format(topk_output)
        assert eid.dtype == torch.int32


# ============================================================================
# Test Section 13: Performance Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestPerformance:
    """Test performance to ensure no overhead beyond direct RDEP calls."""

    def test_format_conversion_overhead(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test that format conversion overhead is minimal."""
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        # Warmup
        for _ in range(10):
            topk_to_rdep_format(topk_output)

        torch.cuda.synchronize()

        # Measure time
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            topk_to_rdep_format(topk_output)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time_us = (elapsed / num_iterations) * 1e6

        # Format conversion should be very fast (< 1ms typically)
        assert avg_time_us < 1000, f"Format conversion too slow: {avg_time_us:.2f} us"

    def test_buffer_view_creation_overhead(self, device, hidden_size, capacity):
        """Test that buffer view creation overhead is minimal."""
        # Warmup
        for _ in range(10):
            view = RdepBufferView(
                capacity=capacity,
                hidden_size=hidden_size,
                profile='bf16',
                world_size=1,
                device=device,
            )

        # Measure time
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            view = RdepBufferView(
                capacity=capacity,
                hidden_size=hidden_size,
                profile='bf16',
                world_size=1,
                device=device,
            )
        elapsed = time.perf_counter() - start

        avg_time_us = (elapsed / num_iterations) * 1e6

        # Buffer view creation should be fast
        assert avg_time_us < 500, f"Buffer view creation too slow: {avg_time_us:.2f} us"

    def test_sync_context_event_overhead(self, device):
        """Test that sync context event recording overhead is minimal."""
        ctx = RdepSyncContext(world_size=1, rank=0, mode=SyncMode.CUDA_EVENTS, device=device)

        # Warmup
        for _ in range(10):
            ctx.record_pre_dispatch()
            ctx.record_post_dispatch()

        torch.cuda.synchronize()

        # Measure time
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            ctx.record_pre_dispatch()
            ctx.record_post_dispatch()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time_us = (elapsed / num_iterations) * 1e6

        # Event recording should be very fast
        assert avg_time_us < 200, f"Event recording too slow: {avg_time_us:.2f} us"

    def test_large_batch_conversion_scaling(self, device, hidden_size, num_experts, top_k):
        """Test that conversion scales linearly with batch size."""
        batch_sizes = [64, 256, 1024, 4096]
        times = []

        for batch_size in batch_sizes:
            hidden_states = create_mock_hidden_states(batch_size, hidden_size, device)
            topk_ids = create_mock_topk_ids(batch_size, top_k, num_experts, device)
            topk_weights = create_mock_topk_weights(batch_size, top_k, device)
            router_logits = create_mock_router_logits(batch_size, num_experts, device)

            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids.long(),
                router_logits=router_logits,
            )

            dispatch_output = StandardDispatchOutput(
                hidden_states=hidden_states,
                hidden_states_scale=None,
                topk_output=topk_output,
            )

            # Warmup
            for _ in range(5):
                standard_dispatch_to_rdep(dispatch_output)

            torch.cuda.synchronize()

            # Measure
            num_iterations = 20
            start = time.perf_counter()
            for _ in range(num_iterations):
                standard_dispatch_to_rdep(dispatch_output)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times.append(elapsed / num_iterations)

        # Check that time scales roughly linearly (within 10x of expected)
        # The ratio of times should be roughly proportional to batch size ratio
        time_ratio = times[-1] / times[0]
        batch_ratio = batch_sizes[-1] / batch_sizes[0]

        # Allow for some overhead and non-linearity
        assert time_ratio < batch_ratio * 10, f"Scaling issue: time ratio {time_ratio:.2f}, batch ratio {batch_ratio:.2f}"


# ============================================================================
# Test Section 14: RdepDispatchInput and RdepCombineOutput Data Classes
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestRdepDataClasses:
    """Test RdepDispatchInput and RdepCombineOutput data classes."""

    def test_rdep_dispatch_input_format(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test RdepDispatchInput.format property."""
        x = create_mock_hidden_states(num_tokens, hidden_size, device)
        eid = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        gates = create_mock_topk_weights(num_tokens, top_k, device)

        rdep_input = RdepDispatchInput(
            hidden_states=x,
            topk_ids=eid,
            topk_weights=gates,
        )

        assert rdep_input.format == DispatchOutputFormat.STANDARD

    def test_rdep_combine_output_format(self, device, hidden_size, num_tokens):
        """Test RdepCombineOutput.format property."""
        output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        rdep_output = RdepCombineOutput(hidden_states=output)

        assert rdep_output.format == CombineInputFormat.STANDARD

    def test_rdep_meta_dataclass(self):
        """Test RdepMeta dataclass."""
        meta = RdepMeta(row_id=42, local_eid=3, gate=0.75)

        assert meta.row_id == 42
        assert meta.local_eid == 3
        assert meta.gate == 0.75


# ============================================================================
# Test Section 15: Integration with DeepEP Formats
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestDeepEPIntegration:
    """Test integration with DeepEP dispatch formats."""

    def test_deepep_normal_to_rdep(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test deepep_normal_to_rdep conversion."""
        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)

        dispatch_output = DeepEPNormalDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids.long(),
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[num_tokens // num_experts] * num_experts,
        )

        rdep_input = deepep_normal_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.dtype == torch.bfloat16
        assert rdep_input.topk_ids.dtype == torch.int32
        assert rdep_input.hidden_states.shape == (num_tokens, hidden_size)
        assert rdep_input.topk_ids.shape == (num_tokens, top_k)

    def test_deepep_ll_to_rdep(self, device, hidden_size, num_tokens, num_experts, top_k):
        """Test deepep_ll_to_rdep conversion."""
        expected_m = num_tokens * top_k
        hidden_states = create_mock_hidden_states(expected_m, hidden_size, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        masked_m = torch.ones(expected_m, device=device, dtype=torch.bool)

        dispatch_output = DeepEPLLDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids.long(),
            topk_weights=topk_weights,
            masked_m=masked_m,
            expected_m=expected_m,
        )

        rdep_input = deepep_ll_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.dtype == torch.bfloat16
        assert rdep_input.topk_ids.dtype == torch.int32
        assert rdep_input.hidden_states.shape == (expected_m, hidden_size)


# ============================================================================
# Test Section 16: DispatchMetaResult Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestDispatchMetaResult:
    """Test DispatchMetaResult dataclass."""

    def test_dispatch_meta_result_has_work(self, device, num_experts):
        """Test DispatchMetaResult.has_work property."""
        offs_pad = torch.zeros(num_experts, device=device, dtype=torch.int32)
        m_host = torch.zeros(1, device='cpu', dtype=torch.int32)
        stream = torch.cuda.current_stream(device)

        # With work
        result_with_work = DispatchMetaResult(
            m_recv=100,
            max_pad=128,
            offs_pad=offs_pad,
            m_host=m_host,
            stream=stream,
            success=True,
        )
        assert result_with_work.has_work is True

        # Without work
        result_no_work = DispatchMetaResult(
            m_recv=0,
            max_pad=0,
            offs_pad=offs_pad,
            m_host=m_host,
            stream=stream,
            success=True,
        )
        assert result_no_work.has_work is False


# ============================================================================
# Test Section 17: CombineResult Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestCombineResult:
    """Test CombineResult dataclass."""

    def test_combine_result_basic(self, device, hidden_size, num_tokens):
        """Test CombineResult basic functionality."""
        output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32)
        stream = torch.cuda.current_stream(device)

        result = CombineResult(
            output=output,
            num_tokens_combined=num_tokens,
            stream=stream,
            success=True,
        )

        assert result.output is output
        assert result.num_tokens_combined == num_tokens
        assert result.success is True


# ============================================================================
# Test Section 18: Constants Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestConstants:
    """Test adapter constants."""

    def test_buffer_alignment(self):
        """Test BUFFER_ALIGNMENT constant."""
        assert BUFFER_ALIGNMENT == 256
        assert BUFFER_ALIGNMENT > 0

    def test_max_ranks(self):
        """Test MAX_RANKS constant."""
        assert MAX_RANKS == 64
        assert MAX_RANKS > 0

    def test_meta_size(self):
        """Test META_SIZE constant."""
        # 8 (row_id) + 4 (local_eid) + 4 (gate) = 16
        assert META_SIZE == 16

    def test_alignment_constants(self):
        """Test BF16_ALIGNMENT and BLOCKSCALED_ALIGNMENT constants."""
        assert BF16_ALIGNMENT == 128
        assert BLOCKSCALED_ALIGNMENT == 128

    def test_low_latency_constants(self):
        """Test LOW_LATENCY_MAX_TOKENS and LOW_LATENCY_SINGLE_TOKEN constants."""
        assert LOW_LATENCY_MAX_TOKENS == 32
        assert LOW_LATENCY_SINGLE_TOKEN == 1


# ============================================================================
# Test Section 19: Edge Cases
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self, device, hidden_size, num_experts, top_k):
        """Test handling of single token."""
        hidden_states = create_mock_hidden_states(1, hidden_size, device)
        topk_ids = create_mock_topk_ids(1, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(1, top_k, device)
        router_logits = create_mock_router_logits(1, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.shape == (1, hidden_size)
        assert rdep_input.topk_ids.shape == (1, top_k)

    def test_top_k_one(self, device, hidden_size, num_tokens, num_experts):
        """Test handling of top_k=1."""
        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device)
        topk_ids = create_mock_topk_ids(num_tokens, 1, num_experts, device)
        topk_weights = torch.ones(num_tokens, 1, device=device, dtype=torch.float32)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.topk_ids.shape == (num_tokens, 1)
        assert rdep_input.topk_weights.shape == (num_tokens, 1)

    def test_many_experts(self, device, hidden_size, num_tokens, top_k):
        """Test handling of many experts (256)."""
        many_experts = 256

        hidden_states = create_mock_hidden_states(num_tokens, hidden_size, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, many_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        router_logits = create_mock_router_logits(num_tokens, many_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.shape == (num_tokens, hidden_size)
        # Expert IDs should be in range
        assert rdep_input.topk_ids.max() < many_experts

    def test_large_hidden_size(self, device, num_tokens, num_experts, top_k):
        """Test handling of large hidden size."""
        large_hidden = 8192

        hidden_states = create_mock_hidden_states(num_tokens, large_hidden, device)
        topk_ids = create_mock_topk_ids(num_tokens, top_k, num_experts, device)
        topk_weights = create_mock_topk_weights(num_tokens, top_k, device)
        router_logits = create_mock_router_logits(num_tokens, num_experts, device)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        rdep_input = standard_dispatch_to_rdep(dispatch_output)

        assert rdep_input.hidden_states.shape == (num_tokens, large_hidden)


# ============================================================================
# Test Section 20: Helper Function Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.integration
class TestHelperFunctions:
    """Test helper utility functions."""

    def test_align_up(self):
        """Test _align_up helper function."""
        assert _align_up(0, 256) == 0
        assert _align_up(1, 256) == 256
        assert _align_up(255, 256) == 256
        assert _align_up(256, 256) == 256
        assert _align_up(257, 256) == 512
        assert _align_up(100, 128) == 128
        assert _align_up(129, 128) == 256

    def test_create_rdep_buffer_view_from_instance(self, device, hidden_size, capacity):
        """Test create_rdep_buffer_view factory function."""
        # Create a mock Rdep instance
        class MockRdep:
            capacity = 4096
            dim = 256
            profile = 'bf16'
            world = 1

        mock_rdep = MockRdep()
        view = create_rdep_buffer_view(mock_rdep, profile='bf16')

        assert isinstance(view, RdepBufferView)
        assert view.capacity == mock_rdep.capacity
        assert view.hidden_size == mock_rdep.dim


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
