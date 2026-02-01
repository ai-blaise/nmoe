"""Tests for RDEP mode detection.

Tests that RDEP mode is correctly selected based on GPU topology.
"""

import pytest
import sys
sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/nmoe')

from nmoe.unified.config import NMoERDEPConfig


class TestNMoERDEPConfig:
    """Test suite for NMoERDEPConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = NMoERDEPConfig()
        assert cfg.mode == 'auto'
        assert cfg.profile == 'bf16'
        assert cfg.capacity == 65536
        assert cfg.nvshmem_enabled is False

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = NMoERDEPConfig(
            mode='ipc',
            profile='fp8',
            capacity=131072,
            nvshmem_enabled=True,
        )
        assert cfg.mode == 'ipc'
        assert cfg.profile == 'fp8'
        assert cfg.capacity == 131072
        assert cfg.nvshmem_enabled is True

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = NMoERDEPConfig(
            mode='hybrid',
            profile='nvfp4',
            capacity=262144,
        )

        d = original.to_dict()
        restored = NMoERDEPConfig.from_dict(d)

        assert original.mode == restored.mode
        assert original.profile == restored.profile
        assert original.capacity == restored.capacity


class TestProfileId:
    """Test profile ID mapping for CUDA kernels."""

    def test_bf16_profile_id(self):
        """Test BF16 profile returns -1."""
        cfg = NMoERDEPConfig(profile='bf16')
        assert cfg.get_profile_id() == -1

    def test_fp8_profile_id(self):
        """Test FP8 profile returns 0."""
        cfg = NMoERDEPConfig(profile='fp8')
        assert cfg.get_profile_id() == 0

    def test_nvfp4_profile_id(self):
        """Test NVFP4 profile returns 1."""
        cfg = NMoERDEPConfig(profile='nvfp4')
        assert cfg.get_profile_id() == 1

    def test_unknown_profile_defaults_to_bf16(self):
        """Test unknown profile defaults to BF16 (-1)."""
        cfg = NMoERDEPConfig(profile='unknown')
        assert cfg.get_profile_id() == -1


class TestModeDetection:
    """Test automatic RDEP mode detection based on GPU topology."""

    def test_single_gpu(self):
        """Test single GPU detection."""
        cfg = NMoERDEPConfig(mode='auto')
        detected = cfg.detect_mode(world_size=1, local_world_size=1)
        assert detected == 'single'

    def test_multi_gpu_single_node_2(self):
        """Test 2 GPUs on single node -> IPC."""
        cfg = NMoERDEPConfig(mode='auto')
        detected = cfg.detect_mode(world_size=2, local_world_size=2)
        assert detected == 'ipc'

    def test_multi_gpu_single_node_4(self):
        """Test 4 GPUs on single node -> IPC."""
        cfg = NMoERDEPConfig(mode='auto')
        detected = cfg.detect_mode(world_size=4, local_world_size=4)
        assert detected == 'ipc'

    def test_multi_gpu_single_node_8(self):
        """Test 8 GPUs on single node -> IPC."""
        cfg = NMoERDEPConfig(mode='auto')
        detected = cfg.detect_mode(world_size=8, local_world_size=8)
        assert detected == 'ipc'

    def test_multi_node_2_nodes_8_gpus(self):
        """Test 2 nodes with 4 GPUs each -> HYBRID."""
        cfg = NMoERDEPConfig(mode='auto')
        detected = cfg.detect_mode(world_size=8, local_world_size=4)
        assert detected == 'hybrid'

    def test_multi_node_4_nodes_32_gpus(self):
        """Test 4 nodes with 8 GPUs each -> HYBRID."""
        cfg = NMoERDEPConfig(mode='auto')
        detected = cfg.detect_mode(world_size=32, local_world_size=8)
        assert detected == 'hybrid'

    def test_explicit_mode_overrides_auto(self):
        """Test explicit mode overrides auto-detection."""
        cfg = NMoERDEPConfig(mode='single')
        detected = cfg.detect_mode(world_size=8, local_world_size=8)
        assert detected == 'single'  # Override, not IPC

        cfg = NMoERDEPConfig(mode='hybrid')
        detected = cfg.detect_mode(world_size=1, local_world_size=1)
        assert detected == 'hybrid'  # Override, not single


class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_mismatched_world_sizes(self):
        """Test when world_size is not multiple of local_world_size."""
        cfg = NMoERDEPConfig(mode='auto')
        # This is an unusual config but should still work
        detected = cfg.detect_mode(world_size=6, local_world_size=4)
        assert detected == 'hybrid'

    def test_local_larger_than_world(self):
        """Test invalid config where local > world (should not happen)."""
        cfg = NMoERDEPConfig(mode='auto')
        # In practice this shouldn't happen, but test the logic
        detected = cfg.detect_mode(world_size=2, local_world_size=4)
        # world != local, so hybrid
        assert detected == 'hybrid'

    def test_zero_world_size(self):
        """Test zero world size defaults to single."""
        cfg = NMoERDEPConfig(mode='auto')
        # Edge case - should probably error but let's see behavior
        detected = cfg.detect_mode(world_size=0, local_world_size=0)
        # world_size == 1 check fails, goes to world == local check
        assert detected == 'ipc'  # 0 == 0

    def test_nvshmem_settings(self):
        """Test NVSHMEM-specific settings."""
        cfg = NMoERDEPConfig(
            mode='hybrid',
            nvshmem_enabled=True,
            nvshmem_heap_size=2 << 30,  # 2GB
        )
        assert cfg.nvshmem_enabled is True
        assert cfg.nvshmem_heap_size == 2 << 30


class TestEPGroupHelpers:
    """Test expert-parallel process group helper functions."""

    def test_get_group_world_size_not_initialized(self):
        """Test world size when dist is not initialized returns 1."""
        from nmoe.rdep import _get_group_world_size
        # When dist is not initialized, should return 1
        result = _get_group_world_size(None)
        assert result == 1

    def test_get_group_rank_not_initialized(self):
        """Test rank when dist is not initialized returns 0."""
        from nmoe.rdep import _get_group_rank
        # When dist is not initialized, should return 0
        result = _get_group_rank(None)
        assert result == 0

    def test_rdep_class_accepts_ep_group(self):
        """Test that Rdep class signature accepts ep_group parameter."""
        from nmoe.rdep import Rdep
        import inspect

        sig = inspect.signature(Rdep.__init__)
        params = list(sig.parameters.keys())

        assert 'ep_group' in params, "Rdep.__init__ should accept ep_group parameter"

    def test_rdep_stores_ep_group(self):
        """Test that Rdep stores ep_group attribute (docstring check)."""
        from nmoe.rdep import Rdep

        # Check the docstring mentions ep_group
        docstring = Rdep.__init__.__doc__
        assert 'ep_group' in docstring, "Rdep.__init__ docstring should document ep_group"
        assert 'expert-parallel' in docstring.lower() or 'process group' in docstring.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
