"""Comprehensive unit tests for NMoEModelInterface.

Tests cover:
- Abstract method contracts
- Interface implementation validation
- Mock implementation behavior
- Edge cases and error handling
"""

import pytest
from abc import ABC
from typing import Optional, Tuple, Dict, Any, List


class TestNMoEModelInterfaceContract:
    """Tests for the abstract interface contract."""

    def test_interface_is_abstract(self):
        """NMoEModelInterface is an abstract base class."""
        from nmoe.unified.interface import NMoEModelInterface

        assert issubclass(NMoEModelInterface, ABC)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate NMoEModelInterface directly."""
        from nmoe.unified.interface import NMoEModelInterface

        with pytest.raises(TypeError):
            NMoEModelInterface()

    def test_required_abstract_methods(self):
        """Interface defines all required abstract methods."""
        from nmoe.unified.interface import NMoEModelInterface

        abstract_methods = set()
        for name in dir(NMoEModelInterface):
            method = getattr(NMoEModelInterface, name, None)
            if callable(method) and getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        # Core methods that must be abstract (matching actual interface)
        expected = {
            "forward",
            "generate",
            "forward_with_log_probs",
            "refresh_expert_caches",
            "get_expert_load_stats",
            "get_router_aux_loss",  # Actual name in interface
            "update_router_biases",
            "gradient_checkpointing_enable",
            "gradient_checkpointing_disable",
            "get_input_embeddings",
            "get_output_embeddings",
            "param_sets",
            "named_parameters_by_type",
            "state_dict_for_save",
            "load_state_dict_from_checkpoint",
        }

        assert expected.issubset(abstract_methods), f"Missing abstract methods: {expected - abstract_methods}"


class MockNMoEModel:
    """Mock implementation for testing interface contracts."""

    def __init__(self):
        self._gradient_checkpointing = False
        self._expert_caches = {}
        self._last_aux_loss = 0.0
        self._load_stats = {}

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ) -> Dict[str, Any]:
        batch_size = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
        seq_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 10
        return {
            "logits": [[0.0] * 201088] * seq_len,  # Mock logits
            "past_key_values": None,
        }

    def generate(
        self,
        input_ids,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        **kwargs,
    ):
        return input_ids  # Return input as mock

    def forward_with_log_probs(
        self,
        input_ids,
        attention_mask=None,
        action_ids=None,
    ) -> Tuple[Any, Any]:
        return (0.0, [0.0])  # Mock log probs

    def refresh_expert_caches(self) -> None:
        self._expert_caches = {"refreshed": True}

    def get_expert_load_stats(self) -> Dict[str, Any]:
        return {"expert_0": 0.1, "expert_1": 0.15}

    def get_router_aux_loss(self) -> float:
        """Get router auxiliary loss."""
        return self._last_aux_loss

    def update_router_biases(self, gamma: float = 0.001) -> None:
        """Update router biases."""
        pass

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    @property
    def config(self):
        return {}

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return "float32"

    @property
    def uses_quantized_experts(self) -> bool:
        return False

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None

    def param_sets(self):
        return ([], [])

    def named_parameters_by_type(self):
        return {}

    def state_dict_for_save(self):
        return {}

    def load_state_dict_from_checkpoint(self, state_dict, strict=True):
        pass


class TestMockImplementation:
    """Tests for mock implementation behavior."""

    def test_forward_returns_dict(self):
        """Forward returns dict with logits."""
        model = MockNMoEModel()
        result = model.forward([[1, 2, 3]])

        assert isinstance(result, dict)
        assert "logits" in result

    def test_generate_returns_tokens(self):
        """Generate returns token tensor."""
        model = MockNMoEModel()
        input_ids = [[1, 2, 3]]
        result = model.generate(input_ids, max_new_tokens=10)

        assert result == input_ids

    def test_forward_with_log_probs_returns_tuple(self):
        """forward_with_log_probs returns tuple."""
        model = MockNMoEModel()
        logits, log_probs = model.forward_with_log_probs([[1, 2, 3]])

        assert logits == 0.0
        assert log_probs == [0.0]

    def test_refresh_expert_caches(self):
        """Expert cache refresh updates caches."""
        model = MockNMoEModel()
        model.refresh_expert_caches()

        assert model._expert_caches.get("refreshed") is True

    def test_get_expert_load_stats(self):
        """Load stats returns dict."""
        model = MockNMoEModel()
        stats = model.get_expert_load_stats()

        assert isinstance(stats, dict)
        assert "expert_0" in stats

    def test_get_router_aux_loss(self):
        """Router aux loss returns float."""
        model = MockNMoEModel()
        loss = model.get_router_aux_loss()

        assert isinstance(loss, float)

    def test_gradient_checkpointing_toggle(self):
        """Gradient checkpointing can be toggled."""
        model = MockNMoEModel()

        assert not model.is_gradient_checkpointing

        model.gradient_checkpointing_enable()
        assert model.is_gradient_checkpointing

        model.gradient_checkpointing_disable()
        assert not model.is_gradient_checkpointing


class TestInterfaceMethodSignatures:
    """Tests for method signature correctness."""

    def test_forward_signature(self):
        """Forward has correct signature."""
        import inspect
        from nmoe.unified.interface import NMoEModelInterface

        sig = inspect.signature(NMoEModelInterface.forward)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "input_ids" in params
        assert "attention_mask" in params
        assert "position_ids" in params
        assert "past_key_values" in params
        assert "use_cache" in params

    def test_generate_signature(self):
        """Generate has correct signature."""
        import inspect
        from nmoe.unified.interface import NMoEModelInterface

        sig = inspect.signature(NMoEModelInterface.generate)
        params = list(sig.parameters.keys())

        assert "input_ids" in params
        assert "max_new_tokens" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "top_k" in params
        assert "do_sample" in params

    def test_forward_with_log_probs_signature(self):
        """forward_with_log_probs has correct signature."""
        import inspect
        from nmoe.unified.interface import NMoEModelInterface

        sig = inspect.signature(NMoEModelInterface.forward_with_log_probs)
        params = list(sig.parameters.keys())

        assert "input_ids" in params
        assert "attention_mask" in params
        assert "action_ids" in params


class TestInterfaceDocumentation:
    """Tests for interface documentation."""

    def test_interface_has_docstring(self):
        """NMoEModelInterface has docstring."""
        from nmoe.unified.interface import NMoEModelInterface

        assert NMoEModelInterface.__doc__ is not None
        assert len(NMoEModelInterface.__doc__) > 50

    def test_forward_has_docstring(self):
        """Forward method has docstring."""
        from nmoe.unified.interface import NMoEModelInterface

        assert NMoEModelInterface.forward.__doc__ is not None
        assert "input_ids" in NMoEModelInterface.forward.__doc__
        assert "logits" in NMoEModelInterface.forward.__doc__

    def test_generate_has_docstring(self):
        """Generate method has docstring."""
        from nmoe.unified.interface import NMoEModelInterface

        assert NMoEModelInterface.generate.__doc__ is not None
        assert "max_new_tokens" in NMoEModelInterface.generate.__doc__


class TestInterfaceEdgeCases:
    """Tests for edge cases."""

    def test_optional_parameters_default_none(self):
        """Optional parameters default to None."""
        model = MockNMoEModel()

        # Should work without optional params
        result = model.forward([[1, 2, 3]])
        assert result is not None

    def test_empty_input(self):
        """Interface handles empty input gracefully."""
        model = MockNMoEModel()

        # Should not raise
        result = model.forward([[]])
        assert result is not None


class TestSubclassValidation:
    """Tests for subclass validation."""

    def test_incomplete_subclass_fails(self):
        """Incomplete subclass cannot be instantiated."""
        from nmoe.unified.interface import NMoEModelInterface

        class IncompleteModel(NMoEModelInterface):
            def forward(self, input_ids, **kwargs):
                return {}

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_complete_subclass_works(self):
        """Complete subclass can be instantiated."""
        from nmoe.unified.interface import NMoEModelInterface

        class CompleteModel(NMoEModelInterface):
            def forward(self, input_ids, **kwargs):
                return {"logits": None}

            def generate(self, input_ids, **kwargs):
                return input_ids

            def forward_with_log_probs(self, input_ids, **kwargs):
                return (0.0, [0.0])

            def refresh_expert_caches(self):
                pass

            def get_expert_load_stats(self):
                return {}

            def get_router_aux_loss(self):
                return 0.0

            def update_router_biases(self, gamma=0.001):
                pass

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                pass

            def gradient_checkpointing_disable(self):
                pass

            @property
            def is_gradient_checkpointing(self):
                return False

            @property
            def config(self):
                return {}

            @property
            def device(self):
                return "cpu"

            @property
            def dtype(self):
                return "float32"

            @property
            def uses_quantized_experts(self):
                return False

            def get_input_embeddings(self):
                return None

            def get_output_embeddings(self):
                return None

            def param_sets(self):
                return ([], [])

            def named_parameters_by_type(self):
                return {}

            def state_dict_for_save(self):
                return {}

            def load_state_dict_from_checkpoint(self, state_dict, strict=True):
                pass

        # Should not raise
        model = CompleteModel()
        assert model is not None
