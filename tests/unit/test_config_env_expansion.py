"""Unit tests for nmoe.config environment variable expansion.

Tests cover:
- Environment variable expansion with ${VAR} syntax
- Default value syntax ${VAR:-default}
- Nested expansion in dicts and lists
- Allowlist enforcement (_ENV_VAR_PREFIXES)
- load_toml() with env var placeholders
- _check_unresolved() for catching typos
- Error handling for disallowed prefixes and unresolved vars
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any

from nmoe.config import (
    _expand_env_vars,
    _check_unresolved,
    load_toml,
    ConfigEnvError,
    _ENV_VAR_PREFIXES,
)


class TestEnvVarExpansionBasic:
    """Tests for basic ${VAR} expansion."""

    def test_simple_expansion(self, monkeypatch):
        """${VAR} expands to environment value."""
        monkeypatch.setenv("NMOE_TEST_VAR", "test_value")
        result = _expand_env_vars("${NMOE_TEST_VAR}")
        assert result == "test_value"

    def test_expansion_in_string(self, monkeypatch):
        """${VAR} expands within larger string."""
        monkeypatch.setenv("NMOE_PATH", "/data/models")
        result = _expand_env_vars("prefix/${NMOE_PATH}/suffix")
        assert result == "prefix//data/models/suffix"

    def test_multiple_vars_in_string(self, monkeypatch):
        """Multiple ${VAR} patterns expand in same string."""
        monkeypatch.setenv("NMOE_USER", "alice")
        monkeypatch.setenv("NMOE_PROJECT", "nmoe")
        result = _expand_env_vars("/home/${NMOE_USER}/${NMOE_PROJECT}")
        assert result == "/home/alice/nmoe"

    def test_hydra_prefix_allowed(self, monkeypatch):
        """HYDRA_ prefix is in allowlist."""
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        result = _expand_env_vars("${HYDRA_FULL_ERROR}")
        assert result == "1"

    def test_non_string_passthrough(self):
        """Non-string scalars pass through unchanged."""
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None


class TestEnvVarExpansionDefaults:
    """Tests for ${VAR:-default} syntax."""

    def test_default_when_var_unset(self, monkeypatch):
        """${VAR:-default} uses default when VAR not set."""
        monkeypatch.delenv("NMOE_UNSET_VAR", raising=False)
        result = _expand_env_vars("${NMOE_UNSET_VAR:-default_val}")
        assert result == "default_val"

    def test_var_value_overrides_default(self, monkeypatch):
        """${VAR:-default} uses VAR value when set."""
        monkeypatch.setenv("NMOE_SET_VAR", "actual_value")
        result = _expand_env_vars("${NMOE_SET_VAR:-default_val}")
        assert result == "actual_value"

    def test_empty_default(self, monkeypatch):
        """${VAR:-} allows empty default."""
        monkeypatch.delenv("NMOE_UNSET", raising=False)
        result = _expand_env_vars("prefix${NMOE_UNSET:-}suffix")
        assert result == "prefixsuffix"

    def test_default_with_path(self, monkeypatch):
        """Default can contain path characters."""
        monkeypatch.delenv("NMOE_DATA_DIR", raising=False)
        result = _expand_env_vars("${NMOE_DATA_DIR:-/default/data/path}")
        assert result == "/default/data/path"

    def test_default_with_special_chars(self, monkeypatch):
        """Default can contain special characters."""
        monkeypatch.delenv("NMOE_PATTERN", raising=False)
        result = _expand_env_vars("${NMOE_PATTERN:-*.txt}")
        assert result == "*.txt"


class TestEnvVarExpansionNested:
    """Tests for nested expansion in dicts and lists."""

    def test_dict_expansion(self, monkeypatch):
        """Env vars expand in dict values."""
        monkeypatch.setenv("NMOE_VALUE", "expanded")
        obj = {"key": "${NMOE_VALUE}", "nested": {"inner": "${NMOE_VALUE}"}}
        result = _expand_env_vars(obj)
        assert result == {"key": "expanded", "nested": {"inner": "expanded"}}

    def test_list_expansion(self, monkeypatch):
        """Env vars expand in list items."""
        monkeypatch.setenv("NMOE_ITEM", "value")
        obj = ["${NMOE_ITEM}", "literal", "${NMOE_ITEM}"]
        result = _expand_env_vars(obj)
        assert result == ["value", "literal", "value"]

    def test_mixed_nested_structure(self, monkeypatch):
        """Env vars expand in complex nested structures."""
        monkeypatch.setenv("NMOE_A", "alpha")
        monkeypatch.setenv("NMOE_B", "beta")
        obj = {
            "paths": ["${NMOE_A}", "${NMOE_B}"],
            "config": {
                "name": "${NMOE_A}",
                "items": [{"val": "${NMOE_B}"}],
            },
        }
        result = _expand_env_vars(obj)
        assert result["paths"] == ["alpha", "beta"]
        assert result["config"]["name"] == "alpha"
        assert result["config"]["items"][0]["val"] == "beta"

    def test_dict_keys_not_expanded(self, monkeypatch):
        """Dict keys are not expanded (only values)."""
        monkeypatch.setenv("NMOE_KEY", "new_key")
        obj = {"${NMOE_KEY}": "value"}
        result = _expand_env_vars(obj)
        # Key should remain unchanged
        assert "${NMOE_KEY}" in result
        assert result["${NMOE_KEY}"] == "value"


class TestEnvVarPrefixAllowlist:
    """Tests for _ENV_VAR_PREFIXES allowlist enforcement."""

    def test_allowed_prefixes_constant(self):
        """Verify allowed prefixes are NMOE_ and HYDRA_."""
        assert "NMOE_" in _ENV_VAR_PREFIXES
        assert "HYDRA_" in _ENV_VAR_PREFIXES
        assert len(_ENV_VAR_PREFIXES) == 2

    def test_disallowed_prefix_raises(self, monkeypatch):
        """Non-allowlisted prefix raises ConfigEnvError."""
        monkeypatch.setenv("SECRET_KEY", "sensitive")
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${SECRET_KEY}")
        assert "SECRET_KEY" in str(exc_info.value)
        assert "not allowed" in str(exc_info.value)

    def test_hf_prefix_blocked(self, monkeypatch):
        """HF_ prefix is intentionally blocked (security)."""
        monkeypatch.setenv("HF_TOKEN", "secret_token")
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${HF_TOKEN}")
        assert "HF_TOKEN" in str(exc_info.value)

    def test_path_var_blocked(self, monkeypatch):
        """Common env vars like PATH are blocked."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${PATH}")
        assert "PATH" in str(exc_info.value)

    def test_home_var_blocked(self):
        """HOME env var is blocked."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${HOME}")
        assert "HOME" in str(exc_info.value)

    def test_aws_prefix_blocked(self, monkeypatch):
        """AWS credentials prefix is blocked."""
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${AWS_SECRET_ACCESS_KEY}")
        assert "AWS_SECRET_ACCESS_KEY" in str(exc_info.value)

    def test_error_message_includes_allowed_prefixes(self, monkeypatch):
        """Error message lists allowed prefixes."""
        monkeypatch.setenv("BAD_VAR", "value")
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${BAD_VAR}")
        error_msg = str(exc_info.value)
        assert "NMOE_" in error_msg
        assert "HYDRA_" in error_msg


class TestCheckUnresolved:
    """Tests for _check_unresolved() function."""

    def test_no_placeholders_passes(self):
        """Strings without ${...} pass silently."""
        _check_unresolved("normal string")
        _check_unresolved({"key": "value"})
        _check_unresolved(["item1", "item2"])

    def test_unresolved_placeholder_raises(self):
        """Remaining ${...} placeholder raises error."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved("path/${NMOE_TYPO}/file")
        assert "NMOE_TYPO" in str(exc_info.value)
        assert "Unresolved" in str(exc_info.value)

    def test_unresolved_in_dict_raises(self):
        """Unresolved placeholder in dict value raises error."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved({"key": "${NMOE_MISSING}"})
        assert "NMOE_MISSING" in str(exc_info.value)

    def test_unresolved_in_list_raises(self):
        """Unresolved placeholder in list raises error."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved(["valid", "${NMOE_BAD}"])
        assert "NMOE_BAD" in str(exc_info.value)

    def test_unresolved_nested_raises(self):
        """Unresolved placeholder in nested structure raises error."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved({"outer": {"inner": [{"deep": "${NMOE_DEEP}"}]}})
        assert "NMOE_DEEP" in str(exc_info.value)

    def test_malformed_placeholder_detected(self):
        """Malformed placeholder like ${bad is detected."""
        # Note: This should still detect partial patterns
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved("${NMOE_MALFORMED}")
        assert "Unresolved" in str(exc_info.value)

    def test_source_in_error_message(self):
        """Source path appears in error message."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved("${NMOE_VAR}", source="/path/to/config.toml")
        assert "/path/to/config.toml" in str(exc_info.value)


class TestLoadToml:
    """Tests for load_toml() with env var expansion."""

    def test_load_simple_toml(self, monkeypatch, tmp_path):
        """load_toml() parses basic TOML file."""
        toml_content = """
[model]
name = "test_model"
layers = 32
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        result = load_toml(toml_file)
        assert result["model"]["name"] == "test_model"
        assert result["model"]["layers"] == 32

    def test_load_toml_with_env_expansion(self, monkeypatch, tmp_path):
        """load_toml() expands ${VAR} placeholders."""
        monkeypatch.setenv("NMOE_MODEL_PATH", "/models/nmoe-7b")
        toml_content = """
[paths]
model = "${NMOE_MODEL_PATH}"
data = "${NMOE_DATA_PATH:-/default/data}"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        result = load_toml(toml_file)
        assert result["paths"]["model"] == "/models/nmoe-7b"
        assert result["paths"]["data"] == "/default/data"

    def test_load_toml_nested_expansion(self, monkeypatch, tmp_path):
        """load_toml() expands env vars in nested structures."""
        monkeypatch.setenv("NMOE_ROOT", "/nmoe")
        toml_content = """
[training]
checkpoint_dir = "${NMOE_ROOT}/checkpoints"
log_dir = "${NMOE_ROOT}/logs"

[[training.stages]]
name = "pretrain"
data = "${NMOE_ROOT}/data/pretrain"

[[training.stages]]
name = "finetune"
data = "${NMOE_ROOT}/data/finetune"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        result = load_toml(toml_file)
        assert result["training"]["checkpoint_dir"] == "/nmoe/checkpoints"
        assert result["training"]["stages"][0]["data"] == "/nmoe/data/pretrain"
        assert result["training"]["stages"][1]["data"] == "/nmoe/data/finetune"

    def test_load_toml_disallowed_var_raises(self, monkeypatch, tmp_path):
        """load_toml() raises ConfigEnvError for disallowed var."""
        monkeypatch.setenv("SECRET_API_KEY", "secret123")
        toml_content = """
[api]
key = "${SECRET_API_KEY}"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        with pytest.raises(ConfigEnvError) as exc_info:
            load_toml(toml_file)
        assert "SECRET_API_KEY" in str(exc_info.value)

    def test_load_toml_unresolved_var_raises(self, monkeypatch, tmp_path):
        """load_toml() raises ConfigEnvError for unresolved var."""
        monkeypatch.delenv("NMOE_REQUIRED_VAR", raising=False)
        toml_content = """
[paths]
required = "${NMOE_REQUIRED_VAR}"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        with pytest.raises(ConfigEnvError) as exc_info:
            load_toml(toml_file)
        assert "NMOE_REQUIRED_VAR" in str(exc_info.value)

    def test_load_toml_path_object(self, monkeypatch, tmp_path):
        """load_toml() accepts Path objects."""
        monkeypatch.setenv("NMOE_VALUE", "42")
        toml_content = """
value = "${NMOE_VALUE}"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        result = load_toml(Path(toml_file))
        assert result["value"] == "42"

    def test_load_toml_string_path(self, monkeypatch, tmp_path):
        """load_toml() accepts string paths."""
        monkeypatch.setenv("NMOE_VALUE", "test")
        toml_content = """
value = "${NMOE_VALUE}"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        result = load_toml(str(toml_file))
        assert result["value"] == "test"

    def test_load_toml_file_path_in_error(self, monkeypatch, tmp_path):
        """Error message includes file path."""
        monkeypatch.delenv("NMOE_MISSING", raising=False)
        toml_content = """
path = "${NMOE_MISSING}"
"""
        toml_file = tmp_path / "my_config.toml"
        toml_file.write_text(toml_content)

        with pytest.raises(ConfigEnvError) as exc_info:
            load_toml(toml_file)
        assert "my_config.toml" in str(exc_info.value)


class TestLoadTomlComplexScenarios:
    """Tests for complex TOML loading scenarios."""

    def test_mixture_config_style(self, monkeypatch, tmp_path):
        """Test realistic mixture.toml style config."""
        monkeypatch.setenv("NMOE_DATA_ROOT", "/data")
        toml_content = """
[mixture]
name = "pretrain_v1"

[[mixture.sources]]
name = "fineweb"
path = "${NMOE_DATA_ROOT}/fineweb"
weight = 0.5

[[mixture.sources]]
name = "code"
path = "${NMOE_DATA_ROOT}/code"
weight = 0.3

[[mixture.sources]]
name = "math"
path = "${NMOE_DATA_ROOT:-/default}/math"
weight = 0.2
"""
        toml_file = tmp_path / "mixture.toml"
        toml_file.write_text(toml_content)

        result = load_toml(toml_file)
        assert result["mixture"]["sources"][0]["path"] == "/data/fineweb"
        assert result["mixture"]["sources"][1]["path"] == "/data/code"
        assert result["mixture"]["sources"][2]["path"] == "/data/math"

    def test_eval_tasks_style(self, monkeypatch, tmp_path):
        """Test realistic eval tasks.toml style config."""
        monkeypatch.setenv("NMOE_EVAL_DATA", "/eval")
        monkeypatch.setenv("HYDRA_SWEEP_DIR", "/sweeps")
        toml_content = """
[eval]
output_dir = "${HYDRA_SWEEP_DIR}/results"

[[eval.tasks]]
name = "mmlu"
data_path = "${NMOE_EVAL_DATA}/mmlu"
max_examples = 1000

[[eval.tasks]]
name = "hellaswag"
data_path = "${NMOE_EVAL_DATA}/hellaswag"
max_examples = 500
"""
        toml_file = tmp_path / "tasks.toml"
        toml_file.write_text(toml_content)

        result = load_toml(toml_file)
        assert result["eval"]["output_dir"] == "/sweeps/results"
        assert result["eval"]["tasks"][0]["data_path"] == "/eval/mmlu"

    def test_all_types_preserved(self, monkeypatch, tmp_path):
        """TOML types are preserved after expansion."""
        monkeypatch.setenv("NMOE_NAME", "test")
        toml_content = """
string_val = "${NMOE_NAME}"
int_val = 42
float_val = 3.14
bool_val = true
array_val = [1, 2, 3]
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        result = load_toml(toml_file)
        assert result["string_val"] == "test"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["array_val"] == [1, 2, 3]


class TestErrorMessages:
    """Tests for error message quality."""

    def test_disallowed_var_shows_allowed_prefixes(self, monkeypatch):
        """Error for disallowed var lists what's allowed."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${BAD_PREFIX_VAR}")
        error_msg = str(exc_info.value)
        # Should mention allowed prefixes
        for prefix in _ENV_VAR_PREFIXES:
            assert prefix in error_msg

    def test_unresolved_var_suggests_default(self, monkeypatch):
        """Error for unresolved var suggests default syntax."""
        monkeypatch.delenv("NMOE_MISSING", raising=False)
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${NMOE_MISSING}")
        error_msg = str(exc_info.value)
        assert ":-" in error_msg  # Suggests default syntax

    def test_source_path_in_expansion_error(self, monkeypatch):
        """Source path appears in expansion error."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _expand_env_vars("${BAD_VAR}", source="/etc/nmoe/config.toml")
        assert "/etc/nmoe/config.toml" in str(exc_info.value)

    def test_source_path_in_unresolved_error(self):
        """Source path appears in unresolved error."""
        with pytest.raises(ConfigEnvError) as exc_info:
            _check_unresolved("${NMOE_VAR}", source="/custom/path.toml")
        assert "/custom/path.toml" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string(self):
        """Empty string passes through."""
        assert _expand_env_vars("") == ""

    def test_dollar_without_brace(self, monkeypatch):
        """Plain $ without {} is not expanded."""
        result = _expand_env_vars("$NMOE_VAR")
        assert result == "$NMOE_VAR"

    def test_incomplete_placeholder(self):
        """Incomplete ${... without closing brace."""
        # This should pass _expand_env_vars but fail _check_unresolved
        result = _expand_env_vars("${NMOE_INCOMPLETE")
        assert result == "${NMOE_INCOMPLETE"  # Regex doesn't match

    def test_escaped_dollar_literal(self, monkeypatch):
        """$$ is not treated as variable (no special escaping)."""
        # The regex specifically looks for ${...} pattern
        result = _expand_env_vars("$$NMOE_VAR")
        assert result == "$$NMOE_VAR"

    def test_deeply_nested_structure(self, monkeypatch):
        """Deeply nested structures expand correctly."""
        monkeypatch.setenv("NMOE_DEEP", "found")
        obj = {"a": {"b": {"c": {"d": {"e": "${NMOE_DEEP}"}}}}}
        result = _expand_env_vars(obj)
        assert result["a"]["b"]["c"]["d"]["e"] == "found"

    def test_var_with_numbers(self, monkeypatch):
        """Variable names can contain numbers."""
        monkeypatch.setenv("NMOE_VAR123", "numeric")
        result = _expand_env_vars("${NMOE_VAR123}")
        assert result == "numeric"

    def test_var_with_underscores(self, monkeypatch):
        """Variable names can contain underscores."""
        monkeypatch.setenv("NMOE_MY_LONG_VAR_NAME", "underscored")
        result = _expand_env_vars("${NMOE_MY_LONG_VAR_NAME}")
        assert result == "underscored"

    def test_default_with_equals_sign(self, monkeypatch):
        """Default value can contain equals sign."""
        monkeypatch.delenv("NMOE_FLAGS", raising=False)
        result = _expand_env_vars("${NMOE_FLAGS:-key=value}")
        assert result == "key=value"

    def test_empty_env_value(self, monkeypatch):
        """Empty env value is valid (not treated as unset)."""
        monkeypatch.setenv("NMOE_EMPTY", "")
        result = _expand_env_vars("prefix${NMOE_EMPTY}suffix")
        assert result == "prefixsuffix"

    def test_whitespace_in_value(self, monkeypatch):
        """Env value can contain whitespace."""
        monkeypatch.setenv("NMOE_SPACED", "hello world")
        result = _expand_env_vars("${NMOE_SPACED}")
        assert result == "hello world"


class TestConfigEnvErrorException:
    """Tests for ConfigEnvError exception class."""

    def test_exception_is_exception(self):
        """ConfigEnvError is an Exception subclass."""
        assert issubclass(ConfigEnvError, Exception)

    def test_exception_message(self):
        """ConfigEnvError stores message correctly."""
        error = ConfigEnvError("test message")
        assert str(error) == "test message"

    def test_exception_can_be_raised(self):
        """ConfigEnvError can be raised and caught."""
        with pytest.raises(ConfigEnvError):
            raise ConfigEnvError("test")

    def test_exception_in_try_except(self):
        """ConfigEnvError works in try/except blocks."""
        try:
            raise ConfigEnvError("caught")
        except ConfigEnvError as e:
            assert "caught" in str(e)
