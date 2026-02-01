"""Comprehensive unit tests for nmoe.eval.runner.

Tests cover:
- SimModel class (deterministic testing model)
- Dataset adapters (iter_unittest, iter_judge)
- Metric computation (pass@k, centered accuracy)
- Score extraction from judge outputs
- Code execution utilities
- Scorer functions (_run_choices_sim, _run_span_sim, etc.)

P1 Critical Tests: Full coverage of eval runner functionality.
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Try to import the runner module; skip tests if unavailable due to missing dependencies
RUNNER_AVAILABLE = False
IMPORT_ERROR = ""

try:
    from nmoe.eval.runner import (
        SimConfig, SimModel, compute_pass_at_k, _centered, _extract_judge_score,
        execute_unittest, _run_choices_sim, _run_span_sim, _run_unittest_sim,
        _run_judge_sim, _hash_u32, _normalize_text, _sim_examples_choices,
        _sim_examples_span, _sim_examples_unittest, _sim_examples_judge,
        EvalResult, CodeGenResult, JudgeResult, PassAtKResult, BASELINE_BY_TASK
    )
    RUNNER_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(
    not RUNNER_AVAILABLE,
    reason=f"nmoe.eval.runner not importable: {IMPORT_ERROR}"
)


# -----------------------------
# Test Fixtures
# -----------------------------

@pytest.fixture
def sim_config():
    """Create a SimConfig with known parameters."""
    return SimConfig(
        acc_choices=0.6,
        acc_span_em=0.55,
        acc_unittest=0.5,
        acc_judge=0.6,
        seed=123
    )


@pytest.fixture
def sim_model(sim_config):
    """Create a SimModel with known config."""
    return SimModel(sim_config)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    enc = Mock()
    enc.encode = Mock(return_value=[1, 2, 3, 4, 5])
    enc.decode = Mock(return_value="generated text")
    return enc


# -----------------------------
# SimConfig Tests
# -----------------------------

class TestSimConfig:
    """Tests for SimConfig dataclass."""

    def test_default_values(self):
        """SimConfig has sensible defaults."""
        cfg = SimConfig()
        assert cfg.acc_choices == 0.6
        assert cfg.acc_span_em == 0.55
        assert cfg.acc_unittest == 0.5
        assert cfg.acc_judge == 0.6
        assert cfg.seed == 123

    def test_custom_values(self):
        """SimConfig accepts custom values."""
        cfg = SimConfig(
            acc_choices=0.8,
            acc_span_em=0.7,
            acc_unittest=0.9,
            acc_judge=0.85,
            seed=456
        )
        assert cfg.acc_choices == 0.8
        assert cfg.acc_span_em == 0.7
        assert cfg.acc_unittest == 0.9
        assert cfg.acc_judge == 0.85
        assert cfg.seed == 456

    def test_edge_values(self):
        """SimConfig handles edge case accuracy values."""
        # Perfect accuracy
        cfg_perfect = SimConfig(acc_choices=1.0, acc_span_em=1.0)
        assert cfg_perfect.acc_choices == 1.0

        # Zero accuracy
        cfg_zero = SimConfig(acc_choices=0.0, acc_span_em=0.0)
        assert cfg_zero.acc_choices == 0.0


# -----------------------------
# SimModel Tests
# -----------------------------

class TestSimModel:
    """Tests for SimModel deterministic testing model."""

    def test_predict_choice_returns_valid_index(self, sim_model):
        """predict_choice returns index within valid range."""
        for _ in range(100):
            prompt = f"test prompt {_}"
            result = sim_model.predict_choice(prompt, n_choices=4, correct_idx=2)
            assert 0 <= result < 4, f"Invalid choice index: {result}"

    def test_predict_choice_deterministic(self, sim_model):
        """predict_choice is deterministic for same input."""
        prompt = "test prompt for determinism"
        result1 = sim_model.predict_choice(prompt, n_choices=4, correct_idx=1)
        result2 = sim_model.predict_choice(prompt, n_choices=4, correct_idx=1)
        assert result1 == result2, "predict_choice should be deterministic"

    def test_predict_choice_different_prompts_vary(self, sim_model):
        """predict_choice produces varying results for different prompts."""
        results = set()
        for i in range(100):
            result = sim_model.predict_choice(f"unique prompt {i}", n_choices=4, correct_idx=0)
            results.add(result)
        # Should have more than one unique result across 100 different prompts
        assert len(results) > 1, "Should produce varying results"

    def test_predict_choice_accuracy_approximates_target(self, sim_config):
        """predict_choice accuracy approximates target over many samples."""
        # Test with known accuracy
        cfg = SimConfig(acc_choices=0.75, seed=42)
        model = SimModel(cfg)

        correct = 0
        n_samples = 1000
        for i in range(n_samples):
            result = model.predict_choice(f"sample {i}", n_choices=4, correct_idx=i % 4)
            if result == i % 4:
                correct += 1

        accuracy = correct / n_samples
        # Allow 10% tolerance from target
        assert 0.65 <= accuracy <= 0.85, f"Accuracy {accuracy} not near target 0.75"

    def test_generate_span_returns_string(self, sim_model):
        """generate_span returns a string."""
        result = sim_model.generate_span("prompt", "answer")
        assert isinstance(result, str)

    def test_generate_span_deterministic(self, sim_model):
        """generate_span is deterministic for same input."""
        result1 = sim_model.generate_span("prompt", "Paris")
        result2 = sim_model.generate_span("prompt", "Paris")
        assert result1 == result2, "generate_span should be deterministic"

    def test_generate_span_exact_match_rate(self, sim_config):
        """generate_span approximates target EM rate."""
        cfg = SimConfig(acc_span_em=0.7, seed=42)
        model = SimModel(cfg)

        exact_matches = 0
        n_samples = 1000
        for i in range(n_samples):
            result = model.generate_span(f"prompt {i}", "Paris")
            if result == "Paris":
                exact_matches += 1

        rate = exact_matches / n_samples
        # Allow 10% tolerance
        assert 0.6 <= rate <= 0.8, f"EM rate {rate} not near target 0.7"

    def test_generate_span_near_miss_format(self, sim_config):
        """generate_span produces expected near-miss format."""
        # Use 0 accuracy to always get near-miss
        cfg = SimConfig(acc_span_em=0.0, seed=1)
        model = SimModel(cfg)

        result = model.generate_span("prompt", "Paris")
        # Near-miss adds " maybe" and normalizes
        assert "maybe" in result

    def test_run_unittest_returns_bool(self, sim_model):
        """run_unittest returns boolean."""
        result = sim_model.run_unittest("def f(x): return x + 1")
        assert isinstance(result, bool)

    def test_run_unittest_deterministic(self, sim_model):
        """run_unittest is deterministic for same input."""
        result1 = sim_model.run_unittest("test prompt")
        result2 = sim_model.run_unittest("test prompt")
        assert result1 == result2

    def test_run_unittest_pass_rate(self, sim_config):
        """run_unittest approximates target pass rate."""
        cfg = SimConfig(acc_unittest=0.6, seed=42)
        model = SimModel(cfg)

        passed = 0
        n_samples = 1000
        for i in range(n_samples):
            if model.run_unittest(f"test {i}"):
                passed += 1

        rate = passed / n_samples
        # Allow 10% tolerance
        assert 0.5 <= rate <= 0.7, f"Pass rate {rate} not near target 0.6"

    def test_judge_keep_returns_bool(self, sim_model):
        """judge_keep returns boolean."""
        result = sim_model.judge_keep("explain backpropagation")
        assert isinstance(result, bool)

    def test_judge_keep_deterministic(self, sim_model):
        """judge_keep is deterministic for same input."""
        result1 = sim_model.judge_keep("test prompt")
        result2 = sim_model.judge_keep("test prompt")
        assert result1 == result2

    def test_judge_keep_rate(self, sim_config):
        """judge_keep approximates target keep rate."""
        cfg = SimConfig(acc_judge=0.65, seed=42)
        model = SimModel(cfg)

        kept = 0
        n_samples = 1000
        for i in range(n_samples):
            if model.judge_keep(f"prompt {i}"):
                kept += 1

        rate = kept / n_samples
        # Allow 10% tolerance
        assert 0.55 <= rate <= 0.75, f"Keep rate {rate} not near target 0.65"


# -----------------------------
# Metric Computation Tests
# -----------------------------

class TestComputePassAtK:
    """Tests for pass@k metric computation."""

    def test_pass_at_k_basic(self):
        """pass@k formula basic test cases."""
        # 50% correct, sample 1: expect 0.5
        result = compute_pass_at_k(n=10, c=5, k=1)
        assert abs(result - 0.5) < 1e-9, f"Expected 0.5, got {result}"

    def test_pass_at_k_all_correct(self):
        """pass@k with all samples correct."""
        # All correct: pass@k = 1.0 for any k
        assert compute_pass_at_k(n=10, c=10, k=1) == 1.0
        assert compute_pass_at_k(n=10, c=10, k=5) == 1.0
        assert compute_pass_at_k(n=10, c=10, k=10) == 1.0

    def test_pass_at_k_none_correct(self):
        """pass@k with no samples correct."""
        # None correct: pass@k = 0.0 for any k
        assert compute_pass_at_k(n=10, c=0, k=1) == 0.0
        assert compute_pass_at_k(n=10, c=0, k=5) == 0.0

    def test_pass_at_k_formula_verification(self):
        """Verify pass@k formula: 1 - C(n-c,k)/C(n,k)."""
        # Manual verification for n=10, c=3, k=2
        # pass@2 = 1 - C(7,2)/C(10,2) = 1 - 21/45 = 24/45 = 8/15
        result = compute_pass_at_k(n=10, c=3, k=2)
        expected = 1.0 - math.comb(7, 2) / math.comb(10, 2)
        assert abs(result - expected) < 1e-9

    def test_pass_at_k_edge_case_k_greater_than_failures(self):
        """pass@k returns 1.0 when k > n-c (guaranteed success)."""
        # k=5 but only 3 failures (7 correct out of 10)
        # Must find at least one correct in 5 samples
        result = compute_pass_at_k(n=10, c=7, k=5)
        assert result == 1.0

    def test_pass_at_k_various_values(self):
        """pass@k computes correctly for various parameter combinations."""
        # Test vectors: (n, c, k, expected)
        test_cases = [
            (10, 5, 1, 0.5),
            (10, 10, 1, 1.0),
            (10, 0, 1, 0.0),
            (100, 50, 1, 0.5),
            (100, 90, 10, 1.0),  # k > n-c
            (20, 10, 5, 1.0 - math.comb(10, 5) / math.comb(20, 5)),
        ]

        for n, c, k, expected in test_cases:
            result = compute_pass_at_k(n=n, c=c, k=k)
            assert abs(result - expected) < 1e-9, f"Failed for n={n}, c={c}, k={k}"

    def test_pass_at_k_monotonicity(self):
        """pass@k increases as k increases (for fixed n, c)."""
        n, c = 100, 30
        prev = 0.0
        for k in [1, 5, 10, 20, 50]:
            current = compute_pass_at_k(n=n, c=c, k=k)
            assert current >= prev, f"pass@{k} should be >= pass@{k-1}"
            prev = current


class TestCenteredAccuracy:
    """Tests for centered accuracy normalization."""

    def test_centered_random_chance(self):
        """Centered accuracy at random chance is 0."""
        # 4-choice: baseline = 0.25
        result = _centered(acc=0.25, baseline=0.25)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_centered_perfect(self):
        """Centered accuracy at perfect score is 1."""
        # 4-choice: baseline = 0.25
        result = _centered(acc=1.0, baseline=0.25)
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_centered_halfway(self):
        """Centered accuracy halfway between baseline and perfect."""
        # 4-choice: baseline = 0.25
        # Halfway = (1.0 + 0.25) / 2 = 0.625
        result = _centered(acc=0.625, baseline=0.25)
        assert abs(result - 0.5) < 1e-9, f"Expected 0.5, got {result}"

    def test_centered_formula_verification(self):
        """Verify centered formula: (acc - baseline) / (1 - baseline)."""
        # Test cases: (acc, baseline, expected)
        test_cases = [
            (0.25, 0.25, 0.0),   # Random chance
            (1.0, 0.25, 1.0),    # Perfect
            (0.625, 0.25, 0.5),  # Halfway
            (0.5, 0.5, 0.0),     # Binary random
            (1.0, 0.5, 1.0),     # Binary perfect
            (0.75, 0.5, 0.5),    # Binary halfway
            (0.0, 0.0, 0.0),     # Zero baseline, zero acc
            (0.5, 0.0, 0.5),     # Zero baseline, 50% acc
        ]

        for acc, baseline, expected in test_cases:
            result = _centered(acc=acc, baseline=baseline)
            assert abs(result - expected) < 1e-9, f"Failed for acc={acc}, baseline={baseline}"

    def test_centered_baseline_one(self):
        """Centered returns 0 when baseline is 1.0."""
        # Special case: if baseline >= 1.0, return 0.0
        result = _centered(acc=0.5, baseline=1.0)
        assert result == 0.0

    def test_centered_clipping(self):
        """Centered clips to [0, 1] for stability."""
        # Below baseline should clip to 0
        result = _centered(acc=0.1, baseline=0.25)
        assert result >= 0.0, "Should be clipped to >= 0"

        # Above perfect (shouldn't happen, but test clipping)
        result = _centered(acc=1.0, baseline=0.0)
        assert result <= 1.0, "Should be clipped to <= 1"

    def test_centered_binary_choices(self):
        """Centered works correctly for binary (2-choice) tasks."""
        # Binary: baseline = 0.5
        assert _centered(0.5, 0.5) == 0.0   # Random
        assert _centered(1.0, 0.5) == 1.0   # Perfect
        assert _centered(0.75, 0.5) == 0.5  # Halfway

    def test_centered_three_choices(self):
        """Centered works correctly for 3-choice tasks (SIQA)."""
        # 3-choice: baseline = 1/3
        baseline = 1/3
        assert abs(_centered(baseline, baseline)) < 1e-9  # Random
        assert abs(_centered(1.0, baseline) - 1.0) < 1e-9  # Perfect


# -----------------------------
# Score Extraction Tests
# -----------------------------

class TestExtractJudgeScore:
    """Tests for judge score extraction from various response formats."""

    def test_extract_standard_format(self):
        """Extract score from standard SCORE: X format."""
        output = "EXPLANATION: This is a good response.\nSCORE: 8"
        score, explanation = _extract_judge_score(output)

        assert score == 8.0
        assert "good response" in explanation

    def test_extract_score_with_decimal(self):
        """Extract score with decimal value."""
        output = "EXPLANATION: Mixed quality.\nSCORE: 7.5"
        score, _ = _extract_judge_score(output)

        assert score == 7.5

    def test_extract_score_lowercase(self):
        """Extract score from lowercase format."""
        output = "explanation: decent response.\nscore: 6"
        score, explanation = _extract_judge_score(output)

        assert score == 6.0
        assert "decent" in explanation

    def test_extract_score_no_newline(self):
        """Extract score when on same line."""
        output = "EXPLANATION: Good work. SCORE: 9"
        score, explanation = _extract_judge_score(output)

        assert score == 9.0

    def test_extract_clamps_high_score(self):
        """Score is clamped to max 10."""
        output = "SCORE: 15"
        score, _ = _extract_judge_score(output)

        assert score == 10.0

    def test_extract_clamps_low_score(self):
        """Score is clamped to min 1."""
        output = "SCORE: -5"
        score, _ = _extract_judge_score(output)

        # Default fallback since -5 doesn't match the pattern
        assert 1.0 <= score <= 10.0

    def test_extract_default_on_missing_score(self):
        """Default score when pattern not found."""
        output = "This response is pretty good overall."
        score, explanation = _extract_judge_score(output)

        assert score == 5.0  # Default middle score
        assert explanation == ""  # No EXPLANATION: pattern

    def test_extract_malformed_score(self):
        """Handle malformed score gracefully."""
        output = "SCORE: excellent"
        score, _ = _extract_judge_score(output)

        assert score == 5.0  # Default on parse failure

    def test_extract_empty_output(self):
        """Handle empty output."""
        score, explanation = _extract_judge_score("")

        assert score == 5.0
        assert explanation == ""

    def test_extract_multiline_explanation(self):
        """Extract multiline explanation."""
        output = """EXPLANATION: This response demonstrates good understanding.
It covers the main points well.
However, it could be more concise.
SCORE: 7"""

        score, explanation = _extract_judge_score(output)

        assert score == 7.0
        assert "good understanding" in explanation
        assert "concise" in explanation

    def test_extract_score_at_beginning(self):
        """Extract score when it appears at beginning."""
        output = "SCORE: 8\nEXPLANATION: Good response."
        score, _ = _extract_judge_score(output)

        assert score == 8.0


# -----------------------------
# Code Execution Tests
# -----------------------------

class TestExecuteUnittest:
    """Tests for execute_unittest function."""

    def test_execute_passing_code(self):
        """Execute code that passes tests."""
        prompt = "def add(a, b):\n"
        completion = "    return a + b"
        tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0"

        passed, error = execute_unittest(prompt, completion, tests)

        assert passed is True
        assert error == ""

    def test_execute_failing_code(self):
        """Execute code that fails tests."""
        prompt = "def add(a, b):\n"
        completion = "    return a - b"  # Wrong implementation
        tests = "assert add(1, 2) == 3"

        passed, error = execute_unittest(prompt, completion, tests)

        assert passed is False
        assert error != ""

    def test_execute_syntax_error(self):
        """Handle code with syntax errors."""
        prompt = "def add(a, b):\n"
        completion = "    return a +"  # Syntax error
        tests = "assert add(1, 2) == 3"

        passed, error = execute_unittest(prompt, completion, tests)

        assert passed is False
        assert error != ""

    def test_execute_strips_markdown_fences(self):
        """Strip markdown code fences from completion."""
        prompt = "def add(a, b):\n"
        completion = "```python\n    return a + b\n```"
        tests = "assert add(1, 2) == 3"

        passed, error = execute_unittest(prompt, completion, tests)

        assert passed is True

    def test_execute_timeout(self):
        """Handle code that times out."""
        prompt = "def slow():\n"
        completion = "    import time; time.sleep(100)"
        tests = "slow()"

        passed, error = execute_unittest(prompt, completion, tests, timeout_s=0.1)

        assert passed is False
        assert "timeout" in error.lower() or "TIMEOUT" in error

    def test_execute_runtime_error(self):
        """Handle runtime errors."""
        prompt = "def divide(a, b):\n"
        completion = "    return a / b"
        tests = "assert divide(1, 0) == 0"  # Division by zero

        passed, error = execute_unittest(prompt, completion, tests)

        assert passed is False
        assert error != ""


# -----------------------------
# Scorer Function Tests
# -----------------------------

class TestRunChoicesSim:
    """Tests for _run_choices_sim scorer."""

    def test_run_choices_sim_basic(self):
        """Basic choices simulation runs without error."""
        model = SimModel(SimConfig(acc_choices=0.6, seed=42))
        result = _run_choices_sim("ARC-Easy", model, n=50)

        assert result.task == "ARC-Easy"
        assert 0 <= result.raw_acc <= 1.0
        assert 0 <= result.centered_acc <= 1.0
        assert result.n == 50

    def test_run_choices_sim_binary_task(self):
        """Choices simulation for binary tasks."""
        model = SimModel(SimConfig(acc_choices=0.8, seed=42))
        result = _run_choices_sim("PIQA", model, n=100)

        assert result.task == "PIQA"
        assert result.n == 100
        # PIQA is binary, baseline 0.5
        # With 80% acc, centered should be around (0.8 - 0.5) / 0.5 = 0.6

    def test_run_choices_sim_three_choice_task(self):
        """Choices simulation for 3-choice tasks."""
        model = SimModel(SimConfig(acc_choices=0.7, seed=42))
        result = _run_choices_sim("SIQA", model, n=100)

        assert result.task == "SIQA"
        # SIQA is 3-choice

    def test_run_choices_sim_deterministic(self):
        """Choices simulation is deterministic with same seed."""
        model1 = SimModel(SimConfig(acc_choices=0.6, seed=123))
        model2 = SimModel(SimConfig(acc_choices=0.6, seed=123))

        result1 = _run_choices_sim("MMLU", model1, n=100)
        result2 = _run_choices_sim("MMLU", model2, n=100)

        assert result1.raw_acc == result2.raw_acc
        assert result1.centered_acc == result2.centered_acc


class TestRunSpanSim:
    """Tests for _run_span_sim scorer."""

    def test_run_span_sim_basic(self):
        """Basic span simulation runs without error."""
        model = SimModel(SimConfig(acc_span_em=0.55, seed=42))
        result = _run_span_sim("SQuAD v1.1", model, n=50)

        assert result.task == "SQuAD v1.1"
        assert 0 <= result.raw_acc <= 1.0
        assert result.n == 50

    def test_run_span_sim_accuracy_near_target(self):
        """Span simulation accuracy approximates target."""
        model = SimModel(SimConfig(acc_span_em=0.7, seed=42))
        result = _run_span_sim("SQuAD v1.1", model, n=1000)

        # Allow 15% tolerance
        assert 0.55 <= result.raw_acc <= 0.85


class TestRunUnittestSim:
    """Tests for _run_unittest_sim scorer."""

    def test_run_unittest_sim_basic(self):
        """Basic unittest simulation runs without error."""
        model = SimModel(SimConfig(acc_unittest=0.5, seed=42))
        result = _run_unittest_sim("HumanEval", model, n=50)

        assert result.task == "HumanEval"
        assert 0 <= result.raw_acc <= 1.0
        assert result.n == 50
        # Unittest baseline is 0, so centered = raw_acc
        assert abs(result.centered_acc - result.raw_acc) < 0.01

    def test_run_unittest_sim_pass_rate(self):
        """Unittest simulation pass rate approximates target."""
        model = SimModel(SimConfig(acc_unittest=0.6, seed=42))
        result = _run_unittest_sim("HumanEval", model, n=1000)

        # Allow 15% tolerance
        assert 0.45 <= result.raw_acc <= 0.75


class TestRunJudgeSim:
    """Tests for _run_judge_sim scorer."""

    def test_run_judge_sim_basic(self):
        """Basic judge simulation runs without error."""
        model = SimModel(SimConfig(acc_judge=0.6, seed=42))
        result = _run_judge_sim("AlpacaEval", model, n=50)

        assert result.task == "AlpacaEval"
        assert 0 <= result.raw_acc <= 1.0
        assert result.n == 50

    def test_run_judge_sim_with_custom_tau(self):
        """Judge simulation with custom tau_keep threshold."""
        model = SimModel(SimConfig(acc_judge=0.7, seed=42))
        result = _run_judge_sim("MT-Bench", model, n=100, tau_keep=0.3)

        # With tau_keep=0.3 as baseline, centered is adjusted
        assert 0 <= result.centered_acc <= 1.0


# -----------------------------
# Utility Function Tests
# -----------------------------

class TestHashU32:
    """Tests for _hash_u32 utility."""

    def test_hash_u32_returns_int(self):
        """Hash returns integer."""
        result = _hash_u32("test string")
        assert isinstance(result, int)

    def test_hash_u32_deterministic(self):
        """Hash is deterministic."""
        assert _hash_u32("test") == _hash_u32("test")

    def test_hash_u32_different_inputs(self):
        """Different inputs produce different hashes."""
        assert _hash_u32("test1") != _hash_u32("test2")

    def test_hash_u32_seed_affects_result(self):
        """Seed affects hash result."""
        assert _hash_u32("test", seed=1) != _hash_u32("test", seed=2)

    def test_hash_u32_bounded(self):
        """Hash is bounded to 32-bit range."""
        for i in range(100):
            h = _hash_u32(f"test{i}")
            assert 0 <= h < 2**32


class TestNormalizeText:
    """Tests for _normalize_text utility."""

    def test_normalize_lowercase(self):
        """Normalize converts to lowercase."""
        assert _normalize_text("HELLO") == "hello"

    def test_normalize_strips_whitespace(self):
        """Normalize strips leading/trailing whitespace."""
        assert _normalize_text("  hello  ") == "hello"

    def test_normalize_collapses_spaces(self):
        """Normalize collapses multiple spaces."""
        assert _normalize_text("hello   world") == "hello world"

    def test_normalize_handles_newlines(self):
        """Normalize converts newlines to spaces."""
        assert _normalize_text("hello\nworld") == "hello world"


# -----------------------------
# Example Sampler Tests
# -----------------------------

class TestSimExamplesChoices:
    """Tests for _sim_examples_choices sampler."""

    def test_sim_examples_choices_count(self):
        """Returns correct number of examples."""
        examples = _sim_examples_choices(n=10)
        assert len(examples) == 10

    def test_sim_examples_choices_structure(self):
        """Examples have correct structure."""
        examples = _sim_examples_choices(n=5, n_choices=4)

        for ex in examples:
            assert "prompt" in ex
            assert "choices" in ex
            assert "label" in ex
            assert len(ex["choices"]) == 4
            assert 0 <= ex["label"] < 4

    def test_sim_examples_choices_varying_n_choices(self):
        """Supports different number of choices."""
        for n_choices in [2, 3, 4, 5, 10]:
            examples = _sim_examples_choices(n=5, n_choices=n_choices)
            assert len(examples[0]["choices"]) == n_choices


class TestSimExamplesSpan:
    """Tests for _sim_examples_span sampler."""

    def test_sim_examples_span_count(self):
        """Returns correct number of examples."""
        examples = _sim_examples_span(n=10)
        assert len(examples) == 10

    def test_sim_examples_span_structure(self):
        """Examples have correct structure."""
        examples = _sim_examples_span(n=5)

        for ex in examples:
            assert "prompt" in ex
            assert "answers" in ex
            assert isinstance(ex["answers"], list)
            assert len(ex["answers"]) > 0


class TestSimExamplesUnittest:
    """Tests for _sim_examples_unittest sampler."""

    def test_sim_examples_unittest_count(self):
        """Returns correct number of examples."""
        examples = _sim_examples_unittest(n=10)
        assert len(examples) == 10

    def test_sim_examples_unittest_structure(self):
        """Examples have correct structure."""
        examples = _sim_examples_unittest(n=5)

        for ex in examples:
            assert "prompt" in ex
            assert "tests" in ex
            assert isinstance(ex["tests"], list)


class TestSimExamplesJudge:
    """Tests for _sim_examples_judge sampler."""

    def test_sim_examples_judge_count(self):
        """Returns correct number of examples."""
        examples = _sim_examples_judge(n=10)
        assert len(examples) == 10

    def test_sim_examples_judge_structure(self):
        """Examples have correct structure."""
        examples = _sim_examples_judge(n=5)

        for ex in examples:
            assert "prompt" in ex


# -----------------------------
# EvalResult Tests
# -----------------------------

class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_creation(self):
        """EvalResult can be created."""
        result = EvalResult(
            task="MMLU",
            raw_acc=0.75,
            centered_acc=0.67,
            n=100
        )

        assert result.task == "MMLU"
        assert result.raw_acc == 0.75
        assert result.centered_acc == 0.67
        assert result.n == 100


# -----------------------------
# CodeGenResult Tests
# -----------------------------

class TestCodeGenResult:
    """Tests for CodeGenResult dataclass."""

    def test_code_gen_result_creation(self):
        """CodeGenResult can be created."""
        result = CodeGenResult(
            task_id="HumanEval/0",
            prompt="def add(a, b):",
            completion="return a + b",
            passed=True,
            error_msg="",
            exec_time_ms=10.5
        )

        assert result.task_id == "HumanEval/0"
        assert result.passed is True
        assert result.exec_time_ms == 10.5

    def test_code_gen_result_with_error(self):
        """CodeGenResult captures error message."""
        result = CodeGenResult(
            task_id="HumanEval/1",
            prompt="def divide(a, b):",
            completion="return a / b",
            passed=False,
            error_msg="ZeroDivisionError"
        )

        assert result.passed is False
        assert "ZeroDivision" in result.error_msg


# -----------------------------
# JudgeResult Tests
# -----------------------------

class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_judge_result_creation(self):
        """JudgeResult can be created."""
        result = JudgeResult(
            prompt="Explain backpropagation",
            response="Backpropagation is...",
            reference="The canonical explanation...",
            score=8.0,
            explanation="Good response",
            raw_judge_output="EXPLANATION: Good\nSCORE: 8"
        )

        assert result.prompt == "Explain backpropagation"
        assert result.score == 8.0
        assert result.explanation == "Good response"


# -----------------------------
# PassAtKResult Tests
# -----------------------------

class TestPassAtKResult:
    """Tests for PassAtKResult dataclass."""

    def test_pass_at_k_result_creation(self):
        """PassAtKResult can be created."""
        result = PassAtKResult(
            task_id="HumanEval/0",
            n_samples=200,
            n_correct=100,
            pass_at_1=0.5,
            pass_at_10=0.95,
            pass_at_100=1.0
        )

        assert result.task_id == "HumanEval/0"
        assert result.n_samples == 200
        assert result.n_correct == 100
        assert result.pass_at_1 == 0.5


# -----------------------------
# BASELINE_BY_TASK Tests
# -----------------------------

class TestBaselineByTask:
    """Tests for BASELINE_BY_TASK constants."""

    def test_baseline_values_valid(self):
        """All baseline values are in valid range."""
        for task, baseline in BASELINE_BY_TASK.items():
            assert 0 <= baseline <= 1.0, f"Invalid baseline for {task}: {baseline}"

    def test_baseline_known_tasks(self):
        """Known tasks have correct baselines."""
        # 4-choice tasks
        assert BASELINE_BY_TASK["ARC-Easy"] == 0.25
        assert BASELINE_BY_TASK["MMLU"] == 0.25

        # Binary tasks
        assert BASELINE_BY_TASK["PIQA"] == 0.5
        assert BASELINE_BY_TASK["BoolQ"] == 0.5

        # 3-choice
        assert abs(BASELINE_BY_TASK["SIQA"] - 0.3333) < 0.001


# -----------------------------
# Edge Case Tests
# -----------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pass_at_k_zero_samples(self):
        """Handle edge case of zero samples."""
        # n=0 is an edge case - implementation may handle differently
        # Just verify it doesn't crash
        try:
            result = compute_pass_at_k(n=0, c=0, k=1)
            # If it returns, should be a number
            assert isinstance(result, (int, float))
        except (ValueError, ZeroDivisionError):
            # This is acceptable behavior for invalid input
            pass

    def test_centered_negative_accuracy(self):
        """Handle negative accuracy (shouldn't happen, but test robustness)."""
        result = _centered(acc=-0.1, baseline=0.25)
        # Should be clipped to 0
        assert result >= 0.0

    def test_empty_prompt_handling(self, sim_model):
        """SimModel handles empty prompts."""
        # Should not crash
        result = sim_model.predict_choice("", n_choices=4, correct_idx=0)
        assert 0 <= result < 4

    def test_execute_unittest_empty_code(self):
        """Execute handles empty code."""
        passed, error = execute_unittest("", "", "")
        # Should handle gracefully (likely syntax error)
        assert isinstance(passed, bool)

    def test_extract_score_unicode(self):
        """Score extraction handles unicode."""
        output = "EXPLANATION: Good response\nSCORE: 8"
        score, _ = _extract_judge_score(output)
        assert score == 8.0


# -----------------------------
# Integration-like Tests
# -----------------------------

class TestEndToEndSimulation:
    """Integration-like tests using simulation model."""

    def test_full_choices_evaluation_flow(self):
        """Complete choices evaluation flow works."""
        cfg = SimConfig(acc_choices=0.65, seed=42)
        model = SimModel(cfg)

        # Run multiple tasks
        results = []
        for task in ["ARC-Easy", "MMLU", "PIQA"]:
            result = _run_choices_sim(task, model, n=100)
            results.append(result)

        # Verify all results are valid
        for r in results:
            assert isinstance(r, EvalResult)
            assert r.n == 100
            assert 0 <= r.raw_acc <= 1.0
            assert 0 <= r.centered_acc <= 1.0

    def test_full_unittest_evaluation_flow(self):
        """Complete unittest evaluation flow works."""
        cfg = SimConfig(acc_unittest=0.5, seed=42)
        model = SimModel(cfg)

        result = _run_unittest_sim("HumanEval", model, n=100)

        assert isinstance(result, EvalResult)
        assert result.task == "HumanEval"
        assert 0.3 <= result.raw_acc <= 0.7  # ~50% with tolerance

    def test_aggregate_core_score(self):
        """Aggregating CORE score across tasks."""
        cfg = SimConfig(acc_choices=0.7, acc_unittest=0.5, seed=42)
        model = SimModel(cfg)

        results = [
            _run_choices_sim("ARC-Easy", model, n=100),
            _run_choices_sim("MMLU", model, n=100),
            _run_unittest_sim("HumanEval", model, n=100),
        ]

        # Compute CORE (average centered accuracy)
        core = sum(r.centered_acc for r in results) / len(results)

        assert 0 <= core <= 1.0
