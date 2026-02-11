from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from nmoe.metrics import start_metrics, stop_metrics
from nmoe.eval.adapters import iter_choices, iter_span
from nmoe.eval.execution import run_python


# -----------------------------
# Utilities
# -----------------------------

def _load_snapshot(path: Path) -> tuple[dict, dict]:
    obj = torch.load(str(path), map_location="cpu")
    cfg_dict = obj.get("config", {})
    model_state = obj.get("model_state", {})
    return cfg_dict, model_state


def _hash_u32(s: str, seed: int = 0) -> int:
    h = hashlib.blake2s((str(seed) + s).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# -----------------------------
# Simulation model (deterministic)
# -----------------------------

@dataclass
class SimConfig:
    acc_choices: float = 0.6  # target accuracy for choices tasks
    acc_span_em: float = 0.55  # exact-match rate for span tasks
    acc_unittest: float = 0.5  # pass rate
    acc_judge: float = 0.6     # keep rate
    seed: int = 123


class SimModel:
    """Deterministic pseudo-model for end-to-end eval plumbing tests.

    Produces controlled accuracies without requiring a trained model.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def predict_choice(self, prompt: str, n_choices: int, correct_idx: int) -> int:
        # Hash prompt → pseudo RNG; hit target accuracy by thresholding
        r = (_hash_u32(prompt, self.cfg.seed) % 10000) / 10000.0
        if r < self.cfg.acc_choices:
            return int(correct_idx)
        else:
            # Pick a wrong answer deterministically
            return (correct_idx + 1 + (int(r * 17) % max(1, n_choices - 1))) % n_choices

    def generate_span(self, prompt: str, answer: str) -> str:
        r = (_hash_u32(prompt + answer, self.cfg.seed) % 10000) / 10000.0
        if r < self.cfg.acc_span_em:
            return answer
        # Return a near-miss string
        return answer.lower().strip().rstrip(".") + " maybe"

    def run_unittest(self, prompt: str) -> bool:
        r = (_hash_u32("UT:" + prompt, self.cfg.seed) % 10000) / 10000.0
        return r < self.cfg.acc_unittest

    def judge_keep(self, prompt: str) -> bool:
        r = (_hash_u32("J:" + prompt, self.cfg.seed) % 10000) / 10000.0
        return r < self.cfg.acc_judge


# -----------------------------
# Task loading (TOML)
# -----------------------------

def _load_tasks_toml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        import tomllib
        with path.open("rb") as f:
            obj = tomllib.load(f)
        tasks = obj.get("task", [])
        return tasks if isinstance(tasks, list) else []
    except Exception:
        return []


# -----------------------------
# Example samplers (simulation)
# -----------------------------

def _sim_examples_choices(n: int, n_choices: int = 4) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"Q{i}: What is the best option?"
        choices = [f"Option {chr(65+j)}" for j in range(n_choices)]
        correct = i % n_choices
        out.append({"prompt": prompt, "choices": choices, "label": correct})
    return out


def _sim_examples_span(n: int) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"S{i}: The capital of France is"
        answer = "Paris"
        out.append({"prompt": prompt, "answers": [answer]})
    return out


def _sim_examples_unittest(n: int) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"Write a function f{i}(x) that returns x+1"
        out.append({"prompt": prompt, "tests": ["assert f(1)==2", "assert f(5)==6"]})
    return out


def _sim_examples_judge(n: int) -> list[dict]:
    out = []
    for i in range(n):
        prompt = f"Explain backpropagation in 2 sentences (item {i})"
        out.append({"prompt": prompt})
    return out


# -----------------------------
# Unittest Dataset Adapters
# -----------------------------

def iter_unittest(name: str, source: str, max_examples: int) -> Iterator[Dict]:
    """Iterate over code generation datasets (HumanEval, MBPP, etc.)."""
    from datasets import load_dataset

    scheme, args = _parse_source(source)
    assert scheme == "hf", f"Unsupported scheme for unittest: {scheme}"

    if len(args) == 3:
        dataset, subset, split = args
        ds = load_dataset(dataset, subset, split=split, trust_remote_code=True)
    elif len(args) == 2:
        dataset, split = args
        ds = load_dataset(dataset, split=split, trust_remote_code=True)
    else:
        raise ValueError(f"Bad source spec: {source}")

    n = 0
    dataset_base = dataset.split("/")[-1]
    for ex in ds:
        try:
            if dataset_base in ("openai_humaneval", "humaneval"):
                # HumanEval format: prompt (function signature + docstring), test, entry_point
                prompt = ex.get("prompt", "")
                test = ex.get("test", "")
                entry_point = ex.get("entry_point", "")
                canonical_solution = ex.get("canonical_solution", "")
                task_id = ex.get("task_id", f"HumanEval/{n}")
                yield {
                    "prompt": prompt,
                    "tests": test,
                    "entry_point": entry_point,
                    "canonical_solution": canonical_solution,
                    "task_id": task_id,
                }
            elif dataset_base in ("mbpp", "google-research-datasets/mbpp", "sanitized_mbpp"):
                # MBPP format: text (description), code (solution), test_list
                text = ex.get("text", "")
                code = ex.get("code", "")
                test_list = ex.get("test_list", [])
                task_id = ex.get("task_id", n)
                # Format prompt as function request
                prompt = f"# {text}\n\ndef solution("
                yield {
                    "prompt": prompt,
                    "tests": "\n".join(test_list) if isinstance(test_list, list) else str(test_list),
                    "entry_point": "solution",
                    "canonical_solution": code,
                    "task_id": f"MBPP/{task_id}",
                }
            elif dataset_base in ("apps", "codeparrot/apps"):
                # APPS format: question, solutions (json list), input_output (json)
                question = ex.get("question", "")
                solutions = ex.get("solutions", "[]")
                input_output = ex.get("input_output", "{}")
                difficulty = ex.get("difficulty", "unknown")
                problem_id = ex.get("problem_id", n)
                try:
                    solutions_list = json.loads(solutions) if isinstance(solutions, str) else solutions
                    io_dict = json.loads(input_output) if isinstance(input_output, str) else input_output
                except json.JSONDecodeError:
                    solutions_list = []
                    io_dict = {}
                # Build test cases from input_output
                inputs = io_dict.get("inputs", [])
                outputs = io_dict.get("outputs", [])
                test_code = ""
                for inp, out in zip(inputs[:3], outputs[:3]):  # Limit to 3 test cases
                    test_code += f"assert solution({repr(inp)}) == {repr(out)}\n"
                yield {
                    "prompt": f"# {question}\n\ndef solution(",
                    "tests": test_code,
                    "entry_point": "solution",
                    "canonical_solution": solutions_list[0] if solutions_list else "",
                    "task_id": f"APPS/{problem_id}",
                    "difficulty": difficulty,
                }
            else:
                continue
            n += 1
            if n >= max_examples:
                break
        except Exception:
            continue


def _parse_source(src: str) -> Tuple[str, List[str]]:
    parts = src.split(":")
    scheme = parts[0]
    args = parts[1:]
    return scheme, args


# -----------------------------
# Judge Dataset Adapters
# -----------------------------

def iter_judge(name: str, source: str, max_examples: int) -> Iterator[Dict]:
    """Iterate over open-ended evaluation datasets for judge scoring."""
    from datasets import load_dataset

    scheme, args = _parse_source(source)
    assert scheme == "hf", f"Unsupported scheme for judge: {scheme}"

    if len(args) == 3:
        dataset, subset, split = args
        ds = load_dataset(dataset, subset, split=split, trust_remote_code=True)
    elif len(args) == 2:
        dataset, split = args
        ds = load_dataset(dataset, split=split, trust_remote_code=True)
    else:
        raise ValueError(f"Bad source spec: {source}")

    n = 0
    dataset_base = dataset.split("/")[-1]
    for ex in ds:
        try:
            if dataset_base in ("alpaca_eval", "tatsu-lab/alpaca_eval"):
                # AlpacaEval format
                instruction = ex.get("instruction", "")
                output = ex.get("output", "")
                generator = ex.get("generator", "")
                yield {
                    "prompt": instruction,
                    "reference": output,
                    "reference_generator": generator,
                }
            elif dataset_base in ("mt_bench", "lmsys/mt_bench_human_judgments"):
                # MT-Bench format
                question = ex.get("question", "")
                reference = ex.get("reference", "")
                category = ex.get("category", "")
                yield {
                    "prompt": question,
                    "reference": reference,
                    "category": category,
                }
            elif dataset_base in ("lima", "GAIR/lima"):
                # LIMA format
                conversations = ex.get("conversations", [])
                if conversations and len(conversations) >= 1:
                    prompt = conversations[0] if isinstance(conversations[0], str) else conversations[0].get("value", "")
                    reference = conversations[1] if len(conversations) > 1 else ""
                    if isinstance(reference, dict):
                        reference = reference.get("value", "")
                    yield {
                        "prompt": prompt,
                        "reference": reference,
                    }
            elif dataset_base in ("wildchat", "allenai/WildChat"):
                # WildChat format
                conversation = ex.get("conversation", [])
                if conversation and len(conversation) >= 1:
                    prompt = conversation[0].get("content", "") if isinstance(conversation[0], dict) else ""
                    yield {
                        "prompt": prompt,
                        "reference": "",  # No reference for WildChat
                    }
            else:
                # Generic format: look for common keys
                prompt = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ex.get("input", "")
                reference = ex.get("output") or ex.get("response") or ex.get("answer", "")
                if prompt:
                    yield {
                        "prompt": prompt,
                        "reference": reference,
                    }
                else:
                    continue
            n += 1
            if n >= max_examples:
                break
        except Exception:
            continue


# -----------------------------
# Code Generation & Execution
# -----------------------------

def generate_code_completion(
    model: torch.nn.Module,
    enc: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    stop_tokens: Optional[List[str]] = None,
) -> str:
    """Generate code completion from the model."""
    if stop_tokens is None:
        stop_tokens = ["\nclass ", "\ndef ", "\n#", "\nif __name__"]

    toks = torch.tensor(enc.encode(prompt), device="cuda", dtype=torch.long)[None, :]
    generated = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(toks)
            if temperature > 0:
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = int(torch.argmax(logits[0, -1]).item())

            generated.append(next_id)
            next_tok = torch.tensor([[next_id]], device="cuda", dtype=torch.long)
            toks = torch.cat([toks, next_tok], dim=1)

            # Check for stop tokens
            try:
                partial = enc.decode(generated)
                for stop in stop_tokens:
                    if stop in partial:
                        # Truncate at stop token
                        idx = partial.find(stop)
                        return partial[:idx]
            except Exception:
                pass

    try:
        return enc.decode(generated)
    except Exception:
        return ""


def execute_unittest(
    prompt: str,
    completion: str,
    tests: str,
    entry_point: str = "",
    timeout_s: float = 10.0,
) -> Tuple[bool, str]:
    """Execute generated code against unit tests.

    Returns:
        Tuple of (passed, error_message)
    """
    # Strip markdown code fences if present
    code = completion.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # Remove first fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)

    # Combine prompt + completion + tests
    full_code = prompt + code + "\n\n" + tests

    # Add check function call if entry_point is specified and tests use check()
    if entry_point and "check(" not in tests.lower():
        full_code += f"\n\ncheck({entry_point})\n"

    result = run_python(full_code, timeout_s=timeout_s)

    if result.ok:
        return True, ""
    else:
        error = result.stderr.strip()
        if "TIMEOUT" in error:
            return False, "Execution timed out"
        # Extract just the last line of error for brevity
        error_lines = error.split("\n")
        return False, error_lines[-1] if error_lines else "Unknown error"


# -----------------------------
# LLM-as-Judge Scoring
# -----------------------------

JUDGE_PROMPT_TEMPLATE = """You are an expert judge evaluating the quality of AI assistant responses.

## Task
Evaluate the following response to the given instruction. Score the response on a scale of 1-10 based on:
- Helpfulness: Does it address the user's request?
- Accuracy: Is the information correct?
- Clarity: Is it well-written and easy to understand?
- Completeness: Does it fully answer the question?

## Instruction
{instruction}

## Response to Evaluate
{response}

{reference_section}

## Evaluation
Provide a brief explanation (2-3 sentences) followed by your score.
Format your response as:
EXPLANATION: [your explanation]
SCORE: [1-10]
"""

JUDGE_PROMPT_WITH_REFERENCE = """## Reference Response (for comparison)
{reference}
"""


@dataclass
class JudgeResult:
    """Result of LLM judge evaluation."""
    prompt: str
    response: str
    reference: str
    score: float
    explanation: str
    raw_judge_output: str


def _extract_judge_score(judge_output: str) -> Tuple[float, str]:
    """Extract score and explanation from judge output."""
    score = 5.0  # Default middle score
    explanation = ""

    # Try to extract SCORE: X pattern
    score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", judge_output, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(1.0, min(10.0, score))  # Clamp to 1-10
        except ValueError:
            pass

    # Try to extract EXPLANATION: pattern
    expl_match = re.search(r"EXPLANATION:\s*(.+?)(?:SCORE:|$)", judge_output, re.IGNORECASE | re.DOTALL)
    if expl_match:
        explanation = expl_match.group(1).strip()

    return score, explanation


def run_judge_evaluation(
    model: torch.nn.Module,
    enc: Any,
    prompt: str,
    response: str,
    reference: str = "",
    max_new_tokens: int = 256,
) -> JudgeResult:
    """Run LLM-as-judge evaluation on a single response."""
    # Build judge prompt
    reference_section = ""
    if reference:
        reference_section = JUDGE_PROMPT_WITH_REFERENCE.format(reference=reference)

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=prompt,
        response=response,
        reference_section=reference_section,
    )

    # Generate judge output
    toks = torch.tensor(enc.encode(judge_prompt), device="cuda", dtype=torch.long)[None, :]
    generated = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(toks)
            next_id = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_id)
            next_tok = torch.tensor([[next_id]], device="cuda", dtype=torch.long)
            toks = torch.cat([toks, next_tok], dim=1)

            # Stop if we see clear end markers
            try:
                partial = enc.decode(generated)
                if "SCORE:" in partial and len(partial) > partial.find("SCORE:") + 10:
                    break
            except Exception:
                pass

    try:
        judge_output = enc.decode(generated)
    except Exception:
        judge_output = ""

    score, explanation = _extract_judge_score(judge_output)

    return JudgeResult(
        prompt=prompt,
        response=response,
        reference=reference,
        score=score,
        explanation=explanation,
        raw_judge_output=judge_output,
    )


def run_external_judge_evaluation(
    prompt: str,
    response: str,
    reference: str = "",
    api_provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> JudgeResult:
    """Run LLM-as-judge evaluation using external API (OpenAI/Anthropic).

    Args:
        prompt: Original instruction/question
        response: Model response to evaluate
        reference: Optional reference answer
        api_provider: "openai" or "anthropic"
        model_name: Model identifier (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")
        api_key: API key (defaults to env var)

    Returns:
        JudgeResult with score, explanation, and raw output
    """
    # Build judge prompt
    reference_section = ""
    if reference:
        reference_section = JUDGE_PROMPT_WITH_REFERENCE.format(reference=reference)

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=prompt,
        response=response,
        reference_section=reference_section,
    )

    judge_output = ""

    try:
        if api_provider == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            judge_output = completion.choices[0].message.content or ""

        elif api_provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=model_name,
                max_tokens=256,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            judge_output = message.content[0].text if message.content else ""

        else:
            raise ValueError(f"Unknown API provider: {api_provider}")

    except Exception as e:
        judge_output = f"ERROR: {str(e)}"

    score, explanation = _extract_judge_score(judge_output)

    return JudgeResult(
        prompt=prompt,
        response=response,
        reference=reference,
        score=score,
        explanation=explanation,
        raw_judge_output=judge_output,
    )


# -----------------------------
# Pass@k Evaluation (HumanEval-style)
# -----------------------------

def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """Compute pass@k metric.

    Args:
        n: Total number of samples per problem
        c: Number of correct samples
        k: k in pass@k

    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0
    # Use combinatorial formula: 1 - C(n-c, k) / C(n, k)
    # Computed numerically for stability
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


@dataclass
class PassAtKResult:
    """Results for pass@k evaluation."""
    task_id: str
    n_samples: int
    n_correct: int
    pass_at_1: float
    pass_at_10: float = 0.0
    pass_at_100: float = 0.0


def evaluate_pass_at_k(
    model: torch.nn.Module,
    enc: Any,
    items: List[Dict],
    n_samples: int = 200,
    temperature: float = 0.8,
    max_gen_tokens: int = 512,
    timeout_s: float = 10.0,
    ks: List[int] = None,
) -> Tuple[Dict[str, float], List[PassAtKResult]]:
    """Evaluate pass@k on code generation dataset.

    Args:
        model: Model to evaluate
        enc: Tokenizer
        items: List of evaluation items from iter_unittest
        n_samples: Number of samples per problem
        temperature: Sampling temperature (>0 for diversity)
        max_gen_tokens: Max tokens to generate
        timeout_s: Execution timeout per sample
        ks: List of k values to compute (default [1, 10, 100])

    Returns:
        Tuple of (aggregate_metrics, per_task_results)
    """
    if ks is None:
        ks = [1, 10, 100]

    per_task_results = []
    all_correct = []

    for it in items:
        task_id = it.get("task_id", "unknown")
        prompt = it["prompt"]
        tests = it.get("tests", "")
        entry_point = it.get("entry_point", "")

        # Generate n_samples completions
        correct_count = 0
        for _ in range(n_samples):
            with torch.inference_mode():
                completion = generate_code_completion(
                    model,
                    enc,
                    prompt,
                    max_new_tokens=max_gen_tokens,
                    temperature=temperature,
                )

            success, _ = execute_unittest(
                prompt=prompt,
                completion=completion,
                tests=tests,
                entry_point=entry_point,
                timeout_s=timeout_s,
            )
            if success:
                correct_count += 1

        # Compute pass@k for this task
        result = PassAtKResult(
            task_id=task_id,
            n_samples=n_samples,
            n_correct=correct_count,
            pass_at_1=compute_pass_at_k(n_samples, correct_count, 1),
            pass_at_10=compute_pass_at_k(n_samples, correct_count, 10) if 10 in ks else 0.0,
            pass_at_100=compute_pass_at_k(n_samples, correct_count, 100) if 100 in ks else 0.0,
        )
        per_task_results.append(result)
        all_correct.append(correct_count)

    # Aggregate metrics
    n_tasks = len(per_task_results)
    aggregate = {}
    for k in ks:
        if k <= n_samples:
            pass_at_k_values = [
                compute_pass_at_k(n_samples, c, k) for c in all_correct
            ]
            aggregate[f"pass@{k}"] = sum(pass_at_k_values) / max(1, n_tasks)

    return aggregate, per_task_results


# -----------------------------
# Scorers
# -----------------------------

def _centered(acc: float, baseline: float) -> float:
    # (acc - b) / (1 - b), clipped to [0,1] for stability in simulation
    if baseline >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (acc - baseline) / (1.0 - baseline)))


BASELINE_BY_TASK: Dict[str, float] = {
    # Random baselines (approx). For public parity, swap with DCLM v2 values later.
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "HellaSwag": 0.25,
    "MMLU": 0.25,
    "OpenBookQA": 0.25,
    "PIQA": 0.5,         # 2-choice
    "SIQA": 0.3333,      # 3-choice
    "WinoGrande": 0.5,   # binary
    "BoolQ": 0.5,
    "COPA": 0.5,         # 2-choice
    "StoryCloze": 0.5,   # 2-choice
    "SQuAD v1.1": 0.0,   # EM baseline treated as 0
}


@dataclass
class EvalResult:
    task: str
    raw_acc: float
    centered_acc: float
    n: int


def _normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().replace("\n", " ").split())


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("nmoe.eval.runner")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tasks", default="core")
    ap.add_argument("--max-examples", type=int, default=500)
    ap.add_argument("--max-time", type=int, default=300)
    ap.add_argument("--tasks-file", default="configs/eval/tasks.toml")
    # Simulation controls (for pre-training validation)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-acc-choices", type=float, default=0.6)
    ap.add_argument("--sim-acc-span", type=float, default=0.55)
    ap.add_argument("--sim-acc-unittest", type=float, default=0.5)
    ap.add_argument("--sim-acc-judge", type=float, default=0.6)
    ap.add_argument("--sim-seed", type=int, default=123)
    # External judge options
    ap.add_argument("--external-judge", action="store_true", help="Use external API for judge evaluation")
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--judge-model", default="gpt-4o-mini", help="Model for external judge")
    # Pass@k options
    ap.add_argument("--pass-at-k", action="store_true", help="Run pass@k evaluation for code tasks")
    ap.add_argument("--pass-k-samples", type=int, default=200, help="Number of samples for pass@k")
    ap.add_argument("--pass-k-temp", type=float, default=0.8, help="Temperature for pass@k sampling")
    args = ap.parse_args()

    snap = Path(args.snapshot)
    cfg_dict, _model_state = _load_snapshot(snap)

    # Metrics context (use same run id if present)
    run_id = cfg_dict.get("experiment_id", "eval")
    ctx = start_metrics(run_id=run_id, metrics_dir=cfg_dict.get("metrics_dir", "/data/metrics"))

    # Load task specs
    tasks_spec = _load_tasks_toml(Path(args.tasks_file))
    # Map name->spec for quick lookup
    spec_by_name = {t.get("name"): t for t in tasks_spec if isinstance(t, dict) and t.get("name")}

    # Build the real model (GPU) for scoring
    from nmoe.config import Config
    from nmoe.model import Transformer
    cfg = Config(**cfg_dict)
    model = Transformer(cfg).cuda().eval()
    # Load state if present
    try:
        miss = model.load_state_dict(_model_state, strict=False)
    except Exception as e:
        logging.getLogger(__name__).error("model.load_state_dict failed: %s. Evaluation may produce invalid results.", e)
        raise

    t0 = _now_ms()
    results: list[EvalResult] = []

    # Determine task list
    if args.tasks == "core":
        # Prefer tasks file ordering if available
        if tasks_spec:
            task_names = [t.get("name") for t in tasks_spec if isinstance(t, dict) and t.get("name")]
        else:
            task_names = []
    elif args.tasks.startswith("chat"):
        parts = args.tasks.split(":", 1)
        task_names = parts[1].split("|") if len(parts) == 2 else ["ARC-Easy","ARC-Challenge","MMLU","GSM8K","HumanEval"]
    else:
        task_names = [args.tasks]

    # Tokenizer (for prompt encoding); keep minimal deps
    import tiktoken
    enc = tiktoken.get_encoding(cfg.tokenizer)

    # Run scorers per task
    for name in task_names:
        spec = spec_by_name.get(name, {})
        src = spec.get("source")
        if not src:
            # Skip tasks that are not present in tasks file
            continue
        scorer = str(spec.get("scorer", "choices")).lower()
        n = max(1, min(int(args.max_examples), 500))  # cap for simulation speed
        if scorer == "choices":
            # Batched option log-likelihood; choose argmax
            correct = 0
            total = 0
            items = list(iter_choices(name, src, n))
            for it in items:
                prompt = it["prompt"]
                options = it["options"]
                gold = int(it["label"]) if it.get("label") is not None else 0
                # Score each option continuation
                scores = []
                with torch.inference_mode():
                    for opt in options:
                        text = prompt + " " + str(opt)
                        tokens = enc.encode(text)
                        toks = torch.tensor(tokens, device="cuda", dtype=torch.long)[None, :]
                        logits = model(toks)  # [1, T, V]
                        # Compute simple next-token LL over the option span only
                        # Split prompt and option by lengths
                        p_ids = enc.encode(prompt)
                        opt_ids = enc.encode(" " + str(opt))
                        if len(opt_ids) == 0:
                            scores.append(float("-inf"))
                            continue
                        start = len(p_ids) - 1
                        # Sum log-probs across the option tokens
                        ll = 0.0
                        for i, tok in enumerate(opt_ids):
                            pos = start + i
                            if pos < 0 or pos >= logits.size(1):
                                break
                            lp = torch.log_softmax(logits[0, pos, :], dim=-1)
                            ll += float(lp[tok].item())
                        scores.append(ll)
                pred = int(max(range(len(options)), key=lambda j: scores[j])) if scores else 0
                correct += int(pred == gold)
                total += 1
            acc = correct / max(1, total)
            baseline = BASELINE_BY_TASK.get(name, 0.0)
            results.append(EvalResult(task=name, raw_acc=acc, centered_acc=_centered(acc, baseline), n=total))

        elif scorer == "span":
            # Greedy decode and EM against provided answers
            em = 0
            total = 0
            items = list(iter_span(name, spec.get("source", ""), n))
            for it in items:
                prompt = it["prompt"]; answers = it.get("answers", [])
                # Greedy one-pass decode for a small budget (no KV cache)
                toks = torch.tensor(enc.encode(prompt), device="cuda", dtype=torch.long)[None, :]
                with torch.inference_mode():
                    logits = model(toks)
                    next_id = int(torch.argmax(logits[0, -1]).item())
                out = enc.decode([next_id])
                if _normalize_text(out) in {_normalize_text(a) for a in answers}:
                    em += 1
                total += 1
            acc = em / max(1, total)
            baseline = BASELINE_BY_TASK.get(name, 0.0)
            results.append(EvalResult(task=name, raw_acc=acc, centered_acc=_centered(acc, baseline), n=total))

        elif scorer == "unittest":
            # Code generation with sandbox execution
            timeout_s = float(spec.get("timeout", 10.0))
            max_gen_tokens = int(spec.get("max_gen_tokens", 512))
            items = list(iter_unittest(name, spec.get("source", ""), n))

            if args.pass_at_k:
                # Run pass@k evaluation
                aggregate, per_task = evaluate_pass_at_k(
                    model=model,
                    enc=enc,
                    items=items,
                    n_samples=args.pass_k_samples,
                    temperature=args.pass_k_temp,
                    max_gen_tokens=max_gen_tokens,
                    timeout_s=timeout_s,
                    ks=[1, 10, 100],
                )
                # Use pass@1 as the primary metric
                acc = aggregate.get("pass@1", 0.0)
                # Store pass@k details in summary
                results.append(EvalResult(task=name, raw_acc=acc, centered_acc=acc, n=len(items)))
                # Add pass@k breakdown to summary later
            else:
                # Standard greedy evaluation (pass@1 with temp=0)
                passed = 0
                total = 0
                for it in items:
                    prompt = it["prompt"]
                    tests = it.get("tests", "")
                    entry_point = it.get("entry_point", "")
                    task_id = it.get("task_id", f"{name}/{total}")

                    # Generate code completion
                    with torch.inference_mode():
                        completion = generate_code_completion(
                            model,
                            enc,
                            prompt,
                            max_new_tokens=max_gen_tokens,
                            temperature=0.0,
                        )

                    # Execute and check
                    success, error_msg = execute_unittest(
                        prompt=prompt,
                        completion=completion,
                        tests=tests,
                        entry_point=entry_point,
                        timeout_s=timeout_s,
                    )
                    if success:
                        passed += 1
                    total += 1

                acc = passed / max(1, total)
                # Unittest baseline is 0 (no code passes without writing)
                results.append(EvalResult(task=name, raw_acc=acc, centered_acc=acc, n=total))

        elif scorer == "judge":
            # LLM-as-judge evaluation for open-ended responses
            total_score = 0.0
            total = 0
            max_gen_tokens = int(spec.get("max_gen_tokens", 256))
            tau_keep = float(spec.get("tau_keep", 0.5))  # Threshold for "good" response
            items = list(iter_judge(name, spec.get("source", ""), n))

            for it in items:
                prompt = it["prompt"]
                reference = it.get("reference", "")

                # Generate response from the model
                with torch.inference_mode():
                    response_completion = generate_code_completion(
                        model,
                        enc,
                        prompt,
                        max_new_tokens=max_gen_tokens,
                        temperature=0.0,
                        stop_tokens=["\n\n", "Human:", "User:"],
                    )

                # Run judge evaluation (external API or local model)
                if args.external_judge:
                    judge_result = run_external_judge_evaluation(
                        prompt=prompt,
                        response=response_completion,
                        reference=reference,
                        api_provider=args.judge_provider,
                        model_name=args.judge_model,
                    )
                else:
                    judge_result = run_judge_evaluation(
                        model=model,
                        enc=enc,
                        prompt=prompt,
                        response=response_completion,
                        reference=reference,
                        max_new_tokens=256,
                    )

                # Normalize score to 0-1 range (from 1-10)
                normalized_score = (judge_result.score - 1.0) / 9.0
                total_score += normalized_score
                total += 1

            acc = total_score / max(1, total)
            # Use tau_keep as baseline for centering
            centered = _centered(acc, tau_keep)
            results.append(EvalResult(task=name, raw_acc=acc, centered_acc=centered, n=total))
        else:
            results.append(EvalResult(task=name, raw_acc=0.0, centered_acc=0.0, n=0))

    # Aggregate CORE (average centered accuracy across tasks present)
    core = sum(r.centered_acc for r in results) / max(1, len(results))

    # Write summary JSON next to snapshot
    summary = {
        "suite": "core" if args.tasks == "core" else "custom",
        "CORE": core,
        "tasks": [
            {"task": r.task, "raw_acc": r.raw_acc, "centered_acc": r.centered_acc, "n": r.n}
            for r in results
        ],
        "elapsed_ms": _now_ms() - t0,
    }
    with (snap.parent / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Persist to metrics DB
    try:
        if ctx is not None and ctx.writer is not None:
            step = int(cfg_dict.get("global_step", 0))
            # Per-task
            items = []
            for r in results:
                items.append((f"eval/task/{r.task}/acc", float(r.raw_acc)))
                items.append((f"eval/task/{r.task}/centered", float(r.centered_acc)))
                items.append((f"eval/task/{r.task}/n", float(r.n)))
            # CORE aggregate
            items.append(("eval/CORE", float(core)))
            ctx.writer.insert_many(step, items)
    except Exception:
        logging.getLogger(__name__).debug("Metrics DB write failed", exc_info=True)

    stop_metrics(ctx)


if __name__ == "__main__":
    main()
