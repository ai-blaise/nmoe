#!/usr/bin/env python3
"""Static guards against hot-path sync/fallback regressions."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _line_no(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def main() -> int:
    failures: list[str] = []

    model_path = ROOT / "nmoe" / "model.py"
    model_text = _read(model_path)
    if re.search(r"_last_load_cv\s*=.*\.item\(", model_text):
        failures.append(
            f"{model_path}: found `.item()` in MoE load-CV assignment (reintroduces forward sync)."
        )

    moe_path = ROOT / "nmoe" / "moe.py"
    moe_text = _read(moe_path)
    for m in re.finditer(r"while\s+not\s+[A-Za-z0-9_\.]+\s*\.query\(\)\s*:", moe_text):
        line = _line_no(moe_text, m.start())
        failures.append(f"{moe_path}:{line}: found Python busy-spin wait on CUDA event query().")

    rdep_path = ROOT / "nmoe" / "csrc" / "rdep.cu"
    rdep_text = _read(rdep_path)
    ceilings = [
        (
            r"cudaStreamSynchronize\(stream\);",
            9,
            "host stream synchronizations in rdep.cu",
        ),
        (
            r"cudaMemcpyAsync\(&dropped,",
            2,
            "D2H dropped-counter copies",
        ),
        (
            r"cudaMemcpy\(&M_recv,",
            1,
            "blocking D2H M_recv copies",
        ),
        (
            r"cudaMemcpyAsync\(&M_recv,",
            0,
            "async D2H M_recv copies",
        ),
    ]
    for pattern, max_allowed, label in ceilings:
        found = _count(pattern, rdep_text)
        if found > max_allowed:
            failures.append(
                f"{rdep_path}: {label} increased ({found} > {max_allowed})."
            )

    if failures:
        print("Hot-path regression guard FAILED:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("Hot-path regression guard checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
