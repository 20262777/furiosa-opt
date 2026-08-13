#!/usr/bin/env python3
"""Ground truth for rung 03 `dot`: out = sum_i lhs[i] * rhs[i].

The kernel multiplies bf16 operands, widens the products to f32, accumulates in f32,
and rounds once to bf16 at the end. The reference does the same.

Usage:
    python src/ops/dot/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "dot" / "dot.safetensors"

A = 2048
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    lhs = torch.randn(A).bfloat16()
    rhs = torch.randn(A).bfloat16()

    # f32 accumulation, single bf16 rounding at the end. Shape [1] to match m![1].
    expected = (lhs.float() * rhs.float()).sum().bfloat16().reshape(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lhs": lhs, "rhs": rhs, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (A={A}, seed={SEED}, expected={expected.item():.6f})")


if __name__ == "__main__":
    main()
