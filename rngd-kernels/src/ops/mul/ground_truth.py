#!/usr/bin/env python3
"""Ground truth for rung 02 `mul`: out[i] = lhs[i] * rhs[i].

The kernel widens both operands to f32, multiplies, and rounds once to bf16.

Usage:
    python src/ops/mul/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "mul" / "mul.safetensors"

A = 2048
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    lhs = torch.randn(A).bfloat16()
    rhs = torch.randn(A).bfloat16()

    expected = (lhs.float() * rhs.float()).bfloat16()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lhs": lhs, "rhs": rhs, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (A={A}, seed={SEED})")


if __name__ == "__main__":
    main()
