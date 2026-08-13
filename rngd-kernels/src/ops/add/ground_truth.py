#!/usr/bin/env python3
"""Ground truth for rung 01 `add`: out[i] = lhs[i] + rhs[i].

The kernel widens bf16 -> f32, adds, then casts back to bf16, so the reference does
the same. Writes lhs / rhs / expected into data/add/add.safetensors.

Usage:
    python src/ops/add/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

# src/ops/add/ground_truth.py -> parents[3] == the package root
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "add" / "add.safetensors"

A = 2048
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    lhs = torch.randn(A).bfloat16()
    rhs = torch.randn(A).bfloat16()

    # Match the kernel's arithmetic exactly: widen, add, narrow.
    expected = (lhs.float() + rhs.float()).bfloat16()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lhs": lhs, "rhs": rhs, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (A={A}, seed={SEED})")


if __name__ == "__main__":
    main()
