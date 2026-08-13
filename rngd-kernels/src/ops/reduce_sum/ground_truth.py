#!/usr/bin/env python3
"""Ground truth for rung 06 `reduce_sum`: out = sum_i x[i].

The kernel widens bf16 -> f32, accumulates in f32 across three stages (packet tree,
time accumulator, inter-slice ring), and rounds once to bf16. torch sums pairwise, so
the summation *order* differs; only the order-independent f32 total is comparable.

Usage:
    python src/ops/reduce_sum/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "reduce_sum" / "reduce_sum.safetensors"

A = 8192
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(A).bfloat16()

    expected = x.float().sum().bfloat16().reshape(1)

    # float64 reference, to show how much of any mismatch is bf16 rounding vs. order.
    exact = x.double().sum().item()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": x, "expected": expected}, str(OUT))
    print(
        f"wrote {OUT}  (A={A}, seed={SEED}, "
        f"expected={expected.item():.6f}, f64={exact:.6f})"
    )


if __name__ == "__main__":
    main()
