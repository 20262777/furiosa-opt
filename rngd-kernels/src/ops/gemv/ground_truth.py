#!/usr/bin/env python3
"""Ground truth for rung 04 `gemv`: y = A x.

Usage:
    python src/ops/gemv/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "gemv" / "gemv.safetensors"

I, J = 256, 2048
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    # Row-major [I, J] matches the kernel's HBM element mapping m![I, J].
    matrix = torch.randn(I, J).bfloat16()
    vector = torch.randn(J).bfloat16()

    # f32 accumulation over J, single bf16 rounding at the end.
    expected = (matrix.float() @ vector.float()).bfloat16()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"matrix": matrix, "vector": vector, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (I={I}, J={J}, seed={SEED})")


if __name__ == "__main__":
    main()
