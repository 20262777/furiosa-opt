#!/usr/bin/env python3
"""Ground truth for rung 05 `gemm`: C = A B^T  (einsum IK, JK -> IJ).

`b` is stored transposed (shape [J, K]) to match the kernel's HBM mapping m![J, K],
so the reference multiplies `a @ b.T`.

Usage:
    python src/ops/gemm/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "gemm" / "gemm.safetensors"

I, J, K = 512, 512, 64
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    a = torch.randn(I, K).bfloat16()  # m![I, K]
    b = torch.randn(J, K).bfloat16()  # m![J, K]  (already transposed)

    # f32 accumulation over K, single bf16 rounding at the end.
    expected = (a.float() @ b.float().T).bfloat16()  # [I, J]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"a": a, "b": b, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (I={I}, J={J}, K={K}, seed={SEED})")


if __name__ == "__main__":
    main()
