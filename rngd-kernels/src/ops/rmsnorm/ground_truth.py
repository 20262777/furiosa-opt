#!/usr/bin/env python3
"""Ground truth for rung 07 `rmsnorm`.

    y[s, h] = x[s, h] * rsqrt(mean_h(x[s, h]^2) + eps) * weight[h]

The kernel keeps every intermediate in f32 — the variance goes to DM as f32 and the
reciprocal RMS sits in the VRF as f32 — and rounds to bf16 exactly once, at the final
commit. The reference does the same, so `.float()` everywhere until the last `.bfloat16()`.

Usage:
    python src/ops/rmsnorm/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "rmsnorm" / "rmsnorm.safetensors"

S, H = 256, 1024
EPS = 1.0e-6  # must match ops::rmsnorm::EPS
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(S, H).bfloat16()
    weight = torch.randn(H).bfloat16()

    xf = x.float()
    # Same order as the kernel: sum of squares -> divide by H -> add eps -> rsqrt.
    var = (xf * xf).sum(dim=1) / H
    inv_rms = 1.0 / torch.sqrt(var + EPS)

    expected = (xf * inv_rms.unsqueeze(1) * weight.float()).bfloat16()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": x, "weight": weight, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (S={S}, H={H}, eps={EPS}, seed={SEED})")
    print(f"  inv_rms[0] = {inv_rms[0].item():.6f}   expected[0,0] = {expected[0, 0].item():.6f}")


if __name__ == "__main__":
    main()
