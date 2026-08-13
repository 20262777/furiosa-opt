#!/usr/bin/env python3
"""Ground truth for rung 08 `softmax`.

    y[s, r] = exp(x[s, r] - max) / sum_r exp(x[s, r] - max)

RP = 1000 slots per row, R_REAL = 999 real, tail = NEG_SENTINEL. The sentinel is the
identity for both reductions: it never wins the Max, and exp(sentinel - max) underflows
to 0 so it contributes nothing to the sum. Padded output positions are therefore exactly
0.0, and the reference encodes that.

The kernel multiplies by the reciprocal (1/sum) rather than dividing, because FpDiv is a
single dedicated unit; the reference does the same so the rounding matches.

Usage:
    python src/ops/softmax/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "softmax" / "softmax.safetensors"

S = 256
R_REAL = 999
RP = 1000
NEG_SENTINEL = -3.3895314e38  # must match ops::softmax::NEG_SENTINEL

SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    real = torch.randn(S, R_REAL).bfloat16()

    pad = torch.full((S, RP - R_REAL), NEG_SENTINEL).bfloat16()
    x = torch.cat([real, pad], dim=1).contiguous()
    assert x.shape == (S, RP)

    xf = real.float()
    mx = xf.max(dim=1, keepdim=True).values
    e = torch.exp(xf - mx)
    inv_sum = 1.0 / e.sum(dim=1, keepdim=True)

    # Multiply by reciprocal, matching the kernel.
    y_real = e * inv_sum

    # Padded positions: exp(sentinel - max) underflows to 0, so the output is 0 there.
    assert torch.exp(pad.float() - mx).max().item() == 0.0, "sentinel does not underflow"
    y_pad = torch.zeros(S, RP - R_REAL)

    expected = torch.cat([y_real, y_pad], dim=1).bfloat16()

    row_sums = expected.float().sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(S), atol=2e-2), row_sums[:4]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": x, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (S={S}, R_REAL={R_REAL}, RP={RP}, seed={SEED})")
    print(f"  expected[0, 0]  = {expected[0, 0].item():.6e}")
    print(f"  expected[0, -1] = {expected[0, -1].item():.6e}  (padded slot)")
    print(f"  row sum[0]      = {row_sums[0].item():.6f}")


if __name__ == "__main__":
    main()
