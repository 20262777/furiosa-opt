#!/usr/bin/env python3
"""Ground truth for rung 07b `reduce_sum_exp`: out[s] = sum_r exp(x[s, r]).

RP = 1000 slots per row, R_REAL = 999 real. The tail holds NEG_SENTINEL, whose exp
underflows to exactly 0 — the additive identity *after* the exp. This is the test of
whether the identity-element contract extends to sum(exp(x)).

Inputs are drawn from N(0, 1) and left un-shifted (no max-subtraction), so exp stays
well inside f32 range: sum ~ 999 * e^0.5 ~ 1650.

Usage:
    python src/ops/reduce_sum_exp/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "reduce_sum_exp" / "reduce_sum_exp.safetensors"

S = 256
R_REAL = 999
RP = 1000
NEG_SENTINEL = -3.3895314e38  # must match ops::reduce_sum_exp::NEG_SENTINEL

SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    real = torch.randn(S, R_REAL).bfloat16()

    pad = torch.full((S, RP - R_REAL), NEG_SENTINEL).bfloat16()
    x = torch.cat([real, pad], dim=1).contiguous()
    assert x.shape == (S, RP)

    # Sanity: the sentinel really does underflow to zero under exp.
    assert torch.exp(pad.float()).max().item() == 0.0, "sentinel does not underflow"

    expected = real.float().exp().sum(dim=1).bfloat16()
    assert torch.isfinite(expected).all(), "expected overflowed"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": x, "expected": expected}, str(OUT))
    print(f"wrote {OUT}  (S={S}, R_REAL={R_REAL}, RP={RP}, seed={SEED})")
    print(f"  expected[0] = {expected[0].item():.4f}   exp(sentinel) = 0.0 (verified)")


if __name__ == "__main__":
    main()
