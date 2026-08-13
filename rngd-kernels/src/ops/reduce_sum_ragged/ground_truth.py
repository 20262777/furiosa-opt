#!/usr/bin/env python3
"""Ground truth for rung 06b `reduce_sum_ragged`: out[s] = sum_r x[s, r].

`RP = 1000` slots per row, of which `R_REAL = 999` carry data. The kernel reduces over
all 1000, so the tail **must** hold the reduce operation's identity element (0.0 for
Add) — that is the caller contract, and it is what this generator supplies.

It also emits `x_poisoned`, the same tensor with a loud sentinel in the tail instead of
zero, so the test can prove the tail is genuinely being summed (i.e. that a violated
contract really does corrupt the result, rather than the kernel accidentally ignoring
those slots).

Usage:
    python src/ops/reduce_sum_ragged/ground_truth.py
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "reduce_sum_ragged" / "reduce_sum_ragged.safetensors"

S = 256  # rows
R_REAL = 999  # slots carrying data
RP = 1000  # padded reduction length the kernel reduces over
SENTINEL = 1.0e4

SEED = 42


def main() -> None:
    torch.manual_seed(SEED)
    real = torch.randn(S, R_REAL).bfloat16()

    # Contract-satisfying input: identity element (0.0) in the tail.
    zeros = torch.zeros(S, RP - R_REAL).bfloat16()
    x = torch.cat([real, zeros], dim=1).contiguous()

    # Contract-violating input, to show the tail is really summed.
    pad = torch.full((S, RP - R_REAL), SENTINEL).bfloat16()
    x_poisoned = torch.cat([real, pad], dim=1).contiguous()

    assert x.shape == (S, RP)

    expected = real.float().sum(dim=1).bfloat16()
    expected_poisoned = x_poisoned.float().sum(dim=1).bfloat16()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "x": x,
            "expected": expected,
            "x_poisoned": x_poisoned,
            "expected_poisoned": expected_poisoned,
        },
        str(OUT),
    )
    print(f"wrote {OUT}  (S={S}, R_REAL={R_REAL}, RP={RP}, seed={SEED})")
    print(f"  expected[0]           = {expected[0].item():.4f}")
    print(f"  expected_poisoned[0]  = {expected_poisoned[0].item():.4f}")


if __name__ == "__main__":
    main()
