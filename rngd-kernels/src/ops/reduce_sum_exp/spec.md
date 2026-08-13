# 07b — `reduce_sum_exp`

## Math

```
out[s] = sum_r exp(x[s, r])        s in 0..256, r in 0..1000
```

Softmax's denominator. `RP = 1000` slots per row, `R_REAL = 999` real; the tail holds
`NEG_SENTINEL = -3.3895314e38`.

## Why this rung exists

Rung 06b established two things: a **padded reduce axis does not lower to vISA**, and the
shippable alternative is to fill the tail with the reduce operation's **identity element**.
It also recorded a conclusion that turns out to be **wrong**:

> identities exist for `Add` (`0`), `Max` (`-inf`), `Min` (`+inf`), but not for
> `sum(exp(x))` — no `p` satisfies `exp(p) = 0`. **Softmax still needs an answer.**

That is true in exact real arithmetic and false in floating point. **`exp` underflows.**
Any `p` below about `-88` gives `0` in `f32`; `-3.39e38` gives exactly `0`. So the identity
element for the *composed* operation `x -> exp(x) -> sum` is simply a large negative number.

The identity is a property of the whole reduce chain, not of the reduce op alone.

And this is not a trick invented here — it is what masked softmax already does.
`furiosa-opt-examples/src/transformer/attention/softmax.rs` fills masked positions with
exactly `-3.3895314e38` for the same reason. **Padding and masking are the same
mechanism**, so a kernel that already masks gets ragged-length handling for free.

`bf16` shares `f32`'s 8-bit exponent, so the sentinel is representable without widening.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `x` | `m![S, RP]` | 256 x 1000 `bf16` (512 KB) |
| `out` | `m![S]` | 256 `bf16` |

## Mapping plan

| dim | mapping | size |
|---|---|---|
| `Chip` | `m![1]` | 1 |
| `Cluster` | `m![1 # 2]` | 2 |
| `Slice` | `m![S]` | 256 — one row per slice, never enters a reducer |
| `Element` | `m![RP]` | 1000 slots, **no `#` padding** (a padded reduce axis does not lower) |

```
RP = 1000 :  RP % 4 (Packet, 4)  x  RP / 4 (Time, 250)
```

## Pipeline trace

| stage | `Time` | `Packet` | way | note |
|---|---|---|---|---|
| `fetch` | 250 | 4 `bf16` | — | 8 B |
| `fetch_cast::<f32>` | 250 | 4 `f32` | — | 16 B |
| `collect` | 250 | `RP % 4 # 8` | — | padded to the 32 B flit |
| `narrow_trim` | 250 | 4 `f32` | 8 → 4 | upper 4 were collect padding |
| `fp_unary(Exp)` | 250 | 4 | 4 | `FpExp`; **sentinel underflows to 0 here** |
| `intra_slice_reduce(Add)` | 1 | `1 # 4` | 4 | tail contributes nothing |
| `widen_pad` | 1 | `1 # 8` | 4 → 8 | |
| `cast` / `commit_trim` | 1 | `1 # 4` | | 8 B write unit |

## ALU budget

`FpExp` for the exponential, plus the reducer's dedicated accumulator tree. `FpFpu`,
`FpFma`, `FpMul0`, `FpMul1` all remain free — which is what lets the full softmax fuse
the max-subtract and the division into neighbouring passes.

## Test design

Inputs are `N(0, 1)` and **not** max-shifted, so `exp` stays well inside range and the row
sums land near `999 * e^0.5 ~ 1650`. The generator asserts `exp(sentinel) == 0.0` before
writing, so a broken assumption fails at generation rather than silently.

The test adds a **finiteness guard** on every row before the value comparison. If the
sentinel ever failed to underflow, each row would be `+inf` rather than ~1650 — a
decisive, cheap check on the padding contract that does not depend on tolerances.

**Result: `max abs diff: 0e0`** across all 256 rows, and it **lowers to vISA**.

## Constraints checked

- `Slice::SIZE == 256`, `Cluster::SIZE == 2`
- Reduce axis `RP` carries no padding → lowers (contrast rung 06b's first design)
- `intra_slice_reduce`: only reduce dim is outermost → `InnerTime = m![1]`, 1 slot
- Per-row DM footprint 2000 B, 8-byte aligned
- `commit_trim` valid size = 8 B

## What this unblocks

Rung 08 `softmax` no longer needs the VCG, `fetch_mask`, or a `Filter` stage to handle a
ragged key length. The recipe is:

1. Pad the reduce axis to a hardware-friendly width.
2. Fill the tail with a large negative value — the same write the causal/padding mask
   already performs.
3. Reduce over the **unpadded** padded-width axis.

`Max` needs the same treatment and gets it for free: the sentinel is also the identity for
`Max` (it is more negative than any real score), so softmax's row-max pass is covered by
the same fill.

## Follow-up

- Confirm the same underflow argument holds for `f8e4m3` / `f8e5m2` inputs in WBS 20 —
  their exponent range is much narrower, so the sentinel must be chosen per dtype.
- `RP` too long for one slice: the reduce axis then lands partly in `Slice` and needs an
  `inter_slice_reduce`, which does not change the padding story.
