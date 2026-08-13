# 08 — `softmax`

## Math

```
y[s, r] = exp(x[s, r] - max_r x[s, r]) / sum_r exp(x[s, r] - max_r x[s, r])
```

`S = 256` rows, `RP = 1000` slots per row, `R_REAL = 999` real. Numerically stable form
(max-subtract before `exp`). `bf16` in, `f32` throughout, one `bf16` rounding at commit.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `x` | `m![S, RP]` | 256 x 1000 `bf16` (512 KB) |
| `out` | `m![S, RP]` | 256 x 1000 `bf16` (512 KB) |

## Mapping plan

| dim | mapping | size |
|---|---|---|
| `Chip` | `m![1]` | 1 |
| `Cluster` | `m![1 # 2]` | 2 |
| `Slice` | `m![S]` | 256 — one row per slice, never enters a reducer |
| `Element` | `m![RP]` | 1000 slots, no `#` padding |

```
RP = 1000 :  RP % 4 (Packet, 4)  x  RP / 4 (Time, 250)
```

## One sentinel, both reductions

The tail holds `NEG_SENTINEL = -3.3895314e38`, and that single value is the identity for
**both** reductions in this kernel:

| reduction | why the sentinel is the identity |
|---|---|
| `Max` | more negative than any real score, so it never wins |
| `Add` after `exp` | `exp(sentinel - max)` underflows to exactly `0` |

So: no Valid Count Generator, no `fetch_mask`, no `Filter`. And it is the same write a
causal or padding mask already performs — upstream's
`transformer/attention/softmax.rs` uses this exact constant. **Padding and masking are
one mechanism.** Rung 07b established the `Add` half; this rung uses both.

Padded output positions come out as exactly `0.0`, which the test asserts.

## Five passes

| pass | ctx | computes | ALUs |
|---|---|---|---|
| 1 | main | `max_r x` -> DM | reduce tree |
| 2 | sub | DM -> VRF | — |
| 3 | main | `1 / sum_r exp(x - max)` -> DM | `FpFma`, `FpExp`, reduce tree, `FpDiv` |
| 4 | sub | DM -> VRF | — |
| 5 | main | `exp(x - max) * inv_sum` -> DM | `FpFma`, `FpExp`, `FpMul0` |

Passes 3 and 5 each pack **three** Float-cluster operations into one invocation, legal
only because `SubF`, `Exp` and `MulF(Mul0)` land on `FpFma`, `FpExp` and `FpMul0`. The
chain's fixed stage order does the rest in pass 3: `Float` (6) → `IntraSliceReduce` (7) →
`FpDiv` (8), so subtract, exp, sum and reciprocal all fit one trip down the pipeline.

Pass 5 multiplies by the reciprocal rather than dividing, keeping the op on `FpMul0` — a
`DivF` there would contend with nothing, but `FpDiv` is a single dedicated unit and the
multiply is free.

The two sub-context passes overlap main-context work.

## ⚠ Why the reductions round-trip through DM

`.to_vrf()` placed **directly after** `vector_intra_slice_reduce` passes emulation and
fails vISA lowering:

```
visa: while lowering StoVrf
  StoVrf: kernel-declared fn_output_shape does not match vrf_shape.
    fn_output: InSlice: [RP / 4 = 250, [RP % 4 + 4] = 8]     <- pre-reduce stream shape
    vrf:       InSlice: [[] + 7] = 8                          <- the reduced m![1 # 8]
```

The store uses the stream's shape *before* the reduce, not after it. So a reduced result
cannot go straight to VRF; it must be committed to DM and lifted by a separate pass.
`rmsnorm` has the same structure for the same reason (its variance goes to DM, then pass
3 reads it back).

This is the third lowering limitation this ladder has found that emulation accepts —
after rung 06b's `reduce axis should not have padding` and rung 07's
`vISA scalar operand must be known at compile time`. **Pattern: a rung is not validated
until `schedule.json` exists.**

## Pipeline trace (pass 5, the normalise)

| stage | `Time` | `Packet` | way | note |
|---|---|---|---|---|
| `fetch` | 250 | 4 `bf16` | — | 8 B |
| `fetch_cast::<f32>` | 250 | 4 `f32` | — | 16 B |
| `collect` | 250 | `RP % 4 # 8` | — | padded to the 32 B flit |
| `narrow_trim` | 250 | 4 `f32` | 8 → 4 | |
| `fp_binary(SubF, max_vrf)` | 250 | 4 | 4 | `FpFma`; `x - max` |
| `fp_unary(Exp)` | 250 | 4 | 4 | `FpExp`; **tail becomes 0.0** |
| `fp_binary(MulF(Mul0), inv_sum_vrf)` | 250 | 4 | 4 | `FpMul0` |
| `widen_pad` | 250 | `RP % 4 # 8` | 4 → 8 | |
| `cast` / `commit_trim` | 250 | `RP % 4` | | 8 B write unit |

## Test design

The value comparison alone is weak here: outputs are ~`1/999 ≈ 1e-3`, so the ladder's
default `1e-2` absolute floor would pass *anything*. Tightened to `1e-5`, and backed by
two reference-independent guards:

**Guard 1 — every row sums to 1.** The defining property of softmax. A broken max, exp,
reciprocal or reduction all break it, and it does not depend on the torch reference at all.

**Guard 2 — padded slots are exactly `0.0`.** If the sentinel ever failed to underflow
through `exp`, these would be non-zero and guard 1 would fail too.

Both guards are skipped when `got` is empty, because under `--backend typecheck` the
tensors are phantom and there is nothing to index. (Getting this wrong was a real failure:
the guards panicked under typecheck while the emulation run passed.)

**Result: `max abs diff: 3.05e-5`** over 256,000 elements — well under one `bf16` ULP at
this magnitude — with both guards holding.

## Constraints checked

- `Slice::SIZE == 256`, `Cluster::SIZE == 2`
- Reduce axis `RP` carries no padding → lowers
- ALU disjointness within every pass (table above)
- VRF: two per-row scalars, 32 B each, against 8 KB per slice
- `commit_trim` 32 B (passes 1, 3) and 8 B (pass 5), both legal
- **Lowers to vISA** — `schedule.json` produced

## Cost, and what FlashAttention would change

`x` is read from DM three times (passes 1, 3, 5). Nothing of row width is ever written
back — only two per-row scalars — so this is materially better than upstream's masked
softmax, which materialises the full `S x T` score matrix in DM and reads it back three
times.

It is still not FlashAttention: that fuses the whole thing into the QK^T contraction's
`Time` loop with a *running* max and sum, so the scores never leave the pipeline at all.
Rungs 09–11 are where that becomes possible.

## Follow-up

- **Masked softmax** — the causal mask writes `NEG_SENTINEL` into disallowed positions,
  which this kernel already handles with zero extra machinery. Wiring the mask in is the
  next increment.
- **GQA head grouping** — softmax runs independently per group; upstream loops `G = 7`
  groups over separate workspaces.
- `f8` variants (WBS 20) need a per-dtype sentinel: `f8e4m3` has a far narrower exponent
  range, so `-3.39e38` is not representable.
