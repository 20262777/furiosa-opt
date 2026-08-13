# 07 — `rmsnorm`

## Math

```
y[s, h] = x[s, h] * rsqrt( mean_h(x[s, h]^2) + eps ) * weight[h]
```

`S = 256` rows, `H = 1024` hidden features, `eps = 1e-6`. `bf16` in, **everything
intermediate in `f32`**, one `bf16` rounding at the final commit.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `x` | `m![S, H]` | 256 x 1024 `bf16` (512 KB) |
| `weight` | `m![H]` | 1024 `bf16` (2 KB) |
| `out` | `m![S, H]` | 256 x 1024 `bf16` (512 KB) |

## Mapping plan

| dim | mapping | size |
|---|---|---|
| `Chip` | `m![1]` | 1 |
| `Cluster` | `m![1 # 2]` | 2 |
| `Slice` | `m![S]` | 256 — one row per slice, never enters a reducer |
| `Element` | `m![H]` | 1024 `bf16` = 2 KB per row |

`H = 1024` factors cleanly (`1024 / 4 = 256`), so no padded-axis handling is needed here —
that problem is rung 06b's, and it remains unsolved for `softmax`.

`weight` has no `S` axis, so its `to_dm` is a **broadcast** (stride-0 entry): every slice
gets a private copy. Same rule as `gemv`'s vector.

## The point of this rung: one op, four passes

Each ALU may fire **at most once per Tensor Unit invocation**, and the intra-slice chain's
stages run in a fixed order. That is what forces the split — not the arithmetic.

| pass | ctx | computes | ALUs used |
|---|---|---|---|
| 1 | sub | `weight` -> VRF | — |
| 2 | main | `sum(x*x) / H + eps` -> DM (`f32`) | `FpMul0`, reduce tree, `FpDiv`, `ClipAdd` |
| 3 | main | `1 / sqrt(var)` -> VRF | `FpFpu`, `FpDiv` |
| 4 | main | `x * inv_rms * weight` -> DM | `FpMul0`, `FpMul1` |

Passes 2 and 3 cannot merge: both need the dedicated `FpDiv` divider. Pass 4 fits **two**
multiplies in one invocation only because the Float cluster exposes `FpMul0` *and*
`FpMul1` as separate ALUs — routing both through `Mul0` panics with "already in use".

Pass 1 runs on the **sub** context, so it overlaps pass 2, which does not read the VRF.

## `vector_stash` — there is no square op

```rust
.vector_stash()
.vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)   // x * x
```

The chain is a single-value pipeline: each stage overwrites the running value. `Stash` is
the one side-register that lets an earlier value survive, and here the operand *is* the
stream, one stage delayed. Read-once and write-once, which is exactly enough.

## Pipeline trace (pass 4, the scale)

| stage | `Time` | `Packet` | way | note |
|---|---|---|---|---|
| `fetch` | 256 | 4 `bf16` | — | 8 B |
| `fetch_cast::<f32>` | 256 | 4 `f32` | — | 16 B |
| `collect` | 256 | `H % 4 # 8` | — | padded to the 32 B flit |
| `narrow_trim` | 256 | 4 `f32` | 8 → 4 | upper 4 were collect padding |
| `fp_binary(MulF(Mul0), inv_rms_vrf)` | 256 | 4 | 4 | per-row scalar |
| `fp_binary(MulF(Mul1), weight_vrf)` | 256 | 4 | 4 | per-feature vector |
| `widen_pad` | 256 | `H % 4 # 8` | 4 → 8 | |
| `cast` / `commit_trim` | 256 | `H % 4` | | 8 B write unit |

## Two mistakes worth recording

Both passed `typecheck` **and** `emulation`, and were caught only by `compile`.

**1. `vector_fp_div(H::SIZE as f32)` — rejected at MIR translation.**

```
mir: a vISA scalar operand must be known at compile time.
help: wrap computed values in `const { ... }`
```

`H::SIZE` const-folds, but the surrounding `as f32` survives into the MIR as a `Cast`.
A vISA scalar operand is encoded *in the instruction*, so it must fold completely. Fix:
`const { H::SIZE as f32 }`. This is the exact case upstream's `scalar_cast_diag.rs`
negative fixture exists to pin.

**2. `.to_vrf()` inherits collect padding.**

```
StoVrf: fn_output_shape does not match vrf_shape
  fn_output: [H / 4 = 256, [H % 4 + 4] = 8]    <- 2048 slots
  vrf:       [H = 1024]
```

`.to_vrf()` flattens `Element = [Time, Packet]`, so a 4-element `f32` packet padded to
the 32 B flit stores **8** slots per time step, not 4 — doubling the VRF footprint to
2048 `f32` = **exactly the 8 KB per-slice budget**, leaving no room for `inv_rms_vrf`.

Fix: fetch an 8-element packet in pass 1. 8 `f32` = 32 B is exactly one flit, nothing is
padded, and the VRF holds a clean `m![H]` = 4 KB.

**Rule of thumb: choose the VRF store's packet so the collect is an identity.**

## Numerics — the first rung that is not bit-exact

```
max abs diff: 1.5625e-2  (at index 121897, n = 262144)
```

`1.5625e-2 = 2^-6`, exactly **one `bf16` ULP** for a value in [2, 4). The kernel reduces
`sum(x*x)` as a 2-level packet tree then a temporal accumulator; torch sums pairwise. The
orders differ, the `f32` variance differs in its last bits, and `rsqrt` carries that into
a 1-ULP difference on the worst element out of 262,144.

This is legitimate, not a defect — but it is the first rung where reduction order is
*visible*, and it is worth watching: if a later change makes this figure jump by orders
of magnitude, the cause is a real bug rather than rounding.

## Constraints checked

- `Slice::SIZE == 256`, `Cluster::SIZE == 2`
- VRF budget: `weight` 4 KB + `inv_rms` 32 B ≤ 8 KB per slice
- `intra_slice_reduce`: only reduce dim is outermost → `InnerTime = m![1]`, 1 slot
- ALU disjointness within every pass (see the table above)
- `commit_trim` 16 B (pass 2) and 8 B (pass 4), both legal
- **Lowers to vISA** — `schedule.json` produced

## Follow-up (WBS 19)

- **Non-factoring `H`.** Real hidden sizes (896, 4864) do not divide by 4/8 cleanly.
  Since the reduce here is `Add`, rung 06b's identity-element contract applies —
  zero-fill the tail and reduce over the padded width.
- **`H` too large for one slice** (4096, 8192 at `f32` in VRF exceeds 8 KB): `H` must
  then live partly in `Slice`, which adds an `inter_slice_reduce` to pass 2 and makes the
  `weight` VRF per-slice-partial. This is what `norm.rs` actually does.
- **Fuse the residual add** (`hidden = residual + x` before the norm), as `norm.rs` does
  with `begin_interleaved` + `clip_zip`.
