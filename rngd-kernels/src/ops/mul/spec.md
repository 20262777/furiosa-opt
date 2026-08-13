# 02 — `mul`

## Math

```
out[i] = lhs[i] * rhs[i]        i in 0..2048
```

`bf16` in, `f32` multiply, `bf16` out.

## Shapes

Identical to rung 01 `add`: `lhs`, `rhs`, `out` all `m![A]`, 2048 `bf16`.

## Mapping plan

| dim | mapping | size |
|---|---|---|
| `Chip` | `m![1]` | 1 |
| `Cluster` | `m![1 # 2]` | 2 |
| `Slice` | `m![A / 8]` | 256 |
| `Element` | `m![A % 8]` | 8 `bf16` |

## Why this is not just `add` with a different op

`add` and `mul` take **different paths through the Vector Engine**, and the reason is
where each arithmetic op lives:

| op | cluster | way | narrow/widen needed? |
|---|---|---|---|
| `ClipBinaryOpF32::Add` | Clip | **8** | no |
| `FpBinaryOp::MulF` | Float | **4** | **yes** |
| `FxpBinaryOp::MulInt` | Fxp | 8 | no — but `i32` only |

There is no full-rate 8-way `f32` multiply. The base template sidesteps this by using
`i32` (`elementwise_mul_kernel.rs`); this rung keeps `bf16` for ladder consistency and
pays the narrow/widen, because `rmsnorm` and `softmax` will have to anyway.

## Two new concepts

**VRF operand.** `rhs` is pre-loaded into the Vector Register File by the sub context and
read as an operand every cycle. Contrast the three operand sources:

| source | value | reads |
|---|---|---|
| constant | same for every element | unlimited |
| **VRF** | **different per element, loaded beforehand** | **many** |
| `Stash` | this element's own earlier value | exactly one |

`.to_vrf()` requires a `VeScalar` element type (`i32`/`f32`), so `fetch_cast::<f32>()`
happens in the sub context before the store. 8 `f32` = 32 B/slice, far under the 8 KB
VRF budget.

**Narrow / widen.** `_split` and `_concat` (not `_trim`/`_pad`) because both halves of
the packet hold real data:

```
narrow_split : [T], [P]            -> [T, P / 2], [P % 4]
             : m![1], m![A % 8]    -> m![A / 4 % 2], m![A % 4]
widen_concat : [T, P / 2], [P % 4] -> [T], [P]
```

## Pipeline trace

| stage | `Time` | `Packet` | way | bytes |
|---|---|---|---|---|
| `fetch` | 1 | 8 `bf16` | — | 16 B |
| `fetch_cast::<f32>` | 1 | 8 `f32` | — | 32 B |
| `collect` | 1 | 8 `f32` | — | one flit, identity |
| `narrow_split` | 2 | 4 `f32` | 8 → 4 | |
| `fp_binary(MulF(Mul0))` | 2 | 4 `f32` | 4 | operand from VRF |
| `widen_concat` | 1 | 8 `f32` | 4 → 8 | |
| `cast::<bf16>` | 1 | 16 `bf16` | | 32 B flit |
| `commit_trim` | 1 | 8 `bf16` | | 16 B write unit |

## ALU budget

One ALU used: `FpMul0`. `FpMul1` and `FpFma` remain free, so a second and third multiply
could fuse into this same pass — which is exactly what `rmsnorm`'s final scale does
(`x * inv_rms * weight` via `Mul0` + `Mul1`).

## Constraints checked

- VRF element type is `f32` (`VeScalar`); 32 B/slice against 8 KB
- Collect output is exactly one 32-byte flit
- `commit_trim` valid size = 16 B, legal
- Float cluster entered only in the Way4 phase

## Known inefficiency

Narrowing halves throughput on the float path — the same logical work takes twice as
many packets as `add` does. Unavoidable for `f32` multiply.

## Follow-up

- `i32` variant using `FxpBinaryOp::MulInt` (8-way, no narrow) for WBS 20 data-type
  coverage, and as a direct throughput comparison against this one.
