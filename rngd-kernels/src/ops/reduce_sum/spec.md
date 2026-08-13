# 06 — `reduce_sum`

## Math

```
out = sum_i x[i]        i in 0..8192
```

Einsum `A -> 1`. `bf16` in, `f32` accumulate, `bf16` out.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `x` | `m![A]` | 8192 `bf16` (16 KB) |
| `out` | `m![1]` | 1 `bf16` |

## Mapping plan

`A = 8192` is chosen so it factors exactly across all three levels with nothing padded:

```
A = 8192 :  A % 4      (Packet, 4)    x  A % 32 / 4 (Time, 8)  x  A / 32 (Slice, 256)
            4 x 8 x 256 = 8192
```

| dim | mapping | size |
|---|---|---|
| `Chip` | `m![1]` | 1 |
| `Cluster` | `m![1 # 2]` | 2 |
| `Slice` | `m![A / 32]` | 256 — exact, no padding |
| `Element` | `m![A % 32]` | 32 `bf16` = 64 B per slice |

## The point of this rung: two reducers, not one

The Contraction Engine is **not used at all**. The reduction is done entirely in the
Vector Engine, and it takes two different reducers because they cover disjoint dimensions:

| reducer | covers | mechanism |
|---|---|---|
| `vector_intra_slice_reduce` | the `Time` **and** `Packet` factors of `A` | 2-level tree over `Packet`, accumulator over `Time` |
| `vector_inter_slice_reduce` | the `Slice` factor of `A` | ring across the 256 slices, `O(r)` cycles |

Whatever portion of the reduce axis lives in `Slice` **survives** the intra-slice stage —
that is the rule to internalise, and it is why `rmsnorm`'s variance pass in
`transformer/common/norm.rs` has exactly this same pair of calls.

## Pipeline trace

| stage | `Time` | `Packet` | way | note |
|---|---|---|---|---|
| `fetch` | 8 | 4 `bf16` | — | 8 B |
| `fetch_cast::<f32>` | 8 | 4 `f32` | — | 16 B |
| `collect` | 8 | `A % 4 # 8` | — | padded to the 32 B flit |
| `narrow_trim` | 8 | 4 `f32` | 8 → 4 | `_trim`: upper 4 slots were collect padding |
| `intra_slice_reduce(Add)` | 1 | `1 # 4` | 4 | `A` gone from Time and Packet |
| `widen_pad` | 1 | `1 # 8` | 4 → 8 | the reducer exit requires 8-way |
| `inter_slice_reduce(Add)` | 1 | `1 # 8` | 8 | `Slice` → `1 # 256` dummy |
| `cast` / `commit_trim` | 1 | `1 # 4` | | 8 B write unit |

## Constraints checked

- `Slice::SIZE == 256`, `Cluster::SIZE == 2` — exact hardware match
- `intra_slice_reduce`: the only reduce dim is outermost, so `InnerTime = m![1]`
  (1 accumulator slot against a capacity of 8) ✓
- Both `vector_final()` and the transition into the inter-slice reducer require Way8 —
  hence the `widen_pad` between the two reducers
- `commit_trim` valid size = 8 B, the minimum legal unit

## No VCG needed

`A = 8192` divides evenly by 4, 8 and 256, so no factor of the reduce axis is padded and
the Valid Count Generator never has to mask anything. **A length that does not factor
cleanly is a materially harder kernel** — see rung 06b `reduce_sum_ragged`, where the
padded-axis approach the book documents turns out not to lower to vISA at all.

## Numerical note

torch sums pairwise; the kernel sums tree(4) → accumulator(8) → ring(256). The orders
differ, so a couple of `bf16` ULP of drift would be legitimate. Measured at this seed:
**0e0** — no visible difference. The test allows `2e-2` relative with a `5e-1` absolute
floor, the floor being there because a sum of 8192 zero-mean values can land near zero
where relative tolerance is meaningless.

## Follow-up (WBS 19)

- `Max` / `Min` variants (`IntraSliceReduceOpF32::Max`, `InterSliceReduceOpF32::Max`),
  needed by `softmax`'s row-max pass.
- Row-wise reduction (reduce the last axis of a 2-D tensor, keeping the first) — done in
  rung 06b; this rung reduces to a single scalar, which no real LLM op does.
