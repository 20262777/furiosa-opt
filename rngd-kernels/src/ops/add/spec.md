# 01 — `add`

## Math

```
out[i] = lhs[i] + rhs[i]        i in 0..2048
```

`bf16` in, `bf16` out. The kernel widens to `f32` for the add and narrows back, so the
ground truth must do the same: `(lhs.float() + rhs.float()).bfloat16()`.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `lhs` | `m![A]` | 2048 `bf16` |
| `rhs` | `m![A]` | 2048 `bf16` |
| `out` | `m![A]` | 2048 `bf16` |

## Mapping plan

`A = 2048` splits cleanly over the hardware with nothing left over:

```
A = 2048 :  A / 8 (Slice, 256)  x  A % 8 (Element, 8)
```

| dim | mapping | size | why |
|---|---|---|---|
| `Chip` | `m![1]` | 1 | single chip |
| `Cluster` | `m![1 # 2]` | 2 | 1 active + 1 padding; hardware requires exactly 2 |
| `Slice` | `m![A / 8]` | 256 | exactly the per-cluster slice count, no padding |
| `Element` | `m![A % 8]` | 8 | 8 `bf16` = 16 B per slice |

## Pipeline trace

| stage | `Time` | `Packet` | bytes |
|---|---|---|---|
| `begin_interleaved` | `m![I]` = 2 | `m![A % 8]` | — |
| `fetch` | 2 | 8 `bf16` | 16 B |
| `fetch_cast::<f32>` | 2 | 8 `f32` | 32 B |
| `collect` | 2 | 8 `f32` | 32 B — one flit, identity |
| `unzip` | 1 | 8 `f32` | two groups in parallel |
| `clip_zip(Add)` | 1 | 8 `f32` | merged |
| `cast::<bf16>` | 1 | 16 `bf16` | 32 B flit |
| `commit_trim` | 1 | 8 `bf16` | 16 B write unit |

## Constraints checked

- `Cluster::SIZE == 2`, `Slice::SIZE == 256` — match hardware exactly
- `Element::SIZE * size_of::<bf16>()` = 16 B, far under the 512 KB/slice DM budget
- Collect output is exactly one 32-byte flit
- `commit_trim` valid size = 16 B, one of the legal 8/16/24/32
- ALU budget: one op only (`ClipAdd`), nothing contends

## Notes

- `Cluster = m![1 # 2]` wastes half the chip. Using both clusters would mean
  `Cluster = m![A / 1024 % 2]` and halving the per-cluster share of `A` — worth doing
  once the ladder reaches a rung where throughput matters.
- The interleave costs a 2x read of DM (each operand fetched once, but as two separate
  time steps). For an op this cheap the kernel is entirely memory bound.
