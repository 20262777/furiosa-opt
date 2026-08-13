# 03 — `dot`

## Math

```
out = sum_i lhs[i] * rhs[i]        i in 0..2048
```

Einsum `I, I -> 1`. No broadcast; both operands reduce along the same axis.
`bf16` in, `f32` accumulate, `bf16` out.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `lhs` | `m![A]` | 2048 `bf16` |
| `rhs` | `m![A]` | 2048 `bf16` |
| `out` | `m![1]` | 1 `bf16` |

## Mapping plan

The whole vector lives in **one slice**. This is the teaching configuration, not a fast
one — `Slice = m![1 # 256]` means 255 of 256 slices idle.

| dim | mapping | size | role |
|---|---|---|---|
| `Chip` | `m![1]` | 1 | single chip |
| `Cluster` | `m![1 # 2]` | 2 | 1 active + 1 padding |
| `Slice` | `m![1 # 256]` | 256 | **1 active**; `m![A / 8 # 256]` would distribute |
| `Lane` | `m![1]` | 1 | 1 of 8 lanes |

Reduction split across the contraction stages:

```
A = 2048 :  A % 32 (Packet, 32)  x  A / 32 (Time, 64)  x  Lane (1)
```

## Pipeline trace

| stage | `Time` | `Packet` | note |
|---|---|---|---|
| `fetch` | 1 | 2048 `bf16` | one 4096 B packet, multi-read |
| `collect` | 128 | 16 `bf16` | 32 B flits |
| `contract_outer` | 64 | 32 `bf16` | 64 B, `PackSize = 2`, full MAC width |
| `contract_packet` | 64 | 1 | depth-5 tree over 32 |
| `contract_time` | 1 | 1 | accumulate 64 → `A` fully reduced |
| `contract_lane` | 1 | `1 # 8` | trivial fold, 1 of 8 bus slots used |
| `cast::<bf16>` | 1 | `1 # 16` | 32 B flit |
| `commit_trim` | 1 | `1 # 8` | 16 B write unit |

## Constraints checked

- TRF holds `m![A]` = 2048 `bf16` = 4 KB per lane, within the 8 KB/lane budget
- `OutPacket::SIZE * size_of::<bf16>()` = 64 B → `PackSize = 2` ✓
- `commit_trim` valid size = 16 B, legal (8/16/24/32)

## Known inefficiency

Sub-context `read_size` is fixed at 8 B, so loading 4 KB into the TRF costs ~512 reads
versus ~128 cycles to stream `lhs` in main. **The TRF preload dominates.** A real kernel
amortises it across many streams or double-buffers `FirstHalf`/`SecondHalf`.

## Follow-up (WBS 18)

- Distribute across slices (`Slice = m![A / 8 # 256]`) and finish with
  `vector_inter_slice_reduce`, as `matmul_4096` does.
- LLM-realistic lengths (`A` = 4096, 8192) exceeding a single pass → tiling.
