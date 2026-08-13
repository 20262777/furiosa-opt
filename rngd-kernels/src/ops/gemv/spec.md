# 04 — `gemv`

## Math

```
y[i] = sum_j matrix[i, j] * vector[j]      i in 0..256, j in 0..2048
```

Einsum `IJ, J -> I`. `bf16` in, `f32` accumulate, `bf16` out.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `matrix` | `m![I, J]` | 256 x 2048 `bf16` (1 MB) |
| `vector` | `m![J]` | 2048 `bf16` (4 KB) |
| `out` | `m![I]` | 256 `bf16` |

## Mapping plan

| dim | mapping | size | role |
|---|---|---|---|
| `Chip` | `m![1]` | 1 | single chip |
| `Cluster` | `m![1 # 2]` | 2 | 1 active + 1 padding |
| `Slice` | `m![I]` | 256 | **output rows**, exactly the slice count — no padding |
| `Lane` | `m![1]` | 1 | 1 of 8 lanes |

`I` never reaches the reduce stages; it is carried entirely by `Slice`. `J` is split:

```
J = 2048 :  J % 32 (Packet, 32)  x  J / 32 (Time, 64)
```

## The two DMAs do opposite things

Both call `to_dm`; only the destination type differs in what it implies.

| source | dest `Slice` | effect |
|---|---|---|
| `matrix: m![I, J]` — **has** `I` | `m![I]` | **distribution** — row `i` to slice `i`, 4 KB/slice |
| `vector: m![J]` — **no** `I` | `m![I]` | **broadcast** — stride-0 entry, full 4 KB copy to every slice (1 MB total) |

This is the DMA sequencer's broadcast rule (an axis in the destination but not in the
buffer gets stride 0), *not* the Switch Engine.

## Pipeline trace

| stage | `Time` | `Packet` | note |
|---|---|---|---|
| `fetch` | 128 | 16 `bf16` | 32 B, one read per packet |
| `collect` | 128 | 16 `bf16` | identity |
| `contract_outer` | 64 | 32 `bf16` | 64 B, `PackSize = 2` |
| `contract_packet` | 64 | 1 | depth-5 tree |
| `contract_time` | 1 | 1 | `J` fully reduced → `y[i]` |
| `contract_lane` | 1 | `1 # 8` | trivial fold |
| `cast` / `commit_trim` | 1 | `1 # 4` | 8 B write unit |

## Constraints checked

- `Slice::SIZE == 256`, `Cluster::SIZE == 2` — exact hardware match
- TRF: `m![J]` = 4 KB/lane, within the 8 KB budget
- `commit_trim` valid size = 8 B, the minimum legal unit

## Known inefficiency

- `Lane = m![1]` — 7 of 8 lanes idle, and `contract_lane` fills 1 of 8 bus slots.
  `gemm` uses that slot for its second output dimension.
- `Cluster = m![1 # 2]` — half the chip unused.
- Sub-context TRF load (~512 reads at 8 B) likely dominates the ~128 cycles of main
  streaming. Amortise across many matrices, or double-buffer the TRF halves.

## Follow-up (WBS 18)

LLM-realistic GEMV shapes (`J` = 4096/8192, `I` = hidden or vocab) exceed one pass and
need the `matmul_split_reduce` tiling pattern.
