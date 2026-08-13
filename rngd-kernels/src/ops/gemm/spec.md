# 05 — `gemm`

## Math

```
C[i, j] = sum_k a[i, k] * b[j, k]      i, j in 0..512, k in 0..64
```

Einsum `IK, JK -> IJ`, i.e. `C = A B^T`. **`b` is stored transposed** so `K` is innermost
in both operands. `bf16` in, `f32` accumulate, `bf16` out.

## Shapes

| tensor | HBM mapping | elements |
|---|---|---|
| `a` | `m![I, K]` | 512 x 64 `bf16` (64 KB) |
| `b` | `m![J, K]` | 512 x 64 `bf16` (64 KB) |
| `out` | `m![I, J]` | 512 x 512 `bf16` (512 KB) |

## Mapping plan

Every axis is split two or three ways. This is the point of the rung.

| axis | placement | sizes |
|---|---|---|
| `I` (512) | `I / 32` → `Slice`, `I % 32` → `Time` | 16 x 32 |
| `J` (512) | `J / 32` → `Slice`, `J / 8 % 4` → `Time`, `J % 8` → **`Lane`** | 16 x 4 x 8 |
| `K` (64) | `K % 32` → `Packet`, `K / 32` → `Time` | 32 x 2 |

`Slice = m![I / 32, J / 32]` is a **16 x 16 grid of slices**, and each slice owns a
**32 x 32 output tile** (`m![I % 32, J % 32]`, 1024 elements = 2 KB).

> The base-template comment says "16 x 16 output tile" — that is wrong, and the same
> error appears in the book's Quick Start. 16 x 16 is the *slice grid*; the tile is
> 32 x 32, as the result type states.

## Both DMAs broadcast

| source | missing axis | effect |
|---|---|---|
| `a: m![I, K]` | no `J` | `J / 32` becomes a stride-0 entry — A replicated across the 16 J-columns |
| `b: m![J, K]` | no `I` | `I / 32` becomes a stride-0 entry — B replicated across the 16 I-rows |

Classic 2-D outer-product tiling: each operand replicated 16x so every slice holds the
pair it needs, 4 KB each.

## Why the TRF fetch ordering matters

```rust
.fetch::<m![J % 8, J / 8 % 4], m![K]>()
```

`.to_trf()` peels `Lane` off the **outermost** `Time` factor (`Lane = Time / FlitsPerLane`),
so putting `J % 8` first is what gives each lane a distinct set of B rows. Here
`FlitsPerLane = 128 / 8 = 16`, and 16 flits x 16 elements = the 256-element `Element`.

## Pipeline trace

| stage | `Time` | `Packet` | note |
|---|---|---|---|
| `fetch` | 128 | 64 `bf16` | 128 B; `J / 8 % 4` is a stride-0 broadcast |
| `collect` | 512 | 16 `bf16` | 32 B flits |
| `contract_outer` | 256 | 32 `bf16` | 64 B, `PackSize = 2`, `Lane = 8` |
| `contract_packet` | 256 | 1 | depth-5 tree over `K % 32` |
| `contract_time` | 128 | 1 | reduces `K / 32` only; `InnerTime = 1`, one slot |
| `contract_lane` | 128 | 8 | Interleaved — **full 8-wide flit** |
| `cast` / `commit_trim` | 128 | `J % 8` | 16 B write unit |

Conservation holds at every step: `512 x 16 = 256 x 32 = 8192`.

## Constraints checked

- `Slice::SIZE = 16 x 16 = 256`, `Cluster::SIZE = 2` — exact hardware match
- `Lane::SIZE = 8` ∈ {1, 2, 4, 8}
- TRF: `m![J / 8 % 4, K]` = 256 `bf16` = 512 B per lane, well under 8 KB
- `contract_time` `InnerTime::SIZE = 1`, far under the `Interleaved` slot capacity
- `commit_trim` valid size = 16 B, legal

## Efficiency

**100% MAC utilisation on the active cluster.** Multiplies issued equals `I x J x K`
exactly — no zero-padded packet half (`PackSize = 2`), no idle lanes (`Lane = 8`), no
padding slots. Reuse is high too: A is read 4x temporally and 16x spatially; B is
broadcast across all 32 `I` steps from the TRF.

The only waste is `Cluster = m![1 # 2]` — 256 of the chip's 512 slices. Splitting
`I / 32` as `[I / 64, I / 32 % 2]` with the outer factor in `Cluster` would use both.

## Follow-up (WBS 18)

- Two-cluster variant (above).
- LLM-realistic shapes: `K` = 4096/8192 far exceeds one pass, so `K` must be tiled with
  partial sums accumulated in DM — the `matmul_split_reduce` pattern.
- `f8e4m3` / `i8` variants for WBS 20.
