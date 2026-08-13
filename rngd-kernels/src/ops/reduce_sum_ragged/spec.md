# 06b — `reduce_sum_ragged`

## Math

```
out[s] = sum_r x[s, r]        s in 0..256, r in 0..1000
```

Row-wise sum. `bf16` in, `f32` accumulate, `bf16` out. Only the first `R_REAL = 999`
slots of each row carry data; the tail must hold the reduce operation's identity element.

## Why this rung exists

Rung 06 `reduce_sum` chose `A = 8192` so the reduce axis factored exactly as
`4 x 8 x 256`. Nothing was padded. **Real shapes are not that kind**, and reducing over a
padded axis without excluding the pad silently folds arbitrary data into the result.

This rung changes two things at once, both required by `rmsnorm` / `softmax`:

| | rung 06 | rung 06b |
|---|---|---|
| output | one scalar | **one value per row** (`S` stays in `Slice`) |
| reduce length | factors exactly | **999 real, padded to 1000** |

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
| `Slice` | `m![S]` | 256 — one row per slice, **never enters a reducer** |
| `Element` | `m![RP]` | 1000 slots per row, **no `#` padding** |

```
RP = 1000 :  RP % 4 (Packet, 4)  x  RP / 4 (Time, 250)
```

## Pad at the boundary, not at the DMA

The obvious approach — an HBM tensor of `m![S, R]` with `R = 999` DMA'd into a padded DM
region — makes the DMA write a **1998-byte** tail per row. HBM↔DM requires 8-byte
alignment on address and packet size, and 1998 is not a multiple of 8; this is the same
class of failure as `matmul_cluster_reduce`'s
`DMA tail alignment violation: reachable destination tail end ... = 1`.

Declaring the padded width from HBM onward makes every row 1000 slots = 2000 B, which
**is** 8-byte aligned. The DMA moves a clean rectangle. This part held up across both
designs below and is the reusable lesson.

## ⚠ The VCG approach does not lower to vISA — verified

The first version declared the axis as `m![R # 1000]` with `R = 999` and let the **Valid
Count Generator** mask the pad, exactly as the book's `vcg.md` describes. It passed
`emulation` bit-exactly and **failed vISA lowering**. Two placements were tried:

| reduce axis placement | lowering error |
|---|---|
| `R # 1000` split across `Time` (`/4`) and `Packet` (`%4`) | `cannot reduce pack alias` |
| `R # 1000` in `Time` only, `Packet = m![1]` | **`reduce axis should not have padding`** |
| *(control)* rung 06's unpadded `A = 8192` | lowers fine ✓ |

The second message is unambiguous and placement-independent: **`vector_intra_slice_reduce`
will not lower over a padded reduce axis.** Since every example in `vcg.md` is built on
`R # PADDED_SIZE` plus `vector_intra_slice_reduce::<R, ...>`, the documented capability
appears unreachable through this API path on hardware.

Why this was invisible: the `vcg.md` examples are live doctests, but doctests exercise
`emulation` / `typecheck` — never the vISA lowering. A feature can be documented, tested,
and still not lower.

**This should go to FuriosaAI.** Either the restriction is a lowering bug, or the
supported VCG placements are far narrower than the chapter implies and it needs to say
so. It is a concrete instance of the interpreter-vs-hardware divergence that indicator 10's
measurement method exists to catch.

## What ships instead: identity-element padding

The axis is declared at its **padded** size (`RP = 1000`, no `#`), and the caller puts the
reduce operation's identity element in the tail.

| | VCG version | identity version |
|---|---|---|
| axis | `m![R # 1000]`, `R = 999` | `m![RP]`, `RP = 1000` |
| tail handling | masked by hardware | **caller must supply `0.0`** |
| emulation | ✅ bit-exact | ✅ bit-exact |
| vISA lowering | ❌ | ✅ |

The cost is a precondition the kernel cannot check, and one that **does not generalise**:
identities exist for `Add` (`0`), `Max` (`-inf`), `Min` (`+inf`), but not for
`sum(exp(x))` — no `p` satisfies `exp(p) = 0`. **Softmax still needs an answer**, and
that is now the open question blocking rung 08, not rung 07.

## Pipeline trace

| stage | `Time` | `Packet` | way | note |
|---|---|---|---|---|
| `fetch` | 250 | 4 `bf16` | — | 8 B |
| `fetch_cast::<f32>` | 250 | 4 `f32` | — | 16 B |
| `collect` | 250 | `RP % 4 # 8` | — | padded to the 32 B flit |
| `narrow_trim` | 250 | 4 `f32` | 8 → 4 | upper 4 were collect padding |
| `intra_slice_reduce(Add)` | 1 | `1 # 4` | 4 | identity tail contributes 0 |
| `widen_pad` | 1 | `1 # 8` | 4 → 8 | |
| `cast` / `commit_trim` | 1 | `1 # 4` | | 8 B write unit |

No `inter_slice_reduce`: `S` lives in `Slice` and must survive.

## Test design

Two tests, and the second is the interesting one.

**`test_reduce_sum_ragged`** — contract satisfied, tail is `0.0`. Result must equal the
sum of the 999 real elements. `max abs diff: 0e0`.

**`test_reduce_sum_ragged_tail_is_summed`** — contract *violated*, tail is a `1e4`
sentinel. The result must equal the sum **including** the sentinel:

```
expected[0]          =    0.6484
expected_poisoned[0] = 9984.0000
```

This is a guard, not a feature. It proves the tail is genuinely summed, so the caller
contract is load-bearing rather than decorative. If it ever starts matching the clean
expectation, the kernel has silently changed semantics.

(The earlier VCG design used the sentinel the other way round — masking was supposed to
*exclude* it. Keeping a sentinel test in both designs is deliberate: zero-padding alone
would make either a broken mask or a broken contract look correct.)

## Constraints checked

- `Slice::SIZE == 256`, `Cluster::SIZE == 2`
- `intra_slice_reduce`: only reduce dim is outermost → `InnerTime = m![1]`, 1 slot
- Per-row DM footprint 2000 B, 8-byte aligned
- `commit_trim` valid size = 8 B
- **Lowers to vISA** — `schedule.json` produced (2982 B)

## Follow-up

- **`Max` variant** for `softmax`'s row-max pass; identity is `-inf`.
- **Softmax's `sum(exp(x))`** has no identity padding. Options to explore: mask to zero
  with the Fetch Adapter's `fetch_mask` *before* the exp; use `TagMode::Comparison` +
  `Filter` to drop pad lanes; or a two-pass shape that commits the masked values to DM
  and reduces over an unpadded axis in a second pass.
- **`RP` too long for one slice** — needs the reduce axis partly in `Slice`, which is the
  `SliceMajor` / `TimeMajor` VCG territory that does not currently lower.
