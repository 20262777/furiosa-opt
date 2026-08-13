//! Rung 06: full reduction to a scalar.
//!
//! `out = sum_i x[i]` over `A = 8192` `bf16` elements.
//!
//! New concept: the reduction axis is spread over **all three** of `Slice`, `Time` and
//! `Packet`, so it takes two different reducers to collapse it — the Contraction Engine
//! is not involved at all.
//!
//! ```text
//! A = 8192 :  A % 4        (Packet, 4)    -> intra-slice tree reduce
//!             A % 32 / 4   (Time,   8)    -> intra-slice temporal accumulate
//!             A / 32       (Slice,  256)  -> inter-slice reducer
//!             4 x 8 x 256 = 8192
//! ```
//!
//! `vector_intra_slice_reduce` handles the `Time` and `Packet` portions; whatever sits in
//! `Slice` survives it and needs `vector_inter_slice_reduce` after. This split is the
//! foundation for `rmsnorm` (sum of squares) and `softmax` (row max, sum of exp).
//!
//! See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![A = 8192];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 32]; // 8192 / 32 = 256 slices exactly
pub type Element = m![A % 32]; // 32 bf16 per slice

/// `out = sum_i x[i]`
#[device(chip = 1)]
pub fn reduce_sum_kernel(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {
    let x: DmTensor<bf16, Chip, Cluster, Slice, Element> = x.to_dm(&mut ctx.tdma);

    let total: DmTensor<bf16, Chip, Cluster, m![1 # 256], m![1 # 4]> = ctx
        .main
        .begin(x.view())
        // Split the per-slice 32 elements into 8 time steps x 4 packet elements.
        .fetch::<m![A % 32 / 4], m![A % 4]>()
        // The Vector Engine takes only i32/f32 and there is no contraction upstream.
        .fetch_cast::<f32>()
        // 4 x f32 = 16 B, padded to the 32 B flit (8 slots, upper 4 are padding).
        .collect::<m![A % 32 / 4], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 -> Way4. `_trim` (not `_split`) because the upper 4 slots are the
        // collect padding, not data.
        .vector_narrow_trim::<m![A % 4]>()
        // Collapses every factor of `A` living in Time and Packet:
        // a 2-level tree over `A % 4`, then accumulation over `A % 32 / 4`.
        .vector_intra_slice_reduce::<A, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        // Way4 -> Way8; the exit to the inter-slice reducer requires 8-way.
        .vector_widen_pad::<m![1 # 8]>()
        // The `A / 32` portion of the reduction still lives across 256 slices.
        // The freed Slice slot becomes a `1 # 256` dummy.
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>() // 4 bf16 = 8 B, the minimum write unit
        .commit();

    total.to_hbm(&mut ctx.tdma)
}
