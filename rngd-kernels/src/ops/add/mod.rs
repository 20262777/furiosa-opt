//! Rung 01: elementwise binary add.
//!
//! `C[i] = A[i] + B[i]` over `A = 2048` `bf16` elements.
//!
//! New concept vs. the base template: combining **two** DM tensors in one Tensor Unit
//! pass. The pipeline has a single stream input, so the two operands are read by one
//! interleaved fetch sequencer (`begin_interleaved`), split back apart inside the
//! Vector Engine (`vector_intra_slice_unzip`), and merged by a zip op
//! (`vector_clip_zip`).
//!
//! See `spec.md` for the mapping plan.

use furiosa_opt_std::prelude::*;

axes![
    A = 2048, // the vector length
    I = 2,    // interleave: which of the two operands this time step came from
];

pub type Chip = m![1];
pub type Cluster = m![1 # 2]; // 1 active cluster, padded to the hardware's 2
pub type Slice = m![A / 8]; // 2048 / 8 = 256 slices exactly, no padding needed
pub type Element = m![A % 8]; // 8 bf16 per slice (16 B)

/// `out[i] = lhs[i] + rhs[i]`
#[device(chip = 1)]
pub fn add_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![A]> {
    // HBM -> DM. `begin_interleaved` requires both views to have *identical* types,
    // so both land in the same Cluster/Slice/Element layout.
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, Element> = lhs.to_dm(&mut ctx.tdma);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, Element> = rhs.to_dm(&mut ctx.tdma);

    let sum: DmTensor<bf16, Chip, Cluster, Slice, Element> = ctx
        .main
        // One sequencer walking two buffers: t=0 reads lhs, t=1 reads rhs.
        // Time becomes m![I] instead of the usual m![1].
        .begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view())
        .fetch::<m![I], Element>()
        // The Vector Engine only accepts i32/f32, and there is no Contraction Engine
        // upstream to widen for us, so the Fetch Adapter has to do it.
        .fetch_cast::<f32>()
        // 8 x f32 = 32 B = exactly one flit, so this is an identity collect.
        .collect::<m![I], Element>()
        .vector_init()
        // De-interleave: TileTime = Time with I -> `1 # 2`, SplitTime = Time without I.
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        // Merge the two groups. f32 add at full 8-way rate lives in the Clip cluster
        // (the Float cluster's AddF would run 4-way and need narrow/widen around it).
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        // f32 -> bf16; 8 bf16 = 16 B, repadded to a full 32 B flit (16 elements).
        .cast::<bf16, m![A % 8 # 16]>()
        // Drop the flit padding: 8 bf16 = 16 B, a legal write unit (8/16/24/32 B).
        .commit_trim::<Element>()
        .commit();

    sum.to_hbm(&mut ctx.tdma)
}
