//! Rung 02: elementwise multiply, VRF operand.
//!
//! `out[i] = lhs[i] * rhs[i]` over `A = 2048` `bf16` elements.
//!
//! Two new concepts vs. rung 01 `add`:
//!
//! 1. **VRF operand.** Instead of interleaving both operands into one stream, `rhs` is
//!    pre-loaded into the Vector Register File by the sub context and read as an operand
//!    every cycle. Unlike a `Stash` (read-once, from the stream itself) a VRF tensor is
//!    read-many and holds a *different* value per element.
//! 2. **Narrow / widen.** `f32` multiply lives in the Float cluster, which runs **4-way**,
//!    so the chain has to drop from Way8 to Way4 around it and come back. (`add` avoided
//!    this because `ClipBinaryOpF32::Add` is the one full-rate 8-way `f32` add.)
//!
//! The base template's `elementwise_mul_kernel.rs` uses `i32` + `FxpBinaryOp::MulInt`,
//! which is 8-way and needs no narrow/widen. This rung deliberately takes the `bf16`
//! path because that is what `rmsnorm` and `softmax` will need.
//!
//! See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8]; // 2048 / 8 = 256 slices exactly
pub type Element = m![A % 8]; // 8 bf16 per slice

/// `out[i] = lhs[i] * rhs[i]`
#[device(chip = 1)]
pub fn mul_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![A]> {
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, Element> = lhs.to_dm(&mut ctx.tdma);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, Element> = rhs.to_dm(&mut ctx.tdma);

    // Sub context: park `rhs` in the VRF. `.to_vrf()` requires a `VeScalar` element type
    // (i32 or f32), so the widening happens here rather than downstream.
    let rhs_vrf: VrfTensor<f32, Chip, Cluster, Slice, Element> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<m![1], Element>()
        .fetch_cast::<f32>()
        .collect::<m![1], Element>() // 8 x f32 = 32 B = one flit
        .to_vrf();

    // Main context: stream `lhs`, multiply by its VRF counterpart.
    let product: DmTensor<bf16, Chip, Cluster, Slice, Element> = ctx
        .main
        .begin(lhs.view())
        .fetch::<m![1], Element>()
        .fetch_cast::<f32>()
        .collect::<m![1], Element>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 -> Way4. `_split` (not `_trim`) because both halves hold real data:
        // [T], [P] -> [T, P / 2], [P % 4].
        .vector_narrow_split::<m![A / 4 % 2], m![A % 4]>()
        // The Float cluster's FpMul0 ALU. Operand comes from VRF, one value per element.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rhs_vrf)
        // Way4 -> Way8, the inverse of `_split`: [T, P / 2], [P % 4] -> [T], [P].
        .vector_widen_concat::<m![1], Element>()
        .vector_final()
        .cast::<bf16, m![A % 8 # 16]>()
        .commit_trim::<Element>()
        .commit();

    product.to_hbm(&mut ctx.tdma)
}
