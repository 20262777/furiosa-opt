//! Rung 08: row-wise softmax over a padded axis.
//!
//! ```text
//! y[s, r] = exp(x[s, r] - max_r x[s, r]) / sum_r exp(x[s, r] - max_r x[s, r])
//! ```
//!
//! `S = 256` rows, `RP = 1000` slots per row of which `R_REAL = 999` carry data.
//!
//! # Padding costs nothing here
//!
//! The tail holds `NEG_SENTINEL = -3.3895314e38`, and that single value is the identity
//! for **both** reductions in this kernel:
//!
//! - `Max` — it is more negative than any real score, so the row max is unaffected.
//! - `Add` after `exp` — `exp(sentinel - max)` underflows to exactly `0`.
//!
//! So no Valid Count Generator, no `fetch_mask`, no `Filter` stage. And it is the same
//! write a causal or padding mask already performs, which is why upstream's
//! `transformer/attention/softmax.rs` uses this exact constant. Rung 07b
//! [`reduce_sum_exp`](../reduce_sum_exp) established the `Add` half.
//!
//! Padded output positions come out as exactly `0.0`, which the test checks.
//!
//! # Three passes
//!
//! | pass | computes | ALUs |
//! |---|---|---|
//! | 1 | `max_r x` -> VRF | reduce tree |
//! | 2 | `1 / sum_r exp(x - max)` -> VRF | `FpFma`, `FpExp`, reduce tree, `FpDiv` |
//! | 3 | `exp(x - max) * inv_sum` -> DM | `FpFma`, `FpExp`, `FpMul0` |
//!
//! Passes 2 and 3 each pack three Float-cluster operations into one invocation, which is
//! legal only because `SubF`, `Exp` and `MulF(Mul0)` land on `FpFma`, `FpExp` and
//! `FpMul0` respectively. Note the chain's fixed stage order does the rest: `Float` (6)
//! runs before `IntraSliceReduce` (7), which runs before `FpDiv` (8) — so pass 2's
//! subtract, exp, sum and reciprocal all fit in a single trip down the pipeline.
//!
//! Unlike upstream's masked softmax, which materialises the full score matrix in DM and
//! reads it back three times, this reads `x` from DM three times but never writes an
//! intermediate of row width — only two per-row scalars, both in VRF.
//!
//! See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![
    S = 256,   // rows, one per slice
    RP = 1000, // padded row width; slots >= R_REAL must be NEG_SENTINEL
];

/// Number of leading slots of `RP` that carry real data.
pub const R_REAL: usize = 999;

/// Identity for `Max` *and*, after `exp`, for `Add`.
pub const NEG_SENTINEL: f32 = -3.3895314e38;

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![S];
pub type Element = m![RP]; // no `#` padding: a padded reduce axis does not lower

/// Per-row scalar staged in DM by a main pass, then lifted to VRF by a sub pass.
///
/// The round trip through DM is **required**, not a stylistic choice: `.to_vrf()` placed
/// directly after `vector_intra_slice_reduce` fails vISA lowering with
/// `StoVrf: fn_output_shape does not match vrf_shape`, because the store uses the
/// stream's *pre-reduce* `[Time, Packet]` (here `[RP / 4 = 250, RP % 4 # 8 = 8]`) rather
/// than the reduced `m![1 # 8]`. Emulation accepts it; the hardware path does not.
/// `rmsnorm` has the same shape for the same reason.
///
/// 8 `f32` = 32 B is exactly one flit, so the collect is an identity and nothing is
/// padded into the VRF.
type RowScalarDm = DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]>;
type RowScalarVrf = VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]>;

/// Row-wise softmax. Correct only if `x[s, r >= R_REAL] == NEG_SENTINEL`.
#[device(chip = 1)]
pub fn softmax_kernel(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![S, RP]>,
) -> HbmTensor<bf16, Chip, m![S, RP]> {
    let x: DmTensor<bf16, Chip, Cluster, Slice, Element> = x.to_dm(&mut ctx.tdma);

    // ---- pass 1 (main): row max -> DM ---------------------------------------------------
    // The sentinel is below every real score, so it never wins the max.
    let max_dm: RowScalarDm = ctx
        .main
        .begin(x.view())
        .fetch::<m![RP / 4], m![RP % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![RP / 4], m![RP % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![RP % 4]>()
        .vector_intra_slice_reduce::<RP, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>() // 8 f32 = 32 B
        .commit();

    // ---- pass 2 (sub): row max -> VRF ---------------------------------------------------
    let max_vrf: RowScalarVrf = ctx
        .sub
        .begin(max_dm.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>() // identity: already one flit
        .to_vrf();

    // ---- pass 3 (main): 1 / sum(exp(x - max)) -> DM -------------------------------------
    // Subtract, exp, reduce and reciprocate all fit one invocation because the chain's
    // stage order is Float(6) -> IntraSliceReduce(7) -> FpDiv(8).
    let inv_sum_dm: RowScalarDm = ctx
        .main
        .begin(x.view())
        .fetch::<m![RP / 4], m![RP % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![RP / 4], m![RP % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![RP % 4]>()
        // Mode01 (default): op(stream, operand) = x - max.
        .vector_fp_binary(FpBinaryOp::SubF, &max_vrf) // FpFma
        // The sentinel underflows to 0 right here, so the tail adds nothing below.
        .vector_fp_unary(FpUnaryOp::Exp) // FpExp
        .vector_intra_slice_reduce::<RP, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        // Mode10 swaps the slots: constant / stream = 1 / sum.
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // ---- pass 4 (sub): 1 / sum -> VRF ---------------------------------------------------
    let inv_sum_vrf: RowScalarVrf = ctx
        .sub
        .begin(inv_sum_dm.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // ---- pass 5 (main): normalise -------------------------------------------------------
    // Multiply by the reciprocal rather than dividing: FpDiv is a single dedicated unit,
    // and this keeps the op on FpMul0 so nothing contends.
    let out: DmTensor<bf16, Chip, Cluster, Slice, Element> = ctx
        .main
        .begin(x.view())
        .fetch::<m![RP / 4], m![RP % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![RP / 4], m![RP % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![RP % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max_vrf) // FpFma
        .vector_fp_unary(FpUnaryOp::Exp) // FpExp  -> tail becomes 0.0
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_sum_vrf) // FpMul0
        .vector_widen_pad::<m![RP % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![RP % 4 # 16]>()
        .commit_trim::<m![RP % 4]>()
        .commit();

    out.to_hbm(&mut ctx.tdma)
}
