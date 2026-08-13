//! Rung 07b: row-wise `sum(exp(x))` over a padded axis — softmax's denominator.
//!
//! `out[s] = sum_r exp(x[s, r])` with `S = 256` rows and `RP = 1000` slots per row, of
//! which `R_REAL = 999` carry data.
//!
//! # The question this rung answers
//!
//! Rung 06b established that a padded reduce axis does not lower to vISA, and that the
//! shippable alternative is to fill the tail with the reduce operation's **identity
//! element**. That works for `Add` (`0`), `Max` (`-inf`), `Min` (`+inf`) — but softmax
//! reduces `sum(exp(x))`, and the received wisdom is that no `p` satisfies `exp(p) = 0`,
//! so the trick supposedly cannot apply.
//!
//! **That reasoning holds in exact arithmetic and fails in floating point.** `exp`
//! underflows: any `p` below about `-88` gives `0` in `f32`, and `-3.39e38` (representable
//! in `bf16`, which shares `f32`'s 8-bit exponent) gives exactly `0`. So the identity
//! element for the *composed* operation `x -> exp(x) -> sum` is simply a large negative
//! number.
//!
//! Which is what masked softmax already does anyway: `transformer/attention/softmax.rs`
//! upstream fills masked positions with `-3.3895314e38` for exactly this reason. Padding
//! and masking turn out to be the same mechanism.
//!
//! See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![
    S = 256,   // rows, one per slice
    RP = 1000, // padded reduction length; slots >= R_REAL must be NEG_SENTINEL
];

/// Number of leading slots of `RP` that carry real data.
pub const R_REAL: usize = 999;

/// The additive identity *after* `exp`. `exp(-3.39e38)` underflows to exactly `0.0`.
/// Same constant upstream's masked softmax uses for masked positions.
pub const NEG_SENTINEL: f32 = -3.3895314e38;

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![S];
pub type Element = m![RP]; // no `#` padding: a padded reduce axis does not lower

/// `out[s] = sum_r exp(x[s, r])`, correct only if `x[s, r >= R_REAL] == NEG_SENTINEL`.
#[device(chip = 1)]
pub fn reduce_sum_exp_kernel(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![S, RP]>,
) -> HbmTensor<bf16, Chip, m![S]> {
    let x: DmTensor<bf16, Chip, Cluster, Slice, Element> = x.to_dm(&mut ctx.tdma);

    let sums: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 4]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![RP / 4], m![RP % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![RP / 4], m![RP % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![RP % 4]>()
        // FpExp. The sentinel underflows to 0 here, which is what makes the tail
        // contribute nothing to the sum below.
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_intra_slice_reduce::<RP, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit();

    sums.to_hbm(&mut ctx.tdma)
}
