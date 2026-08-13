//! Rung 06b: row-wise sum over a padded reduction axis.
//!
//! `out[s] = sum_r x[s, r]` with `S = 256` rows and a reduction axis of `RP = 1000`
//! slots, of which only the first `R_REAL = 999` carry data.
//!
//! Two things change vs. rung 06 `reduce_sum`:
//!
//! 1. **Row-wise, not scalar.** `S` lives in `Slice` and never enters a reducer, so each
//!    slice produces its own sum. This is the shape `rmsnorm` and `softmax` need.
//! 2. **A reduction length that is not hardware-friendly**, handled by padding up to
//!    `RP` and requiring the tail to hold the reduce operation's **identity element**
//!    (`0.0` for `Add`).
//!
//! # Why not the Valid Count Generator
//!
//! The book's `vcg.md` documents exactly this situation: declare the axis as `R # RP`
//! and let the compiler configure the Valid Count Generator to mask the pad. That
//! **passes emulation and does not lower to vISA** — see `spec.md` for the two distinct
//! lowering errors. Until that is resolved upstream, the identity-element contract below
//! is the only form that reaches hardware.
//!
//! # Caller contract
//!
//! `x[s, r]` for `r >= R_REAL` **must** hold `0.0`. The kernel cannot check this, and a
//! non-zero tail silently corrupts every row. This precondition does **not** generalise:
//! it works for `Add`/`Max`/`Min` (identities `0`, `-inf`, `+inf`) but **not** for
//! `sum(exp(x))`, since no value `p` satisfies `exp(p) = 0` — so softmax needs a
//! different answer.
//!
//! See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![
    S = 256,   // rows, one per slice
    RP = 1000, // padded reduction length; slots >= R_REAL must be the identity element
];

/// Number of leading slots of `RP` that carry real data.
pub const R_REAL: usize = 999;

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![S]; // 256 rows across 256 slices, exactly
pub type Element = m![RP]; // 1000 slots per row - NOT a padded mapping

/// `out[s] = sum_r x[s, r]`, correct only if `x[s, r >= R_REAL] == 0.0`.
#[device(chip = 1)]
pub fn reduce_sum_ragged_kernel(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![S, RP]>,
) -> HbmTensor<bf16, Chip, m![S]> {
    // Row s -> slice s. Each slice holds 1000 bf16 = 2000 B (8-byte aligned).
    let x: DmTensor<bf16, Chip, Cluster, Slice, Element> = x.to_dm(&mut ctx.tdma);

    let sums: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 4]> = ctx
        .main
        .begin(x.view())
        // 250 time steps x 4 elements. `RP` carries no `#` padding, which is what makes
        // this lowerable: the vISA lowering rejects a padded reduce axis.
        .fetch::<m![RP / 4], m![RP % 4]>()
        .fetch_cast::<f32>()
        // 4 x f32 = 16 B padded to the 32 B flit.
        .collect::<m![RP / 4], m![RP % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 -> Way4; the upper 4 slots are collect padding, so `_trim`.
        .vector_narrow_trim::<m![RP % 4]>()
        // Reduces every factor of `RP` in Time and Packet. The identity-valued tail
        // contributes nothing, which is why the caller contract matters.
        .vector_intra_slice_reduce::<RP, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        // NOTE: no inter-slice reduce. `S` lives in Slice and must survive - one sum per
        // row is the whole point.
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit();

    sums.to_hbm(&mut ctx.tdma)
}
