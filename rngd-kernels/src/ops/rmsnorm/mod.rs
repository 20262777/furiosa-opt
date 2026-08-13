//! Rung 07: RMS normalisation.
//!
//! `y[s, h] = x[s, h] * rsqrt(mean_h(x[s, h]^2) + eps) * weight[h]`
//!
//! with `S = 256` rows (one per slice) and `H = 1024` hidden features per row.
//!
//! New concept: **one mathematical op does not fit in one Tensor Unit pass.** Each ALU
//! may fire at most once per invocation, and the chain's stages run in a fixed order, so
//! this needs four:
//!
//! | pass | context | does | ALUs |
//! |---|---|---|---|
//! | 1 | sub | `weight` -> VRF | — |
//! | 2 | main | `sum(x*x) / H + eps` -> DM | `FpMul0`, reduce tree, `FpDiv`, `ClipAdd` |
//! | 3 | main | `1 / sqrt(var)` -> VRF | `FpFpu`, `FpDiv` |
//! | 4 | main | `x * inv_rms * weight` -> DM | `FpMul0`, `FpMul1` |
//!
//! Pass 2 is where `vector_stash` earns its place: there is no square op, so the stream
//! is snapshotted and multiplied by its own snapshot.
//!
//! Pass 4 fits two multiplies in one invocation only because the Float cluster exposes
//! `FpMul0` **and** `FpMul1` as separate ALUs — route both through `Mul0` and it panics
//! with "already in use".
//!
//! Modelled on `furiosa-opt-examples/src/transformer/common/norm.rs`, simplified: `H`
//! lives entirely inside one slice here, so the variance needs no inter-slice reduce.
//!
//! See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![
    S = 256,  // rows, one per slice
    H = 1024, // hidden features per row
];

/// Added before the reciprocal square root, for numerical stability.
pub const EPS: f32 = 1.0e-6;

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![S]; // 256 rows across 256 slices, exactly
pub type Element = m![H]; // 1024 bf16 = 2 KB per row

/// Per-row scalar produced by pass 2 and consumed by pass 3.
type RowScalar = DmTensor<f32, Chip, Cluster, Slice, m![1 # 4]>;

/// `y[s, h] = x[s, h] * rsqrt(mean(x[s, :]^2) + eps) * weight[h]`
#[device(chip = 1)]
pub fn rmsnorm_kernel(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![S, H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![S, H]> {
    // `x` has an S axis, so S moves into Slice: row s -> slice s (distribution).
    let x: DmTensor<bf16, Chip, Cluster, Slice, Element> = x.to_dm(&mut ctx.tdma);
    // `weight` has no S axis, so the Slice mapping's S becomes a stride-0 entry:
    // every slice gets its own full copy (broadcast), same rule as gemv's vector.
    let weight: DmTensor<bf16, Chip, Cluster, Slice, Element> = weight.to_dm(&mut ctx.tdma);

    // ---- pass 1 (sub): weight -> VRF -------------------------------------------------
    // Runs concurrently with pass 2, which does not read it.
    // `.to_vrf()` flattens `Element = [Time, Packet]`, so any collect padding lands in
    // the VRF too. An 8-element packet of f32 is exactly one 32 B flit, so nothing is
    // padded and the VRF holds a clean `m![H]` = 4 KB. Using a 4-element packet instead
    // would store `[H / 4, H % 4 # 8]` = 2048 slots = the entire 8 KB budget, leaving no
    // room for `inv_rms_vrf`.
    let weight_vrf: VrfTensor<f32, Chip, Cluster, Slice, Element> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .to_vrf();

    // ---- pass 2 (main): variance ------------------------------------------------------
    let variance: RowScalar = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 4], m![H % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 -> Way4; the upper 4 slots are collect padding.
        .vector_narrow_trim::<m![H % 4]>()
        // No square op exists. Snapshot the stream, then multiply it by its own
        // snapshot one stage later.
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        // `H` lives entirely in Time and Packet (never Slice), so the intra-slice
        // reducer collapses all of it — no inter-slice reduce needed here.
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        // sum -> mean. Dedicated divider, not a Float-cluster ALU.
        // The `const { }` is mandatory: a vISA scalar operand is encoded in the
        // instruction, so it must const-fold. A bare `H::SIZE as f32` folds the SIZE but
        // leaves the cast in MIR and is rejected at translation — even though typecheck
        // and emulation both accept it. See `spec.md`.
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        // + eps, on ClipAdd (the full-rate 8-way f32 add).
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 4]>() // 4 f32 = 16 B write unit
        .commit();

    // ---- pass 3 (main): reciprocal square root -> VRF ----------------------------------
    let inv_rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(variance.view())
        .fetch::<m![1], m![1 # 4]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt) // FpFpu
        // Mode10 swaps the slots so this is `constant / stream`, i.e. 1 / sqrt(var).
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf();

    // ---- pass 4 (main): scale ----------------------------------------------------------
    let out: DmTensor<bf16, Chip, Cluster, Slice, Element> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 4], m![H % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 4], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H % 4]>()
        // Two multiplies in one pass: different ALUs, so they coexist.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_pad::<m![H % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();

    out.to_hbm(&mut ctx.tdma)
}
