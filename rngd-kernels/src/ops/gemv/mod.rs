//! Rung 04: matrix-vector product.
//!
//! `y = A x`, einsum `IJ, J -> I`, with `I = 256` rows and `J = 2048` columns.
//!
//! New concept vs. `dot`: the **output axis lives in `Slice`**, so all 256 slices
//! compute a different output element at once, and the stationary operand has to be
//! **broadcast** to every slice. That broadcast is done by the DMA (an axis present in
//! the destination `Slice` but absent from the source buffer becomes a stride-0
//! sequencer entry) — not by the Switch Engine.
//!
//! Ported from the base-template `gemv_kernel.rs`. See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![I = 256, J = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![I]; // output rows across slices, exactly 256
pub type Time = m![J / 32]; // 64 temporal steps over the reduction axis
pub type Packet = m![J % 32]; // 32 elements reduced spatially per step
pub type Lane = m![1];

/// `y[i] = sum_j matrix[i, j] * vector[j]`
#[device(chip = 1)]
pub fn gemv_kernel(
    ctx: &mut Context,
    matrix: &HbmTensor<bf16, Chip, m![I, J]>,
    vector: &HbmTensor<bf16, Chip, m![J]>,
) -> HbmTensor<bf16, Chip, m![I]> {
    // `matrix` has an I axis, so I moves into Slice: row i -> slice i (distribution).
    let matrix: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = matrix.to_dm(&mut ctx.tdma);
    // `vector` has no I axis, so the Slice mapping's I becomes a stride-0 entry:
    // every slice receives a full private copy of x (broadcast).
    let vector: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = vector.to_dm(&mut ctx.tdma);

    // Sub context: x becomes the stationary operand, replicated per slice.
    let vector_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J]> = ctx
        .sub
        .begin(vector.view())
        .fetch::<m![1], m![J]>()
        .collect::<m![J / 16], m![J % 16]>()
        .to_trf();

    // Main context: each slice streams its own row of A and contracts along J.
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 4]> = ctx
        .main
        .begin(matrix.view())
        // 32 B packets directly — no multi-read, unlike `dot`.
        .fetch::<m![J / 16], m![J % 16]>()
        .collect::<m![J / 16], m![J % 16]>() // identity: already one flit
        .contract_outer::<Time, Packet, _, _, _>(&vector_trf)
        .contract_packet::<m![1]>() // depth-5 tree over 32
        .contract_time::<m![1]>() // accumulate 64 -> y[i]
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>() // 4 bf16 = 8 B, the minimum write unit
        .commit();

    // Slice collapses back into the HBM element mapping: 256 per-slice scalars
    // gathered into one contiguous vector.
    result.to_hbm(&mut ctx.tdma)
}
