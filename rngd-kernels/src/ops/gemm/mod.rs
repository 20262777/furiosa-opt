//! Rung 05: matrix-matrix product.
//!
//! `C = A B^T`, einsum `IK, JK -> IJ`, with `I = J = 512` and `K = 64`.
//! Note `B` is stored **transposed** (`m![J, K]`), so `K` is innermost in both operands.
//!
//! New concept vs. `gemv`: `Slice` carries **both** output dimensions as a 16 x 16 grid,
//! and `Lane` carries part of `J`. This is the first rung where every hardware axis does
//! useful work — full 8 lanes, `PackSize = 2`, no padding anywhere:
//!
//! ```text
//! multiplies issued = 256 steps x 8 lanes x 32 packet x 256 slices = 16,777,216
//! I x J x K         = 512 x 512 x 64                               = 16,777,216
//! ```
//!
//! Ported from the base-template `gemm_kernel.rs`. See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![I = 512, J = 512, K = 64];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
/// 16 x 16 = 256 slices; each slice owns a **32 x 32** output tile (`I % 32` x `J % 32`).
pub type Slice = m![I / 32, J / 32];
/// All 8 lanes, indexed by the low 3 bits of `J`.
pub type Lane = m![J % 8];

/// `C[i, j] = sum_k a[i, k] * b[j, k]`
#[device(chip = 1)]
pub fn gemm_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![I, K]>,
    b: &HbmTensor<bf16, Chip, m![J, K]>,
) -> HbmTensor<bf16, Chip, m![I, J]> {
    // Both destinations carry the same 2-D slice grid, and each source is missing one of
    // its axes, so each gets broadcast along the axis it lacks (stride-0 DMA entries):
    //   a has no J -> replicated across the 16 J-columns of the grid
    //   b has no I -> replicated across the 16 I-rows of the grid
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![I % 32, K]> = a.to_dm(&mut ctx.tdma);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![J % 32, K]> = b.to_dm(&mut ctx.tdma);

    // Sub context: B into the TRF. `J % 8` is the OUTERMOST fetch time factor because
    // `.to_trf()` peels `Lane` off the outermost factor - that is what gives each lane
    // its own distinct set of B rows.
    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J / 8 % 4, K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![J % 8, J / 8 % 4], m![K]>()
        .collect::<m![J % 8, J / 8 % 4, K / 16], m![K % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![I % 32, J % 32]> = ctx
        .main
        .begin(a.view())
        // `J / 8 % 4` is not an axis of A -> stride-0 broadcast: each A row is re-read
        // 4 times, once per j-group. The temporal half of "A broadcasts across J".
        .fetch::<m![I % 32, J / 8 % 4], m![K]>()
        .collect::<m![I % 32, J / 8 % 4, K / 16], m![K % 16]>()
        // 64 B packet => PackSize 2. The A packet is lane-broadcast to all 8 lanes and
        // each lane multiplies it against its own B rows.
        .contract_outer::<m![I % 32, J / 8 % 4, K / 32], m![K % 32], _, _, _>(&b_trf)
        .contract_packet::<m![1]>() // depth-5 tree over the 32 K values in the packet
        // `K / 32` is innermost in Time, so InnerTime = 1: a single accumulator slot.
        .contract_time::<m![I % 32, J / 8 % 4]>()
        // Interleaved: Lane folds into OutPacket -> a full 8-wide flit, no empty slots.
        .contract_lane::<m![I % 32, J / 8 % 4], m![J % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![J % 8 # 16]>()
        .commit_trim::<m![J % 8]>() // 8 bf16 = 16 B write unit
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
