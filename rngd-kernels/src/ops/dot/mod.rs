//! Rung 03: dot product.
//!
//! `out = sum_i lhs[i] * rhs[i]` over `A = 2048` `bf16` elements.
//!
//! First rung to use the **Contraction Engine**. One operand streams from DM, the
//! other sits stationary in the TRF, and the reduction is split across two of the
//! contraction stages:
//!
//! ```text
//! A = 2048 : 32 (Packet, adder tree) x 64 (Time, accumulator) x 1 (Lane)
//! ```
//!
//! Ported from the base-template `dot_product_kernel.rs`. See `spec.md`.

use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![1 # 256]; // 1 active slice; m![A / 8 # 256] would distribute
pub type Time = m![1]; // no temporal iteration at fetch
pub type Lane = m![1]; // no lane parallelism

/// `out = lhs . rhs`
#[device(chip = 1)]
pub fn dot_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {
    // HBM -> DM
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = lhs.to_dm(&mut ctx.tdma);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = rhs.to_dm(&mut ctx.tdma);

    // Sub context: park `rhs` in the TRF as the stationary operand.
    // `TrfAddress::Full` (the `.to_trf()` default) dedicates the whole TRF to it.
    let rhs: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![A]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![{ Time }, A / 16], m![A % 16]>()
        .to_trf();

    // Main context: stream `lhs` through the Contraction Engine, reduce along A.
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![A / 16], m![A % 16]>()
        // OutPacket = 32 bf16 = 64 B => PackSize 2: pairs consecutive 32 B flits,
        // halving the time steps (A/16 = 128 -> A/32 = 64) and filling all MACs.
        .contract_outer::<m![A / 32], m![A % 32], _, _, _>(&rhs)
        // Packet Reducer: depth-5 tree sums the 32 products in each packet.
        .contract_packet::<m![1]>()
        // Time Reducer: all 64 partial sums accumulate into one f32 slot.
        .contract_time::<m![1]>()
        // Lane Folder: trivial at Lane = m![1]; the scalar lands in bus position 0.
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        // f32 accumulator -> bf16, repadded to one 32 B flit.
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 8]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
