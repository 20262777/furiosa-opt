//! Rung 06b `reduce_sum_ragged`: row-wise sum over a padded reduction axis.
//!
//! Regenerate the ground truth first:
//!     python src/ops/reduce_sum_ragged/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::reduce_sum_ragged::{RP, S, reduce_sum_ragged_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/reduce_sum_ragged/reduce_sum_ragged.safetensors");

/// Contract satisfied: the tail holds the Add identity (0.0), so the kernel's sum over
/// all `RP` slots equals the sum over the 999 real ones.
#[tokio::test]
async fn test_reduce_sum_ragged() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![S, RP]>::from_safetensors(&gt.tensor("x").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![S]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let x_hbm = x.to_hbm(&mut ctx.pdma).await;

    let got = launch(reduce_sum_ragged_kernel, (&mut *ctx, &x_hbm))
        .await
        .to_host::<m![S]>(&mut ctx.pdma)
        .await
        .into_vec();

    // A 999-term sum of N(0,1) has std ~31.6; the absolute floor is generous but still
    // ~300x below the sentinel the sibling test uses.
    common::assert_close(&expected, &got, 2e-2, 5e-1);
}

/// Contract violated: a `1e4` sentinel in the tail. The kernel has no way to exclude it,
/// so the result must equal the sum *including* the sentinel.
///
/// This is a guard, not a feature: it proves the tail is genuinely summed, so the
/// caller contract in the kernel docs is load-bearing rather than decorative. If this
/// ever starts matching the clean expectation, the kernel has silently changed
/// semantics.
#[tokio::test]
async fn test_reduce_sum_ragged_tail_is_summed() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let x =
        HostTensor::<bf16, m![S, RP]>::from_safetensors(&gt.tensor("x_poisoned").unwrap()).unwrap();
    let expected =
        HostTensor::<bf16, m![S]>::from_safetensors(&gt.tensor("expected_poisoned").unwrap())
            .unwrap()
            .into_vec();

    let x_hbm = x.to_hbm(&mut ctx.pdma).await;

    let got = launch(reduce_sum_ragged_kernel, (&mut *ctx, &x_hbm))
        .await
        .to_host::<m![S]>(&mut ctx.pdma)
        .await
        .into_vec();

    common::assert_close(&expected, &got, 2e-2, 5e-1);
}
