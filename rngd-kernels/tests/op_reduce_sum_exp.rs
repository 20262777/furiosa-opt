//! Rung 07b `reduce_sum_exp`: row-wise sum(exp(x)) over a padded axis.
//!
//! Regenerate the ground truth first:
//!     python src/ops/reduce_sum_exp/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::reduce_sum_exp::{RP, S, reduce_sum_exp_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/reduce_sum_exp/reduce_sum_exp.safetensors");

#[tokio::test]
async fn test_reduce_sum_exp() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![S, RP]>::from_safetensors(&gt.tensor("x").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![S]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let x_hbm = x.to_hbm(&mut ctx.pdma).await;

    let got = launch(reduce_sum_exp_kernel, (&mut *ctx, &x_hbm))
        .await
        .to_host::<m![S]>(&mut ctx.pdma)
        .await
        .into_vec();

    // If the sentinel failed to underflow, every row would be +inf rather than ~1650,
    // so a finiteness check is a cheap and decisive guard on the padding contract.
    for (i, g) in got.iter().enumerate() {
        assert!(
            g.to_f32().is_finite(),
            "row {i} is not finite ({}): exp(sentinel) did not underflow to 0",
            g.to_f32()
        );
    }

    common::assert_close_default(&expected, &got);
}
