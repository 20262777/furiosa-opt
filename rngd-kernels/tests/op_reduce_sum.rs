//! Rung 06 `reduce_sum`: compares the kernel against the torch ground truth.
//!
//! Regenerate the ground truth first:
//!     python src/ops/reduce_sum/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::reduce_sum::{A, reduce_sum_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/reduce_sum/reduce_sum.safetensors");

#[tokio::test]
async fn test_reduce_sum() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![A]>::from_safetensors(&gt.tensor("x").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![1]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let x_hbm = x.to_hbm(&mut ctx.pdma).await;

    let got = launch(reduce_sum_kernel, (&mut *ctx, &x_hbm))
        .await
        .to_host::<m![1]>(&mut ctx.pdma)
        .await
        .into_vec();

    // A 8192-term sum: the kernel's tree/accumulator/ring order differs from torch's
    // pairwise sum, so a couple of bf16 ULP of drift is expected and legitimate.
    common::assert_close(&expected, &got, 2e-2, 5e-1);
}
