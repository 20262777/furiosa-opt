//! Rung 03 `dot`: compares the kernel against the torch ground truth.
//!
//! Regenerate the ground truth first:
//!     python src/ops/dot/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::dot::{A, dot_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/dot/dot.safetensors");

#[tokio::test]
async fn test_dot() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let lhs = HostTensor::<bf16, m![A]>::from_safetensors(&gt.tensor("lhs").unwrap()).unwrap();
    let rhs = HostTensor::<bf16, m![A]>::from_safetensors(&gt.tensor("rhs").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![1]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

    let got = launch(dot_kernel, (&mut *ctx, &lhs_hbm, &rhs_hbm))
        .await
        .to_host::<m![1]>(&mut ctx.pdma)
        .await
        .into_vec();

    common::assert_close_default(&expected, &got);
}
