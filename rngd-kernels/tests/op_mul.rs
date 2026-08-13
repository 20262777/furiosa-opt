//! Rung 02 `mul`: compares the kernel against the torch ground truth.
//!
//! Regenerate the ground truth first:
//!     python src/ops/mul/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::mul::{A, mul_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/mul/mul.safetensors");

#[tokio::test]
async fn test_mul() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let lhs = HostTensor::<bf16, m![A]>::from_safetensors(&gt.tensor("lhs").unwrap()).unwrap();
    let rhs = HostTensor::<bf16, m![A]>::from_safetensors(&gt.tensor("rhs").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![A]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

    let got = launch(mul_kernel, (&mut *ctx, &lhs_hbm, &rhs_hbm))
        .await
        .to_host::<m![A]>(&mut ctx.pdma)
        .await
        .into_vec();

    common::assert_close_default(&expected, &got);
}
