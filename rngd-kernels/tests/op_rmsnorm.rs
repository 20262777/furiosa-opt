//! Rung 07 `rmsnorm`: compares the kernel against the torch ground truth.
//!
//! Regenerate the ground truth first:
//!     python src/ops/rmsnorm/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::rmsnorm::{H, S, rmsnorm_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/rmsnorm/rmsnorm.safetensors");

#[tokio::test]
async fn test_rmsnorm() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![S, H]>::from_safetensors(&gt.tensor("x").unwrap()).unwrap();
    let weight =
        HostTensor::<bf16, m![H]>::from_safetensors(&gt.tensor("weight").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![S, H]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let x_hbm = x.to_hbm(&mut ctx.pdma).await;
    let weight_hbm = weight.to_hbm(&mut ctx.pdma).await;

    let got = launch(rmsnorm_kernel, (&mut *ctx, &x_hbm, &weight_hbm))
        .await
        .to_host::<m![S, H]>(&mut ctx.pdma)
        .await
        .into_vec();

    common::assert_close_default(&expected, &got);
}
