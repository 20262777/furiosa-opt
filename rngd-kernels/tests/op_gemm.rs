//! Rung 05 `gemm`: compares the kernel against the torch ground truth.
//!
//! Regenerate the ground truth first:
//!     python src/ops/gemm/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::gemm::{I, J, K, gemm_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/gemm/gemm.safetensors");

#[tokio::test]
async fn test_gemm() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let a = HostTensor::<bf16, m![I, K]>::from_safetensors(&gt.tensor("a").unwrap()).unwrap();
    let b = HostTensor::<bf16, m![J, K]>::from_safetensors(&gt.tensor("b").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![I, J]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let a_hbm = a.to_hbm(&mut ctx.pdma).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma).await;

    let got = launch(gemm_kernel, (&mut *ctx, &a_hbm, &b_hbm))
        .await
        .to_host::<m![I, J]>(&mut ctx.pdma)
        .await
        .into_vec();

    common::assert_close_default(&expected, &got);
}
