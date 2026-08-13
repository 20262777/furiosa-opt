//! Rung 04 `gemv`: compares the kernel against the torch ground truth.
//!
//! Regenerate the ground truth first:
//!     python src/ops/gemv/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::gemv::{I, J, gemv_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/gemv/gemv.safetensors");

#[tokio::test]
async fn test_gemv() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let matrix =
        HostTensor::<bf16, m![I, J]>::from_safetensors(&gt.tensor("matrix").unwrap()).unwrap();
    let vector =
        HostTensor::<bf16, m![J]>::from_safetensors(&gt.tensor("vector").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![I]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let matrix_hbm = matrix.to_hbm(&mut ctx.pdma).await;
    let vector_hbm = vector.to_hbm(&mut ctx.pdma).await;

    let got = launch(gemv_kernel, (&mut *ctx, &matrix_hbm, &vector_hbm))
        .await
        .to_host::<m![I]>(&mut ctx.pdma)
        .await
        .into_vec();

    common::assert_close_default(&expected, &got);
}
