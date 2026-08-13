//! Rung 08 `softmax`: row-wise softmax over a padded axis.
//!
//! Regenerate the ground truth first:
//!     python src/ops/softmax/ground_truth.py

mod common;

use furiosa_opt_std::prelude::*;
use rngd_kernels::ops::softmax::{R_REAL, RP, S, softmax_kernel};
use safetensors::SafeTensors;

const DATA: &[u8] = include_bytes!("../data/softmax/softmax.safetensors");

#[tokio::test]
async fn test_softmax() {
    let gt = SafeTensors::deserialize(DATA).unwrap();
    let mut ctx = Context::acquire();

    let x = HostTensor::<bf16, m![S, RP]>::from_safetensors(&gt.tensor("x").unwrap()).unwrap();
    let expected = HostTensor::<bf16, m![S, RP]>::from_safetensors(&gt.tensor("expected").unwrap())
        .unwrap()
        .into_vec();

    let x_hbm = x.to_hbm(&mut ctx.pdma).await;

    let got = launch(softmax_kernel, (&mut *ctx, &x_hbm))
        .await
        .to_host::<m![S, RP]>(&mut ctx.pdma)
        .await
        .into_vec();

    let rp = <m![RP]>::SIZE;

    // Under `--backend typecheck` the tensors are phantom and `got` is empty, so the
    // index-based guards below have nothing to walk. `assert_close` already tolerates
    // this (both lengths are 0 and the loop is empty); the guards need it said explicitly.
    if !got.is_empty() {
        // Guard 1: every row must sum to 1. This is the defining property of softmax and is
        // independent of the reference — a broken max, exp or reciprocal all break it.
        for s in 0..<m![S]>::SIZE {
            let sum: f32 = got[s * rp..(s + 1) * rp].iter().map(|v| v.to_f32()).sum();
            assert!((sum - 1.0).abs() <= 2e-2, "row {s} sums to {sum}, not 1.0");
        }

        // Guard 2: padded slots must be exactly zero. If the sentinel failed to underflow
        // through exp, these would be non-zero and the row sums above would also be wrong.
        for s in 0..<m![S]>::SIZE {
            for r in R_REAL..rp {
                let v = got[s * rp + r].to_f32();
                assert_eq!(v, 0.0, "padded slot [{s}, {r}] is {v}, not 0");
            }
        }
    }

    // Values are ~1/999, so the default 1e-2 absolute floor would make the comparison
    // vacuous. Tighten it to well below one bf16 ULP at that magnitude.
    common::assert_close(&expected, &got, 2e-2, 1e-5);
}
