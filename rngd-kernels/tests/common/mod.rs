//! Shared test helpers for the op ladder.
//!
//! Every rung compares a kernel result against a torch-generated ground truth with the
//! same tolerance rule, so the reports are directly comparable across ops.

use furiosa_opt_std::prelude::*;

/// Element-wise comparison with a relative tolerance and an absolute floor.
///
/// `bf16` carries 8 mantissa bits, so one ULP is ~0.4% relative. Kernels accumulate in
/// `f32` and round once at the end, exactly as the ground truth does, so the only
/// expected difference is summation order — a couple of ULP at most. `rel = 2e-2` is
/// therefore loose enough to be stable and tight enough to catch a real mapping error.
///
/// Returns the worst absolute difference seen, for the run report.
#[allow(dead_code)] // each test binary compiles this module but uses only one helper
pub fn assert_close(expected: &[bf16], got: &[bf16], rel: f64, abs: f64) -> f64 {
    assert_eq!(expected.len(), got.len(), "output length mismatch");

    let mut worst = 0f64;
    let mut worst_at = 0usize;
    for (i, (e, g)) in expected.iter().zip(got).enumerate() {
        let (x, y) = (f64::from(e.to_f32()), f64::from(g.to_f32()));

        if x.is_nan() || y.is_nan() {
            assert!(x.is_nan() && y.is_nan(), "index {i}: expected {x}, got {y}");
            continue;
        }

        let diff = (x - y).abs();
        let tol = abs.max(x.abs() * rel);
        assert!(
            diff <= tol,
            "index {i}: expected {x}, got {y} (diff {diff:e} > tol {tol:e})"
        );
        if diff > worst {
            worst = diff;
            worst_at = i;
        }
    }

    // Printed into results/<op>/test.log and picked up by `pipeline.sh report`.
    println!(
        "max abs diff: {worst:e} (at index {worst_at}, n = {})",
        expected.len()
    );
    worst
}

/// Default tolerance: 2% relative, 1e-2 absolute floor.
#[allow(dead_code)] // each test binary compiles this module but uses only one helper
pub fn assert_close_default(expected: &[bf16], got: &[bf16]) -> f64 {
    assert_close(expected, got, 2e-2, 1e-2)
}
