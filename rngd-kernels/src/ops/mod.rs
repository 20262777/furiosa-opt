//! Pipeline operations, one module per rung of the ladder in `../../PIPELINE.md`.
//!
//! Each `<op>/` folder holds its kernel (`mod.rs`), its spec (`spec.md`) and its
//! ground-truth generator (`ground_truth.py`). Tests live in `tests/op_<op>.rs`,
//! generated data in `data/<op>/`, and results in `results/<op>/`.
//!
//! | rung | module | WBS |
//! |---|---|---|
//! | 01 | [`add`] | — (warm-up) |
//! | 02 | [`mul`] | 19 LLM 추론 연산 커널 (VRF operand, narrow/widen) |
//! | 03 | [`dot`] | 18 기본 행렬 곱 연산 커널 |
//! | 04 | [`gemv`] | 18 기본 행렬 곱 연산 커널 |
//! | 05 | [`gemm`] | 18 기본 행렬 곱 연산 커널 |
//! | 06 | [`reduce_sum`] | 19 LLM 추론 연산 커널 (intra + inter slice reduce) |
//! | 06b | [`reduce_sum_ragged`] | 19 LLM 추론 연산 커널 (padded reduce axis) |

pub mod add;
pub mod dot;
pub mod gemm;
pub mod gemv;
pub mod mul;
pub mod reduce_sum;
pub mod reduce_sum_ragged;
