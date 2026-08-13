//! Low-level vISA kernel library for FuriosaAI RNGD (Tensor Contraction Processor).
//!
//! Each operation is developed as a *rung* on a ladder, from `add` up to a working
//! attention block, and every rung carries its own spec, torch ground truth, test,
//! schedule dump and run report. See `PIPELINE.md` for the development cycle and the
//! full ladder.
//!
//! Built against the `furiosa-opt` Virtual ISA. This package is standalone: it does not
//! depend on the `base-template` scaffold, which stays untouched in `../lab/`.

#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod ops;
