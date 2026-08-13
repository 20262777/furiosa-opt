# Kernel Development Pipeline

A repeatable cycle for taking one operation from "I want this op" to "it passes and I
know what it costs", climbing from `add` up to a working attention block.

Everything for one op lives in **one folder**: `src/ops/<op>/`.

## The cycle

```
 1. select        pick the next rung on the LADDER
 2. spec          write src/ops/<op>/spec.md   — math, shapes, mapping plan
 3. ground truth  write src/ops/<op>/ground_truth.py  → data/<op>/<op>.safetensors
 4. kernel        write src/ops/<op>/mod.rs    — the #[device] fn
 5. check         ./pipeline.sh check <op>     — mapping errors only, ~1s
 6. test          ./pipeline.sh test <op>      — numeric compare vs ground truth
 7. schedule      ./pipeline.sh schedule <op>  → results/<op>/schedule.json
 8. report        ./pipeline.sh report <op>    → results/<op>/report.md
```

Steps 5–8 are one command:

```bash
./pipeline.sh all add
```

**Do step 5 before step 6.** Every real bug we have hit so far was a mapping
assertion, and `--backend typecheck` catches those in under a second while a full
emulation run costs minutes.

## Layout

Per the official docs (`introduction.md`), a kernel package must keep:

- `[package.metadata.furiosa-opt]` in `Cargo.toml` — opts into kernel compilation
- every `#[device]` fn under `src/`, reachable from `src/lib.rs`
- host programs as direct `src/*.rs` files with explicit `[[bin]] path = "src/<name>.rs"`
- **never** `src/bin/`, `examples/` — the rustc plugin skips those

Integration tests in `tests/` are fine: they only *call* kernels that live in `src/`,
which is exactly what `furiosa-opt-examples` does.

```
rngd-kernels/
├── pipeline.sh                    # the driver
├── PIPELINE.md                    # this file
├── src/
│   ├── lib.rs                     # pub mod ops;
│   └── ops/
│       ├── mod.rs
│       └── <op>/
│           ├── mod.rs             # the #[device] kernel
│           ├── spec.md            # math, shapes, mapping plan, notes
│           └── ground_truth.py    # torch reference → data/<op>/<op>.safetensors
├── tests/
│   └── op_<op>.rs                 # loads the safetensors, launches, compares
├── data/<op>/<op>.safetensors     # generated, gitignored
└── results/<op>/
    ├── schedule.json              # for furiosa-schedule-viewer
    └── report.md                  # pass/fail, cycles, observations
```

## The ladder

Each rung adds exactly one concept. Do not skip: rung N's concept is assumed by N+1.

| # | op | new concept | status |
|---|---|---|---|
| 00 | `const_add` | Vector Engine, single stream (`../lab/src/kernel/constant_add_kernel.rs`) | template |
| 01 | **`add`** | `begin_interleaved` → `unzip` → `clip_zip` — binary op on two DM tensors | **done** |
| 02 | **`mul`** | VRF operand + Float cluster (4-way) → `narrow`/`widen` | **done** |
| 03 | **`dot`** | Contraction Engine: packet tree + time accumulator | **done** — WBS 18 |
| 04 | **`gemv`** | output axis in `Slice`; DMA broadcast of the stationary operand | **done** — WBS 18 |
| 05 | **`gemm`** | both output axes in `Slice`, `Lane` carries `J`; 100% MAC utilisation | **done** — WBS 18 |
| 06 | **`reduce_sum`** | `vector_intra_slice_reduce` + `vector_inter_slice_reduce` | **done** |
| 06b | **`reduce_sum_ragged`** | row-wise reduce over a padded axis (identity-element contract) | **done** |
| 07 | `rmsnorm` | `vector_stash` for `x*x`, multi-pass, VRF scalars, ALU budgeting | todo |
| 08 | `softmax` | max → exp → sum → div; `TagMode::Comparison` for masking | todo |
| 09 | `qk_matmul` | batched matmul, GQA broadcast of KV heads | todo |
| 10 | `attn_output` | score × V matmul | todo |
| 11 | `attention` | 09 → 08 → 10 composed | **goal** |

Rungs 03–05 are ported from the `base-template` scaffold (the originals stay untouched in
`../lab/src/kernel/`) and now run under the pipeline with torch ground truth, schedules
and reports. Together they are the **WBS 18 — 기본 행렬 곱 연산 커널** deliverable.

This package is **standalone**: it does not depend on `../lab/`, which is kept only as a
pristine reference copy of the template.

> `data/` is gitignored, so a fresh clone has no ground truth and `include_bytes!` will
> fail until it is regenerated. `pipeline.sh all` / `ladder` run `gt` as their first
> step, so CI is fine; a bare `cargo test` is not. Run `./pipeline.sh gt <op>` first.

Copy-paste is expected and encouraged — `src/ops/add/` is the template. What must stay
separate is the *folder, data, results and docs* for each op, so every rung has its own
reproducible record.

## Known environment gaps

- **`cargo furiosa-opt compile` fails at the final LIR → C step** with
  `bits/wordsize.h: No such file or directory`. The host is x86_64 but
  `/usr/include/x86_64-linux-gnu` is missing, i.e. `libc6-dev` is incomplete in this
  image. Fix with `apt install libc6-dev` (the README also wants `libclang-dev` and
  `gcc-aarch64-linux-gnu`).
  **This does not block the pipeline**: `--dump-schedule` writes its JSON *before* that
  step, so `results/<op>/schedule.json` is still valid and viewer-ready. `pipeline.sh`
  judges the step on the artefact, not the exit code.
- `--backend npu` needs the Furiosa SDK and physical hardware. `typecheck` and
  `emulation` need neither, and both are what the pipeline uses.

## Two different `compile` failures — tell them apart

`./pipeline.sh schedule` can fail in two ways, and only one of them is environmental:

| symptom | stage | meaning |
|---|---|---|
| `schedule.json` **is** produced, log ends `lir: failed to compile ... .c` | codegen, after the dump | environmental (`libc6-dev`); the kernel lowered fine |
| **no** `schedule.json`, log says `visa: while lowering ...` | vISA lowering, before the dump | **the kernel is not expressible on hardware as written** |

The second is a real finding, not a setup problem. `reduce_sum_ragged` hit it
(`reduce axis should not have padding`) despite passing the numeric test, and had to be
redesigned — see its `spec.md`. A rung that passes `emulation` but fails lowering **has
not been validated for hardware**, so treat a missing `schedule.json` as a red flag, not noise.

## Reference material

- Book: `../furiosa-opt/docs/src/` (`mdbook serve` or read the markdown directly)
- Worked kernels: `../furiosa-opt/furiosa-opt-examples/src/`
  - `matmul/matmul_4096.rs` — contraction + inter-slice reduce
  - `matmul/matmul_split_reduce.rs` — tiling, accumulation, `commit_view`
  - `mnist/mod.rs` — composition, Switch/Transpose relayout
  - `transformer/common/norm.rs` — RMSNorm, multi-pass, stash
  - `transformer/attention/softmax.rs` — masked softmax (3-pass, materialized)
- Schedule viewer: `cargo install furiosa-schedule-viewer && furiosa-schedule-viewer`
