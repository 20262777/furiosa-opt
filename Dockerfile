# syntax=docker/dockerfile:1
# =============================================================================
#  furiosa-opt lab image
#
#  Everything needed to work through
#    - https://github.com/furiosa-ai/furiosa-opt
#    - https://developer.furiosa.ai/furiosa-opt/book
#  on a Slurm cluster via enroot/pyxis, with no NPU hardware
#  (the `emulation` and `typecheck` backends run entirely host-side).
#
#  Build:  podman build -t furiosa-opt:0.5.1 .      (rootless, on the cluster)
#          docker build -t furiosa-opt:0.5.1 .      (on your own machine)
#
#  x86_64 only. Upstream publishes cargo-furiosa-opt and the prebuilt static
#  libraries for x86_64-unknown-linux-gnu alone; arm64 is not supported.
# =============================================================================
FROM --platform=linux/amd64 ubuntu:22.04

# furiosa-mapping/furiosa-opt-lower crate version == GitHub release tag (vN.N.N)
ARG FURIOSA_OPT_VERSION=0.5.1
# ABI-locked: cargo-furiosa-opt is a rustc driver, valid only against this nightly
ARG RUST_TOOLCHAIN=nightly-2026-05-01
ARG RUST_TARGET=x86_64-unknown-linux-gnu
ARG FURIOSA_OPT_REPO=furiosa-ai/furiosa-opt

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System packages
#   libclang-dev          furiosa-opt-std/build.rs runs bindgen -> loads libclang
#   gcc-aarch64-linux-gnu aarch64-linux-gnu-{gcc,as,ld,objcopy} for NPU device binaries
# Ubuntu 22.04 is deliberate: cargo-furiosa-opt needs GCC_12.0.0 in libgcc_s,
# which jammy has but RHEL/Rocky 9 (the cluster's host OS) does not.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git make pkg-config \
        libclang-dev gcc-aarch64-linux-gnu \
        tar gzip xz-utils procps file less vim-tiny \
    && rm -rf /var/lib/apt/lists/*

ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib

# -----------------------------------------------------------------------------
# Rust toolchain
#
# NOTE ON PATHS: rustup's own default is $HOME/.rustup + $HOME/.cargo. That
# cannot work here. Under enroot/pyxis the container process runs as the
# *invoking cluster user's* UID with their real home bind-mounted over $HOME,
# so anything installed into a build-time home directory is either unreadable
# (mode 0700 /root) or masked by the mount. /usr/local/{rustup,cargo} is the
# layout the official `rust` Docker image uses for exactly this reason, and it
# is world-readable. This is the one path deviation in the image.
# -----------------------------------------------------------------------------
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# rustc-dev ships librustc_driver-<hash>.so and libLLVM, which furiosa-opt-driver
# links against. llvm-tools ships llvm-objcopy etc. used on the NPU codegen path.
# Neither is mentioned in the upstream README, but the driver will not start
# without rustc-dev.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal \
          --default-toolchain "${RUST_TOOLCHAIN}" \
          -c rustfmt -c clippy -c rustc-dev -c llvm-tools \
    && rustc --version && cargo --version

# furiosa-opt-driver has no RUNPATH, so the dynamic loader needs to be told
# where librustc_driver lives.
ENV LD_LIBRARY_PATH=/usr/local/rustup/toolchains/${RUST_TOOLCHAIN}-${RUST_TARGET}/lib

# -----------------------------------------------------------------------------
# cargo-furiosa-opt (+ helper tooling), installed the way the book prescribes
# -----------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSfL \
      https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh \
      | bash \
    && cargo binstall -y \
         "cargo-furiosa-opt@${FURIOSA_OPT_VERSION}" \
         cargo-generate \
         mdbook \
         mdbook-mermaid \
    && cargo furiosa-opt --help >/dev/null

# -----------------------------------------------------------------------------
# Prebuilt static libraries
#
# furiosa-mapping/build.rs and furiosa-opt-lower/build.rs curl a matching .a
# from the GitHub release on every cold build. Baking them in and exporting the
# two *_LOCAL_PREBUILT variables makes the build hermetic: a compute node with
# no outbound network still compiles.
# -----------------------------------------------------------------------------
ENV FURIOSA_PREBUILT_DIR=/opt/furiosa/prebuilt
RUN mkdir -p "${FURIOSA_PREBUILT_DIR}" && cd "${FURIOSA_PREBUILT_DIR}" \
    && base="https://github.com/${FURIOSA_OPT_REPO}/releases/download/v${FURIOSA_OPT_VERSION}" \
    && for f in \
         "SHA256SUMS" \
         "libfuriosa_mapping_impl-v${FURIOSA_OPT_VERSION}-${RUST_TARGET}.a" \
         "libfuriosa_mapping_impl-v${FURIOSA_OPT_VERSION}-${RUST_TARGET}.LICENSES.txt" \
         "libfuriosa_opt_lower_impl-v${FURIOSA_OPT_VERSION}-${RUST_TARGET}.a" \
         "libfuriosa_opt_lower_impl-v${FURIOSA_OPT_VERSION}-${RUST_TARGET}.LICENSES.txt" \
       ; do curl -fsSL -O "${base}/${f}"; done \
    && sha256sum --ignore-missing --check SHA256SUMS

ENV FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT=${FURIOSA_PREBUILT_DIR}/libfuriosa_mapping_impl-v${FURIOSA_OPT_VERSION}-${RUST_TARGET}.a \
    FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT=${FURIOSA_PREBUILT_DIR}/libfuriosa_opt_lower_impl-v${FURIOSA_OPT_VERSION}-${RUST_TARGET}.a

# -----------------------------------------------------------------------------
# Book source + the five worked Quick Start examples
#
# /opt/furiosa/furiosa-opt   full repo: `make test`, mdbook sources, examples
# /opt/furiosa/lab           base-template rendered as a ready-to-run project
# -----------------------------------------------------------------------------
# Pinned to the release tag, not main: the driver and the static libraries above
# are locked to ${FURIOSA_OPT_VERSION}, so the book and base-template must come
# from the matching commit. Cloning main would silently drift as upstream lands
# commits and pair a newer template with an older driver.
RUN git clone --depth 1 --branch "v${FURIOSA_OPT_VERSION}" \
      "https://github.com/${FURIOSA_OPT_REPO}.git" \
      /opt/furiosa/furiosa-opt

# USER is set only because cargo-generate refuses to run without it.
RUN cd /opt/furiosa \
    && USER=root cargo generate --path /opt/furiosa/furiosa-opt/base-template \
         --name furiosa-opt-lab --vcs none --destination /opt/furiosa \
    && mv /opt/furiosa/furiosa-opt-lab /opt/furiosa/lab

# Warm a shared target dir so the first `cargo furiosa-opt run` inside a job is
# incremental rather than a from-scratch dependency build.
ENV CARGO_TARGET_DIR=/opt/furiosa/target
RUN cd /opt/furiosa/lab \
    && cargo furiosa-opt build --release \
    && cargo furiosa-opt test --release --no-run \
    && cargo furiosa-opt --backend typecheck build --release

# mdbook needs its mermaid assets installed into the book dir before serving
RUN cd /opt/furiosa/furiosa-opt && mdbook-mermaid install docs

# Every path a cluster user touches must be writable by an arbitrary UID
# (enroot --rw / pyxis --container-writable give a private, ephemeral layer).
RUN chmod -R a+rwX /opt/furiosa /usr/local/cargo /usr/local/rustup

run curl -fsSL https://claude.ai/install.sh | bash && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

WORKDIR /opt/furiosa/lab
CMD ["/bin/bash", "-l"]

LABEL org.opencontainers.image.source="https://github.com/20262777/furiosa-opt"
