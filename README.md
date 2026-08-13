# furiosa-opt on the PSAL cluster

A single-image environment for working through
[furiosa-ai/furiosa-opt](https://github.com/furiosa-ai/furiosa-opt) and the
[furiosa-opt book](https://developer.furiosa.ai/furiosa-opt/book), with no NPU
hardware — the `emulation` (default) and `typecheck` backends run entirely
host-side.

**Assumed setup:** you log into `ohpc-master` over SSH and run `podman` and
`enroot` there. Actual kernel runs go to a compute node through Slurm — never on
the login node.

Two ways in:

- **[Use the shared image](#1-use-the-shared-image-most-users)** — one `srun`, no
  build. This is what most people want.
- **[Build your own](#2-build-your-own-image)** — only if you are the person
  maintaining the image or you need to change it.

---

## 0. Why a container at all

The pinned toolchain cannot run on the cluster's host OS. `cargo-furiosa-opt`
requires `GCC_12.0.0` in `libgcc_s.so.1`; the nodes are Rocky Linux 9.7, which
ships GCC 11. Ubuntu 22.04 inside the container provides it.

Two things the upstream docs omit, both handled in the Dockerfile:

- `furiosa-opt-driver` links `librustc_driver-<hash>.so` and has **no RUNPATH**,
  so the `rustc-dev` rustup component and an explicit `LD_LIBRARY_PATH` are
  mandatory. Without them the driver cannot start.
- `furiosa-mapping/build.rs` and `furiosa-opt-lower/build.rs` `curl` a prebuilt
  `.a` from GitHub on every cold build. The image bakes them in and exports
  `FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT` / `FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT`,
  so builds work on a node with no outbound network.

`x86_64` only — upstream publishes the driver and static libraries for
`x86_64-unknown-linux-gnu` alone.

---

## 1. Use the shared image (most users)

The image is published once to `/home/shared`, which is world-readable and
visible from every node. You do not need podman, you do not need to import
anything, and you do not need a single `ENROOT_*` variable.

```bash
FURIOSA_SQSH=/home/shared/furiosa-opt/furiosa-opt-0.5.1.sqsh

srun -p allcpu-nolimit -N1 -c 16 -t 04:00:00 --pty \
     --container-image=$FURIOSA_SQSH \
     --container-writable \
     --container-workdir=/opt/furiosa/lab \
     /bin/bash
```

Then jump to [§4 Run the book's examples](#4-run-the-books-examples).

> If that path does not exist yet, someone needs to publish it once — see
> [§2c](#2c-publish-it-for-everyone-once). Until then, build your own.

**Please prefer the shared image.** The master's root filesystem has ~27 GB free
and a single import peaks near 11 GB, so a handful of people importing their own
copy at the same time can fill `/` on the login node.

---

## 2. Build your own image

Only needed to create or change the image. The `Dockerfile` is self-contained —
it copies nothing from the build context, so it is the only file you need.

### 2a. Build

You are not in the `docker` group and the compute nodes have no podman, so
rootless podman on `ohpc-master` is the build path:

```bash
nice -n 19 podman build --format docker -t furiosa-opt:0.5.1 .
```

`nice` matters — this is a login node and the build saturates cores otherwise.
Expect ~40 minutes cold and roughly 31 GB in `~/.local/share/containers`
(see [§6 Disk footprint](#6-disk-footprint-and-cleanup)).

On your own machine instead, Docker works the same way:

```bash
docker build --platform linux/amd64 -t furiosa-opt:0.5.1 .
```

Version overrides, if you need a release other than 0.5.1:

```bash
podman build --format docker \
  --build-arg FURIOSA_OPT_VERSION=0.5.1 \
  --build-arg RUST_TOOLCHAIN=nightly-2026-05-01 \
  -t furiosa-opt:0.5.1 .
```

The toolchain is ABI-locked to the release: bumping one without the other breaks
the driver. Check that release's `rust-toolchain.toml` before changing either.

### 2b. Convert to an enroot image (`.sqsh`)

`env | grep ENROOT` should print nothing. If it does, the variables were exported
somewhere in your session — unset them and restart your shell (or the VS Code
server, which inherits its environment at launch). Pointing them at `~/enroot/*`
forces every container rootfs onto NFS home: slower, and several GB against a
shared filesystem that is ~90% full.

The exception is importing **on `ohpc-master`**, where `/var/tmp/enroot` already
exists owned by another (unprivileged) user with mode `0700`:

```
mkdir: cannot create directory '/var/tmp/enroot': Permission denied
```

The compute nodes are unaffected (`drwxrwxrwt`); only the master is. **Do not
chmod or chown that directory** — it belongs to another user. Give yourself a
private path for this command alone:

```bash
mkdir -p /tmp/$USER-enroot/data

ENROOT_TEMP_PATH=/tmp/$USER-enroot ENROOT_DATA_PATH=/tmp/$USER-enroot/data \
  enroot import -o ~/furiosa-opt-0.5.1.sqsh podman://furiosa-opt:0.5.1

rm -rf /tmp/$USER-enroot        # ~11 GB of scratch, on a filesystem with ~27 GB free
```

Keep those variables on that one line — do not export them. `srun` propagates
your environment by default, so an exported `ENROOT_DATA_PATH` would follow your
jobs onto the compute nodes and override the correct node-local default there.

If you built with a local Docker daemon instead, swap the URI for
`dockerd://furiosa-opt:0.5.1`; from a registry, `docker://ghcr.io#<you>/furiosa-opt:0.5.1`.

### 2c. Publish it for everyone (once)

So that nobody else has to repeat §2a–2b:

```bash
mkdir -p /home/shared/furiosa-opt
cp ~/furiosa-opt-0.5.1.sqsh /home/shared/furiosa-opt/
chmod 644 /home/shared/furiosa-opt/furiosa-opt-0.5.1.sqsh
```

`/home/shared` is `drwxrwxrwx` and on NFS, so the file is readable from every
node. Note it has **no sticky bit** — anyone can delete anyone's files there, so
keep your own copy of anything you care about. The `chmod` above applies only to
the file you just created, never to anyone else's.

---

## 3. Run it on a compute node

### 3a. Slurm + pyxis (recommended)

pyxis takes the `.sqsh` directly, with no unpack step, and tears the container
down with the job.

```bash
srun -p allcpu-nolimit -N1 -c 16 -t 04:00:00 --pty \
     --container-image=/home/shared/furiosa-opt/furiosa-opt-0.5.1.sqsh \
     --container-writable \
     --container-workdir=/opt/furiosa/lab \
     /bin/bash
```

`--container-writable` is required: cargo writes to its target directory, and
enroot mounts the image read-only by default. Your home is bind-mounted in, so
edits under `$HOME` persist after the job ends.

Batch — save as `furiosa-gemm.sbatch`, then `sbatch furiosa-gemm.sbatch`:

```bash
#!/bin/bash
# allcpu-nolimit has no time limit, so -t is optional; set one anyway so a
# runaway job does not sit on cores forever.
#SBATCH -J furiosa-gemm
#SBATCH -p allcpu-nolimit
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o furiosa-gemm.%j.out

srun --container-image=/home/shared/furiosa-opt/furiosa-opt-0.5.1.sqsh \
     --container-writable \
     --container-workdir=/opt/furiosa/lab \
     cargo furiosa-opt run --release --bin gemm
```

### 3b. Plain enroot

`enroot create` unpacks into `ENROOT_DATA_PATH`, which is **node-local**
(`/var/tmp/enroot/data`). It does not persist across jobs and is not visible from
other nodes, so create it **inside** the allocation — not on the master:

```bash
srun -p allcpu-nolimit -N1 -c 16 -t 04:00:00 --pty /bin/bash -c '
  enroot create --name furiosa-opt /home/shared/furiosa-opt/furiosa-opt-0.5.1.sqsh
  enroot start --rw furiosa-opt /bin/bash
'
```

`--rw` is the enroot equivalent of `--container-writable`. Unpacking costs ~4 GB
of node-local disk and about a minute. **Prefer 3a** — pyxis skips the unpack
entirely.

### 3c. VS Code / Cursor tunnel inside the container

`stunnel` is the site wrapper at `/usr/local/bin/stunnel`. Its first argument is
an IDE name that must exist under `/opt/ohpc/pub/ide` (`code`, `cursor_x64`,
`cursor_arm64`); everything else is forwarded to `srun` verbatim, and `--image=`
switches it into pyxis container mode. This gives you an IDE whose terminal and
language server run *inside* the container, seeing `cargo furiosa-opt` and
`/opt/furiosa/lab`.

```bash
tmux new -s stunnel 'stunnel code \
    --image=/home/shared/furiosa-opt/furiosa-opt-0.5.1.sqsh \
    --partition=allcpu-nolimit \
    -w n04 --mem=80G --cpus-per-task=20 \
    --container-mount-home'
```

> **`--container-mount-home` is not optional here, and is not a stunnel default.**
> stunnel passes `--container-remap-root`, and enroot's `10-home.sh` hook only
> mounts your home when `ENROOT_MOUNT_HOME` is set. Under remap-root it mounts
> your home at **`/root`** and sets `HOME=/root` — so with the flag, `$HOME` is
> your real NFS home and your work persists. Without it, `/root` is a throwaway
> directory in the container's ephemeral layer and **everything you write is lost
> when the job ends** (verified: a file written to `$HOME` did not survive).

Pick the node with headroom — `sinfo -p allcpu-nolimit -N -o "%N %C %e %m"`.
Avoid `n07`, which has only 50 GB of RAM in total and can never satisfy
`--mem=80G`. `tmux` keeps the tunnel alive across SSH disconnects; note the
session ends if the tunnel command itself exits.

---

## 4. Run the book's examples

`/opt/furiosa/lab` inside the image is the Quick Start project, but it lives in
the container's ephemeral layer — edits there vanish when the job ends. Copy it
into your home first:

```bash
cp -a /opt/furiosa/lab ~/furiosa-opt-lab
cd ~/furiosa-opt-lab
```

The copy is ~100 KB of source; the warm build cache stays behind at
`/opt/furiosa/target`, so the first build here is still incremental.

The five worked examples from the
[Quick Start](https://developer.furiosa.ai/furiosa-opt/book/quick-start.html)
chapter:

```bash
cargo furiosa-opt run --release --bin constant_add
cargo furiosa-opt run --release --bin elementwise_mul
cargo furiosa-opt run --release --bin dot_product
cargo furiosa-opt run --release --bin gemv
cargo furiosa-opt run --release --bin gemm
```

Verify against the host-side reference, and check mappings only:

```bash
cargo furiosa-opt test --release --bin gemm
cargo furiosa-opt --backend typecheck run --release --bin gemm
```

Adding your own kernel follows the book's layout rules — `src/kernel/<name>_kernel.rs`,
a `pub mod` line in `src/kernel/mod.rs`, a host program at `src/<name>.rs`, and a
matching `[[bin]]` in `Cargo.toml`. Host programs must stay directly under `src/`;
the rustc plugin skips `src/bin/`, `examples/`, and `tests/`.

The full repository is at `/opt/furiosa/furiosa-opt`, so `make check`, `make test`,
and `cargo furiosa-opt test --test mnist_tests` work there too.

### Build cache

`CARGO_TARGET_DIR=/opt/furiosa/target` is pre-warmed in the image, so the first
build inside a job is incremental rather than a from-scratch dependency build. It
lives in the container's ephemeral write layer and is discarded when the job
ends. To keep artifacts across jobs:

```bash
export CARGO_TARGET_DIR=$HOME/.cache/furiosa-opt-target
```

Home is on NFS and ~90% full; a Rust target directory runs to several GB.

### Reading the book offline

The book sources ship with the image:

```bash
cd /opt/furiosa/furiosa-opt
mdbook serve docs --hostname 0.0.0.0 --port 3000
```

From your laptop, tunnel to the node the job landed on (`squeue` shows it):

```bash
ssh -p 7777 -L 3000:<node>:3000 <user>@psal-cluster.postech.ac.kr
```

Then open <http://localhost:3000>.

---

## 5. NPU backend

`--backend npu` needs a physical Furiosa NPU plus the Furiosa SDK
(`furiosa-driver-rngd`, `furiosa-smi`). This cluster has none, so that backend is
out of reach here; `emulation` and `typecheck` cover everything in the book
except real hardware dispatch.

---

## 6. Disk footprint and cleanup

Only relevant if you built your own image (§2). Using the shared `.sqsh` costs
you nothing.

| What | Size | Needed afterwards? |
|---|---|---|
| `~/.local/share/containers` (podman) | **~31 GB** | No — only to rebuild |
| `/tmp/$USER-enroot` (import scratch, master) | ~11 GB peak | No — delete immediately |
| `/var/tmp/enroot/data/<name>` (`enroot create`) | ~4 GB | No — node-local, pyxis needs no unpack |
| `~/furiosa-opt-0.5.1.sqsh` | 3.3 GB | Only if you are not using the shared copy |

The podman figure is not a typo: `podman images` reports ~7.4 GB logical, but each
failed or repeated build branches after the rustup layer, so several copies of the
toolchain and cargo layers pile up.

```bash
podman rmi -af                      # drop all images
```

**`podman rmi -af` is not enough on this cluster.** With the overlay driver on
NFS, layer directories survive image deletion: after removing every image you can
still be left with tens of GB of orphans, and `podman system prune -a` reports
`Total reclaimed space: 0B`. Verify and, if so, reset:

```bash
du -sh ~/.local/share/containers          # if this is still GB-sized...
podman system reset -f                    # ...wipe the storage tree outright
```

`podman system reset` destroys **all** your podman images and containers, so check
`podman images` first if you keep unrelated ones.

Also note `enroot import podman://` leaves a stopped container behind on every
run. Clear them with `podman ps -a` / `podman rm <name>`.

---

## 7. Troubleshooting

**`mkdir: cannot create directory '/var/tmp/enroot': Permission denied`**
You are importing on the master. Use the private paths in [§2b](#2b-convert-to-an-enroot-image-sqsh).

**`tar (child): /home/enroot/cache/...: Cannot open: Permission denied`**
The shared layer cache is world-writable, but blobs land with each user's umask —
72 of 176 are mode `0640` and unreadable to anyone else. This affects `docker://`
registry pulls only; `podman://` and `dockerd://` bypass the cache. Those blobs
belong to other users, so **do not chmod them**. Point that one command at your
own cache instead:

```bash
ENROOT_CACHE_PATH=/tmp/$USER-enroot-cache enroot import -o img.sqsh docker://…
```

The cost is re-downloading layers the shared cache already holds.

**`enroot list` / imports behave oddly, or containers appear under your home**
`env | grep ENROOT` — it should print nothing. See [§2b](#2b-convert-to-an-enroot-image-sqsh).

**Cargo rebuilds everything on each job**
Expected: `/opt/furiosa/target` is inside the container's ephemeral layer. Set
`CARGO_TARGET_DIR` to a path under `$HOME` to persist it.

---

## Partitions

`sinfo` for current state. GPU partitions are irrelevant here — the emulation
backend is CPU-only.

| Partition | Nodes | Notes |
|---|---|---|
| `allcpu-nolimit` | n01–n10 | default, no time limit — use this |
| `amd-cpu` / `intel-cpu` | n01–n04, n08–n10 / n05–n06 | if you need a specific µarch |
| `allgpu`, `gpu-3090`, `gpu-2080Ti` | g01, g05–g07 | 11-day limit, not needed here |

---

## Verified

Built and run end-to-end on this cluster on 2026-08-06.

- Image built with rootless podman on `ohpc-master`: 7.38 GB → 3.3 GB `.sqsh`
- Ran on compute node n05 as uid 1060 (not root), Ubuntu 22.04.5 inside
- `rustc 1.97.0-nightly (f53b654a8 2026-04-30)`, `cargo-furiosa-opt 0.5.1`
- All five Quick Start binaries ran; `cargo furiosa-opt test --release --bin gemm`
  reported `1 passed; 0 failed`; the `typecheck` backend ran
- Both routes work: pyxis `--container-image` and `enroot create` + `enroot start --rw`
- Re-verified with **no `ENROOT_*` variables set at all** (`srun --export=NONE`),
  using only the site defaults
- Import on the master verified with the private-path form in §2b, changing no
  permissions anywhere
- A copy of `/opt/furiosa/lab` in `$HOME` rebuilt in 1.6 s against the pre-warmed cache
- `mdbook build docs` renders the book offline

Cosmetic: mdbook-mermaid warns it was built against mdbook 0.5.0 while binstall
installs 0.5.4. The book renders correctly; pin `mdbook@0.5.0` in the Dockerfile
if you want it silent.

Not verified, because this cluster has no Furiosa NPU: `--backend npu`.

---

## For the cluster admin

Two site-level issues make enroot harder to use than it should be, and both have
the same root cause: `/etc/enroot/enroot.conf` uses **single shared paths** for
state that is inherently per-user, so whoever runs enroot first owns the
directory and everyone else is locked out by their umask.

Symptoms:

1. **`/var/tmp/enroot` on `ohpc-master`** was created by the first user to run
   enroot there and is `drwx------`, so nobody else can import an image on the
   login node. The compute nodes happen to be `drwxrwxrwt` and work fine. With
   every user working on the master, this affects everyone.
2. **`/home/enroot/cache`** accumulates blobs owned by individual users; 72 of 176
   are mode `0640`, so a `docker://` pull fails for everyone except the user who
   cached that layer.

The robust fix is a **configuration change, not a permission change** — enroot
expands `${VAR}` and `$(cmd)` in `enroot.conf` (its own shipped defaults use
`${XDG_CONFIG_HOME}` and `$(nproc)`), so the per-user paths upstream intends can
be restored:

```bash
# /etc/enroot/enroot.conf
ENROOT_RUNTIME_PATH  /run/enroot/user-$(id -u)
ENROOT_DATA_PATH     /var/tmp/enroot-$(id -u)/data
ENROOT_TEMP_PATH     /var/tmp/enroot-$(id -u)
ENROOT_CACHE_PATH    /home/enroot/cache            # shared is fine if readable
```

Each user then gets their own directory on every node and nothing collides.

If the shared cache is kept for its deduplication benefit, the umask is what needs
fixing — new blobs should land group/world-readable — rather than retroactively
changing the mode of files other users own.

Worth knowing: the master's root filesystem is ~88% full (~27 GB free) and
`/var/tmp` and `/tmp` both live on it. Since every user imports on the master and
one import peaks near 11 GB, concurrent imports can fill `/` on the login node.
Publishing one shared `.sqsh` under `/home/shared` (§2c) is the practical
mitigation.

**Deliberately not recommended:** `chmod`/`chown` on `/var/tmp/enroot` or on blobs
under `/home/enroot/cache`. Those are unprivileged users' files, and rewriting
their permissions is not a safe fix.

Contact: 박정기 <jkpark85@postech.ac.kr>
