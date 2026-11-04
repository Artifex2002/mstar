# IRIS full-server qualification

Run this after the [eager CUDA qualification](CUDA.md) passes. This is the next
part of Step 5: validate the real HTTP → data worker → conductor → native graph
runtime → GPU worker → streamed response path. It uses eager FP32 reference
attention and real FlashInfer sampling. It does not measure performance or
qualify CUDA graph capture.

Use Linux with an NVIDIA GPU. The [Colab notebook](qualify_server_colab.ipynb)
contains these commands. A T4 is suitable for the existing eager configuration;
the full-server run has not yet been qualified on it. Keep the previous CUDA
report so both runs' commits and environment versions can be compared.

## Prepare a fresh or existing Colab runtime

Select a GPU runtime. Opening the notebook does not clone its repository;
Colab runtimes also lose files when they reset. These setup cells create
`/content/mstar-iris` and its Python 3.12 environment when missing, or reuse
the existing checkout, virtualenv and cached checkpoint.

Run this Python cell first:

```python
%pip install -q uv
```

Run this shell cell. Rust installation/build can take several minutes. The
compiler version comes from this checkout's `rust-toolchain.toml`.

```bash
%%bash
set -euo pipefail
if [ ! -e /content/mstar-iris ]; then
  git clone --branch codex/iris --single-branch https://github.com/Artifex2002/mstar.git /content/mstar-iris
fi
cd /content/mstar-iris
git switch codex/iris
git pull --ff-only origin codex/iris
if [ ! -e .venv ]; then
  uv venv --python 3.12 .venv
fi
export PATH="$HOME/.cargo/bin:$PWD/.venv/bin:$PATH"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
uv pip install --python .venv/bin/python --torch-backend=auto \
  -c scripts/iris/requirements-cuda-tests.txt -e '.[iris,metrics]' maturin
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  .venv/bin/maturin develop --release --locked -m rust/Cargo.toml
.venv/bin/python -c 'import mstar_rust; print(mstar_rust.__version__)'
```

This adds the actual serving dependencies and the optional `iris` and `metrics`
extras. The constraint file keeps the previous CUDA qualification's torch,
torchvision and FlashInfer package versions. It does not freeze every dependency;
the report records the served checkout and relevant versions. A source Rust
build needs a C/C++ compiler, which standard Colab runtimes provide.

## Run and collect evidence

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
mkdir -p .ci-results/iris
OMP_NUM_THREADS=2 .venv/bin/python -m scripts.iris.qualify_server \
  --download-checkpoint --output .ci-results/iris/server-check-01 \
  2>&1 | tee .ci-results/iris/server-check-01.log
```

Choose a new output directory on a rerun, such as `server-check-02`; the runner
refuses to overwrite a previous run. An existing checkpoint is checksum-verified
and reused. No reference-source checkout or Atari ROM is needed for this stage.

Download the complete evidence directory from a Python cell:

```python
import shutil

from google.colab import files

archive = shutil.make_archive(
    "/content/iris-server-check-01", "zip",
    "/content/mstar-iris/.ci-results/iris/server-check-01",
)
files.download(archive)
```

Send the archive, or at least `report.json` and `server.log`. On installation or
Rust build failures, send the setup cell output. On failure during execution,
the report retains the failing stage and traceback and the managed server is
shut down.

For a cluster checkout, use the same commands from its root, remove `%%bash`,
and substitute its checkout path. Build the Rust extension on the tested commit,
including after any rebase that changes `rust/`.

## Acceptance checks

- The pinned checkpoint runs through the local resource driver to produce a
  seeded H=20 baseline using the real image loader and output serialization.
- A managed `mstar serve iris` process starts with its normal native graph,
  control transport and asynchronous worker. The harness supplies unique IPC
  paths, a private upload directory, an unused local port, and metrics.
- HTTP warmup and H=20 rollouts return exactly one RGB24 frame, float32 reward
  and boolean ending per action. H=20 crosses the reference's context refresh.
  Two serial requests reuse the same client request ID and explicit seed; each
  must match the local baseline's reward/ending sequence exactly and RGB bytes
  within one intensity level. Their full outputs are saved as NPZ files.
- H=20/H=20 requests overlap at the API. An H=20/H=2 pair exercises departure
  while another request continues. Batch-size metrics are recorded; concurrent
  admission alone does not prove every world-model call ran as a batched forward.
- Action 4 is rejected for Breakout. Closing a stream after its first frame must
  increment the server's abort counter. A later request reuses that client ID.
- After every case, API in-flight requests, conductor waiting/running/draining
  counts, pending worker steps and used/cached world KV pages must be zero over
  a stable window, with fresh conductor status. Missing metrics fail the check.
  Each cleanup snapshot is saved as Prometheus text. Health is checked again at
  the end, and the managed server's process group is shut down.

Read `status` and `stage` first. A passed report establishes this eager full-server
correctness gate on the reported hardware and commit. It does not establish
general cancellation correctness for all models, output quality, graph replay
correctness or a speedup. The [learning walkthrough](../../docs/design/iris-step-05-server.md)
connects these checks to M*'s layers and the following capture/benchmark work.
