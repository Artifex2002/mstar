# IRIS Step 5b: FP32 attention and engine-managed capture

The [eager full-server gate](../../docs/design/iris-t4-server-qualification.json)
passed on T4 at `ac98d64a`. This experimental step adds capture recipes and a
fixed-layout attention backend. Its CUDA acceptance run is **pending**. There
is no speedup claim, and the existing `configs/iris.yaml` remains the qualified
eager configuration.

Use [qualify_capture_colab.ipynb](qualify_capture_colab.ipynb) on a Linux NVIDIA
GPU. The notebook can prepare a fresh checkout/environment or reuse the previous
Colab runtime. It installs the reference requirements as well as the full server,
because this gate regenerates official CUDA traces. It also rebuilds the native
extension; the changes themselves are Python-only.

## Run in an existing prepared checkout

These commands assume `/content/mstar-iris` and its `.venv` from the previous
notebook. Keep Colab's runtime set to GPU. Run the setup cell first:

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
git pull --ff-only origin codex/iris
uv pip install --python .venv/bin/python --torch-backend=auto \
  -r scripts/iris/requirements-cuda-tests.txt -e '.[iris,metrics]'
mkdir -p .ci-results/iris
if [ ! -e .ci-results/iris/reference ]; then
  git clone https://github.com/eloialonso/iris.git .ci-results/iris/reference
fi
git -C .ci-results/iris/reference checkout --detach 24326aaaa283c527f42b89b44cfdecf2665a7a16
```

Then run the direct engine qualification. Capture sizing and first-use kernel
builds can take several minutes; this is a correctness diagnostic.

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
mkdir -p .ci-results/iris
OMP_NUM_THREADS=2 .venv/bin/python -m scripts.iris.qualify_capture \
  --download-checkpoint --output .ci-results/iris/capture-check-01 \
  2>&1 | tee .ci-results/iris/capture-check-01.log
```

Only after that reports `PASSED`, run the captured full-server gate:

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
mkdir -p .ci-results/iris
OMP_NUM_THREADS=2 .venv/bin/python -m scripts.iris.qualify_server \
  --config configs/iris_capture.yaml --download-checkpoint \
  --startup-timeout 1800 --request-timeout 600 \
  --output .ci-results/iris/capture-server-check-01 \
  2>&1 | tee .ci-results/iris/capture-server-check-01.log
```

Use new output directory names on reruns. The notebook's final cell downloads
both reports and available logs. Send them even after failure. It is useful to
retain the full output directories locally, including official traces.

If an earlier run stopped at `engine_capture` with "Expected a torch.device
with a specified index ... got:cuda", update `codex/iris` and retry into a new
directory. The diagnostic adapter now resolves the current logical CUDA index
before loading the engine, while preserving explicitly selected indices.
This fix requires no Rust rebuild. The pinned Gym package's maintenance notice
is separate from that exception; the qualification environment pins NumPy 1.26.

## What a pass establishes

1. On CUDA, `torch_static` matches fresh official traces at B=1/B=2 through
   20 actions and context refresh, at the existing numerical tolerances.
2. The actual `Engine` builds every requested capture bucket. The direct driver
   refuses dropped buckets or any real step that falls back to eager. Its audit
   wraps the real runner's `run_forward`, which stages inputs and calls replay;
   it records calls by node and slot.
3. Captured rollouts match an uncaptured `torch_static` run with the same
   checkpoint, inputs, precision and sampler seeds. H=20/H=2 exercises departure,
   both slots, preplanning and context refresh. A reused handle runs H=1 using
   a deliberately padded B=2 replay. World carriers/rewards/endings must match
   exactly; RGB24 may differ by at most one intensity level.
4. The subsequent real HTTP/conductor/async-worker run exercises the same
   server cases as [SERVER.md](SERVER.md). It sets the existing
   `MSTAR_REQUIRE_ACCELERATOR_GRAPHS=1` only in that child server's environment,
   so missing declared capture buckets fail startup.

The direct driver synchronizes comparisons and clones outputs for diagnostic
ownership. Its runtime is not a performance baseline. Official trace checks
and seeded capture-versus-eager checks answer different questions; report both.
If a seeded sequence diverges, retain its failing stage and investigate rather
than relaxing correctness checks without numerical evidence.

The next gate is a controlled benchmark under `benchmark/`: official PyTorch,
M* eager, and validated M* captured execution, with matching workloads, warmup,
repeated runs and worker-phase evidence. Measure before claiming a benefit from
capture or attempting further kernel/compilation tuning.
