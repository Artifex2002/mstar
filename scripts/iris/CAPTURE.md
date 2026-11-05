# IRIS Step 5b: FP32 attention and engine-managed capture

The [eager full-server gate](../../docs/design/iris-t4-server-qualification.json)
passed on T4 at `ac98d64a`. This experimental step adds capture recipes and a
fixed-layout attention backend. **Direct capture/replay and captured full-server
correctness passed on Tesla T4 at clean commit `2ea19ee4`**, using PyTorch
2.12.1+cu132. See the [direct report](../../docs/design/iris-t4-capture-qualification.json),
[server report](../../docs/design/iris-t4-capture-server-qualification.json) and
[evidence notes](../../docs/design/iris-step-05b.md#t4-qualification-evidence).
Server multirow batching was not established by that run, and performance is
unmeasured. `configs/iris.yaml` remains the eager configuration.

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

An older comparison also stopped at `INITIAL_DECODER` on request-key equality.
That node deliberately emits no tensors: engine collection returns `{}`, while
the direct driver retains empty entries per request. Update the branch for the
initial-node representation adapter and rerun into a new output directory.
Admission, replay auditing and all output-producing node comparisons remain
strict. This diagnostic fix also needs no Rust rebuild.

The `capture-check-03` run reached a sampled observation-token mismatch.
The original gate incorrectly required identical seeded draws from eager and
captured execution. In `sample_accelerator_graphable_gpu`, M* uses `fused_sample`
for these small vocabularies, while eager `_sample_cuda` uses FlashInfer. The
sampler implementation explicitly documents different draws for the same seed.
This explains why token identity is not a valid cross-path acceptance contract;
it does not by itself prove that captured logits or cache state are correct.
Update the branch and retry as `capture-check-04`. No Rust rebuild is needed.

## What a pass establishes

1. On CUDA, `torch_static` matches fresh official traces at B=1/B=2 through
   20 actions and context refresh, at the existing numerical tolerances.
2. The actual `Engine` builds every requested capture bucket. The direct driver
   refuses dropped buckets or any real step that falls back to eager. Its audit
   wraps the real runner's `run_forward`, which stages inputs and calls replay;
   it records calls by node and slot.
3. Captured hidden states and logits match the fresh official trace under
   teacher forcing (`rtol=2e-5`, `atol=1e-5`). A diagnostic world submodule returns
   these tensors as graph outputs, so replayed values are actually inspected.
   Real sampler draws execute and are checked for valid ranges and untouched
   carrier columns. Only the copied carriers fed into the next step receive the
   official samples. Decoded RGB24 must match within one intensity level.
4. Separate free-running captured rollouts repeat exactly with the same seeds
   and schedule after teardown. H=20/H=2 exercises departure, both slots,
   preplanning and context refresh. A reused handle runs H=1 in a deliberately
   padded B=2 replay. RNG counters must equal 16 observation draws and one
   reward/ending draw per frame, and request sampler registrations and KV pages
   must be released. Eager-versus-captured sampled token identity is not required.
5. The subsequent real HTTP/conductor/async-worker run exercises the same
   server cases as [SERVER.md](SERVER.md). It sets the existing
   `MSTAR_REQUIRE_ACCELERATOR_GRAPHS=1` only in that child server's environment,
   so missing declared capture buckets fail startup. Its standalone baseline
   also uses the real captured engine with ordinary production submodules;
   it must use the same sampler algorithm as the captured server.

The direct driver synchronizes comparisons and clones outputs for diagnostic
ownership. Its runtime is not a performance baseline. Numerical parity and
sampling repeatability answer different questions; report both. Failures retain
frame/tick/request context and numerical difference details. These checks do not
replace statistical sampler tests or benchmark output-quality evaluation.

The next gate is a controlled benchmark under `benchmark/`: official PyTorch,
M* eager, and validated M* captured execution, with matching workloads, warmup,
repeated runs and worker-phase evidence. Measure before claiming a benefit from
capture or attempting further kernel/compilation tuning.
