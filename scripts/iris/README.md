# IRIS reference and qualification tooling

`reference_harness.py` calls the external official IRIS implementation. It
establishes a numerical baseline for the model integrated into M*
through the [guided contribution plan](../../docs/design/iris.md).

For the new batched M* implementation, use the [CUDA qualification guide](CUDA.md)
or `qualify_colab.ipynb`. It runs the official/M* numerical comparison on the
same GPU, followed by actual engine sampling and request-state checks.

The workload is a learned-world prediction: initial frames and supplied actions
produce imagined frames, rewards and ending signals. No Atari emulator, ROM,
actor training, or actor inference is required. The reference source and weights
are kept in Git-ignored `.ci-results/iris/`, not vendored into M*.

## This chat's checkout

The implementation branch is `codex/iris`, based on `main`, in:

```text
/Users/ashutoshpanda/.codex/worktrees/d8dc/mstar
```

Your original checkout remains on `dreamer_v3`. Each worktree has a separate
working directory and `.venv`. Run IRIS commands from the IRIS checkout root.

## One-time setup

An environment, reference checkout and verified Breakout checkpoint have already
been prepared in this worktree. For another checkout, recreate them with:

```bash
cd "$(git rev-parse --show-toplevel)"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r scripts/iris/requirements-reference.txt
mkdir -p .ci-results/iris
git clone https://github.com/eloialonso/iris.git .ci-results/iris/reference
git -C .ci-results/iris/reference checkout --detach 24326aaaa283c527f42b89b44cfdecf2665a7a16
.venv/bin/python scripts/iris/reference_harness.py \
  --download-checkpoint --checkpoint .ci-results/iris/Breakout.pt
```

These requirements are inference-only CPU reference tooling. They intentionally
do not reproduce upstream's Atari training dependencies. The pinned source's
inference APIs were checked with the listed modern PyTorch versions; the reference
checkout remains unmodified. `gym` is imported by upstream, but no simulator is
constructed here. It prints a maintenance notice even with the pinned NumPy 1.x.
The `with_lpips=False` inference constructor avoids downloading VGG/LPIPS weights.

The download verifies SHA-256 before moving a `.partial` file into place. A failed
download leaves that partial file for inspection; remove an incomplete partial
before retrying. Existing completed checkpoints are checksum-verified, not replaced.

## Your first run

```bash
source .venv/bin/activate
python scripts/iris/reference_harness.py \
  --reference-root .ci-results/iris/reference \
  --checkpoint .ci-results/iris/Breakout.pt \
  --horizon 3 --batch-size 2 --output .ci-results/iris/my-first-rollout
```

Default inputs are a deterministic synthetic image pattern and a short legal
action sequence. The checkpoint is pretrained; the initial image is not a real
Breakout observation. This tests mechanics and numerical behavior, not game
quality or episode return. Existing output directories are refused.

Repeat with identical arguments and a new output directory:

```bash
python scripts/iris/reference_harness.py \
  --reference-root .ci-results/iris/reference \
  --checkpoint .ci-results/iris/Breakout.pt \
  --horizon 3 --batch-size 2 \
  --compare .ci-results/iris/my-first-rollout/trace.npz \
  --output .ci-results/iris/my-repeat-rollout
```

Use `--horizon 20` in a separate output directory to cross the reference's
context limit and observe a refresh. The reference resumes from the last image's
16 tokens; it does not silently slide old keys while retaining old positions.

## Supply real saved inputs later

`--observations frames.npy` takes float32 `[B,3,64,64]` images in `[0,1]`.
`--actions actions.npy` takes int64 `[B,H]` actions in `[0,4)` for this Breakout
checkpoint. Set `--batch-size B --horizon H` consistently. The helper deliberately
does not resize or normalize incompatible inputs silently.

The action sequence is external. Predicted ending signals are recorded; the
harness continues the requested horizon rather than treating itself as a real
environment episode. To assess learned behavior, use actual captured observations
and compare several examples against the reference and the eventual M* port.

## What to inspect

- `report.json`: pinned code/checkpoint, parameter counts, architecture, input
  provenance, software versions, dense-versus-cached comparison and refresh count.
- `trace.npz`: initial image/tokens/reconstruction; every Transformer call's input
  tokens, hidden features, logits and cache size; every frame's sampled tokens,
  decoded image, reward and ending signal.

```python
import numpy as np

with np.load(".ci-results/iris/my-first-rollout/trace.npz", allow_pickle=False) as trace:
    print(trace["initial/tokens"].shape)  # [2, 16]
    print(trace["step/0000/frame"].shape)  # [2, 3, 64, 64]
    print(trace["step/0000/cache_size"])  # 33
```

Before rollout, the harness compares one dense forward over 33 known tokens with
the same sequence run as a 16-token prefill followed by cached one-token forwards.
It checks hidden features and all three prediction heads. The float32 comparison
uses `rtol=2e-5`, `atol=1e-5`; measured absolute differences are saved in the report.
Each rollout trace is checked for finite values. Repeat comparisons also require
matching keys, shapes and dtypes and exact integer/boolean outputs.

This is the external reference's cache, not M*'s cache. Later we will compare the
M* path against these recorded tensors using controlled input tokens and draws.
Reference sampling agreement alone cannot locate the first numerical difference;
the saved intermediate logits and tokens help diagnose it.

## CPU utility tests

The prepared environment also includes local test/lint tools. To recreate that
part in a new environment:

```bash
uv pip install --python .venv/bin/python pytest 'ruff<=0.16' PyYAML msgpack pyzmq fastapi
```

From the IRIS repository root:

```bash
.venv/bin/ruff check .
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_reference_harness.py \
  test/modular/test_model_registration.py -q
```

## Native graph-routing tests

The graph tests use synthetic CPU tensors and the real Rust runtime. They do not
load IRIS weights. Install Rust/Cargo as described in `docs/installation.rst`, then
build the extension into this checkout's environment:

```bash
uv pip install --python .venv/bin/python 'maturin>=1,<2'
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  .venv/bin/maturin develop --release -m rust/Cargo.toml
OMP_NUM_THREADS=2 .venv/bin/python -m pytest test/modular/test_iris_graph.py -q
```

The `env` command selects `.venv` explicitly if Conda base is active. Rebuild the
extension when upstream Rust source changes. Without it, the native test cases
skip; the structural tests still run. Local macOS modular fixtures stub optional
accelerator imports; they do not emulate GPU computation.

Full serving still needs M*'s normal dependencies. The graph-routing tests use
synthetic outputs; learned-model parity is checked separately below. Neither is
a performance benchmark.

## Eager M* inference qualification

The checkpoint loader and neural submodules now execute through real engine
resources. For the optional parity tests, keep the verified checkpoint at
`.ci-results/iris/Breakout.pt` and generate the 20-frame, two-row official trace:

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/iris/reference_harness.py \
  --reference-root .ci-results/iris/reference \
  --checkpoint .ci-results/iris/Breakout.pt \
  --output .ci-results/iris/context-refresh --horizon 20 --batch-size 2
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_execution.py \
  test/modular/test_torch_reference_attention.py -q -s
```

The trace generator requires a new output directory; reuse an existing verified
trace instead of rerunning that command over it. The parity cases skip when
their checkpoint, trace or reference checkout is absent. They compare two
trajectories with the reference's sampled values supplied to the M* samplers;
CPU tests do not execute M*'s CUDA/XPU sampling kernel.

On a CUDA host with M* installed and the checkpoint cached, qualify actual
sampling with:

```bash
OMP_NUM_THREADS=2 python -m pytest test/modular/test_iris_execution.py \
  -k cuda_eager_rollout -q -s
```

This must report a pass, not a skip. It executes a 20-frame rollout through the
real eager submodules/resources, including context refresh. It does not launch
the API server or establish numerical GPU parity against official IRIS. Those
checks remain required before benchmarking. `configs/iris.yaml` opts into the
eager `torch` attention backends. The later [Step 5b lesson](../../docs/design/iris-step-05b.md)
records passed eager/captured qualification on T4 and the limits of the batching
evidence. Use [CAPTURE.md](CAPTURE.md) for the separate captured configuration.
See [the measured T4 baseline](../../docs/design/iris-t4-benchmark.md) for performance
and [the Step 4 lesson](../../docs/design/iris-step-04.md)
for the original eager implementation.

For the next experiment, use the [baseline benchmark](../../benchmark/iris/README.md)
and [Step 5c lesson](../../docs/design/iris-step-05c.md). Start with worker batching
evidence, then compare official/eager/captured standalone rollouts and M* HTTP
serving as separate scopes. The T4 audit and uninstrumented sweep passed;
[the measured baseline](../../docs/design/iris-t4-benchmark.md) preserves rates,
ranges and limits for official/eager/captured execution.

The next optional graph-layout experiment is
[Step 5d: async RGB decoding](../../docs/design/iris-step-05d.md). Start with
[decoder_pipeline_colab.ipynb](decoder_pipeline_colab.ipynb) to qualify both
pipeline servers against the serial implementation before any graph A/B.
CUDA correctness and performance of this experimental layout are pending.
