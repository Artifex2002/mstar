# IRIS reference rollout harness

This tool calls the external official IRIS implementation; it is not an M* model
integration. It establishes a numerical baseline before we implement the model
through the [guided contribution plan](../../docs/design/iris.md).

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

Full serving/graph-runtime validation later needs M*'s normal dependencies and
Rust extension. This harness has not established M* inference or a speedup.
