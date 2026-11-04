# IRIS CUDA qualification

This checks correctness of eager FP32 inference and batching. It does not measure
serving performance, run an API server or enable CUDA graphs. Use a fresh Colab
GPU runtime, or a Linux NVIDIA host with a compatible driver/CUDA toolkit.

The notebook [qualify_colab.ipynb](qualify_colab.ipynb) contains the same commands.
In Colab, select a GPU under **Runtime → Change runtime type** first.

## Colab commands

Run this Python notebook cell:

```python
%pip install -q uv
```

Then run this shell cell in a fresh runtime:

```bash
%%bash
set -euo pipefail
git clone --branch codex/iris --single-branch https://github.com/Artifex2002/mstar.git /content/mstar-iris
cd /content/mstar-iris
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=auto -r scripts/iris/requirements-cuda-tests.txt
mkdir -p .ci-results/iris
git clone https://github.com/eloialonso/iris.git .ci-results/iris/reference
git -C .ci-results/iris/reference checkout --detach 24326aaaa283c527f42b89b44cfdecf2665a7a16
```

This uses [uv's GPU-aware PyTorch installation](https://docs.astral.sh/uv/guides/integration/pytorch/)
in a separate Python 3.12 environment, so the notebook kernel's Python and torch
packages need not be replaced. The requirements include FlashInfer for M*'s
sampler; its [installation guide](https://docs.flashinfer.ai/installation.html)
describes toolkit/kernel requirements.

Run the qualification cell:

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
OMP_NUM_THREADS=2 .venv/bin/python -m scripts.iris.qualify_cuda \
  --download-checkpoint --output .ci-results/iris/cuda-check-01 \
  2>&1 | tee .ci-results/iris/cuda-check-01.log
```

The checkpoint download is approximately 127 MB and SHA256-verified. No ROMs or
Atari simulator are needed. Inputs are synthetic test patterns, so the result
establishes numerical mechanics rather than gameplay quality. First sampling
use may compile kernels. The command refuses to reuse an output directory; for
a rerun, change both `cuda-check-01` names to `cuda-check-02` without rerunning
the clone/setup cell.

Finally download the report, whether the command passed or failed:

```python
from google.colab import files

files.download("/content/mstar-iris/.ci-results/iris/cuda-check-01/report.json")
```

Send back `report.json`. If installation or a compiler fails before a report is
written, send the cell output; if useful, also download `cuda-check-01.log`.

## Existing cluster checkout

Use the same requirements and command from the checkout root, dropping the
`%%bash` notebook magic and using your checkout's directory. Fetch and fast-forward
`codex/iris` first. Keep a separate qualification virtualenv if the checkout
already has a serving environment. The driver imports directly from the source
tree, so no editable M* install or Rust build is needed for this particular run.
Full-server/native-runtime tests still require the normal installation.

## How to read the result

Success prints `PASSED` and records `"status": "passed"`. It means:

- A real CUDA build, GPU, Triton and FlashInfer sampling kernels executed.
- Official and M* encoder/world/decoder computations passed the declared
  teacher-forced tolerances at B=1 and B=2, through context refresh.
- Real sampling and the 2→1 request transition completed, with finite outputs,
  valid frame/reward/ending schemas and clean request-state reuse.

Failure exits nonzero and records its stage and traceback. Missing CUDA never
silently runs CPU instead. The JSON includes GPU/torch/CUDA/FlashInfer versions,
source/checkpoint identities, per-tensor maximum errors and the M* commit.
Inspect these before calling a run comparable with another result.

This test deliberately uses eager reference attention and synchronizing
diagnostics. It does **not** establish CUDA-graph correctness, native worker/API
concurrency, model quality or a speedup. Those are separate acceptance gates.
