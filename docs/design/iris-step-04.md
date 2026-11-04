# IRIS Step 4 / 4a: neural submodules and declared execution

This follows **Step 4 — Implement the submodules** and **Step 4a — Declare the
step** in `docs/adding_models.rst`. Step 3 is published on `codex/iris` at
`750df8b4`; the work described here is the next local increment.

Status: eager FP32 computation and CPU numerical qualification are implemented.
CUDA sampling, numerical GPU parity and full API-server execution still require
cluster validation. Batching, compilation and accelerator graphs belong to the
next increment. There is no measured speedup yet.

## What this adds

| File | Responsibility |
|---|---|
| `mstar/model/iris/components.py` | CNN tokenizer/codebook, decoder and ten-layer Transformer; attention is an injected engine operation |
| `mstar/model/iris/checkpoint.py` | Resolve the pinned checkpoint, verify its SHA256, map names and load every inference weight strictly |
| `mstar/model/iris/submodules.py` | `NodeSubmodule` wrappers: prepare, declare, forward and stop |
| `mstar/model/iris/iris_model.py` | Load on demand in `get_submodule`; share components on the same device; preserve weight-free dummy mode |
| `mstar/engine/resources/attn/torch_reference.py` | Opt-in eager attention for FP32 numerical checks, inside the existing resource lifecycle |
| `configs/iris.yaml` | Select `backend: torch` for world and spatial attention |
| `test/modular/test_iris_execution.py` | Real resources and neural forwards; optional checkpoint/reference comparisons and CUDA smoke test |
| `test/modular/test_torch_reference_attention.py` | Causal mask, GQA, independent segments/pages and explicit capture refusal |

Inference requires no upstream training package. The reference checkout stays
external, unmodified and pinned; no upstream source is vendored. The existing
license/reuse discussion in the design note still applies before an upstream PR.
For review, the engine backend is a useful separate commit/PR below the model.

## Read one real execution from input to output

Start with `ImageSubmodule.prepare_inputs`, then `declare_step`, then `forward`.
It takes one float32 CHW image. The wrapper declares spatial segments of 16, 64
and 256 positions; its attention layers use those labels at resolutions 4, 8
and 16. Encoding normalizes pixels to [-1, 1], produces a 4×4 grid of 512-wide
vectors and finds their nearest learned codebook entries. The graph carries
16 integer codes, not the continuous encoder activations, to the world model.

Next read `WorldSubmodule.prepare_inputs` beside `frame_calls` in `graph.py`.
The runtime's host-side frame/token loop counters choose one precomputed
`WorldCall`. Preparation does not inspect GPU tensor values. A frame has either
17 calls or 18 when a new context needs its initial 16-code prime. Returning
`None` vetoes a speculative call beyond the known frame length before it can
write cache entries or draw another sample.

Now read `WorldSubmodule.declare_step`. Its segment names a request, an epoch
label and a token span. It declares KV, attention and positions every call;
samplers appear only when their heads will be sampled. The engine executes
admit → plan → forward → commit. The model neither allocates cache pages nor
advances counters itself.

Then read `compute` and `WorldNetwork.forward` together. Planned position IDs
index a **learned absolute embedding table**; no RoPE rotation is applied.
Each Transformer block projects Q/K/V, writes through `KVManager.write_kv` and
attends through the engine resource. LayerNorm, residuals, GELU and the three
prediction heads retain the checkpoint's inference math. Dropout is absent
from these inference-only networks because reference evaluation disables it.

The action call predicts reward, ending and the first observation code. The
following observation calls predict the remaining codes; consuming code 15
commits its K/V but samples nothing. The prime computes the same sliced heads
as the reference even though its logits are discarded. Reward/end predictions
do not stop the supplied trajectory.

Finally read the decoder branch. It looks up the generated codes, runs the CNN
decoder, rescales/clamps RGB and serializes a frame, reward and ending. It also
returns the same code carrier to the graph, so the next frame does not encode
the generated RGB again. A newly sampled code is written into a cloned carrier,
because another node can still be reading the preceding carrier. The initial
decoder executes the reference reset's reconstruction without emitting a client
frame; its learned decoder is shared with predicted-frame decoding.

## Checkpoint loading and parameter count

The loader pins HF revision `c0539e99adf924d44be0f19b12de00639daa5550`, file
`pretrained_models/Breakout.pt`, and the SHA256 recorded in `config.py`. A local
file can be supplied with `model_kwargs.checkpoint_path`; it must have the same
hash. Otherwise `get_submodule` downloads the pinned file via Hugging Face Hub.
The constructor performs no download or neural allocation.

Tokenizer weights load under their checkpoint names. World-model names are
mapped to the inference modules and checked strictly for missing, unexpected or
wrongly shaped tensors. Saved head-index tables and causal masks become engine
spans/masks. LPIPS is training-only; the actor is outside supplied-action
imagination. Neither network is silently included in the serving count.

- Tokenizer, including encoder, codebook and decoder: **4,027,523 parameters**.
- World model: **8,447,493 parameters**.
- Unique parameters for this workload: **12,475,016**.

The initial and predicted-frame decoder wrappers share weights; counting their
parameters twice would overstate the model size. The current loader caches a
full network bundle per local device, so a process assigned just one node can
retain unused components. Qualify memory separately before deployment tuning.

## Why an engine reference-attention backend?

Models must declare attention instead of implementing private attention/cache
machinery. `AttnBackend.TORCH` and `RaggedAttnBackend.TORCH` offer an explicit
FP32 eager path for correctness. Existing `AUTO` choices are unchanged.

Paged attention gathers each request's declared pages and applies a causal mask
aligned to the **last** query positions. With one query and 17 keys, that query
can see all 17; a top-left causal mask would incorrectly expose just key zero.
Ragged attention keeps each image segment independent. Neither implementation
supports capture or pre-plan; no model manually synchronizes, manages streams,
indexes cache storage or runs resource admission/commit.

This backend allocates/gathers intermediate tensors. It is a numerical baseline,
not the optimized M* result. Such overhead must remain visible in future
benchmarks. The submodules opt out of compilation/autocast and ask the engine
for full FP32 matmuls/convolutions through existing per-node precision hooks.

## Acceptance evidence and reproduction

Run commands from the checkout root, with `OMP_NUM_THREADS=2`. Setup and trace
generation are in `scripts/iris/README.md`. The optional parity cases use the
local ignored checkpoint and the official two-row, twenty-frame trace.

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_execution.py \
  test/modular/test_torch_reference_attention.py -q -s
```

The checks drive real `StepRunner` admission/planning/commit and the actual
`prepare_inputs`/`declare_step`/`forward` methods. CPU draws are supplied in the
tests because the production sampler requires CUDA or XPU. Reference-trace tests
feed back the reference samples: this is **teacher forcing**, which isolates
neural errors from different random-number consumption.

Locally on macOS, PyTorch 2.12.1, with two CPU threads:

- Both trajectories matched all 342 world calls, including the refresh into
  `epoch_1`. Maximum world tensor absolute errors were `7.6293945e-05` and
  `9.1552734e-05`, within combined `rtol=2e-5, atol=1e-5` bounds.
- Encoder codes matched exactly. Two fresh same-batch official tokenizer
  comparisons produced identical RGB tensors.
- Float frames are checked before byte conversion. Comparison against the
  saved B=2 trace allows `rtol=2e-5, atol=5e-5`, plus at most one RGB24 level
  after truncation. Batch-dependent FP32 differences also exist within the
  official decoder: its B=1 versus saved B=2 maximum error was `1.7642975e-05`.
- Positions ended at 339 for `epoch_0` and 33 for `epoch_1`.
- The focused regression command from Step 3 plus the two new test modules
  passed **253 tests and two subtests**, with **one CUDA-only test skipped**.
- Ruff and `git diff --check` pass.

An additional eleven-module engine regression selection passed 142 tests and
38 subtests, skipped 36 accelerator cases and failed six existing cases. The
same six test IDs fail on clean `main` at `a9e81ed8` in this environment; local
Triton stubs and missing optional dependencies prevent their execution. This is
not a clean full CPU-suite result. The focused IRIS/native-routing checks ran.

## Before proceeding to optimization

On the cluster, run the CUDA eager-rollout test described in the README and
confirm it passes rather than skips. It uses the actual sampler through a full
20-frame context refresh. Then compare teacher-forced official/M* tensors on
that GPU and exercise the API server through the native graph, including two
concurrent requests and cleanup. Record the GPU model, torch/CUDA versions and
commit being tested. The CPU results do not establish these properties.

Only then add batching/capture and choose optimized attention backends. Keep
official eager, M* eager and M* optimized measurements separate; preserve
weights, precision, TF32 policy, action horizon and output/synchronization
boundaries across each comparison. Worker phase timing and end-to-end timing
answer different questions, as the benchmark plan explains.

To check your understanding, explain why positions reset while old epoch pages
remain allocated; why the final image token still needs a neural call; why only
the action call samples reward/end; why teacher forcing is useful; and why GPU
sampling correctness cannot be inferred from these CPU tests.
