# IRIS Step 5a: eager continuous batching and CUDA qualification

Step 4 was published in `4eddbe16` (reference attention) and `36c9b856` (IRIS
neural submodules). This increment follows the **continuous batching** part of
Step 5 in `docs/adding_models.rst`. It adds executable batching and a CUDA
qualification command. Accelerator graphs remain disabled: the selected FP32
attention resources explicitly refuse capture. No performance improvement has
been measured or claimed.

## What to read and implement on your learning branch

Start from `36c9b856` when comparing this increment. Read these changes in order:

1. `config.py`, `iris_model.py` and `configs/iris.yaml` add a deployment
   `max_batch_size`, default 2. The existing 64-page pool can hold two complete
   20-frame requests (25 pages each, plus its sink page). To increase concurrency,
   tune both the batch limit and the resource pool; changing just the limit does
   not create more KV capacity.
2. `IrisSubmodule.preprocess` stacks compatible requests into a batch. `forward`
   and `forward_batched` share `forward_tensors`, then return the required
   per-request lists of tensors. Inspect frame shapes: `[B, 1, 64, 64, 3]`
   becomes `[1, 64, 64, 3]` for one request, preserving the API contract.
3. `WorldSubmodule.prepare_inputs` selects the current action or observation
   codes using host loop counters. It passes only those tokens, avoiding a stack
   of full action sequences with different horizons. No GPU values are read.
4. `WorldSubmodule.can_batch` requires the same call kind, epoch label and sampled
   code index. Requests can be at different frame positions in that epoch, since
   each has its own positions/cache. Prime, action, final observation and different
   epochs must not accidentally share the wrong forward/declaration.
5. `WorldNetwork.forward` reshapes packed position IDs to `[B, T]` before learned
   embedding lookup. A one-token batch has B independent positions, not one
   common position or a broadcast that invents a `[B, B, D]` sequence.
6. Read the tests in `test_iris_batching.py`: real batched calls are compared with
   independent executions using separate resource pools. Different cache lengths,
   reversed request order and per-request stop checks exercise row addressing.

The scheduler selects the requests; model code only declares whether the given
selection is compatible. The current eager scheduler does not regroup requests
by IRIS phase. An incompatible selection falls back to per-request execution,
which preserves correctness but may limit batching opportunities. Phase-aware
scheduling is not claimed here, and no global worker behavior was changed.

The concepts to learn are **batch selection versus collation**, packed token
order, per-request state isolation, head/sequence/batch tensor dimensions,
fixed-shape versus mixed-phase compute, and resource capacity/backpressure.
Explain why compatible rows can have different absolute positions but cannot
use different epoch labels in this implementation.

## The diagnostic driver

`scripts/iris/local_engine.py` builds real resources through their factories and
drives real admission → planning → forward → commit. It uses the existing local
transfer engine with no remote KV consumers. It does not substitute a Python
graph runtime or copy worker/engine methods into fakes. It does not run the
worker, scheduler, native graph, HTTP server or multi-GPU transport.

`qualification.py` provides two checks:

- Teacher-forced official-trace comparisons at batch sizes 1 and 2: encoder
  codes, initial reconstruction, every hidden feature/logit tensor, float and
  RGB24 frames, rewards/endings, positions and teardown. It includes the full
  context refresh after frame 19. Only sampler draws are replaced in this check.
- Free-running sampling: real CUDA/Triton/FlashInfer draws. Two requests run
  horizons 20 and 2, so the second leaves and the first continues alone. The
  driver checks finite outputs, schemas/ranges, frame counts, teardown and a
  reused handle's positions/RNG counters before executing it again.

`qualify_cuda.py` runs a CUDA/sampling preflight, generates fresh official
traces at B=1 and B=2 on that GPU with TF32 disabled, then runs both checks. It records
the M* commit, reference/checkpoint pins, hardware and software versions, maximum
tensor errors and any failure traceback in `report.json`. It exits nonzero on
failure, with no CPU fallback or skipped acceptance cases. Reference trace hooks
and diagnostic finite checks synchronize deliberately; **do not use this runner
for timings**.

For local CPU testing, the sampler draw is supplied or replaced with argmax.
That validates the driver and numerical batching, not CUDA sampling. The
checkpoint cases skip explicitly when the local ignored assets are absent.

## Run on Colab or a CUDA host

The executable commands and downloadable-report instructions are in
[the CUDA run guide](../../scripts/iris/CUDA.md). A notebook containing the same
commands is provided as `scripts/iris/qualify_colab.ipynb`.

This is an isolated qualification environment. It does not build Rust, install
the whole serving stack or enable attention/CUDA-graph optimizations. FlashInfer
is needed for the **sampler** even while attention uses the PyTorch reference
backend. First use may compile kernels.

After a successful run, review `report.json` and keep it with the tested commit.
Before performance comparisons, full server/native-runtime concurrency and
cleanup still need GPU validation. Then implement capture using engine-owned
configurations and a qualified capturable attention backend. Do not flip
`can_capture()` on the eager reference backend: replay would freeze host-selected
layouts and would not automatically support changing request contexts.

## Local acceptance

The new CPU checks cover incompatible batch rejection before resource planning,
batched-versus-independent outputs, row order, position broadcasting, per-row
loop stops, two full official trajectories, batch departure, handle reuse,
finite-value checks and failure-report behavior without a GPU. Run:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_batching.py \
  test/modular/test_iris_execution.py \
  test/modular/test_iris_graph.py \
  test/modular/test_iris_model.py \
  test/modular/test_iris_resources.py -q
```

Local focused regression result: **263 tests and two subtests passed**, with one
CUDA-only case skipped. This includes the real-weights CPU comparisons and
native Rust routing tests. Ruff, formatting and `git diff --check` passed. The
broader local-environment failures on clean `main` are documented in Step 4;
this is not a claim that the entire CPU Core suite passed.

The CUDA command has not been run locally; this machine has no NVIDIA GPU.
Passing CPU tests and a prepared Colab notebook are not a CUDA qualification
result. The next acceptance gate is the report from that run.
