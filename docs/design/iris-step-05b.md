# IRIS Step 5b: make execution capturable

This follows the [passed eager server check](iris-t4-server-qualification.json).
The implementation is CPU-checked; **CUDA capture/replay qualification is pending**.
Use the [run guide](../../scripts/iris/CAPTURE.md) or its Colab notebook.

## What changes and why

`configs/iris_capture.yaml` opts into `capture_graphs: true` and `torch_static`
attention. `configs/iris.yaml` keeps the already-qualified eager path. Both use
the same FP32 checkpoint, learned positions, action schedule and resource-owned
KV and sampler state. Compilation remains disabled for this first capture gate.

`mstar/engine/resources/attn/torch_static.py` provides opt-in paged and cacheless
attention. The paged path gathers into a fixed context ceiling. Page addresses,
valid-key masks and bottom-right causal masks are staged in persistent buffers
per capture bucket, slot and label. It zeros invalid keys/values with a select
before either matmul, so poisoned sink/tail storage cannot contaminate live rows.
Padding rows are independent and return zero attention outputs.

The resource supports preplanning and forces double buffering. A preplan writes
the next slot and is promoted when that step executes; it does not replace the
in-flight step's current plan. Eager steps use their live width; captured steps
use the leased bucket width. A layout change within one capture bucket is an
error. The ragged path batches equal-length independent spans with ordinary
PyTorch math. Both paths preserve input precision, including FP32.

This backend is deliberately limited to uniform positive spans per label and
one segment per request per label. Paged storage must be contiguous NHD, without
context-only attention, windows or LSE output. It performs work up to the fixed
context ceiling; it is not a fused attention kernel or a demonstrated speedup.
The factory selects it only when explicitly requested. AUTO and other models'
backends are unchanged.

## Read the model contracts

1. `IrisConfig.capture_graphs` and `IrisModel.get_submodule` pass the deployment
   opt-in to submodules. No model code captures a graph or indexes resource storage.
2. `ImageSubmodule.get_accelerator_graph_configs` declares batched encoder and
   decoder inputs. The initial reconstruction is captured too, despite emitting
   no client result.
3. `WorldSubmodule` declares one recipe per `(kind, context label, sample index)`.
   These distinguish action versus observation embeddings, sampler heads,
   the column updated in the code carrier, and the KV epoch. Position values
   remain runtime resource inputs, so each frame position needs no new graph.
4. `cg_key_info`, each recipe's `additional_key_info`, and `declare_step` agree.
   Per-step metadata is authoritative even when shared request counters advance.
   Overshoot has no capture key and returns before neural work.
5. Captured mode groups compatible phases using the existing scheduler hook.
   Static input dimensions explicitly name batch axis 0; guessing from a size
   equal to the token count would confuse `[B, 16]` at B=1 with its code axis.
6. `WorldCall` is an immutable dataclass. Capture templates can safely share it;
   a tuple-like phase record could not pass `NodeInputs.clone`'s tuple copying.

## Checks before the GPU run

`test_torch_static_attention.py` checks explicit reference numerics, unequal KV
lengths, GQA, bottom-right masks, NaN-poisoned padding, stable buffer addresses,
preplan isolation/promotion and layout rejection. With cached weights/traces it
also checks all 342 B=2 world calls and 40 decoded frames against official IRIS.

`test_iris_capture_contract.py` checks every phase through context refresh,
capture template cloning, runtime/step/config key agreement, image input schemas,
strict opt-in and the batch-axis declarations. The existing IRIS tests still
cover eager execution and sparse serving metadata.

Local verification on macOS: 382 tests and two subtests passed, with five GPU
cases skipped. One resource import probe fails because its fresh subprocess has
no Triton installed; the identical test fails on the clean main snapshot too
(its test and importing KV source match upstream `c4277651`). Repeating the
selection with that probe deselected gives 382 passed, five skipped and one
deselected. Ruff, notebook syntax/metadata and diff checks passed. This is not a
claim of CUDA execution or a full CPU Core CI run.

`scripts/iris/captured_engine.py` is a diagnostic adapter around the real Engine,
not a second capture implementation. It audits real runner replay calls, tests
padding through an actual engine lease, and fails on eager fallback.
`qualify_capture.py` first compares CUDA static-attention results with fresh
official traces, then checks real capture/replay numerics under shared official
inputs, request departure, preplanning and handle reuse. Finally rerun the server with
the capture config to check the actual asynchronous worker.

A reported CUDA attempt completed the numerical stage, then failed at engine
capture sizing because the adapter supplied `torch.device("cuda")` without an
index. The graph backend's `torch.accelerator.set_device_index` requires an
explicit ordinal. The adapter now resolves an implicit device to the current
logical CUDA index and preserves explicit indices. CPU regressions exercise
the exact PyTorch validation that rejected the original descriptor. The 37
focused checks for this fix pass; CUDA capture/replay still needs its rerun.

The subsequent attempt passed capture setup and encoder comparison, then
stopped on the initial decoder's empty output mapping. The engine intentionally
omits empty per-request entries during collection; the direct driver preserves
them. `copy_step_outputs` adapts this representation only for the initial
reconstruction, after admission/failure/replay checks. A CPU regression calls
the actual `Engine._collect_outputs` for eager and captured addressing and
reproduces the empty mapping. Unexpected initial output data and missing rows
on output-producing nodes still fail. Complete CUDA rollout replay remains
pending its rerun.

The next attempt reached `world_model/1/observation_tokens` disagreement. The
gate was requiring identical seeded draws across algorithms, which is not M*'s
sampler contract: `sample_accelerator_graphable_gpu` uses `fused_sample` for
IRIS's vocabularies (512/3/2), whereas eager `_sample_cuda` uses FlashInfer.
The sampler's docstring explicitly says those draws differ for the same seed.
The mismatch alone therefore cannot establish a neural or cache error.

The corrected diagnostic returns hidden states and logits as captured graph
outputs and compares them directly with the official trace before feeding back
official samples in copied carriers. Numerical tolerances are unchanged. Real
sampling still executes; range, carrier preservation, RNG-counter and teardown
checks remain active. Separate free-running captured trajectories must repeat
exactly after handle reuse. The captured HTTP gate now uses a captured local
baseline with the same sampling algorithm. CPU regressions exercise real IRIS
submodules/resources with both argmax and argmin draws: shared-input numerical
parity passes for both, while perturbed logits, invalid draws and corruption of
untouched carrier columns fail with step context. CUDA qualification of these
corrected checks is still pending.

For this correction, 52 focused CPU tests passed on macOS with PyTorch 2.12.1
and `OMP_NUM_THREADS=2`, including cached-checkpoint numerical tests, real sampler
buffer reset checks and server-baseline selection. `ruff check .` and
`git diff --check` passed. These results do not exercise CUDA replay.

## What to learn

Explain why a CUDA graph needs stable addresses and shapes while values can
change. Trace which buffers plan N+1 may write while graph N is running, why
padding must describe valid independent rows, and why Python branch choices
belong in a capture key. Distinguish compiling tensor programs from replaying
their launches: this step tests replay without also changing compiler behavior.

After both GPU reports pass, build a timing harness and collect eager and
captured baselines before further tuning. Read the repository's benchmarking
skill and use worker phases to identify CPU versus GPU work. Neither successful
capture nor this diagnostic driver's wall time establishes a speedup.
