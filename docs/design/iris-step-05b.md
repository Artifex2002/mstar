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
official traces, then checks real capture/replay, request departure, preplanning
and handle reuse against static eager execution. Finally rerun the server with
the capture config to check the actual asynchronous worker.

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
