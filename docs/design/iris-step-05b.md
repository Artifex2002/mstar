# IRIS Step 5b: make execution capturable

This follows the [passed eager server check](iris-t4-server-qualification.json).
**CUDA capture/replay and captured full-server correctness passed on Tesla T4
at clean commit `2ea19ee4`** (PyTorch 2.12.1+cu132). This qualifies the recorded
Breakout workloads; server multirow batching and performance remain to be measured.
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
focused checks for this fix passed before the next CUDA attempt.

The subsequent attempt passed capture setup and encoder comparison, then
stopped on the initial decoder's empty output mapping. The engine intentionally
omits empty per-request entries during collection; the direct driver preserves
them. `copy_step_outputs` adapts this representation only for the initial
reconstruction, after admission/failure/replay checks. A CPU regression calls
the actual `Engine._collect_outputs` for eager and captured addressing and
reproduces the empty mapping. Unexpected initial output data and missing rows
on output-producing nodes still fail. This fix preceded the sampled-token
comparison correction below.

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
untouched carrier columns fail with step context. The corrected CUDA checks
then passed in the run recorded below.

For this correction, 52 focused CPU tests passed on macOS with PyTorch 2.12.1
and `OMP_NUM_THREADS=2`, including cached-checkpoint numerical tests, real sampler
buffer reset checks and server-baseline selection. `ruff check .` and
`git diff --check` passed. These results do not exercise CUDA replay.

## T4 qualification evidence

The user supplied `iris-capture-check-04.zip`, containing reports and logs from
clean commit `2ea19ee46d380fdad3a285a15f882b276253c92b` on Tesla T4, PyTorch
2.12.1+cu132, CUDA build 13.2 and Rust extension 0.1.1. These are user-run GPU
results inspected locally; the GPU commands were not rerun on the macOS host.
The original report bytes are preserved in
[iris-t4-capture-qualification.json](iris-t4-capture-qualification.json) and
[iris-t4-capture-server-qualification.json](iris-t4-capture-server-qualification.json).

The direct report establishes:

- Static attention matches fresh official B=1/B=2 traces through H=20, including
  342 world calls per batch and the context refresh.
- All 78 requested buckets captured with two slots each: 72 world buckets and
  two each for encoder, decoder and initial reconstruction.
- Captured teacher-forced hidden states/logits pass 1,589 tensor comparisons at
  the existing tolerances. Decoded RGB24 differs by at most one intensity level.
- Free-running captured repeats match 2,912 tensor fingerprints exactly.
  The driver audits 1,080 actual world replays, split 540/540 across slots,
  with preplanning, H20/H2 departure and padded H1 handle reuse.
- RNG counters equal 320/20/20 for H20, 32/2/2 for H2 and 16/1/1 for H1
  (observation/reward/ending). Sampler teardown checks pass and no KV pages remain.

The server report uses FlashInfer 0.7.2 and strict capture startup. The server
log independently lists the same 78 captured buckets. Two seeded H20 HTTP
responses (including request-ID reuse) match the standalone captured baseline's
frame/reward/ending digests exactly, with reported RGB24 error zero. Both H20/H20
and H20/H2 cases reach two API requests in flight. Invalid actions return 400;
disconnect/abort and request-ID reuse pass. All eight cleanup snapshots are
complete and idle, ending with eight successful requests and one abort.

Limits of this evidence:

- The server report records `steps_with_more_than_one_row=0` throughout.
  It does not establish multirow worker execution; API concurrency alone is not
  batching evidence. The archive omits raw Prometheus snapshots, so verify the
  histogram's presence and per-node values in the next measured run before
  attributing the zero to scheduling. The direct driver does exercise real B=2
  execution, separately from worker batch selection.
- The archive contains reports/logs, not NPZ/frame payloads or official trace
  arrays. Local inspection verifies JSON digest consistency and log evidence;
  numerical and byte comparisons were executed by the GPU harness.
- Managed shutdown reports completion, but the log has `KeyboardInterrupt` while
  the conductor joins a worker during process-group SIGINT teardown. Graceful
  process shutdown is not established. The invalid-action traceback is the
  expected validation case.
- Compilation remains disabled. No latency/throughput or speedup was measured.

SHA256 provenance:

| Artifact | SHA256 |
| --- | --- |
| Uploaded ZIP | `36ede42da0c3b45a8c4855879ef1eb8261b8cfb4c4faedf45d82bb42d52fdbcd` |
| Direct report | `5929750ce00e8592146b85951307665a8f92b087ec1bb24b5ac9ced831ad3f73` |
| Server report | `54ceb2332308dc46de54d0fa49d6fe69034a29326db28f746e3ae91aa6f72298` |
| Server log | `4c3f8cfd140fdeb221f28061f6b4adcd61467fb687f951050027df64a56c59a8` |

## What to learn

Explain why a CUDA graph needs stable addresses and shapes while values can
change. Trace which buffers plan N+1 may write while graph N is running, why
padding must describe valid independent rows, and why Python branch choices
belong in a capture key. Distinguish compiling tensor programs from replaying
their launches: this step tests replay without also changing compiler behavior.

Both GPU reports now pass for the recorded scope. The [Step 5c harness](iris-step-05c.md)
is ready for its first GPU run: verify actual worker batch sizes and collect
official/eager/captured baselines before further tuning. Read the repository's benchmarking
skill and use worker phases to identify CPU versus GPU work. Neither successful
capture nor this diagnostic driver's wall time establishes a speedup.
