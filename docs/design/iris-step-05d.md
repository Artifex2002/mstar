# Step 5d: decouple generation from RGB decoding

This is an **opt-in graph experiment**, following the measured baseline in
[iris-t4-benchmark.md](iris-t4-benchmark.md). Its CUDA correctness and performance
are not yet measured. It revisits the Step 3 graph declaration and the Step 5
serving/capture contracts; it does not introduce a new neural architecture.

## The question and the dependency

For supplied actions, the next world-model call reads generated observation
codes, reward/end classes and the action sequence. It never reads decoded RGB.
See `WorldSubmodule.prepare_inputs` and `forward_tensors` in
`mstar/model/iris/submodules.py`. The serial graph nonetheless places the RGB
decoder between successive frame iterations:

```text
Serial:    encode -> [generate codes -> decode RGB] -> next frame
Pipeline:  encode -> generate codes -> next frame
                          |
                          +-> FIFO -> decode RGB -> client
```

In both layouts, a separate initial reconstruction retains the reference's
reset work and emits no predicted frame. H=20 still performs 342 world-model
calls and 21 image decodes, including that initial reconstruction. Sampling,
weights, supplied actions, FP32 precision and context refresh are unchanged.

Putting generation and decoding inside a `Parallel` **within** the frame loop
would retain a join before the next frame. Instead, the experiment gives the
decoder an independent async partition. This exposes scheduling freedom;
it is not evidence of concurrent GPU kernels. All nodes remain on rank 0 in
one worker process, without a side stream or additional GPU. Extra stream
bookkeeping might cost more than the removed dependency saves, and a scheduler
that favors generation could delay delivery. Both are hypotheses to measure.

## Where the change lives

| File | Responsibility |
| --- | --- |
| `mstar/model/iris/config.py` | `decoder_pipeline=False` by default; experiment bounds H to 20 |
| `mstar/model/iris/graph.py` | Producer `imagine` walk carries codes straight into the next frame; consumer `decode_frames` walk loops H times |
| `mstar/model/iris/iris_model.py` | Partitions, three streaming connections, one-item chunk policies, request-specific loop caps and resource/submodule selection |
| `mstar/model/iris/submodules.py` | Emit streamed carriers only on a completed frame; validate FIFO positions; declare decoder capture under its consumer walk |
| `configs/iris_pipeline.yaml`, `configs/iris_capture_pipeline.yaml` | Same deployment as the respective baseline, with only `decoder_pipeline: true` added |
| `benchmark/iris/graph_ab.py` | Correctness gate, separate worker-phase audit and fresh-process HTTP A/B |

There are three streamed signals: `frame_codes`, `frame_reward_class` and
`frame_ending_class`. Distinct names prevent the stream manager from mistaking
the producer's ordinary loop-back inputs for consumers. `StopCheck.withheld`
suppresses these edges on intermediate token calls. Each final frame publishes
one item on each stream. The decoder requires matching `start_offset` values
equal to its frame-loop index, so a dropped or reordered carrier fails loudly.

The decoder's H-iteration consuming loop keeps its partition active between
frames. It finishes after exactly H frames, independently of when the producer
finishes. The conductor owns request completion across both partitions. Existing
stream-buffer and request teardown mechanisms own queued tensors; the model
has no private queue, cursor, resource allocator, stream or capture manager.

## Tensor lifetime and bounds

Accelerator graph outputs may be overwritten on replay. The existing engine
`Engine._merge_per_rid` clones per-request outputs before routing them. The
new test calls that real method with aliased capture-style outputs, overwrites
the original buffers, and verifies the queued carrier snapshots retain their
values. No extra model-level clone or lifecycle mechanism is needed.

The model accepts at most 20 actions in this experimental layout. The diagnostic
server caps concurrency at two, so even if decoding is completely delayed there
are at most 20 produced items per stream per request (40 across two requests).
This is a bounded-rollout experiment, not a general streaming backpressure
implementation. Consumed tensor references are released by the framework.

## Local acceptance checks

`test/modular/test_iris_decoder_pipeline.py` exercises:

- Actual Rust graph runtime, tensor store and stream buffers for H=1, 2 and 20,
  plus mixed H=20/H=2 requests. It deliberately generates the entire rollout
  before decoding, verifies every carrier and FIFO position, and checks that
  the consumer stays active until its Hth frame.
- Deployment validation, independent capture walk, stream alignment checks,
  speculative overshoot and the engine's output-snapshot mechanism.
- Cached real Breakout weights on CPU at B=2/H=20, comparing all generated codes
  and delayed RGB/reward/end outputs against the serial layout, bit for bit.
  That CPU check uses argmax sampling and is **not** a CUDA or RNG qualification.

`test/modular/test_iris_graph_ab.py` checks that failed/stale qualifications block
timing and that the four timing arms run uninstrumented in alternating orders.

Local verification on macOS, PyTorch 2.12.1, `OMP_NUM_THREADS=2`:

```bash
ruff check .
python -m pytest test/modular/test_iris* test/modular/test_partition_done.py \
  test/modular/test_stream_buffer* test/ci/test_cpu_selection.py -q \
  --basetemp=/private/tmp/iris-final-check-01
```

Lint passed; the selected suite passed with **281 passed, 1 skipped and 2
subtests passed**, including the cached real-weights CPU check. A separate
`test_partition_bookkeeping.py` run with socket access and a short temporary
path had **4 passed, 1 failed**: `test_parallel_stream_connections_are_both_tracked`
encountered the macOS test stub's missing `triton.language.int64`. Running that
same module from a clean local `main` snapshot (`b6134ed6`) produced the identical
failure and pass set. No engine, worker or Rust runtime implementation changed.
These checks do not establish CUDA correctness; that gate follows below.

## Run on the GPU

Use [decoder_pipeline_colab.ipynb](../../scripts/iris/decoder_pipeline_colab.ipynb)
for a fresh runtime or the previous benchmark runtime. Activate `.venv` so
FlashInfer can find `ninja`. This experiment adds no dependencies beyond the
existing benchmark setup. Use new output directory names on reruns.

First qualify both eager and captured pipeline servers against the **serial**
local baseline on the exact checkout. This runs the existing seeded H20 output
comparison, request-id reuse, concurrent H20/H20 and H20/H2 cases, refusal,
disconnect/abort, reuse and resource cleanup checks:

```bash
cd /content/mstar-iris
git pull --ff-only origin codex/iris
source .venv/bin/activate
mkdir -p .ci-results/iris
OMP_NUM_THREADS=2 python -m benchmark.iris.graph_ab --stage qualify \
  --download-checkpoint --output .ci-results/iris/graph-qualify-01 \
  2>&1 | tee .ci-results/iris/graph-qualify-01.log
```

Send that directory and log first. A failed qualification blocks measurement.
The top-level controller has no live CUDA context during the server runs.

After qualification passes, inspect scheduling with an **untimed** phase audit:

```bash
OMP_NUM_THREADS=2 python -m benchmark.iris.graph_ab --stage audit \
  --qualification .ci-results/iris/graph-qualify-01/qualification \
  --output .ci-results/iris/graph-audit-01 \
  2>&1 | tee .ci-results/iris/graph-audit-01.log
```

Compare `worker-phases.txt`, raw `.prom` histograms and `server.log` across
serial/pipeline layouts within each capture mode. Check `event_sync`, scheduling,
preparation and postprocess work; do not add wait phases to CPU work. This can
show a scheduling cost or changed batching, not prove GPU kernel overlap.

Then measure, with phase instrumentation disabled:

```bash
OMP_NUM_THREADS=2 python -m benchmark.iris.graph_ab --stage measure \
  --qualification .ci-results/iris/graph-qualify-01/qualification \
  --rounds 3 --requests 10 --warmup-waves 5 \
  --output .ci-results/iris/graph-ab-01 \
  2>&1 | tee .ci-results/iris/graph-ab-01.log
```

The four arms are eager serial/pipeline and captured serial/pipeline. Each uses
a fresh server, the same GPU/checkpoint/actions/H20, C=1/2, five warmup waves per
cell and ten measured requests. Successive rounds reverse arm order. Existing
HTTP client validation and binary output-volume checks apply to every response.
Setup/capture/warmup are outside timing. All servers use the versioned benchmark
launcher and SHM arena limits. Do not pull or edit code between qualification
and measurement; the gate checks commit, clean tracked sources and checkpoint.

Compare FPS, p50/p95 request latency, first-frame latency and batching within a
capture mode. `summary.tsv` retains each round; `report.json` retains individual
requests, GPU/software/source pins and cleanup evidence. This experiment only
changes native serving, so it does not rerun the unchanged official standalone
arm or mix its timing boundary into HTTP comparisons. Archive both graph layouts
on the same new commit; old T4 measurements are context, not the A/B control.

No new speedup is claimed. Retain this layout only if the measurements justify
it across the tested workloads; otherwise remove the opt-in implementation.
