# IRIS Step 5c: measure the baseline

This follows the [capture correctness gates](iris-step-05b.md#t4-qualification-evidence).
**The batching/phase audit passed on Tesla T4 at clean commit `4b8db96e`.**
Eager multirow worker selection was observed; captured serving used single rows in this
run. **The standalone GPU and uninstrumented three-round timing sweep passed
at `b16393e6`**; see [the baseline results](iris-t4-benchmark.md) for measured
rates, variability, scope and independent checks. This completes the initial
baseline experiment. Retain raw metrics even if an audit fails.

Local validation on macOS with `OMP_NUM_THREADS=2`: 83 focused tests passed
(18 benchmark tests plus IRIS capture/server and worker-phase regressions),
along with 15 CPU-selection checks. Ruff, diff checks and notebook shell/Python
syntax checks passed. This is not a CUDA benchmark result.

The first Colab audit reached eager server startup, then the arena capacity
guard rejected its default 8 GiB per-entity ceiling against 5,752 MiB tmpfs.
The benchmark launcher now supplies 16 MiB segments and a four-segment cap
(64 MiB per entity), with a 128 MiB pin budget, for both arms. It records those
values in `launch.json` and leaves M*'s global defaults alone. A CPU regression
calls the real arena constructor with reported Colab capacity: the old defaults
raise the original error, while both benchmark launch paths pass that guard and
reach native arena creation (intercepted before allocation). The subsequent
GPU startup and audit passed as recorded below, followed by the timing sweep
linked above.
For this correction, all 85 focused CPU checks pass, including the eager and
captured launcher regressions; Ruff and diff checks pass too.

## T4 batching audit evidence

The user supplied `iris-benchmark-audit-02.zip` from clean commit
`4b8db96e39fc1ceeba4b572885d050b08e375b9a`, Tesla T4, PyTorch 2.12.1+cu132,
CUDA build 13.2, FlashInfer 0.7.2 and Triton 3.7.1. The original report bytes
are preserved in [iris-t4-batching-audit.json](iris-t4-batching-audit.json).
This is user-run GPU evidence, inspected locally rather than rerun on macOS.
All seven source/config hashes match the measured checkout's files at inspection.

Local recomputation from the archive's raw before/after `.prom` files matches
all four reported per-node histogram deltas. Each cell completed ten H20
requests, delivered 200 frames and 2,458,600 payload bytes, and ended complete
and idle with no KV pages held. Each has exactly 3,420 world row-steps,
200 decoder row-steps and ten encoder/initial-decoder row-steps.

| Arm | Concurrency | World worker steps | One-row steps | Two-row steps | Mean rows/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 1 | 3,420 | 3,420 | 0 | 1.000 |
| Eager | 2 | 1,915 | 410 | 1,505 | 1.786 |
| Captured | 1 | 3,420 | 3,420 | 0 | 1.000 |
| Captured | 2 | 3,420 | 3,420 | 0 | 1.000 |

The logs contain 100 eager and 133 captured worker-phase records, matching the
report. Captured startup lists all 78 requested buckets (72 world plus six
image-node buckets). Both launch manifests specify the scoped SHM arena budget.
Neither log contains arena-spill, error or traceback messages. No runtime
failure was observed in this audit.

Worker step width is not necessarily neural forward width. In
`mstar/worker/worker_metrics.py`, `WorkerMetrics.step_done` records
`len(inputs)`. The Engine can run an incompatible eager batch through
`Engine._exec_per_request`, issuing one neural forward per row. Summing
`step.world_model.imagine.engine.forward` samples in the eager phase log gives
11,970 calls, equal to its final raw histogram's 11,970 row-steps across all
35 requests (warmup included). This is consistent with its two-row worker
steps being split into single-request forwards. The archive has no per-call
row/key trace, so the precise incompatibility cause is not established.
Do not present the 1,505 two-row steps as proof of batched neural execution.

Captured B=2 **worker selection** remains unobserved for this workload. The
separate direct-engine gate already demonstrated B=2 replay correctness, so
this finding does not establish an inability to capture B=2. Its scheduling
cause is not yet verified. Record the current behavior as a baseline before
considering a scheduling change. The earlier zero-multirow qualification report
also cannot be diagnosed retrospectively from this later run alone.

These boots enabled `MSTAR_PHASE_TIMING=100`. Their latency/throughput fields
are diagnostic and excluded from the final performance comparison. The phase
segments mix node/step shapes; per-step means across arms also reflect different
batching. Do not derive an uninstrumented speedup or a kernel-level cause from
them. The next command uses `--stage timings` to retain the interleaved fresh
process rounds while skipping this completed audit.

SHA256 provenance:

- Archive: `8fc3bd3aa95a4adcf02df868fa3b336fc83d0df96d79c17c279ad343f8ed5eed`.
- Original report: `06e38e927554a5ee0f8f1bfd2bad0f39ba0d2d6b4710e792d523e16f39eccc2a`.

Use [benchmark/iris/README.md](../../benchmark/iris/README.md) for the exact
contracts, commands, parameters and interpretation, or
[benchmark_colab.ipynb](../../scripts/iris/benchmark_colab.ipynb) for setup and
evidence download on Colab. This is a subdivision of the guide's batching and
CUDA graph work, with an experiment to measure the resulting behavior.

## Read these files in order

1. `benchmark/iris/common.py`: fixed images/actions, latency/throughput statistics,
   and raw batch histogram deltas. Missing metrics and observed single-row
   execution have different meanings. Per-node row counts say more than API
   concurrency or a graph bucket's padded width.
2. `benchmark/iris/http.py`: native binary streaming, closed-loop request
   submission, response validation and drained before/after snapshots. The
   versioned benchmarking `server.sh` owns server startup/teardown. Audit boots
   enable worker phases; timed boots disable them. SHM arena is explicit.
3. `benchmark/iris/standalone.py`: the official arm invokes unmodified
   `WorldModelEnv`; the two M* arms invoke the real Engine. Model construction,
   weights loading and engine warmup/capture are outside measured rollouts.
   Request ingest/teardown and synchronous per-frame CPU output delivery are
   included. Captured forwards require a lease and complete capture buckets.
   No qualification trace hooks or per-step output clones are timed.
4. `benchmark/iris/run.py`: CUDA/system/source preflight, sequential processes,
   alternating arm order, warmup, fresh restarts and incremental evidence.
   The controller probes CUDA in a child so it holds no context alongside a
   timed model. `summary.tsv` contains comparable rows, with scope and workload
   dimensions retained. It does not silently merge standalone and HTTP rates.
5. `test/modular/test_iris_benchmark.py`: invalid timings, histogram resets,
   truncated/error responses and concurrency bounds. With cached weights, the
   real eager Engine adapter matches the qualified resource driver through H20
   and repeats exactly after request teardown on CPU. CUDA capture and sampling
   execution still require the notebook run.

## Concepts to understand

**Measurement boundary.** A raw PyTorch rollout does not include HTTP,
preprocessing, conductor, native worker scheduling or transport. Compare all
three standalone arms first, then compare eager/captured HTTP serving separately.
The difference between these scopes is not solely framework overhead: one is a
synchronous caller and the other pipelines work asynchronously.

**Latency versus throughput.** At concurrency two, summing individual request
latencies double-counts overlapping time. Throughput uses the whole closed-loop
interval. Standalone latency at B=2 is the time for one two-row rollout, whereas
HTTP latency is per request. Frames/second keeps the fixed output volume visible.

**Warmup and variance.** First-shape kernel JIT and graph capture must not appear
as steady inference time. Warm each shape/concurrency, then repeat across fresh
processes in alternating order on the same allocation. Inspect the range across
boots, GPU monitoring and host load before attributing a difference to an arm.

**Host versus device work.** Read `benchmark/worker_phases/README.md` and the
benchmarking skill. `event_sync` is the actual GPU wait; a host synchronization
inside another phase moves the wait there. Phase instrumentation adds overhead,
so its diagnostic boots are excluded from throughput comparisons.

**Equivalent work.** H1 and H20 are separate workloads, and H20 crosses a context
refresh. Initial reconstruction, output format, CPU delivery, dtype and sampler
settings matter. The three sampler algorithms can draw different tokens from
the same seed; fixed actions/horizon keep the amount of neural work and output
volume fixed. Numerical parity and output-quality checks remain separate from
the timing result. This experiment retains FP32 and disables compilation; it
does not establish the performance of a fully tuned deployment.

## Acceptance

Run the audit first and inspect per-node B=1/B=2 counts. If metrics are absent or
unchanged, diagnose that evidence path before trusting batching claims. If only
single-row steps are observed, record that result and use the phase evidence to
investigate; do not change the global scheduler on an unverified explanation.

Then collect the full sweep's JSON, TSV, logs, raw `.prom` snapshots and GPU
monitoring. Every request and output schema must validate. Compare the same
scope and workload across the fresh-process rounds. No speedup is claimed until
those GPU results have been inspected.
