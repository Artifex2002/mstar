# IRIS Step 5c: measure the baseline

This follows the [capture correctness gates](iris-step-05b.md#t4-qualification-evidence).
The benchmark implementation is CPU-checked. **CUDA benchmark validation and
performance results are pending.** The first GPU run is the batching/phase
audit; retain its raw metrics even if the audit fails.

Local validation on macOS with `OMP_NUM_THREADS=2`: 83 focused tests passed
(18 benchmark tests plus IRIS capture/server and worker-phase regressions),
along with 15 CPU-selection checks. Ruff, diff checks and notebook shell/Python
syntax checks passed. This is not a CUDA benchmark result.

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
