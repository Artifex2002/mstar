# IRIS baseline benchmark

This follows the [passed T4 correctness gates](../../docs/design/iris-step-05b.md#t4-qualification-evidence).
The benchmark is CPU-checked; **its CUDA execution and performance results are
pending**. Do not infer a speedup from successful capture or from qualification
wall times. Run on one otherwise idle Linux NVIDIA allocation, using the same
GPU and checkout for every arm. Stop the qualification servers first.

## What is compared

| Scope | Arms | Workload | Measured boundary |
| --- | --- | --- | --- |
| Standalone synchronous rollout | Pinned official PyTorch, M* eager Engine, M* captured Engine | B=1/2, H=1/20 | GPU-resident input through reset, initial reconstruction, H supplied actions, per-frame RGB24/reward/end delivery to CPU and request teardown |
| HTTP closed loop | M* eager, M* captured | Concurrency 1/2, H=20 | PNG upload, preprocessing, conductor/native worker, transport and complete binary response |
| Untimed batching/phase audit | M* eager, M* captured | Concurrency 1/2, H=20 | Worker batch histograms and `benchmark.worker_phases` records; excluded from throughput comparisons |

Compare numbers within a scope and matching B/concurrency/horizon. Standalone
latency is per batched rollout, not per individual row. HTTP latency is per
request, from its actual submission (excluding client queue time) to full stream
receipt. First-frame latency is to receipt of the first complete video chunk.
HTTP throughput uses total closed-loop wall time, not summed request latencies.
That wall time includes client response validation; individual response latency
stops before schema checks/hashing. Input PNG generation is outside timing.

All arms use the pinned Breakout checkpoint, the same RGB64 PNG-derived images,
action patterns, H, FP32, highest matmul precision and disabled TF32. M* uses
production submodules/resources with compilation disabled, matching qualification.
The official arm calls the unmodified `WorldModelEnv.reset_from_initial_observations`
and `.step`: its per-frame reward/end CPU synchronization is included. M*'s
standalone arm also delivers every frame/reward/end to CPU synchronously. Initial
reconstruction is computed by all arms, but not delivered as a predicted frame.
No trace hooks or diagnostic fingerprints are in the timed path. The standalone
Engine adapter does not emulate the worker, scheduler or asynchronous serving.

The sampler algorithms differ: official PyTorch `Categorical`, eager FlashInfer,
captured `fused_sample`. Identical seeded samples are not required. Supplied
actions and fixed horizons keep output volume and neural call counts fixed;
reward/end flags never terminate the rollout. These are reference-preserving,
uncompiled baselines, not claims about optimally tuned M* or another engine.
There is no vLLM-Omni arm in this harness.

## Run on Colab

[benchmark_colab.ipynb](../../scripts/iris/benchmark_colab.ipynb) supports both
the previously prepared runtime and a fresh runtime. Its setup reuses the
qualification environment, installs the Linux listener-management tools and
rebuilds the Rust extension from the checkout. This change adds no Python
dependency beyond that environment. Use a new output directory on each rerun.

In the existing prepared runtime, update and run the audit first:

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
git pull --ff-only origin codex/iris
source .venv/bin/activate
# server.sh checks listener ownership using ss; Colab may lack iproute2.
if ! command -v ss >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y iproute2
fi
mkdir -p .ci-results/iris
OMP_NUM_THREADS=2 python -m benchmark.iris.run --stage audit \
  --download-checkpoint --output .ci-results/iris/benchmark-audit-01 \
  2>&1 | tee .ci-results/iris/benchmark-audit-01.log
```

Send the audit directory and log first, especially if it reports a missing or
unchanged histogram. A `passed` audit can still show only single-row forwards;
read `world_multirow_observed` and each component's `two_row_steps`. Missing
metrics are an error, never evidence of zero batching. Keep the raw `.prom`
files: the earlier qualification archive did not include them.

After the audit is understood, a complete sweep repeats the audit and runs
three interleaved rounds, each with fresh model/server processes:

```bash
%%bash
set -euo pipefail
cd /content/mstar-iris
source .venv/bin/activate
mkdir -p .ci-results/iris
OMP_NUM_THREADS=2 python -m benchmark.iris.run \
  --download-checkpoint --rounds 3 --samples 5 --requests 10 --warmup-waves 5 \
  --output .ci-results/iris/benchmark-01 \
  2>&1 | tee .ci-results/iris/benchmark-01.log
```

This starts several servers and can take many minutes on T4. Per-stage logs and
results are saved incrementally. `--stage standalone` and `--stage http` run
those scopes separately after the audit; retain the same allocation when
comparing arms. The benchmark never checks out or pulls code during a sweep.
It refuses tracked modifications. `--port` selects a free loopback server port
(default 8123). On a cluster, activate the environment and set
`CUDA_VISIBLE_DEVICES` to the GPU allocated to this job; the server uses its
visible device 0. The launcher requires `ss`, `setsid`, `curl`, `ps` and
`nvidia-smi` on PATH (Debian/Ubuntu: `iproute2`, `util-linux`, `curl`, `procps`).

## How to read the evidence

Start with `report.json`: `status`, `stage`, errors, source/checkpoint pins,
hardware/software, precision/sampler contracts and workload settings. Every
failed or truncated response invalidates its cell. Standalone setup, weights
loading and graph capture are outside timing, followed by five warmup rollouts
per shape and five measured rollouts. Each HTTP cell warms five waves at that
cell's concurrency, then measures ten requests (at least 5x maximum concurrency).
`summary.tsv` keeps each fresh-process round separate. Standalone output hashes
are computed outside timing and must repeat within one process after request
teardown; cross-algorithm hashes are not required to agree.

`batching_audit` has per-node deltas from drained before/after metric snapshots:
step count, row-step sum, B=1/B=2 step counts and mean rows. These describe the
forward's actual rows, not requests in flight or padded graph bucket width.
For each audit server, `worker-phases.txt` uses the repo's existing parser and
segmentation. Check steady segments in `server.log` before interpreting means.
`worker.postprocess.event_sync` is the real GPU wait; `await_gpu` is a host-thread
wait. Do not add waits to work totals or infer GPU utilization from their sum.
See [worker_phases/README.md](../worker_phases/README.md).

Timed HTTP boots set `MSTAR_PHASE_TIMING=0`, use the versioned benchmarking
`server.sh` for ownership-checked startup/teardown, and explicitly use
`MSTAR_SHM_ARENA=1`. Captured boots require every declared graph bucket and
standalone captured steps require a replay lease. The engine's leased-forward
path calls `AcceleratorGraphRunner.run_forward`; no diagnostic wrapper is timed.
Check per-arm logs for unexpected fallback/errors before accepting a comparison.

The sweep reverses arm order on alternate rounds and holds no controller CUDA
context. Model processes run sequentially. `gpu-monitor.csv`, per-arm GPU
snapshots and host load averages help identify contention. Compare the spread
across fresh boots, not only the fastest sample; keep H=1 and H=20 separate.
Output lengths are fixed, but content may differ by sampler. Inspect output
volume and quality independently before making broader performance claims.

The download cell includes raw metrics, logs, generated serving configs,
per-request timings and the top-level report even after a failure.
