# IRIS T4 baseline results

The supplied `iris-benchmark-01.zip` contains a passed, uninstrumented timing
sweep at clean commit `b16393e6d65a27cac1a2f596792511d81a3aaec2`, starting
2026-11-05 01:53 UTC. These are user-run Colab GPU measurements inspected
locally. The original report bytes are preserved in
[iris-t4-benchmark.json](iris-t4-benchmark.json).

## Hardware, workload and method

- One Tesla T4 (same GPU UUID in every before/after snapshot), PyTorch
  2.12.1+cu132, CUDA build 13.2, FlashInfer 0.7.2, Triton 3.7.1, NumPy 1.26.4.
- Official source `24326aaaa283c527f42b89b44cfdecf2665a7a16`; checkpoint
  SHA256 `169e3669b7d1901990b4ebce35ff8f3355610fae393e6a587fd62c40faf790f4`.
- Breakout world-model rollouts, RGB64 synthetic PNG-derived images, supplied
  actions, fixed H=1/20, FP32, highest matmul precision, TF32 disabled and no
  `torch.compile`. The actor is outside this workload.
- Three sequential, alternating-order fresh-process rounds per arm on one
  allocation. Five warmup rollouts per standalone shape, then five measured
  rollouts; five HTTP warmup waves per concurrency, then ten measured requests.
- Standalone includes request reset/initial reconstruction, inference,
  synchronous per-frame RGB24/reward/end CPU delivery and teardown. Model
  loading and graph capture are outside timing. HTTP additionally includes the
  PNG upload, preprocessing, conductor, native worker, tensor transport and
  streamed response. Compare arms within each scope.
- The sampler/attention implementations differ between arms; prior numerical
  qualification supplies shared-input parity evidence. Sampled outputs are not
  required to agree across algorithms. These results compare configured
  pipelines; they are not an isolated CUDA-graph ablation.

## Standalone throughput

Predicted frames/second: **median (minimum–maximum)** of the three process-level
rates. Each process-level rate uses all five measured rollouts. B is the explicit
neural batch width. H20 includes a context refresh.

| Horizon | Batch | Official PyTorch | M* eager | M* captured |
| --- | --- | --- | --- | --- |
| 1 | 1 | 5.14 (4.08–5.26) | 4.49 (4.36–4.51) | 29.54 (29.24–33.61) |
| 1 | 2 | 9.94 (7.57–10.20) | 6.88 (5.93–7.10) | 61.24 (54.79–61.74) |
| 20 | 1 | 5.60 (5.42–5.68) | 4.77 (4.73–4.93) | 29.25 (28.80–38.31) |
| 20 | 2 | 11.07 (10.75–11.09) | 7.40 (7.29–7.42) | 66.24 (55.74–67.05) |

For H20, the ratio of median captured to median official throughput is **5.23x
at B=1** and **5.98x at B=2**. Eager throughput is **14.8% lower** than official
at B=1 and **33.2% lower** at B=2. At B=1 the median per-rollout mean latencies
are 3.574 s official, 4.194 s eager and 0.684 s captured. At B=2 they are
3.613 s, 5.406 s and 0.604 s respectively, for a two-row rollout.

The captured H20 B=1 rate varies 28.80–38.31 frames/s across boots. Per-round
captured/official throughput ratios span 5.15–6.75x (B=1) and 5.03–6.16x (B=2).
The ratio-of-medians headline is a summary of this allocation, not a confidence
interval or a general hardware/model guarantee.

## HTTP throughput and latency

All cells use H20. C is client concurrency, not neural batch width. Rates are
predicted frames/second. Latencies are the median across boots of each boot's
per-request p50; first-frame latency is to the first complete video chunk.

| Arm | Concurrency | Frames/s, median (range) | Request p50, ms | First-frame p50, ms |
| --- | --- | --- | --- | --- |
| M* eager | 1 | 3.59 (3.55–3.62) | 5538.19 | 307.05 |
| M* eager | 2 | 3.76 (3.68–3.81) | 10687.32 | 616.75 |
| M* captured | 1 | 15.50 (14.96–15.72) | 1249.54 | 79.06 |
| M* captured | 2 | 15.16 (14.58–15.27) | 2536.79 | 154.56 |

Captured/eager ratios of median HTTP throughput are **4.31x at C=1** and
**4.03x at C=2**. Every fresh-process paired comparison favors captured:
4.21–4.34x at C=1 and 3.97–4.03x at C=2. C=2 does not increase captured serving
throughput in this experiment, even though explicit B=2 standalone throughput
is higher than B=1. This identifies a serving investigation, not its cause.

At C=2 every eager measured cell reports 1,915 world worker steps (410 one-row
and 1,505 two-row) for 3,420 row-steps; every captured cell reports 3,420 one-row
steps. The [earlier phase audit](iris-step-05c.md#t4-batching-audit-evidence)
explains why worker width does not establish neural forward width: eager phase
counts are consistent with per-request splitting. The timing boots contain no
phase instrumentation, so do not treat that earlier diagnosis as a new per-call
trace of these boots.

## Independent local checks

- All seven benchmark/config hashes match the measured checkout's files at
  inspection. The final report is `passed`, and the controller's source check
  refused tracked edits or a commit change during the sweep.
- All 36 standalone cells match their child JSON reports. Recomputing every
  duration summary reproduces the recorded rates/latencies: 180 measured batched
  rollouts, 2,835 predicted frames. Per-cell output hashes also agree across all
  three fresh boots within each arm; cross-algorithm equality is not asserted.
- All 12 HTTP cells match their per-server JSON reports. Recomputed timing
  summaries match: 120 measured requests, 2,400 predicted frames, zero failures.
  Every cell delivers 2,458,600 payload bytes and finishes complete/idle, without
  aborts or retained KV pages.
- Recomputing all 12 before/after histogram deltas from the 24 raw Prometheus
  snapshots exactly reproduces the report. Every H20 cell has the expected
  3,420 world row-steps.
- The six HTTP launch manifests set `MSTAR_PHASE_TIMING=0`; the six logs contain
  no phase records, errors, tracebacks or spill messages. Each captured boot
  lists all 78 capture buckets; eager boots list none. The standalone captured
  driver independently requires complete buckets and a replay lease per step.
- Every per-arm GPU snapshot reports the same GPU UUID and zero allocated GPU
  memory before/after the arm. Continuous GPU monitoring and host load snapshots
  are included, but do not prove absence of all external contention.
- All 48 summary TSV rows agree with the JSON. The old failed `benchmark-audit-01`
  also present in the ZIP is separate historical evidence and is excluded here.

No GPU commands were rerun on the local macOS host. The archive contains hashes
and validation results, not standalone frame arrays, so local inspection cannot
repeat the output-byte comparisons. Output-quality/task-score evaluation, other
checkpoints/hardware, a vLLM-Omni port, and attribution of the gains to individual
mechanisms remain separate work. No runtime optimization was made from these
measurements.

## Reproduce and next investigation

See [benchmark/iris/README.md](../../benchmark/iris/README.md) and
[benchmark_colab.ipynb](../../scripts/iris/benchmark_colab.ipynb). Keep the
same scope and workload dimensions when comparing results. The baseline stage
is complete for this recorded experiment. Before optimizing serving, establish
why requests fail to form neural batches using targeted phase/call-key evidence;
retain this baseline and remeasure any change with the same harness.

SHA256 provenance:

- Uploaded ZIP: `e3445c140a5be684accf1f8261febb9693546b9a24cda5e3e47d58a378e54329`.
- Original timing report: `9eee20acbb75c1f4c46f9913ec46bae621592e248ee2257ebb03f9b1b4f673f2`.
