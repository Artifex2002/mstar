# IRIS Step 5 continuation: validate the real server

The earlier [T4 qualification](iris-t4-qualification.json) passed the eager
neural/resource checks at `caf2f45a`. Its driver called real resource lifecycle
methods directly. This continuation validates M*'s normal server, conductor,
graph runtime and worker; those layers were outside that driver.
The [full-server T4 run](iris-t4-server-qualification.json) passed at clean commit
`ac98d64a`. This remains eager integration validation; CUDA graph capture and
performance measurement follow.

## Read the implementation in this order

1. [The run guide](../../scripts/iris/SERVER.md) describes the workload and evidence.
2. `scripts/iris/server_checks.py` constructs multipart `/generate` requests and
   reads M*'s public binary-frame format. Follow `generate`, `StreamCheck.add`,
   `metric_snapshot` and `wait_idle`. Rewards and ending tensors are binary data;
   decoding them as text would silently lose their meaning.
3. `scripts/iris/qualify_server.py` first builds a seeded local baseline, then
   launches the production server using the existing `benchmark.eval.server.ManagedServer`
   lifecycle helper. Read `local_baseline`, `qualify`, then `suite`. The generated
   config points all four nodes to one GPU and uses the existing eager deployment.
   `prepare_server_config` runs the conductor's actual deployment validator
   before checkpoint download or GPU inference. The serving YAML must declare
   `max_seq_len: 340` for the context window, separately from the action horizon.
4. `test/modular/test_iris_server_checks.py` covers malformed/incomplete streams,
   output divergence, missing/nonidle metrics, disconnect handling on the client,
   missing-CUDA reporting, and the real CPU image-loader/serialization baseline
   when the checkpoint is cached. These CPU tests do not run a GPU server.
   The deployment regression also validates the generated serving YAML without
   weights; it reproduces the missing-`max_seq_len` startup failure before the fix.
5. `test/modular/test_iris_serving_metadata.py` is a standalone CPU regression
   for the first world-model step. It uses the real request-state manager,
   normal/columnar worker snapshot helpers and engine wrapper before calling
   IRIS's real `prepare_inputs` and `check_stop`. A new request can have an empty
   snapshot: missing loop counters mean iteration zero. The snapshot remains
   authoritative when shared counters advance. Cases cover first entry, token
   and frame advancement, context refresh, stopping and speculative overshoot.

No worker, engine, graph-runtime or model behavior is changed by this harness.
The `iris` installation extra supplies runtime dependencies; `metrics` is an
existing extra. Rebuild `rust/` from the tested checkout before serving.

The second Colab server run (`e5ba135b`, Tesla T4) passed startup and the local
H=20 baseline, then failed on HTTP warmup: IRIS indexed an absent `rollout_loop`
counter. The standalone metadata regression reproduced that `KeyError` without
a GPU. IRIS now reads sparse counters with a zero default in preparation and
stopping, matching the worker's initial/snapshot contract. No worker scheduling
or resource lifecycle change was needed. The next run, at `ac98d64a`, passed
the complete eager full-server gate described below.

## Trace the M* path

Start at `/generate` in `mstar/api_server/entrypoint.py`. The data worker loads
the PNG and calls `IrisModel.process_prompt`; the conductor admits the request
and advances its graph. The worker schedules node inputs. The engine executes
declared attention/KV/position/sampler resources and submodules. Tensor edges
return outputs to the data worker, which calls model postprocessing and streams
framed bytes. The graph runtime and control transport use the Rust extension.

Follow an H=2 request on paper: image encoding and initial reconstruction occur
once, the world model consumes action 0 and predicts the first observation,
the decoder emits its frame/reward/ending, then action 1 produces the second.
The request ends after two actions even if a predicted ending flag is true.
For H=20, also identify the fresh cache label used at the context refresh.

## Concepts to learn and explain

- **Numerical equivalence versus system correctness.** A correct forward can
  still receive the wrong tensor, cache row, request identity or loop position.
  Seeded local-versus-server output checks add a second integration boundary to
  the official-versus-M* numerical comparisons already completed.
- **Concurrency versus batching.** Two requests can be in flight while executing
  separate forwards. Their phases must be compatible for this IRIS implementation
  to batch them. Inspect recorded batch-size metrics before making batching claims.
- **Ownership and teardown.** A finished HTTP stream does not by itself prove
  worker resources are freed. Conductor drain state and KV gauges provide
  observable cleanup evidence. Fresh status and a stable zero window help avoid
  accepting an old idle snapshot.
- **Cancellation and identity reuse.** A disconnected client must trigger an
  abort and release state. Reusing a client ID afterward checks the public
  lifecycle; it does not prove which recycled worker-local integer handle was
  allocated. The earlier resource-driver test checks handle reuse directly.
- **Transport contracts.** Image loading, tensor edges, output postprocessing
  and binary framing are real work across process boundaries. Keep this
  complete serving boundary separate from model-only timing later.

## Acceptance and the next step

Run [the notebook](../../scripts/iris/qualify_server_colab.ipynb) or equivalent
commands and review the complete report plus server log. A failure is evidence
to investigate at the named stage, not a reason to add speculative fixes to
several layers. Preserve the commit and environment in every rerun.

The user-provided `iris-server-check-03.zip` records a passed run on Tesla T4,
PyTorch 2.12.1+cu132, CUDA build 13.2, FlashInfer 0.7.2 and Rust extension 0.1.1.
The [qualification record](iris-t4-server-qualification.json) retains case
results, cleanup snapshots, artifact hashes and limits. Local inspection
independently verified the saved HTTP arrays against the local eager baseline
and checked all eight raw Prometheus snapshots against the report.

| Check | Observed result |
| --- | --- |
| H=20 HTTP versus local eager driver | Frames, rewards and endings bit-exact, including context refresh |
| Same client ID and seed after completion | Second H=20 rollout also bit-exact |
| H=20/H=20 and H=20/H=2 | Both pairs reached two API requests in flight and returned their exact output counts |
| Invalid Breakout action 4 | HTTP 400 with an action validation message |
| Disconnect after first frame | Abort counter incremented, resources drained, later same-ID request succeeded |
| Cleanup after every case | API/conductor/pending steps/world KV idle with fresh status; final counters: eight successes, one abort |

The local baseline here is M*'s resource driver. The earlier
[official-versus-M* numerical run](iris-t4-qualification.json) supplies the
separate reference comparison. Multirow scheduling appears in the metrics,
but those histograms alone do not establish that every world-model step ran
one batched forward. The log's invalid-action traceback is the intentional
negative case. At final process-group teardown it also contains a
`KeyboardInterrupt` while the conductor joins a child after SIGINT; this
qualification does not claim graceful process shutdown. No performance result
comes from this run.

Proceed to capture-compatible attention and
engine-managed CUDA graphs. Those changes need their own numerical, lifecycle
and capture/replay checks. Afterward add a non-diagnostic benchmark under
`benchmark/`, read the repository's benchmarking skill, and measure official
PyTorch, M* eager and M* optimized execution at matching workloads. None of the
current diagnostic runs is a performance result.
