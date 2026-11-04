# IRIS, step 3: declare the computation graph

Step 2a is committed and pushed as `0bde3c29`; `91966ae8` isolates its CPU
lifecycle test from accelerator staging on hosts with MPS. The Step 3 changes
are separate and uncommitted.

## What this step implements

`IrisModel.get_graph_walk_graphs()` now returns one `imagine` walk built from
M*'s `GraphNode`, `Sequential`, `Parallel` and `Loop` primitives. It describes
dependencies, carried tensors, client outputs and loop bounds. It does not
execute the neural network or load weights.

Read these files in order:

1. `mstar/model/iris/graph.py`: `imagination_graph` and `frame_calls`.
2. `mstar/model/iris/iris_model.py`: the graph hook and per-request loop caps.
3. `configs/iris.yaml`: placement of the four logical nodes on rank zero.
4. `test/modular/test_iris_graph.py`: structural checks and a synthetic-tensor
   driver over the real Rust runtime, using M*'s worker-graph construction.
5. `mstar/graph/runtime/rust.py`: `worker_graph_args`, the construction boundary
   between Python graph declarations and the Rust runtime.

## Dataflow

```text
initial_image -> image_encoder -> observation_tokens
                                    |           |
                                    |           +-> initial_image_decoder -> discard reconstruction
                                    |
actions -----------------------> rollout_loop (H frames)
                                   |
                                   +-> token_loop (17 or 18 world_model calls)
                                   |       |
                                   |       +-- observation_tokens, reward_class, ending_class
                                   |           carried between calls
                                   |
                                   +-> image_decoder -> frame, reward, ending -> client
                                           |
                                           +-- carry generated codes/classes into the next frame
```

`Sequential` and `Parallel` declare composition and dependencies; the worker
still schedules ready nodes. Here the one-time initial reconstruction and the
rollout independently consume the encoded image tokens, so their graph sections
are parallel. Request completion waits for both. No request-level state dictionary
or simulator is introduced.

The four node names are logical execution identities. YAML maps them to hardware.
`initial_image_decoder` is a second execution site for the same decoder weights;
step 4 must share its decoder component with `image_decoder` when they run in the
same process/device. It must not initialize unrelated or duplicate decoder weights.
The existing `decoder_attn` resource now names both decoder sites.

Why include the initial reconstruction? Official
`WorldModelEnv.reset_from_initial_observations` performs image encoding, context
prefill and an initial decode. Its initial RGB reconstruction is not one of the
H predicted frames. Including the decode retains that computation for complete
rollout comparisons. Its independent scheduling is legal in inference mode;
the M* graph does not promise the reference's exact Python operation order.

## Inner call schedule

`frame_calls(frame_index)` is the host-known schedule for one frame. `WorldCall`
names each call's kind, cache label, starting position, input image-code index
and sampled image-code destination. Its span is 16 for prefill and one otherwise.

| Call | Input | Sampling |
|---|---|---|
| `prime`, only for frame 0 or a context refresh | Latest image's 16 codes | None |
| `action` | Supplied action for this frame | Reward, ending, image code 0 |
| `observation`, input indices 0 through 14 | Previous sampled image code | Image codes 1 through 15 |
| Final `observation`, input index 15 | Last sampled image code | None; commit its KV |

The final call matters even though it samples nothing: the next action must see
that last image code in context. Normal frames therefore take 17 calls. Frames
0, 19, 38, ... take 18 including a new-context prefill. A 20-frame rollout takes
342 world-model calls and 21 decoder calls, including the initial reconstruction.

Keep these immutable schedules once per configuration when implementing the
submodules; do not rebuild tuples per row on the execution hot path. Step 4 will
use the runtime's `dynamic_loop_iter_counts` for the frame and inner-call index,
rather than incrementing model-owned cursors.

## Carried tensors and outputs

| Name | One request's shape/dtype | Role |
|---|---|---|
| `actions` | `[H]`, int64 | Unchanged conditioning, held by enclosing loops |
| `observation_tokens` | `[16]`, int64 | Latest image, progressively replaced by generated codes |
| `reward_class` | `[1]`, int64 | This frame's sampled reward class |
| `ending_class` | `[1]`, int64 | This frame's sampled ending class |

The encoder wrapper will emit its real image tokens and seed the class carriers
with reward class 1 and ending class 0. Those initial carriers are not predictions
or client outputs; the first action call overwrites them with actual samples.
The world-model node preserves the class carriers on other calls. Every carried
edge must be produced for every row on every call, as required by speculation.

The inner loop exposes its last carried values to the decoder exactly once per
frame. It does not accumulate all intermediate buffers. The decoder feeds the
completed token row back to the next frame unchanged. Decoding followed by
re-encoding the generated RGB would add computation and could change the tokens;
it is absent from the reference rollout.

Routed inputs must remain immutable to their producers/consumers. The initial
decoder may read the same encoded token tensor while the world-model path starts;
progressive sampling must create a new carried value, not overwrite that shared
input tensor.

`image_decoder` emits `frames` as `video_frame`, `reward` as `scalar` and `ending`
as `tensor`, using the Step 2 serialization contract. Only `frames` has
`conductor_new_token=True`: the request budget counts predicted frames rather
than image codes or its three emitted modalities. The initial reconstruction is
discarded, and produces no client chunk or output-budget increment.

## Bounds and stopping

The outer `rollout_loop` has the deployment maximum as its ceiling; the request
cap is H. The inner `token_loop` has a ceiling/cap of 18. Its actual frame length
is `len(frame_calls(frame_index))`, either 17 or 18.

Step 4 must register an inner-loop stop after the scheduled final call, using
host-known loop indices. Ending predictions must not stop the fixed-horizon
rollout. A speculative call beyond the scheduled frame length must exit during
input preparation before GPU work or resource mutation. The current graph keeps
the default async capability, but there are no executable submodules yet; these
guards are required before enabling real execution.

The graph runtime performs the iteration transitions and resets the inner loop
for each outer iteration. Selecting a fresh epoch label re-primes positions at
zero, as declared in Step 2a. No model code allocates, resets or frees KV storage.
Completed epoch labels still remain allocated until request teardown; that
earlier memory tradeoff remains.

## Validation and its boundaries

The native routing tests construct worker graphs through `get_worker_graphs`
using the deployment YAML, then instantiate `RustGraphRuntime` and the real Rust
tensor store/bookkeeper. Synthetic CPU tensors replace neural computation only.
The driver uses real readiness, input pops, loop counters, stop registration,
routing, frame counts and request cleanup. It checks:

- H = 1, 2, 20 and 39, including two context refreshes.
- Interleaved requests with different horizons and distinct token contents.
- One encoder, one initial decode, H predicted-frame decodes, and the reference
  count of world-model calls.
- Complete token rows and corresponding reward/ending classes at every decode.
- Three client outputs per predicted frame, with one output-budget increment.
- Intermediate outputs released; only the held action conditioning survives
  until request teardown, after which the tensor store is empty.

A separate check compared all 342 scheduled inputs, spans and positions with the
saved official 20-frame trace in `.ci-results/iris/context-refresh/trace.npz`.
That trace is local and ignored by Git; the routing tests do not need it, a
checkpoint, or the external reference package. Structural tests run without the
extension; native routing cases skip explicitly until it is built.

No Rust source/runtime behavior was changed. These tests establish graph
structure and routing, not learned predictions, numerical parity of the M* neural
port, CUDA correctness or speed. `get_submodule` still raises outside dummy mode.
FP32 GPU kernel qualification remains required before equal-precision benchmarks.

The local extension was built from this checkout using `maturin develop --release
-m rust/Cargo.toml` with an isolated Rust toolchain. See `docs/installation.rst`
and the [reference-tool README](../../scripts/iris/README.md) for build requirements.

Acceptance result: **233 tests and two subtests passed**, with the native cases
running against the freshly built extension. Ruff, formatting and
`git diff --check` passed. The focused regression command was:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_graph.py \
  test/modular/test_iris_model.py \
  test/modular/test_iris_resources.py \
  test/modular/test_iris_reference_harness.py \
  test/modular/test_model_registration.py \
  test/ci/test_cpu_selection.py \
  test/modular/test_graph_io.py \
  test/modular/test_position_fork_rows.py \
  test/modular/test_ragged_attention_config.py \
  test/rust/test_runtime_parity.py \
  test/rust/test_runtime_loop_output_ownership.py -q
```

Next is step 4: implement the submodules, the exact checkpoint loader and
resource-step declarations, then compare actual tensor computations against the
official reference before compilation or capture optimization.
