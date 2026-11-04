# IRIS, step 2: the request and Model contract

Implemented on `codex/iris`, following step 2 of `docs/adding_models.rst`.
Step 1 is commit `69f05b7f`, rebased onto upstream/fork `main` at `a9e81ed8`
and pushed to `origin/codex/iris`. This contract, benchmark plan and reference
tooling were committed and pushed as `8cae2069`.

## What now works

`IrisModel` can be constructed without a checkpoint, network access or reference
package. Its real methods validate a request, create the initial conductor
arguments, report completion of the planned walk, and encode output tensors.

At the Step 2 commit, resource declarations, the computation graph and real
submodule loading raise `NotImplementedError`. Dummy submodule lookup returns `None` for the three known
nodes. This is a CPU-tested contract, not a runnable IRIS server.

## First read these paths

1. `mstar/model/iris/config.py`: checkpoint identity, interface constants and the
   deployment's rollout bound. The exact file/revision/checksum will be enforced
   by the later loader; there is no weight loading here.
2. `mstar/model/iris/iris_model.py`: follow `process_prompt`, then
   `get_initial_forward_pass_args`, then `get_partition_forward_pass_args`.
3. `mstar/model/base.py`: `ProcessPromptOutput`, `ForwardPassArgs`, and the
   `PREPROCESS_ONLY_KWARGS`, `INTERNAL_KWARGS` and `LENGTH_KWARGS` contracts.
4. `mstar/api_server/data_worker.py`: `_prepare_input` and `_finish_input`.
5. `mstar/conductor/conductor.py`: `_do_ingest_request`, `_max_output_tokens`
   and `_loop_caps`.
6. `test/modular/test_iris_model.py`: concrete accepted/rejected examples and
   the actual data-worker handoff test.

## One request is one bounded imagined trajectory

```text
one prepared RGB image + actions[H]
    -> API data worker: validate, name tensors, derive H
    -> conductor: describe imagine walk and route tensor descriptors
    -> future graph: encode image, prime context, predict H frames
    -> client: predicted RGB frames, rewards and ending flags
    -> conductor: finish request after the walk
```

Actions are supplied. No actor, Atari emulator, training or interactive session
is introduced. Each request will start a fresh world-model context; state does
not survive across requests. Ending flags are predictions returned to the caller,
not an early-stop condition: the reference harness also executes every supplied
action. This keeps fixed-horizon benchmark workloads comparable.

`H` is the action sequence's length. The YAML now explicitly sets
`max_rollout_horizon: 20`; this is a configurable deployment limit, not IRIS's
architectural context length. Twenty frames also exercise the reference's context
refresh after nineteen frames. Implementing that refresh belongs to the later
resource/graph work.

## Input contract

The request declares `input_modalities=["image"]` and supplies one image.
`process_prompt` receives the loaded image under `tensors["image_inputs"]` and
actions through `model_kwargs["actions"]`.

| Input | Required representation |
|---|---|
| Image | CPU float32 `[3,64,64]`, finite values in `[0,1]` |
| Actions | Python list of integer IDs, or CPU int64 `[H]` tensor |
| Action values | `0 <= id < 4`; floats, bools and negative/out-of-range IDs rejected |
| Horizon | `1 <= H <= max_rollout_horizon` |
| Text | No text conditioning; prompt is `None` or empty |
| Ordered prompt parts | Absent, or one image part at index zero |

These shapes describe one request, without a batch dimension. The list in
`{"initial_image": [tensor]}` is M*'s tensor-bundle convention, not the batch.
Batching independent requests comes later.

The inherited `Model.load_image` decodes an uploaded file to RGB CHW float32
`[0,1]`. The caller must supply a prepared 64x64 image. We reject other sizes
rather than silently introducing a resize/crop that changes the reference
workload. Original-resolution Atari preprocessing and game evaluation remain
separate work.

Preparation emits:

```python
ProcessPromptOutput(
    new_input_tensors={
        "image_inputs": [],        # discard the superseded input edge
        "initial_image": [image],  # [3,64,64] -> image_encoder
        "actions": [actions],      # [H] -> world_model
    },
    metadata={"iris_horizon": H},
)
```

Noncontiguous tensors are made contiguous without changing their values.
Validation happens on CPU, before admission and GPU scheduling.

## Control plane versus data plane

The data worker retains real tensors. The conductor sees `TensorPointerInfo`
descriptors (shape, dtype, location and identity), and `GraphEdge` describes their
destinations. The conductor does not read the image or execute a neural network.

`PREPROCESS_ONLY_KWARGS=("actions",)` makes `_finish_input` remove the original
action payload from conductor kwargs after storing its tensor edge. The API worker
retains those original kwargs locally. This avoids serializing bulk data twice.

`INTERNAL_KWARGS=("iris_horizon",)` makes `_prepare_input` discard a client-supplied
horizon. The model always derives and returns its own value from validated actions.
The handoff test calls the real preparation and finish methods, with only transport
storage and the communicator replaced at their boundaries. It verifies that
`actions` is absent from conductor kwargs and a forged horizon is overwritten.

`LENGTH_KWARGS=("iris_horizon",)` marks this trajectory length as fixed. The
conductor's `_do_ingest_request` consequently bypasses automatic length fitting;
future resource admission must reject an unsupported trajectory instead of quietly
shortening it. An explicit `max_output_tokens` request is rejected during
preprocessing, since action count defines this workload's length.

`get_max_output_tokens` returns H, and `get_loop_caps` returns
`{"rollout_loop": H}`. The future graph must count one completed frame toward the
output cap, and execute the bounded frame loop H times. Image-token generation
and reward/ending emissions must not each increment that frame count.

## Routing and completion

The initial arguments select the `imagine` walk in the `default` partition.
The image goes to `image_encoder`; actions go to `world_model`. The only step
metadata is the integer horizon. Descriptor shape/dtype checks backstop the
preprocessing contract.

There is one conductor-level walk. The token and frame loops will live inside
its graph. Thus `get_partition_forward_pass_args` reports `request_done=True`
after that walk completes. It does not report completion after a single token.
The initial tensors are listed in `unpersist_tensors`: no later conductor walk
needs them. The future graph/runtime must still hold and route the references
needed by its internal loops.

## Output contract

The initial interface requires
`output_modalities=["video_frame", "scalar", "tensor"]`. These are existing
native M* modalities. Once serving is implemented it will require `stream=True`,
as raw `video_frame` output is streaming-only. There is no OpenAI adapter here.

| Modality | One future frame's output tensor | Encoded payload |
|---|---|---|
| `video_frame` | uint8 `[1,64,64,3]` | RGB24 bytes, 12,288 bytes |
| `scalar` | float32 `[1]`, value -1, 0 or +1 | One little-endian float32 reward |
| `tensor` | bool `[1]` | One ending byte, 0 or 1 |

Each modality will emit H chunks. Associate the nth reward and ending with the
nth predicted frame, not with cross-modality message arrival order. The data
worker adds frame indices, dimensions and RGB24 metadata for video frames.
Playback is 15 fps, matching the official game's recorded-video setting in
`src/game/game.py`; this is presentation timing, not measured generation speed.
The later decoder wrapper must convert its clamped float image to these bytes.
Numerical parity checks should compare float reconstructions before quantization.

## Acceptance and next boundary

From this checkout root:

```bash
.venv/bin/ruff check .
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_model.py \
  test/modular/test_model_registration.py \
  test/modular/test_iris_reference_harness.py \
  test/ci/test_cpu_selection.py -q
git diff --check
```

The local macOS modular-test fixtures stub optional accelerator imports. Passing
CPU tests does not establish a production import, GPU execution, numerical parity
of M* inference, or performance. No GPU code is executed by this step.

Acceptance result on 2026-11-04: the command above passed **72 tests and two
subtests**, including the real data-worker preparation/handoff case. Ruff,
format checks for the changed Python files and `git diff --check` also passed.

Next is step 2a: inspect and declare KV storage, causal attention, learned
positions and sampling, including the context-refresh boundary and tokenizer's
spatial attention. Resource lifecycle work follows `.claude/skills/engine-resources/SKILL.md`.
Graph declarations follow in step 3, and neural computation/loading in step 4.
The subsequent [Step 2a implementation](iris-step-02a.md) adds resource declarations
and `kv_layout` metadata to the original contract shown here.

For learning, explain why the data worker handles real tensors while the conductor
handles descriptors, why the image has no batch dimension here, why H is derived
instead of client-controlled, and why registration/contract tests can pass before
there is a runnable model.
