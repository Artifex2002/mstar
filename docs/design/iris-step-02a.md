# IRIS, step 2a: declare engine resources

Step 2 and the benchmark plan were committed and pushed as `8cae2069`.
This step adds declarations and CPU resource-lifecycle checks. Graph execution,
neural computation, checkpoint loading and GPU qualification remain later work.

## Read the declarations first

1. `mstar/model/iris/config.py`: checkpoint geometry and shared resource keys.
2. `mstar/model/iris/resources.py`: resource factories, context segments and
   per-request cache footprints.
3. `mstar/model/iris/iris_model.py`: `get_node_resources`, `kv_labels`,
   `get_request_resource_configs` and the new `kv_layout` preparation metadata.
4. `configs/iris.yaml`: deployment page size and pool capacity.
5. `test/modular/test_iris_resources.py`: real CPU admit/plan/commit checks.

The repository's `.claude/skills/engine-resources/SKILL.md` explains why models
declare resources instead of allocating caches or planning attention themselves.
No engine, worker or Rust implementation changes are introduced here.

## The eight resource declarations

| Resource | Node | Reason |
|---|---|---|
| `world_kv` | `world_model` | Paged K/V for 10 layers, 4 heads, head dimension 64 |
| `world_attn` | `world_model` | Attention over that cache; the future step must declare causal attention |
| `world_pos` | `world_model` | Position IDs for the learned absolute-position embedding |
| `observation_sampler` | `world_model` | Categorical draws over 512 image codes |
| `reward_sampler` | `world_model` | Categorical draws over 3 reward classes |
| `ending_sampler` | `world_model` | Categorical draws over 2 ending classes |
| `encoder_attn` | `image_encoder` | Bidirectional spatial self-attention, without persistent K/V |
| `decoder_attn` | `image_decoder` | The decoder's corresponding spatial attention |

`world_attn` and `world_pos` depend on `world_kv`. The engine resolves those
dependencies and plans KV first. Sampling and cacheless spatial attention have
no KV dependency. Each node receives only its declared resources.

IRIS uses learned position embeddings, not rotary embeddings. M*'s
`PositionSpec` currently builds a manager named `RopeManager`, but it also owns
the ordinary position counter. IRIS will read its planned IDs and index the
checkpoint's embedding table; it must never call `apply_qk` or rotate Q/K.

## Preserving the context refresh

The pinned reference has 16 image tokens, 17 tokens per predicted frame, and a
340-token context. Therefore one context holds:

```text
16 + 19 * 17 = 339 tokens
```

Before frame 20, `WorldModelEnv.step` re-primes a fresh cache with the latest
image's 16 tokens and restarts learned positions at zero. It does not implement
a sliding window retaining the last 340 old K/V values: those old values include
information from an earlier history, whereas re-priming recomputes them from
the latest image alone.

The current `KVStep` has no stream-reset declaration. Instead, this initial port
names each independent context `epoch_0`, `epoch_1`, and so on. The paged manager
and position manager already isolate state by request and label. A new label
therefore starts empty and at position zero. The future graph must route the
latest image's token IDs into its new-label prefill, and subsequent steps must
attend only that label.

`context_epochs(H)` specifies the frame and token counts of each segment:

| H | Context token counts |
|---|---|
| 1 | 33 |
| 19 | 339 |
| 20 | 339, 33 |
| 38 | 339, 339 |
| 39 | 339, 339, 33 |

This is a storage-policy choice, not a change to the context mathematics. It
also has a cost: old labels remain allocated until request teardown. The
reference reuses one fixed cache. The initial M* footprint therefore grows with
the number of context segments, within the configured request horizon. A future
declarative resource-lifecycle extension could recycle old segments; model code
must not call manager reset/free methods to do it.

Report this limitation in memory comparisons. Avoid describing the initial port
as having the reference's constant per-request KV footprint over arbitrarily
long trajectories.

## Declare the complete request footprint

`process_prompt` now returns `kv_layout` beside `iris_horizon`. Each used label
has one fixed `rows` span covering all its known writes in the `imagine` walk.
There is no prefix reuse. These spans describe reservations and position bounds;
they do not mean that all neural tokens are computed at once.

Since the action sequence fixes every write count, reserving the whole footprint
also accounts for completed contexts held until teardown. There is no separate
loop-growth term to double-count those writes.

The data worker converts each `LabelLayout` through `.row()` before sending it
to the conductor. `get_request_resource_configs` opens only the labels that this
request needs. The model's `kv_labels` lists every label permitted by the
deployment's maximum horizon.

At page size 16, a 20-frame request needs:

```text
ceil(339 / 16) + ceil(33 / 16) = 22 + 3 = 25 request pages
```

The YAML declares 64 pool pages, including one sink page used by the engine.
The pool can hold two such full-footprint requests simultaneously. Higher
concurrency, longer horizons and offload capacity are deployment sizing choices
under `resources.world_kv`; they do not belong in architecture constants.
Admission validates the request's footprint before GPU work. Tests show that a
25-page pool, which leaves only 24 request pages, rejects this request.

## Sampling and tokenizer attention

The official environment samples `Categorical(logits=...)`. All three sampling
configs explicitly use temperature 1, top-k disabled, top-p 1 and no repetition
penalty. Do not inherit M*'s generic temperature default of 0.6. Ending samples
are returned to the caller without shortening the fixed-horizon workload.

Each sampler has an independent engine RNG stream derived from the request seed
and resource key. The distributions must match the reference; identical seeds
are not a promise of identical categorical draws across implementations.

The tokenizer's `ch_mult=[1,1,1,1,1]` keeps its attention width at 64. Its
attention blocks use one head over spatial grids at 16x16, 8x8 and the 4x4 middle
block. Declare separate cacheless resources for encoder and decoder. Each
resolution will have its own label and noncausal `AttentionStep`; the maximum
per-label span is 256. The ragged manager plans each label independently, so one
image segment per request per label suffices.

## What is verified and what remains

CPU checks cover geometry, resource ownership and dependencies, YAML overrides,
request footprints, sampler parameters and seed streams, plus real KV and
position managers under `StepRunner`. The lifecycle test admits, plans and commits
the initial 16-token prefill and all 17 single-token calls per frame. It verifies
frame-end cache lengths `33, 50, ..., 339, 33`, new-label position IDs starting at
zero, and the actual 25-page total. Transport is stubbed; resource lifecycle
methods run unchanged.

The graph and submodules are still unimplemented. CPU planning tests do not
establish GPU attention or sampling numerics. In particular, the FP32 reference
requires qualification of the selected paged and ragged GPU attention kernels;
declaring FP32 tensors is not evidence that a backend supports the full path.
Resolve any unsupported precision in the engine's opt-in resource/backend layer
before claiming equal-precision parity or timing results. Do not silently
downcast the M* model to make a kernel run.

The later `declare_step` implementations must provide causal world attention,
noncausal spatial attention, matching KV/position segments and sampling only on
the appropriate calls. Those effects cannot be tested end to end before the
submodules exist. No CUDA execution or performance measurement occurred here.

From the checkout root:

```bash
.venv/bin/ruff check .
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  test/modular/test_iris_model.py \
  test/modular/test_iris_resources.py \
  test/modular/test_iris_reference_harness.py \
  test/modular/test_model_registration.py \
  test/ci/test_cpu_selection.py \
  test/modular/test_position_fork_rows.py \
  test/modular/test_ragged_attention_config.py -q
```

Acceptance result: **90 tests and two subtests passed**. Ruff, format checks for
the changed Python files and `git diff --check` passed. The declared geometry was
also checked directly against the pinned reference's world-model and tokenizer
YAML files at `24326aaaa283c527f42b89b44cfdecf2665a7a16`.

Adding the broader `test_resource_runner.py` suite produced **123 passed and
three failures**. All three failure names were reproduced on a clean archive of
upstream `main` at `a9e81ed8`, using the same Python environment:

- `test_graph_buffer_sizing_reaches_recurrent_and_linear_attention[gdn]`
- `test_graph_buffer_sizing_reaches_recurrent_and_linear_attention[mamba2]`
- `test_kv_layer_imports_without_the_conductor_or_the_sampling_kernels`

The first two reach `triton.language.int64`, absent from the local macOS fixture
stub; the subprocess in the third has no installed Triton at all. These are
baseline environment failures, not new skips or changes to the core tests.

Next is step 3: declare the graph and its token/frame loops, including new-context
prefill and routing from the latest generated image tokens. Keep neural
computation and checkpoint loading at their explicit unimplemented boundary
until step 4.
