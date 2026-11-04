# IRIS: first contribution and learning plan

Status: `codex/iris` is the implementation branch, based on `main`. The official
reference harness is executable, and M* registration, the CPU request contract,
resource declarations, computation graph and batched eager FP32 neural submodules are
implemented. CPU checks and eager CUDA numerical/sampling qualification pass;
the user-provided [Tesla T4 report](iris-t4-qualification.json) records clean
commit `caf2f45a`, PyTorch 2.12.1+cu132 and FlashInfer 0.7.2. Full server
execution, CUDA graphs and performance are not yet qualified. The learner can
repeat the changes independently on an `iris` branch using the
[Step 1 lesson](iris-step-01.md).
Continue with the [Step 2 contract walkthrough](iris-step-02.md).
Resource declarations and the context-storage tradeoff are explained in
[Step 2a](iris-step-02a.md).
The [Step 3 walkthrough](iris-step-03.md) explains nested loops and native routing tests.
The [Step 4 walkthrough](iris-step-04.md) covers checkpoint loading, neural
computation, resource-step declarations and the remaining GPU acceptance checks.
The [Step 5a walkthrough](iris-step-05.md) adds eager batching and an executable
[CUDA qualification guide](../../scripts/iris/CUDA.md) for Colab or the cluster.
The [server continuation](iris-step-05-server.md) adds a real HTTP/native-worker
qualification harness; its GPU run is the next acceptance gate.

## Benchmark objective

The goal is correct IRIS world-model inference in M*, followed by a measured
comparison with the official PyTorch implementation and, if implemented there,
vLLM-Omni. On 2026-11-04, no native IRIS entry was found in vLLM-Omni's
[supported-model list](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/),
[autoregressive registry](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/registry.py),
or [diffusion registry](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/registry.py).
A direct comparison therefore needs a separate IRIS integration in vLLM-Omni;
running official PyTorch code beside that runtime would not establish its serving
performance. A private or unmerged implementation has not been ruled out.

Keep checkpoint, initial images, supplied actions, horizon, precision and context
refresh semantics fixed across implementations. First compare encoder tokens,
teacher-forced logits and decoding numerically; stochastic rollouts can diverge
with different random-number consumption even when these computations agree.

### Faithful port and comparison boundaries

The serving implementation must preserve the selected checkpoint's inference
computation: tokenizer normalization and codebook, action/observation embedding
order, causal attention, learned absolute positions, prediction heads, sampling
distributions, decoding and context refresh. Changes in cache storage, scheduling,
batching or execution kernels must pass numerical checks against the reference.
The 20-action request limit and native output serialization are M* interface
choices, not features copied from upstream. Compare the same H and include the
same output conversions when timing the complete workload.

Both implementations use PyTorch. The experiment measures official IRIS execution
versus execution managed by M*, rather than comparing two numerical frameworks.
Record these configurations separately:

1. Official eager PyTorch reference, without diagnostic trace hooks or file I/O.
2. M* eager execution of equivalent computation, with compilation and accelerator
   graph replay disabled, to expose the initial integration overhead.
3. M* with its validated batching, compilation and graph replay enabled, recording
   each setting and comparing at matching batch sizes and concurrency.

Where practical, also measure a standalone PyTorch compiled/captured control and
batched reference runs. This helps attribute gains to compilation/capture versus
M*'s scheduling and resource management. Do not compare a batch-1 reference solely
against a batched M* workload and describe the whole difference as runtime speedup.

Model-only timing must use the same input/output device boundaries. Complete
rollout timing must include the reference's reward/ending CPU materialization and
equivalent M* outputs; serving timing adds preprocessing, routing, transport and
serialization and must be reported separately. The current diagnostic harness
copies intermediates to CPU on every world-model call and is a correctness tool,
not a raw-performance baseline. No timing claims may be derived from its trace runs.
Both complete-rollout paths include the reference's one-time initial image
reconstruction, even though it is discarded rather than returned as a prediction.

After functional parity, add a timing harness under `benchmark/` and measure on
the same GPU and software environment. Exclude model loading and compilation
from steady-state timings, warm up first, disable intermediate trace recording,
and report cold-start costs separately. Measure rollout latency (including
p50/p95), generated frames per second, memory use and concurrency sweeps; distinguish
model execution from end-to-end request serving. Use M*'s worker-phase harness
to explain overhead, and report exact commits, hardware, versions, workload and
repeated-run variation. No performance result has been measured yet.

## Scope we will implement first

One request supplies an initial image and a bounded action sequence. It returns
predicted images, rewards and ending signals for that sequence. Actions are supplied
by the caller; adding the recurrent actor is a later capability. This scope
deliberately executes IRIS's Transformer world model: the upstream `Agent.act`
path alone does not call that world model.

```text
initial frame -> image encoder/codebook -> 16 observation tokens
                                             |
provided action -> Transformer -> next observation tokens, reward, end
                                             |
                                      image decoder -> frame
```

IRIS is its own model-based RL algorithm, not a Transformer port of DreamerV3.
Its normal actor is CNN+LSTM; the world model is autoregressive self-attention.

## Reproducible reference

| Item | Decision |
|---|---|
| Official source | [eloialonso/iris](https://github.com/eloialonso/iris) |
| Source revision | `24326aaaa283c527f42b89b44cfdecf2665a7a16` |
| Checkpoint | `eloialonso/iris`, `pretrained_models/Breakout.pt` |
| HF revision | `c0539e99adf924d44be0f19b12de00639daa5550` |
| Checkpoint SHA-256 | `169e3669b7d1901990b4ebce35ff8f3355610fae393e6a587fd62c40faf790f4` |
| Checkpoint size | 126,502,241 bytes |
| First inputs | Synthetic image pattern plus supplied legal actions; real frames supported as `.npy` |
| External harness | [`scripts/iris/reference_harness.py`](../../scripts/iris/reference_harness.py) |
| Setup and run instructions | [README](../../scripts/iris/README.md) |

Reference source, checkpoint and generated traces live in ignored
`.ci-results/iris/`. The harness requires the pinned source to be clean and the
checkpoint to match its checksum. It calls upstream modules without copying their
implementations into M*. Upstream is GPL-3.0 and M* is Apache-2.0; agree the
implementation/code-reuse approach with maintainers before porting source files.
This note does not make a licensing-compatibility determination.

The modern CPU reference environment is separate from the Dreamer environment
in the original checkout. It does not download Atari ROMs or create an emulator.
Training-only LPIPS is disabled through the upstream constructor; only LPIPS
checkpoint keys are omitted, with all tokenizer/world-model inference state
loaded strictly. Actor checkpoint weights are unused by this workload.

## Tensor contract and reference mechanics

| Value | Shape | Meaning |
|---|---|---|
| Initial image | `[B,3,64,64]`, float32 `[0,1]` | Prepared channel-first image |
| Supplied actions | `[B,H]`, int64 `[0,4)` | Breakout's four action IDs |
| Observation tokens | `[B,16]`, int64 `[0,512)` | Quantized 4x4 image grid |
| Frame prediction | `[B,3,64,64]` | Decoded generated tokens |
| Reward | `[B]` | Sampled from three classes representing -1, 0, +1 |
| Ending signal | `[B]`, bool | Sampled binary ending prediction |

The default Transformer has 10 layers, 4 heads, width 256 and learned absolute
position embeddings. One block has 16 image tokens and one action token; 20
blocks bound the context to 340 tokens. It uses causal attention in this config.

The upstream environment prefills the initial 16 image tokens. For each supplied
action it runs 17 one-token calls: the action call begins prediction of the next
image, then the generated image tokens are fed back to complete the frame's
context. Reward/ending heads are read on the action call. Preserve this alignment;
moving a head by one position changes the modeled conditional distribution.

When the next step would exceed 340 tokens, the reference replaces its context
with the current image's 16 tokens and restarts positions. Generic sliding-window
KV eviction is not equivalent to this behavior. The baseline must preserve it.

The harness records predicted endings and continues the supplied horizon. This
is an imagined prediction request, not a real environment evaluation. Quality
needs representative game observations; synthetic inputs are a mechanics test.

## How this maps to M*

Proposed graph components are `image_encoder`, `world_model`, and `image_decoder`.
The graph declaration, exact walk names and output encoding will be worked out in
their guide steps. Start on one GPU so placement changes do not obscure numerical
correctness; the YAML, not model code, will control placement later.

The Transformer needs declared KV storage, attention planning, learned-position
tracking and categorical sampling. The cache lifecycle belongs to engine resources;
do not import upstream `KVCache` into a serving model. The tokenizer also contains
spatial attention blocks: inspect their resource/wrapper needs in step 2a rather
than assuming every non-Transformer component needs no resources.

The actor's LSTM state is not required by the initial supplied-action workload.
If autonomous rollouts are added later, declare its per-request state through the
engine's recurrent resources. No simulator runs in a model forward.

## Guide sequence and teaching workflow

| Step | Your implementation | Systems concepts to learn |
|---|---|---|
| 0: baseline (prepared here) | Inspect pinned source, checkpoint and trace; run the harness | Numerical references, tensor layouts, conditional prediction, cached versus dense attention |
| 1: registration | Lazy registry entry, model class scaffold, YAML, CLI default, truthful docs | Discovery versus loading, Python packages, logical versus physical placement |
| 2: Model contract | Input preparation, bounded action sequence, initial routing and completion | Control/data planes, request lifecycle, serialization |
| 2a: resources | KV, attention, positions, sampling; inspect tokenizer attention needs | Admission, planning, commit, bounded state, cache reset semantics |
| 3: graph | Image encoding, context priming, token/frame loops and decoding | Dataflow, readiness, loops and output routing |
| 4 / 4a: submodules | Tensor computations, checkpoint loading and step declarations | Numerical parity, weight layouts, host/device boundaries |
| 5: batching and capture | Independent rollout rows and engine-owned capture configurations | Variable lengths, padding, fixed buffers, asynchronous execution |
| 6: deployment | Final YAML, client example, real-input validation and measurement | Placement, communication costs, concurrency and performance evidence |

On `codex/iris`, the assistant implements each guide step, explains the code path,
and runs its acceptance check before advancing. The lessons also describe the
edits and concepts for the learner to reproduce later on a separate branch.

Useful existing M* reading: Orpheus's autoregressive declarations, V-JEPA2's
world-model input/output routing, and Waypoint's prime/rollout lifecycle. Use
their patterns selectively; IRIS's learned positions and cache refresh have
different semantics from those models.

## Validation record

The pretrained CPU baseline uses Python 3.12.13, PyTorch 2.12.1,
torchvision 0.27.1 and NumPy 1.26.4, with two intra-op threads.

Initial validation:

- Strict inference-weight loading and checksum verification passed.
- Three-frame rollout with batch 2 passed finite checks.
- Cached versus dense hidden features and prediction heads passed with
  `rtol=2e-5`, `atol=1e-5`; maximum absolute differences were about `3.34e-6`
  for hidden features, `5.34e-5` for observation logits, `9.54e-6` for reward
  logits and `2.86e-6` for ending logits. Relative tolerance applies as well.
- Tokenizer/codebook/decoder: **4,027,523 parameters**, excluding LPIPS.
- Transformer world-model module: **8,447,493 parameters**.
- Initial rollout components total: **12,475,016 parameters**; this excludes
  the actor and training-only perceptual network.
- Five harness utility tests and four existing registration tests passed.

- The three-frame second process matched all recorded tensors within the trace
  comparison tolerances (integer/boolean values exactly).
- The 20-frame run crossed the context boundary and recorded one refresh:
  frame-end cache sizes reached 339, then returned to 33 after re-priming from
  the last image. It made 342 world-model calls (initial prefill, 20x17 calls,
  and one refresh prefill).
- `ruff check .`, formatter checks, and all 15 CPU test-selection checks passed.

Generated evidence is in `.ci-results/iris/{baseline,repeat,context-refresh}/`.
This is not yet validation of M* execution, game quality, or a performance claim.
