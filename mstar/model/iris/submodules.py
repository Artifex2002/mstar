"""Batched eager FP32 serving nodes for supplied-action imagination."""

import torch

from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SubmoduleStep
from mstar.model.iris.components import Tokenizer
from mstar.model.iris.config import (
    DECODER_ATTN,
    ENCODER_ATTN,
    ENDING_SAMPLER,
    OBSERVATION_SAMPLER,
    REWARD_SAMPLER,
    ROLLOUT_LOOP,
    TOKEN_LOOP,
    WORLD_ATTN,
    WORLD_CARRY,
    WORLD_KV,
    WORLD_POS,
)
from mstar.model.iris.graph import frame_calls
from mstar.model.submodule_base import NodeInputs, NodeSubmodule


class IrisSubmodule(NodeSubmodule):
    disable_autocast = True
    disable_torch_compile = True
    float32_matmul_precision = "highest"
    cudnn_allow_tf32 = False

    def __init__(self, max_batch_size=2):
        super().__init__()
        self.batch_limit = max_batch_size

    def max_batch_size(self, graph_walk):
        return self.batch_limit

    def can_batch(self, batch, model_inputs):
        return 0 < len(model_inputs) <= self.batch_limit

    @staticmethod
    def _row(outputs, index):
        return {name: [value[index]] for name, value in outputs.items()}

    def forward(self, graph_walk, engine_inputs, **inputs):
        return self._row(self.forward_tensors(engine_inputs, **inputs), 0)

    def forward_batched(self, graph_walk, engine_inputs, **inputs):
        outputs = self.forward_tensors(engine_inputs, **inputs)
        return {rid: self._row(outputs, index) for index, rid in enumerate(engine_inputs.request_ids)}

    def preprocess(self, graph_walk, engine_inputs, inputs):
        if not self.can_batch(None, inputs):
            raise ValueError("IRIS rows have incompatible call phases or exceed max_batch_size")
        return {name: torch.stack([row.tensor_inputs[name] for row in inputs]) for name in inputs[0].tensor_inputs}


class ImageSubmodule(IrisSubmodule):
    def __init__(self, tokenizer, decode=False, initial=False, max_batch_size=2):
        super().__init__(max_batch_size)
        self.decode, self.initial = decode, initial
        self.embedding = tokenizer.embedding
        if decode:
            self.decoder, self.post_quant_conv = tokenizer.decoder, tokenizer.post_quant_conv
        else:
            self.encoder, self.pre_quant_conv = tokenizer.encoder, tokenizer.pre_quant_conv
        self.attention_key = DECODER_ATTN if decode else ENCODER_ATTN

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        names = ("observation_tokens",) if self.initial else WORLD_CARRY if self.decode else ("initial_image",)
        return NodeInputs(tensor_inputs={name: inputs[name][0] for name in names})

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        segments = tuple(Segment(rid, f"spatial_{size}", size * size) for size in (4, 8, 16) for rid in request_ids)
        return SubmoduleStep(steps={self.attention_key: AttentionStep(segments=segments, causal=False)})

    def forward_tensors(self, engine_inputs, **inputs):
        attend = engine_inputs.resources[self.attention_key].run
        if not self.decode:
            tokens = Tokenizer.encode(self, inputs["initial_image"], attend).tokens
            return {
                "observation_tokens": tokens,
                "reward_class": tokens.new_ones((tokens.shape[0], 1)),
                "ending_class": tokens.new_zeros((tokens.shape[0], 1)),
            }
        frames = Tokenizer.decode_tokens(self, inputs["observation_tokens"], attend)
        if self.initial:
            # Preserve the reference reset's reconstruction even though it is not emitted.
            return {}
        return {
            **{name: inputs[name] for name in WORLD_CARRY},
            "frames": (frames * 255).to(torch.uint8).permute(0, 2, 3, 1).unsqueeze(1).contiguous(),
            "reward": inputs["reward_class"].to(torch.float32) - 1,
            "ending": inputs["ending_class"].to(torch.bool),
        }


class WorldSubmodule(IrisSubmodule):
    def __init__(self, network, max_horizon, max_batch_size=2):
        super().__init__(max_batch_size)
        self.network = network
        self.schedules = tuple(frame_calls(frame) for frame in range(max_horizon))

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        frame = fwd_info.dynamic_loop_iter_counts[ROLLOUT_LOOP]
        token = fwd_info.dynamic_loop_iter_counts[TOKEN_LOOP]
        if frame >= fwd_info.step_metadata["iris_horizon"] or token >= len(self.schedules[frame]):
            return None
        call = self.schedules[frame][token]
        observation = inputs["observation_tokens"][0]
        if call.kind == "prime":
            tokens = observation
        elif call.kind == "action":
            tokens = inputs["actions"][0][frame : frame + 1]
        else:
            tokens = observation[call.input_index : call.input_index + 1]
        return NodeInputs(
            tensor_inputs={"tokens": tokens, **{name: inputs[name][0] for name in WORLD_CARRY}},
            kwargs={"call": call},
            resource_step_info=call,
            input_seq_len=call.span,
        )

    @staticmethod
    def call_key(call):
        return call.kind, call.label, call.sample_index

    def can_batch(self, batch, model_inputs):
        return (
            super().can_batch(batch, model_inputs)
            and len({self.call_key(row.resource_step_info) for row in model_inputs}) == 1
        )

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {**super().preprocess(graph_walk, engine_inputs, inputs), "call": inputs[0].resource_step_info}

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        call = inputs[0].resource_step_info
        segments = tuple(Segment(rid, call.label, call.span) for rid in request_ids)
        steps = {WORLD_KV: KVStep(), WORLD_POS: PositionStep(), WORLD_ATTN: AttentionStep(causal=True)}
        if call.sample_index is not None:
            steps[OBSERVATION_SAMPLER] = SamplerStep(apply_penalty=False, apply_filters=False)
        if call.kind == "action":
            steps[REWARD_SAMPLER] = SamplerStep(apply_penalty=False, apply_filters=False)
            steps[ENDING_SAMPLER] = SamplerStep(apply_penalty=False, apply_filters=False)
        if not self.can_batch(None, inputs):
            raise ValueError("IRIS cannot declare an incompatible batch")
        return SubmoduleStep(segments=segments, steps=steps)

    def compute(self, engine_inputs, tokens, call):
        resources = engine_inputs.resources
        kv, attention = resources[WORLD_KV], resources[WORLD_ATTN]

        def attend(layer, q, k, v):
            if attention.requires_kv_write:
                kv.write_kv(k, v, layer_idx=layer, label=call.label)
            return attention.run(q, k=k, v=v, layer_idx=layer, label=call.label, kv_cache_layer=kv.layer_view(layer))

        return self.network(tokens, resources[WORLD_POS].pos_ids(call.label), call, attend)

    def forward_tensors(self, engine_inputs, tokens, observation_tokens, reward_class, ending_class, call):
        result = self.compute(engine_inputs, tokens, call)
        resources, rids = engine_inputs.resources, engine_inputs.request_ids
        if call.kind == "action":
            reward_class = resources[REWARD_SAMPLER].sample(rids, result.logits_rewards[:, 0]).reshape(-1, 1)
            ending_class = resources[ENDING_SAMPLER].sample(rids, result.logits_ends[:, 0]).reshape(-1, 1)
        if call.sample_index is not None:
            # Other nodes may still read the preceding frame's carrier.
            observation_tokens = observation_tokens.clone()
            observation_tokens[:, call.sample_index] = resources[OBSERVATION_SAMPLER].sample(
                rids, result.logits_observations[:, 0]
            )
        return {
            "observation_tokens": observation_tokens,
            "reward_class": reward_class,
            "ending_class": ending_class,
        }

    def check_stop(self, request_id, request_info, outputs):
        frame = request_info.dynamic_loop_iter_counts[ROLLOUT_LOOP]
        token = request_info.dynamic_loop_iter_counts[TOKEN_LOOP]
        return {TOKEN_LOOP} if token + 1 == len(self.schedules[frame]) else set()
