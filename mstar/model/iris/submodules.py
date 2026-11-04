"""Eager FP32 serving nodes for supplied-action imagination."""

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

    def max_batch_size(self, graph_walk):
        return 1


class ImageSubmodule(IrisSubmodule):
    def __init__(self, tokenizer, decode=False, initial=False):
        super().__init__()
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

    def forward(self, graph_walk, engine_inputs, **inputs):
        attend = engine_inputs.resources[self.attention_key].run
        if not self.decode:
            tokens = Tokenizer.encode(self, inputs["initial_image"].unsqueeze(0), attend).tokens[0]
            return {
                "observation_tokens": [tokens],
                "reward_class": [tokens.new_ones(1)],
                "ending_class": [tokens.new_zeros(1)],
            }
        frames = Tokenizer.decode_tokens(self, inputs["observation_tokens"].unsqueeze(0), attend)
        if self.initial:
            # Preserve the reference reset's reconstruction even though it is not emitted.
            return {}
        return {
            **{name: [inputs[name]] for name in WORLD_CARRY},
            "frames": [(frames * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()],
            "reward": [inputs["reward_class"].to(torch.float32) - 1],
            "ending": [inputs["ending_class"].to(torch.bool)],
        }


class WorldSubmodule(IrisSubmodule):
    def __init__(self, network, max_horizon):
        super().__init__()
        self.network = network
        self.schedules = tuple(frame_calls(frame) for frame in range(max_horizon))

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        frame = fwd_info.dynamic_loop_iter_counts[ROLLOUT_LOOP]
        token = fwd_info.dynamic_loop_iter_counts[TOKEN_LOOP]
        if frame >= fwd_info.step_metadata["iris_horizon"] or token >= len(self.schedules[frame]):
            return None
        call = self.schedules[frame][token]
        return NodeInputs(
            tensor_inputs={name: inputs[name][0] for name in ("actions", *WORLD_CARRY)},
            kwargs={"call": call, "frame": frame},
            resource_step_info=call,
            input_seq_len=call.span,
        )

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        call = inputs[0].resource_step_info
        segments = tuple(Segment(rid, call.label, call.span) for rid in request_ids)
        steps = {WORLD_KV: KVStep(), WORLD_POS: PositionStep(), WORLD_ATTN: AttentionStep(causal=True)}
        if call.sample_index is not None:
            steps[OBSERVATION_SAMPLER] = SamplerStep(apply_penalty=False, apply_filters=False)
        if call.kind == "action":
            steps[REWARD_SAMPLER] = SamplerStep(apply_penalty=False, apply_filters=False)
            steps[ENDING_SAMPLER] = SamplerStep(apply_penalty=False, apply_filters=False)
        return SubmoduleStep(segments=segments, steps=steps)

    def compute(self, engine_inputs, tokens, call):
        resources = engine_inputs.resources
        kv, attention = resources[WORLD_KV], resources[WORLD_ATTN]

        def attend(layer, q, k, v):
            if attention.requires_kv_write:
                kv.write_kv(k, v, layer_idx=layer, label=call.label)
            return attention.run(q, k=k, v=v, layer_idx=layer, label=call.label, kv_cache_layer=kv.layer_view(layer))

        return self.network(tokens, resources[WORLD_POS].pos_ids(call.label), call, attend)

    def forward(self, graph_walk, engine_inputs, actions, observation_tokens, reward_class, ending_class, call, frame):
        if call.kind == "prime":
            tokens = observation_tokens.unsqueeze(0)
        elif call.kind == "action":
            tokens = actions[frame : frame + 1].unsqueeze(0)
        else:
            tokens = observation_tokens[call.input_index : call.input_index + 1].unsqueeze(0)
        result = self.compute(engine_inputs, tokens, call)
        resources, rids = engine_inputs.resources, engine_inputs.request_ids
        if call.kind == "action":
            reward_class = resources[REWARD_SAMPLER].sample(rids, result.logits_rewards[:, 0])
            ending_class = resources[ENDING_SAMPLER].sample(rids, result.logits_ends[:, 0])
        if call.sample_index is not None:
            # Other nodes may still read the preceding frame's carrier.
            observation_tokens = observation_tokens.clone()
            observation_tokens[call.sample_index : call.sample_index + 1] = resources[OBSERVATION_SAMPLER].sample(
                rids, result.logits_observations[:, 0]
            )
        return {
            "observation_tokens": [observation_tokens],
            "reward_class": [reward_class],
            "ending_class": [ending_class],
        }

    def check_stop(self, request_id, request_info, outputs):
        frame = request_info.dynamic_loop_iter_counts[ROLLOUT_LOOP]
        token = request_info.dynamic_loop_iter_counts[TOKEN_LOOP]
        return {TOKEN_LOOP} if token + 1 == len(self.schedules[frame]) else set()
