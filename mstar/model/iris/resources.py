"""IRIS declarations; the engine owns cache storage, positions and sampling."""

from typing import NamedTuple

import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    SamplerSpec,
)
from mstar.model.base import LabelLayout, Span
from mstar.model.iris.config import (
    CONTEXT_TOKENS,
    DECODER_ATTN,
    ENCODER_ATTN,
    ENDING_SAMPLER,
    FRAMES_PER_CONTEXT,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    IMAGINE_WALK,
    OBSERVATION_SAMPLER,
    OBSERVATION_TOKENS,
    OBSERVATION_VOCAB_SIZE,
    REWARD_SAMPLER,
    TOKENIZER_CHANNELS,
    TOKENS_PER_FRAME,
    TRANSFORMER_HEAD_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    WORLD_ATTN,
    WORLD_KV,
    WORLD_MODEL,
    WORLD_POS,
)


class ContextEpoch(NamedTuple):
    label: str
    frames: int
    tokens: int


def context_epochs(horizon: int) -> tuple[ContextEpoch, ...]:
    """Independent contexts, each primed from the latest image's 16 tokens."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("IRIS context planning needs an integer horizon >= 1")
    return tuple(
        ContextEpoch(
            label=f"epoch_{start // FRAMES_PER_CONTEXT}",
            frames=min(FRAMES_PER_CONTEXT, horizon - start),
            tokens=OBSERVATION_TOKENS + TOKENS_PER_FRAME * min(FRAMES_PER_CONTEXT, horizon - start),
        )
        for start in range(0, horizon, FRAMES_PER_CONTEXT)
    )


def request_kv_layout(horizon: int) -> dict[str, dict[str, LabelLayout]]:
    # Every write is known from the supplied actions. Reserve its full footprint,
    # including finished epochs retained until teardown, without prefix reuse.
    return {
        WORLD_KV: {
            epoch.label: LabelLayout(spans=(Span("rows", IMAGINE_WALK, epoch.tokens, epoch.tokens, WORLD_MODEL),))
            for epoch in context_epochs(horizon)
        }
    }


def iris_resources() -> list[NodeResourceSpec]:
    world_nodes = {WORLD_MODEL}
    specs = [
        KVSpec(
            resource_key=WORLD_KV,
            nodes=world_nodes,
            config=PagedKVConfig(
                num_layers=TRANSFORMER_LAYERS,
                num_kv_heads=TRANSFORMER_HEADS,
                num_qo_heads=TRANSFORMER_HEADS,
                head_dim=TRANSFORMER_HEAD_DIM,
                max_seq_len=CONTEXT_TOKENS,
                prefix_cache=False,
            ),
        ),
        AttentionSpec(
            resource_key=WORLD_ATTN,
            nodes=world_nodes,
            config=AttentionConfig(kv_cache=WORLD_KV),
        ),
        # Only the planned ids index IRIS's learned table. Never apply RoPE.
        PositionSpec(
            resource_key=WORLD_POS,
            nodes=world_nodes,
            config=PositionConfig(kv_cache=WORLD_KV),
        ),
    ]
    for key, vocab in ((OBSERVATION_SAMPLER, OBSERVATION_VOCAB_SIZE), (REWARD_SAMPLER, 3), (ENDING_SAMPLER, 2)):
        specs.append(
            SamplerSpec(
                resource_key=key,
                nodes=world_nodes,
                vocab_size=vocab,
                enable_repetion_penalty=False,
            )
        )
    for key, node in ((ENCODER_ATTN, IMAGE_ENCODER), (DECODER_ATTN, IMAGE_DECODER)):
        specs.append(
            RaggedAttentionSpec(
                resource_key=key,
                nodes={node},
                config=RaggedAttentionConfig(
                    num_qo_heads=1,
                    num_kv_heads=1,
                    head_dim=TOKENIZER_CHANNELS,
                    dtype=torch.float32,
                    max_segments_per_request=1,
                    max_tokens_per_request=16 * 16,
                ),
            )
        )
    return specs
