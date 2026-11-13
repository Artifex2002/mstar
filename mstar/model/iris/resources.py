"""IRIS resource declarations; the engine owns their execution and lifecycle."""

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
from mstar.model.iris.config import (
    CONTEXT_TOKENS,
    OBSERVATION_VOCAB_SIZE,
    TOKENIZER_ATTENTION_CHANNELS,
    TRANSFORMER_HEAD_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)


def iris_resources() -> list[NodeResourceSpec]:
    world_nodes = {"world_model"}

    specs: list[NodeResourceSpec] = [
        KVSpec(
            resource_key="world_kv",
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
            resource_key="world_attn",
            nodes=world_nodes,
            config=AttentionConfig(kv_cache="world_kv"),
        ),
        PositionSpec(
            resource_key="world_pos",
            nodes=world_nodes,
            config=PositionConfig(kv_cache="world_kv"),
        ),
    ]

    for key, vocab_size in (
        ("observation_sampler", OBSERVATION_VOCAB_SIZE),
        ("reward_sampler", 3),
        ("ending_sampler", 2),
    ):
        specs.append(
            SamplerSpec(
                resource_key=key,
                nodes=world_nodes,
                vocab_size=vocab_size,
                enable_repetion_penalty=False,
            )
        )

    for key, node in (
        ("encoder_attn", "image_encoder"),
        ("decoder_attn", "image_decoder"),
    ):
        specs.append(
            RaggedAttentionSpec(
                resource_key=key,
                nodes={node},
                config=RaggedAttentionConfig(
                    num_qo_heads=1,
                    num_kv_heads=1,
                    head_dim=TOKENIZER_ATTENTION_CHANNELS,
                    dtype=torch.float32,
                    max_segments_per_request=1,
                    max_tokens_per_request=16 * 16,
                ),
            )
        )

    return specs
