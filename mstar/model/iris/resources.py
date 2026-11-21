"""IRIS state declarations and the request's independent context layouts."""

from typing import NamedTuple

import torch

from mstar.engine.state import (
    Array,
    Attend,
    Block,
    Causal,
    Entry,
    KVEntry,
    Layout,
    Positions,
    Read,
    Sampler,
    Span,
    TensorType,
)
from mstar.model.iris.config import (
    CONTEXT_TOKENS,
    FRAMES_PER_CONTEXT,
    OBSERVATION_TOKENS,
    OBSERVATION_VOCAB_SIZE,
    TOKENIZER_ATTENTION_CHANNELS,
    TOKENS_PER_BLOCK,
    TRANSFORMER_HEAD_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)


def iris_state(max_rollout_horizon: int) -> list[Array | Sampler]:
    world_nodes = {"world_model"}
    declarations: list[Array | Sampler] = [
        Array(
            "world_kv", nodes=world_nodes,
            layers=frozenset(range(TRANSFORMER_LAYERS)),
            entry=KVEntry(heads=TRANSFORMER_HEADS, head_dim=TRANSFORMER_HEAD_DIM),
            max_len=CONTEXT_TOKENS,
            labels={epoch.label for epoch in context_epochs(max_rollout_horizon)},
            positions=Positions(),
            reads={"world_attn": Read(mask=Causal(), op=Attend(q_heads=TRANSFORMER_HEADS))},
        ),
    ]
    for key, vocab_size in (
        ("observation_sampler", OBSERVATION_VOCAB_SIZE),
        ("reward_sampler", 3), ("ending_sampler", 2),
    ):
        declarations.append(Sampler(key, nodes=world_nodes, vocab_size=vocab_size, penalty=False))
    field = TensorType((1, TOKENIZER_ATTENTION_CHANNELS), dtype=torch.float32, shard=0)
    for kind, node in (("encoder", "image_encoder"), ("decoder", "image_decoder")):
        # Each resolution needs an independent plan; identical reads share the engine's ragged resource.
        for resolution in (16, 8, 4):
            read = f"{kind}_attn" if resolution == 16 else f"{kind}_attn_{resolution}"
            declarations.append(Array(
                f"{kind}_features_{resolution}", nodes={node}, layers={0},
                entry=Entry({"k": field, "v": field}), max_len=0,
                labels={f"spatial_{resolution}"},
                reads={read: Read(mask=Block(), op=Attend(q_heads=1))},
            ))
    return declarations


class ContextEpoch(NamedTuple):
    label: str
    frames: int
    tokens: int


def context_epochs(horizon: int) -> tuple[ContextEpoch, ...]:
    """Independent contexts, each primed from the latest image tokens."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("IRIS requires an integer horizon greater than zero.")

    epochs = []
    for start in range(0, horizon, FRAMES_PER_CONTEXT):
        frames = min(FRAMES_PER_CONTEXT, horizon - start)
        epochs.append(
            ContextEpoch(
                label=f"epoch_{len(epochs)}",
                frames=frames,
                tokens=OBSERVATION_TOKENS + frames * TOKENS_PER_BLOCK,
            )
        )
    return tuple(epochs)


def request_kv_layout(horizon: int) -> dict[str, dict[str, Layout]]:
    return {
        "world_kv": {
            epoch.label: Layout(
                spans=(
                    Span(
                        kind="rows",
                        walk="imagine",
                        length=epoch.tokens,
                        advance=epoch.tokens,
                        source="world_model",
                    ),
                ),
            )
            for epoch in context_epochs(horizon)
        }
    }
