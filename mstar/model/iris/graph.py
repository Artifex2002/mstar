"""IRIS dataflow and the reference's host-known per-frame call schedule."""

from dataclasses import dataclass
from typing import Literal

from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.iris.config import (
    FRAMES_PER_CONTEXT,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    INITIAL_DECODER,
    OBSERVATION_TOKENS,
    ROLLOUT_LOOP,
    TOKEN_LOOP,
    TOKENS_PER_FRAME,
    WORLD_CARRY,
    WORLD_MODEL,
)


@dataclass(frozen=True, slots=True)
class WorldCall:
    kind: Literal["prime", "action", "observation"]
    label: str
    position: int
    input_index: int | None
    sample_index: int | None

    @property
    def span(self) -> int:
        return OBSERVATION_TOKENS if self.kind == "prime" else 1


def frame_calls(frame_index: int) -> tuple[WorldCall, ...]:
    """Prefill on refresh, action, then consume all 16 generated image codes."""
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("IRIS frame_index must be an integer >= 0")
    epoch, local_frame = divmod(frame_index, FRAMES_PER_CONTEXT)
    label = f"epoch_{epoch}"
    start = OBSERVATION_TOKENS + local_frame * TOKENS_PER_FRAME
    prime = (WorldCall("prime", label, 0, None, None),) if local_frame == 0 else ()
    return (
        prime
        + (WorldCall("action", label, start, None, 0),)
        + tuple(
            WorldCall(
                "observation",
                label,
                start + index + 1,
                index,
                index + 1 if index + 1 < OBSERVATION_TOKENS else None,
            )
            for index in range(OBSERVATION_TOKENS)
        )
    )


def imagination_graph(max_frames: int) -> GraphSection:
    encode = GraphNode(
        name=IMAGE_ENCODER,
        input_names={"initial_image"},
        outputs=[
            *(GraphEdge(next_node=WORLD_MODEL, name=name) for name in WORLD_CARRY),
            GraphEdge(next_node=INITIAL_DECODER, name="observation_tokens"),
        ],
    )
    initial_decode = GraphNode(name=INITIAL_DECODER, input_names={"observation_tokens"}, outputs=[])
    tokens = Loop(
        name=TOKEN_LOOP,
        max_iters=TOKENS_PER_FRAME + 1,
        section=GraphNode(
            name=WORLD_MODEL,
            input_names={"actions", *WORLD_CARRY},
            outputs=[GraphEdge(next_node=WORLD_MODEL, name=name) for name in WORLD_CARRY],
        ),
        # Cache the final state of this frame, then hand it to the decoder once.
        outputs=[GraphEdge(next_node=IMAGE_DECODER, name=name) for name in WORLD_CARRY],
    )
    decode = GraphNode(
        name=IMAGE_DECODER,
        input_names=set(WORLD_CARRY),
        outputs=[
            # Feed generated codes directly into the next frame; never re-encode RGB.
            *(GraphEdge(next_node=WORLD_MODEL, name=name) for name in WORLD_CARRY),
            GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="frames",
                output_modality="video_frame",
                conductor_new_token=True,
            ),
            GraphEdge(next_node=EMIT_TO_CLIENT, name="reward", output_modality="scalar"),
            GraphEdge(next_node=EMIT_TO_CLIENT, name="ending", output_modality="tensor"),
        ],
    )
    return Sequential(
        [
            encode,
            Parallel(
                [
                    initial_decode,
                    Loop(name=ROLLOUT_LOOP, max_iters=max_frames, section=Sequential([tokens, decode]), outputs=[]),
                ]
            ),
        ]
    )
