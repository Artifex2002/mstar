"""IRIS's single-walk, three-node imagination graph."""

from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.iris.config import TOKENS_PER_BLOCK

WORLD_CARRY = ("observation_tokens", "reward_class", "ending_class")


def imagination_graph(max_frames: int) -> GraphSection:
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be an integer greater than zero.")

    encode = GraphNode(
        name="image_encoder",
        input_names={"initial_image"},
        outputs=[
            GraphEdge(next_node="world_model", name=name)
            for name in WORLD_CARRY
        ],
    )
    tokens = Loop(
        name="token_loop",
        # Up to one context prefill plus one action and 16 observation calls.
        max_iters=TOKENS_PER_BLOCK + 1,
        section=GraphNode(
            name="world_model",
            input_names={"actions", *WORLD_CARRY},
            outputs=[
                GraphEdge(next_node="world_model", name=name)
                for name in WORLD_CARRY
            ],
        ),
        # Publish only the final carry of this frame to the image decoder.
        outputs=[
            GraphEdge(next_node="image_decoder", name=name)
            for name in WORLD_CARRY
        ],
    )
    decode = GraphNode(
        name="image_decoder",
        input_names=set(WORLD_CARRY),
        outputs=[
            # Preserve generated codes directly for the next frame.
            *(GraphEdge(next_node="world_model", name=name) for name in WORLD_CARRY),
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
    return Sequential(sections=[
        encode,
        Loop(
            name="rollout_loop",
            max_iters=max_frames,
            section=Sequential(sections=[tokens, decode]),
            outputs=[],
        ),
    ])
