"""Host-known Transformer calls needed to generate one IRIS frame."""

from dataclasses import dataclass
from typing import Literal

from mstar.model.iris.config import (
    FRAMES_PER_CONTEXT,
    OBSERVATION_TOKENS,
    TOKENS_PER_BLOCK,
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
    """Return the call schedule for a zero-based generated-frame index."""
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError(
            "frame_index must be an integer greater than or equal to zero."
        )

    epoch, local_frame = divmod(frame_index, FRAMES_PER_CONTEXT)
    label = f"epoch_{epoch}"
    start = OBSERVATION_TOKENS + local_frame * TOKENS_PER_BLOCK

    prime = (
        (WorldCall("prime", label, 0, None, None),)
        if local_frame == 0 else ()
    )

    return (
        prime
        + (WorldCall("action", label, start, None, 0),)
        + tuple(
            WorldCall(
                kind="observation",
                label=label,
                position=start + index + 1,
                input_index=index,
                sample_index=(
                    index + 1
                    if index + 1 < OBSERVATION_TOKENS
                    else None
                ),
            )
            for index in range(OBSERVATION_TOKENS)
        )
    )
