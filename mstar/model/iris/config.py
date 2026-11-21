from dataclasses import dataclass

HF_REPO = "eloialonso/iris"
IMAGE_SHAPE = (3, 64, 64)

OBSERVATION_TOKENS = 16
OBSERVATION_VOCAB_SIZE = 512

TRANSFORMER_LAYERS = 10
TRANSFORMER_HEADS = 4
TRANSFORMER_HEAD_DIM = 64

TOKENS_PER_BLOCK = OBSERVATION_TOKENS + 1
CONTEXT_BLOCKS = 20
CONTEXT_TOKENS = TOKENS_PER_BLOCK * CONTEXT_BLOCKS

FRAMES_PER_CONTEXT = (
    CONTEXT_TOKENS - OBSERVATION_TOKENS
) // TOKENS_PER_BLOCK

TOKENIZER_ATTENTION_CHANNELS = 64


@dataclass(frozen=True)
class IRISConfig:
    """Checkpoint selection and the deployment's supplied-action limit."""

    game: str = "Breakout"
    max_rollout_horizon: int = 20

    def __post_init__(self) -> None:
        if self.game != "Breakout":
            raise ValueError(
                "Only Breakout is supported until other checkpoints are validated."
            )

        if (
            type(self.max_rollout_horizon) is not int
            or self.max_rollout_horizon < 1
        ):
            raise ValueError(
                "max_rollout_horizon must be an integer greater than zero."
            )

    @property
    def checkpoint_file(self) -> str:
        return f"pretrained_models/{self.game}.pt"

    @property
    def num_actions(self) -> int:
        return 4
