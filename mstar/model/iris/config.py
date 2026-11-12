from dataclasses import dataclass

HF_REPO = "eloialonso/iris"
IMAGE_SHAPE = (3, 64, 64)


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
