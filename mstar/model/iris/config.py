from dataclasses import dataclass

HF_REPO = "eloialonso/iris"
CHECKPOINT_FILE = "pretrained_models/Breakout.pt"
CHECKPOINT_REVISION = "c0539e99adf924d44be0f19b12de00639daa5550"
CHECKPOINT_SHA256 = "169e3669b7d1901990b4ebce35ff8f3355610fae393e6a587fd62c40faf790f4"

IMAGE_SHAPE = (3, 64, 64)
NUM_ACTIONS = 4
IMAGE_ENCODER = "image_encoder"
WORLD_MODEL = "world_model"
IMAGE_DECODER = "image_decoder"
IMAGINE_WALK = "imagine"
ROLLOUT_LOOP = "rollout_loop"
OUTPUT_MODALITIES = ("video_frame", "scalar", "tensor")


@dataclass(frozen=True)
class IrisConfig:
    """Selected checkpoint and the deployment's bound on supplied actions."""

    game: str = "Breakout"
    max_rollout_horizon: int = 20

    def __post_init__(self):
        if self.game != "Breakout":
            raise ValueError("IRIS currently supports only the pinned Breakout checkpoint")
        if type(self.max_rollout_horizon) is not int or self.max_rollout_horizon < 1:
            raise ValueError("max_rollout_horizon must be an integer >= 1")
