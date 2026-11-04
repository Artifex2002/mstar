from dataclasses import dataclass

HF_REPO = "eloialonso/iris"
CHECKPOINT_FILE = "pretrained_models/Breakout.pt"
CHECKPOINT_REVISION = "c0539e99adf924d44be0f19b12de00639daa5550"
CHECKPOINT_SHA256 = "169e3669b7d1901990b4ebce35ff8f3355610fae393e6a587fd62c40faf790f4"

IMAGE_SHAPE = (3, 64, 64)
NUM_ACTIONS = 4
OBSERVATION_TOKENS = 16
OBSERVATION_VOCAB_SIZE = 512
TOKENS_PER_FRAME = OBSERVATION_TOKENS + 1
CONTEXT_TOKENS = 340
FRAMES_PER_CONTEXT = (CONTEXT_TOKENS - OBSERVATION_TOKENS) // TOKENS_PER_FRAME
TRANSFORMER_LAYERS = 10
TRANSFORMER_HEADS = 4
TRANSFORMER_HEAD_DIM = 64
TOKENIZER_CHANNELS = 64
IMAGE_ENCODER = "image_encoder"
WORLD_MODEL = "world_model"
IMAGE_DECODER = "image_decoder"
INITIAL_DECODER = "initial_image_decoder"
IMAGINE_WALK = "imagine"
ROLLOUT_LOOP = "rollout_loop"
TOKEN_LOOP = "token_loop"
WORLD_CARRY = ("observation_tokens", "reward_class", "ending_class")
OUTPUT_MODALITIES = ("video_frame", "scalar", "tensor")
WORLD_KV = "world_kv"
WORLD_ATTN = "world_attn"
WORLD_POS = "world_pos"
OBSERVATION_SAMPLER = "observation_sampler"
REWARD_SAMPLER = "reward_sampler"
ENDING_SAMPLER = "ending_sampler"
ENCODER_ATTN = "encoder_attn"
DECODER_ATTN = "decoder_attn"


@dataclass(frozen=True)
class IrisConfig:
    """Selected checkpoint and the deployment's bound on supplied actions."""

    game: str = "Breakout"
    max_rollout_horizon: int = 20
    max_batch_size: int = 2

    def __post_init__(self):
        if self.game != "Breakout":
            raise ValueError("IRIS currently supports only the pinned Breakout checkpoint")
        if type(self.max_rollout_horizon) is not int or self.max_rollout_horizon < 1:
            raise ValueError("max_rollout_horizon must be an integer >= 1")
        if type(self.max_batch_size) is not int or self.max_batch_size < 1:
            raise ValueError("max_batch_size must be an integer >= 1")
