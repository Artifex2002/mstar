"""Pinned checkpoint loading, with strict inference-weight coverage."""

import hashlib
from pathlib import Path
from typing import NamedTuple

import torch

from mstar.model.iris.components import Tokenizer, WorldNetwork
from mstar.model.iris.config import CHECKPOINT_FILE, CHECKPOINT_REVISION, CHECKPOINT_SHA256, HF_REPO


class IrisNetworks(NamedTuple):
    tokenizer: Tokenizer
    world: WorldNetwork


def resolve_checkpoint(local_path=None, repository=HF_REPO, cache_dir=None):
    if local_path is not None:
        path = Path(local_path)
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repository, CHECKPOINT_FILE, revision=CHECKPOINT_REVISION, cache_dir=cache_dir))
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"IRIS checkpoint SHA256 mismatch: {path}")
    return path


def inference_state(state):
    tokenizer, world = {}, {}
    for key, value in state.items():
        if key.startswith(("actor_critic.", "tokenizer.lpips.")):
            continue
        if key.startswith("tokenizer."):
            tokenizer[key.removeprefix("tokenizer.")] = value
            continue
        if not key.startswith("world_model."):
            raise ValueError(f"Unknown IRIS checkpoint key: {key}")
        key = key.removeprefix("world_model.")
        # Saved causal masks and head index tables are replaced by declared spans.
        if key.endswith((".mask", ".indices")):
            continue
        key = key.removeprefix("transformer.")
        key = key.replace("embedder.embedding_tables.0.", "action_embedding.")
        key = key.replace("embedder.embedding_tables.1.", "observation_embedding.")
        key = key.replace(".head_module.", ".")
        world[key] = value
    return tokenizer, world


def load_networks(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer_state, world_state = inference_state(state)
    tokenizer, world = Tokenizer(), WorldNetwork()
    tokenizer.load_state_dict(tokenizer_state, strict=True)
    world.load_state_dict(world_state, strict=True)
    return IrisNetworks(tokenizer.eval().to(device), world.eval().to(device))
