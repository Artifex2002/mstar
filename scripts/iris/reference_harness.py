#!/usr/bin/env python3
"""Run a pinned, external IRIS world model as a numerical learning baseline.

No IRIS implementation is vendored here. See README.md for setup and limits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

REFERENCE_COMMIT = "24326aaaa283c527f42b89b44cfdecf2665a7a16"
CHECKPOINT_REVISION = "c0539e99adf924d44be0f19b12de00639daa5550"
CHECKPOINT_SHA256 = "169e3669b7d1901990b4ebce35ff8f3355610fae393e6a587fd62c40faf790f4"
CHECKPOINT_URL = f"https://huggingface.co/eloialonso/iris/resolve/{CHECKPOINT_REVISION}/pretrained_models/Breakout.pt"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path):
    actual = file_sha256(path)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(f"expected the pinned Breakout checkpoint {CHECKPOINT_SHA256}; got {actual}")
    return actual


def download_checkpoint(path):
    if path.exists():
        verify_checkpoint(path)
        print(f"Already present and verified: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response, partial.open("xb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    verify_checkpoint(partial)
    partial.rename(path)
    print(f"Downloaded verified checkpoint: {path}")


def compare_traces(expected, actual, *, rtol=2e-5, atol=2e-6):
    if expected.keys() != actual.keys():
        raise ValueError("trace tensor names differ")
    for name in sorted(expected):
        left, right = np.asarray(expected[name]), np.asarray(actual[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{name}: shape/dtype mismatch")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"{name}: non-finite values")
        if left.dtype.kind in "biu":
            np.testing.assert_array_equal(left, right, err_msg=name)
        else:
            np.testing.assert_allclose(left, right, rtol=rtol, atol=atol, err_msg=name)


def validate_inputs(observations, actions, batch_size, horizon, num_actions):
    if observations.shape != (batch_size, 3, 64, 64) or observations.dtype != np.float32:
        raise ValueError("observations must be float32 [batch, 3, 64, 64]")
    if not np.isfinite(observations).all() or observations.min() < 0 or observations.max() > 1:
        raise ValueError("observations must be finite and in [0, 1]")
    if actions.shape != (batch_size, horizon) or actions.dtype != np.int64:
        raise ValueError("actions must be int64 [batch, horizon]")
    if actions.min() < 0 or actions.max() >= num_actions:
        raise ValueError(f"actions must be in [0, {num_actions})")


def load_reference(root, checkpoint, device):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != REFERENCE_COMMIT or dirty:
        raise ValueError(f"reference must be clean at {REFERENCE_COMMIT}; got {commit}, dirty={bool(dirty)}")
    sys.path.insert(0, str(root.resolve() / "src"))
    from models.tokenizer import Decoder, Encoder, EncoderDecoderConfig, Tokenizer
    from models.transformer import TransformerConfig
    from models.world_model import WorldModel
    from omegaconf import OmegaConf

    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tok_config = OmegaConf.load(root / "config/tokenizer/default.yaml")
    encoder_config = dict(OmegaConf.to_container(tok_config.encoder.config, resolve=True))
    encoder_config.pop("_target_")
    geometry = EncoderDecoderConfig(**encoder_config)
    tokenizer = Tokenizer(
        tok_config.vocab_size,
        tok_config.embed_dim,
        Encoder(geometry),
        Decoder(geometry),
        with_lpips=False,
    )
    world_config = dict(OmegaConf.to_container(OmegaConf.load(root / "config/world_model/default.yaml")))
    world_config.pop("_target_")
    num_actions = weights["world_model.embedder.embedding_tables.0.weight"].shape[0]
    world_model = WorldModel(tok_config.vocab_size, num_actions, TransformerConfig(**world_config))
    omitted = []
    for prefix, model in (("tokenizer", tokenizer), ("world_model", world_model)):
        selected = {}
        for name, value in weights.items():
            if not name.startswith(prefix + "."):
                continue
            relative = name.removeprefix(prefix + ".")
            if prefix == "tokenizer" and relative.startswith("lpips."):
                omitted.append(name)
            else:
                selected[relative] = value
        model.load_state_dict(selected, strict=True)
        model.to(device).eval()
    return tokenizer, world_model, num_actions, omitted


@torch.inference_mode()
def check_cached_forward(world_model, observation_tokens, num_actions):
    """Same token sequence, once densely and once as prefill + cached decode."""
    batch, width = observation_tokens.shape
    action = torch.arange(batch, device=observation_tokens.device).remainder(num_actions).view(batch, 1)
    sequence = torch.cat((observation_tokens, action, observation_tokens), dim=1)
    dense = world_model(sequence)
    cache = world_model.transformer.generate_empty_keys_values(batch, world_model.config.max_tokens)
    parts = [world_model(sequence[:, :width], past_keys_values=cache)]
    for offset in range(width, sequence.shape[1]):
        parts.append(world_model(sequence[:, offset : offset + 1], past_keys_values=cache))
    errors = {}
    for field in ("output_sequence", "logits_observations", "logits_rewards", "logits_ends"):
        expected = getattr(dense, field)
        actual = torch.cat([getattr(part, field) for part in parts], dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5, msg=field)
        errors[field] = float((actual - expected).abs().max().item())
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--observations", type=Path, help="optional .npy float32 [B,3,64,64], range [0,1]")
    parser.add_argument("--actions", type=Path, help="optional .npy int64 [B,H]")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compare", type=Path, help="previous trace.npz with identical setup")
    args = parser.parse_args(argv)
    if min(args.horizon, args.batch_size, args.threads) < 1 or args.seed < 0:
        parser.error("horizon, batch size and threads must be positive; seed must be nonnegative")
    if not args.download_checkpoint and (args.reference_root is None or args.output is None):
        parser.error("rollout requires --reference-root and --output")
    return args


@torch.inference_mode()
def run(args):
    checksum = verify_checkpoint(args.checkpoint)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer, world_model, num_actions, omitted = load_reference(args.reference_root, args.checkpoint, args.device)
    from envs.world_model_env import WorldModelEnv

    if args.observations:
        observations = np.load(args.observations, allow_pickle=False)
    else:
        pattern = np.linspace(0, 1, 3 * 64 * 64, dtype=np.float32).reshape(3, 64, 64)
        observations = np.stack([np.roll(pattern, row * 3, axis=2) for row in range(args.batch_size)])
    actions = (
        np.load(args.actions, allow_pickle=False)
        if args.actions
        else ((np.arange(args.batch_size)[:, None] + np.arange(args.horizon)[None, :]) % num_actions).astype(np.int64)
    )
    validate_inputs(observations, actions, args.batch_size, args.horizon, num_actions)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = torch.from_numpy(observations).to(args.device)
    tokens = tokenizer.encode(inputs, should_preprocess=True).tokens
    parity = check_cached_forward(world_model, tokens, num_actions)
    trace = {"initial/observations": observations.copy(), "input/actions": actions.copy()}
    call_number = 0
    step = -1
    prefills = 0

    def record(name, tensor):
        array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: non-finite tensor")
        trace[name] = array.copy()

    def hook(module, inputs, kwargs, output):
        nonlocal call_number, prefills
        call_tokens = inputs[0]
        if call_tokens.shape[1] == tokens.shape[1]:
            prefills += 1
        prefix = f"call/{call_number:04}"
        record(f"{prefix}/frame_step", np.int64(step))
        record(f"{prefix}/tokens", call_tokens)
        record(f"{prefix}/cache_size", np.int64(kwargs["past_keys_values"].size))
        for field in ("output_sequence", "logits_observations", "logits_rewards", "logits_ends"):
            record(f"{prefix}/{field}", getattr(output, field))
        call_number += 1

    # Reset after module construction: checkpoint loading must not choose the
    # categorical draws used by this reference rollout.
    torch.manual_seed(args.seed)
    handle = world_model.register_forward_hook(hook, with_kwargs=True)
    try:
        environment = WorldModelEnv(tokenizer, world_model, args.device)
        record("initial/reconstruction", environment.reset_from_initial_observations(inputs))
        record("initial/tokens", environment.obs_tokens)
        for step in range(args.horizon):
            frame, reward, done, _ = environment.step(torch.from_numpy(actions[:, step].copy()).to(args.device))
            prefix = f"step/{step:04}"
            record(f"{prefix}/frame", frame)
            record(f"{prefix}/reward", reward)
            record(f"{prefix}/done", done)
            record(f"{prefix}/tokens", environment.obs_tokens)
            record(f"{prefix}/cache_size", np.int64(environment.keys_values_wm.size))
    finally:
        handle.remove()
    if args.compare:
        with np.load(args.compare, allow_pickle=False) as previous:
            compare_traces(dict(previous), trace)
    np.savez_compressed(args.output / "trace.npz", **trace)
    counts = {
        name: sum(p.numel() for p in model.parameters())
        for name, model in (
            ("tokenizer_without_lpips", tokenizer),
            ("world_model", world_model),
        )
    }
    report = {
        "reference_commit": REFERENCE_COMMIT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": checksum,
        "checkpoint": str(args.checkpoint.resolve()),
        "game": "Breakout",
        "parameter_counts": counts,
        "rollout_parameters": sum(counts.values()),
        "num_actions": num_actions,
        "tokens_per_frame": tokens.shape[1],
        "transformer": vars(world_model.config),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "seed": args.seed,
        "threads": args.threads,
        "device": args.device,
        "input_kind": "provided_frames" if args.observations else "synthetic_test_pattern",
        "cached_vs_dense_max_abs_errors": parity,
        "cached_vs_dense_tolerances": {"rtol": 2e-5, "atol": 1e-5},
        "context_refreshes": prefills - 1,
        "trace_comparison": "passed" if args.compare else "not_run",
        "world_model_calls": call_number,
        "omitted_training_only_lpips_tensors": len(omitted),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "numpy",
                "einops",
                "gym",
                "omegaconf",
            )
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    args = parse_args(argv)
    if args.download_checkpoint:
        download_checkpoint(args.checkpoint)
    else:
        run(args)


if __name__ == "__main__":
    main()
