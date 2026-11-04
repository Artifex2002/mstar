"""Numerical checks shared by CPU tests and the CUDA qualification command."""

from collections import Counter
from contextlib import ExitStack

import numpy as np
import torch

from mstar.model.iris.config import (
    ENDING_SAMPLER,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    INITIAL_DECODER,
    OBSERVATION_SAMPLER,
    REWARD_SAMPLER,
    WORLD_KV,
    WORLD_MODEL,
    WORLD_POS,
)
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.iris_model import IrisModel
from scripts.iris.local_engine import LocalIrisEngine


class Comparisons:
    def __init__(self):
        self.max_abs_errors = Counter()
        self.tensors = 0

    def check(self, name, actual, expected, rtol=2e-5, atol=1e-5):
        actual = actual.detach().cpu()
        expected = torch.as_tensor(np.array(expected, copy=True))
        if not torch.isfinite(actual).all():
            raise AssertionError(f"{name}: non-finite output")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
        if actual.numel():
            self.max_abs_errors[name] = max(
                self.max_abs_errors[name], (actual.double() - expected.double()).abs().max().item()
            )
        self.tensors += 1

    def report(self):
        return {"tensors_checked": self.tensors, "max_abs_errors": dict(self.max_abs_errors)}


def _modules(model, device):
    return {
        node: model.get_submodule(node, device=device)
        for node in (IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER)
    }


def decoded_rgb(raw):
    if not torch.isfinite(raw).all():
        raise AssertionError("decoder produced non-finite RGB values")
    return ((raw + 1) / 2).clamp(0, 1)


@torch.inference_mode()
def compare_trace(checkpoint, trace, device="cpu", batch_size=2, config_path=None):
    """Actual batched submodules/resources, with official samples fed back."""
    model = IrisModel(checkpoint_path=str(checkpoint), max_batch_size=batch_size)
    modules = _modules(model, device)
    horizon = trace["input/actions"].shape[1]
    if horizon > model.config.max_rollout_horizon:
        raise ValueError("trace exceeds the deployment horizon")
    compare, calls = Comparisons(), Counter()
    captured_world, captured_rgb = [], []
    world_hook = modules[WORLD_MODEL].network.register_forward_hook(
        lambda module, args, output: captured_world.append(output)
    )
    rgb_hook = modules[IMAGE_DECODER].decoder.register_forward_hook(
        lambda module, args, output: captured_rgb.append(output)
    )
    rids = list(range(1, batch_size + 1))
    try:
        with LocalIrisEngine(model, device, config_path=config_path) as engine:
            for rid in rids:
                engine.open(rid, horizon, seed=rid)
            inputs = {
                rid: {"initial_image": [torch.from_numpy(trace["initial/observations"][rid - 1].copy()).to(device)]}
                for rid in rids
            }
            carry = engine.run(modules[IMAGE_ENCODER], inputs)
            actions = {rid: torch.from_numpy(trace["input/actions"][rid - 1].copy()).to(device) for rid in rids}
            codes = torch.stack([carry[rid]["observation_tokens"][0] for rid in rids])
            compare.check("encoder_codes", codes, trace["initial/tokens"][:batch_size], rtol=0, atol=0)
            assert all(not output for output in engine.run(modules[INITIAL_DECODER], carry).values())
            compare.check(
                "initial_rgb",
                decoded_rgb(captured_rgb.pop()),
                trace["initial/reconstruction"][:batch_size],
                atol=5e-5,
            )
            call_number = 0
            for frame in range(horizon):
                for token, call in enumerate(frame_calls(frame)):
                    draws = {}
                    if call.sample_index is not None:
                        draws[OBSERVATION_SAMPLER] = torch.from_numpy(
                            trace[f"step/{frame:04}/tokens"][:batch_size, call.sample_index].copy()
                        ).to(device)
                    if call.kind == "action":
                        draws[REWARD_SAMPLER] = torch.from_numpy(
                            (trace[f"step/{frame:04}/reward"][:batch_size] + 1).astype(np.int64)
                        ).to(device)
                        draws[ENDING_SAMPLER] = torch.from_numpy(
                            trace[f"step/{frame:04}/done"][:batch_size].astype(np.int64)
                        ).to(device)
                    inputs = {rid: {**carry[rid], "actions": [actions[rid]]} for rid in rids}
                    carry = engine.run(modules[WORLD_MODEL], inputs, {rid: (frame, token) for rid in rids}, draws)
                    output = captured_world.pop()
                    for field, tensor in zip(output._fields, output, strict=True):
                        compare.check(field, tensor, trace[f"call/{call_number:04}/{field}"][:batch_size])
                    for rid in rids:
                        assert engine.resources[WORLD_POS].position(rid, call.label) == int(
                            trace[f"call/{call_number:04}/cache_size"]
                        )
                    call_number += 1
                output = engine.run(modules[IMAGE_DECODER], carry)
                compare.check(
                    "predicted_rgb",
                    decoded_rgb(captured_rgb.pop()),
                    trace[f"step/{frame:04}/frame"][:batch_size],
                    atol=5e-5,
                )
                for rid in rids:
                    compare.check(
                        "reward",
                        output[rid]["reward"][0],
                        trace[f"step/{frame:04}/reward"][rid - 1 : rid],
                        rtol=0,
                        atol=0,
                    )
                    compare.check(
                        "ending",
                        output[rid]["ending"][0],
                        trace[f"step/{frame:04}/done"][rid - 1 : rid],
                        rtol=0,
                        atol=0,
                    )
                    expected_bytes = (
                        (torch.from_numpy(trace[f"step/{frame:04}/frame"][rid - 1 : rid].copy()) * 255)
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                    )
                    byte_error = (
                        (output[rid]["frames"][0].cpu().to(torch.int16) - expected_bytes.to(torch.int16))
                        .abs()
                        .max()
                        .item()
                    )
                    if byte_error > 1:
                        raise AssertionError(f"frame {frame}, row {rid}: RGB24 differs by {byte_error}")
                    calls["decoded_frames"] += 1
            calls["world_batch_calls"] = call_number
            calls["world_row_calls"] = call_number * batch_size
            for rid in rids:
                engine.close(rid)
            assert not engine.horizons
            assert not engine.resources[WORLD_KV]._streams
    finally:
        world_hook.remove()
        rgb_hook.remove()
    return {"batch_size": batch_size, "horizon": horizon, **dict(calls), **compare.report()}


@torch.inference_mode()
def sampling_smoke(checkpoint, device="cuda", batch_size=2):
    """Free-running CUDA/XPU sampling, including a request leaving the batch."""
    model = IrisModel(checkpoint_path=str(checkpoint), max_batch_size=batch_size)
    modules = _modules(model, device)
    horizons = {1: 20, 2: 2} if batch_size == 2 else {1: 20}
    frames = Counter()

    def check_world(module, args, output):
        for value in output:
            if not torch.isfinite(value).all():
                raise AssertionError("world model produced non-finite features/logits")

    def check_rgb(module, args, output):
        decoded_rgb(output)

    # These diagnostic hooks intentionally synchronize. They are not used by
    # serving or timing, and keep a bad value from being hidden by RGB clamping.
    with ExitStack() as stack, LocalIrisEngine(model, device) as engine:
        stack.callback(modules[WORLD_MODEL].network.register_forward_hook(check_world).remove)
        stack.callback(modules[IMAGE_DECODER].decoder.register_forward_hook(check_rgb).remove)
        for rid, horizon in horizons.items():
            engine.open(rid, horizon, seed=rid)
        inputs = {
            rid: {"initial_image": [torch.linspace(0, 1, 3 * 64 * 64, device=device).reshape(3, 64, 64).roll(rid, -1)]}
            for rid in horizons
        }
        carry = engine.run(modules[IMAGE_ENCODER], inputs)
        engine.run(modules[INITIAL_DECODER], carry)
        actions = {rid: torch.arange(horizon, device=device) % 4 for rid, horizon in horizons.items()}
        for frame in range(20):
            for token in range(len(frame_calls(frame))):
                inputs = {rid: {**carry[rid], "actions": [actions[rid]]} for rid in carry}
                carry = engine.run(modules[WORLD_MODEL], inputs, {rid: (frame, token) for rid in carry})
            output = engine.run(modules[IMAGE_DECODER], carry)
            for rid, values in output.items():
                assert values["frames"][0].shape == (1, 64, 64, 3)
                assert values["frames"][0].dtype == torch.uint8
                assert values["reward"][0].item() in (-1, 0, 1)
                assert values["ending"][0].dtype == torch.bool
                codes = carry[rid]["observation_tokens"][0]
                assert ((codes >= 0) & (codes < 512)).all()
                frames[rid] += 1
            for rid in list(carry):
                if frame + 1 == horizons[rid]:
                    engine.close(rid)
                    del carry[rid]
        assert dict(frames) == horizons
        # Reusing the handle must start with fresh KV, positions and sampler state.
        engine.open(1, 1, seed=1)
        assert engine.resources[WORLD_POS].position(1, "epoch_0") == 0
        assert engine.resources[WORLD_POS].position(1, "epoch_1") == 0
        for key in (OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
            assert engine.resources[key]._sampler._step_offset[1] == 0
        fresh_image = torch.zeros(3, 64, 64, device=device)
        fresh_actions = torch.zeros(1, dtype=torch.long, device=device)
        fresh = engine.run(modules[IMAGE_ENCODER], {1: {"initial_image": [fresh_image]}})
        for token in range(len(frame_calls(0))):
            fresh = engine.run(modules[WORLD_MODEL], {1: {**fresh[1], "actions": [fresh_actions]}}, {1: (0, token)})
        assert engine.resources[WORLD_POS].position(1, "epoch_0") == 33
    return {"batch_size": batch_size, "decoded_frames_by_request": dict(frames), "handle_reuse": "passed"}
