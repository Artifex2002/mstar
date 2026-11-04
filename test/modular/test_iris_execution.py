"""Real submodule/resource execution; cached official weights/traces are optional."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.engine.resources import StepContext, StepRunner, apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.model.iris.checkpoint import inference_state, resolve_checkpoint
from mstar.model.iris.config import (
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    ENDING_SAMPLER,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    IMAGINE_WALK,
    INITIAL_DECODER,
    OBSERVATION_SAMPLER,
    REWARD_SAMPLER,
    ROLLOUT_LOOP,
    TOKEN_LOOP,
    WORLD_KV,
    WORLD_MODEL,
    WORLD_POS,
)
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.iris_model import IrisModel
from mstar.model.iris.resources import request_kv_layout
from mstar.model.submodule_base import ModelInputsFromEngine

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".ci-results/iris"


def test_checkpoint_hash_is_checked_before_deserialization(tmp_path):
    path = tmp_path / "Breakout.pt"
    path.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_checkpoint(path)


def test_unknown_checkpoint_keys_cannot_be_silently_discarded():
    with pytest.raises(ValueError, match="Unknown IRIS checkpoint key"):
        inference_state({"unexpected.weight": torch.empty(1)})


@pytest.fixture
def execution(monkeypatch, request):
    from mstar.engine.resources.kv import manager as manager_mod

    class NoTransfer:
        def __init__(self, *args, **kwargs):
            pass

        def remove_request(self, rid):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(manager_mod, "KVTransferManager", NoTransfer)
    device = torch.device(getattr(request, "param", "cpu"))
    if device.type == "cpu":
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    model = IrisModel(checkpoint_path=str(CACHE / "Breakout.pt"))
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, yaml.safe_load((ROOT / "configs/iris.yaml").read_text()))
    specs = resolve_spec_dependencies(specs)
    resources = {
        key: build_resource(
            spec,
            EngineResourceInfo(
                device=device,
                kv_dtype=torch.float32,
                dependencies={dep: specs[dep] for dep in spec.depends_on()},
            ),
        )
        for key, spec in specs.items()
    }
    nodes = {
        node: [key for key, spec in specs.items() if node in spec.nodes]
        for node in (IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER)
    }
    runner = StepRunner(resources, node_resources=nodes)
    configs = model.get_request_resource_configs({}, {"iris_horizon": 20})
    configs[WORLD_KV].apply_conductor_config(
        kv_layout={label: layout.row() for label, layout in request_kv_layout(20)[WORLD_KV].items()}, loop_caps={}
    )
    for key in (OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
        configs[key].apply_conductor_config(seed=0, resource_key=key)
    runner.ingest_request(1, configs)
    yield model, resources, runner
    runner.remove_request(1)
    for resource in resources.values():
        resource.cleanup()


def info(frame=0, token=0):
    return CurrentForwardPassInfo(
        "request",
        IMAGINE_WALK,
        0,
        0,
        20,
        rid_handle=1,
        step_metadata={"iris_horizon": 20},
        dynamic_loop_iter_counts={ROLLOUT_LOOP: frame, TOKEN_LOOP: token},
    )


def execute(module, runner, resources, inputs, metadata):
    row = module.prepare_inputs(IMAGINE_WALK, metadata, inputs)
    assert row is not None
    step = module.declare_step(IMAGINE_WALK, [1], [row])
    step.set_ctx(StepContext((1,), IMAGINE_WALK, 0, False))
    assert runner.admit(step).outcome.ok
    runner.plan(step)
    engine = ModelInputsFromEngine([1], {1: metadata}, resources=resources, step=step)
    with fp32_precision_scope(node_fp32_precision(type(module).__name__, module)):
        result = module(IMAGINE_WALK, engine, **module.preprocess(IMAGINE_WALK, engine, [row]))
    runner.commit(step)
    return result


def test_eager_factory_keeps_existing_backend_defaults_and_resource_lifecycle(execution, monkeypatch):
    from mstar.engine.resources.attn.config import AttnBackend
    from mstar.engine.resources.attn.torch_reference import TorchPagedAttention, TorchRaggedAttention

    model, resources, runner = execution
    assert model.get_node_resources()[1].config.backend is AttnBackend.AUTO
    assert isinstance(resources["world_attn"], TorchPagedAttention)
    assert isinstance(resources["encoder_attn"], TorchRaggedAttention)
    # Exercise admission, actual writes, attention and positions together without a checkpoint.
    from mstar.model.iris.components import WorldNetwork
    from mstar.model.iris.submodules import WorldSubmodule

    world = WorldSubmodule(WorldNetwork().eval(), 20)
    # The production sampler supports CUDA/XPU. Supply deterministic CPU draws
    # while retaining its real ingest/admit/plan/commit lifecycle.
    for key in (OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
        monkeypatch.setattr(resources[key], "sample", lambda rids, logits: logits.argmax(-1))
    carry = {
        "observation_tokens": [torch.zeros(16, dtype=torch.long)],
        "reward_class": [torch.ones(1, dtype=torch.long)],
        "ending_class": [torch.zeros(1, dtype=torch.long)],
    }
    actions = torch.zeros(20, dtype=torch.long)
    with torch.inference_mode():
        for token in range(18):
            previous = carry["observation_tokens"][0].clone()
            old_tensor = carry["observation_tokens"][0]
            carry = execute(world, runner, resources, {**carry, "actions": [actions]}, info(token=token))
            torch.testing.assert_close(old_tensor, previous, rtol=0, atol=0)
            assert world.check_stop(1, info(token=token), carry) == ({TOKEN_LOOP} if token == 17 else set())
        assert resources[WORLD_POS].position(1, "epoch_0") == 33
        assert carry["reward_class"][0].numel() == 1
        assert 0 <= carry["reward_class"][0].item() < 3
        assert 0 <= carry["ending_class"][0].item() < 2
        assert world.prepare_inputs(IMAGINE_WALK, info(token=18), {**carry, "actions": [actions]}) is None


@pytest.mark.skipif(
    not (CACHE / "Breakout.pt").is_file() or not (CACHE / "context-refresh/trace.npz").is_file(),
    reason="pinned official checkpoint and reference trace are not cached",
)
@torch.inference_mode()
@pytest.mark.parametrize("trace_row", [0, 1])
def test_pinned_weights_match_official_teacher_forced_twenty_frame_trace(execution, monkeypatch, trace_row):
    model, resources, runner = execution
    report = json.loads((CACHE / "context-refresh/report.json").read_text())
    assert report["checkpoint_revision"] == CHECKPOINT_REVISION
    assert report["checkpoint_sha256"] == CHECKPOINT_SHA256
    with np.load(CACHE / "context-refresh/trace.npz", allow_pickle=False) as data:
        trace = dict(data)
    encoder, world, decoder, initial = (
        model.get_submodule(node) for node in (IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER)
    )
    assert initial.decoder is decoder.decoder
    assert encoder.embedding is decoder.embedding
    assert callable(model.checkpoint_path)
    assert model.get_submodule(WORLD_MODEL) is world
    assert sum(p.numel() for p in world.parameters()) == 8_447_493
    assert sum(p.numel() for p in model._networks["cpu"].tokenizer.parameters()) == 4_027_523
    with pytest.raises(ValueError, match="requires FP32"):
        model.get_submodule(WORLD_MODEL, autocast_dtype=torch.bfloat16)
    image = torch.from_numpy(trace["initial/observations"][trace_row].copy())
    carry = execute(encoder, runner, resources, {"initial_image": [image]}, info())
    torch.testing.assert_close(
        carry["observation_tokens"][0], torch.from_numpy(trace["initial/tokens"][trace_row]), rtol=0, atol=0
    )
    # The initial decoder really runs, but creates no client frame.
    decoder_calls = []
    handle = initial.decoder.register_forward_hook(lambda module, args, output: decoder_calls.append(output))
    assert execute(initial, runner, resources, carry, info()) == {}
    handle.remove()
    assert len(decoder_calls) == 1
    reconstruction = ((decoder_calls.pop() + 1) / 2).clamp(0, 1)
    torch.testing.assert_close(
        reconstruction,
        torch.from_numpy(trace["initial/reconstruction"][trace_row : trace_row + 1].copy()),
        rtol=2e-5,
        atol=1e-5,
    )
    handle_decode = decoder.decoder.register_forward_hook(lambda module, args, output: decoder_calls.append(output))
    actions = torch.from_numpy(trace["input/actions"][trace_row].copy())
    calls, errors = [], []
    handle = world.network.register_forward_hook(lambda module, args, output: calls.append(output))
    call_number = 0
    try:
        for frame in range(20):
            expected_tokens = torch.from_numpy(trace[f"step/{frame:04}/tokens"][trace_row].copy())
            for token, call in enumerate(frame_calls(frame)):
                if call.sample_index is not None:
                    monkeypatch.setattr(
                        resources[OBSERVATION_SAMPLER],
                        "sample",
                        lambda *args, index=call.sample_index, values=expected_tokens: values[index : index + 1],
                    )
                monkeypatch.setattr(
                    resources[REWARD_SAMPLER],
                    "sample",
                    lambda *args, f=frame: torch.from_numpy(
                        np.asarray(trace[f"step/{f:04}/reward"][trace_row] + 1, dtype=np.int64).reshape(1)
                    ),
                )
                monkeypatch.setattr(
                    resources[ENDING_SAMPLER],
                    "sample",
                    lambda *args, f=frame: torch.from_numpy(
                        np.asarray(trace[f"step/{f:04}/done"][trace_row], dtype=np.int64).reshape(1)
                    ),
                )
                carry = execute(world, runner, resources, {**carry, "actions": [actions]}, info(frame, token))
                output = calls.pop()
                for field, actual in zip(output._fields, output, strict=True):
                    expected = torch.from_numpy(
                        trace[f"call/{call_number:04}/{field}"][trace_row : trace_row + 1].copy()
                    )
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5, msg=f"{call_number}/{field}")
                    if actual.numel():
                        errors.append((actual - expected).abs().max().item())
                assert resources[WORLD_POS].position(1, call.label) == int(trace[f"call/{call_number:04}/cache_size"])
                call_number += 1
            result = execute(decoder, runner, resources, carry, info(frame))
            expected_frame = torch.from_numpy(trace[f"step/{frame:04}/frame"][trace_row : trace_row + 1].copy())
            decoded = ((decoder_calls.pop() + 1) / 2).clamp(0, 1)
            torch.testing.assert_close(decoded, expected_frame, rtol=2e-5, atol=5e-5)
            # Conversion to RGB24 can cross a truncation boundary with FP32 roundoff.
            expected_bytes = (expected_frame * 255).to(torch.uint8).permute(0, 2, 3, 1)
            assert (result["frames"][0].to(torch.int16) - expected_bytes.to(torch.int16)).abs().max() <= 1
            torch.testing.assert_close(
                result["reward"][0],
                torch.from_numpy(np.asarray(trace[f"step/{frame:04}/reward"][trace_row], dtype=np.float32).reshape(1)),
            )
            assert result["ending"][0].item() == bool(trace[f"step/{frame:04}/done"][trace_row])
        assert call_number == 342
        assert resources[WORLD_POS].position(1, "epoch_0") == 339
        assert resources[WORLD_POS].position(1, "epoch_1") == 33
        print(f"342 official world calls: maximum FP32 absolute error {max(errors):.8g}")
    finally:
        handle.remove()
        handle_decode.remove()


@pytest.mark.skipif(
    not (CACHE / "Breakout.pt").is_file() or not (CACHE / "reference/.git").exists(),
    reason="official checkpoint and source checkout are not cached",
)
@torch.inference_mode()
def test_tokenizer_matches_fresh_official_execution_at_the_same_batch_size(execution):
    from scripts.iris.reference_harness import load_reference

    model, resources, runner = execution
    reference, _, _, _ = load_reference(CACHE / "reference", CACHE / "Breakout.pt", "cpu")
    encoder, decoder = model.get_submodule(IMAGE_ENCODER), model.get_submodule(IMAGE_DECODER)
    generator = torch.Generator().manual_seed(4)
    errors = []
    for image in (torch.rand(3, 64, 64, generator=generator), torch.rand(3, 64, 64, generator=generator)):
        carry = execute(encoder, runner, resources, {"initial_image": [image]}, info())
        expected = reference.encode(image.unsqueeze(0), should_preprocess=True)
        torch.testing.assert_close(carry["observation_tokens"][0], expected.tokens[0], rtol=0, atol=0)
        outputs = []
        handle = decoder.decoder.register_forward_hook(
            lambda module, args, output, values=outputs: values.append(output)
        )
        try:
            execute(decoder, runner, resources, carry, info())
        finally:
            handle.remove()
        actual = ((outputs[0] + 1) / 2).clamp(0, 1)
        expected_rgb = reference.decode(expected.z_quantized, should_postprocess=True).clamp(0, 1)
        torch.testing.assert_close(actual, expected_rgb, rtol=2e-5, atol=1e-5)
        errors.append((actual - expected_rgb).abs().max().item())
    print(f"Same-batch official tokenizer maximum RGB absolute error {max(errors):.8g}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (CACHE / "Breakout.pt").is_file(),
    reason="CUDA and the pinned checkpoint are required",
)
@pytest.mark.parametrize("execution", ["cuda"], indirect=True)
@torch.inference_mode()
def test_cuda_eager_rollout_uses_real_sampling_through_the_context_refresh(execution):
    model, resources, runner = execution
    encoder, world, decoder, initial = (
        model.get_submodule(node, device="cuda")
        for node in (IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER)
    )
    image = torch.linspace(0, 1, 3 * 64 * 64, device="cuda").reshape(3, 64, 64)
    carry = execute(encoder, runner, resources, {"initial_image": [image]}, info())
    assert execute(initial, runner, resources, carry, info()) == {}
    actions = torch.arange(20, device="cuda") % 4
    for frame in range(20):
        for token in range(len(frame_calls(frame))):
            carry = execute(world, runner, resources, {**carry, "actions": [actions]}, info(frame, token))
        result = execute(decoder, runner, resources, carry, info(frame))
        assert result["frames"][0].shape == (1, 64, 64, 3)
        assert result["frames"][0].dtype == torch.uint8
        assert result["reward"][0].item() in (-1.0, 0.0, 1.0)
        assert result["ending"][0].dtype == torch.bool
        assert ((carry["observation_tokens"][0] >= 0) & (carry["observation_tokens"][0] < 512)).all()
    assert resources[WORLD_POS].position(1, "epoch_0") == 339
    assert resources[WORLD_POS].position(1, "epoch_1") == 33
    assert all(parameter.dtype == torch.float32 for parameter in world.parameters())
    print(f"CUDA eager rollout: {torch.cuda.get_device_name()}, torch={torch.__version__}, CUDA={torch.version.cuda}")
