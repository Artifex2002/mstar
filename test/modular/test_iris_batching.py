"""Real batched neural execution versus independent request contexts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mstar.model.iris.components import WorldNetwork
from mstar.model.iris.config import ENDING_SAMPLER, OBSERVATION_SAMPLER, REWARD_SAMPLER, TOKEN_LOOP, WORLD_KV, WORLD_POS
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.iris_model import IrisModel
from mstar.model.iris.submodules import WorldSubmodule
from scripts.iris.local_engine import LocalIrisEngine
from scripts.iris.qualification import compare_trace, decoded_rgb, sampling_smoke
from scripts.iris.qualify_cuda import main

CACHE = Path(__file__).resolve().parents[2] / ".ci-results/iris"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_batch_limit_requires_a_positive_integer(limit):
    with pytest.raises(ValueError, match="max_batch_size"):
        IrisModel(max_batch_size=limit)


@pytest.fixture
def cpu_staging(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def carrier(rid):
    return {
        "observation_tokens": [(torch.arange(16) + rid * 23) % 512],
        "reward_class": [torch.ones(1, dtype=torch.long)],
        "ending_class": [torch.zeros(1, dtype=torch.long)],
        "actions": [torch.tensor([rid % 4, (rid + 1) % 4])],
    }


def draws(rids, frame, call):
    result = {}
    if call.sample_index is not None:
        result[OBSERVATION_SAMPLER] = torch.tensor(
            [(rid * 31 + frame[rid] * 17 + call.sample_index) % 512 for rid in rids]
        )
    if call.kind == "action":
        result[REWARD_SAMPLER] = torch.tensor([(rid + frame[rid]) % 3 for rid in rids])
        result[ENDING_SAMPLER] = torch.tensor([(rid + frame[rid]) % 2 for rid in rids])
    return result


@torch.inference_mode()
def test_mixed_frame_positions_and_reordered_rows_match_independent_execution(cpu_staging):
    model = IrisModel(skip_weight_loading=True)
    world = WorldSubmodule(WorldNetwork().eval(), 20)
    captured = []
    hook = world.network.register_forward_hook(lambda module, args, output: captured.append(output))
    with LocalIrisEngine(model) as batched, LocalIrisEngine(model) as independent:
        for engine in (batched, independent):
            engine.open(1, 2)
            engine.open(2, 1)
        carry = {rid: carrier(rid) for rid in (1, 2)}
        reference = {rid: carrier(rid) for rid in (1, 2)}

        def advance(indices):
            rids = list(indices)
            first_rid = rids[0]
            frame, tick = indices[first_rid]
            call = frame_calls(frame)[tick]
            frame_indices = {rid: indices[rid][0] for rid in rids}
            forced = draws(rids, frame_indices, call)
            before = {rid: carry[rid]["observation_tokens"][0].clone() for rid in rids}
            original = {rid: carry[rid]["observation_tokens"][0] for rid in rids}
            output = batched.run(world, {rid: carry[rid] for rid in rids}, indices, forced)
            batch_output = captured.pop()
            reference_outputs = []
            for index, rid in enumerate(rids):
                expected_stop = {TOKEN_LOOP} if indices[rid][1] + 1 == len(frame_calls(indices[rid][0])) else set()
                assert world.check_stop(rid, batched.info(rid, *indices[rid]), output[rid]) == expected_stop
                single = independent.run(
                    world,
                    {rid: reference[rid]},
                    {rid: indices[rid]},
                    {key: value[index : index + 1] for key, value in forced.items()},
                )
                reference_outputs.append(captured.pop())
                reference[rid] = {**single[rid], "actions": reference[rid]["actions"]}
                torch.testing.assert_close(original[rid], before[rid], rtol=0, atol=0)
                for name, values in output[rid].items():
                    torch.testing.assert_close(values[0], single[rid][name][0], rtol=0, atol=0)
                carry[rid] = {**output[rid], "actions": carry[rid]["actions"]}
            for field in batch_output._fields:
                expected = torch.cat([getattr(value, field) for value in reference_outputs])
                torch.testing.assert_close(getattr(batch_output, field), expected, rtol=2e-5, atol=1e-5)

        # Prime both, then advance only request 1 through its first frame.
        advance({1: (0, 0), 2: (0, 0)})
        for tick in range(1, 18):
            advance({1: (0, tick)})
        # Different context lengths, different actions and reversed row order.
        for tick in range(17):
            advance({2: (0, tick + 1), 1: (1, tick)})
        assert batched.resources[WORLD_POS].position(1, "epoch_0") == 50
        assert batched.resources[WORLD_POS].position(2, "epoch_0") == 33
        for engine in (batched, independent):
            engine.close(2)
            engine.close(1)
            assert not engine.resources[WORLD_KV]._streams
            engine.open(1, 1)
            assert engine.resources[WORLD_POS].position(1, "epoch_0") == 0
    hook.remove()


def test_incompatible_phases_and_epochs_are_rejected_before_planning(cpu_staging):
    model = IrisModel(skip_weight_loading=True)
    world = WorldSubmodule(WorldNetwork(), 20)
    with LocalIrisEngine(model) as engine:
        engine.open(1, 20)
        engine.open(2, 20)
        inputs = carrier(1)
        inputs["actions"] = [torch.arange(20) % 4]
        prime = world.prepare_inputs("imagine", engine.info(1, 0, 0), inputs)
        action = world.prepare_inputs("imagine", engine.info(2, 0, 1), inputs)
        epoch_one = world.prepare_inputs("imagine", engine.info(1, 19, 1), inputs)
        assert not world.can_batch(None, [prime, action])
        assert not world.can_batch(None, [action, epoch_one])
        with pytest.raises(ValueError, match="incompatible"):
            engine.run(world, {1: inputs, 2: inputs}, {1: (0, 0), 2: (0, 1)})
        assert engine.resources[WORLD_POS].position(1, "epoch_0") == 0
        assert engine.resources[WORLD_POS].position(2, "epoch_0") == 0
        assert world.get_accelerator_graph_configs(torch.device("cpu")) == []


@pytest.mark.skipif(
    not (CACHE / "Breakout.pt").is_file() or not (CACHE / "context-refresh/trace.npz").is_file(),
    reason="pinned checkpoint and official trace are not cached",
)
def test_batched_checkpoint_matches_both_official_rollouts_through_refresh(cpu_staging):
    with np.load(CACHE / "context-refresh/trace.npz", allow_pickle=False) as data:
        report = compare_trace(CACHE / "Breakout.pt", dict(data))
    assert report["world_batch_calls"] == 342
    assert report["world_row_calls"] == 684
    assert report["decoded_frames"] == 40


@pytest.mark.skipif(not (CACHE / "Breakout.pt").is_file(), reason="pinned checkpoint is not cached")
def test_sampling_driver_covers_batch_departure_and_handle_reuse_on_cpu(cpu_staging, monkeypatch):
    from mstar.engine.resources.sampler.resource import SamplerResource

    # Check the driver on CPU; the CUDA command uses the real implementation.
    monkeypatch.setattr(SamplerResource, "sample", lambda self, rids, logits: logits.argmax(-1))
    report = sampling_smoke(CACHE / "Breakout.pt", device="cpu", batch_size=2)
    assert report["decoded_frames_by_request"] == {1: 20, 2: 2}
    assert report["handle_reuse"] == "passed"


def test_nonfinite_rgb_is_detected_before_clamping():
    with pytest.raises(AssertionError, match="non-finite"):
        decoded_rgb(torch.tensor([float("inf")]))


def test_cuda_command_fails_without_a_gpu_and_writes_a_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "qualification"
    assert main(["--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "cuda_preflight"
    assert "no CPU fallback" in report["error"]
