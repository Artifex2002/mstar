"""Capture diagnostics must reject math errors without requiring draw identity."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.model.iris.config import (
    ENDING_SAMPLER,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    INITIAL_DECODER,
    OBSERVATION_SAMPLER,
    REWARD_SAMPLER,
    WORLD_MODEL,
)
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.iris_model import IrisModel
from scripts.iris.capture_checks import (
    DiagnosticWorldSubmodule,
    check_decoded,
    check_rng_offsets,
    checked_world_outputs,
    output_digests,
)
from scripts.iris.local_engine import LocalIrisEngine
from scripts.iris.qualification import Comparisons, _modules

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / ".ci-results/iris/Breakout.pt"
TRACE = ROOT / ".ci-results/iris/context-refresh/trace.npz"


def test_digests_distinguish_steps_and_mutations():
    outputs = {1: {"tokens": [torch.tensor([1, 2])]}}
    before = output_digests(outputs, "frame=0")
    assert before == output_digests(outputs, "frame=0")
    assert before.keys() != output_digests(outputs, "frame=1").keys()
    outputs[1]["tokens"][0][0] = 9
    assert before != output_digests(outputs, "frame=0")


def test_rng_diagnostic_reads_real_masters_and_detects_reuse_resets():
    from mstar.engine.resources.sampler.utils import SamplerBuffers

    engine = SimpleNamespace(resources={})
    for key, offset in ((OBSERVATION_SAMPLER, 320), (REWARD_SAMPLER, 20), (ENDING_SAMPLER, 20)):
        buffers = SamplerBuffers.allocate(2, torch.device("cpu"))
        buffers.register_request(1)
        slots, offsets = buffers.eager_offsets([1])
        buffers.advance_eager_offsets(slots, offsets, offset)
        engine.resources[key] = SimpleNamespace(_cg_buffers=buffers)
    assert check_rng_offsets(engine, 1, 20) == {
        OBSERVATION_SAMPLER: 320,
        REWARD_SAMPLER: 20,
        ENDING_SAMPLER: 20,
    }
    with pytest.raises(AssertionError, match="no registered sampler state"):
        check_rng_offsets(engine, 2, 20)
    for resource in engine.resources.values():
        resource._cg_buffers.unregister_request(1)
        resource._cg_buffers.register_request(1)
    # Registration defers GPU resets until the next read; inspect those real
    # buffers rather than a stand-in counter or a stale graph slot.
    assert set(check_rng_offsets(engine, 1, 0).values()) == {0}
    with pytest.raises(AssertionError, match="RNG counter 0, expected 320"):
        check_rng_offsets(engine, 1, 20)


@pytest.fixture
def real_world(monkeypatch):
    if not CHECKPOINT.is_file() or not TRACE.is_file():
        pytest.skip("pinned checkpoint/official trace not cached")
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    model = IrisModel(checkpoint_path=str(CHECKPOINT))
    modules = _modules(model, "cpu")
    modules[WORLD_MODEL] = DiagnosticWorldSubmodule(modules[WORLD_MODEL].network, 20)
    try:
        with np.load(TRACE, allow_pickle=False) as data:
            trace = dict(data)
        with LocalIrisEngine(model, "cpu", ROOT / "configs/iris_capture.yaml") as engine:
            yield engine, modules, trace
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("draw_policy", ["argmax", "argmin"])
def test_official_teacher_forcing_accepts_distinct_valid_draws(real_world, monkeypatch, draw_policy):
    engine, modules, trace = real_world
    for key in (OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
        monkeypatch.setattr(engine.resources[key], "sample", lambda rids, logits: getattr(logits, draw_policy)(dim=-1))
    for rid in (1, 2):
        engine.open(rid, 1, rid)
    carry = engine.run(
        modules[IMAGE_ENCODER],
        {rid: {"initial_image": [torch.from_numpy(trace["initial/observations"][rid - 1].copy())]} for rid in (1, 2)},
    )
    engine.run(modules[INITIAL_DECODER], carry)
    comparisons, distinct_draws = Comparisons(), 0
    for tick, call in enumerate(frame_calls(0)):
        inputs = {
            rid: {**values, "actions": [torch.from_numpy(trace["input/actions"][rid - 1, :1].copy())]}
            for rid, values in carry.items()
        }
        output = engine.run(modules[WORLD_MODEL], inputs, {rid: (0, tick) for rid in carry})
        if call.sample_index is not None:
            logits = output[1]["diagnostic_logits_observations"][0][0]
            distinct_draws += int(logits.argmax() != logits.argmin())
        carry = checked_world_outputs(output, carry, 0, tick, tick, comparisons, trace)
    assert distinct_draws == 16
    assert comparisons.tensors == 4 * 18 * 2
    check_decoded(engine.run(modules[IMAGE_DECODER], carry), carry, 0, trace)


@pytest.mark.parametrize("corruption", ["logits", "untouched_carrier", "invalid_draw"])
def test_numerical_and_carrier_errors_still_fail_with_step_context(real_world, corruption):
    engine, modules, trace = real_world
    engine.open(1, 1)
    carry = engine.run(
        modules[IMAGE_ENCODER], {1: {"initial_image": [torch.from_numpy(trace["initial/observations"][0].copy())]}}
    )
    output = engine.run(modules[WORLD_MODEL], {1: {**carry[1], "actions": [torch.tensor([0])]}}, {1: (0, 0)})
    if corruption == "logits":
        output[1]["diagnostic_logits_observations"][0] = output[1]["diagnostic_logits_observations"][0] + 1
    else:
        # Prime has no sampled column: every carrier value must be preserved.
        output[1]["observation_tokens"] = [carry[1]["observation_tokens"][0].clone()]
        output[1]["observation_tokens"][0][0] = (
            512 if corruption == "invalid_draw" else (int(carry[1]["observation_tokens"][0][0]) + 1) % 512
        )
    with pytest.raises(AssertionError, match="frame=0 tick=0 phase=prime request=1") as caught:
        checked_world_outputs(output, carry, 0, 0, 0, Comparisons(), trace)
    if corruption == "logits":
        assert "logits_observations" in str(caught.value) and "Greatest absolute difference" in str(caught.value)
