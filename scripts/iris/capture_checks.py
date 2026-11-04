"""Diagnostic graph outputs and independent numerical/sampling checks.

These synchronize and copy tensors deliberately. They are never serving code
or timing instrumentation.
"""

import hashlib

import numpy as np
import torch

from mstar.model.iris.components import WorldOutput
from mstar.model.iris.config import ENDING_SAMPLER, OBSERVATION_SAMPLER, REWARD_SAMPLER, WORLD_CARRY
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.submodules import WorldSubmodule

DEBUG_PREFIX = "diagnostic_"


class DiagnosticWorldSubmodule(WorldSubmodule):
    """Return intermediates as graph outputs; Python hooks cannot inspect replay."""

    def compute(self, engine_inputs, tokens, call):
        result = super().compute(engine_inputs, tokens, call)
        self._diagnostic_output = result
        return result

    def forward_tensors(self, *args, **kwargs):
        outputs = super().forward_tensors(*args, **kwargs)
        outputs.update(
            {
                DEBUG_PREFIX + name: tensor
                for name, tensor in zip(WorldOutput._fields, self._diagnostic_output, strict=True)
            }
        )
        return outputs


def checked_world_outputs(outputs, previous, frame, tick, call_number, comparisons, trace=None):
    """Check real draws first, then optionally feed official samples to the next step.

    The copied carriers can be teacher-forced after replay without replacing a
    sampler inside a captured graph. Logits/hidden states must match the official
    trace before this substitution. Unmodified carrier columns remain exact.
    """
    call = frame_calls(frame)[tick]
    assert outputs.keys() == previous.keys(), f"frame={frame} tick={tick}: world output rows changed"
    carry = {}
    for rid, values in outputs.items():
        context = f"frame={frame} tick={tick} phase={call.kind} request={rid}"
        try:
            assert set(values) == set(WORLD_CARRY) | {DEBUG_PREFIX + field for field in WorldOutput._fields}
            row = {name: [values[name][0].clone()] for name in WORLD_CARRY}
            expected = {name: [previous[rid][name][0].clone()] for name in WORLD_CARRY}
            for name, size, upper in (("observation_tokens", 16, 512), ("reward_class", 1, 3), ("ending_class", 1, 2)):
                value = row[name][0]
                assert value.dtype == torch.long and value.shape == (size,), f"invalid {name} schema"
                assert ((value >= 0) & (value < upper)).all(), f"invalid {name} draw"
            if call.sample_index is not None:
                expected["observation_tokens"][0][call.sample_index] = row["observation_tokens"][0][call.sample_index]
            if call.kind == "action":
                expected["reward_class"] = row["reward_class"]
                expected["ending_class"] = row["ending_class"]
            for name in WORLD_CARRY:
                torch.testing.assert_close(row[name][0], expected[name][0], rtol=0, atol=0)
            for field in WorldOutput._fields:
                actual = values[DEBUG_PREFIX + field][0]
                assert torch.isfinite(actual).all(), f"non-finite {field}"
                if trace is not None:
                    comparisons.check(field, actual, trace[f"call/{call_number:04}/{field}"][rid - 1])
            if trace is not None:
                if call.sample_index is not None:
                    row["observation_tokens"][0][call.sample_index] = int(
                        trace[f"step/{frame:04}/tokens"][rid - 1, call.sample_index]
                    )
                if call.kind == "action":
                    row["reward_class"][0].fill_(int(trace[f"step/{frame:04}/reward"][rid - 1]) + 1)
                    row["ending_class"][0].fill_(int(trace[f"step/{frame:04}/done"][rid - 1]))
            carry[rid] = row
        except AssertionError as exc:
            raise AssertionError(f"{context}: {exc}") from exc
    return carry


def output_digests(outputs, context):
    """Per-tensor fingerprints retain the first differing step on a repeat run."""
    return {
        f"{context}/request={rid}/{name}": hashlib.sha256(
            value[0].detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for rid, values in outputs.items()
        for name, value in values.items()
    }


def check_rng_offsets(engine, rid, frames):
    """Read real sampler masters outside replay; diagnostic use only."""
    result = {}
    for key, draws in ((OBSERVATION_SAMPLER, 16 * frames), (REWARD_SAMPLER, frames), (ENDING_SAMPLER, frames)):
        buffers = engine.resources[key]._cg_buffers
        assert buffers is not None, f"{key}: captured sampler buffers missing"
        assert buffers.has_slot(rid), f"{key}/request={rid}: no registered sampler state"
        _, offsets = buffers.eager_offsets([rid])
        actual = offsets.item()
        assert actual == draws, f"{key}/request={rid}: RNG counter {actual}, expected {draws}"
        result[key] = actual
    return result


def check_decoded(outputs, carry, frame, trace=None):
    max_error = 0
    assert outputs.keys() == carry.keys(), f"frame={frame}: decoded output rows changed"
    for rid, values in outputs.items():
        assert set(values) == set(WORLD_CARRY) | {"frames", "reward", "ending"}
        for name in WORLD_CARRY:
            torch.testing.assert_close(values[name][0], carry[rid][name][0], rtol=0, atol=0)
        rgb = values["frames"][0]
        assert rgb.dtype == torch.uint8 and rgb.shape == (1, 64, 64, 3)
        reward, ending = values["reward"][0], values["ending"][0]
        assert reward.dtype == torch.float32 and reward.shape == (1,)
        assert ending.dtype == torch.bool and ending.shape == (1,)
        torch.testing.assert_close(reward, carry[rid]["reward_class"][0].float() - 1, rtol=0, atol=0)
        torch.testing.assert_close(ending, carry[rid]["ending_class"][0].bool(), rtol=0, atol=0)
        if trace is not None:
            reference = (
                (torch.from_numpy(np.array(trace[f"step/{frame:04}/frame"][rid - 1 : rid], copy=True)) * 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
            )
            error = (rgb.cpu().to(torch.int16) - reference.to(torch.int16)).abs().max().item()
            assert error <= 1, f"frame={frame} request={rid}: RGB24 error {error}"
            max_error = max(max_error, error)
            assert values["reward"][0].item() == trace[f"step/{frame:04}/reward"][rid - 1]
            assert values["ending"][0].item() == trace[f"step/{frame:04}/done"][rid - 1]
    return max_error
