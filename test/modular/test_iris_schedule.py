"""IRIS call ordering, sampling alignment, and context boundaries."""

import pytest

from mstar.model.iris.resources import context_epochs
from mstar.model.iris.schedule import WorldCall, frame_calls


def test_first_frame_primes_then_consumes_action_and_all_codes():
    calls = frame_calls(0)

    assert len(calls) == 18
    assert calls[0] == WorldCall("prime", "epoch_0", 0, None, None)
    assert calls[0].span == 16
    assert calls[1] == WorldCall("action", "epoch_0", 16, None, 0)
    assert [
        call.input_index for call in calls[2:]
    ] == list(range(16))
    assert [
        call.sample_index
        for call in calls
        if call.sample_index is not None
    ] == list(range(16))
    assert calls[-1] == WorldCall(
        "observation", "epoch_0", 32, 15, None
    )


def test_regular_frame_and_context_refresh():
    assert len(frame_calls(1)) == 17
    assert frame_calls(1)[0] == WorldCall(
        "action", "epoch_0", 33, None, 0
    )
    assert frame_calls(18)[-1].position == 338
    assert frame_calls(19)[0] == WorldCall(
        "prime", "epoch_1", 0, None, None
    )
    assert frame_calls(19)[1] == WorldCall(
        "action", "epoch_1", 16, None, 0
    )


@pytest.mark.parametrize("horizon", [1, 19, 20, 38, 39, 60])
def test_schedule_matches_declared_context_footprints(horizon):
    cursors = {}

    for frame in range(horizon):
        for call in frame_calls(frame):
            assert call.position == cursors.get(call.label, 0)
            cursors[call.label] = call.position + call.span
            assert cursors[call.label] <= 340

    assert cursors == {
        epoch.label: epoch.tokens
        for epoch in context_epochs(horizon)
    }


@pytest.mark.parametrize("invalid", [True, -1, 1.5, "0", None])
def test_invalid_frame_index(invalid):
    with pytest.raises(ValueError, match="frame_index"):
        frame_calls(invalid)
