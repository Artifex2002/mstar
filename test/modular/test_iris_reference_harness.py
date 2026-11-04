"""CPU checks of reference trace validation; no checkpoint or IRIS imports."""

import numpy as np
import pytest

from scripts.iris.reference_harness import compare_traces, validate_inputs, verify_checkpoint


def valid_inputs():
    return np.zeros((2, 3, 64, 64), np.float32), np.zeros((2, 3), np.int64)


def test_valid_inputs_and_reject_wrong_layout_or_action_shape():
    observations, actions = valid_inputs()
    validate_inputs(observations, actions, 2, 3, 4)
    with pytest.raises(ValueError, match="observations must be float32"):
        validate_inputs(observations.transpose(0, 2, 3, 1), actions, 2, 3, 4)
    with pytest.raises(ValueError, match="actions must be int64"):
        validate_inputs(observations, actions.T, 2, 3, 4)


def test_reject_wrong_dtype_nonfinite_and_out_of_range_inputs():
    observations, actions = valid_inputs()
    with pytest.raises(ValueError, match="observations must be float32"):
        validate_inputs(observations.astype(np.float64), actions, 2, 3, 4)
    observations[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and in"):
        validate_inputs(observations, actions, 2, 3, 4)
    observations[0, 0, 0, 0] = 0
    actions[0, 0] = 4
    with pytest.raises(ValueError, match="actions must be in"):
        validate_inputs(observations, actions, 2, 3, 4)


def test_trace_comparison_detects_missing_data_dtype_drift_and_nonfinite():
    expected = {"tokens": np.array([1, 2], np.int64)}
    with pytest.raises(ValueError, match="names differ"):
        compare_traces(expected, {})
    with pytest.raises(ValueError, match="shape/dtype"):
        compare_traces(expected, {"tokens": np.array([1, 2], np.int32)})
    with pytest.raises(ValueError, match="non-finite"):
        compare_traces({"logits": np.array([np.nan])}, {"logits": np.array([np.nan])})


def test_trace_comparison_detects_token_and_float_drift():
    with pytest.raises(AssertionError):
        compare_traces({"tokens": np.array([1])}, {"tokens": np.array([2])})
    with pytest.raises(AssertionError):
        compare_traces({"logits": np.zeros(2)}, {"logits": np.ones(2)})
    compare_traces({"logits": np.ones(2)}, {"logits": np.ones(2)})


def test_wrong_checkpoint_is_rejected_before_loading(tmp_path):
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"not the published checkpoint")
    with pytest.raises(ValueError, match="pinned Breakout checkpoint"):
        verify_checkpoint(checkpoint)
