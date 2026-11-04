"""CPU serving-boundary regression: worker snapshots -> real IRIS hooks.

No network weights, CUDA runtime, HTTP server or kernel execution is required.
"""

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.iris.config import IMAGINE_WALK, ROLLOUT_LOOP, TOKEN_LOOP, WORLD_MODEL
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.submodules import WorldSubmodule


@pytest.fixture(params=["normal", "columnar"])
def worker_snapshot(monkeypatch, request):
    import triton.language as tl

    # Complete only an import-time annotation in the local macOS Triton stub.
    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)

    from mstar.worker.executing_batch import input_metadata_and_fwd_info, make_executing_batch
    from mstar.worker.node_manager_utils import RequestStateManager
    from mstar.worker.spec_rows import loop_counter_snapshots
    from mstar.worker.stream_buffers import StreamBufferManager

    def build(counters, increment=None):
        info = CurrentForwardPassInfo(
            "iris-serving",
            IMAGINE_WALK,
            0,
            7,
            20,
            rid_handle=0,
            step_metadata={"iris_horizon": 20},
            dynamic_loop_iter_counts=dict(counters),
        )
        state = RequestStateManager()
        state.add_request(0, info)
        streams = StreamBufferManager(state, None, [])
        inputs = {
            "observation_tokens": [torch.arange(16)],
            "reward_class": [torch.ones(1, dtype=torch.long)],
            "ending_class": [torch.zeros(1, dtype=torch.long)],
            "actions": [torch.arange(20) % 4],
        }
        if request.param == "normal":
            metadata, infos = input_metadata_and_fwd_info(
                state,
                streams,
                WORLD_MODEL,
                "default",
                {0: inputs},
                increment_loop_rids=[0] if increment else None,
                increment_loop_name=increment,
            )
        else:
            # The optimized speculative assembler uses this real columnar helper.
            metadata = {0: loop_counter_snapshots([info], increment, n_next_iter=1)[0]}
            infos = {0: info}
        batch = make_executing_batch(
            streams,
            WORLD_MODEL,
            IMAGINE_WALK,
            [0],
            {0: inputs},
            infos,
            per_request_input_metadata=metadata,
        )
        return info, batch.per_request_info_wrapped[0], inputs

    return build


@pytest.mark.parametrize(
    "counters,increment,frame,tick",
    [
        ({}, None, 0, 0),  # the first request's wire metadata has no loop counters
        ({}, TOKEN_LOOP, 0, 1),  # next token snapshot can still omit the outer loop
        ({}, ROLLOUT_LOOP, 1, 0),
        ({ROLLOUT_LOOP: 18}, ROLLOUT_LOOP, 19, 0),  # context refresh
        ({TOKEN_LOOP: 17}, None, 0, 17),  # stop check with a sparse outer counter
        ({ROLLOUT_LOOP: 1, TOKEN_LOOP: 16}, None, 1, 16),
    ],
)
def test_worker_loop_snapshots_prepare_and_stop_the_correct_iris_call(
    worker_snapshot, counters, increment, frame, tick
):
    shared, snapshot, inputs = worker_snapshot(counters, increment)
    # Routing N updates shared state while N+1 retains its own snapshot.
    shared.dynamic_loop_iter_counts.update({ROLLOUT_LOOP: 99, TOKEN_LOOP: 99})
    world = WorldSubmodule(None, 20)
    prepared = world.prepare_inputs(IMAGINE_WALK, snapshot, inputs)
    call = frame_calls(frame)[tick]
    assert prepared.resource_step_info == call
    if call.kind == "prime":
        assert torch.equal(prepared.tensor_inputs["tokens"], inputs["observation_tokens"][0])
    elif call.kind == "action":
        assert prepared.tensor_inputs["tokens"].tolist() == [frame % 4]
    else:
        assert prepared.tensor_inputs["tokens"].tolist() == [call.input_index]
    expected = {TOKEN_LOOP} if tick + 1 == len(frame_calls(frame)) else set()
    assert world.check_stop(0, snapshot, {}) == expected


@pytest.mark.parametrize("counters", [{ROLLOUT_LOOP: 20}, {TOKEN_LOOP: 18}])
def test_sparse_speculative_overshoot_skips_neural_work(worker_snapshot, counters):
    _, snapshot, inputs = worker_snapshot(counters)
    assert WorldSubmodule(None, 20).prepare_inputs(IMAGINE_WALK, snapshot, inputs) is None
