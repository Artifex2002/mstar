"""Real CPU KV and position planning through IRIS's state declarations."""

from pathlib import Path

import torch
import yaml

import mstar
from mstar.engine.resources import (
    KVReqConfig,
    Resource,
    StepContext,
    StepRunner,
    apply_yaml_overrides,
)
from mstar.engine.resources.kv import manager as kv_module
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep
from mstar.engine.state.adapters import (
    StateSetup,
    _resolve,
    build_handle,
    build_specs,
    kv_layouts,
    state_steps,
)
from mstar.model.iris.resources import (
    context_epochs,
    iris_state,
    request_kv_layout,
)


class _NoTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def remove_request(self, rid):
        pass

    def cleanup(self):
        pass


class _PlanningOnlyAttention(Resource):
    @classmethod
    def build(cls, spec, info):
        return cls()

    def depends_on(self):
        return {"world_kv"}

    def plan(self, step, ctx):
        assert step.causal


def test_context_refresh_plans_and_teardown(monkeypatch):
    monkeypatch.setattr(kv_module, "KVTransferManager", _NoTransfer)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    array = next(
        item for item in iris_state(20)
        if item.name == "world_kv"
    )
    declarations = [array]
    specs = build_specs(declarations)

    root = Path(mstar.__file__).resolve().parents[1]
    deployment = yaml.safe_load(
        (root / "configs/iris.yaml").read_text()
    )
    apply_yaml_overrides(specs, deployment)

    by_key = {spec.resource_key: spec for spec in specs}
    config = by_key["world_kv"].config

    layouts = kv_layouts(
        declarations,
        request_kv_layout(20),
        {},
    )["world_kv"]

    request = KVReqConfig(
        needed_labels=list(layouts),
        prefix_cache=False,
    )
    request.apply_conductor_config(
        kv_layout={
            label: layout.row()
            for label, layout in layouts.items()
        },
        loop_caps={},
    )
    request.validate(by_key["world_kv"])

    kv = kv_module.KVManager(
        cfg=config,
        name="world_kv",
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    positions = RopeManager(
        config=by_key["world_pos"].config,
        device=torch.device("cpu"),
        max_seq_len=config.max_seq_len,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )

    resources = {
        "world_kv": kv,
        "world_pos": positions,
        "world_attn": _PlanningOnlyAttention(),
    }
    handle = build_handle(
        array,
        resources,
        _resolve(declarations, StateSetup()),
    )
    runner = StepRunner(resources)

    initial_free = kv._arena.num_free
    frame_lengths = []

    try:
        runner.ingest_request(1, {"world_kv": request})

        for epoch in context_epochs(20):
            cursor = 0
            spans = [16] + [1] * (epoch.frames * 17)

            for span in spans:
                declared = SubmoduleStep(
                    segments=[Segment(1, epoch.label, span)],
                    state={"world_kv": ArrayStep()},
                )
                step = state_steps(
                    declared,
                    {"world_kv": handle},
                    graph_walk="imagine",
                )
                step.set_ctx(
                    StepContext(
                        request_ids=(1,),
                        graph_walk="imagine",
                        slot=0,
                        capture=False,
                    )
                )

                assert runner.admit(step).outcome.ok
                plans = runner.plan(step)

                assert plans["world_pos"][epoch.label].tolist() == list(
                    range(cursor, cursor + span)
                )
                assert set(plans["world_kv"]) == {epoch.label}
                assert (
                    plans["world_kv"][epoch.label].views[0].length
                    == cursor + span
                )

                # No forward runs: only bookkeeping is committed,
                # and no unwritten KV is read.
                runner.commit(step)

                cursor += span
                if cursor > 16 and (cursor - 16) % 17 == 0:
                    frame_lengths.append(cursor)

        assert frame_lengths == list(range(33, 340, 17)) + [33]
        assert positions.position(1, "epoch_0") == 339
        assert positions.position(1, "epoch_1") == 33
        assert initial_free - kv._arena.num_free == 25
    finally:
        runner.remove_request(1)
        kv.cleanup()

    assert kv._arena.num_free == initial_free
    assert 1 not in kv._streams
    assert 1 not in positions._counters
