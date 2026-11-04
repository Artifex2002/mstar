"""IRIS capture recipes, step keys and deployment opt-in, without a GPU."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.iris.components import Tokenizer
from mstar.model.iris.config import IMAGINE_WALK, ROLLOUT_LOOP, TOKEN_LOOP
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.iris_model import IrisModel
from mstar.model.iris.submodules import ImageSubmodule, WorldSubmodule
from mstar.model.submodule_base import InputMetadata


def test_eager_remains_default_and_capture_flag_is_strict():
    assert not IrisModel().config.capture_graphs
    assert WorldSubmodule(None, 20).get_accelerator_graph_configs("cpu") == []
    with pytest.raises(ValueError, match="capture_graphs"):
        IrisModel(capture_graphs="true")
    path = Path(__file__).resolve().parents[2] / "configs/iris_capture.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["model_kwargs"]["capture_graphs"] is True
    assert all(
        config["resources"][name]["backend"] == "torch_static"
        for name in ("world_attn", "encoder_attn", "decoder_attn")
    )


def test_world_captures_every_phase_and_keys_match_step_and_runtime_metadata():
    world = WorldSubmodule(None, 20, capture_graphs=True)
    configs = world.get_accelerator_graph_configs("cpu")
    by_key = {config.additional_key_info: config for config in configs}
    assert len(by_key) == len(configs) == 36
    for frame in range(20):
        for tick, call in enumerate(frame_calls(frame)):
            info = CurrentForwardPassInfo(
                "rid",
                IMAGINE_WALK,
                0,
                0,
                20,
                step_metadata={"iris_horizon": 20},
                dynamic_loop_iter_counts={ROLLOUT_LOOP: 99, TOKEN_LOOP: 99},
            )
            meta = InputMetadata(dynamic_loop_iter_counts={ROLLOUT_LOOP: frame, TOKEN_LOOP: tick})
            key = world.cg_key_info(IMAGINE_WALK, {1: info}, {1: meta})
            config = by_key[key]
            rows = config.get_node_inputs(2, 2 * call.span)
            step = world.declare_step(IMAGINE_WALK, [1, 2], rows)
            assert step.cg_key_info == key == world.call_key(call)
            assert all(segment.span == call.span and segment.label == call.label for segment in step.segments)
            assert config.capture_batch_sizes == [1, 2] and not config.compile
            assert all(dim == 0 for dim in config.input_seq_dims.values())
    assert world.split_batches_by_capture_key(IMAGINE_WALK)
    info = CurrentForwardPassInfo("rid", IMAGINE_WALK, 0, 0, 1, step_metadata={"iris_horizon": 1})
    assert world.cg_key_info(IMAGINE_WALK, {1: info}) in by_key
    meta = InputMetadata(dynamic_loop_iter_counts={ROLLOUT_LOOP: 1})
    assert world.cg_key_info(IMAGINE_WALK, {1: info}, {1: meta}) == ("overshoot",)


@pytest.mark.parametrize("decode,initial", [(False, False), (True, False), (True, True)])
def test_image_capture_templates_match_forward_inputs(decode, initial):
    module = ImageSubmodule(Tokenizer(), decode=decode, initial=initial, capture_graphs=True)
    configs = module.get_accelerator_graph_configs("cpu")
    assert len(configs) == 1 and not configs[0].compile
    rows = configs[0].get_node_inputs(2, 0)
    prepared = module.preprocess(IMAGINE_WALK, None, rows)
    assert all(value.shape[0] == 2 for value in prepared.values())
    assert set(prepared) == (
        {"initial_image"}
        if not decode
        else {"observation_tokens"}
        if initial
        else {"observation_tokens", "reward_class", "ending_class"}
    )
    step = module.declare_step(IMAGINE_WALK, [1, 2], rows)
    assert {segment.span for segment in step.steps[module.attention_key].segments} == {16, 64, 256}


def test_capture_command_requires_cuda_and_saves_failure_report(tmp_path, monkeypatch):
    from scripts.iris.qualify_capture import main

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "capture"
    assert main(["--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "cuda_preflight"


@pytest.mark.parametrize(
    "requested,expected,queries_current",
    [
        ("cuda", "cuda:2", True),
        (torch.device("cuda"), "cuda:2", True),
        ("cuda:0", "cuda:0", False),
        ("cuda:3", "cuda:3", False),
        ("cpu", "cpu", False),
    ],
)
def test_capture_device_is_explicit_and_preserves_requested_ordinals(monkeypatch, requested, expected, queries_current):
    from scripts.iris.qualify_cuda import indexed_cuda_device

    queried = []

    def current():
        queried.append(True)
        return 2

    monkeypatch.setattr(torch.cuda, "current_device", current)
    actual = indexed_cuda_device(requested)
    assert actual == torch.device(expected)
    assert bool(queried) == queries_current
    if actual.type == "cuda":
        from torch.accelerator._utils import _get_device_index

        # Exercise the exact CPU-side validation used by set_device_index;
        # no GPU device switch or accelerator initialization is needed.
        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
        with pytest.raises(ValueError, match="specified index"):
            _get_device_index(torch.device("cuda"), optional=False)
        assert _get_device_index(actual, optional=False) == actual.index


def test_real_engine_loads_capture_model_and_optin_resources_on_cpu(monkeypatch):
    import triton.language as tl

    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources import apply_yaml_overrides
    from mstar.engine.resources.attn.torch_static import TorchStaticPagedAttention, TorchStaticRaggedAttention
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.iris.components import WorldNetwork

    model = IrisModel(capture_graphs=True)
    tokenizer = Tokenizer()
    modules = {
        "image_encoder": ImageSubmodule(tokenizer, capture_graphs=True),
        "image_decoder": ImageSubmodule(tokenizer, decode=True, capture_graphs=True),
        "initial_image_decoder": ImageSubmodule(tokenizer, decode=True, initial=True, capture_graphs=True),
        "world_model": WorldSubmodule(WorldNetwork(), 20, capture_graphs=True),
    }
    specs = model.get_node_resources()
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/iris_capture.yaml").read_text())
    apply_yaml_overrides(specs, config)
    engine = Engine(None, autocast_dtype=None)
    try:
        engine.load_model(
            modules,
            specs,
            WorkerParallelGroups(1, 0),
            torch.device("cpu"),
            TransferEngineInfo("iris-test", "local", LocalTransferEngine("localhost")),
            kv_cache_type=torch.float32,
            model=model,
        )
        assert isinstance(engine._resources["world_attn"], TorchStaticPagedAttention)
        assert isinstance(engine._resources["decoder_attn"], TorchStaticRaggedAttention)
        assert all(module.node_resources for module in modules.values())
    finally:
        engine.shutdown()


@pytest.fixture
def capture_output_tools(monkeypatch):
    import triton.language as tl

    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import BatchedModelOutput, NodeInputs
    from scripts.iris.captured_engine import copy_step_outputs

    return Engine, BatchedModelOutput, NodeInputs, copy_step_outputs


@pytest.mark.parametrize("captured", [False, True])
def test_real_engine_empty_output_collection_matches_direct_initial_decode(capture_output_tools, captured):
    Engine, Output, NodeInputs, copy_outputs = capture_output_tools
    from mstar.engine.resources.step import BucketKey, SlotLease
    from mstar.model.iris.config import INITIAL_DECODER

    module = ImageSubmodule(Tokenizer(), decode=True, initial=True)
    engine = Engine.__new__(Engine)
    management = SimpleNamespace(
        submodule=module,
        accelerator_graph_runner=SimpleNamespace(slot_for=lambda lease: SimpleNamespace(dummy_rids=[-1, -2])),
    )
    lease = SlotLease(0, BucketKey(IMAGINE_WALK, 2, 0)) if captured else None
    raw = Output(per_rid_outputs={rid: {} for rid in ([-1, -2] if captured else [1, 2])})
    collected = engine._collect_outputs(
        management,
        lease,
        raw,
        [NodeInputs(), NodeInputs()],
        {},
        [1, 2],
        (1, 2),
        IMAGINE_WALK,
    )
    assert collected.per_rid_outputs == {}  # engine intentionally drops empty entries
    direct = {1: {}, 2: {}}
    assert copy_outputs(INITIAL_DECODER, [1, 2], collected) == direct


@pytest.mark.parametrize("bad_kind", ["unknown_request", "per_request_tensor", "packed_tensor", "row_tensor"])
def test_initial_decode_adapter_refuses_unexpected_data(capture_output_tools, bad_kind):
    _, Output, _, copy_outputs = capture_output_tools
    from mstar.model.iris.config import INITIAL_DECODER

    output = Output()
    if bad_kind == "unknown_request":
        output.per_rid_outputs = {99: {}}
    elif bad_kind == "per_request_tensor":
        output.per_rid_outputs = {1: {"frames": [torch.zeros(1)]}}
    elif bad_kind == "packed_tensor":
        output.packed_outputs = {"__frames__": torch.zeros(1)}
    else:
        output.row_outputs = {"frames": torch.zeros(1)}
    with pytest.raises(AssertionError, match="unexpectedly emitted"):
        copy_outputs(INITIAL_DECODER, [1, 2], output)


def test_output_producing_nodes_still_require_every_row_and_copy_tensors(capture_output_tools):
    _, Output, _, copy_outputs = capture_output_tools
    from mstar.model.iris.config import WORLD_MODEL

    with pytest.raises(AssertionError, match="expected output requests"):
        copy_outputs(WORLD_MODEL, [1, 2], Output(per_rid_outputs={1: {}}))
    values = torch.arange(16)
    copied = copy_outputs(WORLD_MODEL, [1], Output(per_rid_outputs={1: {"observation_tokens": [values]}}))
    assert torch.equal(copied[1]["observation_tokens"][0], values)
    values.zero_()
    assert copied[1]["observation_tokens"][0].tolist() == list(range(16))
