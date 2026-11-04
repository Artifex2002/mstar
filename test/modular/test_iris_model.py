"""IRIS's real CPU request contract; no weights, reference package or GPU."""

import json
import struct
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import DEFAULT_PARTITION
from mstar.graph.base import TensorPointerInfo
from mstar.model.iris.config import IMAGE_SHAPE, OUTPUT_MODALITIES, ROLLOUT_LOOP
from mstar.model.iris.iris_model import IrisModel
from mstar.model.multimodal import PromptPart
from mstar.model.registry import get_model_class, model_init_kwargs


@pytest.fixture
def model():
    return IrisModel()


def _prepare(model, *, image=None, actions=None, **kwargs):
    if image is None:
        image = torch.linspace(0, 1, 3 * 64 * 64).reshape(IMAGE_SHAPE)
    return model.process_prompt(
        None,
        ["image"],
        list(OUTPUT_MODALITIES),
        tensors={"image_inputs": [image]},
        actions=[0, 1, 3] if actions is None else actions,
        **kwargs,
    )


def _descriptor(tensor, uuid):
    return TensorPointerInfo(
        dims=list(tensor.shape),
        dtype=tensor.dtype,
        nbytes=tensor.numel() * tensor.element_size(),
        address=0,
        stride=list(tensor.stride()),
        uuid=uuid,
        source_session_id="test",
        source_entity="api_server",
    )


def _initial(model, prepared, **kwargs):
    signals = {
        name: [_descriptor(tensor, index) for tensor in values]
        for index, (name, values) in enumerate(prepared.new_input_tensors.items())
    }
    return model.get_initial_forward_pass_args(
        DEFAULT_PARTITION,
        ["image"],
        list(OUTPUT_MODALITIES),
        signals,
        model_kwargs={**prepared.metadata, **kwargs},
    )


def test_registered_constructor_needs_no_checkpoint(monkeypatch):
    def no_load(*args, **kwargs):
        pytest.fail("construction must not load weights")

    monkeypatch.setattr(torch, "load", no_load)
    cls = get_model_class("iris")
    model = cls(**model_init_kwargs("iris"), cache_dir="/does/not/exist")
    assert model.config.game == "Breakout"
    assert model.get_autocast_dtype() is None
    assert model.request_kwargs() == {"actions"}


@pytest.mark.parametrize("options", [{"game": "Pong"}, {"max_rollout_horizon": 0}, {"max_rollout_horizon": True}])
def test_bad_deployment_options(options):
    with pytest.raises(ValueError):
        IrisModel(**options)


def test_preparation_moves_bulk_actions_to_tensor_edges(model):
    result = _prepare(model, iris_horizon=999)
    assert result.metadata == {"iris_horizon": 3}
    assert result.new_input_tensors["image_inputs"] == []
    assert result.new_input_tensors["initial_image"][0].shape == IMAGE_SHAPE
    assert torch.equal(result.new_input_tensors["actions"][0], torch.tensor([0, 1, 3]))
    assert "actions" in model.PREPROCESS_ONLY_KWARGS
    assert "iris_horizon" in model.INTERNAL_KWARGS
    assert json.loads(json.dumps(result.metadata)) == result.metadata


def test_real_data_worker_handoff_removes_bulk_kwargs_and_overwrites_client_horizon(model, tmp_path):
    from PIL import Image

    from mstar.api_server.data_worker import PreprocessWorkerThread
    from mstar.api_server.request_types import PreprocessInput

    image_path = tmp_path / "initial.png"
    Image.new("RGB", (64, 64), (40, 80, 120)).save(image_path)
    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.model = model
    worker.device = "cpu"
    worker._keyed_labels = {}
    worker._client_edges = set()
    worker.request_model_kwargs = {}
    worker.request_output_state = {}
    worker.enable_prof = False
    sent = []
    worker.communicator = SimpleNamespace(send=lambda entity, msg: sent.append((entity, msg)))
    # Transport storage is the boundary: preparation and conductor handoff run unchanged.
    worker._store_persisted = lambda request_id, tensors: {
        name: [_descriptor(tensor, index) for tensor in values] for index, (name, values) in enumerate(tensors.items())
    }
    actions = torch.tensor([3, 1], dtype=torch.int64)
    request = PreprocessInput(
        request_id="iris-test",
        text=None,
        file_paths={"image": [str(image_path)]},
        input_modalities=["image"],
        output_modalities=list(OUTPUT_MODALITIES),
        model_kwargs={"actions": actions, "iris_horizon": 999, "seed": 7},
    )
    prepared = worker._prepare_input(request)
    assert prepared.model_kwargs["iris_horizon"] == 2
    assert prepared.tensors["initial_image"][0].shape == IMAGE_SHAPE
    assert prepared.tensors["image_inputs"] == []
    worker._finish_input(request, prepared, None)
    entity, message = sent[0]
    assert entity == "conductor"
    assert message.body.model_kwargs == {"iris_horizon": 2, "seed": 7}
    assert message.body.initial_signals["actions"][0].dims == [2]
    assert worker.request_model_kwargs["iris-test"]["actions"] is actions
    assert request.model_kwargs["iris_horizon"] == 999


def test_noncontiguous_cpu_inputs_are_made_contiguous_without_changing_values(model):
    image = torch.rand(3, 64, 64).transpose(1, 2)
    actions = torch.tensor([0, 9, 1, 9, 2, 9], dtype=torch.int64)[::2]
    result = _prepare(model, image=image, actions=actions)
    assert result.new_input_tensors["initial_image"][0].is_contiguous()
    assert result.new_input_tensors["actions"][0].is_contiguous()
    torch.testing.assert_close(result.new_input_tensors["initial_image"][0], image)
    assert torch.equal(result.new_input_tensors["actions"][0], actions)


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(64, 64, 3),
        torch.zeros(1, 3, 64, 64),
        torch.zeros(3, 64, 64, dtype=torch.uint8),
        torch.zeros(3, 64, 64, dtype=torch.float64),
        torch.full(IMAGE_SHAPE, float("nan")),
        torch.full(IMAGE_SHAPE, float("inf")),
        torch.full(IMAGE_SHAPE, -0.01),
        torch.full(IMAGE_SHAPE, 1.01),
        torch.empty(IMAGE_SHAPE, device="meta"),
    ],
)
def test_rejects_ambiguous_or_invalid_images(model, image):
    with pytest.raises(ValueError, match="image"):
        _prepare(model, image=image)


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [0] * 21,
        [True],
        [1.0],
        [-1],
        [4],
        [2**100],
        ["1"],
        torch.tensor([1.0]),
        torch.tensor([True]),
        torch.tensor([[0, 1]]),
        torch.tensor([4]),
        torch.empty(2, dtype=torch.int64, device="meta"),
    ],
)
def test_rejects_invalid_action_sequences(model, actions):
    with pytest.raises(ValueError, match="action"):
        _prepare(model, actions=actions)


@pytest.mark.parametrize(
    "input_modalities, output_modalities, prompt, tensors, parts",
    [
        ([], list(OUTPUT_MODALITIES), None, {"image_inputs": [torch.zeros(IMAGE_SHAPE)]}, None),
        (["image"], ["text"], None, {"image_inputs": [torch.zeros(IMAGE_SHAPE)]}, None),
        (["image"], list(OUTPUT_MODALITIES), "play", {"image_inputs": [torch.zeros(IMAGE_SHAPE)]}, None),
        (["image"], list(OUTPUT_MODALITIES), None, {}, None),
        (["image"], list(OUTPUT_MODALITIES), None, {"image_inputs": [torch.zeros(IMAGE_SHAPE)] * 2}, None),
        (
            ["image"],
            list(OUTPUT_MODALITIES),
            None,
            {"image_inputs": [torch.zeros(IMAGE_SHAPE)]},
            [PromptPart("text", "play")],
        ),
    ],
)
def test_rejects_unsupported_request_layout(model, input_modalities, output_modalities, prompt, tensors, parts):
    with pytest.raises(ValueError):
        model.process_prompt(prompt, input_modalities, output_modalities, tensors, parts, actions=[0])


def test_horizon_is_action_count_and_never_silently_truncated(model):
    result = _prepare(model, actions=[0] * 20)
    assert any(result.metadata.get(key) is not None for key in model.LENGTH_KWARGS)
    assert model.get_max_output_tokens(**result.metadata) == 20
    assert model.get_loop_caps(**result.metadata) == {ROLLOUT_LOOP: 20}
    assert _prepare(IrisModel(max_rollout_horizon=30), actions=[0] * 30).metadata["iris_horizon"] == 30
    with pytest.raises(ValueError, match="max_output_tokens"):
        _prepare(model, max_output_tokens=1)
    with pytest.raises(ValueError, match="iris_horizon"):
        model.get_max_output_tokens()


def test_descriptor_routing_then_completion_has_no_request_state_on_model(model):
    first = _initial(model, _prepare(model))
    second = _initial(model, _prepare(model, actions=[2]))
    assert [(edge.name, edge.next_node) for edge in first.inputs] == [
        ("initial_image", "image_encoder"),
        ("actions", "world_model"),
    ]
    assert first.full_metadata.graph_walk == "imagine"
    assert first.full_metadata.is_prefill is True
    assert first.step_metadata == {"iris_horizon": 3}
    assert second.step_metadata == {"iris_horizon": 1}
    assert first.full_metadata.kwargs == {"iris_horizon": 3}
    assert not first.request_done
    assert first.unpersist_tensors == [info for edge in first.inputs for info in edge.tensor_info]
    assert all(not edge.persist for edge in first.inputs)
    done = model.get_partition_forward_pass_args(DEFAULT_PARTITION, first.full_metadata, {})
    assert done.request_done
    assert done.inputs == done.unpersist_tensors == []
    assert first.full_metadata.kwargs == {"iris_horizon": 3}


def test_missing_or_mismatched_descriptors_fail_before_scheduling(model):
    with pytest.raises(ValueError, match="initial_image"):
        model.get_initial_forward_pass_args(
            DEFAULT_PARTITION,
            ["image"],
            list(OUTPUT_MODALITIES),
            {},
            {"iris_horizon": 1},
        )
    with pytest.raises(ValueError, match="actions"):
        _initial(model, _prepare(model), iris_horizon=2)
    with pytest.raises(ValueError, match="partition"):
        model.get_initial_forward_pass_args("other", ["image"], list(OUTPUT_MODALITIES), {})


def test_output_bytes_follow_native_frame_and_numeric_contracts(model):
    frame = torch.arange(64 * 64 * 3).remainder(256).to(torch.uint8).reshape(1, 64, 64, 3)
    assert model.postprocess(frame, "video_frame") == frame.numpy().tobytes()
    assert model.postprocess(torch.tensor([-1.0]), "scalar") == struct.pack("<f", -1)
    assert model.postprocess(torch.tensor([True]), "tensor") == b"\x01"
    assert model.get_output_frame_rate() == 15


@pytest.mark.parametrize(
    "output, modality",
    [
        (torch.zeros(64, 64, 3, dtype=torch.uint8), "video_frame"),
        (torch.zeros(1, 64, 64, 3), "video_frame"),
        (torch.tensor([0]), "scalar"),
        (torch.tensor([float("nan")]), "scalar"),
        (torch.tensor([0.5]), "scalar"),
        (torch.tensor([1]), "tensor"),
        (torch.tensor([True, False]), "tensor"),
        (torch.tensor([0.0]), "text"),
    ],
)
def test_output_schema_errors_are_explicit(model, output, modality):
    with pytest.raises(ValueError):
        model.postprocess(output, modality)


def test_unimplemented_execution_is_explicit_and_dummy_mode_still_works(model):
    with pytest.raises(NotImplementedError, match="resource"):
        model.get_node_resources()
    with pytest.raises(NotImplementedError, match="graph"):
        model.get_graph_walk_graphs()
    with pytest.raises(NotImplementedError, match="submodules"):
        model.get_submodule("world_model")
    dummy = IrisModel(skip_weight_loading=True)
    for node in ("image_encoder", "world_model", "image_decoder"):
        assert dummy.get_submodule(node) is None
    with pytest.raises(ValueError, match="Unknown IRIS node"):
        dummy.get_submodule("typo")
