"""CPU request validation without constructing IRIS compute or resources."""

import pytest
import torch

from mstar.graph.base import TensorPointerInfo
from mstar.model.iris.iris_model import IRISModel
from mstar.model.multimodal import PromptPart


class RequestOnlyIRIS(IRISModel):
    """Expose the real preprocessing methods while execution remains abstract."""

    def get_graph_walk_graphs(self):
        raise NotImplementedError

    def get_submodule(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def model():
    return RequestOnlyIRIS(max_rollout_horizon=3)


def prepare(model, **overrides):
    request = {
        "prompt": None,
        "input_modalities": ["image"],
        "output_modalities": ["video_frame", "scalar", "tensor"],
        "tensors": {"image_inputs": [torch.zeros(3, 64, 64)]},
        "actions": [0, 1, 3],
    }
    request.update(overrides)
    return model.process_prompt(**request)


def test_prepared_request_and_derived_horizon(model):
    image = torch.rand(3, 64, 64).transpose(1, 2)
    result = prepare(
        model,
        tensors={"image_inputs": [image]},
        iris_horizon=999,
        prompt_parts=[PromptPart(modality="image", index=0)],
    )
    assert result.metadata == {"iris_horizon": 3}
    assert result.new_input_tensors["image_inputs"] == []
    prepared_image = result.new_input_tensors["initial_image"][0]
    assert prepared_image.is_contiguous()
    assert torch.equal(prepared_image, image)
    actions = result.new_input_tensors["actions"][0]
    assert actions.dtype == torch.int64
    assert actions.device.type == "cpu"
    assert actions.tolist() == [0, 1, 3]


@pytest.mark.parametrize(
    "actions",
    [None, [], [0] * 4, [True], [False], [-1], [4], [1.0], ["1"], (0, 1)],
)
def test_invalid_actions(model, actions):
    with pytest.raises(ValueError):
        prepare(model, actions=actions)


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(3, 32, 32),
        torch.zeros(3, 64, 64, dtype=torch.uint8),
        torch.full((3, 64, 64), float("nan")),
        torch.full((3, 64, 64), float("inf")),
        torch.full((3, 64, 64), -0.1),
        torch.full((3, 64, 64), 1.1),
        "image",
    ],
)
def test_invalid_images(model, image):
    with pytest.raises(ValueError):
        prepare(model, tensors={"image_inputs": [image]})


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": "hello"},
        {"input_modalities": ["text"]},
        {"output_modalities": ["video_frame"]},
        {"prompt_parts": []},
        {"max_output_tokens": 2},
        {"seed": True},
        {"seed": 2**63},
        {"tensors": {}},
        {"tensors": {"image_inputs": [torch.zeros(3, 64, 64)] * 2}},
        {"tensors": {
            "image_inputs": [torch.zeros(3, 64, 64)],
            "audio_inputs": [torch.zeros(1)],
        }},
    ],
)
def test_invalid_request_layout_and_options(model, overrides):
    with pytest.raises(ValueError):
        prepare(model, **overrides)


def test_request_contract_flags(model):
    assert model.request_kwargs() == frozenset({"actions"})
    assert model.PREPROCESS_ONLY_KWARGS == ("actions",)
    assert model.INTERNAL_KWARGS == ("iris_horizon",)
    assert model.LENGTH_KWARGS == ("iris_horizon",)


def test_shorter_request(model):
    assert prepare(model, actions=[0]).metadata == {"iris_horizon": 1}


def descriptor(tensor, identity):
    return TensorPointerInfo(
        dims=list(tensor.shape),
        dtype=tensor.dtype,
        nbytes=tensor.numel() * tensor.element_size(),
        address=0,
        stride=list(tensor.stride()),
        uuid=identity,
        source_session_id="test",
        source_entity="api_server",
    )


def initial_args(model, horizon=3):
    signals = {
        "initial_image": [descriptor(torch.zeros(3, 64, 64), 1)],
        "actions": [descriptor(torch.zeros(3, dtype=torch.int64), 2)],
    }
    return model.get_initial_forward_pass_args(
        partition_name="default",
        input_modalities=["image"],
        output_modalities=["video_frame", "scalar", "tensor"],
        input_signals=signals,
        model_kwargs={"iris_horizon": horizon},
    )


def test_initial_routing_and_completion(model):
    args = initial_args(model)
    assert args.full_metadata.graph_walk == "imagine"
    assert not args.request_done
    assert args.step_metadata == {"iris_horizon": 3}
    assert [(edge.name, edge.next_node) for edge in args.inputs] == [
        ("initial_image", "image_encoder"),
        ("actions", "world_model"),
    ]
    assert len(args.unpersist_tensors) == 2

    done = model.get_partition_forward_pass_args(
        "default", args.full_metadata, {}
    )
    assert done.request_done
    assert done.inputs == []
    assert model.get_max_output_tokens(iris_horizon=3) == 3
    assert model.get_loop_caps(iris_horizon=3) == {"rollout_loop": 3}


@pytest.mark.parametrize("horizon", [None, True, 0, 4])
def test_invalid_conductor_horizon(model, horizon):
    with pytest.raises(ValueError):
        initial_args(model, horizon)


def test_action_descriptor_must_match_horizon(model):
    with pytest.raises(ValueError):
        initial_args(model, horizon=2)


def test_resource_ownership_and_dependencies(model):
    specs = model.get_node_resources()
    by_key = {spec.resource_key: spec for spec in specs}

    assert len(specs) == len(by_key) == 8
    for key in (
        "world_kv",
        "world_attn",
        "world_pos",
        "observation_sampler",
        "reward_sampler",
        "ending_sampler",
    ):
        assert by_key[key].nodes == {"world_model"}

    assert by_key["encoder_attn"].nodes == {"image_encoder"}
    assert by_key["decoder_attn"].nodes == {"image_decoder"}
    assert by_key["world_attn"].depends_on() == {"world_kv"}
    assert by_key["world_pos"].depends_on() == {"world_kv"}


def test_resource_geometry(model):
    by_key = {spec.resource_key: spec for spec in model.get_node_resources()}
    kv = by_key["world_kv"].config
    assert (
        kv.num_layers,
        kv.num_kv_heads,
        kv.num_qo_heads,
        kv.head_dim,
        kv.max_seq_len,
    ) == (10, 4, 4, 64, 340)
    assert not kv.prefix_cache

    assert by_key["observation_sampler"].vocab_size == 512
    assert by_key["reward_sampler"].vocab_size == 3
    assert by_key["ending_sampler"].vocab_size == 2
    for key in ("encoder_attn", "decoder_attn"):
        config = by_key[key].config
        assert config.num_qo_heads == config.num_kv_heads == 1
        assert config.head_dim == 64
        assert config.max_tokens_per_request == 256
        assert config.dtype == torch.float32
