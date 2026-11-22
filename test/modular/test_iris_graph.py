"""IRIS graph declarations and their compilation into the Rust runtime."""

import pytest

from mstar.graph.runtime.rust import worker_graph_args
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.model.iris.graph import WORLD_CARRY, imagination_graph
from mstar.model.iris.iris_model import IRISModel
from mstar.model.iris.resources import iris_state


@pytest.mark.parametrize("max_frames", [1, 20, 40])
def test_graph_inputs_loops_and_resource_owners(max_frames):
    graph = imagination_graph(max_frames)
    nodes = graph.get_nodes()
    loops = graph.get_loops()

    assert set(nodes) == {"image_encoder", "world_model", "image_decoder"}
    assert graph.get_inputs_outputs().ext_inputs == {
        ("initial_image", "image_encoder"),
        ("actions", "world_model"),
    }
    assert loops["rollout_loop"].max_iters == max_frames
    assert loops["token_loop"].max_iters == 18
    assert {
        (edge.name, edge.next_node)
        for edge in loops["token_loop"].outputs
    } == {(name, "image_decoder") for name in WORLD_CARRY}
    for declaration in iris_state(max_frames):
        assert declaration.nodes <= set(nodes)

    worker_graph = WorkerGraph(
        section=graph,
        graph_walks={"imagine"},
        ranks=[0],
        worker_graph_id=0,
    )
    compiled = worker_graph_args(worker_graph)
    assert {loop["name"]: loop["parent"] for loop in compiled["loops"]} == {
        "rollout_loop": None,
        "token_loop": "rollout_loop",
    }


def test_client_outputs_count_only_frames():
    decoder = imagination_graph(20).get_nodes()["image_decoder"]
    client_edges = [edge for edge in decoder.outputs if edge.next_node == EMIT_TO_CLIENT]
    assert {
        edge.name: (edge.output_modality, edge.conductor_new_token)
        for edge in client_edges
    } == {
        "frames": ("video_frame", True),
        "reward": ("scalar", False),
        "ending": ("tensor", False),
    }


@pytest.mark.parametrize("invalid", [True, 0, -1, 2.5, "20"])
def test_invalid_graph_limits(invalid):
    with pytest.raises(ValueError, match="max_frames"):
        imagination_graph(invalid)


def test_model_exposes_the_walk_and_request_cap():
    class GraphOnlyIRIS(IRISModel):
        def get_submodule(self, *args, **kwargs):
            raise AssertionError("Graph declaration must not load compute.")

    model = GraphOnlyIRIS(max_rollout_horizon=20)
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"imagine"}
    assert walks["imagine"].get_loops()["rollout_loop"].max_iters == 20
    assert model.get_loop_caps(iris_horizon=3) == {"rollout_loop": 3}
    assert set(model.nodes) == {"image_encoder", "world_model", "image_decoder"}
