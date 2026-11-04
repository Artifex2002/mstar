"""IRIS graph construction and real Rust routing with synthetic CPU tensors."""

from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path

import pytest
import torch
import yaml

from mstar.model.iris.config import (
    IMAGE_DECODER,
    IMAGE_ENCODER,
    IMAGINE_WALK,
    INITIAL_DECODER,
    ROLLOUT_LOOP,
    TOKEN_LOOP,
    WORLD_CARRY,
    WORLD_MODEL,
)
from mstar.model.iris.graph import frame_calls
from mstar.model.iris.iris_model import IrisModel


def test_graph_contract_and_counted_outputs():
    from mstar.validation.resources import check_resources

    model = IrisModel()
    graph = model.get_graph_walk_graphs()[IMAGINE_WALK]
    assert set(model.nodes) == {IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER}
    assert set(graph.get_loops()) == {ROLLOUT_LOOP, TOKEN_LOOP}
    assert graph.get_loops()[ROLLOUT_LOOP].max_iters == 20
    assert graph.get_loops()[TOKEN_LOOP].max_iters == 18
    assert graph.get_inputs_outputs().ext_inputs == {
        ("initial_image", IMAGE_ENCODER),
        ("actions", WORLD_MODEL),
    }
    body = graph.get_nodes()
    assert set(body[WORLD_MODEL].input_names) == {"actions", *WORLD_CARRY}
    assert {edge.name for edge in graph.get_loops()[TOKEN_LOOP].outputs} == set(WORLD_CARRY)
    emits = [edge for edge in body[IMAGE_DECODER].outputs if edge.output_modality]
    assert {(edge.name, edge.output_modality) for edge in emits} == {
        ("frames", "video_frame"),
        ("reward", "scalar"),
        ("ending", "tensor"),
    }
    assert [edge.name for edge in emits if edge.conductor_new_token] == ["frames"]
    assert model.get_partitions()[0].graph_walks == {IMAGINE_WALK}
    deployment = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/iris.yaml").read_text())
    assert check_resources(model, model.get_node_resources(), deployment) == []


@pytest.mark.parametrize(
    "frame, label, count, action_position",
    [
        (0, "epoch_0", 18, 16),
        (1, "epoch_0", 17, 33),
        (18, "epoch_0", 17, 322),
        (19, "epoch_1", 18, 16),
        (38, "epoch_2", 18, 16),
    ],
)
def test_call_schedule_consumes_the_last_generated_token(frame, label, count, action_position):
    calls = frame_calls(frame)
    assert len(calls) == count
    assert {call.label for call in calls} == {label}
    action = next(call for call in calls if call.kind == "action")
    assert action.position == action_position and action.sample_index == 0
    assert [call.input_index for call in calls if call.kind == "observation"] == list(range(16))
    assert [call.sample_index for call in calls if call.sample_index is not None] == list(range(16))
    assert calls[-1].input_index == 15
    assert calls[-1].sample_index is None
    assert calls[-1].position == action_position + 16


class _RoutingScript:
    """Real runtime/store; synthetic node outputs test routing, not ML numerics."""

    def __init__(self, horizons):
        pytest.importorskip("mstar_rust", reason="build the Rust extension to check native IRIS routing")
        from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
        from mstar.distributed.base import ShardingConfig
        from mstar.graph.runtime.rust import RustGraphRuntime

        self.model = IrisModel(max_rollout_horizon=max(horizons))
        worker_graphs = self.model.get_worker_graphs(str(Path(__file__).resolve().parents[2] / "configs/iris.yaml"))
        assert len(worker_graphs) == 1
        worker_graph = worker_graphs[0]
        worker_graph.worker_graph_id = 0
        graph = worker_graph.section
        self.book = RustTensorBookkeeping()
        self.store = TensorStore(bookkeeping=self.book)
        self.runtime = RustGraphRuntime(
            my_worker_id="worker_0",
            my_worker_graphs=[worker_graph],
            all_wg_ids_to_graph_walks={0: {IMAGINE_WALK}},
            all_wg_ids_to_dyn_loops={0: set(graph.get_loops())},
            all_wg_ids_to_nodes={0: set(graph.get_nodes())},
            node_to_partition=dict.fromkeys(graph.get_nodes(), "default"),
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
            bookkeeping=self.book,
        )
        self.horizons = {}
        self.counts = Counter()
        self.phases = defaultdict(list)
        self.emits = Counter()
        self.created = []
        self.action_uuids = {}
        self.uuid = 100
        for index, horizon in enumerate(horizons):
            self.add(index, horizon)

    def put(self, rid, tensor):
        from mstar.graph.base import TensorPointerInfo

        self.uuid += 1
        uuid = self.uuid
        self.created.append(uuid)
        info = TensorPointerInfo(
            dims=list(tensor.shape),
            dtype=tensor.dtype,
            nbytes=tensor.numel() * tensor.element_size(),
            address=0,
            stride=list(tensor.stride()),
            uuid=uuid,
            source_session_id="test",
            source_entity="worker_0",
        )
        self.store.put_tensor(rid, uuid, tensor, info)
        self.store.increment_ref(uuid)
        return uuid

    def add(self, index, horizon):
        from mstar.graph.runtime.base import ColumnarEdgeSpecs
        from mstar.utils.containers import ParallelList

        rid = self.runtime.add_request(
            f"iris-{index}",
            "default",
            IMAGINE_WALK,
            [0],
            ParallelList([0], [["worker_0"]]),
            self.model.get_loop_caps(iris_horizon=horizon),
        )
        self.horizons[rid] = horizon
        edges = ColumnarEdgeSpecs.empty()
        edges.add(rid, "initial_image", [self.put(rid, torch.zeros(3, 64, 64))], False, IMAGE_ENCODER)
        self.action_uuids[rid] = self.put(rid, torch.arange(horizon) % 4)
        edges.add(rid, "actions", [self.action_uuids[rid]], False, WORLD_MODEL)
        assert self.runtime.ingest_inputs_batch(edges) == []

    @staticmethod
    def codes(rid, frame):
        return (17 * rid + 31 * frame + torch.arange(16)) % 512

    def finish_teardown(self, freed):
        if freed.uuids:
            self.store.mark_forgotten(list(freed.uuids))

    def step(self, node, rid):
        from mstar.graph.runtime.base import RouteInput
        from mstar.utils.containers import ParallelList

        popped = self.runtime.pop_rids(node, IMAGINE_WALK, [rid], check_ready=True)
        assert popped is not None
        inputs = {name: self.store.get_tensor(uuids[0]) for _, name, uuids, _ in popped.input_edges.edge_tuples()}
        indices = self.runtime.get_dynamic_loop_iters([rid], "default").values[0]
        self.counts[rid, node] += 1
        if node == IMAGE_ENCODER:
            outputs = dict(zip(WORLD_CARRY, (self.codes(rid, -1), torch.tensor([1]), torch.tensor([0])), strict=True))
        elif node == INITIAL_DECODER:
            assert torch.equal(inputs["observation_tokens"], self.codes(rid, -1))
            outputs = {}
        elif node == WORLD_MODEL:
            frame, tick = indices[ROLLOUT_LOOP], indices[TOKEN_LOOP]
            calls = frame_calls(frame)
            assert tick < len(calls), "a surplus neural call was scheduled"
            call = calls[tick]
            self.phases[rid].append((frame, call.kind, call.label, call.position))
            outputs = {name: inputs[name].clone() for name in WORLD_CARRY}
            if call.kind == "prime":
                assert torch.equal(inputs["observation_tokens"], self.codes(rid, frame - 1))
            elif call.kind == "action":
                assert inputs["actions"][frame] == frame % 4
                outputs["reward_class"] = torch.tensor([frame % 3])
                outputs["ending_class"] = torch.tensor([frame % 2])
            else:
                assert inputs["observation_tokens"][call.input_index] == self.codes(rid, frame)[call.input_index]
            if call.sample_index is not None:
                outputs["observation_tokens"][call.sample_index] = self.codes(rid, frame)[call.sample_index]
            if tick + 1 == len(calls):
                self.runtime.stop_loops_batched(
                    "default",
                    IMAGINE_WALK,
                    node,
                    ParallelList([rid], [[TOKEN_LOOP]]),
                )
        else:
            frame = indices[ROLLOUT_LOOP]
            assert torch.equal(inputs["observation_tokens"], self.codes(rid, frame))
            assert inputs["reward_class"].item() == frame % 3
            assert inputs["ending_class"].item() == frame % 2
            outputs = {
                **inputs,
                "frames": torch.full((1, 64, 64, 3), frame, dtype=torch.uint8),
                "reward": inputs["reward_class"].float() - 1,
                "ending": inputs["ending_class"].bool(),
            }
        # Match the worker's order: read inputs, clean up, store outputs, route.
        self.finish_teardown(self.runtime.cleanup_consumed_inputs(node, [rid], [0]))
        names = list(popped.output_signals)
        uuids = [self.put(rid, outputs[name]) for name in names]
        routed = self.runtime.complete_and_route_batch(
            RouteInput(
                partition="default",
                graph_walk=IMAGINE_WALK,
                node_name=node,
                output_signals=names,
                wg_ids=ParallelList([rid], [0]),
                tensors=uuids,
                num_tensors=[1] * len(names),
            )
        )
        self.finish_teardown(routed.freed_inputs)
        self.runtime.clear_pending_loop_stops()
        if node == IMAGE_DECODER:
            assert {names[uuids.index(uuid)] for uuid in routed.register_uuids} == {"frames", "reward", "ending"}
            assert [names[index] for index in routed.new_token_output_idxs] == ["frames"]
            self.emits[rid] += 1
            self.store.dereference_batch_uniform(routed.register_uuids, cleanup=True)
        else:
            assert routed.register_uuids == routed.new_token_output_idxs == []

    def run(self):
        turns = cycle(self.horizons)
        limit = sum((h * 19 + 1) for h in self.horizons.values()) + len(self.horizons)
        for _ in range(limit):
            ready = self.runtime.get_ready_nodes(set())
            if not ready:
                break
            for _ in range(len(self.horizons)):
                rid = next(turns)
                matching = [entry for entry in ready if rid in entry.rids]
                if matching:
                    self.step(matching[0].node_name, rid)
                    break
        assert self.runtime.get_ready_nodes(set()) == []
        for rid, horizon in self.horizons.items():
            assert self.counts[rid, IMAGE_ENCODER] == 1
            assert self.counts[rid, INITIAL_DECODER] == 1
            assert self.counts[rid, IMAGE_DECODER] == self.emits[rid] == horizon
            assert self.counts[rid, WORLD_MODEL] == 17 * horizon + (horizon + 18) // 19
            assert sum(phase == "action" for _, phase, _, _ in self.phases[rid]) == horizon
            self.runtime.remove_request(rid)
        held = [uuid for uuid in self.created if self.book.get_info(uuid) is not None and not self.book.can_gc(uuid)]
        # Loop-external conditioning stays held until the worker tears down the request.
        assert set(held) == set(self.action_uuids.values())
        for rid in self.horizons:
            self.store.release_request(rid, force=True)
        assert self.store.stored_counts() == (0, 0)


@pytest.mark.parametrize("horizons", [(1,), (2,), (20,), (39,), (2, 20)])
def test_native_runtime_routes_complete_frames_and_keeps_requests_independent(horizons):
    _RoutingScript(horizons).run()
