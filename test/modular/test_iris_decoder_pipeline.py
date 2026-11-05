"""Async decoder experiment: real native routing and stream-buffer ownership."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.model.iris.config import (
    DECODE_LOOP,
    DECODER_PARTITION,
    FRAME_DECODE_WALK,
    FRAME_STREAMS,
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
from mstar.model.iris.submodules import WorldSubmodule
from test.modular.test_iris_graph import _RoutingScript

ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_is_optin_bounded_and_uses_independent_partition():
    assert not IrisModel().config.decoder_pipeline
    with pytest.raises(ValueError, match="decoder_pipeline"):
        IrisModel(decoder_pipeline="true")
    with pytest.raises(ValueError, match="20 frames"):
        IrisModel(decoder_pipeline=True, max_rollout_horizon=21)
    model = IrisModel(decoder_pipeline=True)
    assert set(model.get_graph_walk_graphs()) == {IMAGINE_WALK, FRAME_DECODE_WALK}
    assert model.get_loop_caps(iris_horizon=2)[DECODE_LOOP] == 2
    assert [c.edge_name for c in model.get_partition_topology().connections] == list(FRAME_STREAMS)
    assert all(c.chunk_policy_factory().window_size() == 1 for c in model.get_partition_topology().connections)
    assert set(model.get_graph_walk_graphs()[IMAGINE_WALK].get_nodes()) == {
        IMAGE_ENCODER,
        WORLD_MODEL,
        INITIAL_DECODER,
    }


class PipelineRouting(_RoutingScript):
    def __init__(self, horizons):
        pytest.importorskip("mstar_rust")
        from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
        from mstar.distributed.base import ShardingConfig
        from mstar.graph.runtime.base import ColumnarEdgeSpecs
        from mstar.graph.runtime.rust import RustGraphRuntime
        from mstar.utils.containers import ParallelList
        from mstar.worker.stream_buffers import StreamBufferManager

        self.model = IrisModel(decoder_pipeline=True, max_rollout_horizon=max(horizons))
        graphs = self.model.get_worker_graphs(str(ROOT / "configs/iris_pipeline.yaml"))
        assert len(graphs) == 2
        for index, graph in enumerate(graphs):
            graph.worker_graph_id = index
        self.ids = {next(iter(g.graph_walks)): g.worker_graph_id for g in graphs}
        self.book = RustTensorBookkeeping()
        self.store = TensorStore(bookkeeping=self.book)
        self.runtime = RustGraphRuntime(
            my_worker_id="worker_0",
            my_worker_graphs=graphs,
            all_wg_ids_to_graph_walks={g.worker_graph_id: g.graph_walks for g in graphs},
            all_wg_ids_to_dyn_loops={g.worker_graph_id: set(g.section.get_loops()) for g in graphs},
            all_wg_ids_to_nodes={g.worker_graph_id: set(g.section.get_nodes()) for g in graphs},
            node_to_partition={
                node: DECODER_PARTITION if node == IMAGE_DECODER else "default" for node in self.model.nodes
            },
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
            bookkeeping=self.book,
        )
        self.state = SimpleNamespace(per_request_info={})
        self.streams = StreamBufferManager(
            self.state,
            self.store,
            self.model.get_partition_topology().connections,
            self.model.get_graph_walk_graphs(),
            graph_runtime=self.runtime,
        )
        self.horizons, self.counts, self.emits = {}, Counter(), Counter()
        self.created, self.uuid, self.peak, self.action_uuids = [], 100, 0, {}
        self.world = WorldSubmodule(None, max(horizons), stream_frames=True)
        for index, horizon in enumerate(horizons):
            mapping = ParallelList([g.worker_graph_id for g in graphs], [["worker_0"] for g in graphs])
            rid = self.runtime.add_request(
                f"iris-{index}",
                "default",
                IMAGINE_WALK,
                [self.ids[IMAGINE_WALK]],
                mapping,
                self.model.get_loop_caps(iris_horizon=horizon),
            )
            assert (
                self.runtime.add_request(
                    f"iris-{index}",
                    DECODER_PARTITION,
                    FRAME_DECODE_WALK,
                    [self.ids[FRAME_DECODE_WALK]],
                    mapping,
                    self.model.get_loop_caps(iris_horizon=horizon),
                )
                == rid
            )
            self.horizons[rid] = horizon
            info = SimpleNamespace(
                stream_buffers={},
                stream_buffers_by_consumer={},
                partition_name=DECODER_PARTITION,
                stream_lead_items={},
            )
            self.state.per_request_info[rid] = info
            self.streams.add_request(rid, info)
            edges = ColumnarEdgeSpecs.empty()
            edges.add(rid, "initial_image", [self.put(rid, torch.zeros(3, 64, 64))], False, IMAGE_ENCODER)
            edges.add(rid, "actions", [self.put(rid, torch.arange(horizon) % 4)], False, WORLD_MODEL)
            assert self.runtime.ingest_inputs_batch(edges) == []

    def step(self, node, rid):
        from mstar.graph.runtime.base import RouteInput
        from mstar.utils.containers import ParallelList

        consumer = node == IMAGE_DECODER
        walk, partition = (FRAME_DECODE_WALK, DECODER_PARTITION) if consumer else (IMAGINE_WALK, "default")
        popped = self.runtime.pop_rids(node, walk, [rid], check_ready=True)
        assert popped is not None
        inputs = {name: self.store.get_tensor(uuids[0]) for _, name, uuids, _ in popped.input_edges.edge_tuples()}
        counts = self.runtime.get_dynamic_loop_iters([rid], partition).values[0]
        self.counts[rid, node] += 1
        withheld = ()
        if node == IMAGE_ENCODER:
            outputs = dict(zip(WORLD_CARRY, (self.codes(rid, -1), torch.tensor([1]), torch.tensor([0])), strict=True))
        elif node == INITIAL_DECODER:
            assert torch.equal(inputs["observation_tokens"], self.codes(rid, -1))
            outputs = {}
        elif node == WORLD_MODEL:
            frame, tick = counts[ROLLOUT_LOOP], counts[TOKEN_LOOP]
            call = frame_calls(frame)[tick]
            outputs = {name: inputs[name].clone() for name in WORLD_CARRY}
            if call.kind == "action":
                outputs["reward_class"] = torch.tensor([frame % 3])
                outputs["ending_class"] = torch.tensor([frame % 2])
            if call.sample_index is not None:
                outputs["observation_tokens"][call.sample_index] = self.codes(rid, frame)[call.sample_index]
            if tick + 1 == len(frame_calls(frame)):
                outputs.update({s: outputs[n] for s, n in zip(FRAME_STREAMS, WORLD_CARRY, strict=True)})
            stop = self.world.check_stop(rid, SimpleNamespace(dynamic_loop_iter_counts=counts), {})
            withheld = stop.withheld
            if stop.stops:
                self.runtime.stop_loops_batched(partition, walk, node, ParallelList([rid], [list(stop.stops)]))
        else:
            frame = counts[DECODE_LOOP]
            assert frame == self.emits[rid]
            assert torch.equal(inputs[FRAME_STREAMS[0]], self.codes(rid, frame))
            assert inputs[FRAME_STREAMS[1]].item() == frame % 3
            assert inputs[FRAME_STREAMS[2]].item() == frame % 2
            outputs = {
                "frames": torch.full((1, 64, 64, 3), frame, dtype=torch.uint8),
                "reward": inputs[FRAME_STREAMS[1]].float() - 1,
                "ending": inputs[FRAME_STREAMS[2]].bool(),
            }
        self.finish_teardown(self.runtime.cleanup_consumed_inputs(node, [rid], [self.ids[walk]]))
        names = [name for name in popped.output_signals if name not in withheld]
        values = [self.put(rid, outputs[name]) for name in names]
        routed = self.runtime.complete_and_route_batch(
            RouteInput(
                partition=partition,
                graph_walk=walk,
                node_name=node,
                output_signals=names,
                wg_ids=ParallelList([rid], [self.ids[walk]]),
                tensors=values,
                num_tensors=[1] * len(names),
            )
        )
        for name, pairs in routed.local_streaming_by_signal.items():
            for row, uuid in pairs:
                buffer = self.state.per_request_info[row].stream_buffers[name]
                buffer.pre_read_register_joined([uuid])
                buffer.put(uuid, uuid)
                self.peak = max(self.peak, buffer._num_tensors_registered - buffer._consumed)
        self.finish_teardown(routed.freed_inputs)
        self.runtime.clear_pending_loop_stops()
        if consumer:
            assert len(routed.register_uuids) == 3
            self.emits[rid] += 1
            assert self.runtime.is_partition_idle(rid, DECODER_PARTITION) == (self.emits[rid] == self.horizons[rid])
            self.store.dereference_batch_uniform(routed.register_uuids, cleanup=True)

    def poll(self):
        polled = self.streams.poll()
        if polled:
            refused = self.runtime.ingest_inputs_batch(
                self.streams.edge_block(polled),
                can_buffer=False,
                is_streaming=True,
            )
            for index, (rid, edge) in enumerate(polled):
                if index in refused:
                    self.streams.return_edge(rid, edge)
                else:
                    self.streams.mark_ingested(rid, edge)

    def run(self):
        # Deliberately withhold all decoder work: producer progress must not
        # depend on RGB decoding, and queued snapshots must remain immutable.
        for _ in range(sum(h * 19 + 2 for h in self.horizons.values())):
            ready = [entry for entry in self.runtime.get_ready_nodes(set()) if entry.node_name != IMAGE_DECODER]
            if not ready:
                break
            entry = ready[0]
            self.step(entry.node_name, entry.rids[0])
        assert all(
            self.counts[rid, WORLD_MODEL] == len(sum((frame_calls(f) for f in range(h)), ()))
            for rid, h in self.horizons.items()
        )
        assert self.peak == max(self.horizons.values())
        for _ in range(sum(self.horizons.values()) + 2):
            self.poll()
            ready = self.runtime.get_ready_nodes(set())
            if not ready:
                break
            for entry in ready:
                assert entry.node_name == IMAGE_DECODER
                self.step(IMAGE_DECODER, entry.rids[0])
        assert dict(self.emits) == self.horizons
        for rid in self.horizons:
            self.runtime.remove_request(rid)
            self.store.release_request(rid, force=True)
        assert self.store.stored_counts() == (0, 0)


@pytest.mark.parametrize("horizons", [(1,), (2,), (20,), (20, 2)])
def test_native_pipeline_can_generate_whole_rollout_before_decoding_without_corrupting_frames(horizons):
    PipelineRouting(horizons).run()


@pytest.mark.parametrize("captured", [False, True])
def test_pipeline_deployment_validates_and_decoder_capture_targets_its_walk(tmp_path, monkeypatch, captured):
    import triton.language as tl
    import yaml

    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)
    from mstar.model.iris.components import Tokenizer
    from mstar.model.iris.submodules import StreamingImageSubmodule
    from scripts.iris.qualify_server import prepare_server_config

    config = ROOT / "configs" / ("iris_capture_pipeline.yaml" if captured else "iris_pipeline.yaml")
    baseline = ROOT / "configs" / ("iris_capture.yaml" if captured else "iris.yaml")
    new_config = yaml.safe_load(config.read_text())
    assert new_config["model_kwargs"].pop("decoder_pipeline") is True
    assert new_config == yaml.safe_load(baseline.read_text())
    prepare_server_config(tmp_path / "Breakout.pt", tmp_path, config)
    module = StreamingImageSubmodule(Tokenizer(), decode=True, capture_graphs=True)
    assert module.get_accelerator_graph_configs("cpu")[0].capture_graph_walk == FRAME_DECODE_WALK


def test_consumer_starts_without_producer_inputs_and_finishes_only_its_own_walk():
    from mstar.model.base import ForwardPassArgs

    model = IrisModel(decoder_pipeline=True)
    started = model.get_initial_forward_pass_args(
        DECODER_PARTITION, ["image"], ["video_frame", "scalar", "tensor"], {}, {"iris_horizon": 2}
    )
    assert isinstance(started, ForwardPassArgs)
    assert not started.inputs and not started.request_done
    assert started.full_metadata.graph_walk == FRAME_DECODE_WALK
    assert started.step_metadata == {"iris_horizon": 2}
    assert model.get_partition_forward_pass_args(DECODER_PARTITION, started.full_metadata, {}).request_done
    with pytest.raises(ValueError, match="unknown walk"):
        model.get_partition_forward_pass_args(DECODER_PARTITION, SimpleNamespace(graph_walk=IMAGINE_WALK), {})


def test_streamed_decoder_rejects_misalignment_and_absorbs_speculative_overshoot():
    from mstar.model.iris.submodules import StreamingImageSubmodule
    from mstar.model.submodule_base import InputMetadata
    from mstar.streaming.stream_buffer import StreamChunkInfo

    module = StreamingImageSubmodule(SimpleNamespace(embedding=None, decoder=None, post_quant_conv=None), decode=True)
    info = SimpleNamespace(dynamic_loop_iter_counts={DECODE_LOOP: 1}, step_metadata={"iris_horizon": 2})
    chunks = {name: StreamChunkInfo(1, 0, 1, False) for name in FRAME_STREAMS}
    inputs = {name: [torch.zeros(1)] for name in FRAME_STREAMS}
    row = module.prepare_inputs(FRAME_DECODE_WALK, info, inputs, InputMetadata(stream_chunks=chunks))
    assert set(row.tensor_inputs) == set(WORLD_CARRY)
    chunks[FRAME_STREAMS[1]] = StreamChunkInfo(0, 0, 1, False)
    with pytest.raises(ValueError, match="misaligned"):
        module.prepare_inputs(FRAME_DECODE_WALK, info, inputs, InputMetadata(stream_chunks=chunks))
    with pytest.raises(ValueError, match="every carrier"):
        module.prepare_inputs(FRAME_DECODE_WALK, info, inputs)
    chunks = {name: StreamChunkInfo(1, 0, 0, True) for name in FRAME_STREAMS}
    with pytest.raises(ValueError, match="misaligned"):
        module.prepare_inputs(FRAME_DECODE_WALK, info, {}, InputMetadata(stream_chunks=chunks))
    info.dynamic_loop_iter_counts[DECODE_LOOP] = 2
    assert module.prepare_inputs(FRAME_DECODE_WALK, info, {}) is None


def test_engine_snapshots_streamed_aliases_before_reusing_capture_outputs(monkeypatch):
    import triton.language as tl

    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import BatchedModelOutput

    engine = Engine.__new__(Engine)
    module = WorldSubmodule(None, 1, stream_frames=True)
    static = {name: torch.full((2,), i, dtype=torch.long) for i, name in enumerate(WORLD_CARRY)}
    raw = {**static, **{s: static[n] for s, n in zip(FRAME_STREAMS, WORLD_CARRY, strict=True)}}
    outputs = {}
    engine._merge_per_rid(outputs, BatchedModelOutput(per_rid_outputs={1: raw}), [1], [1], module, {})
    for value in static.values():
        value.fill_(-1)  # a subsequent capture replay overwrites its static output buffers
    for index, name in enumerate(FRAME_STREAMS):
        assert outputs[1][name][0].tolist() == [index, index]


@pytest.mark.skipif(not (ROOT / ".ci-results/iris/Breakout.pt").is_file(), reason="pinned checkpoint not cached")
def test_real_weights_delayed_decoding_matches_serial_rollout_through_context_refresh(monkeypatch):
    from contextlib import ExitStack

    from benchmark.iris.common import workload
    from mstar.engine.resources.sampler.resource import SamplerResource
    from mstar.model.submodule_base import InputMetadata
    from mstar.streaming.stream_buffer import StreamChunkInfo
    from scripts.iris.local_engine import LocalIrisEngine
    from scripts.iris.qualification import _modules

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(SamplerResource, "sample", lambda self, rids, logits: logits.argmax(-1))
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        checkpoint = str(ROOT / ".ci-results/iris/Breakout.pt")
        serial_model = IrisModel(checkpoint_path=checkpoint)
        pipeline_model = IrisModel(checkpoint_path=checkpoint, decoder_pipeline=True)
        serial_modules, pipeline_modules = _modules(serial_model, "cpu"), _modules(pipeline_model, "cpu")
        images, actions = map(torch.from_numpy, workload(2, 20))
        with ExitStack() as stack:
            serial = stack.enter_context(LocalIrisEngine(serial_model))
            pipeline = stack.enter_context(LocalIrisEngine(pipeline_model))
            for engine in (serial, pipeline):
                for rid in (1, 2):
                    engine.open(rid, 20, 17 + rid)
            inputs = {rid: {"initial_image": [images[rid - 1]]} for rid in (1, 2)}
            left = serial.run(serial_modules[IMAGE_ENCODER], inputs)
            right = pipeline.run(pipeline_modules[IMAGE_ENCODER], inputs)
            serial.run(serial_modules[INITIAL_DECODER], left)
            pipeline.run(pipeline_modules[INITIAL_DECODER], right)
            pending, expected = [], []
            for frame in range(20):
                for tick in range(len(frame_calls(frame))):
                    indices = {rid: (frame, tick) for rid in (1, 2)}
                    left = serial.run(
                        serial_modules[WORLD_MODEL],
                        {rid: {**row, "actions": [actions[rid - 1]]} for rid, row in left.items()},
                        indices,
                    )
                    right = pipeline.run(
                        pipeline_modules[WORLD_MODEL],
                        {rid: {**row, "actions": [actions[rid - 1]]} for rid, row in right.items()},
                        indices,
                    )
                    for rid in (1, 2):
                        for name in WORLD_CARRY:
                            torch.testing.assert_close(left[rid][name][0], right[rid][name][0], rtol=0, atol=0)
                pending.append({rid: {name: right[rid][name] for name in FRAME_STREAMS} for rid in (1, 2)})
                expected.append(serial.run(serial_modules[IMAGE_DECODER], left))
            for frame, inputs in enumerate(pending):
                chunks = {name: StreamChunkInfo(frame, 0, 1, frame == 19) for name in FRAME_STREAMS}
                metadata = {
                    rid: InputMetadata(stream_chunks=chunks, dynamic_loop_iter_counts={DECODE_LOOP: frame})
                    for rid in (1, 2)
                }
                actual = pipeline.run(
                    pipeline_modules[IMAGE_DECODER], inputs, graph_walk=FRAME_DECODE_WALK, input_metadata=metadata
                )
                for rid in (1, 2):
                    for name in ("frames", "reward", "ending"):
                        torch.testing.assert_close(actual[rid][name][0], expected[frame][rid][name][0], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
