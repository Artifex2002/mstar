"""Diagnostic adapter to the real Engine: capture, lease, execute, and audit replay.

No scheduling/runtime emulation and no model-private CUDA capture. This driver
synchronizes for correctness checks and must not be used as a timing baseline.
"""

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext, apply_yaml_overrides
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.iris.config import IMAGINE_WALK, INITIAL_DECODER
from mstar.model.submodule_base import InputMetadata
from scripts.iris.local_engine import LocalIrisEngine
from scripts.iris.qualify_cuda import indexed_cuda_device

ROOT = Path(__file__).resolve().parents[2]


def copy_step_outputs(node, request_ids, output):
    """Copy diagnostic outputs, restoring empty rows only for the no-output node.

    Engine._merge_per_rid omits empty dictionaries. The direct resource driver
    preserves them, so the initial reconstruction needs a representation adapter.
    The caller must already have checked admission, failures and actual replay.
    """
    per_request = output.per_rid_outputs
    expected = set(request_ids)
    if node == INITIAL_DECODER:
        if output.packed_outputs or output.row_outputs or set(per_request) - expected or any(per_request.values()):
            raise AssertionError("initial reconstruction unexpectedly emitted output data or unknown requests")
        return {rid: {} for rid in request_ids}
    if set(per_request) != expected:
        raise AssertionError(f"{node}: expected output requests {sorted(expected)}, got {sorted(per_request)}")
    return {
        rid: {name: [value.clone() for value in values] for name, values in tensors.items()}
        for rid, tensors in per_request.items()
    }


class CapturedIrisEngine(LocalIrisEngine):
    def __init__(self, model, modules, device="cuda"):
        self.model, self.device = model, indexed_cuda_device(device)
        self.horizons = {}
        self.replays, self.slots = Counter(), Counter()
        self.nodes = {id(module): node for node, module in modules.items()}
        self._patches = ExitStack()
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, yaml.safe_load((ROOT / "configs/iris_capture.yaml").read_text()))
        self.specs = {spec.resource_key: spec for spec in specs}
        self.engine = Engine(graph_runtime=None, autocast_dtype=None)
        try:
            self.engine.load_model(
                modules,
                specs,
                WorkerParallelGroups(1, 0),
                self.device,
                TransferEngineInfo("iris-capture-qualification", "local", LocalTransferEngine("localhost")),
                kv_cache_type=torch.float32,
                model=model,
            )
            self.engine.warmup()
            self.resources, self.runner = self.engine._resources, self.engine._runner
            self.captures = {}
            for node, management in self.engine._submodules.items():
                runner = management.accelerator_graph_runner
                expected = sum(
                    len(config.capture_batch_sizes)
                    for config in modules[node].get_accelerator_graph_configs(self.device)
                )
                if runner is None or runner.dropped_buckets or len(runner._buckets) != expected:
                    raise AssertionError(f"{node}: not every requested bucket captured; eager fallback is forbidden")
                self.captures[node] = {
                    "buckets": len(runner._buckets),
                    "slots_per_bucket": runner._num_slots,
                }
                original = runner.run_forward

                def audited(lease, *args, _node=node, _original=original, **kwargs):
                    result = _original(lease, *args, **kwargs)
                    self.replays[_node] += 1
                    self.slots[f"{_node}:{lease.slot}"] += 1
                    return result

                self._patches.enter_context(patch.object(runner, "run_forward", audited))
        except Exception:
            self.cleanup()
            raise

    def open(self, rid, horizon, seed=0):
        if rid in self.horizons:
            raise ValueError(f"request {rid} is already open")
        configs = self.request_configs(horizon, seed)
        self.horizons[rid] = horizon
        info = self.info(rid)
        info.resource_configs = configs
        try:
            self.engine.add_request(rid, configs, info)
        except Exception:
            del self.horizons[rid]
            raise

    @torch.inference_mode()
    def run(self, module, inputs, indices=None, draws=None, pad_to=None, preplan=False):
        if draws:
            raise ValueError("captured sampling is real; teacher-force copied carriers after replay instead")
        node = self.nodes[id(module)]
        infos = {rid: self.info(rid, *(indices or {}).get(rid, (0, 0))) for rid in inputs}
        metadata = {
            rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
            for rid, info in infos.items()
        }
        batch = ExecutingBatch(
            node_name=node,
            per_request_info=infos,
            per_request_input_tensors=inputs,
            per_request_input_metadata=metadata,
            step_context=StepContext(tuple(inputs), IMAGINE_WALK, 0, False),
        )
        self.engine.prepare_inputs(batch)
        if batch.failed_requests or len(batch.request_ids) != len(inputs):
            raise AssertionError(f"capture driver preparation failed: {batch.failed_requests}")
        if pad_to is not None:
            management = self.engine._submodules[node]
            runner = management.accelerator_graph_runner
            slot = management.lease_slot()
            key = module.cg_key_info(IMAGINE_WALK, batch.per_request_info_wrapped, metadata)
            lease = runner.lease_slot(
                IMAGINE_WALK,
                pad_to,
                slot,
                num_tokens=sum(row.input_seq_len for row in batch.inputs) * pad_to // len(inputs),
                cg_key_info=key,
            )
            if lease is None:
                raise AssertionError("no padded replay bucket")
            batch.lease_slot(lease)
        elif self.engine.reserve_replay_slot(batch) is None:
            raise AssertionError(f"{node}: no replay lease for a real step")
        if preplan:
            self.engine.pre_plan_for_batch(batch)
        before = self.replays[node]
        output = self.engine.exec_and_postprocess(batch)
        if batch.admit_error or batch.failed_requests or self.replays[node] != before + 1:
            raise AssertionError(f"{node}: step did not execute exactly one engine-managed replay")
        self.engine.after_output_copy(batch)
        # Diagnostic ownership: a later replay may reuse its graph output buffers.
        return copy_step_outputs(node, inputs, output)

    def close(self, rid):
        self.engine.remove_request(rid)
        del self.horizons[rid]

    def cleanup(self):
        if hasattr(self, "engine"):
            for rid in list(self.horizons):
                self.close(rid)
            self.engine.shutdown()
        self._patches.close()
