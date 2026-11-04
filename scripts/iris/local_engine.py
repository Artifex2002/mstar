"""Single-device diagnostic driver for real submodules and engine resources.

This drives the lifecycle directly. It does not emulate the worker, native graph,
transport or API server, and is not a performance harness.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.engine.resources import StepContext, StepRunner, apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.iris.config import IMAGINE_WALK, ROLLOUT_LOOP, TOKEN_LOOP, WORLD_KV
from mstar.model.iris.resources import request_kv_layout
from mstar.model.submodule_base import ModelInputsFromEngine

ROOT = Path(__file__).resolve().parents[2]


class LocalIrisEngine:
    def __init__(self, model, device="cpu"):
        self.model, self.device = model, torch.device(device)
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, yaml.safe_load((ROOT / "configs/iris.yaml").read_text()))
        self.specs = resolve_spec_dependencies(specs)
        transfer = TransferEngineInfo("iris-qualification", "local", LocalTransferEngine("localhost"))
        self.resources = {
            key: build_resource(
                spec,
                EngineResourceInfo(
                    device=self.device,
                    kv_dtype=torch.float32,
                    transfer_engine_info=transfer,
                    needs_remote_transfer=False,
                    nodes=frozenset(spec.nodes),
                    dependencies={dep: self.specs[dep] for dep in spec.depends_on()},
                ),
            )
            for key, spec in self.specs.items()
        }
        nodes = {node: [key for key, spec in self.specs.items() if node in spec.nodes] for node in model.nodes}
        self.runner = StepRunner(self.resources, node_resources=nodes)
        self.horizons = {}

    def open(self, rid, horizon, seed=0):
        if rid in self.horizons:
            raise ValueError(f"request {rid} is already open")
        configs = self.model.get_request_resource_configs({}, {"iris_horizon": horizon})
        configs[WORLD_KV].apply_conductor_config(
            kv_layout={label: layout.row() for label, layout in request_kv_layout(horizon)[WORLD_KV].items()},
            loop_caps=self.model.get_loop_caps(iris_horizon=horizon),
        )
        for key, config in configs.items():
            config.apply_conductor_config(seed=seed, resource_key=key)
            config.validate(self.specs[key])
        self.runner.ingest_request(rid, configs)
        self.horizons[rid] = horizon

    def info(self, rid, frame=0, token=0):
        return CurrentForwardPassInfo(
            f"iris-{rid}",
            IMAGINE_WALK,
            0,
            0,
            self.horizons[rid],
            rid_handle=rid,
            step_metadata={"iris_horizon": self.horizons[rid]},
            dynamic_loop_iter_counts={ROLLOUT_LOOP: frame, TOKEN_LOOP: token},
        )

    @torch.inference_mode()
    def run(self, module, inputs, indices=None, draws=None):
        rids = list(inputs)
        infos = {rid: self.info(rid, *(indices or {}).get(rid, (0, 0))) for rid in rids}
        rows = [module.prepare_inputs(IMAGINE_WALK, infos[rid], inputs[rid]) for rid in rids]
        if any(row is None for row in rows):
            raise ValueError("diagnostic driver was asked to execute a speculative overshoot")
        if not module.can_batch(None, rows):
            raise ValueError("diagnostic batch is incompatible with the submodule")
        step = module.declare_step(IMAGINE_WALK, rids, rows)
        step.set_ctx(StepContext(tuple(rids), IMAGINE_WALK, 0, False))
        outcome = self.runner.admit(step)
        if not outcome.outcome.ok:
            raise RuntimeError(f"resource admission failed: {outcome}")
        self.runner.plan(step)
        engine = ModelInputsFromEngine(rids, infos, resources=self.resources, step=step)
        with ExitStack() as stack:
            # Teacher forcing replaces only the draw. Admission, planning and
            # commit still run on the real sampler resource.
            for key, value in (draws or {}).items():
                stack.enter_context(patch.object(self.resources[key], "sample", return_value=value))
            stack.enter_context(fp32_precision_scope(node_fp32_precision(type(module).__name__, module)))
            prepared = module.preprocess(IMAGINE_WALK, engine, rows)
            if len(rids) == 1:
                outputs = {rids[0]: module(IMAGINE_WALK, engine, **prepared)}
            else:
                outputs = module.forward_batched(IMAGINE_WALK, engine, **prepared)
        self.runner.commit(step)
        return outputs

    def close(self, rid):
        self.runner.remove_request(rid)
        del self.horizons[rid]

    def cleanup(self):
        for rid in list(self.horizons):
            self.close(rid)
        for resource in self.resources.values():
            resource.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
