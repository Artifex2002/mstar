"""Time synchronous rollouts, including resource lifecycle and per-frame D2H.

No trace hooks, per-step fingerprints or diagnostic output clones run here.
This measures a standalone Engine, not the worker or HTTP serving.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

from benchmark.iris.common import summarize, workload
from scripts.iris.local_engine import LocalIrisEngine
from scripts.iris.qualify_cuda import indexed_cuda_device, require_cuda
from scripts.iris.reference_harness import load_reference, verify_checkpoint

ROOT = Path(__file__).resolve().parents[2]


class EngineRollout(LocalIrisEngine):
    """Use the same real Engine APIs for eager and captured forwards."""

    def __init__(self, checkpoint, capture):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.distributed.communication import WorkerParallelGroups
        from mstar.engine.engine import Engine
        from mstar.engine.resources import apply_yaml_overrides
        from mstar.engine.resources.kv.transfer import TransferEngineInfo
        from mstar.model.iris.iris_model import IrisModel
        from scripts.iris.qualification import _modules

        self.model = IrisModel(checkpoint_path=str(checkpoint), capture_graphs=capture)
        self.device, self.capture, self.horizons = indexed_cuda_device(), capture, {}
        self.modules = _modules(self.model, self.device)
        specs = self.model.get_node_resources()
        config = ROOT / "configs" / ("iris_capture.yaml" if capture else "iris.yaml")
        apply_yaml_overrides(specs, yaml.safe_load(config.read_text()))
        self.specs = {spec.resource_key: spec for spec in specs}
        self.engine = Engine(None, autocast_dtype=None)
        try:
            self.engine.load_model(
                self.modules,
                specs,
                WorkerParallelGroups(1, 0),
                self.device,
                TransferEngineInfo("iris-benchmark", "local", LocalTransferEngine("localhost")),
                kv_cache_type=torch.float32,
                model=self.model,
            )
            self.engine.warmup()
            if capture:
                for node, module in self.modules.items():
                    runner = self.engine._submodules[node].accelerator_graph_runner
                    expected = sum(
                        len(c.capture_batch_sizes) for c in module.get_accelerator_graph_configs(self.device)
                    )
                    if runner is None or runner.dropped_buckets or len(runner._buckets) != expected:
                        raise RuntimeError(f"{node}: incomplete capture; benchmarking eager fallback is forbidden")
        except Exception:
            self.engine.shutdown()
            raise

    def step(self, node, inputs, frame=0, tick=0):
        from mstar.engine.engine import ExecutingBatch
        from mstar.engine.resources import StepContext
        from mstar.model.iris.config import IMAGINE_WALK
        from mstar.model.submodule_base import InputMetadata

        infos = {rid: self.info(rid, frame, tick) for rid in inputs}
        batch = ExecutingBatch(
            node_name=node,
            per_request_info=infos,
            per_request_input_tensors=inputs,
            per_request_input_metadata={
                rid: InputMetadata(dynamic_loop_iter_counts=info.dynamic_loop_iter_counts)
                for rid, info in infos.items()
            },
            step_context=StepContext(tuple(inputs), IMAGINE_WALK, 0, False),
        )
        self.engine.prepare_inputs(batch)
        if batch.failed_requests or len(batch.request_ids) != len(inputs):
            raise RuntimeError(f"{node}: failed to prepare full batch: {batch.failed_requests}")
        lease = self.engine.reserve_replay_slot(batch)
        if self.capture and lease is None:
            raise RuntimeError(f"{node}: captured run would fall back to eager")
        output = self.engine.exec_and_postprocess(batch)
        if batch.admit_error or batch.failed_requests:
            raise RuntimeError(f"{node}: failed execution: {batch.admit_error}, {batch.failed_requests}")
        self.engine.after_output_copy(batch)
        return output.per_rid_outputs

    def rollout(self, images, actions, seed):
        from mstar.model.iris.config import IMAGE_DECODER, IMAGE_ENCODER, INITIAL_DECODER, WORLD_MODEL
        from mstar.model.iris.graph import frame_calls

        batch, horizon = actions.shape
        records = []
        try:
            for rid in range(1, batch + 1):
                self.horizons[rid] = horizon
                self.engine.add_request(rid, self.request_configs(horizon, seed + rid - 1), self.info(rid))
            carry = self.step(IMAGE_ENCODER, {rid: {"initial_image": [images[rid - 1]]} for rid in self.horizons})
            self.step(INITIAL_DECODER, carry)
            for frame in range(horizon):
                for tick in range(len(frame_calls(frame))):
                    carry = self.step(
                        WORLD_MODEL,
                        {rid: {**row, "actions": [actions[rid - 1]]} for rid, row in carry.items()},
                        frame,
                        tick,
                    )
                output = self.step(IMAGE_DECODER, carry, frame)
                # Match a synchronous caller consuming RGB24/reward/end each frame.
                records.append(
                    {
                        rid: {name: row[name][0].cpu() for name in ("frames", "reward", "ending")}
                        for rid, row in output.items()
                    }
                )
        finally:
            for rid in list(self.horizons):
                self.engine.remove_request(rid)
                del self.horizons[rid]
        return records

    def cleanup(self):
        self.engine.shutdown()


class OfficialRollout:
    def __init__(self, reference, checkpoint):
        tokenizer, world, _, _ = load_reference(reference, checkpoint, "cuda")
        from envs.world_model_env import WorldModelEnv

        self.env = WorldModelEnv(tokenizer, world, "cuda")

    def rollout(self, images, actions, seed):
        torch.manual_seed(seed)
        self.env.reset_from_initial_observations(images)
        records = []
        for frame in range(actions.shape[1]):
            rgb, reward, ending, _ = self.env.step(actions[:, frame])
            rgb = (rgb * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
            records.append(
                {
                    rid + 1: {
                        "frames": rgb[rid : rid + 1],
                        "reward": torch.as_tensor(reward[rid : rid + 1]),
                        "ending": torch.as_tensor(ending[rid : rid + 1]),
                    }
                    for rid in range(actions.shape[0])
                }
            )
        return records


def validate_records(records, batch, horizon):
    if len(records) != horizon:
        raise AssertionError("incomplete rollout")
    for frame in records:
        assert set(frame) == set(range(1, batch + 1))
        for row in frame.values():
            assert row["frames"].dtype == torch.uint8 and row["frames"].shape == (1, 64, 64, 3)
            assert row["reward"].shape == (1,) and row["reward"].item() in (-1, 0, 1)
            assert row["ending"].dtype == torch.bool and row["ending"].shape == (1,)


def fingerprint(records):
    digest = hashlib.sha256()
    for frame in records:
        for rid in sorted(frame):
            for name in ("frames", "reward", "ending"):
                digest.update(frame[rid][name].detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("official", "eager", "captured"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.warmup, args.samples) < 1:
        parser.error("warmup and samples must be positive")
    require_cuda()
    verify_checkpoint(args.checkpoint)
    torch.set_num_threads(2)
    from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope

    results = []
    with fp32_precision_scope(Fp32Precision("highest", False)):
        driver = (
            OfficialRollout(args.reference_root, args.checkpoint)
            if args.arm == "official"
            else EngineRollout(args.checkpoint, args.arm == "captured")
        )
        try:
            for horizon in (1, 20):
                for batch in (1, 2):
                    images, actions = [torch.from_numpy(x).to("cuda") for x in workload(batch, horizon)]
                    durations = []
                    expected_digest = None
                    for index in range(args.warmup + args.samples):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        records = driver.rollout(images, actions, 17)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        validate_records(records, batch, horizon)
                        digest = fingerprint(records)
                        if expected_digest is not None and digest != expected_digest:
                            raise AssertionError(f"{args.arm}/B{batch}/H{horizon}: same-process seeded repeat differs")
                        expected_digest = digest
                        if index >= args.warmup:
                            durations.append(elapsed)
                    results.append(
                        {
                            "batch": batch,
                            "horizon": horizon,
                            "output_sha256": expected_digest,
                            "seconds": durations,
                            **summarize(durations, batch * horizon * args.samples),
                        }
                    )
        finally:
            if isinstance(driver, EngineRollout):
                driver.cleanup()
    args.output.write_text(json.dumps({"status": "passed", "arm": args.arm, "results": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
