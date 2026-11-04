"""Qualify FP32 static attention and real engine-managed CUDA capture/replay."""

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.iris.qualify_cuda import git_output, prepare_build_tools, require_cuda
from scripts.iris.reference_harness import REFERENCE_COMMIT, download_checkpoint, verify_checkpoint
from scripts.iris.reference_harness import run as reference_run

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_CONFIG = ROOT / "configs/iris_capture.yaml"


@torch.inference_mode()
def replay_scenarios(captured, modules, trace, stage):
    from mstar.model.iris.config import (
        ENDING_SAMPLER,
        IMAGE_DECODER,
        IMAGE_ENCODER,
        INITIAL_DECODER,
        OBSERVATION_SAMPLER,
        REWARD_SAMPLER,
        WORLD_KV,
        WORLD_MODEL,
        WORLD_POS,
    )
    from mstar.model.iris.graph import frame_calls
    from scripts.iris.capture_checks import (
        check_decoded,
        check_rng_offsets,
        checked_world_outputs,
        output_digests,
    )
    from scripts.iris.qualification import Comparisons

    comparisons = Comparisons()
    max_rgb_error, padded_steps, preplanned_steps = 0, 0, 0
    counters = {}
    device = captured.device

    def trajectory(horizons, teacher_forced, padded=False):
        nonlocal max_rgb_error, padded_steps, preplanned_steps
        fingerprints = {}
        forcing = trace if teacher_forced else None

        def run(node, inputs, indices=None):
            nonlocal padded_steps, preplanned_steps
            preplan = node == WORLD_MODEL
            output = captured.run(modules[node], inputs, indices, pad_to=2 if padded else None, preplan=preplan)
            padded_steps += int(padded)
            preplanned_steps += int(preplan)
            context = f"schedule={tuple(horizons.items())}/padded={padded}/{node}/{indices or 'initial'}"
            fingerprints.update(output_digests(output, context))
            return output

        for rid, horizon in horizons.items():
            captured.open(rid, horizon, seed=10 + rid)
        actions = {
            rid: torch.from_numpy(trace["input/actions"][rid - 1, :horizon].copy()).to(device)
            for rid, horizon in horizons.items()
        }
        inputs = {
            rid: {"initial_image": [torch.from_numpy(trace["initial/observations"][rid - 1].copy()).to(device)]}
            for rid in horizons
        }
        carry = run(IMAGE_ENCODER, inputs)
        for rid, values in carry.items():
            comparisons.check("encoder_codes", values["observation_tokens"][0], trace["initial/tokens"][rid - 1], 0, 0)
        run(INITIAL_DECODER, carry)
        call_number = 0
        for frame in range(max(horizons.values())):
            for tick, call in enumerate(frame_calls(frame)):
                indices = {rid: (frame, tick) for rid in carry}
                output = run(
                    WORLD_MODEL, {rid: {**values, "actions": [actions[rid]]} for rid, values in carry.items()}, indices
                )
                carry = checked_world_outputs(output, carry, frame, tick, call_number, comparisons, forcing)
                for rid in carry:
                    expected = int(trace[f"call/{call_number:04}/cache_size"])
                    actual = captured.resources[WORLD_POS].position(rid, call.label)
                    assert actual == expected, (
                        f"frame={frame} tick={tick} request={rid}: position {actual} != {expected}"
                    )
                call_number += 1
            decoded = run(IMAGE_DECODER, carry, {rid: (frame, 0) for rid in carry})
            max_rgb_error = max(max_rgb_error, check_decoded(decoded, carry, frame, forcing))
            for rid in list(carry):
                if frame + 1 == horizons[rid]:
                    counters[f"H{horizons[rid]}"] = check_rng_offsets(captured, rid, frame + 1)
                    captured.close(rid)
                    for key in (OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
                        resource = captured.resources[key]
                        assert rid not in resource._sampler._sampling_config
                        assert rid not in resource._cg_buffers._rid_to_slot
                    del carry[rid]
        counts = captured.resources[WORLD_KV].page_counts()
        assert counts.used == counts.cached == 0
        return fingerprints

    # Numerical parity uses identical official inputs, independent of which valid
    # token each sampler draws. The real draws still execute and advance counters.
    stage("captured_teacher_forced_parity")
    trajectory({1: 20, 2: 2}, teacher_forced=True)
    trajectory({1: 1}, teacher_forced=True, padded=True)
    # Repeat the same captured algorithm and schedule after teardown/handle reuse.
    stage("captured_sampling_first_run")
    first = trajectory({1: 20, 2: 2}, teacher_forced=False)
    first.update(trajectory({1: 1}, teacher_forced=False, padded=True))
    stage("captured_sampling_repeat")
    second = trajectory({1: 20, 2: 2}, teacher_forced=False)
    second.update(trajectory({1: 1}, teacher_forced=False, padded=True))
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], f"captured sampling repeat mismatch: {name}"
    assert padded_steps and preplanned_steps
    for node in modules:
        assert captured.replays[node] > 0
    assert captured.slots[f"{WORLD_MODEL}:0"] and captured.slots[f"{WORLD_MODEL}:1"]
    return {
        "teacher_forced_official_parity": comparisons.report(),
        "max_rgb24_error": max_rgb_error,
        "captured_sampling_repeatability": {"status": "exact", "tensor_fingerprints_compared": len(first)},
        "cross_algorithm_seed_identity": "not required: eager FlashInfer and captured fused_sample differ",
        "rng_counters": counters,
        "horizons": [20, 2],
        "handle_reuse": "passed",
        "padded_replays": padded_steps,
        "preplanned_world_steps": preplanned_steps,
        "engine_run_forward_calls": dict(captured.replays),
        "replay_slots": dict(captured.slots),
        "kv_pages_after_cleanup": {"used": 0, "cached": 0},
    }


def qualify(args, report, save):
    report["stage"] = "cuda_preflight"
    save()
    require_cuda()
    report["build_tools"] = prepare_build_tools()
    import mstar_rust

    from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope
    from mstar.model.iris.config import WORLD_MODEL
    from mstar.model.iris.iris_model import IrisModel
    from scripts.iris.capture_checks import DiagnosticWorldSubmodule
    from scripts.iris.captured_engine import CapturedIrisEngine
    from scripts.iris.qualification import _modules, compare_trace

    report["hardware"] = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "rust_extension": mstar_rust.__version__,
    }
    if args.download_checkpoint:
        download_checkpoint(args.checkpoint)
    report["checkpoint_sha256"] = verify_checkpoint(args.checkpoint)
    if git_output("rev-parse", "HEAD", cwd=args.reference_root) != REFERENCE_COMMIT:
        raise ValueError(f"official source must be pinned at {REFERENCE_COMMIT}")
    report["reference_commit"] = REFERENCE_COMMIT
    torch.set_num_threads(2)
    with fp32_precision_scope(Fp32Precision("highest", False)):
        report["static_attention_parity"] = []
        for batch in (1, 2):
            report["stage"] = f"static_attention_reference_batch_{batch}"
            save()
            reference_output = args.output / f"official-batch-{batch}"
            reference_run(
                SimpleNamespace(
                    reference_root=args.reference_root,
                    checkpoint=args.checkpoint,
                    device="cuda",
                    threads=2,
                    seed=0,
                    observations=None,
                    actions=None,
                    batch_size=batch,
                    horizon=20,
                    output=reference_output,
                    compare=None,
                )
            )
            with np.load(reference_output / "trace.npz", allow_pickle=False) as trace:
                report["static_attention_parity"].append(
                    compare_trace(args.checkpoint, dict(trace), "cuda", batch, CAPTURE_CONFIG)
                )
        report["stage"] = "engine_capture"
        save()
        model = IrisModel(checkpoint_path=str(args.checkpoint), capture_graphs=True)
        modules = _modules(model, "cuda")
        modules[WORLD_MODEL] = DiagnosticWorldSubmodule(modules[WORLD_MODEL].network, 20, capture_graphs=True)
        with np.load(args.output / "official-batch-2/trace.npz", allow_pickle=False) as data:
            trace = dict(data)
        with CapturedIrisEngine(model, modules) as captured:
            report["captures"] = captured.captures

            def stage(name):
                report["stage"] = name
                save()

            report["replay"] = replay_scenarios(captured, modules, trace, stage)
    report.update(status="passed", stage="complete")
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(".ci-results/iris/Breakout.pt"))
    parser.add_argument("--reference-root", type=Path, default=Path(".ci-results/iris/reference"))
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "stage": "initialization",
        "started_utc": datetime.now(UTC).isoformat(),
        "mstar_commit": git_output("rev-parse", "HEAD"),
        "working_tree_status": git_output("status", "--porcelain", "--untracked-files=no"),
        "purpose": "FP32 attention parity and engine-managed CUDA capture/replay correctness; no timing claims",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        qualify(args, report, save)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        print(report["traceback"])
        print(f"FAILED at {report['stage']}; send {args.output / 'report.json'}")
        return 1
    print(f"PASSED; send {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
