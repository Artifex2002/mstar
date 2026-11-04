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
def replay_scenarios(captured, eager, modules, reference_modules):
    from mstar.model.iris.config import IMAGE_DECODER, IMAGE_ENCODER, INITIAL_DECODER, WORLD_KV, WORLD_MODEL
    from mstar.model.iris.graph import frame_calls

    max_rgb_error, compared, padded_steps, preplanned_steps = 0, 0, 0, 0

    def run_pair(node, left, right, indices=None, padded=False):
        nonlocal max_rgb_error, compared, padded_steps, preplanned_steps
        preplan = node == WORLD_MODEL
        actual = captured.run(modules[node], left, indices, pad_to=2 if padded else None, preplan=preplan)
        expected = eager.run(reference_modules[node], right, indices)
        padded_steps += int(padded)
        preplanned_steps += int(preplan)
        assert actual.keys() == expected.keys()
        for rid in actual:
            assert actual[rid].keys() == expected[rid].keys()
            for name in actual[rid]:
                a, b = actual[rid][name][0], expected[rid][name][0]
                if name == "frames":
                    error = (a.to(torch.int16) - b.to(torch.int16)).abs().max().item()
                    max_rgb_error = max(max_rgb_error, error)
                    if error > 1:
                        raise AssertionError(f"{node}/{rid}: RGB24 error {error}")
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f"{node}/{rid}/{name}")
                compared += 1
        return actual, expected

    def trajectory(horizons, padded=False):
        for rid, horizon in horizons.items():
            captured.open(rid, horizon, seed=10 + rid)
            eager.open(rid, horizon, seed=10 + rid)
        actions = {rid: torch.arange(horizon, device="cuda") % 4 for rid, horizon in horizons.items()}
        inputs = {
            rid: {"initial_image": [torch.linspace(0, 1, 3 * 64 * 64, device="cuda").reshape(3, 64, 64)]}
            for rid in horizons
        }
        left, right = run_pair(IMAGE_ENCODER, inputs, inputs, padded=padded)
        run_pair(INITIAL_DECODER, left, right, padded=padded)
        for frame in range(max(horizons.values())):
            for tick in range(len(frame_calls(frame))):
                indices = {rid: (frame, tick) for rid in left}
                left, right = run_pair(
                    WORLD_MODEL,
                    {rid: {**values, "actions": [actions[rid]]} for rid, values in left.items()},
                    {rid: {**values, "actions": [actions[rid]]} for rid, values in right.items()},
                    indices,
                    padded,
                )
            decoded, _ = run_pair(IMAGE_DECODER, left, right, padded=padded)
            assert all(tuple(values["frames"][0].shape) == (1, 64, 64, 3) for values in decoded.values())
            for rid in list(left):
                if frame + 1 == horizons[rid]:
                    captured.close(rid)
                    eager.close(rid)
                    del left[rid], right[rid]

    # Real RNG draws, context refresh, two -> one row departure, then handle
    # reuse at one row replaying the two-row bucket (a real padded replay).
    trajectory({1: 20, 2: 2})
    trajectory({1: 1}, padded=True)
    counts = captured.resources[WORLD_KV].page_counts()
    assert counts.used == counts.cached == 0
    assert padded_steps and preplanned_steps
    for node in modules:
        assert captured.replays[node] > 0
    assert captured.slots[f"{WORLD_MODEL}:0"] and captured.slots[f"{WORLD_MODEL}:1"]
    return {
        "tensor_outputs_compared": compared,
        "max_rgb24_error": max_rgb_error,
        "world_carriers_rewards_endings": "exact",
        "horizons": [20, 2],
        "handle_reuse": "passed",
        "padded_replays": padded_steps,
        "preplanned_world_steps": preplanned_steps,
        "engine_run_forward_calls": dict(captured.replays),
        "replay_slots": dict(captured.slots),
        "kv_pages_after_cleanup": {"used": counts.used, "cached": counts.cached},
    }


def qualify(args, report, save):
    report["stage"] = "cuda_preflight"
    save()
    require_cuda()
    report["build_tools"] = prepare_build_tools()
    import mstar_rust

    from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope
    from mstar.model.iris.iris_model import IrisModel
    from scripts.iris.captured_engine import CapturedIrisEngine
    from scripts.iris.local_engine import LocalIrisEngine
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
        reference_model = IrisModel(checkpoint_path=str(args.checkpoint))
        modules, reference_modules = _modules(model, "cuda"), _modules(reference_model, "cuda")
        with LocalIrisEngine(reference_model, "cuda", CAPTURE_CONFIG) as eager:
            with CapturedIrisEngine(model, modules) as captured:
                report["captures"] = captured.captures
                report["stage"] = "engine_replay"
                save()
                report["replay"] = replay_scenarios(captured, eager, modules, reference_modules)
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
