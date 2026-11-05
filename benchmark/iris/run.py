"""Run sequential, interleaved IRIS arms on one Linux NVIDIA allocation."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from scripts.iris.qualify_cuda import git_output, prepare_build_tools
from scripts.iris.reference_harness import REFERENCE_COMMIT, download_checkpoint, verify_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def write_summary(report, output):
    with (output / "summary.tsv").open("w") as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(
            ["scope", "arm", "round", "batch", "concurrency", "horizon", "frames_per_second", "p50_ms", "p95_ms"]
        )
        for scope, key in (("standalone", "results"), ("http", "cells")):
            for boot in report[scope]:
                for cell in boot[key]:
                    writer.writerow(
                        [
                            scope,
                            boot["arm"],
                            boot["round"],
                            cell.get("batch", ""),
                            cell.get("concurrency", ""),
                            cell["horizon"],
                            cell["frames_per_second"],
                            cell["latency_ms"]["p50"],
                            cell["latency_ms"]["p95"],
                        ]
                    )


def verify_sources(report):
    if git_output("rev-parse", "HEAD") != report["mstar_commit"]:
        raise RuntimeError("checkout commit changed during the sweep; discard this comparison")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files changed during the sweep; discard this comparison")
    for name, expected in report["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"benchmark source changed during the sweep: {name}")


def gpu_snapshot(path):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu", "--format=csv"],
        capture_output=True,
        text=True,
        check=True,
    )
    path.write_text(result.stdout)


def preflight(args):
    if platform.system() != "Linux":
        raise RuntimeError("Run this benchmark on Linux with an NVIDIA GPU; there is no CPU fallback")
    for executable in ("ss", "setsid", "curl", "ps", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"missing {executable}; see benchmark/iris/README.md for system prerequisites")
    tools = prepare_build_tools()
    if args.download_checkpoint:
        download_checkpoint(args.checkpoint)
    checksum = verify_checkpoint(args.checkpoint)
    if git_output("rev-parse", "HEAD", cwd=args.reference_root) != REFERENCE_COMMIT:
        raise ValueError(f"official source must be at {REFERENCE_COMMIT}")
    if git_output("status", "--porcelain", "--untracked-files=no", cwd=args.reference_root):
        raise ValueError("official source has tracked modifications")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("benchmark requires a clean tracked checkout; do not edit it during a sweep")
    # Probe in a short-lived child, so the controller holds no CUDA context
    # alongside any timed arm. Model processes must never overlap on the GPU.
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,torch; from scripts.iris.qualify_cuda import require_cuda; "
            "require_cuda(); print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda, "
            "'gpu':torch.cuda.get_device_name(0)}))",
        ],
        text=True,
        cwd=ROOT,
    )
    return {"cuda_probe": json.loads(probe), "build_tools": tools, "checkpoint_sha256": checksum}


@contextmanager
def gpu_monitor(output):
    with (output / "gpu-monitor.csv").open("w") as log:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw",
                "--format=csv",
                "--loop=5",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            yield
        finally:
            process.terminate()
            process.wait(timeout=30)


def run(args, report, save):
    report["stage"] = "preflight"
    save()
    report["preflight"] = preflight(args)
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "mstar-ai",
            "torch",
            "numpy",
            "flashinfer-python",
            "triton",
        )
    }
    with gpu_monitor(args.output):
        run_cells(args, report, save)


def run_cells(args, report, save):
    from benchmark.iris.http import http_cells, server
    from benchmark.worker_phases.phases import parse_log, render, segment

    if args.stage in ("all", "audit"):
        report["batching_audit"] = {}
        for arm in ("eager", "captured"):
            report["stage"] = f"batching_audit_{arm}"
            save()
            output = args.output / f"audit-{arm}"
            gpu_snapshot(args.output / f"audit-{arm}-gpu-before.csv")
            with server(args.checkpoint, output, arm == "captured", phase_period=100, port=args.port) as url:
                cells = http_cells(url, output, args.warmup_waves, args.requests, audit=True)
            records = parse_log((output / "server.log").read_text())
            if not records:
                raise RuntimeError("worker phase audit emitted no parseable records; retain server.log for inspection")
            (output / "worker-phases.txt").write_text(render(segment(records)))
            report["batching_audit"][arm] = {
                "cells": cells,
                "phase_records": len(records),
                "world_multirow_observed": any(
                    cell["batching"]["components"]["world_model"]["two_row_steps"] > 0 for cell in cells
                ),
            }
            gpu_snapshot(args.output / f"audit-{arm}-gpu-after.csv")
            save()
    report["standalone"], report["http"] = [], []
    for round_index in range(args.rounds):
        # Reverse each successive round to expose ordering/load drift.
        if args.stage in ("all", "timings", "standalone"):
            order = ("official", "eager", "captured")
            for arm in order[:: 1 if round_index % 2 == 0 else -1]:
                label = f"round-{round_index + 1}-standalone-{arm}"
                report.setdefault("host_load", {})[label] = list(os.getloadavg())
                report["stage"] = label
                save()
                gpu_snapshot(args.output / f"{label}-gpu-before.csv")
                result_path = args.output / f"{label}.json"
                command = [
                    sys.executable,
                    "-m",
                    "benchmark.iris.standalone",
                    "--arm",
                    arm,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--reference-root",
                    str(args.reference_root),
                    "--output",
                    str(result_path),
                    "--warmup",
                    str(args.warmup_waves),
                    "--samples",
                    str(args.samples),
                ]
                with (args.output / f"{label}.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, MSTAR_PHASE_TIMING="0", OMP_NUM_THREADS="2"),
                    )
                report["standalone"].append({"round": round_index + 1, **json.loads(result_path.read_text())})
                gpu_snapshot(args.output / f"{label}-gpu-after.csv")
                save()
        if args.stage in ("all", "timings", "http"):
            for arm in ("eager", "captured")[:: 1 if round_index % 2 == 0 else -1]:
                label = f"round-{round_index + 1}-http-{arm}"
                report.setdefault("host_load", {})[label] = list(os.getloadavg())
                report["stage"] = label
                save()
                gpu_snapshot(args.output / f"{label}-gpu-before.csv")
                output = args.output / label
                with server(args.checkpoint, output, arm == "captured", port=args.port) as url:
                    cells = http_cells(url, output, args.warmup_waves, args.requests)
                report["http"].append({"round": round_index + 1, "arm": arm, "cells": cells})
                gpu_snapshot(args.output / f"{label}-gpu-after.csv")
                save()
    verify_sources(report)
    write_summary(report, args.output)
    report.update(status="passed", stage="complete")
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(".ci-results/iris/Breakout.pt"))
    parser.add_argument("--reference-root", type=Path, default=Path(".ci-results/iris/reference"))
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--stage", choices=("all", "audit", "timings", "standalone", "http"), default="all")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-waves", type=int, default=5)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args(argv)
    if min(args.rounds, args.samples, args.warmup_waves) < 1 or args.requests < 10:
        parser.error("rounds/samples/warmup-waves must be positive; requests must be at least 10 (5x concurrency 2)")
    args.output, args.checkpoint, args.reference_root = (
        p.resolve()
        for p in (
            args.output,
            args.checkpoint,
            args.reference_root,
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        *sorted((ROOT / "benchmark/iris").glob("*.py")),
        ROOT / "configs/iris.yaml",
        ROOT / "configs/iris_capture.yaml",
    ]
    report = {
        "status": "running",
        "stage": "initialization",
        "started_utc": datetime.now(UTC).isoformat(),
        "mstar_commit": git_output("rev-parse", "HEAD"),
        "reference_commit": REFERENCE_COMMIT,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "contract": {
            "standalone": "GPU inputs; reset/initial decode, H actions, RGB24/reward/end D2H per frame, teardown",
            "http": "PNG upload through native binary stream; closed loop; phase timing disabled in timed boots",
            "precision": "FP32, highest matmul, TF32 disabled; torch.compile disabled in every arm",
            "sampling": "official Categorical / eager FlashInfer / captured fused_sample; draws can differ by backend",
            "workload": "Breakout supplied actions, synthetic RGB64 PNG variants 0/1; H=1/20 standalone, H=20 HTTP",
            "comparison_scope": "compare rates only within the same scope and B or concurrency; no vLLM-Omni arm",
        },
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"IRIS benchmark: {report['stage']}", flush=True)

    try:
        run(args, report, save)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        print(report["traceback"])
        return 1
    print(f"PASSED; inspect {args.output / 'report.json'} and raw metrics/logs before drawing conclusions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
