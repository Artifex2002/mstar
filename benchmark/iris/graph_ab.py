"""Qualify the async decoder and A/B its graph against the serial baseline."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from benchmark.iris.http import http_cells, server
from benchmark.iris.run import gpu_monitor, gpu_snapshot, preflight, verify_sources
from scripts.iris.qualify_cuda import git_output

ROOT = Path(__file__).resolve().parents[2]


def qualified(path, commit, checkpoint_sha256):
    for arm in ("eager", "captured"):
        report = json.loads((path / arm / "report.json").read_text())
        if report["status"] != "passed" or report["mstar_commit"] != commit or not report.get("decoder_pipeline"):
            raise ValueError(f"{arm}: need a passed decoder-pipeline server qualification on this exact commit")
        if (report["capture"] != "disabled") != (arm == "captured"):
            raise ValueError(f"{arm}: qualification used the wrong capture mode")
        if report["working_tree_status"] or report["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError(f"{arm}: qualification requires clean sources and the same checkpoint")


def write_summary(report, output):
    with (output / "summary.tsv").open("w") as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(["round", "capture", "graph", "concurrency", "fps", "p50_ms", "p95_ms", "first_frame_p50_ms"])
        for boot in report["cells"]:
            for cell in boot["results"]:
                writer.writerow([
                    boot["round"], "captured" if boot["captured"] else "eager",
                    "pipeline" if boot["decoder_pipeline"] else "serial", cell["concurrency"],
                    cell["frames_per_second"], cell["latency_ms"]["p50"], cell["latency_ms"]["p95"],
                    cell["first_frame_ms"]["p50"],
                ])


def experiment(args, report, save):
    report["stage"] = "preflight"
    save()
    report["preflight"] = preflight(args)
    report["versions"] = {
        name: importlib.metadata.version(name) for name in ("torch", "flashinfer-python", "triton", "numpy")
    }
    qualification = args.qualification or args.output / "qualification"
    if args.stage in ("qualify", "all"):
        for arm in ("eager", "captured"):
            report["stage"] = f"qualify_pipeline_{arm}"
            save()
            qualification.mkdir(parents=True, exist_ok=True)
            config = ROOT / "configs" / ("iris_capture_pipeline.yaml" if arm == "captured" else "iris_pipeline.yaml")
            with (qualification / f"{arm}.log").open("w") as log:
                subprocess.run([
                    sys.executable, "-m", "scripts.iris.qualify_server", "--config", str(config),
                    "--checkpoint", str(args.checkpoint), "--output", str(qualification / arm),
                    "--startup-timeout", "1800", "--request-timeout", "600",
                ], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, MSTAR_PHASE_TIMING="0", OMP_NUM_THREADS="2"))
    qualified(qualification, report["mstar_commit"], report["preflight"]["checkpoint_sha256"])
    report["qualification"] = str(qualification)
    report["cells"] = []
    if args.stage in ("audit", "measure", "all"):
        with gpu_monitor(args.output):
            if args.stage in ("audit", "all"):
                from benchmark.worker_phases.phases import parse_log, render, segment

                report["audit"] = []
                for captured, pipeline in ((False, False), (False, True), (True, False), (True, True)):
                    arm = f"{'captured' if captured else 'eager'}-{'pipeline' if pipeline else 'serial'}"
                    report["stage"] = f"audit-{arm}"
                    save()
                    output = args.output / f"audit-{arm}"
                    with server(
                        args.checkpoint, output, captured, phase_period=100, port=args.port, decoder_pipeline=pipeline
                    ) as url:
                        cells = http_cells(url, output, args.warmup_waves, args.requests, audit=True)
                    records = parse_log((output / "server.log").read_text())
                    if not records:
                        raise RuntimeError("missing worker phase records; retain server.log")
                    (output / "worker-phases.txt").write_text(render(segment(records)))
                    report["audit"].append({"arm": arm, "phase_records": len(records), "results": cells})
                    save()
            for round_index in range(args.rounds if args.stage in ("measure", "all") else 0):
                arms = ((False, False), (False, True), (True, False), (True, True))
                for captured, pipeline in arms[::1 if round_index % 2 == 0 else -1]:
                    arm = f"{'captured' if captured else 'eager'}-{'pipeline' if pipeline else 'serial'}"
                    label = f"round-{round_index + 1}-{arm}"
                    report["stage"] = label
                    save()
                    gpu_snapshot(args.output / f"{label}-gpu-before.csv")
                    output = args.output / label
                    with server(args.checkpoint, output, captured, port=args.port, decoder_pipeline=pipeline) as url:
                        cells = http_cells(url, output, args.warmup_waves, args.requests)
                    report["cells"].append({
                        "round": round_index + 1, "captured": captured, "decoder_pipeline": pipeline,
                        "host_load_after": list(os.getloadavg()), "results": cells,
                    })
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
    parser.add_argument("--stage", choices=("qualify", "audit", "measure", "all"), default="qualify")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-waves", type=int, default=5)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args(argv)
    if args.stage in ("audit", "measure") and args.qualification is None:
        parser.error("--stage audit/measure requires --qualification from this exact commit")
    if args.rounds < 1 or args.requests < 10 or args.warmup_waves < 1:
        parser.error("positive rounds/warmup and at least ten requests required")
    for key in ("output", "checkpoint", "reference_root", "qualification"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        *sorted((ROOT / "mstar/model/iris").glob("*.py")),
        *sorted((ROOT / "benchmark/iris").glob("*.py")),
        *sorted((ROOT / "configs").glob("iris*.yaml")),
        *sorted((ROOT / "scripts/iris").glob("*.py")),
    ]
    report = {
        "status": "running", "stage": "initialization", "started_utc": datetime.now(UTC).isoformat(),
        "mstar_commit": git_output("rev-parse", "HEAD"),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hypothesis": "Decoupling generation from RGB decoding may improve scheduling and serving; unmeasured",
        "contract": "same single GPU, checkpoint, H20, actions, precision, kernels and capture mode per graph A/B",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"IRIS graph experiment: {report['stage']}", flush=True)

    try:
        experiment(args, report, save)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        print(report["traceback"])
        return 1
    print(f"PASSED; send {args.output / 'report.json'} with its per-server logs and metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
