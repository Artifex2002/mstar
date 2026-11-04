"""Qualify the real IRIS HTTP/conductor/native-worker pipeline on one CUDA GPU."""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import numpy as np
import requests
import torch
import yaml

from benchmark.eval.server import ManagedServer
from scripts.iris.qualify_cuda import git_output, prepare_build_tools, require_cuda
from scripts.iris.reference_harness import download_checkpoint, verify_checkpoint
from scripts.iris.server_checks import (
    Case,
    ServerRefusalError,
    StreamCheck,
    compare_streams,
    generate,
    image_png,
    metrics,
    wait_idle,
)

ROOT = Path(__file__).resolve().parents[2]


def prepare_server_config(checkpoint, output):
    from mstar.model.iris.iris_model import IrisModel
    from mstar.validation import validate_deployment

    config = yaml.safe_load((ROOT / "configs/iris.yaml").read_text())
    config["model_kwargs"]["checkpoint_path"] = str(checkpoint.resolve())
    config["max_concurrent_requests"] = 2
    config_path = output / "server.yaml"
    config_path.write_text(yaml.safe_dump(config))
    # This is the conductor's real validation path, without loading weights or
    # starting workers. Catch invalid deployments before the GPU baseline.
    validate_deployment(IrisModel(**config["model_kwargs"]), config, str(config_path))
    return config_path


def save_arrays(path, stream):
    frames = np.frombuffer(b"".join(stream.frames), np.uint8).reshape(-1, 64, 64, 3)
    np.savez_compressed(
        path, frames=frames, rewards=np.asarray(stream.rewards, np.float32), endings=np.asarray(stream.endings, bool)
    )


@torch.inference_mode()
def local_baseline(checkpoint, case, image_path, device="cuda"):
    from mstar.model.iris.config import IMAGE_DECODER, IMAGE_ENCODER, INITIAL_DECODER, WORLD_MODEL
    from mstar.model.iris.graph import frame_calls
    from mstar.model.iris.iris_model import IrisModel
    from scripts.iris.local_engine import LocalIrisEngine

    model = IrisModel(checkpoint_path=str(checkpoint), max_batch_size=1)
    modules = {
        node: model.get_submodule(node, device) for node in (IMAGE_ENCODER, WORLD_MODEL, IMAGE_DECODER, INITIAL_DECODER)
    }
    image_path.write_bytes(image_png(case.image_variant))
    loaded = model.load_image(str(image_path), "cpu").data
    prepared = model.process_prompt(
        None,
        ["image"],
        ["video_frame", "scalar", "tensor"],
        tensors={"image_inputs": [loaded]},
        actions=list(case.actions),
    )
    image = prepared.new_input_tensors["initial_image"][0].to(device)
    actions = prepared.new_input_tensors["actions"][0].to(device)
    check = StreamCheck(len(case.actions))
    with LocalIrisEngine(model, device) as engine:
        engine.open(1, len(case.actions), seed=case.seed)
        carry = engine.run(modules[IMAGE_ENCODER], {1: {"initial_image": [image]}})
        engine.run(modules[INITIAL_DECODER], carry)
        for frame in range(len(case.actions)):
            for token in range(len(frame_calls(frame))):
                carry = engine.run(modules[WORLD_MODEL], {1: {**carry[1], "actions": [actions]}}, {1: (frame, token)})
            output = engine.run(modules[IMAGE_DECODER], carry)[1]
            for name, modality in (("frames", "video_frame"), ("reward", "scalar"), ("ending", "tensor")):
                metadata = (
                    {
                        "width": 64,
                        "height": 64,
                        "frame_count": 1,
                        "frame_index": frame,
                        "fps": 15.0,
                        "pixel_format": "rgb24",
                    }
                    if modality == "video_frame"
                    else {}
                )
                check.add(
                    {"modality": modality, "bytes": model.postprocess(output[name][0], modality), "metadata": metadata}
                )
    return check.finish()


def concurrent(url, cases, timeout):
    barrier = Barrier(len(cases))

    def one(case):
        barrier.wait(timeout=10)
        return generate(url, case, timeout)

    pool = ThreadPoolExecutor(len(cases))
    futures = {case.request_id: pool.submit(one, case) for case in cases}
    peak = 0
    deadline = time.monotonic() + timeout
    try:
        while not all(future.done() for future in futures.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError("concurrent HTTP requests did not finish")
            _, snapshot = metrics(url)
            peak = max(peak, sum(snapshot["api_in_flight"]))
            time.sleep(0.025)
        return {rid: future.result() for rid, future in futures.items()}, peak
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def suite(server, output, expected, replay, report, save, timeout):
    report["cases"] = {}
    success = 0

    def stage(name):
        report["stage"] = name
        save()

    def drained(name, aborts=0):
        text, snapshot = wait_idle(server.url, success, aborts)
        (output / f"{name}.prom").write_text(text)
        report.setdefault("cleanup", {})[name] = snapshot
        save()

    stage("http_warmup")
    warmup = generate(server.url, Case("iris-warmup", (0,), 7), timeout)
    report["cases"]["warmup"] = warmup.report()
    success += 1
    drained("warmup")

    for label in ("native_vs_local", "reused_request_id"):
        stage(label)
        result = generate(server.url, replay, timeout)
        save_arrays(output / f"{label}.npz", result)
        report["cases"][label] = {**result.report(), "local_comparison": compare_streams(result, expected)}
        success += 1
        drained(label)

    for label, horizons in (("concurrent", (20, 20)), ("request_departure", (20, 2))):
        stage(label)
        cases = [
            Case(f"iris-{label}-{index}", tuple((index + step) % 4 for step in range(horizon)), 30 + index, index)
            for index, horizon in enumerate(horizons)
        ]
        results, peak = concurrent(server.url, cases, timeout)
        if label == "concurrent" and peak < 2:
            raise AssertionError("the API never observed two requests in flight; concurrency was not exercised")
        report["cases"][label] = {
            "peak_api_in_flight": peak,
            "requests": {rid: result.report() for rid, result in results.items()},
        }
        success += len(cases)
        drained(label)

    stage("invalid_action")
    try:
        generate(server.url, Case("iris-invalid", (4,), 11), timeout)
    except ServerRefusalError as exc:
        if "action" not in str(exc).lower():
            raise AssertionError(f"invalid action failed without a validation message: {exc}") from exc
        report["cases"]["invalid_action"] = {"status": exc.status, "message": str(exc)}
    else:
        raise AssertionError("invalid action was accepted")
    drained("invalid_action")

    stage("client_disconnect")
    _, before = metrics(server.url)
    abort = generate(server.url, Case("iris-disconnect", tuple(range(4)) * 5, 71), timeout, abort_after_frame=True)
    report["cases"]["client_disconnect"] = abort.report()
    drained("client_disconnect", aborts=before["aborts"] + 1)

    stage("after_disconnect")
    result = generate(server.url, Case("iris-disconnect", (1,), 73), timeout)
    report["cases"]["after_disconnect"] = result.report()
    success += 1
    drained("after_disconnect", aborts=before["aborts"] + 1)
    requests.get(f"{server.url}/health", timeout=10).raise_for_status()


def qualify(args, report, save):
    report["stage"] = "cuda_and_native_preflight"
    save()
    require_cuda()
    report["build_tools"] = prepare_build_tools()
    import mstar_rust

    if platform.system() != "Linux":
        raise RuntimeError("Full-server qualification requires Linux (native eventfd/worker transport)")
    for name in ("GraphRuntime", "ZmqCommunicator", "TensorBookkeeping"):
        if not hasattr(mstar_rust, name):
            raise RuntimeError(f"rebuild rust/: extension is missing {name}")
    for module in ("mstar.api_server.entrypoint", "mstar.worker.worker", "prometheus_client"):
        __import__(module)
    report["hardware"] = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "rust_extension": mstar_rust.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
    }
    report["stage"] = "deployment_validation"
    save()
    config_path = prepare_server_config(args.checkpoint, args.output)
    if args.download_checkpoint:
        download_checkpoint(args.checkpoint)
    report["checkpoint_sha256"] = verify_checkpoint(args.checkpoint)
    args.checkpoint = args.checkpoint.resolve()
    replay = Case("iris-seeded-reuse", tuple(step % 4 for step in range(20)), 17)
    report["stage"] = "local_seeded_baseline"
    save()
    torch.set_num_threads(2)
    expected = local_baseline(args.checkpoint, replay, args.output / "seed.png")
    save_arrays(args.output / "local-baseline.npz", expected)
    report["local_baseline"] = expected.report()
    report["stage"] = "server_startup"
    save()
    with ManagedServer(
        "iris",
        config=str(config_path),
        python=sys.executable,
        cwd=ROOT,
        extra_args=["--enable-metrics"],
        log_path=args.output / "server.log",
        startup_timeout_s=args.startup_timeout,
    ) as server:
        report["server"] = server.info()
        try:
            suite(server, args.output, expected, replay, report, save, args.request_timeout)
        except Exception:
            try:
                (args.output / "failure-metrics.prom").write_text(metrics(server.url)[0])
            except requests.RequestException:
                pass
            raise
    report.update(status="passed", stage="complete", managed_server_shutdown="completed")
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(".ci-results/iris/Breakout.pt"))
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "stage": "initialization",
        "started_utc": datetime.now(UTC).isoformat(),
        "mstar_commit": git_output("rev-parse", "HEAD"),
        "working_tree_status": git_output("status", "--porcelain", "--untracked-files=no"),
        "purpose": "full-server eager FP32 correctness; no performance measurement",
        "capture": "disabled",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        qualify(args, report, save)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        print(report["traceback"])
        print(f"FAILED at {report['stage']}; send report.json and server.log from {args.output}")
        return 1
    print(f"PASSED; send {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
