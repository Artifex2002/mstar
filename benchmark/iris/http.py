"""Closed-loop HTTP measurements and untimed worker batching observations."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Barrier

import requests

from benchmark.iris.common import batching_delta, summarize
from mstar.client.media import BINARY_STREAM_MEDIA_TYPE, iter_binary_frames
from scripts.iris.qualify_server import prepare_server_config
from scripts.iris.server_checks import StreamCheck, image_png, wait_idle

ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = ROOT / ".claude/skills/benchmarking/scripts/server.sh"
# RGB64, H<=20 and two requests need small tensor transfers. Ceilings are per
# entity: one GPU worker plus the API data worker can grow to 128 MiB total.
ARENA_ENV = {
    "MSTAR_SHM_ARENA": "1",
    "MSTAR_SHM_ARENA_SEGMENT_MB": "16",
    "MSTAR_SHM_ARENA_MAX_SEGMENTS": "4",
    "MSTAR_SHM_ARENA_PIN_MAX_MB": "128",
}


@dataclass(frozen=True)
class RequestResult:
    seconds: float
    first_frame_seconds: float
    frames: int
    output_bytes: int


def request(url, index, horizon, png, timeout, variant=0):
    data = {
        "input_modalities": "image",
        "output_modalities": "video_frame,scalar,tensor",
        "streaming": "true",
        "request_id": f"iris-benchmark-{index}",
        "model_kwargs": json.dumps(
            {"actions": [(variant + step) % 4 for step in range(horizon)], "seed": 17 + variant}
        ),
    }
    start = time.perf_counter()
    first_frame, chunks = None, []
    with requests.post(
        url + "/generate",
        data=data,
        files={"files": ("seed.png", png, "image/png")},
        headers={"Accept": BINARY_STREAM_MEDIA_TYPE, "Accept-Encoding": "identity"},
        stream=True,
        timeout=(10, timeout),
    ) as response:
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").startswith(BINARY_STREAM_MEDIA_TYPE):
            raise AssertionError("server did not negotiate native binary streaming")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise AssertionError("compressed response would change the benchmark contract")
        for chunk in iter_binary_frames(response.raw):
            if first_frame is None and chunk["modality"] == "video_frame":
                first_frame = time.perf_counter() - start
            chunks.append(chunk)
        elapsed = time.perf_counter() - start
    # Validate every response, but keep hashing/schema checks outside latency.
    check = StreamCheck(horizon)
    for chunk in chunks:
        check.add(chunk)
    check.finish()
    if first_frame is None:
        raise AssertionError("response had no video frames")
    return RequestResult(elapsed, first_frame, horizon, sum(len(chunk["bytes"]) for chunk in chunks))


def closed_loop(url, concurrency, count, horizon, timeout=600):
    if concurrency < 1 or count < concurrency or horizon not in (1, 20):
        raise ValueError("need positive concurrency, at least one full wave and horizon 1 or 20")
    # The same two PNG/action variants are used in standalone B=1/B=2.
    images = [image_png(index % concurrency) for index in range(count)]
    barrier = Barrier(concurrency)

    def one(index):
        if index < concurrency:
            barrier.wait(timeout=30)
        return request(url, index, horizon, images[index], timeout, variant=index % concurrency)

    with ThreadPoolExecutor(concurrency) as pool:
        start = time.perf_counter()
        futures = [pool.submit(one, index) for index in range(count)]
        results = [future.result() for future in futures]  # any failure invalidates the cell
        elapsed = time.perf_counter() - start
    report = summarize([r.seconds for r in results], count * horizon, elapsed)
    report.update(
        requests_per_second=count / elapsed,
        first_frame_ms=summarize([r.first_frame_seconds for r in results], count)["latency_ms"],
        output_bytes=sum(r.output_bytes for r in results),
        requests=[asdict(r) for r in results],
    )
    return report


@contextmanager
def server(checkpoint, output, captured, phase_period=0, port=8123, decoder_pipeline=False):
    output.mkdir(parents=True, exist_ok=False)
    name = "iris_capture" if captured else "iris"
    deployment = ROOT / "configs" / f"{name}{'_pipeline' if decoder_pipeline else ''}.yaml"
    config = prepare_server_config(checkpoint, output, deployment)
    name = "iris-benchmark"
    env = dict(os.environ)
    env.update(
        SERVER_STATE_DIR=str(output / "server-state"),
        MSTAR_PHASE_TIMING=str(phase_period),
        MSTAR_REQUIRE_ACCELERATOR_GRAPHS="1" if captured else "0",
        TORCHINDUCTOR_CACHE_DIR=str(output / "inductor"),
        OMP_NUM_THREADS="2",
    )
    env.update(ARENA_ENV)
    env.pop("TRITON_CACHE_DIR", None)
    # Keep the versioned script responsible for listener ownership and teardown.
    start = [
        "bash",
        str(SERVER_SCRIPT),
        "start",
        name,
        str(port),
        "--timeout",
        "1800",
        "--log",
        str(output / "server.log"),
        "--",
        sys.executable,
        "-m",
        "mstar.cli.main",
        "serve",
        "iris",
        "--gpus",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--config",
        str(config),
        "--tensor-comm-protocol",
        "SHM",
        "--enable-metrics",
    ]
    (output / "launch.json").write_text(
        json.dumps(
            {
                "argv": start,
                "env": {
                    key: env[key] for key in (*ARENA_ENV, "MSTAR_PHASE_TIMING", "MSTAR_REQUIRE_ACCELERATOR_GRAPHS")
                },
            },
            indent=2,
        )
        + "\n"
    )
    try:
        subprocess.run(start, env=env, cwd=ROOT, check=True)
        yield f"http://127.0.0.1:{port}"
    finally:
        if (output / "server-state" / f"{name}.pid").exists():
            subprocess.run(["bash", str(SERVER_SCRIPT), "stop", name], env=env, cwd=ROOT, check=True)


def http_cells(url, output, warmup_waves, count, audit=False):
    results = []
    for concurrency in (1, 2):
        closed_loop(url, concurrency, concurrency * warmup_waves, 20)
        before, initial = wait_idle(url)
        (output / f"c{concurrency}-before.prom").write_text(before)
        result = closed_loop(url, concurrency, count, 20)
        after, cleanup = wait_idle(url, minimum_success=initial["success"] + count)
        (output / f"c{concurrency}-after.prom").write_text(after)
        batching = batching_delta(before, after)
        world = batching["components"].get("world_model", {})
        if not world.get("steps"):
            raise RuntimeError(
                "worker batch histogram absent or unchanged; inspect raw .prom files before timing claims"
            )
        result.update(concurrency=concurrency, horizon=20, batching=batching, cleanup=cleanup)
        if audit:
            result["timing_scope"] = (
                "diagnostic only: phase instrumentation enabled; excluded from throughput comparisons"
            )
        results.append(result)
        (output / "results.json").write_text(json.dumps({"status": "passed", "cells": results}, indent=2) + "\n")
    return results
