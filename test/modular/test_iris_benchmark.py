"""CPU checks for benchmark accounting, HTTP errors and real eager Engine use."""

import io
import json
import struct
import time
from pathlib import Path
from threading import Lock

import numpy as np
import pytest
import requests
import torch

from benchmark.iris import http
from benchmark.iris.common import batching_delta, summarize, workload


def histogram(count, singles, rows, component="world_model"):
    labels = f'model_name="iris",component="{component}",graph_walk="imagine"'
    return (
        f"mstar_batch_size_count{{{labels}}} {count}\n"
        f"mstar_batch_size_sum{{{labels}}} {rows}\n"
        f'mstar_batch_size_bucket{{{labels},le="1.0"}} {singles}\n'
        f'mstar_batch_size_bucket{{{labels},le="2.0"}} {count}\n'
    )


def test_batching_deltas_distinguish_absence_single_rows_and_true_two_row_steps():
    assert batching_delta("", "") == {"available": False, "components": {}}
    before = histogram(10, 8, 12)
    after = histogram(15, 11, 19)
    assert batching_delta(before, after)["components"]["world_model"] == {
        "steps": 5,
        "row_steps": 7,
        "single_row_steps": 3,
        "two_row_steps": 2,
        "mean_rows": 1.4,
    }
    single = batching_delta("", histogram(5, 5, 5))
    assert single["available"] and single["components"]["world_model"]["two_row_steps"] == 0
    assert batching_delta("", histogram(5, 5, 5, "image_encoder"))["components"].keys() == {"image_encoder"}


@pytest.mark.parametrize(
    "before,after",
    [
        (histogram(10, 8, 12), histogram(1, 1, 1)),
        ("", histogram(3, 4, 3)),
        ("", histogram(3, 1, 3)),
        ("", histogram(3, 1, 5).splitlines()[0] + "\n"),
    ],
)
def test_invalid_or_reset_histograms_never_become_batching_evidence(before, after):
    with pytest.raises(ValueError):
        batching_delta(before, after)


def test_throughput_uses_wall_time_at_concurrency_not_sum_of_request_latencies():
    report = summarize([2, 2], frames=40, elapsed=2)
    assert report["frames_per_second"] == 20 and report["latency_ms"]["p50"] == 2000
    assert summarize([2, 2], frames=40)["frames_per_second"] == 10


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_samples_cannot_produce_plausible_rates(samples):
    with pytest.raises(ValueError):
        summarize(samples, frames=20)


def test_workload_preserves_http_pixels_action_ids_and_dtype():
    images, actions = workload(2, 20)
    assert images.shape == (2, 3, 64, 64) and images.dtype == np.float32
    assert actions.dtype == np.int64 and actions.shape == (2, 20)
    np.testing.assert_array_equal(actions[1], (np.arange(20) + 1) % 4)
    assert (images >= 0).all() and (images <= 1).all() and not np.array_equal(images[0], images[1])


def response(complete=True, status=200):
    chunks = [
        {
            "modality": "video_frame",
            "bytes": bytes(64 * 64 * 3),
            "metadata": {
                "width": 64,
                "height": 64,
                "fps": 15.0,
                "pixel_format": "rgb24",
                "frame_index": 0,
                "frame_count": 1,
            },
        }
    ]
    if complete:
        chunks.extend(
            [
                {"modality": "scalar", "bytes": struct.pack("<f", 0.0), "metadata": {}},
                {"modality": "tensor", "bytes": b"\x00", "metadata": {}},
            ]
        )
    body = b"".join(
        json.dumps({"modality": c["modality"], "metadata": c["metadata"], "nbytes": len(c["bytes"])}).encode()
        + b"\n"
        + c["bytes"]
        for c in chunks
    )
    result = requests.Response()
    result.status_code = status
    result.headers["Content-Type"] = "application/vnd.mstar.frames"
    result.raw = io.BytesIO(body)
    return result


def test_http_measurement_checks_entire_stream_and_sends_fixed_variant(monkeypatch):
    sent = {}

    def post(*args, **kwargs):
        sent.update(kwargs)
        return response()

    monkeypatch.setattr(requests, "post", post)
    result = http.request("http://localhost:1", 9, 1, b"png", 30, variant=1)
    assert 0 < result.first_frame_seconds <= result.seconds
    assert result.frames == 1 and result.output_bytes == 64 * 64 * 3 + 5
    assert json.loads(sent["data"]["model_kwargs"]) == {"actions": [1], "seed": 18}


@pytest.mark.parametrize("complete,status,error", [(False, 200, AssertionError), (True, 500, requests.HTTPError)])
def test_http_failure_is_not_counted_as_fast_inference(monkeypatch, complete, status, error):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response(complete, status))
    with pytest.raises(error):
        http.request("http://localhost:1", 0, 1, b"png", 30)


def test_closed_loop_bounds_active_requests_and_cycles_workload_variants(monkeypatch):
    lock, active, peak, variants = Lock(), 0, 0, []

    def request(url, index, horizon, png, timeout, variant):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            variants.append(variant)
        time.sleep(0.005)
        with lock:
            active -= 1
        return http.RequestResult(0.005, 0.001, horizon, 123)

    monkeypatch.setattr(http, "request", request)
    report = http.closed_loop("unused", 2, 10, 20)
    assert peak == 2 and active == 0
    assert variants.count(0) == variants.count(1) == 5
    assert report["completed"] == 10 and report["frames"] == 200 and report["failed"] == 0


def test_command_saves_preflight_failure_without_cpu_timing(tmp_path, monkeypatch):
    from benchmark.iris import run

    monkeypatch.setattr(run.platform, "system", lambda: "Darwin")
    output = tmp_path / "result"
    assert run.main(["--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "preflight"
    assert "no CPU fallback" in report["error"] and "standalone" not in report


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".ci-results/iris/Breakout.pt").is_file(),
    reason="pinned checkpoint not cached",
)
def test_standalone_adapter_runs_real_eager_engine_with_cpu_weights(monkeypatch):
    import triton.language as tl

    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)
    from benchmark.iris import standalone
    from mstar.engine.resources.sampler.resource import SamplerResource

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(standalone, "indexed_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(SamplerResource, "sample", lambda self, rids, logits: logits.argmax(-1))
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    checkpoint = Path(__file__).resolve().parents[2] / ".ci-results/iris/Breakout.pt"
    try:
        driver = standalone.EngineRollout(checkpoint, False)
        try:
            images, actions = [torch.from_numpy(value) for value in workload(2, 20)]
            first = driver.rollout(images, actions, 17)
            standalone.validate_records(first, 2, 20)
            from mstar.model.iris.config import IMAGE_DECODER, IMAGE_ENCODER, INITIAL_DECODER, WORLD_MODEL
            from mstar.model.iris.graph import frame_calls
            from scripts.iris.local_engine import LocalIrisEngine

            # Compare the new Engine adapter with the already-qualified direct
            # resource driver, not a copy of Engine's implementation.
            with LocalIrisEngine(driver.model, "cpu") as reference:
                for rid in (1, 2):
                    reference.open(rid, 20, 16 + rid)
                carry = reference.run(
                    driver.modules[IMAGE_ENCODER], {rid: {"initial_image": [images[rid - 1]]} for rid in (1, 2)}
                )
                reference.run(driver.modules[INITIAL_DECODER], carry)
                for frame in range(20):
                    for tick in range(len(frame_calls(frame))):
                        carry = reference.run(
                            driver.modules[WORLD_MODEL],
                            {rid: {**row, "actions": [actions[rid - 1]]} for rid, row in carry.items()},
                            {rid: (frame, tick) for rid in carry},
                        )
                    decoded = reference.run(driver.modules[IMAGE_DECODER], carry)
                    for rid in (1, 2):
                        for name in first[frame][rid]:
                            torch.testing.assert_close(first[frame][rid][name], decoded[rid][name][0], rtol=0, atol=0)
            second = driver.rollout(images, actions, 17)
            for left, right in zip(first, second, strict=True):
                for rid in left:
                    for name in left[rid]:
                        torch.testing.assert_close(left[rid][name], right[rid][name], rtol=0, atol=0)
            assert not driver.horizons
        finally:
            driver.cleanup()
    finally:
        torch.set_num_threads(previous)
