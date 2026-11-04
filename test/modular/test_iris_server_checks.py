"""Protocol and qualification-driver checks; full CUDA serving runs separately."""

import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
import yaml

from scripts.iris import server_checks
from scripts.iris.qualify_server import local_baseline, main, prepare_server_config, save_arrays
from scripts.iris.server_checks import (
    Case,
    ServerRefusalError,
    StreamCheck,
    compare_streams,
    generate,
    metric_snapshot,
    wait_idle,
)


def frame(index=0):
    return {
        "modality": "video_frame",
        "bytes": bytes(64 * 64 * 3),
        "metadata": {
            "width": 64,
            "height": 64,
            "fps": 15.0,
            "pixel_format": "rgb24",
            "frame_index": index,
            "frame_count": 1,
        },
    }


def chunks(index=0):
    return [
        frame(index),
        {"modality": "scalar", "bytes": struct.pack("<f", -1.0), "metadata": {}},
        {"modality": "tensor", "bytes": b"\x01", "metadata": {}},
    ]


def response(chunks):
    body = b"".join(
        json.dumps(
            {"modality": chunk["modality"], "nbytes": len(chunk["bytes"]), "metadata": chunk["metadata"]}
        ).encode()
        + b"\n"
        + chunk["bytes"]
        for chunk in chunks
    )
    reply = requests.Response()
    reply.status_code = 200
    reply.headers["Content-Type"] = "application/vnd.mstar.frames"
    reply.raw = io.BytesIO(body)
    return reply


def test_http_request_preserves_scalar_and_ending_bytes(monkeypatch):
    reply, sent = response([*chunks(0), *chunks(1)]), {}

    def post(url, **kwargs):
        sent.update(url=url, **kwargs)
        return reply

    monkeypatch.setattr(requests, "post", post)
    result = generate("http://localhost:1", Case("request", (0, 3), 42))
    assert result.rewards == [-1.0, -1.0]
    assert result.endings == [True, True]
    assert result.report()["counts"] == {"video_frame": 2, "scalar": 2, "tensor": 2}
    assert json.loads(sent["data"]["model_kwargs"]) == {"actions": [0, 3], "seed": 42}
    assert sent["data"]["request_id"] == "request"
    assert sent["files"]["files"][2] == "image/png"
    assert reply.raw.closed


def test_disconnect_closes_the_response_after_a_frame(monkeypatch):
    reply = response(chunks())
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply)
    result = generate("http://localhost:1", Case("request", (0,) * 20, 42), abort_after_frame=True)
    assert result.disconnected
    assert len(result.frames) == 1
    assert reply.raw.closed


def test_server_error_is_not_confused_with_clean_end_of_stream(monkeypatch):
    reply = response([{"modality": "error", "bytes": b"invalid action", "metadata": {"status": 400}}])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply)
    with pytest.raises(ServerRefusalError, match="invalid action") as exc:
        generate("http://localhost:1", Case("request", (4,), 42))
    assert exc.value.status == 400


def test_incomplete_stream_is_not_a_success(monkeypatch):
    reply = response(chunks())
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply)
    with pytest.raises(AssertionError, match="incomplete stream"):
        generate("http://localhost:1", Case("request", (0, 1), 42))


def test_seeded_comparison_detects_pixel_and_reward_divergence():
    expected, actual = StreamCheck(1), StreamCheck(1)
    for chunk in chunks():
        expected.add(chunk)
        actual.add(chunk)
    assert compare_streams(actual, expected)["max_rgb24_error"] == 0
    actual.frames[0] = b"\x02" + actual.frames[0][1:]
    with pytest.raises(AssertionError, match="frame 0"):
        compare_streams(actual, expected)
    actual.rewards[0] = 0.0
    with pytest.raises(AssertionError, match="reward/ending"):
        compare_streams(actual, expected)


@pytest.mark.parametrize(
    "chunk,message",
    [
        (frame(1), "indices"),
        ({"modality": "scalar", "bytes": struct.pack("<f", float("nan")), "metadata": {}}, "reward"),
        ({"modality": "scalar", "bytes": b"123", "metadata": {}}, "float32"),
        ({"modality": "tensor", "bytes": b"\x02", "metadata": {}}, "boolean"),
    ],
)
def test_bad_output_schema_is_rejected(chunk, message):
    with pytest.raises(AssertionError, match=message):
        StreamCheck(1).add(chunk)


def metric_text(used=0, age=0.1):
    return f"""mstar_requests_in_flight{{model_name="iris"}} 0
mstar_conductor_requests{{model_name="iris",state="waiting"}} 0
mstar_conductor_requests{{model_name="iris",state="running"}} 0
mstar_conductor_requests{{model_name="iris",state="draining"}} 0
mstar_kv_pages_used{{model_name="iris",component="world_model",resource="world_kv"}} {used}
mstar_kv_pages_cached{{model_name="iris",component="world_model",resource="world_kv"}} 0
mstar_conductor_status_age_seconds{{model_name="iris"}} {age}
mstar_component_pending_steps{{model_name="iris",component="world_model"}} 0
mstar_requests_total{{model_name="iris",outcome="success"}} 3
mstar_requests_total{{model_name="iris",outcome="abort"}} 1
mstar_batch_size_bucket{{model_name="iris",component="world_model",le="1.0"}} 8
mstar_batch_size_count{{model_name="iris",component="world_model"}} 10
"""


def test_metric_snapshot_requires_request_and_kv_cleanup():
    pytest.importorskip("prometheus_client")
    result = metric_snapshot(metric_text())
    assert result["complete"] and result["idle"]
    assert result["success"] == 3 and result["aborts"] == 1
    assert result["steps_with_more_than_one_row"] == 2
    assert not metric_snapshot(metric_text(used=1))["idle"]
    assert not metric_snapshot(metric_text(age=4.0))["idle"]
    assert not metric_snapshot("")["idle"]
    missing = "\n".join(line for line in metric_text().splitlines() if not line.startswith("mstar_kv_pages_used"))
    assert not metric_snapshot(missing)["complete"]


def test_idle_requires_a_stable_window_after_the_last_nonidle_sample(monkeypatch):
    clock, calls = [0.0], []

    def snapshot(url):
        clock[0] += 1
        calls.append(url)
        return "metrics", {"idle": len(calls) != 2, "success": 3, "aborts": 1}

    monkeypatch.setattr(server_checks, "metrics", snapshot)
    monkeypatch.setattr(server_checks.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(server_checks.time, "sleep", lambda seconds: None)
    _, result = wait_idle("http://localhost:1", minimum_success=3, minimum_aborts=1, stable_seconds=2)
    assert result["idle"]
    assert len(calls) == 5


def test_server_command_reports_missing_cuda_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "server"
    assert main(["--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["stage"] == "cuda_and_native_preflight"
    assert not (output / "server.log").exists()


def test_real_server_deployment_preflight_accepts_shipped_yaml_without_weights(tmp_path, monkeypatch):
    import triton.language as tl

    # The macOS conftest stub lacks this import-time kernel annotation. No
    # kernel runs during validation; leave real Triton types unchanged.
    if not hasattr(tl, "int64"):
        monkeypatch.setattr(tl, "int64", int, raising=False)

    def no_load(*args, **kwargs):
        raise AssertionError("deployment validation must not load checkpoint tensors")

    monkeypatch.setattr(torch, "load", no_load)
    checkpoint = tmp_path / "not-downloaded.pt"
    config_path = prepare_server_config(checkpoint, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    assert config["max_seq_len"] == 340
    assert config["max_concurrent_requests"] == 2
    assert config["model_kwargs"]["checkpoint_path"] == str(checkpoint)
    assert not checkpoint.exists()


CACHE = Path(__file__).resolve().parents[2] / ".ci-results/iris/Breakout.pt"


@pytest.mark.skipif(not CACHE.is_file(), reason="pinned checkpoint is not cached")
def test_baseline_uses_the_real_image_loader_and_output_serialization(tmp_path, monkeypatch):
    from mstar.engine.resources.sampler.resource import SamplerResource

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(SamplerResource, "sample", lambda self, rids, logits: logits.argmax(-1))
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = local_baseline(CACHE, Case("baseline", (0, 1), 17), tmp_path / "seed.png", device="cpu")
        assert result.report()["counts"] == {"video_frame": 2, "scalar": 2, "tensor": 2}
        save_arrays(tmp_path / "outputs.npz", result)
        with np.load(tmp_path / "outputs.npz", allow_pickle=False) as arrays:
            assert arrays["frames"].shape == (2, 64, 64, 3)
    finally:
        torch.set_num_threads(previous)
