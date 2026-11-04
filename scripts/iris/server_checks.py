"""HTTP stream and cleanup checks for a real IRIS server; no timing claims."""

import hashlib
import io
import json
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import requests
from PIL import Image

from mstar.client.media import BINARY_STREAM_MEDIA_TYPE, iter_binary_frames
from mstar.client.types import VideoFrameChunk


@dataclass(frozen=True)
class Case:
    request_id: str
    actions: tuple[int, ...]
    seed: int
    image_variant: int = 0


class ServerRefusalError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def image_png(variant):
    pixels = ((np.arange(3 * 64 * 64).reshape(64, 64, 3) + variant * 37) % 256).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


@dataclass
class StreamCheck:
    horizon: int
    digests: dict[str, list[str]] = field(
        default_factory=lambda: {key: [] for key in ("video_frame", "scalar", "tensor")}
    )
    rewards: list[float] = field(default_factory=list)
    endings: list[bool] = field(default_factory=list)
    frames: list[bytes] = field(default_factory=list, repr=False)
    disconnected: bool = False

    def add(self, chunk):
        modality, payload, metadata = chunk["modality"], chunk["bytes"], chunk["metadata"]
        if modality == "error":
            raise ServerRefusalError(payload.decode("utf-8", "replace"), metadata.get("status"))
        if modality == "video_frame":
            frame = VideoFrameChunk(payload, metadata)
            if (frame.width, frame.height, frame.frame_count, frame.fps) != (64, 64, 1, 15.0):
                raise AssertionError(f"unexpected IRIS frame metadata: {metadata}")
            if frame.frame_index != len(self.digests[modality]):
                raise AssertionError("frame indices are missing, duplicated or reordered")
            self.frames.append(payload)
        elif modality == "scalar":
            if len(payload) != 4:
                raise AssertionError("reward must be one little-endian float32")
            reward = struct.unpack("<f", payload)[0]
            if reward not in (-1.0, 0.0, 1.0):
                raise AssertionError(f"invalid reward {reward}")
            self.rewards.append(reward)
        elif modality == "tensor":
            if payload not in (b"\x00", b"\x01"):
                raise AssertionError("ending must be one boolean byte")
            self.endings.append(payload == b"\x01")
        else:
            raise AssertionError(f"unexpected output modality {modality!r}")
        self.digests[modality].append(hashlib.sha256(payload).hexdigest())
        if len(self.digests[modality]) > self.horizon:
            raise AssertionError(f"surplus {modality} output beyond the action horizon")

    def finish(self):
        counts = {key: len(values) for key, values in self.digests.items()}
        if any(count != self.horizon for count in counts.values()):
            raise AssertionError(f"incomplete stream: expected {self.horizon} of each output, got {counts}")
        return self

    def report(self):
        return {
            "counts": {key: len(value) for key, value in self.digests.items()},
            "digests": self.digests,
            "rewards": self.rewards,
            "endings": self.endings,
            "client_disconnected": self.disconnected,
        }


def generate(url, case, timeout=600.0, abort_after_frame=False):
    data = {
        "input_modalities": "image",
        "output_modalities": "video_frame,scalar,tensor",
        "streaming": "true",
        "request_id": case.request_id,
        "model_kwargs": json.dumps({"actions": list(case.actions), "seed": case.seed}),
    }
    headers = {"Accept": BINARY_STREAM_MEDIA_TYPE, "Accept-Encoding": "identity"}
    with requests.post(
        f"{url}/generate",
        data=data,
        files={"files": ("seed.png", image_png(case.image_variant), "image/png")},
        headers=headers,
        stream=True,
        timeout=(10.0, timeout),
    ) as response:
        if not response.ok:
            raise ServerRefusalError(response.text[:4096], response.status_code)
        if not response.headers.get("Content-Type", "").startswith(BINARY_STREAM_MEDIA_TYPE):
            raise AssertionError("server did not negotiate the native binary stream")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise AssertionError("binary stream must not be compressed")
        check = StreamCheck(len(case.actions))
        for chunk in iter_binary_frames(response.raw):
            check.add(chunk)
            if abort_after_frame and check.digests["video_frame"]:
                check.disconnected = True
                return check
        return check.finish()


def compare_streams(actual, expected):
    if actual.rewards != expected.rewards or actual.endings != expected.endings:
        raise AssertionError("seeded reward/ending sequence differs from the local eager baseline")
    if len(actual.frames) != len(expected.frames):
        raise AssertionError("seeded frame count differs from the local eager baseline")
    maximum = 0
    for index, (left, right) in enumerate(zip(actual.frames, expected.frames, strict=True)):
        error = np.abs(np.frombuffer(left, np.uint8).astype(np.int16) - np.frombuffer(right, np.uint8)).max().item()
        if error > 1:
            raise AssertionError(f"frame {index}: RGB24 differs from the local eager baseline by {error}")
        maximum = max(maximum, error)
    return {"max_rgb24_error": maximum, "rewards_endings": "exact"}


def metric_snapshot(text):
    from prometheus_client.parser import text_string_to_metric_families

    samples = [
        sample
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.labels.get("model_name") == "iris"
    ]

    def values(name, **labels):
        return [
            sample.value
            for sample in samples
            if sample.name == name and all(sample.labels.get(key) == value for key, value in labels.items())
        ]

    api = values("mstar_requests_in_flight")
    conductor = {state: values("mstar_conductor_requests", state=state) for state in ("waiting", "running", "draining")}
    kv = values("mstar_kv_pages_used", resource="world_kv")
    cached = values("mstar_kv_pages_cached", resource="world_kv")
    age = values("mstar_conductor_status_age_seconds")
    pending = values("mstar_component_pending_steps")
    complete = bool(api and all(conductor.values()) and kv and cached and age and pending)
    idle = (
        complete
        and all(
            value == 0
            for value in [*api, *kv, *cached, *pending, *(value for rows in conductor.values() for value in rows)]
        )
        and max(age) < 3.0
    )
    batch_counts = values("mstar_batch_size_count")
    single_buckets = values("mstar_batch_size_bucket", le="1.0")
    return {
        "complete": complete,
        "idle": bool(idle),
        "api_in_flight": api,
        "conductor": conductor,
        "kv_pages_used": kv,
        "kv_pages_cached": cached,
        "status_age_seconds": age,
        "success": sum(values("mstar_requests_total", outcome="success")),
        "aborts": sum(values("mstar_requests_total", outcome="abort")),
        "steps_with_more_than_one_row": sum(batch_counts) - sum(single_buckets),
    }


def metrics(url):
    response = requests.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    return response.text, metric_snapshot(response.text)


def wait_idle(url, minimum_success=0, minimum_aborts=0, timeout=60.0, stable_seconds=3.0):
    deadline, since = time.monotonic() + timeout, None
    latest = {}
    while time.monotonic() < deadline:
        text, latest = metrics(url)
        if latest["idle"] and latest["success"] >= minimum_success and latest["aborts"] >= minimum_aborts:
            since = time.monotonic() if since is None else since
            if time.monotonic() - since >= stable_seconds:
                return text, latest
        else:
            since = None
        time.sleep(0.25)
    raise TimeoutError(f"server did not drain requests/KV pages: {latest}")
