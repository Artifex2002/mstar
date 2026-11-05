"""Workload, statistics and worker batching evidence shared by the IRIS runs."""

import io
import math
from collections import defaultdict

import numpy as np
from PIL import Image

from scripts.iris.server_checks import image_png


def workload(batch, horizon):
    # Decode the exact PNGs used by the HTTP client, before standalone timing.
    images = np.stack(
        [
            np.asarray(Image.open(io.BytesIO(image_png(row))), dtype=np.float32).transpose(2, 0, 1) / 255
            for row in range(batch)
        ]
    )
    actions = ((np.arange(batch)[:, None] + np.arange(horizon)[None, :]) % 4).astype(np.int64)
    return images, actions


def summarize(seconds, frames, elapsed=None):
    if not seconds or any(not math.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError("timings must be finite positive seconds with no failed requests")
    wall = sum(seconds) if elapsed is None else elapsed
    if not math.isfinite(wall) or wall <= 0 or frames <= 0:
        raise ValueError("elapsed time and completed frame count must be positive")
    return {
        "completed": len(seconds),
        "failed": 0,
        "wall_seconds": wall,
        "latency_ms": {
            name: float(np.percentile(seconds, p) * 1000) for name, p in (("p50", 50), ("p95", 95), ("p99", 99))
        },
        "mean_latency_ms": float(np.mean(seconds) * 1000),
        "frames": frames,
        "frames_per_second": frames / wall,
    }


def batch_histograms(text):
    """Keep absent histograms distinct from observed single-row execution."""
    from prometheus_client.parser import text_string_to_metric_families

    rows = defaultdict(dict)
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.labels.get("model_name") != "iris" or not sample.name.startswith("mstar_batch_size_"):
                continue
            component = sample.labels["component"]
            key = sample.name.removeprefix("mstar_batch_size_")
            if key == "bucket":
                bound = float(sample.labels["le"])
                key = f"le_{bound:g}"
            if key in ("count", "sum", "le_1", "le_2"):
                rows[component][key] = rows[component].get(key, 0) + sample.value
    for component, row in rows.items():
        if set(row) != {"count", "sum", "le_1", "le_2"}:
            raise ValueError(f"incomplete batch histogram for {component}: {row}")
    return dict(rows)


def batching_delta(before, after):
    left, right = batch_histograms(before), batch_histograms(after)
    result = {}
    for component, values in right.items():
        delta = {key: value - left.get(component, {}).get(key, 0) for key, value in values.items()}
        if any(not math.isfinite(v) or v < 0 for v in delta.values()):
            raise ValueError(f"batch histogram reset during measurement: {component}")
        if not (delta["le_1"] <= delta["le_2"] <= delta["count"]):
            raise ValueError(f"invalid cumulative batch histogram: {component}")
        if delta["count"] != delta["le_2"]:
            raise ValueError(f"unexpected batch larger than qualified B=2: {component}")
        if delta["sum"] != 2 * delta["count"] - delta["le_1"]:
            raise ValueError(f"batch histogram row sum disagrees with its buckets: {component}")
        result[component] = {
            "steps": delta["count"],
            "row_steps": delta["sum"],
            "single_row_steps": delta["le_1"],
            "two_row_steps": delta["count"] - delta["le_1"],
            "mean_rows": delta["sum"] / delta["count"] if delta["count"] else None,
        }
    return {"available": bool(result), "components": result}
