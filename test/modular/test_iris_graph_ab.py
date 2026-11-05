"""Correctness gates and controlled process order for the graph experiment."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmark.iris import graph_ab


def qualification(path, **updates):
    for arm in ("eager", "captured"):
        target = path / arm
        target.mkdir(parents=True)
        report = {
            "status": "passed", "mstar_commit": "commit", "decoder_pipeline": True,
            "capture": "disabled" if arm == "eager" else "enabled; every declared bucket required",
            "working_tree_status": "", "checkpoint_sha256": "checksum",
        }
        report.update(updates)
        (target / "report.json").write_text(json.dumps(report))


@pytest.mark.parametrize("changes", [
    {"status": "failed"}, {"mstar_commit": "old"}, {"decoder_pipeline": False},
    {"capture": "disabled"}, {"working_tree_status": " M model.py"}, {"checkpoint_sha256": "other"},
])
def test_qualification_gate_rejects_failed_stale_dirty_or_wrong_arms(tmp_path, changes):
    qualification(tmp_path, **changes)
    with pytest.raises(ValueError):
        graph_ab.qualified(tmp_path, "commit", "checksum")


def test_graph_sweep_uses_all_four_arms_in_reverse_order_and_disables_phase_timing(tmp_path, monkeypatch):
    qualification(tmp_path / "qualification")
    args = SimpleNamespace(
        stage="measure", qualification=tmp_path / "qualification", output=tmp_path,
        checkpoint=tmp_path / "Breakout.pt", rounds=2, port=8123, warmup_waves=5, requests=10,
    )
    monkeypatch.setattr(graph_ab, "preflight", lambda args: {"checkpoint_sha256": "checksum"})
    monkeypatch.setattr(graph_ab.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(graph_ab, "verify_sources", lambda report: None)
    monkeypatch.setattr(graph_ab, "gpu_snapshot", lambda path: None)

    @contextmanager
    def monitor(output):
        yield

    arms = []

    @contextmanager
    def server(checkpoint, output, captured, phase_period=0, port=8123, decoder_pipeline=False):
        assert phase_period == 0
        arms.append((captured, decoder_pipeline))
        yield "http://unused"

    monkeypatch.setattr(graph_ab, "gpu_monitor", monitor)
    monkeypatch.setattr(graph_ab, "server", server)
    monkeypatch.setattr(graph_ab, "http_cells", lambda *args: [{
        "concurrency": 1, "frames_per_second": 20, "latency_ms": {"p50": 1000, "p95": 1000},
        "first_frame_ms": {"p50": 100},
    }])
    report = {"mstar_commit": "commit"}
    graph_ab.experiment(args, report, lambda: None)
    expected = [(False, False), (False, True), (True, False), (True, True)]
    assert arms == expected + list(reversed(expected))
    assert report["status"] == "passed" and len(report["cells"]) == 8
    assert len((tmp_path / "summary.tsv").read_text().splitlines()) == 9
    assert "audit" not in report
