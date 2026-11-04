"""Fixed-layout attention: numerical reference, padding and slot ownership."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.engine.resources import AttentionStep, Segment, StepContext
from mstar.engine.resources.attn.torch_reference import reference_attention
from mstar.engine.resources.attn.torch_static import TorchStaticPagedAttention, TorchStaticRaggedAttention
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import SequenceView
from mstar.engine.resources.step import BucketKey, SlotLease


def cfg():
    return PagedKVConfig(
        num_layers=1, num_kv_heads=2, num_qo_heads=4, head_dim=8, max_seq_len=12, page_size=4, max_num_pages=8
    )


def context(views, slot=0, bucket=None, preplan=False):
    ctx = StepContext(
        tuple(view.request_id for view in views),
        "walk",
        slot,
        False,
        is_preplan=preplan,
        plan_results={"kv": {"label": SimpleNamespace(views=views)}},
    )
    if bucket is not None:
        ctx.slot_lease = SlotLease(slot, bucket)
    return ctx


@pytest.mark.parametrize("queries", [1, 3])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_reference_different_lengths_and_poisoned_padding(queries, causal):
    torch.manual_seed(4)
    manager = TorchStaticPagedAttention("kv", "cpu", cfg())
    views = [SequenceView(1, "label", [3, 1], 5, queries), SequenceView(2, "label", [2], 3, queries)]
    bucket = BucketKey("walk", 3, 3 * queries)
    ctx = context(views, bucket=bucket)
    ctx.set_padded_rids([1, 2, -1])
    views.append(SequenceView(-1, "label", [0], queries, queries))
    manager.plan(AttentionStep(causal=causal), ctx)
    cache = torch.randn(8, 2, 4, 2, 8)
    cache[0].fill_(float("nan"))
    cache[1, :, 1:].fill_(float("nan"))
    cache[2, :, 3:].fill_(float("nan"))
    q = torch.randn(3 * queries, 4, 8)
    q[2 * queries :].fill_(float("nan"))
    actual = manager.run(q, "label", kv_cache_layer=cache)
    for index, view in enumerate(views[:2]):
        pages = cache[view.page_idxs]
        expected = reference_attention(
            q[index * queries : (index + 1) * queries],
            pages[:, 0].flatten(0, 1)[: view.length],
            pages[:, 1].flatten(0, 1)[: view.length],
            causal,
        )
        torch.testing.assert_close(actual[index * queries : (index + 1) * queries], expected, rtol=2e-6, atol=2e-6)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[2 * queries :]) == 0


def test_slot_preplan_does_not_overwrite_current_plan_and_reuses_addresses():
    manager = TorchStaticPagedAttention("kv", "cpu", cfg())
    views = [SequenceView(1, "label", [1], 2, 1)]
    bucket = BucketKey("walk", 1, 1)
    step = AttentionStep(causal=True)
    manager.plan(step, context(views, bucket=bucket))
    first = manager._current["label"]
    before = first.indices.clone()
    second = [SequenceView(2, "label", [3], 4, 1)]
    pre = context(second, 1, bucket, True)
    manager.plan(step, pre)
    assert manager._current["label"] is first
    assert torch.equal(first.indices, before)
    manager.plan(step, context(second, 1, bucket))
    assert manager._current["label"].indices.data_ptr() != first.indices.data_ptr()
    manager.plan(step, context(second, bucket=bucket))
    assert manager._current["label"] is first
    assert not torch.equal(first.indices, before)
    assert manager.force_double_buffer and manager.supports_preplan
    manager.plan(step, context(second, 1, bucket, True))
    manager.clear_preplan()
    manager.plan(step, context(views, 1, bucket))
    assert manager._current["label"].valid.sum() == 2


def test_paged_refuses_layout_changes_and_unequal_query_spans():
    manager = TorchStaticPagedAttention("kv", "cpu", cfg())
    step = AttentionStep(causal=True)
    views = [SequenceView(1, "label", [1], 3, 1)]
    ctx = context(views, bucket=BucketKey("walk", 1, 1))
    manager.plan(step, ctx)
    views[0] = views[0]._replace(to_compute=2)
    with pytest.raises(ValueError, match="layout changed"):
        manager.plan(step, ctx)
    views.append(views[0]._replace(request_id=2, to_compute=1))
    with pytest.raises(ValueError, match="uniform"):
        manager.plan(step, context(views))


@pytest.mark.parametrize("causal", [False, True])
def test_ragged_isolated_uniform_spans_and_preplan(causal):
    manager = TorchStaticRaggedAttention(0.25, "cpu")
    step = AttentionStep(segments=(Segment(1, "image", 3), Segment(2, "image", 3)), causal=causal)
    ctx = StepContext((1, 2), "walk", 0, False)
    manager.plan(step, ctx)
    q, k, v = (torch.randn(6, 2, 8) for _ in range(3))
    actual = manager.run(q, k, v, "image")
    expected = torch.cat([reference_attention(q[i : i + 3], k[i : i + 3], v[i : i + 3], causal, 0.25) for i in (0, 3)])
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    pre = StepContext((1,), "walk", 1, False, is_preplan=True, slot_lease=SlotLease(1, BucketKey("walk", 1, 1)))
    manager.plan(AttentionStep(segments=(Segment(1, "next", 2),)), pre)
    torch.testing.assert_close(manager.run(q, k, v, "image"), actual)
    pre.is_preplan = False
    manager.plan(step, pre)
    with pytest.raises(KeyError):
        manager.run(q, k, v, "image")


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (ROOT / ".ci-results/iris/Breakout.pt").is_file()
    or not (ROOT / ".ci-results/iris/context-refresh/trace.npz").is_file(),
    reason="pinned checkpoint/official trace not cached",
)
def test_static_backend_matches_full_official_batched_trajectory(monkeypatch):
    from scripts.iris.qualification import compare_trace

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with np.load(ROOT / ".ci-results/iris/context-refresh/trace.npz", allow_pickle=False) as trace:
            report = compare_trace(
                ROOT / ".ci-results/iris/Breakout.pt", dict(trace), config_path=ROOT / "configs/iris_capture.yaml"
            )
        assert report["world_batch_calls"] == 342
        assert report["decoded_frames"] == 40
    finally:
        torch.set_num_threads(threads)
