"""Correctness-only backend: explicit masks, segment isolation and eager refusal."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from mstar.engine.resources import AttentionStep, Segment, StepContext
from mstar.engine.resources.attn.torch_reference import TorchPagedAttention, TorchRaggedAttention, reference_attention


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2)])
@pytest.mark.parametrize("queries,keys,causal", [(3, 3, False), (3, 3, True), (1, 7, True), (3, 7, True)])
def test_matches_explicit_bottom_right_sdpa_mask(heads, kv_heads, queries, keys, causal):
    generator = torch.Generator().manual_seed(2)
    q = torch.randn(queries, heads, 8, generator=generator)
    k, v = (torch.randn(keys, kv_heads, 8, generator=generator) for _ in range(2))
    mask = None
    if causal:
        mask = torch.arange(keys)[None, :] <= torch.arange(queries)[:, None] + keys - queries
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(heads // kv_heads, 1).transpose(0, 1),
        v.repeat_interleave(heads // kv_heads, 1).transpose(0, 1),
        attn_mask=mask,
    ).transpose(0, 1)
    actual = reference_attention(q, k, v, causal)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_ragged_segments_are_independent_and_plans_are_replaced():
    attention = TorchRaggedAttention(scale=0.25)
    ctx = StepContext((1, 2), "encode", 0, False)
    attention.plan(AttentionStep(segments=(Segment(1, "image", 2), Segment(2, "image", 3)), causal=False), ctx)
    q, k, v = (torch.randn(5, 1, 8) for _ in range(3))
    expected = torch.cat(
        [reference_attention(q[:2], k[:2], v[:2], scale=0.25), reference_attention(q[2:], k[2:], v[2:], scale=0.25)]
    )
    torch.testing.assert_close(attention.run(q, k, v, "image"), expected)
    attention.plan(AttentionStep(segments=(Segment(1, "next", 5),)), ctx)
    with pytest.raises(KeyError):
        attention.run(q, k, v, "image")
    with pytest.raises(ValueError, match="rows disagree"):
        attention.run(q[:2], k, v, "next")


def test_paged_reads_only_each_requests_declared_pages():
    attention = TorchPagedAttention("kv", torch.device("cpu"))
    views = [
        SimpleNamespace(page_idxs=[2, 1], length=5, to_compute=1),
        SimpleNamespace(page_idxs=[3], length=2, to_compute=2),
    ]
    ctx = StepContext((1, 2), "decode", 0, False, plan_results={"kv": {"epoch": SimpleNamespace(views=views)}})
    attention.plan(AttentionStep(causal=True), ctx)
    cache = torch.randn(4, 2, 4, 2, 8)
    cache[0].fill_(float("nan"))
    q = torch.randn(3, 2, 8)
    expected = []
    offset = 0
    for view in views:
        pages = cache[view.page_idxs]
        expected.append(
            reference_attention(
                q[offset : offset + view.to_compute],
                pages[:, 0].flatten(0, 1)[: view.length],
                pages[:, 1].flatten(0, 1)[: view.length],
                causal=True,
            )
        )
        offset += view.to_compute
    torch.testing.assert_close(attention.run(q, "epoch", kv_cache_layer=cache), torch.cat(expected))


@pytest.mark.parametrize("attention", [TorchPagedAttention("kv", torch.device("cpu")), TorchRaggedAttention()])
@pytest.mark.parametrize("capture,preplan", [(True, False), (False, True)])
def test_reference_backend_explicitly_refuses_capture_and_preplan(attention, capture, preplan):
    assert not attention.can_capture()
    assert not attention.supports_preplan
    with pytest.raises(ValueError, match="eager-only"):
        attention.plan(AttentionStep(), StepContext((1,), "walk", 0, capture, is_preplan=preplan))
