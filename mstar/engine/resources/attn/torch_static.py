"""Opt-in fixed-layout PyTorch attention, including FP32 CUDA graph replay.

Paged steps require one uniform positive query span per row. Cache lengths and
page addresses vary through slotted buffers; the context ceiling stays fixed.
This is a correctness/capture backend, not a fused-kernel performance claim.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.base import AttentionResource


def _attend(q, k, v, mask, scale):
    # [batch, tokens, heads, dim], with an explicit bottom-right causal mask.
    if q.dtype != k.dtype or q.dtype != v.dtype or q.shape[2] % k.shape[2]:
        raise ValueError("torch_static requires matching Q/K/V dtypes and divisible head counts")
    repeat = q.shape[2] // k.shape[2]
    if repeat != 1:
        k, v = (x.repeat_interleave(repeat, dim=2) for x in (k, v))
    scores = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)


class _Planned:
    """Promote a preplan without changing what the in-flight forward reads."""

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        self._preplanned = None

    def _promote(self, ctx):
        if not ctx.is_preplan and self._preplanned is not None:
            self._current, self._preplanned = self._preplanned, None
            self.reset_default_cursors()
            return True
        if ctx.is_preplan and (ctx.slot_lease is None or self._preplanned is not None):
            raise ValueError("torch_static preplan needs a lease and no pending preplan")
        return False

    def _publish(self, plans, ctx):
        if ctx.is_preplan:
            self._preplanned = plans
        else:
            self._current = plans
            self.reset_default_cursors()


@dataclass
class _PagedPlan:
    indices: torch.Tensor
    valid: torch.Tensor
    mask: torch.Tensor
    live: torch.Tensor
    host: tuple[torch.Tensor, ...]
    queries: int
    causal: bool


class TorchStaticPagedAttention(_Planned, AttentionManager):
    def __init__(self, kv_key, device, config):
        self._kv_key, self._device, self._config = kv_key, torch.device(device), config
        self._states, self._current, self._preplanned = {}, {}, None
        self._geometries = {}

    def depends_on(self):
        return {self._kv_key}

    @property
    def force_double_buffer(self):
        return True

    def _allocate(self, rows, queries, causal):
        ceiling = self._config.max_seq_len
        shapes = ((rows, ceiling), (rows, ceiling), (rows, 1, queries, ceiling), (rows, 1, 1, 1))
        dtypes = (torch.long, torch.bool, torch.bool, torch.bool)
        host = tuple(
            torch.empty(shape, dtype=dtype, pin_memory=self._device.type == "cuda")
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        buffers = tuple(torch.empty_like(value, device=self._device) for value in host)
        return _PagedPlan(*buffers, host, queries, causal)

    def plan(self, step: AttentionStep, ctx):
        refuse_context_only(step, type(self).__name__)
        if self._promote(ctx):
            return
        plans = {}
        for label, output in ctx.plan_results[self._kv_key].items():
            views = output.views
            spans = {view.to_compute for view in views}
            if len(spans) != 1 or not spans or min(spans) < 1:
                raise ValueError("torch_static paged attention needs uniform positive query spans")
            queries = next(iter(spans))
            lease = ctx.slot_lease
            rows = lease.bucket.bs if lease is not None and lease.bucket is not None else len(views)
            if len(views) > rows:
                raise ValueError("torch_static supports one segment per request per label")
            bucket = lease.bucket if lease is not None else None
            if bucket is not None:
                identity = (bucket, ctx.slot, label)
                geometry = (rows, queries, step.causal)
                if identity in self._geometries and self._geometries[identity] != geometry:
                    raise ValueError("torch_static paged layout changed within a capture bucket")
                self._geometries[identity] = geometry
            key = (bucket, ctx.slot, label, rows, queries, step.causal)
            plan = self._states.get(key)
            if plan is None:
                plan = self._states[key] = self._allocate(rows, queries, step.causal)
            indices, valid, mask, live = (value.numpy() for value in plan.host)
            indices.fill(0)
            valid.fill(False)
            mask.fill(False)
            mask[..., 0] = True  # a well-formed dummy row, selected to zero below
            live.fill(False)
            page_size = self._config.tokens_per_page
            for row, view in enumerate(views):
                if ctx.is_padding_row(view.request_id):
                    continue
                if view.start != 0 or not queries <= view.length <= self._config.max_seq_len:
                    raise ValueError("torch_static requires non-windowed contexts within max_seq_len")
                if len(view.page_idxs) * page_size < view.length:
                    raise ValueError("torch_static KV plan has too few pages")
                positions = np.arange(view.length)
                # A layer view is [page, K/V, token, head, dim].
                addresses = np.asarray(view.page_idxs, dtype=np.int64)[positions // page_size]
                indices[row, : view.length] = addresses * (2 * page_size) + positions % page_size
                valid[row, : view.length] = True
                mask[row].fill(False)
                allowed = positions[None, :] <= np.arange(queries)[:, None] + view.length - queries
                mask[row, 0, :, : view.length] = allowed if step.causal else True
                live[row] = True
            for destination, source in zip((plan.indices, plan.valid, plan.mask, plan.live), plan.host, strict=True):
                destination.copy_(source, non_blocking=True)
            plans[label] = plan
        self._publish(plans, ctx)

    @torch.compiler.disable
    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        if kv_cache_layer is None or not kv_cache_layer.is_contiguous():
            raise ValueError("torch_static requires a contiguous NHD KV layer view")
        plan = self._current[self._default_label if label is None else label]
        rows, ceiling = plan.indices.shape
        if q.shape[0] != rows * plan.queries:
            raise ValueError("torch_static query rows disagree with the planned bucket")
        heads, dim = kv_cache_layer.shape[-2:]
        flat = kv_cache_layer.view(-1, heads, dim)
        indices = plan.indices.reshape(-1)
        keys = flat.index_select(0, indices).view(rows, ceiling, heads, dim)
        values = flat.index_select(0, indices + self._config.tokens_per_page).view(rows, ceiling, heads, dim)
        # Sink/unwritten storage may contain NaNs. Select it out before either matmul.
        keep = plan.valid[:, :, None, None]
        keys, values = (torch.where(keep, x, 0) for x in (keys, values))
        queries = q.view(rows, plan.queries, *q.shape[1:])
        output = _attend(queries, keys, values, plan.mask, q.shape[-1] ** -0.5)
        return torch.where(plan.live, output, 0).reshape_as(q)

    def cleanup(self):
        self._states.clear()
        self._geometries.clear()
        self._current.clear()
        self.clear_preplan()


class TorchStaticRaggedAttention(_Planned, AttentionResource):
    """Uniform independent spans; no device-side schedule or persistent KV."""

    prefix_skip_safe = True

    @classmethod
    def build(cls, spec, info):
        joint = info.joint_comm_group
        spec.config.shard(info.head_shards if spec.config.shard_over_sp or joint is None else joint.tp_group.world_size)
        return cls(spec.config.sm_scale, info.device)

    def __init__(self, scale, device):
        self._scale, self._device = scale, torch.device(device)
        self._current, self._preplanned, self._layouts = {}, None, {}

    def plan(self, step: AttentionStep, ctx):
        refuse_context_only(step, type(self).__name__)
        if self._promote(ctx):
            return
        grouped = {}
        for segment in step.segments or ():
            grouped.setdefault(segment.label, []).append(segment.span)
        plans = {}
        for label, spans in grouped.items():
            if len(set(spans)) != 1 or min(spans) < 1:
                raise ValueError("torch_static ragged attention needs uniform positive spans per label")
            lease = ctx.slot_lease
            bucket = lease.bucket if lease is not None else None
            rows = bucket.bs if bucket is not None else len(spans)
            if len(spans) > rows:
                raise ValueError("torch_static supports one segment per request per label")
            layout = (rows, spans[0], step.causal)
            key = (bucket, ctx.slot, label) if bucket is not None else (None, ctx.slot, label, layout)
            if key in self._layouts and self._layouts[key][:3] != layout:
                raise ValueError("torch_static ragged layout changed within a capture bucket")
            if key not in self._layouts:
                mask = (
                    torch.ones(spans[0], spans[0], dtype=torch.bool, device=self._device).tril()
                    if step.causal
                    else None
                )
                self._layouts[key] = (*layout, mask)
            plans[label] = self._layouts[key]
        self._publish(plans, ctx)

    @torch.compiler.disable
    def run(self, q, k, v, label=None, return_lse=False):
        if return_lse:
            raise NotImplementedError("torch_static does not return LSE")
        rows, span, _, mask = self._current[self._default_label if label is None else label]
        if any(value.shape[0] != rows * span for value in (q, k, v)):
            raise ValueError("torch_static ragged rows disagree with the planned bucket")
        q, k, v = (value.reshape(rows, span, *value.shape[1:]) for value in (q, k, v))
        return _attend(q, k, v, mask, self._scale).reshape(rows * span, *q.shape[2:])

    def cleanup(self):
        self._current.clear()
        self._layouts.clear()
        self.clear_preplan()
