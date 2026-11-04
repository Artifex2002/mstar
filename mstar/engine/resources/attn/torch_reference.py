"""Opt-in eager attention preserving PyTorch dtypes, including FP32.

This correctness backend allocates gathered contexts and score matrices. It is
not capturable and makes no performance claim; optimized backends stay unchanged.
"""

from typing import NamedTuple

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.base import AttentionResource


def reference_attention(q, k, v, causal=False, scale=None):
    """Packed [tokens, heads, dim]; causal queries align to the key suffix."""
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("torch attention requires matching Q/K/V dtypes")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be a multiple of KV heads")
    repeat = q.shape[1] // k.shape[1]
    if repeat != 1:
        k, v = (value.repeat_interleave(repeat, dim=1) for value in (k, v))
    scores = (q.transpose(0, 1) @ k.permute(1, 2, 0)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    if causal:
        query = torch.arange(q.shape[0], device=q.device)[:, None] + k.shape[0] - q.shape[0]
        key = torch.arange(k.shape[0], device=k.device)[None, :]
        scores = scores.masked_fill(key > query, float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v.transpose(0, 1)).transpose(0, 1)


class _PagedRow(NamedTuple):
    pages: torch.Tensor
    length: int
    queries: int


class TorchPagedAttention(AttentionManager):
    def __init__(self, kv_key, device):
        self._kv_key = kv_key
        self._device = device
        self._plans = {}

    def depends_on(self):
        return {self._kv_key}

    def can_capture(self):
        return False

    @property
    def requires_kv_write(self):
        return True

    def plan(self, step: AttentionStep, ctx):
        refuse_context_only(step, type(self).__name__)
        if ctx.capture or ctx.is_preplan or ctx.slot_lease is not None:
            raise ValueError("torch reference attention is eager-only")
        self.reset_default_cursors()
        self._causal = step.causal
        self._plans = {
            label: tuple(
                _PagedRow(
                    torch.tensor(view.page_idxs, dtype=torch.long, device=self._device),
                    view.length,
                    view.to_compute,
                )
                for view in plan.views
            )
            for label, plan in ctx.plan_results[self._kv_key].items()
        }

    @torch.compiler.disable
    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        if kv_cache_layer is None:
            raise ValueError("torch paged attention needs the KV resource's layer view")
        rows = self._plans[self._default_label if label is None else label]
        if sum(row.queries for row in rows) != q.shape[0]:
            raise ValueError("torch paged query rows disagree with the declared plan")
        outputs, offset = [], 0
        for row in rows:
            pages = kv_cache_layer.index_select(0, row.pages)
            keys = pages[:, 0].flatten(0, 1)[: row.length]
            values = pages[:, 1].flatten(0, 1)[: row.length]
            outputs.append(reference_attention(q[offset : offset + row.queries], keys, values, self._causal))
            offset += row.queries
        return torch.cat(outputs, dim=0)


class TorchRaggedAttention(AttentionResource):
    prefix_skip_safe = True

    @classmethod
    def build(cls, spec, info):
        joint = info.joint_comm_group
        spec.config.shard(info.head_shards if spec.config.shard_over_sp or joint is None else joint.tp_group.world_size)
        return cls(spec.config.sm_scale)

    def __init__(self, scale=None):
        self._scale = scale
        self._plans = {}

    def can_capture(self):
        return False

    def plan(self, step: AttentionStep, ctx):
        refuse_context_only(step, type(self).__name__)
        if ctx.capture or ctx.is_preplan or ctx.slot_lease is not None:
            raise ValueError("torch reference attention is eager-only")
        self.reset_default_cursors()
        self._causal = step.causal
        plans = {}
        for segment in step.segments or ():
            plans.setdefault(segment.label, []).append(segment.span)
        self._plans = plans

    @torch.compiler.disable
    def run(self, q, k, v, label=None, return_lse=False):
        if return_lse:
            raise NotImplementedError("torch reference attention does not return LSE")
        lengths = self._plans[self._default_label if label is None else label]
        total = sum(lengths)
        if any(value.shape[0] != total for value in (q, k, v)):
            raise ValueError("torch ragged rows disagree with the declared plan")
        outputs, offset = [], 0
        for length in lengths:
            end = offset + length
            outputs.append(reference_attention(q[offset:end], k[offset:end], v[offset:end], self._causal, self._scale))
            offset = end
        return torch.cat(outputs, dim=0)
