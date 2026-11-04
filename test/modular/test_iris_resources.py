"""IRIS declarations and real CPU cache/position lifecycle, without weights."""

from pathlib import Path

import pytest
import torch
import yaml

from mstar.engine.resources import (
    KVStep,
    PositionConfig,
    PositionStep,
    RaggedAttentionSpec,
    SamplerSpec,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.model.iris.config import (
    DECODER_ATTN,
    ENCODER_ATTN,
    ENDING_SAMPLER,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    IMAGINE_WALK,
    INITIAL_DECODER,
    OBSERVATION_SAMPLER,
    REWARD_SAMPLER,
    WORLD_ATTN,
    WORLD_KV,
    WORLD_MODEL,
    WORLD_POS,
)
from mstar.model.iris.iris_model import IrisModel
from mstar.model.iris.resources import context_epochs, request_kv_layout


def _specs():
    model = IrisModel()
    specs = model.get_node_resources()
    deployment = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/iris.yaml").read_text())
    apply_yaml_overrides(specs, deployment)
    return model, resolve_spec_dependencies(specs)


def _request_config(model, horizon):
    cfg = model.get_request_resource_configs({}, {"iris_horizon": horizon})[WORLD_KV]
    layouts = request_kv_layout(horizon)[WORLD_KV]
    cfg.apply_conductor_config(kv_layout={label: layout.row() for label, layout in layouts.items()}, loop_caps={})
    return cfg


def test_geometry_binding_and_resource_dependencies():
    model, specs = _specs()
    assert set(specs) == {
        WORLD_KV,
        WORLD_ATTN,
        WORLD_POS,
        OBSERVATION_SAMPLER,
        REWARD_SAMPLER,
        ENDING_SAMPLER,
        ENCODER_ATTN,
        DECODER_ATTN,
    }
    kv = specs[WORLD_KV].config
    assert (kv.num_layers, kv.num_kv_heads, kv.num_qo_heads, kv.head_dim) == (10, 4, 4, 64)
    assert kv.max_seq_len == 340
    assert (kv.tokens_per_page, kv.max_num_pages, kv.prefix_cache) == (16, 64, False)
    assert specs[WORLD_ATTN].depends_on() == specs[WORLD_POS].depends_on() == {WORLD_KV}
    for key in (WORLD_KV, WORLD_ATTN, WORLD_POS, OBSERVATION_SAMPLER, REWARD_SAMPLER, ENDING_SAMPLER):
        assert specs[key].nodes == {WORLD_MODEL}
    for key, node in ((ENCODER_ATTN, IMAGE_ENCODER), (DECODER_ATTN, IMAGE_DECODER)):
        spec = specs[key]
        assert isinstance(spec, RaggedAttentionSpec)
        assert spec.nodes == ({node, INITIAL_DECODER} if node == IMAGE_DECODER else {node})
        assert spec.depends_on() == set()
        assert (spec.config.num_qo_heads, spec.config.num_kv_heads, spec.config.head_dim) == (1, 1, 64)
        assert spec.config.max_tokens_per_request == 256
        assert spec.config.dtype == torch.float32
    # Resource configs are freshly declared, not mutated process-wide singletons.
    specs[WORLD_KV].config.max_num_pages = 7
    assert model.get_node_resources()[0].config.max_num_pages != 7


@pytest.mark.parametrize(
    "horizon, frames, tokens",
    [
        (1, [1], [33]),
        (19, [19], [339]),
        (20, [19, 1], [339, 33]),
        (38, [19, 19], [339, 339]),
        (39, [19, 19, 1], [339, 339, 33]),
    ],
)
def test_context_boundaries_preserve_reference_reprime_schedule(horizon, frames, tokens):
    epochs = context_epochs(horizon)
    assert [epoch.frames for epoch in epochs] == frames
    assert [epoch.tokens for epoch in epochs] == tokens
    assert [epoch.label for epoch in epochs] == [f"epoch_{i}" for i in range(len(frames))]


def test_request_opens_only_used_labels_and_reserves_the_full_footprint():
    model, specs = _specs()
    assert set(model.kv_labels()[WORLD_KV]) == {"epoch_0", "epoch_1"}
    for horizon, labels, pages in ((1, {"epoch_0"}, 3), (20, {"epoch_0", "epoch_1"}, 25)):
        cfg = _request_config(model, horizon)
        assert cfg.opened_labels() == labels
        assert set(cfg.kv_layout) == labels
        assert cfg.pages({WORLD_MODEL}, {}, specs[WORLD_KV].config) == pages
        cfg.validate(specs[WORLD_KV])
    specs[WORLD_KV].config.max_num_pages = 25  # one sink page leaves only 24 for requests
    with pytest.raises(ValueError, match="needs 25 pages"):
        _request_config(model, 20).validate(specs[WORLD_KV])


def test_sampling_matches_unfiltered_categorical_and_has_independent_streams():
    model, specs = _specs()
    configs = model.get_request_resource_configs({}, {"iris_horizon": 1, "temperature": 0})
    seeds = []
    for key, vocab in ((OBSERVATION_SAMPLER, 512), (REWARD_SAMPLER, 3), (ENDING_SAMPLER, 2)):
        spec, cfg = specs[key], configs[key]
        assert isinstance(spec, SamplerSpec)
        assert spec.vocab_size == vocab
        assert not spec.enable_repetion_penalty
        assert (cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty, cfg.min_p) == (1, 0, 1, 1, 0)
        assert cfg.ignore_eos and not cfg.penalize_prompt
        cfg.validate(spec)
        cfg.apply_conductor_config(seed=11, resource_key=key)
        seeds.append(cfg.seed)
        repeated = model.get_request_resource_configs({}, {"iris_horizon": 1})[key]
        repeated.apply_conductor_config(seed=11, resource_key=key)
        assert repeated.seed == cfg.seed
    assert len(set(seeds)) == 3


class _NoTransfer:
    """No inter-worker transport is involved in the CPU lifecycle test."""

    def __init__(self, *args, **kwargs):
        pass

    def remove_request(self, rid):
        pass

    def cleanup(self):
        pass


def test_real_cache_and_position_plans_start_new_epoch_at_zero(monkeypatch):
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.engine.resources.position.manager import RopeManager

    monkeypatch.setattr(manager_mod, "KVTransferManager", _NoTransfer)
    # Exercise pageable CPU staging even on hosts with an available accelerator.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    model, specs = _specs()
    kv = KVManager(
        cfg=specs[WORLD_KV].config,
        name=WORLD_KV,
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    positions = RopeManager(
        config=PositionConfig(kv_cache=WORLD_KV),
        device=torch.device("cpu"),
        max_seq_len=340,
        head_dim=64,
        dtype=torch.float32,
    )
    runner = StepRunner(
        {WORLD_KV: kv, WORLD_POS: positions},
        node_resources={WORLD_MODEL: [WORLD_KV, WORLD_POS]},
    )
    runner.ingest_request(1, {WORLD_KV: _request_config(model, 20)})
    caches_after_frame = []
    try:
        for epoch in context_epochs(20):
            expected_start = 0
            # Each epoch prefills 16 tokens, then one action and 16 observation tokens per frame.
            for span in [16] + [1] * (epoch.frames * 17):
                step = SubmoduleStep(
                    segments=[Segment(1, epoch.label, span)],
                    steps={WORLD_KV: KVStep(), WORLD_POS: PositionStep()},
                )
                step.set_ctx(StepContext(request_ids=(1,), graph_walk=IMAGINE_WALK, slot=0, capture=False))
                assert runner.admit(step).outcome.ok
                plans = runner.plan(step)
                assert plans[WORLD_POS][epoch.label].tolist() == list(range(expected_start, expected_start + span))
                # Only this epoch's pages are in the plan; old epochs are not concatenated.
                assert plans[WORLD_KV][epoch.label].views[0].length == expected_start + span
                runner.commit(step)
                expected_start += span
                if expected_start > 16 and (expected_start - 16) % 17 == 0:
                    caches_after_frame.append(expected_start)
        assert caches_after_frame == list(range(33, 340, 17)) + [33]
        assert positions.position(1, "epoch_0") == 339
        assert positions.position(1, "epoch_1") == 33
        assert sum(len(stream.page_indices) for stream in kv._streams[1].values()) == 25
    finally:
        runner.remove_request(1)
        kv.cleanup()
