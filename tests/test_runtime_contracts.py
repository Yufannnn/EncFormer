#!/usr/bin/env python3


from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_glue import resolve_model_config_name
from src.engines.encformer_model import (
    EncFormerGPT2LM,
    infer_model_config_name,
)
from src.engines.mpc_engine_plain import PlainMpcEngine
from src.fhe.phantom.phantom_native_pipe import validate_native_inputs
from src.inference_runtime import (
    inject_layer_running_denoms,
    prepare_layer_running_denoms,
)
from src.models.model_config import get_config
from src.utils_comm import (
    LAN,
    WAN1,
    WAN2,
    WAN3,
    CommStats,
    compute_latency,
    estimate_bridge_bytes,
    estimate_ct_bytes,
)


class DummyBertModel:
    def __init__(self, *, hidden_size: int, num_layers: int, seq_len: int) -> None:
        self.hidden_size = hidden_size
        self.layers = [object() for _ in range(num_layers)]
        self.seq_len = seq_len


class DummyGPT2Model(EncFormerGPT2LM):
    def __init__(self, *, hidden_size: int, num_layers: int, seq_len: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(nn.Identity() for _ in range(num_layers))
        self.seq_len = seq_len


def _valid_native_weights(cfg_name: str) -> dict[str, np.ndarray]:
    cfg = get_config(cfg_name)
    d = cfg.d_model
    d_ff = cfg.d_ff
    return {
        "WQ": np.zeros((d, d), dtype=np.float64),
        "WK": np.zeros((d, d), dtype=np.float64),
        "WV": np.zeros((d, d), dtype=np.float64),
        "WO": np.zeros((d, d), dtype=np.float64),
        "W1": np.zeros((d, d_ff), dtype=np.float64),
        "W2": np.zeros((d_ff, d), dtype=np.float64),
        "bQ": np.zeros((d,), dtype=np.float64),
        "bK": np.zeros((d,), dtype=np.float64),
        "bV": np.zeros((d,), dtype=np.float64),
        "bO": np.zeros((d,), dtype=np.float64),
        "b1": np.zeros((d_ff,), dtype=np.float64),
        "b2": np.zeros((d,), dtype=np.float64),
        "ln1_w": np.zeros((d,), dtype=np.float64),
        "ln1_b": np.zeros((d,), dtype=np.float64),
        "ln2_w": np.zeros((d,), dtype=np.float64),
        "ln2_b": np.zeros((d,), dtype=np.float64),
    }


_MODELS = [
    ("bert-base", DummyBertModel(hidden_size=768, num_layers=12, seq_len=128)),
    ("bert-large", DummyBertModel(hidden_size=1024, num_layers=24, seq_len=128)),
    ("gpt2-base", DummyGPT2Model(hidden_size=768, num_layers=12, seq_len=64)),
]


def test_get_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model config"):
        get_config("does-not-exist")


def test_prepare_layer_running_denoms_returns_empty_for_none():
    assert prepare_layer_running_denoms(None, 0) == {}


def test_prepare_layer_running_denoms_normalizes_shapes():
    running_denoms = {
        "layer_2_bpmax_running_denominator": np.ones((1, 12, 128, 1)),
        "layer_2_ln1_running_denominator": np.ones((1, 128, 1)),
        "layer_2_ln2_running_denominator": np.ones((128, 1)),
    }
    prepared = prepare_layer_running_denoms(running_denoms, 2)
    assert prepared["bpmax_running_denominator"].shape == (12, 128)
    assert prepared["ln1_running_denominator"].shape == (128, 1)
    assert prepared["ln2_running_denominator"].shape == (128, 1)


def test_prepare_layer_running_denoms_ignores_other_layers():
    running_denoms = {
        "layer_1_bpmax_running_denominator": np.ones((1, 12, 128, 1)),
    }
    assert prepare_layer_running_denoms(running_denoms, 0) == {}


def test_inject_layer_running_denoms():
    engine = PlainMpcEngine()
    prepared = {
        "bpmax_running_denominator": np.ones((2, 2)),
        "ln2_running_denominator": np.ones((2, 2)),
    }
    inject_layer_running_denoms(engine, prepared)
    buffers = engine._batch_state.buffers
    assert "bpmax_running_denominator" in buffers
    assert "ln2_running_denominator" in buffers
    assert "ln1_running_denominator" not in buffers


def test_infer_and_resolve_model_config_name():
    for expected, model in _MODELS:
        assert infer_model_config_name(model) == expected
        assert resolve_model_config_name(model, None) == expected
        assert resolve_model_config_name(model, expected) == expected


def test_resolve_model_config_name_rejects_mismatch():
    model = DummyBertModel(hidden_size=768, num_layers=12, seq_len=128)
    with pytest.raises(ValueError, match="mismatch"):
        resolve_model_config_name(model, "bert-large")


def test_resolve_model_config_name_requires_explicit_when_unknown():
    unknown = DummyBertModel(hidden_size=512, num_layers=7, seq_len=99)
    with pytest.raises(ValueError, match="Could not infer"):
        resolve_model_config_name(unknown, None)


def test_validate_native_inputs_all_models():
    for cfg_name in ["bert-base", "bert-large", "gpt2-base"]:
        cfg = get_config(cfg_name)
        weights = _valid_native_weights(cfg_name)
        embeds = np.zeros((cfg.m, cfg.d_model), dtype=np.float64)
        validate_native_inputs(weights, embeds, model_config_name=cfg_name)

        validate_native_inputs(
            weights,
            embeds,
            attention_mask=np.ones((cfg.m,)),
            model_config_name=cfg_name,
        )
        validate_native_inputs(
            weights,
            embeds,
            attention_mask=np.ones((cfg.m, cfg.m)),
            model_config_name=cfg_name,
        )


def test_validate_native_inputs_rejects_bad_shapes():
    cfg = get_config("bert-base")
    weights = _valid_native_weights("bert-base")
    weights["bQ"] = np.zeros((2, 2), dtype=np.float64)
    embeds = np.zeros((cfg.m, cfg.d_model), dtype=np.float64)
    with pytest.raises(ValueError, match="bQ"):
        validate_native_inputs(weights, embeds, model_config_name="bert-base")


def test_compute_latency_all_profiles():
    stats = CommStats(ct_bytes=100, mpc_msg_bytes=50)
    stats.add_bridge_c2m(2)
    stats.add_bridge_m2c(1)
    stats.add_mpc_rounds("gelu", 4)
    stats.add_mpc_rounds("softmax", 3)

    for profile in [LAN, WAN1, WAN2, WAN3]:
        latency = compute_latency(stats, profile)
        expected_conv = (stats.total_bridge_bytes() * 8) / profile.bandwidth_bps
        expected_mpc = stats.total_mpc_rounds() * profile.rtt_s
        assert latency["T_conv_s"] == pytest.approx(expected_conv)
        assert latency["T_mpc_s"] == pytest.approx(expected_mpc)
        assert latency["T_total_s"] == pytest.approx(expected_conv + expected_mpc)


def test_estimate_byte_helpers_scale_linearly():
    for nslots in [1024, 16384, 32768]:
        assert estimate_ct_bytes(nslots) == nslots * 16
        assert estimate_bridge_bytes(nslots) == nslots * 16


def test_commstats_totals():
    stats = CommStats(ct_bytes=256, mpc_msg_bytes=1024)
    stats.add_bridge_c2m(3)
    stats.add_bridge_m2c(2)
    stats.add_mpc_rounds("softmax", 3)
    stats.add_mpc_rounds("gelu", 4)
    assert stats.total_bridge_bytes() == 5 * 256
    assert stats.total_mpc_rounds() == 7
    assert stats.total_mpc_bytes() == 7 * 1024
    assert stats.total_bytes() == (5 * 256) + (7 * 1024)


def test_cfg_propagates_without_reconfigure():

    from src.encformer import _MODEL_CFG, QKV_M, reconfigure

    reconfigure("bert-base")
    assert QKV_M == 128

    import src.encformer as ef
    from src.encformer import run_with_weights
    from src.engines.ckks_engine_plain import CKKSContext

    cfg_gpt2 = get_config("gpt2-base")

    ef.reconfigure("gpt2-base")
    assert ef.QKV_M == 64
    assert ef.QKV_D == 768

    ef.reconfigure("bert-base")
    assert ef.QKV_M == 128


def test_inject_clears_stale_denominators_on_reuse():

    engine = PlainMpcEngine()

    full = {
        "bpmax_running_denominator": np.ones((12, 128)),
        "ln1_running_denominator": np.ones((128, 1)),
        "ln2_running_denominator": np.ones((128, 1)),
    }
    inject_layer_running_denoms(engine, full)
    assert "bpmax_running_denominator" in engine._batch_state.buffers
    assert "ln1_running_denominator" in engine._batch_state.buffers
    assert "ln2_running_denominator" in engine._batch_state.buffers

    partial = {
        "bpmax_running_denominator": np.ones((12, 128)) * 2,
    }
    inject_layer_running_denoms(engine, partial)
    assert "bpmax_running_denominator" in engine._batch_state.buffers
    assert engine._batch_state.buffers["bpmax_running_denominator"].mean() == 2.0
    assert "ln1_running_denominator" not in engine._batch_state.buffers
    assert "ln2_running_denominator" not in engine._batch_state.buffers

    inject_layer_running_denoms(engine, {})
    assert "bpmax_running_denominator" not in engine._batch_state.buffers


def test_mixed_engine_accepts_running_denominators():

    from src.engines.mpc_engine_factory import get_mpc_engine

    try:
        engine = get_mpc_engine("mixed")
    except ImportError:
        pytest.skip("CrypTen not available for mixed engine")

    denoms = {
        "bpmax_running_denominator": np.ones((12, 128)),
        "ln1_running_denominator": None,
        "ln2_running_denominator": np.ones((128, 1)),
    }

    engine.set_running_denominators(denoms)
    assert "bpmax_running_denominator" in engine._plain._batch_state.buffers
    assert "ln2_running_denominator" in engine._plain._batch_state.buffers
    assert "ln1_running_denominator" not in engine._plain._batch_state.buffers
