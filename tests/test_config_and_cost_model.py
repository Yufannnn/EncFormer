#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_glue import validate_model_matches_config
from src.cost_model import (
    LEGACY_PROFILE,
    PHANTOM_GPU_PROFILE,
    PLAIN_PROFILE,
    baseline_repack_ops,
    calibrate_ct_bytes_from_bridge,
    calibrate_repack_time,
    compute_design_cost,
    encformer_conversions,
    encformer_repack_ops,
    get_backend_profile,
    minimal_baseline_conversions,
)
from src.engines.encformer_model import (
    EncFormerBertForSequenceClassification,
    infer_model_config_name_from_metadata,
)
from src.engines.mpc_engine_plain import PlainMpcEngine
from src.fhe.phantom.phantom_native_pipe import (
    native_supported_model_configs,
    validate_native_inputs,
    validate_native_model_config,
)
from src.inference_runtime import (
    inject_layer_running_denoms,
    prepare_layer_running_denoms,
)
from src.models.model_config import get_config
from src.utils_comm import LAN, estimate_bridge_bytes, estimate_ct_bytes


def test_infer_model_config_name_from_metadata() -> None:
    assert (
        infer_model_config_name_from_metadata(model_type="bert", hidden_size=768, num_layers=12, seq_len=128)
        == "bert-base"
    )
    assert (
        infer_model_config_name_from_metadata(model_type="bert", hidden_size=1024, num_layers=24, seq_len=128)
        == "bert-large"
    )
    assert (
        infer_model_config_name_from_metadata(model_type="gpt2", hidden_size=768, num_layers=12, seq_len=64)
        == "gpt2-base"
    )
    assert infer_model_config_name_from_metadata(model_type="bert", hidden_size=999, num_layers=1, seq_len=16) is None


def test_validate_model_matches_config_rejects_mismatch() -> None:
    model = EncFormerBertForSequenceClassification()
    validate_model_matches_config(model, "bert-base")
    with pytest.raises(ValueError, match="Checkpoint/model_config mismatch"):
        validate_model_matches_config(model, "bert-large")


def test_cost_model_variants_change_expected_terms() -> None:
    cfg = get_config("bert-base")
    assert minimal_baseline_conversions(cfg, use_cc=False) == 2 * minimal_baseline_conversions(cfg)
    assert encformer_conversions(cfg, use_cc=False) == 2 * encformer_conversions(cfg)
    assert encformer_repack_ops(cfg, use_scp=True) == 0
    assert encformer_repack_ops(cfg, use_scp=False) == baseline_repack_ops(cfg)


def test_layer_running_denoms_are_prepared_and_injected() -> None:
    running_denoms = {
        "layer_0_bpmax_running_denominator": np.ones((1, 12, 128, 1), dtype=np.float64),
        "layer_0_ln1_running_denominator": np.ones((1, 128, 1), dtype=np.float64) * 2,
    }
    prepared = prepare_layer_running_denoms(running_denoms, 0)
    assert prepared["bpmax_running_denominator"].shape == (12, 128)
    assert prepared["ln1_running_denominator"].shape == (128, 1)

    engine = PlainMpcEngine()
    inject_layer_running_denoms(engine, prepared)
    assert "bpmax_running_denominator" in engine._batch_state.buffers
    assert "ln1_running_denominator" in engine._batch_state.buffers

    inject_layer_running_denoms(engine, {})


def test_native_backend_guards_are_explicit() -> None:
    supported = native_supported_model_configs()
    assert "bert-base" in supported
    assert "bert-large" in supported
    assert "gpt2-base" in supported
    validate_native_model_config("bert-base")
    validate_native_model_config("bert-large")
    validate_native_model_config("gpt2-base")
    with pytest.raises(ValueError):
        validate_native_model_config("nonexistent-config")
    with pytest.raises(ValueError, match="input_embeds shape"):
        validate_native_inputs({}, np.zeros((64, 768), dtype=np.float64), model_config_name="bert-base")


def test_backend_profiles_affect_cost() -> None:

    cfg = get_config("bert-base")
    convs = encformer_conversions(cfg)
    repacks = baseline_repack_ops(cfg)

    cost_legacy = compute_design_cost("D", convs, repacks, cfg, LAN, backend=LEGACY_PROFILE)
    cost_plain = compute_design_cost("D", convs, repacks, cfg, LAN, backend=PLAIN_PROFILE)
    cost_phantom = compute_design_cost("D", convs, repacks, cfg, LAN, backend=PHANTOM_GPU_PROFILE)

    assert cost_plain.t_repack_s == 0.0

    assert cost_legacy.t_repack_s > 0
    assert cost_phantom.t_repack_s > 0

    assert cost_plain.t_total_s < cost_legacy.t_total_s

    assert PHANTOM_GPU_PROFILE.total_fhe_time_s() > 0


def test_bridge_bytes_vs_analytical() -> None:

    nslots = 16384
    bridge = estimate_bridge_bytes(nslots)
    analytical = estimate_ct_bytes(nslots)

    assert bridge == analytical


def test_calibrate_repack_time() -> None:

    assert calibrate_repack_time(1.0, 1000) == pytest.approx(0.001)
    assert calibrate_repack_time(0.0, 0) == 0.0


def test_calibrate_ct_bytes_from_bridge() -> None:

    assert calibrate_ct_bytes_from_bridge(1_000_000, 100) == 10_000
    assert calibrate_ct_bytes_from_bridge(0, 0) == 0


def test_get_backend_profile() -> None:

    assert get_backend_profile("plain").repack_time_s == 0.0
    assert get_backend_profile("phantom").repack_time_s > 0
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_profile("nonexistent")
