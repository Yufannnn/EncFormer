from __future__ import annotations

import numpy as np
import pytest

from src.engines.mpc_engine_plain import PlainMpcEngine
from src.engines.mpc_gelu_secure import (
    secure_gelu_plain_fixedpoint,
    secure_gelu_preeval_plain_fixedpoint,
)


@pytest.mark.parametrize("seed", [0, 42, 777])
def test_preeval_matches_full_secure_gelu(seed):

    rng = np.random.default_rng(seed)

    x = np.concatenate(
        [
            rng.uniform(-5.0, -2.7, size=64),
            rng.uniform(-2.7, 2.7, size=64),
            rng.uniform(2.7, 5.0, size=64),
        ]
    )
    y_full = secure_gelu_plain_fixedpoint(x)
    y_pre = secure_gelu_preeval_plain_fixedpoint(x)
    assert np.allclose(y_full, y_pre, atol=1e-12), (
        f"Pre-eval diverged from full secure GELU; max diff = {np.max(np.abs(y_full - y_pre))}"
    )


def test_preeval_matches_torch_gelu_within_quantization():

    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=256)
    y_pre = secure_gelu_preeval_plain_fixedpoint(x)

    y_ref = torch.nn.functional.gelu(torch.from_numpy(x.astype(np.float32))).numpy().astype(np.float64)

    assert np.max(np.abs(y_pre - y_ref)) < 1e-2, (
        f"Pre-eval diverged from torch GELU; max diff = {np.max(np.abs(y_pre - y_ref))}"
    )


def test_engine_round_count_drops_to_one():

    eng = PlainMpcEngine()
    rng = np.random.default_rng(1)
    x = rng.uniform(-3.0, 3.0, size=64)

    rounds_before = eng.get_round_summary().get("rounds", {}).copy() if hasattr(eng, "get_round_summary") else None

    y_full = eng.gelu(x)
    y_pre = eng.gelu_preeval(x)

    if hasattr(eng, "get_round_summary"):
        summary = eng.get_round_summary().get("rounds", {})

        assert summary.get("gelu", 0) == 4, f"Expected gelu=4 rounds, got {summary.get('gelu')}"
        assert summary.get("gelu_preeval", 0) == 1, f"Expected gelu_preeval=1 round, got {summary.get('gelu_preeval')}"

    assert np.allclose(y_full, y_pre, atol=1e-12)


def test_engine_accepts_external_f0_f1():

    from src.engines.mpc_gelu_secure import PlainFixedPointOps, load_secure_gelu_config, precompute_f0_f1_fixedpoint

    eng = PlainMpcEngine()
    cfg = load_secure_gelu_config()
    ops = PlainFixedPointOps(cfg)
    rng = np.random.default_rng(2)
    x = rng.uniform(-3.0, 3.0, size=64)

    f0_q, f1_q = precompute_f0_f1_fixedpoint(x, cfg)
    f0_decoded = ops.decode(f0_q)
    f1_decoded = ops.decode(f1_q)

    y_internal = eng.gelu_preeval(x)
    y_external = eng.gelu_preeval(x, f0=f0_decoded, f1=f1_decoded)
    assert np.allclose(y_internal, y_external, atol=1e-12), (
        f"External F0/F1 path diverged: max diff = {np.max(np.abs(y_internal - y_external))}"
    )
