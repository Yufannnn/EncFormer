from __future__ import annotations

import numpy as np
import pytest


def _make_engine(**kwargs):
    from src.engines.mpc_engine_ezpc import EzPCMpcEngine

    return EzPCMpcEngine(mode="emulated", **kwargs)


class TestEzPCEmulatedOps:
    def test_encode_decode_roundtrip(self):
        from src.engines.mpc_engine_ezpc import _EzPCEmulatedOps

        ops = _EzPCEmulatedOps(ring_bits=43, scale_bits=13)
        x = np.array([1.5, -0.25, 0.0, 3.14159], dtype=np.float64)
        encoded = ops.encode(x)
        decoded = ops.decode(encoded)
        np.testing.assert_allclose(decoded, x, atol=1e-3)

    def test_mul_associative(self):
        from src.engines.mpc_engine_ezpc import _EzPCEmulatedOps

        ops = _EzPCEmulatedOps(ring_bits=43, scale_bits=13)
        a = ops.encode(np.array([2.0]))
        b = ops.encode(np.array([3.0]))
        c = ops.encode(np.array([0.5]))
        ab_c = ops.mul(ops.mul(a, b), c)
        a_bc = ops.mul(a, ops.mul(b, c))
        np.testing.assert_allclose(ops.decode(ab_c), ops.decode(a_bc), atol=0.05)

    def test_cmp_lt(self):
        from src.engines.mpc_engine_ezpc import _EzPCEmulatedOps

        ops = _EzPCEmulatedOps()
        a = ops.encode(np.array([-1.0, 0.0, 1.0, 2.0]))
        b = ops.encode(np.array([0.0, 0.0, 0.0, 0.0]))
        result = ops.cmp_lt(a, b)
        np.testing.assert_array_equal(result, [1, 0, 0, 0])


class TestEzPCSoftmax:
    def test_softmax_rows_sum_to_one(self):
        engine = _make_engine()
        x = np.random.randn(4, 8)
        y = engine.softmax_rows(x)
        row_sums = y.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_softmax_positive(self):
        engine = _make_engine()
        x = np.random.randn(2, 5)
        y = engine.softmax_rows(x)
        assert np.all(y >= 0)


class TestEzPCLayerNorm:
    def test_layer_norm_zero_mean_unit_var(self):
        engine = _make_engine()
        x = np.random.randn(3, 16) * 5 + 2
        y = engine.layer_norm(x, eps=1e-5)

        row_means = y.mean(axis=1)
        row_vars = y.var(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-5)
        np.testing.assert_allclose(row_vars, 1.0, atol=0.05)

    def test_layer_norm_with_affine(self):
        engine = _make_engine()
        x = np.random.randn(2, 8)
        gamma = np.full(8, 2.0)
        beta = np.full(8, 1.0)
        y = engine.layer_norm(x, gamma=gamma, beta=beta)

        row_means = y.mean(axis=1)
        np.testing.assert_allclose(row_means, 1.0, atol=0.3)


class TestEzPCGelu:
    def test_gelu_shape_preserved(self):
        engine = _make_engine()
        x = np.random.randn(4, 16)
        y = engine.gelu(x)
        assert y.shape == x.shape

    def test_gelu_positive_for_large_positive(self):
        engine = _make_engine()
        x = np.array([[2.0, 3.0, 4.0]])
        y = engine.gelu(x)
        assert np.all(y > 0), f"GELU should be positive for large inputs, got {y}"

    def test_gelu_near_zero_at_zero(self):
        engine = _make_engine()
        x = np.array([[0.0]])
        y = engine.gelu(x)
        np.testing.assert_allclose(y, 0.0, atol=0.05)

    def test_gelu_matches_reference(self):

        from src.engines.mpc_gelu_secure import secure_gelu_plain_fixedpoint

        engine = _make_engine()
        x = np.linspace(-2.5, 2.5, 32)
        y_ezpc = engine.gelu(x.reshape(1, -1)).flatten()
        y_ref = secure_gelu_plain_fixedpoint(x)
        np.testing.assert_allclose(y_ezpc, y_ref, atol=0.02)


class TestFactoryIntegration:
    def test_factory_creates_ezpc(self):
        from src.engines.mpc_engine_factory import get_mpc_engine

        engine = get_mpc_engine("ezpc")
        assert engine.name == "ezpc"
        assert engine.backend_mode in ("emulated", "native")

    def test_pipeline_resolve(self):
        import os

        from src.engines.mpc_engine_factory import resolve_pipeline

        old = os.environ.get("PIPELINE")
        try:
            os.environ["PIPELINE"] = "phantom-ezpc"
            ckks, mpc = resolve_pipeline()
            assert ckks == "phantom_native"
            assert mpc == "ezpc"
        finally:
            if old is None:
                os.environ.pop("PIPELINE", None)
            else:
                os.environ["PIPELINE"] = old

    def test_pipeline_invalid(self):
        import os

        from src.engines.mpc_engine_factory import resolve_pipeline

        old = os.environ.get("PIPELINE")
        try:
            os.environ["PIPELINE"] = "invalid-combo"
            with pytest.raises(ValueError, match="Unknown PIPELINE"):
                resolve_pipeline()
        finally:
            if old is None:
                os.environ.pop("PIPELINE", None)
            else:
                os.environ["PIPELINE"] = old
