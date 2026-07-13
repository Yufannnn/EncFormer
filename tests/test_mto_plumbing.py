from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.bridges.ckks_mpc_bridge import (
    _maybe_trim_for_c2m,
    complex_ckks_to_mpc,
    real_ckks_to_mpc,
)


class _FakeCipher:
    def __init__(self, vals: np.ndarray, *, chain_level: int = 10):
        self._vals = np.asarray(vals, dtype=np.complex128)
        self.chain_level = chain_level

    def decorypt(self, *, copy=True, readonly=True):
        return self._vals.copy() if copy else self._vals

    def add(self, other):
        if isinstance(other, _FakeCipher):
            return _FakeCipher(self._vals + other._vals, chain_level=self.chain_level)
        return _FakeCipher(self._vals + np.asarray(other), chain_level=self.chain_level)


class _FakeCtx:
    def __init__(self, nslots: int):
        self.nslots = nslots

    @contextmanager
    def decorypt_scope(self):
        yield

    def encorypt(self, arr, **_):
        return _FakeCipher(arr, chain_level=10)


class _FakeCtxWithMTO(_FakeCtx):
    def __init__(self, nslots: int):
        super().__init__(nslots)
        self.mto_call_count = 0

    def mod_switch_to_min_for_c2m(self, ct):
        self.mto_call_count += 1

        if isinstance(ct, _FakeCipher):
            ct.chain_level = 1
        return ct


def _round_trip_value(vals_re: np.ndarray, vals_im: np.ndarray | None = None, *, ctx_cls=_FakeCtx) -> tuple:

    nslots = len(vals_re)
    ctx = ctx_cls(nslots)
    if vals_im is None:
        ct = _FakeCipher(vals_re.astype(np.complex128))
        share = real_ckks_to_mpc(ctx, ct)
        recon = share.share0 + share.share1
        return ctx, recon, None
    else:
        complex_vals = vals_re + 1j * vals_im
        ct = _FakeCipher(complex_vals.astype(np.complex128))
        sh_re, sh_im = complex_ckks_to_mpc(ctx, ct)
        recon_re = sh_re.share0 + sh_re.share1
        recon_im = sh_im.share0 + sh_im.share1
        return ctx, recon_re, recon_im


def test_real_c2m_without_mto_method_passes_through():

    rng = np.random.default_rng(42)
    vals = rng.standard_normal(64) * 0.3
    ctx, recon, _ = _round_trip_value(vals)

    assert np.allclose(recon, vals, atol=1e-3), (
        f"Reconstructed value diverged: max diff = {np.max(np.abs(recon - vals))}"
    )
    assert not hasattr(ctx, "mto_call_count")


def test_complex_c2m_without_mto_method_passes_through():
    rng = np.random.default_rng(43)
    vals_re = rng.standard_normal(64) * 0.3
    vals_im = rng.standard_normal(64) * 0.3
    ctx, rr, ri = _round_trip_value(vals_re, vals_im)
    assert np.allclose(rr, vals_re, atol=1e-3)
    assert np.allclose(ri, vals_im, atol=1e-3)


def test_real_c2m_with_mto_invokes_method():

    rng = np.random.default_rng(44)
    vals = rng.standard_normal(64) * 0.3
    ctx, recon, _ = _round_trip_value(vals, ctx_cls=_FakeCtxWithMTO)
    assert ctx.mto_call_count == 1, f"MTO method should be called once per C2M; got {ctx.mto_call_count}"
    assert np.allclose(recon, vals, atol=1e-3)


def test_complex_c2m_with_mto_invokes_method():
    rng = np.random.default_rng(45)
    vals_re = rng.standard_normal(64) * 0.3
    vals_im = rng.standard_normal(64) * 0.3
    ctx, rr, ri = _round_trip_value(vals_re, vals_im, ctx_cls=_FakeCtxWithMTO)
    assert ctx.mto_call_count == 1, f"MTO method should be called once; got {ctx.mto_call_count}"
    assert np.allclose(rr, vals_re, atol=1e-3)
    assert np.allclose(ri, vals_im, atol=1e-3)


def test_env_disable_mto_short_circuits():

    os.environ["ENCFORMER_DISABLE_MTO"] = "1"
    try:
        rng = np.random.default_rng(46)
        vals = rng.standard_normal(32) * 0.3
        ctx = _FakeCtxWithMTO(32)
        ct = _FakeCipher(vals.astype(np.complex128))
        share = real_ckks_to_mpc(ctx, ct)
        recon = share.share0 + share.share1
        assert ctx.mto_call_count == 0, (
            f"With ENCFORMER_DISABLE_MTO=1, MTO should NOT be called; got {ctx.mto_call_count}"
        )
        assert np.allclose(recon, vals, atol=1e-3)
    finally:
        os.environ.pop("ENCFORMER_DISABLE_MTO", None)


def test_mto_failure_falls_back_silently():

    class _BadMTOCtx(_FakeCtx):
        def mod_switch_to_min_for_c2m(self, ct):
            raise RuntimeError("simulated mod-switch failure")

    rng = np.random.default_rng(47)
    vals = rng.standard_normal(32) * 0.3
    ctx = _BadMTOCtx(32)
    ct = _FakeCipher(vals.astype(np.complex128))

    share = real_ckks_to_mpc(ctx, ct)
    recon = share.share0 + share.share1
    assert np.allclose(recon, vals, atol=1e-3)


def test_maybe_trim_helper_directly():

    ct = _FakeCipher(np.zeros(4, dtype=np.complex128), chain_level=10)

    out = _maybe_trim_for_c2m(_FakeCtx(4), ct)
    assert out is ct
    assert ct.chain_level == 10

    ctx2 = _FakeCtxWithMTO(4)
    ct2 = _FakeCipher(np.zeros(4, dtype=np.complex128), chain_level=10)
    out2 = _maybe_trim_for_c2m(ctx2, ct2)
    assert out2 is ct2
    assert ct2.chain_level == 1
    assert ctx2.mto_call_count == 1
