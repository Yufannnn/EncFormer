from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.engines.mpc_engine_plain import PlainShare


@runtime_checkable
class BridgeCipher(Protocol):
    def decorypt(self, *, copy: bool = True, readonly: bool = True) -> np.ndarray: ...

    def add(self, other: Any) -> "BridgeCipher": ...


@runtime_checkable
class BridgeContext(Protocol):
    @property
    def nslots(self) -> int: ...

    def encorypt(self, arr: Any, **kwargs: Any) -> BridgeCipher: ...

    def decorypt_scope(self) -> Any: ...


def _mod(x: np.ndarray, q: int) -> np.ndarray:
    return np.mod(x, q)


def _center_lift(x_mod: np.ndarray, q: int) -> np.ndarray:
    half = q // 2
    y = x_mod.astype(np.int64, copy=True)
    y[y >= half] -= q
    return y


def _conv_params() -> tuple[int, int, int]:
    q_conv_bits = int(os.environ.get("CKKS_Q_CONV_BITS", "52"))
    ell_bits = int(os.environ.get("CKKS_RING_BITS", "43"))
    fp_bits = int(os.environ.get("CKKS_MPC_PREC_BITS", "13"))
    q_conv = 1 << q_conv_bits
    ring = 1 << ell_bits
    return (q_conv, ring, 1 << fp_bits)


def _is_share_like(x: Any) -> bool:
    return hasattr(x, "share0") and hasattr(x, "share1")


def _to_int_share(x: Any, *, scale: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, PlainShare):
        s0 = np.round(np.asarray(x.share0, dtype=np.float64) * scale).astype(np.int64)
        s1 = np.round(np.asarray(x.share1, dtype=np.float64) * scale).astype(np.int64)
        return (s0, s1)
    if _is_share_like(x):
        s0 = np.round(np.asarray(getattr(x, "share0"), dtype=np.float64) * scale).astype(np.int64)
        s1 = np.round(np.asarray(getattr(x, "share1"), dtype=np.float64) * scale).astype(np.int64)
        return (s0, s1)
    s0 = np.round(np.asarray(x, dtype=np.float64) * scale).astype(np.int64)
    s1 = np.zeros_like(s0, dtype=np.int64)
    return (s0, s1)


def _ctx_nslots(ctx: BridgeContext) -> int:
    n = int(getattr(ctx, "nslots", 0))
    if n <= 0:
        raise TypeError("Bridge context must expose a positive integer 'nslots' property.")
    return n


def _ctx_encrypt(ctx: BridgeContext, arr: np.ndarray, *, encrypt_level: int | None = None) -> BridgeCipher:
    if encrypt_level is None:
        return ctx.encorypt(arr.astype(np.complex128))
    try:
        return ctx.encorypt(arr.astype(np.complex128), level=encrypt_level)
    except TypeError:
        return ctx.encorypt(arr.astype(np.complex128))


def _cipher_decrypt(ctx: BridgeContext, ct: BridgeCipher) -> np.ndarray:
    with ctx.decorypt_scope():
        return ct.decorypt(copy=True, readonly=True)


def _maybe_trim_for_c2m(ctx: BridgeContext, ct: BridgeCipher) -> BridgeCipher:

    if os.environ.get("ENCFORMER_DISABLE_MTO", "0") not in ("0", "", "false", "False"):
        return ct
    trim = getattr(ctx, "mod_switch_to_min_for_c2m", None)
    if trim is None:
        return ct
    try:
        return trim(ct)
    except Exception:
        return ct


def real_ckks_to_mpc(
    ctx: BridgeContext, ct: BridgeCipher, *, dtype: np.dtype = np.float64, comm_stats=None, rng=None
) -> PlainShare:
    q_conv, ring_mod, scale = _conv_params()
    if rng is None:
        rng = np.random.default_rng()
    nslots = _ctx_nslots(ctx)
    ct = _maybe_trim_for_c2m(ctx, ct)
    dec = _cipher_decrypt(ctx, ct)
    x_int = np.round(dec.real * scale).astype(np.int64)
    r = rng.integers(0, q_conv, size=nslots, dtype=np.int64)
    masked = _mod(x_int + r, q_conv).astype(np.float64) / scale
    ct_masked = _ctx_encrypt(ctx, masked.astype(np.complex128))
    x_plus_r = _cipher_decrypt(ctx, ct_masked)
    t0_q = np.round(x_plus_r.real * scale).astype(np.int64)
    t1_q = _mod(-r, q_conv)
    t0_ring = _mod(_center_lift(t0_q, q_conv), ring_mod)
    t1_ring = _mod(_center_lift(t1_q, q_conv), ring_mod)
    s0 = _center_lift(t0_ring, ring_mod).astype(dtype) / scale
    s1 = _center_lift(t1_ring, ring_mod).astype(dtype) / scale
    if comm_stats is not None:
        comm_stats.add_bridge_c2m(1)
    return PlainShare(s0, s1, scale=float(scale))


def real_mpc_to_ckks(ctx: BridgeContext, x: Any, *, encrypt_level: int | None = None, comm_stats=None) -> BridgeCipher:
    q_conv, ring_mod, scale = _conv_params()
    x0, x1 = _to_int_share(x, scale=scale)
    x0_ring = _mod(x0, ring_mod)
    x1_ring = _mod(x1, ring_mod)
    t0_q = _mod(_center_lift(x0_ring, ring_mod), q_conv)
    t1_q = _mod(_center_lift(x1_ring, ring_mod), q_conv)
    v0 = _center_lift(t0_q, q_conv).astype(np.float64) / scale
    v1 = _center_lift(t1_q, q_conv).astype(np.float64) / scale
    if comm_stats is not None:
        comm_stats.add_bridge_m2c(1)
    ct0 = _ctx_encrypt(ctx, v0.astype(np.complex128), encrypt_level=encrypt_level)
    return ct0.add(_ctx_encrypt(ctx, v1.astype(np.complex128)))


def complex_ckks_to_mpc(
    ctx: BridgeContext, ct: BridgeCipher, *, dtype: np.dtype = np.float64, comm_stats=None, rng=None
) -> tuple[PlainShare, PlainShare]:
    q_conv, ring_mod, scale = _conv_params()
    if rng is None:
        rng = np.random.default_rng()
    nslots = _ctx_nslots(ctx)
    ct = _maybe_trim_for_c2m(ctx, ct)
    dec = _cipher_decrypt(ctx, ct)
    x_int = np.round(dec.real * scale).astype(np.int64)
    y_int = np.round(dec.imag * scale).astype(np.int64)
    r_re = rng.integers(0, q_conv, size=nslots, dtype=np.int64)
    r_im = rng.integers(0, q_conv, size=nslots, dtype=np.int64)
    masked = _mod(x_int + r_re, q_conv).astype(np.float64) / scale + 1j * (
        _mod(y_int + r_im, q_conv).astype(np.float64) / scale
    )
    ct_masked = _ctx_encrypt(ctx, masked.astype(np.complex128))
    x_plus_r = _cipher_decrypt(ctx, ct_masked)
    t0_re_q = np.round(x_plus_r.real * scale).astype(np.int64)
    t0_im_q = np.round(x_plus_r.imag * scale).astype(np.int64)
    t1_re_q = _mod(-r_re, q_conv)
    t1_im_q = _mod(-r_im, q_conv)
    t0_re_ring = _mod(_center_lift(t0_re_q, q_conv), ring_mod)
    t1_re_ring = _mod(_center_lift(t1_re_q, q_conv), ring_mod)
    t0_im_ring = _mod(_center_lift(t0_im_q, q_conv), ring_mod)
    t1_im_ring = _mod(_center_lift(t1_im_q, q_conv), ring_mod)
    s0_re = _center_lift(t0_re_ring, ring_mod).astype(dtype) / scale
    s1_re = _center_lift(t1_re_ring, ring_mod).astype(dtype) / scale
    s0_im = _center_lift(t0_im_ring, ring_mod).astype(dtype) / scale
    s1_im = _center_lift(t1_im_ring, ring_mod).astype(dtype) / scale
    sh_re = PlainShare(s0_re, s1_re, scale=float(scale))
    sh_im = PlainShare(s0_im, s1_im, scale=float(scale))
    if comm_stats is not None:
        comm_stats.add_bridge_c2m(1)
    return (sh_re, sh_im)


def complex_mpc_to_ckks(
    ctx: BridgeContext, re: Any, im: Any, *, encrypt_level: int | None = None, comm_stats=None
) -> BridgeCipher:
    q_conv, ring_mod, scale = _conv_params()
    re0, re1 = _to_int_share(re, scale=scale)
    im0, im1 = _to_int_share(im, scale=scale)
    re0_ring = _mod(re0, ring_mod)
    re1_ring = _mod(re1, ring_mod)
    im0_ring = _mod(im0, ring_mod)
    im1_ring = _mod(im1, ring_mod)
    t0_re_q = _mod(_center_lift(re0_ring, ring_mod), q_conv)
    t1_re_q = _mod(_center_lift(re1_ring, ring_mod), q_conv)
    t0_im_q = _mod(_center_lift(im0_ring, ring_mod), q_conv)
    t1_im_q = _mod(_center_lift(im1_ring, ring_mod), q_conv)
    v0 = _center_lift(t0_re_q, q_conv).astype(np.float64) / scale + 1j * (
        _center_lift(t0_im_q, q_conv).astype(np.float64) / scale
    )
    v1 = _center_lift(t1_re_q, q_conv).astype(np.float64) / scale + 1j * (
        _center_lift(t1_im_q, q_conv).astype(np.float64) / scale
    )
    if comm_stats is not None:
        comm_stats.add_bridge_m2c(1)
    ct0 = _ctx_encrypt(ctx, v0.astype(np.complex128), encrypt_level=encrypt_level)
    return ct0.add(_ctx_encrypt(ctx, v1.astype(np.complex128)))
