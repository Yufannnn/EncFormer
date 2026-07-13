from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

COEFF_A = 0.020848611754127593
COEFF_B = -0.18352506127082727
COEFF_C = 0.5410550166368381
COEFF_D = -0.03798164612714154
COEFF_E = 0.001620808531841547


@dataclass(frozen=True)
class SecureGeluConfig:
    ring_bits: int
    scale_bits: int
    threshold: float

    @property
    def ring_mod(self) -> int:
        return 1 << int(self.ring_bits)

    @property
    def ring_half(self) -> int:
        return 1 << int(self.ring_bits - 1)

    @property
    def scale(self) -> int:
        return 1 << int(self.scale_bits)


@dataclass(frozen=True)
class Algorithm4Constants:
    a: Any
    b: Any
    c: Any
    p: Any
    m: Any
    e: Any
    neg_thresh: Any
    pos_thresh: Any
    zero: Any = 0
    one: Any = 1


def load_secure_gelu_config() -> SecureGeluConfig:
    ring_bits = int(os.environ.get("MPC_GELU_RING_BITS", "43"))
    scale_bits = int(os.environ.get("MPC_GELU_SCALE_BITS", "13"))
    threshold = float(os.environ.get("MPC_GELU_THRESH", "2.7"))
    if ring_bits <= 1:
        raise ValueError("MPC_GELU_RING_BITS must be > 1.")
    if scale_bits <= 0:
        raise ValueError("MPC_GELU_SCALE_BITS must be > 0.")
    if ring_bits >= 62:
        raise ValueError("MPC_GELU_RING_BITS must be < 62 for int64-safe emulation.")
    return SecureGeluConfig(ring_bits=ring_bits, scale_bits=scale_bits, threshold=threshold)


def quantize_const_int(value: float, cfg: SecureGeluConfig) -> int:
    return int(np.rint(float(value) * float(cfg.scale)))


def quantize_const_real(value: float, cfg: SecureGeluConfig) -> float:
    return float(quantize_const_int(value, cfg)) / float(cfg.scale)


def quantize_real_array(x: np.ndarray, cfg: SecureGeluConfig) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    q = np.rint(arr * float(cfg.scale)).astype(np.int64)
    return q.astype(np.float64) / float(cfg.scale)


def build_plain_algorithm4_constants(cfg: SecureGeluConfig) -> Algorithm4Constants:
    return Algorithm4Constants(
        a=quantize_const_int(COEFF_A, cfg),
        b=quantize_const_int(COEFF_B, cfg),
        c=quantize_const_int(COEFF_C, cfg),
        p=quantize_const_int(0.5 + COEFF_D, cfg),
        m=quantize_const_int(0.5 - COEFF_D, cfg),
        e=quantize_const_int(COEFF_E, cfg),
        neg_thresh=quantize_const_int(-cfg.threshold, cfg),
        pos_thresh=quantize_const_int(cfg.threshold, cfg),
        zero=0,
        one=1,
    )


def build_quantized_real_algorithm4_constants(cfg: SecureGeluConfig) -> Algorithm4Constants:
    return Algorithm4Constants(
        a=quantize_const_real(COEFF_A, cfg),
        b=quantize_const_real(COEFF_B, cfg),
        c=quantize_const_real(COEFF_C, cfg),
        p=quantize_const_real(0.5 + COEFF_D, cfg),
        m=quantize_const_real(0.5 - COEFF_D, cfg),
        e=quantize_const_real(COEFF_E, cfg),
        neg_thresh=quantize_const_real(-cfg.threshold, cfg),
        pos_thresh=quantize_const_real(cfg.threshold, cfg),
        zero=0.0,
        one=1.0,
    )


def secure_gelu_algorithm4_split(
    x_cmp: Any,
    x_poly: Any,
    x_out: Any,
    *,
    ops: Any,
    consts: Algorithm4Constants,
) -> Any:

    x2 = ops.ewmulcc(x_poly, x_poly)
    x3 = ops.ewmulcc(x2, x_poly)
    x4 = ops.ewmulcc(x2, x2)

    ax4 = ops.ewmulcp(x4, consts.a)
    bx3 = ops.ewmulcp(x3, consts.b)
    cx2 = ops.ewmulcp(x2, consts.c)
    p = ops.ewmulcp(x_poly, consts.p)
    m = ops.ewmulcp(x_poly, consts.m)

    f0 = ops.add(ax4, ops.neg(bx3))
    f0 = ops.add(f0, cx2)
    f0 = ops.add(f0, m)
    f0 = ops.addc(f0, consts.e)

    f1 = ops.add(ax4, bx3)
    f1 = ops.add(f1, cx2)
    f1 = ops.add(f1, p)
    f1 = ops.addc(f1, consts.e)

    b0 = ops.cmp(x_cmp, consts.neg_thresh)
    b1 = ops.cmp(x_cmp, consts.zero)
    b2 = ops.cmp(consts.pos_thresh, x_cmp)

    z0 = ops.xor_bits(b0, b1)
    z1 = ops.xor_bits(ops.xor_bits(b1, b2), consts.one)
    z2 = b2

    y0 = ops.mux(f0, z0)
    y1 = ops.mux(f1, z1)
    y2 = ops.mux(x_out, z2)
    return ops.add(ops.add(y0, y1), y2)


def secure_gelu_algorithm4(x: Any, *, ops: Any, consts: Algorithm4Constants) -> Any:
    return secure_gelu_algorithm4_split(x, x, x, ops=ops, consts=consts)


def secure_gelu_preeval_select(
    x_cmp: Any,
    f0: Any,
    f1: Any,
    x_out: Any,
    *,
    ops: Any,
    consts: Algorithm4Constants,
) -> Any:

    b0 = ops.cmp(x_cmp, consts.neg_thresh)
    b1 = ops.cmp(x_cmp, consts.zero)
    b2 = ops.cmp(consts.pos_thresh, x_cmp)

    z0 = ops.xor_bits(b0, b1)
    z1 = ops.xor_bits(ops.xor_bits(b1, b2), consts.one)
    z2 = b2

    y0 = ops.mux(f0, z0)
    y1 = ops.mux(f1, z1)
    y2 = ops.mux(x_out, z2)
    return ops.add(ops.add(y0, y1), y2)


class PlainFixedPointOps:
    def __init__(self, cfg: SecureGeluConfig):
        self.cfg = cfg
        self._scale = int(cfg.scale)
        self._mod = int(cfg.ring_mod)
        self._half = int(cfg.ring_half)

    def _wrap(self, x: np.ndarray | int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.int64)
        mod = np.mod(arr, self._mod).astype(np.int64, copy=False)
        return np.where(mod >= self._half, mod - self._mod, mod).astype(np.int64, copy=False)

    def _trunc_keep_scale(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.int64)
        out = np.empty_like(arr, dtype=np.int64)
        pos = arr >= 0
        out[pos] = (arr[pos] + self._scale // 2) // self._scale
        out[~pos] = -((-arr[~pos] + self._scale // 2) // self._scale)
        return self._wrap(out)

    def encode(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        q = np.rint(arr * float(self._scale)).astype(np.int64)
        return self._wrap(q)

    def decode(self, x: np.ndarray) -> np.ndarray:
        return self._wrap(x).astype(np.float64) / float(self._scale)

    def ewmulcc(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        uu = self._wrap(u)
        vv = self._wrap(v)
        prod = uu * vv
        return self._trunc_keep_scale(prod)

    def ewmulcp(self, u: np.ndarray, c: int) -> np.ndarray:
        uu = self._wrap(u)
        prod = uu * int(c)
        return self._trunc_keep_scale(prod)

    def cmp(self, a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        aa = self._wrap(a)
        bb = self._wrap(b)
        return (aa < bb).astype(np.int64)

    def mux(self, val: np.ndarray, bit: np.ndarray) -> np.ndarray:
        return self._wrap(self._wrap(val) * np.asarray(bit, dtype=np.int64))

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._wrap(self._wrap(a) + self._wrap(b))

    def addc(self, a: np.ndarray, c: int) -> np.ndarray:
        return self._wrap(self._wrap(a) + int(c))

    def neg(self, a: np.ndarray) -> np.ndarray:
        return self._wrap(-self._wrap(a))

    def xor_bits(self, a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
        aa = np.asarray(a, dtype=np.int64)
        bb = np.asarray(b, dtype=np.int64)
        return aa + bb - 2 * aa * bb


def secure_gelu_plain_fixedpoint(x: np.ndarray, cfg: SecureGeluConfig | None = None) -> np.ndarray:
    cfg = load_secure_gelu_config() if cfg is None else cfg
    ops = PlainFixedPointOps(cfg)
    consts = build_plain_algorithm4_constants(cfg)
    x_q = ops.encode(x)
    y_q = secure_gelu_algorithm4(x_q, ops=ops, consts=consts)
    return ops.decode(y_q)


def precompute_f0_f1_fixedpoint(x: np.ndarray, cfg: SecureGeluConfig | None = None) -> tuple[np.ndarray, np.ndarray]:

    cfg = load_secure_gelu_config() if cfg is None else cfg
    ops = PlainFixedPointOps(cfg)
    consts = build_plain_algorithm4_constants(cfg)
    x_q = ops.encode(x)

    x2 = ops.ewmulcc(x_q, x_q)
    x3 = ops.ewmulcc(x2, x_q)
    x4 = ops.ewmulcc(x2, x2)
    ax4 = ops.ewmulcp(x4, consts.a)
    bx3 = ops.ewmulcp(x3, consts.b)
    cx2 = ops.ewmulcp(x2, consts.c)
    p = ops.ewmulcp(x_q, consts.p)
    m = ops.ewmulcp(x_q, consts.m)
    f0 = ops.add(ops.add(ops.add(ax4, ops.neg(bx3)), cx2), m)
    f0 = ops.addc(f0, consts.e)
    f1 = ops.add(ops.add(ops.add(ax4, bx3), cx2), p)
    f1 = ops.addc(f1, consts.e)
    return f0, f1


def secure_gelu_preeval_plain_fixedpoint(x: np.ndarray, cfg: SecureGeluConfig | None = None) -> np.ndarray:

    cfg = load_secure_gelu_config() if cfg is None else cfg
    ops = PlainFixedPointOps(cfg)
    consts = build_plain_algorithm4_constants(cfg)
    x_q = ops.encode(x)
    f0_q, f1_q = precompute_f0_f1_fixedpoint(x, cfg)
    y_q = secure_gelu_preeval_select(x_q, f0_q, f1_q, x_q, ops=ops, consts=consts)
    return ops.decode(y_q)


def secure_gelu_piecewise_reference(x: np.ndarray, cfg: SecureGeluConfig | None = None) -> np.ndarray:
    cfg = load_secure_gelu_config() if cfg is None else cfg
    consts = build_quantized_real_algorithm4_constants(cfg)
    xq = quantize_real_array(x, cfg)
    x2 = xq * xq
    x3 = x2 * xq
    x4 = x2 * x2
    f0 = consts.a * x4 - consts.b * x3 + consts.c * x2 + consts.m * xq + consts.e
    f1 = consts.a * x4 + consts.b * x3 + consts.c * x2 + consts.p * xq + consts.e
    b0 = (xq < consts.neg_thresh).astype(np.float64)
    b1 = (xq < consts.zero).astype(np.float64)
    b2 = (consts.pos_thresh < xq).astype(np.float64)
    z0 = b0 + b1 - 2.0 * b0 * b1
    z1p = b1 + b2 - 2.0 * b1 * b2
    z1 = z1p + 1.0 - 2.0 * z1p
    z2 = b2
    y = f0 * z0 + f1 * z1 + xq * z2
    return quantize_real_array(y, cfg)


def selector_bits_public(
    x: np.ndarray, cfg: SecureGeluConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_secure_gelu_config() if cfg is None else cfg
    consts = build_quantized_real_algorithm4_constants(cfg)
    xq = quantize_real_array(x, cfg)
    b0 = (xq < consts.neg_thresh).astype(np.int64)
    b1 = (xq < consts.zero).astype(np.int64)
    b2 = (consts.pos_thresh < xq).astype(np.int64)
    z0 = b0 + b1 - 2 * b0 * b1
    z1p = b1 + b2 - 2 * b1 * b2
    z1 = z1p + 1 - 2 * z1p
    z2 = b2
    return (z0.astype(np.int64), z1.astype(np.int64), z2.astype(np.int64))
