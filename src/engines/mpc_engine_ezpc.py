from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from src.engines.mpc_batch_method import BatchMethodState, load_batch_method_config
from src.engines.mpc_gelu_secure import (
    SecureGeluConfig,
    build_plain_algorithm4_constants,
    load_secure_gelu_config,
    quantize_real_array,
)

_ezpc_env_path = os.getenv("EZPC_PYTHONPATH", "").strip()
if _ezpc_env_path and _ezpc_env_path not in sys.path:
    sys.path.insert(0, _ezpc_env_path)

try:
    import ezpc_sci as sci  # type: ignore[import-untyped]
except Exception as e:  # pragma: no cover
    sci = None
    _SCI_IMPORT_ERR: Optional[Exception] = e
else:
    _SCI_IMPORT_ERR = None


def _require_sci() -> None:

    if sci is None:
        raise ImportError(
            "ezpc_sci is not importable. Build EzPC/SCI with "
            "ENABLE_PYTHON_BINDING=ON and set EZPC_PYTHONPATH to the "
            "build output directory."
        ) from _SCI_IMPORT_ERR


class _EzPCEmulatedOps:
    def __init__(self, ring_bits: int = 43, scale_bits: int = 13):
        self.ring_bits = ring_bits
        self.scale_bits = scale_bits
        self._mod = 1 << ring_bits
        self._half = 1 << (ring_bits - 1)
        self._scale = 1 << scale_bits

    def _wrap(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.int64)
        mod = np.mod(arr, self._mod).astype(np.int64, copy=False)
        return np.where(mod >= self._half, mod - self._mod, mod).astype(np.int64, copy=False)

    def _trunc(self, x: np.ndarray) -> np.ndarray:

        arr = np.asarray(x, dtype=np.int64)
        out = np.empty_like(arr, dtype=np.int64)
        pos = arr >= 0
        out[pos] = (arr[pos] + self._scale // 2) // self._scale
        out[~pos] = -((-arr[~pos] + self._scale // 2) // self._scale)
        return self._wrap(out)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self._wrap(np.rint(np.asarray(x, dtype=np.float64) * self._scale).astype(np.int64))

    def decode(self, x: np.ndarray) -> np.ndarray:
        return self._wrap(x).astype(np.float64) / float(self._scale)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._trunc(self._wrap(a) * self._wrap(b))

    def mul_const(self, a: np.ndarray, c: int) -> np.ndarray:
        return self._trunc(self._wrap(a) * int(c))

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._wrap(self._wrap(a) + self._wrap(b))

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._wrap(self._wrap(a) - self._wrap(b))

    def neg(self, a: np.ndarray) -> np.ndarray:
        return self._wrap(-self._wrap(a))

    def cmp_lt(self, a: np.ndarray, b: np.ndarray | int) -> np.ndarray:

        return (self._wrap(a) < self._wrap(np.broadcast_to(np.int64(b), np.shape(a)))).astype(np.int64)

    def mux(self, val: np.ndarray, bit: np.ndarray) -> np.ndarray:

        return self._wrap(self._wrap(val) * np.asarray(bit, dtype=np.int64))

    def reciprocal_approx(self, x: np.ndarray, *, iters: int = 3) -> np.ndarray:

        xf = self.decode(x).astype(np.float64)
        safe = np.where(np.abs(xf) < 1e-12, 1e-12, xf)
        result = 1.0 / safe
        return self.encode(result)

    def sqrt_approx(self, x: np.ndarray, *, iters: int = 3) -> np.ndarray:

        xf = self.decode(x).astype(np.float64)
        result = np.sqrt(np.maximum(xf, 0.0))
        return self.encode(result)


class _EzPCGeluOps:
    def __init__(self, emulated: _EzPCEmulatedOps):
        self._ops = emulated

    def ewmulcc(self, u, v):
        return self._ops.mul(u, v)

    def ewmulcp(self, u, c):
        return self._ops.mul_const(u, int(c))

    def cmp(self, a, b):
        aa = np.broadcast_to(np.asarray(a, dtype=np.int64), np.shape(b) if np.ndim(a) == 0 else np.shape(a))
        bb = np.broadcast_to(np.asarray(b, dtype=np.int64), np.shape(a) if np.ndim(b) == 0 else np.shape(b))
        return self._ops.cmp_lt(aa, bb)

    def mux(self, val, bit):
        return self._ops.mux(val, bit)

    def add(self, a, b):
        return self._ops.add(a, b)

    def addc(self, a, c):
        return self._ops.add(a, np.broadcast_to(np.int64(c), np.shape(a)))

    def neg(self, a):
        return self._ops.neg(a)

    def xor_bits(self, a, b):
        aa = np.asarray(a, dtype=np.int64)
        bb = np.asarray(b, dtype=np.int64)
        return aa + bb - 2 * aa * bb


class EzPCMpcEngine:
    name = "ezpc"

    def __init__(
        self, *, mode: str = "auto", role: str = "server", address: str = "127.0.0.1", port: int | None = None
    ):
        self._mode = self._resolve_mode(mode)

        self._role = os.environ.get("MPC_EZPC_ROLE", role)
        self._address = os.environ.get("MPC_EZPC_ADDRESS", address)
        self._port = int(os.environ.get("MPC_EZPC_PORT", str(port if port is not None else 32000)))
        self._batch_cfg = load_batch_method_config()
        self._batch_state = BatchMethodState(self._batch_cfg)
        self._gelu_cfg = load_secure_gelu_config()
        self.comm_stats = None
        self._sci_ctx = None

        if self._mode == "native":
            _require_sci()

        self._ops = _EzPCEmulatedOps(
            ring_bits=self._gelu_cfg.ring_bits,
            scale_bits=self._gelu_cfg.scale_bits,
        )

        self._emu_fp = os.environ.get("MPC_EMULATED_FIXEDPOINT", "0") == "1"

    def set_running_denominators(self, denoms: dict) -> None:

        for key, val in denoms.items():
            if val is not None:
                self._batch_state.buffers[key] = val
            else:
                self._batch_state.buffers.pop(key, None)

    @staticmethod
    def _resolve_mode(mode: str) -> str:
        m = (mode or os.environ.get("MPC_EZPC_MODE", "auto")).strip().lower()
        if m == "native":
            return "native"
        if m == "emulated":
            return "emulated"

        return "native" if sci is not None else "emulated"

    def _get_sci_ctx(self):

        if self._sci_ctx is None:
            self._sci_ctx = sci.SCIContext(
                role=0 if self._role == "server" else 1,
                address=self._address,
                port=self._port,
            )
        return self._sci_ctx

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def backend_mode(self) -> str:
        return self._mode

    def _record_rounds(self, op: str, n: int) -> None:
        if self.comm_stats is not None:
            self.comm_stats.add_mpc_rounds(op, n)

    def softmax_rows(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        bpmax_on = self._batch_cfg.enabled and self._batch_cfg.enable_bpmax

        if self._mode == "native":
            if bpmax_on:
                return self._softmax_bpmax_native(x_arr, head_index=head_index)
            return self._softmax_native(x_arr, head_index=head_index)

        if bpmax_on:
            return self._softmax_bpmax_emulated(x_arr, head_index=head_index)

        x_shifted = x_arr - x_arr.max(axis=1, keepdims=True)
        y = np.exp(x_shifted)
        self._record_rounds("softmax", 3)
        return y / (y.sum(axis=1, keepdims=True) + 1e-12)

    def _softmax_bpmax_native(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:

        p = int(self._batch_cfg.p)
        den = None
        if self._batch_cfg.inference:
            den = self._batch_state.get_bpmax_den(x.shape[0], head_index=head_index)
        if den is None:
            z = np.maximum(x + float(self._batch_cfg.c), 0.0)
            pw = z.copy()
            for _ in range(1, p):
                pw = pw * z
            den = pw.sum(axis=1, keepdims=True)
        inv_rd = 1.0 / (np.asarray(den, dtype=np.float64).reshape(-1) + float(self._batch_cfg.eps))

        safe_floor = -(2.0 ** (self._gelu_cfg.ring_bits - self._gelu_cfg.scale_bits - 4))
        x_safe = np.maximum(x.astype(np.float64), safe_floor)
        result = sci.bpmax_2pc(
            self._get_sci_ctx(),
            x_safe,
            inv_rd.astype(np.float64),
            c=float(self._batch_cfg.c),
            p=p,
            scale_bits=self._gelu_cfg.scale_bits,
            ring_bits=self._gelu_cfg.ring_bits,
        )
        self._record_rounds("bpmax", 3)
        return np.asarray(result, dtype=np.float64)

    def _softmax_native(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:

        result = sci.softmax_2pc(
            self._get_sci_ctx(),
            x.astype(np.float64),
            scale_bits=self._gelu_cfg.scale_bits,
            ring_bits=self._gelu_cfg.ring_bits,
        )
        self._record_rounds("softmax", 3)
        return np.asarray(result, dtype=np.float64)

    def _softmax_bpmax_emulated(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:
        p = int(self._batch_cfg.p)
        c = float(self._batch_cfg.c)
        if self._emu_fp:
            ops = self._ops
            safe_floor = -(2.0 ** (ops.ring_bits - ops.scale_bits - 4))
            Xq = ops.encode(np.maximum(x, safe_floor))
            Zq = ops._wrap(Xq + int(round(c * ops._scale)))
            z = ops.mux(Zq, 1 - ops.cmp_lt(Zq, 0))
            pw_q = z
            for _ in range(1, p):
                pw_q = ops.mul(pw_q, z)
            pw = ops.decode(pw_q)
        else:
            z = np.maximum(x + c, 0.0)
            pw = z.copy()
            for _ in range(1, p):
                pw = pw * z
        den_cur = pw.sum(axis=1, keepdims=True)

        if self._batch_cfg.inference:
            den = self._batch_state.get_bpmax_den(x.shape[0], head_index=head_index)
            if den is None:
                den = den_cur
        else:
            den = den_cur
            h = head_index if head_index is not None else 0
            self._batch_state.update_bpmax_head(h, den_cur)

        inv = 1.0 / (den + float(self._batch_cfg.eps))
        self._record_rounds("bpmax", 3)
        return pw * inv

    def layer_norm(
        self,
        x: np.ndarray,
        *,
        eps: float = 1e-5,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        ln_tag: str | None = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        batchln_on = self._batch_cfg.enabled and self._batch_cfg.enable_batchln

        if self._mode == "native" and batchln_on:
            y = self._layer_norm_mbnorm_native(x_arr, gamma=gamma, beta=beta, ln_tag=ln_tag)
            self._record_rounds("batchln", 0)
            return y

        if self._mode == "native":
            y = self._layer_norm_native(x_arr, eps=eps, ln_tag=ln_tag)
        elif batchln_on:
            y = self._layer_norm_batchln_emulated(x_arr, eps=eps, ln_tag=ln_tag)
        else:
            mean = x_arr.mean(axis=1, keepdims=True)
            xc = x_arr - mean
            var = x_arr.var(axis=1, keepdims=True)
            y = xc / np.sqrt(var + eps)

        if gamma is not None:
            y = y * np.asarray(gamma, dtype=np.float64)
        if beta is not None:
            y = y + np.asarray(beta, dtype=np.float64)

        self._record_rounds("batchln" if batchln_on else "layernorm", 0 if batchln_on else 2)
        return y

    def _layer_norm_mbnorm_native(self, x: np.ndarray, *, gamma, beta, ln_tag) -> np.ndarray:

        cols = x.shape[1]
        den = None
        if self._batch_cfg.inference:
            den = self._batch_state.get_ln_den(x.shape[0], ln_tag=ln_tag)
        if den is None:
            xc = x - x.mean(axis=1, keepdims=True)
            den = np.sqrt((xc * xc).mean(axis=1, keepdims=True) + float(self._batch_cfg.eps))
        inv_rd = 1.0 / (
            float(self._batch_cfg.ln_l) * np.asarray(den, dtype=np.float64).reshape(-1) + float(self._batch_cfg.eps)
        )
        g = np.asarray(gamma, dtype=np.float64) if gamma is not None else np.ones(cols)
        b = np.asarray(beta, dtype=np.float64) if beta is not None else np.zeros(cols)
        result = sci.mbnorm_2pc(
            self._get_sci_ctx(),
            x.astype(np.float64),
            inv_rd.astype(np.float64),
            g.astype(np.float64),
            b.astype(np.float64),
            scale_bits=self._gelu_cfg.scale_bits,
            ring_bits=self._gelu_cfg.ring_bits,
        )
        return np.asarray(result, dtype=np.float64)

    def _layer_norm_native(self, x: np.ndarray, *, eps: float, ln_tag: str | None) -> np.ndarray:

        result = sci.layer_norm_2pc(
            self._get_sci_ctx(),
            x.astype(np.float64),
            eps=eps,
            scale_bits=self._gelu_cfg.scale_bits,
            ring_bits=self._gelu_cfg.ring_bits,
        )
        return np.asarray(result, dtype=np.float64)

    def _layer_norm_batchln_emulated(self, x: np.ndarray, *, eps: float, ln_tag: str | None) -> np.ndarray:
        mean = x.mean(axis=1, keepdims=True)
        xc = x - mean
        rms_cur = np.sqrt((xc * xc).mean(axis=1, keepdims=True) + eps)

        if self._batch_cfg.inference:
            den = self._batch_state.get_ln_den(x.shape[0], ln_tag=ln_tag)
            if den is None:
                den = rms_cur
        else:
            den = rms_cur
            key = "ln2_running_denominator" if ln_tag == "ln2" else "ln1_running_denominator"
            self._batch_state.update_running(key, rms_cur)

        if self._emu_fp:
            ops = self._ops
            cols = x.shape[1]
            inv_rd = 1.0 / (float(self._batch_cfg.ln_l) * den + float(self._batch_cfg.eps))
            Xq = ops.encode(x)
            rowsum = ops._wrap(Xq.sum(axis=1, keepdims=True))
            colsX = ops._wrap(Xq * cols)
            u = ops.sub(colsX, np.broadcast_to(rowsum, Xq.shape))

            inv_q = ops.encode(np.broadcast_to(inv_rd, u.shape))
            return ops.decode(ops.mul(u, inv_q)) / cols

        return xc / (float(self._batch_cfg.ln_l) * den + float(self._batch_cfg.eps))

    def gelu(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)

        if self._mode == "native":
            y = self._gelu_native(x_arr)
        else:
            y = self._gelu_emulated(x_arr)

        self._record_rounds("gelu", 4)
        return y

    def _gelu_native(self, x: np.ndarray) -> np.ndarray:

        result = sci.gelu_2pc(
            self._get_sci_ctx(),
            x.astype(np.float64),
            scale_bits=self._gelu_cfg.scale_bits,
            ring_bits=self._gelu_cfg.ring_bits,
        )
        return np.asarray(result, dtype=np.float64)

    def _gelu_emulated(self, x: np.ndarray) -> np.ndarray:

        from src.engines.mpc_gelu_secure import secure_gelu_algorithm4_split

        cfg = self._gelu_cfg
        consts = build_plain_algorithm4_constants(cfg)
        ops = _EzPCGeluOps(self._ops)

        x_q = self._ops.encode(x)

        clip_lo = int(np.rint(-cfg.threshold * self._ops._scale))
        clip_hi = int(np.rint(cfg.threshold * self._ops._scale))
        x_clipped = np.clip(x_q, clip_lo, clip_hi)

        y_q = secure_gelu_algorithm4_split(x_q, x_clipped, x_q, ops=ops, consts=consts)
        return self._ops.decode(y_q)

    def softmax_rows_shares(self, shares: "ShareMatrix", *, head_index: int | None = None) -> "ShareMatrix":
        from src.share_types import ShareMatrix

        x = shares.reconstruct()
        y = self.softmax_rows(x, head_index=head_index)
        return ShareMatrix(y, np.zeros_like(y))

    def layer_norm_shares(
        self,
        shares: "ShareMatrix",
        *,
        eps: float = 1e-5,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        ln_tag: str | None = None,
    ) -> "ShareMatrix":
        from src.share_types import ShareMatrix

        x = shares.reconstruct()
        try:
            y = self.layer_norm(x, eps=eps, gamma=gamma, beta=beta, ln_tag=ln_tag)
        except TypeError:
            y = self.layer_norm(x, eps=eps, gamma=gamma, beta=beta)
        return ShareMatrix(y, np.zeros_like(y))

    def gelu_shares(self, shares: "ShareMatrix") -> "ShareMatrix":
        from src.share_types import ShareMatrix

        x = shares.reconstruct()
        y = self.gelu(x)
        return ShareMatrix(y, np.zeros_like(y))

    def gelu_preeval_shares(
        self,
        x_shares: "ShareMatrix",
        f0_shares: "ShareMatrix | None" = None,
        f1_shares: "ShareMatrix | None" = None,
    ) -> "ShareMatrix":
        from src.share_types import ShareMatrix

        x = x_shares.reconstruct()
        f0 = f0_shares.reconstruct() if f0_shares is not None else None
        f1 = f1_shares.reconstruct() if f1_shares is not None else None

        y = self.gelu(x)
        return ShareMatrix(y, np.zeros_like(y))
