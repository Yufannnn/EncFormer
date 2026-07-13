#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.engines.mpc_batch_method import BatchMethodState, load_batch_method_config
from src.engines.mpc_gelu_secure import secure_gelu_plain_fixedpoint


@dataclass(frozen=True)
class PlainShare:
    share0: np.ndarray
    share1: np.ndarray
    scale: Optional[float] = None

    def reconstruct(self) -> np.ndarray:
        return self.share0 + self.share1


class PlainMpcEngine:
    name = "plain"
    device = "cpu"

    def __init__(self):
        self._batch_cfg = load_batch_method_config()
        self._batch_state = BatchMethodState(self._batch_cfg)
        self.comm_stats = None

    def set_running_denominators(self, denoms: dict[str, np.ndarray | None]) -> None:

        for key, val in denoms.items():
            if val is not None:
                self._batch_state.buffers[key] = val
            else:
                self._batch_state.buffers.pop(key, None)

    def _pow_by_int(self, x: np.ndarray, p: int) -> np.ndarray:
        if p <= 0:
            return np.ones_like(x, dtype=np.float64)
        out = x.astype(np.float64, copy=True)
        for _ in range(1, p):
            out = out * x
        return out

    def _record_rounds(self, op: str, n: int) -> None:
        cs = getattr(self, "comm_stats", None)
        if cs is not None:
            cs.add_mpc_rounds(op, n)

    def softmax_rows(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._batch_cfg.enabled and self._batch_cfg.enable_bpmax:
            z = np.maximum(x + float(self._batch_cfg.c), 0.0)
            pw = self._pow_by_int(z, int(self._batch_cfg.p))
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
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        self._record_rounds("softmax", 3)
        return y / (y.sum(axis=1, keepdims=True) + 1e-12)

    def layer_norm(
        self,
        x: np.ndarray,
        *,
        eps: float = 1e-5,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        ln_tag: str | None = None,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=1, keepdims=True)
        xc = x - mean
        if self._batch_cfg.enabled and self._batch_cfg.enable_batchln:
            rms_cur = np.sqrt((xc * xc).mean(axis=1, keepdims=True) + float(eps))
            if self._batch_cfg.inference:
                den = self._batch_state.get_ln_den(x.shape[0], ln_tag=ln_tag)
                if den is None:
                    den = rms_cur
            else:
                den = rms_cur
                key = "ln2_running_denominator" if ln_tag == "ln2" else "ln1_running_denominator"
                self._batch_state.update_running(key, rms_cur)
            y = xc / (float(self._batch_cfg.ln_l) * den + float(self._batch_cfg.eps))
        else:
            var = x.var(axis=1, keepdims=True)
            y = xc / np.sqrt(var + eps)
        if gamma is not None:
            y = y * np.asarray(gamma, dtype=np.float64)
        if beta is not None:
            y = y + np.asarray(beta, dtype=np.float64)
        rounds = 0 if (self._batch_cfg.enabled and self._batch_cfg.enable_batchln) else 2
        self._record_rounds("batchln" if rounds == 0 else "layernorm", rounds)
        return y

    def gelu(self, x: np.ndarray) -> np.ndarray:
        self._record_rounds("gelu", 4)
        return secure_gelu_plain_fixedpoint(np.asarray(x, dtype=np.float64))

    def gelu_preeval(self, x: np.ndarray, f0: np.ndarray | None = None, f1: np.ndarray | None = None) -> np.ndarray:

        from src.engines.mpc_gelu_secure import (
            PlainFixedPointOps,
            build_plain_algorithm4_constants,
            load_secure_gelu_config,
            precompute_f0_f1_fixedpoint,
            secure_gelu_preeval_select,
        )

        self._record_rounds("gelu_preeval", 1)
        cfg = load_secure_gelu_config()
        ops = PlainFixedPointOps(cfg)
        consts = build_plain_algorithm4_constants(cfg)
        x_arr = np.asarray(x, dtype=np.float64)
        x_q = ops.encode(x_arr)
        if f0 is None or f1 is None:
            f0_q, f1_q = precompute_f0_f1_fixedpoint(x_arr, cfg)
        else:
            f0_q = ops.encode(np.asarray(f0, dtype=np.float64))
            f1_q = ops.encode(np.asarray(f1, dtype=np.float64))
        y_q = secure_gelu_preeval_select(x_q, f0_q, f1_q, x_q, ops=ops, consts=consts)
        return ops.decode(y_q)

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
        y = self.gelu_preeval(x, f0=f0, f1=f1)
        return ShareMatrix(y, np.zeros_like(y))
