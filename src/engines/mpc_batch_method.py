from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BatchMethodConfig:
    mode: str
    inference: bool
    p: int
    c: float
    eps: float
    ln_l: float
    rd_path: str

    @property
    def enabled(self) -> bool:
        return self.mode in {"on", "encformer", "batch"}

    @property
    def enable_bpmax(self) -> bool:
        return self.mode in {"on", "encformer", "batch", "bpmax"}

    @property
    def enable_batchln(self) -> bool:
        return self.mode in {"on", "encformer", "batch", "batchln"}


def load_batch_method_config() -> BatchMethodConfig:
    mode = os.environ.get("MPC_BATCH_METHOD", "off").strip().lower()
    inference = os.environ.get("MPC_BATCH_INFERENCE", "1").strip() not in {"0", "false", "False"}
    p = int(os.environ.get("MPC_BP_P", "5"))
    c = float(os.environ.get("MPC_BP_C", "5"))
    eps = float(os.environ.get("MPC_BATCH_EPS", "1e-12"))
    ln_l = float(os.environ.get("MPC_BATCHLN_L", "1.0"))
    rd_path = os.environ.get("MPC_BATCH_RD_PATH", "").strip()
    return BatchMethodConfig(
        mode=mode,
        inference=inference,
        p=p,
        c=c,
        eps=eps,
        ln_l=ln_l,
        rd_path=rd_path,
    )


class BatchMethodState:
    def __init__(self, cfg: BatchMethodConfig):
        self.cfg = cfg
        self.buffers: Dict[str, np.ndarray] = {}
        if cfg.rd_path:
            self._load_buffers(cfg.rd_path)

    def _load_buffers(self, p: str) -> None:
        path = Path(p)
        if not path.exists():
            return
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=False)
            for k in data.files:
                self.buffers[k] = np.asarray(data[k], dtype=np.float64)
            return
        if path.suffix.lower() == ".npy":
            self.buffers["running_denominator"] = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
            return
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            for k, v in obj.items():
                self.buffers[str(k)] = np.asarray(v, dtype=np.float64)
            return

    def _find_key(self, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in self.buffers:
                return c
        for c in candidates:
            for k in self.buffers:
                if k.endswith(c):
                    return k
        return None

    @staticmethod
    def _reshape_row_den(arr: np.ndarray, rows: int, *, head_index: int | None = None) -> Optional[np.ndarray]:
        a = np.asarray(arr, dtype=np.float64)

        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]

        if a.ndim >= 2 and head_index is not None and a.shape[0] < rows and a.shape[1] >= rows:
            if a.shape[0] > head_index:
                a = a[head_index]
        elif a.ndim >= 3:
            if head_index is not None and a.shape[0] > head_index:
                a = a[head_index]
            elif a.shape[0] < rows and a.shape[1] >= rows:
                a = a[0]
        if a.ndim == 0:
            return np.full((rows, 1), float(a), dtype=np.float64)
        if a.ndim == 1:
            if a.size == 1:
                return np.full((rows, 1), float(a[0]), dtype=np.float64)
            if a.size >= rows:
                return a[:rows].reshape(rows, 1).astype(np.float64, copy=False)
            return None
        if a.ndim >= 2:
            if a.shape[0] == 1:
                base = a[:, :1].reshape(1, 1)
                return np.repeat(base, rows, axis=0).astype(np.float64, copy=False)
            if a.shape[0] >= rows:
                return a[:rows, :1].astype(np.float64, copy=False)
        return None

    def get_bpmax_den(self, rows: int, *, head_index: int | None = None) -> Optional[np.ndarray]:
        key = self._find_key(
            [
                "bpmax_running_denominator",
                "running_denominator",
                "bpmax",
                "attention.self.batch_method.running_denominator",
            ]
        )
        if key is None:
            return None
        return self._reshape_row_den(self.buffers[key], rows, head_index=head_index)

    def get_ln_den(self, rows: int, *, ln_tag: str | None = None) -> Optional[np.ndarray]:
        if ln_tag == "ln2":
            cands = [
                "ln2_running_denominator",
                "output.LayerNorm.rd.running_denominator",
                "LayerNorm2.rd.running_denominator",
            ]
        else:
            cands = [
                "ln1_running_denominator",
                "attention.output.LayerNorm.rd.running_denominator",
                "LayerNorm1.rd.running_denominator",
            ]
        key = self._find_key(cands)
        if key is None:
            return None
        return self._reshape_row_den(self.buffers[key], rows)

    def update_running_max(self, key: str, cur: np.ndarray) -> None:
        c = np.asarray(cur, dtype=np.float64)
        if key not in self.buffers:
            self.buffers[key] = c.copy()
            return
        prev = np.asarray(self.buffers[key], dtype=np.float64)
        if prev.shape != c.shape:
            self.buffers[key] = c.copy()
            return
        self.buffers[key] = np.maximum(prev, c)

    def update_bpmax_head(self, head: int, cur: np.ndarray) -> None:
        key = "bpmax_running_denominator"
        c = np.asarray(cur, dtype=np.float64).squeeze()
        rows = c.shape[0]
        if key not in self.buffers or self.buffers[key].ndim != 2:
            max_h = max(head + 1, 12)
            self.buffers[key] = np.zeros((max_h, rows), dtype=np.float64)
        buf = self.buffers[key]
        if head >= buf.shape[0]:
            new_buf = np.zeros((head + 1, rows), dtype=np.float64)
            new_buf[: buf.shape[0]] = buf
            self.buffers[key] = new_buf
            buf = new_buf
        buf[head, :rows] = c[:rows]

    def update_running(self, key: str, cur: np.ndarray) -> None:
        c = np.asarray(cur, dtype=np.float64)
        self.buffers[key] = c.copy()
