from __future__ import annotations

import numpy as np

from src.engines.mpc_engine_crypten import CrypTenMpcEngine
from src.engines.mpc_engine_plain import PlainMpcEngine


class MixedMpcEngine:
    name = "mixed"

    def __init__(self, *, cryp_device: str = "auto"):
        self._plain = PlainMpcEngine()
        self._crypten_gelu = CrypTenMpcEngine(device=cryp_device)
        self._comm_stats = None

    @property
    def comm_stats(self):
        return self._comm_stats

    @comm_stats.setter
    def comm_stats(self, val):
        self._comm_stats = val
        self._plain.comm_stats = val
        self._crypten_gelu.comm_stats = val

    @property
    def device(self) -> str:
        return self._crypten_gelu.device

    def softmax_rows(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:
        return self._plain.softmax_rows(x, head_index=head_index)

    def layer_norm(
        self,
        x: np.ndarray,
        *,
        eps: float = 1e-5,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        ln_tag: str | None = None,
    ) -> np.ndarray:
        return self._plain.layer_norm(x, eps=eps, gamma=gamma, beta=beta, ln_tag=ln_tag)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return self._crypten_gelu.gelu(x)

    def set_running_denominators(self, denoms: dict) -> None:

        self._plain.set_running_denominators(denoms)
        self._crypten_gelu.set_running_denominators(denoms)
