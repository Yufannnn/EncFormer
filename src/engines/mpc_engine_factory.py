from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.engines.mpc_engine_plain import PlainMpcEngine


@runtime_checkable
class MpcEngine(Protocol):
    name: str

    def softmax_rows(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray: ...
    def layer_norm(
        self,
        x: np.ndarray,
        *,
        eps: float = 1e-5,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        ln_tag: str | None = None,
    ) -> np.ndarray: ...
    def gelu(self, x: np.ndarray) -> np.ndarray: ...
    def set_running_denominators(self, denoms: dict[str, np.ndarray]) -> None: ...


_PIPELINE_MPC_MAP: dict[str, str] = {
    "desilo-crypten": "crypten",
    "phantom-ezpc": "ezpc",
}

_PIPELINE_CKKS_MAP: dict[str, str] = {
    "desilo-crypten": "desilo",
    "phantom-ezpc": "phantom_native",
}


def resolve_pipeline() -> tuple[str | None, str | None]:

    pipeline = os.environ.get("PIPELINE", "").strip().lower()
    if not pipeline:
        return (None, None)
    if pipeline not in _PIPELINE_MPC_MAP:
        raise ValueError(f"Unknown PIPELINE '{pipeline}'. Expected one of: {', '.join(sorted(_PIPELINE_MPC_MAP))}.")
    return (_PIPELINE_CKKS_MAP[pipeline], _PIPELINE_MPC_MAP[pipeline])


def get_mpc_engine(kind: str | None = None) -> MpcEngine:

    _, pipeline_mpc = resolve_pipeline()
    mode = (kind or pipeline_mpc or os.environ.get("MPC_ENGINE", "plain")).strip().lower()

    if mode == "plain":
        return PlainMpcEngine()
    if mode == "crypten":
        from src.engines.mpc_engine_crypten import CrypTenMpcEngine

        device = os.environ.get("MPC_DEVICE", "auto")
        return CrypTenMpcEngine(device=device)
    if mode in {"mixed", "hybrid", "plain-crypten-gelu"}:
        from src.engines.mpc_engine_mixed import MixedMpcEngine

        device = os.environ.get("MPC_DEVICE", "auto")
        return MixedMpcEngine(cryp_device=device)
    if mode == "ezpc":
        from src.engines.mpc_engine_ezpc import EzPCMpcEngine

        ezpc_mode = os.environ.get("MPC_EZPC_MODE", "auto")
        role = os.environ.get("MPC_EZPC_ROLE", "server")
        return EzPCMpcEngine(mode=ezpc_mode, role=role)
    raise ValueError(f"Unsupported MPC engine '{mode}'. Expected one of: plain, crypten, mixed, ezpc.")
