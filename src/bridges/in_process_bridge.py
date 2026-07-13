from __future__ import annotations

from typing import Any

import numpy as np

from src.bridges.channel import InProcessChannelPair
from src.bridges.ckks_mpc_bridge import (
    BridgeCipher,
    BridgeContext,
    _conv_params,
    _is_share_like,
)
from src.bridges.secure_bridge import Role, SecureBridge
from src.engines.mpc_engine_plain import PlainShare


class InProcessBridge:
    def __init__(self, ctx: BridgeContext, seed: int | None = None) -> None:
        server_ch, client_ch = InProcessChannelPair.create()
        self._server = SecureBridge(Role.SERVER, server_ch, ctx, seed=seed)
        self._client = SecureBridge(Role.CLIENT, client_ch)

    def ckks_to_mpc(
        self,
        ct: BridgeCipher,
        *,
        dtype: np.dtype = np.float64,
        comm_stats: Any = None,
    ) -> PlainShare:

        _, _, scale = _conv_params()

        server_share = self._server.ckks_to_mpc(ct, dtype=dtype, comm_stats=comm_stats)
        client_share = self._client.ckks_to_mpc(None, dtype=dtype)
        return PlainShare(client_share, server_share, scale=float(scale))

    def mpc_to_ckks(
        self,
        x: Any,
        *,
        encrypt_level: int | None = None,
        comm_stats: Any = None,
    ) -> BridgeCipher:

        s0, s1 = _extract_shares(x)

        self._client.mpc_to_ckks(s0)
        ct = self._server.mpc_to_ckks(s1, encrypt_level=encrypt_level, comm_stats=comm_stats)
        return ct

    def complex_ckks_to_mpc(
        self,
        ct: BridgeCipher,
        *,
        dtype: np.dtype = np.float64,
        comm_stats: Any = None,
    ) -> tuple[PlainShare, PlainShare]:

        _, _, scale = _conv_params()
        server_re, server_im = self._server.complex_ckks_to_mpc(ct, dtype=dtype, comm_stats=comm_stats)
        client_re, client_im = self._client.complex_ckks_to_mpc(None, dtype=dtype)
        sh_re = PlainShare(client_re, server_re, scale=float(scale))
        sh_im = PlainShare(client_im, server_im, scale=float(scale))
        return sh_re, sh_im

    def complex_mpc_to_ckks(
        self,
        re: Any,
        im: Any,
        *,
        encrypt_level: int | None = None,
        comm_stats: Any = None,
    ) -> BridgeCipher:

        re0, re1 = _extract_shares(re)
        im0, im1 = _extract_shares(im)
        self._client.complex_mpc_to_ckks(re0, im0)
        ct = self._server.complex_mpc_to_ckks(re1, im1, encrypt_level=encrypt_level, comm_stats=comm_stats)
        return ct


def _extract_shares(x: Any) -> tuple[np.ndarray, np.ndarray]:

    if isinstance(x, PlainShare):
        return np.asarray(x.share0, dtype=np.float64), np.asarray(x.share1, dtype=np.float64)
    if _is_share_like(x):
        return np.asarray(getattr(x, "share0"), dtype=np.float64), np.asarray(getattr(x, "share1"), dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    return arr, np.zeros_like(arr)
