from __future__ import annotations

from typing import Any

import numpy as np

from src.bridges.channel import Channel
from src.bridges.ckks_mpc_bridge import BridgeCipher, BridgeContext, _conv_params
from src.bridges.secure_bridge import Role, SecureBridge
from src.engines.mpc_engine_plain import PlainShare

OP_COMPLEX_C2M = 0
OP_COMPLEX_M2C = 1
OP_REAL_C2M = 2
OP_REAL_M2C = 3
OP_DONE = -1


def _split_public_value_exact(
    x: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    _, _, scale = _conv_params()
    x_arr = np.asarray(x, dtype=np.float64)
    x_int = np.round(x_arr * scale).astype(np.int64)

    mask_bound = 1 << 20
    client_int = rng.integers(
        -mask_bound,
        mask_bound,
        size=x_int.shape,
        dtype=np.int64,
    )
    server_int = x_int - client_int
    return (
        client_int.astype(np.float64) / scale,
        server_int.astype(np.float64) / scale,
    )


class TwoPartyServerBridge:
    def __init__(self, ctx: BridgeContext, channel: Channel, seed: int | None = None) -> None:
        self._sb = SecureBridge(Role.SERVER, channel, ctx, seed=seed)
        self._ch = channel
        self._rng = np.random.default_rng(seed + 1 if seed is not None else None)

    def _dispatch(self, op: int) -> None:

        self._ch.send("bridge_op", np.array([op], dtype=np.int32))

    def complex_ckks_to_mpc(
        self,
        ct: BridgeCipher,
        *,
        dtype: np.dtype = np.float64,
        comm_stats: Any = None,
    ) -> tuple[PlainShare, PlainShare]:
        _, _, scale = _conv_params()
        self._dispatch(OP_COMPLEX_C2M)

        server_re, server_im = self._sb.complex_ckks_to_mpc(ct, dtype=dtype, comm_stats=comm_stats)

        client_re = self._ch.recv("c2m_client_re")
        client_im = self._ch.recv("c2m_client_im")

        return (
            PlainShare(client_re, server_re, scale=float(scale)),
            PlainShare(client_im, server_im, scale=float(scale)),
        )

    def complex_mpc_to_ckks(
        self,
        re: Any,
        im: Any,
        *,
        encrypt_level: int | None = None,
        comm_stats: Any = None,
    ) -> BridgeCipher:
        self._dispatch(OP_COMPLEX_M2C)

        client_re, server_re = _split_public_value_exact(re, self._rng)
        client_im, server_im = _split_public_value_exact(im, self._rng)

        self._ch.send("m2c_result_re", client_re)
        self._ch.send("m2c_result_im", client_im)

        ct = self._sb.complex_mpc_to_ckks(
            server_re,
            server_im,
            encrypt_level=encrypt_level,
            comm_stats=comm_stats,
        )
        return ct

    def ckks_to_mpc(
        self,
        ct: BridgeCipher,
        *,
        dtype: np.dtype = np.float64,
        comm_stats: Any = None,
    ) -> PlainShare:
        _, _, scale = _conv_params()
        self._dispatch(OP_REAL_C2M)
        server_share = self._sb.ckks_to_mpc(ct, dtype=dtype, comm_stats=comm_stats)
        client_share = self._ch.recv("c2m_client_real")
        return PlainShare(client_share, server_share, scale=float(scale))

    def mpc_to_ckks(
        self,
        x: Any,
        *,
        encrypt_level: int | None = None,
        comm_stats: Any = None,
    ) -> BridgeCipher:
        self._dispatch(OP_REAL_M2C)
        client_share, server_share = _split_public_value_exact(x, self._rng)
        self._ch.send("m2c_result_real", client_share)
        ct = self._sb.mpc_to_ckks(
            server_share,
            encrypt_level=encrypt_level,
            comm_stats=comm_stats,
        )
        return ct

    def finish(self) -> None:

        self._dispatch(OP_DONE)


def client_bridge_loop(channel: Channel) -> dict:

    bridge = SecureBridge(Role.CLIENT, channel)
    stats = {"complex_c2m": 0, "complex_m2c": 0, "real_c2m": 0, "real_m2c": 0}

    while True:
        op_arr = channel.recv("bridge_op")
        op = int(op_arr[0])

        if op == OP_DONE:
            break

        elif op == OP_COMPLEX_C2M:
            client_re, client_im = bridge.complex_ckks_to_mpc(None)

            channel.send("c2m_client_re", client_re)
            channel.send("c2m_client_im", client_im)
            stats["complex_c2m"] += 1

        elif op == OP_COMPLEX_M2C:
            client_re = channel.recv("m2c_result_re")
            client_im = channel.recv("m2c_result_im")

            bridge.complex_mpc_to_ckks(client_re, client_im)
            stats["complex_m2c"] += 1

        elif op == OP_REAL_C2M:
            client_share = bridge.ckks_to_mpc(None)
            channel.send("c2m_client_real", client_share)
            stats["real_c2m"] += 1

        elif op == OP_REAL_M2C:
            client_share = channel.recv("m2c_result_real")
            bridge.mpc_to_ckks(client_share)
            stats["real_m2c"] += 1

        else:
            raise ValueError(f"Unknown bridge op code: {op}")

    return stats
