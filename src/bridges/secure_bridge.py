from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from src.bridges.channel import Channel
from src.bridges.ckks_mpc_bridge import (
    BridgeCipher,
    BridgeContext,
    _center_lift,
    _cipher_decrypt,
    _conv_params,
    _ctx_encrypt,
    _ctx_nslots,
    _mod,
)


class Role(Enum):
    SERVER = "server"
    CLIENT = "client"


class SecureBridge:
    def __init__(
        self,
        role: Role,
        channel: Channel,
        ctx: BridgeContext | None = None,
        seed: int | None = None,
    ) -> None:
        self._role = role
        self._ch = channel
        self._ctx = ctx
        self._rng = np.random.default_rng(seed)
        if role == Role.SERVER and ctx is None:
            raise ValueError("Server must be initialized with a BridgeContext.")

    @property
    def role(self) -> Role:
        return self._role

    def ckks_to_mpc(
        self,
        ct: BridgeCipher | None = None,
        *,
        dtype: np.dtype = np.float64,
        comm_stats: Any = None,
    ) -> np.ndarray:

        q_conv, ring_mod, scale = _conv_params()

        if self._role == Role.SERVER:
            if ct is None:
                raise ValueError("Server must provide the ciphertext.")
            nslots = _ctx_nslots(self._ctx)
            dec = _cipher_decrypt(self._ctx, ct)
            x_int = np.round(dec.real * scale).astype(np.int64)

            rng = self._rng
            r = rng.integers(0, q_conv, size=nslots, dtype=np.int64)

            masked = _mod(x_int + r, q_conv).astype(np.float64) / scale
            self._ch.send("c2m_masked", masked)

            t1_q = _mod(-r, q_conv)
            t1_ring = _mod(_center_lift(t1_q, q_conv), ring_mod)
            share = _center_lift(t1_ring, ring_mod).astype(dtype) / scale

            if comm_stats is not None:
                comm_stats.add_bridge_c2m(1)
            return share

        else:
            masked = self._ch.recv("c2m_masked")

            t0_q = np.round(masked * scale).astype(np.int64)
            t0_ring = _mod(_center_lift(t0_q, q_conv), ring_mod)
            share = _center_lift(t0_ring, ring_mod).astype(dtype) / scale
            return share

    def mpc_to_ckks(
        self,
        my_share: np.ndarray,
        *,
        encrypt_level: int | None = None,
        comm_stats: Any = None,
    ) -> BridgeCipher | None:

        q_conv, ring_mod, scale = _conv_params()

        x_int = np.round(np.asarray(my_share, dtype=np.float64) * scale).astype(np.int64)
        x_ring = _mod(x_int, ring_mod)
        t_q = _mod(_center_lift(x_ring, ring_mod), q_conv)
        v = _center_lift(t_q, q_conv).astype(np.float64) / scale

        if self._role == Role.CLIENT:
            self._ch.send("m2c_share", v)
            return None
        else:
            v_client = self._ch.recv("m2c_share")
            if comm_stats is not None:
                comm_stats.add_bridge_m2c(1)
            ct0 = _ctx_encrypt(
                self._ctx,
                v.astype(np.complex128),
                encrypt_level=encrypt_level,
            )
            ct1 = _ctx_encrypt(
                self._ctx,
                v_client.astype(np.complex128),
                encrypt_level=encrypt_level,
            )
            return ct0.add(ct1)

    def complex_ckks_to_mpc(
        self,
        ct: BridgeCipher | None = None,
        *,
        dtype: np.dtype = np.float64,
        comm_stats: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        q_conv, ring_mod, scale = _conv_params()

        if self._role == Role.SERVER:
            if ct is None:
                raise ValueError("Server must provide the ciphertext.")
            nslots = _ctx_nslots(self._ctx)
            dec = _cipher_decrypt(self._ctx, ct)
            x_int = np.round(dec.real * scale).astype(np.int64)
            y_int = np.round(dec.imag * scale).astype(np.int64)

            rng = self._rng
            r_re = rng.integers(0, q_conv, size=nslots, dtype=np.int64)
            r_im = rng.integers(0, q_conv, size=nslots, dtype=np.int64)

            masked_re = _mod(x_int + r_re, q_conv).astype(np.float64) / scale
            masked_im = _mod(y_int + r_im, q_conv).astype(np.float64) / scale
            self._ch.send("c2m_masked_re", masked_re)
            self._ch.send("c2m_masked_im", masked_im)

            t1_re_q = _mod(-r_re, q_conv)
            t1_im_q = _mod(-r_im, q_conv)
            t1_re_ring = _mod(_center_lift(t1_re_q, q_conv), ring_mod)
            t1_im_ring = _mod(_center_lift(t1_im_q, q_conv), ring_mod)
            share_re = _center_lift(t1_re_ring, ring_mod).astype(dtype) / scale
            share_im = _center_lift(t1_im_ring, ring_mod).astype(dtype) / scale

            if comm_stats is not None:
                comm_stats.add_bridge_c2m(1)
            return share_re, share_im

        else:
            masked_re = self._ch.recv("c2m_masked_re")
            masked_im = self._ch.recv("c2m_masked_im")

            t0_re_q = np.round(masked_re * scale).astype(np.int64)
            t0_im_q = np.round(masked_im * scale).astype(np.int64)
            t0_re_ring = _mod(_center_lift(t0_re_q, q_conv), ring_mod)
            t0_im_ring = _mod(_center_lift(t0_im_q, q_conv), ring_mod)
            share_re = _center_lift(t0_re_ring, ring_mod).astype(dtype) / scale
            share_im = _center_lift(t0_im_ring, ring_mod).astype(dtype) / scale
            return share_re, share_im

    def complex_mpc_to_ckks(
        self,
        my_share_re: np.ndarray,
        my_share_im: np.ndarray,
        *,
        encrypt_level: int | None = None,
        comm_stats: Any = None,
    ) -> BridgeCipher | None:

        q_conv, ring_mod, scale = _conv_params()

        def _reduce(arr: np.ndarray) -> np.ndarray:
            x_int = np.round(np.asarray(arr, dtype=np.float64) * scale).astype(np.int64)
            x_ring = _mod(x_int, ring_mod)
            t_q = _mod(_center_lift(x_ring, ring_mod), q_conv)
            return _center_lift(t_q, q_conv).astype(np.float64) / scale

        v_re = _reduce(my_share_re)
        v_im = _reduce(my_share_im)

        if self._role == Role.CLIENT:
            self._ch.send("m2c_share_re", v_re)
            self._ch.send("m2c_share_im", v_im)
            return None
        else:
            vc_re = self._ch.recv("m2c_share_re")
            vc_im = self._ch.recv("m2c_share_im")
            if comm_stats is not None:
                comm_stats.add_bridge_m2c(1)
            v0 = v_re.astype(np.complex128) + 1j * v_im.astype(np.complex128)
            v1 = vc_re.astype(np.complex128) + 1j * vc_im.astype(np.complex128)
            ct0 = _ctx_encrypt(self._ctx, v0, encrypt_level=encrypt_level)
            ct1 = _ctx_encrypt(self._ctx, v1, encrypt_level=encrypt_level)
            return ct0.add(ct1)
