from __future__ import annotations

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.engines.mpc_engine_factory import get_mpc_engine
from src.fhe.simulator.score import mapf
from src.fhe.simulator.value import pack_a
from src.share_types import ShareMatrix
from src.utils import pack_a_groups, score_unpack_vecs, to_mpc_vecs


def _to_mpc_shares(ctx: CKKSContext, cts, *, use_cc: bool, bridge=None):

    from src.bridges.ckks_mpc_bridge import complex_ckks_to_mpc, real_ckks_to_mpc

    shares = []
    for ct in cts:
        if use_cc:
            if bridge is not None:
                sh_re, sh_im = bridge.complex_ckks_to_mpc(ct, dtype=np.float64)
            else:
                sh_re, sh_im = complex_ckks_to_mpc(ctx, ct, dtype=np.float64)
        else:
            if bridge is not None:
                sh_re = bridge.ckks_to_mpc(ct, dtype=np.float64)
                sh_im = bridge.ckks_to_mpc(ct.mul_scalar(-1j), dtype=np.float64)
            else:
                sh_re = real_ckks_to_mpc(ctx, ct, dtype=np.float64)
                sh_im = real_ckks_to_mpc(ctx, ct.mul_scalar(-1j), dtype=np.float64)
        s0 = sh_re.share0.astype(np.complex128) + 1j * sh_im.share0.astype(np.complex128)
        s1 = sh_re.share1.astype(np.complex128) + 1j * sh_im.share1.astype(np.complex128)
        shares.append((s0, s1))
    return shares


def roundA(
    ctx: CKKSContext,
    P: list[Cipher],
    *,
    H: int,
    m: int,
    d_k: int | None = None,
    b_fold: int = 16,
    use_cc: bool = True,
    mpc_engine=None,
    attn_mask: np.ndarray | None = None,
    bridge=None,
) -> tuple[list[Cipher], list]:

    mp_half, _ = mapf(m, b_fold)
    engine = mpc_engine if mpc_engine is not None else get_mpc_engine()

    raw_shares = _to_mpc_shares(ctx, P, use_cc=use_cc, bridge=bridge)

    vecs_s0 = [s0 for s0, _ in raw_shares]
    vecs_s1 = [s1 for _, s1 in raw_shares]
    heads_s0 = score_unpack_vecs(vecs_s0, H=H, m=m, mp_half=mp_half)
    heads_s1 = score_unpack_vecs(vecs_s1, H=H, m=m, mp_half=mp_half)

    A_all = []
    A_share_list = []
    for h in range(H):
        head_shares = ShareMatrix(heads_s0[h].real, heads_s1[h].real)

        if d_k is not None:
            scale = 1.0 / np.sqrt(float(d_k))
            head_shares = ShareMatrix(head_shares.share0 * scale, head_shares.share1 * scale)
        if attn_mask is not None:
            mask_arr = np.asarray(attn_mask, dtype=np.float64)
            if mask_arr.ndim == 1:
                mask_1d = mask_arr.ravel()[:m]
                additive = np.where(mask_1d == 1, 0.0, -1e9)

                head_shares = ShareMatrix(head_shares.share0 + additive[np.newaxis, :], head_shares.share1)
            elif mask_arr.ndim == 2:
                head_shares = ShareMatrix(head_shares.share0 + mask_arr[:m, :m], head_shares.share1)

        result_shares = engine.softmax_rows_shares(head_shares, head_index=h)
        A_share_list.append(result_shares)
        A_all.append(result_shares.reconstruct())

    A_cts = pack_a_groups(ctx, A_all, H=H, m=m, d_h=d_k, pack_a_fn=pack_a)
    return A_cts, A_all
