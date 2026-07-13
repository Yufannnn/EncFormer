from __future__ import annotations

import os
from typing import List

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.engines.mpc_engine_factory import get_mpc_engine
from src.utils import pair_cc, shares_to_ckks_mat, to_mpc_mat_shares


def _expanded_gelu_enabled() -> bool:
    return os.getenv("ENCFORMER_EXPANDED_GELU", "0") in ("1", "true", "yes")


def _expanded_gelu_pack_f0f1() -> bool:

    return os.getenv("ENCFORMER_EXPANDED_GELU_PACK", "1") in ("1", "true", "yes")


def _bridge_paired_f0f1(
    ctx: CKKSContext,
    f0_mat: np.ndarray,
    f1_mat: np.ndarray,
    *,
    m: int,
    d_out: int,
    bridge,
):

    from src.bridges.ckks_mpc_bridge import complex_ckks_to_mpc, complex_mpc_to_ckks
    from src.share_types import ShareMatrix

    n = ctx.nslots
    c_per_block = n // m
    n_blocks = (d_out + c_per_block - 1) // c_per_block

    s0_f0 = np.zeros((m, d_out), dtype=np.float64)
    s1_f0 = np.zeros((m, d_out), dtype=np.float64)
    s0_f1 = np.zeros((m, d_out), dtype=np.float64)
    s1_f1 = np.zeros((m, d_out), dtype=np.float64)

    for bi in range(n_blocks):
        f0_real = np.zeros(n, dtype=np.float64)
        f1_imag = np.zeros(n, dtype=np.float64)
        used = min(c_per_block, max(0, d_out - bi * c_per_block))
        for local_c in range(used):
            col = bi * c_per_block + local_c
            f0_real[local_c * m : (local_c + 1) * m] = f0_mat[:, col]
            f1_imag[local_c * m : (local_c + 1) * m] = f1_mat[:, col]

        ct_paired = complex_mpc_to_ckks(ctx, f0_real, f1_imag)

        if bridge is not None:
            sh_re, sh_im = bridge.complex_ckks_to_mpc(ct_paired, dtype=np.float64)
        else:
            sh_re, sh_im = complex_ckks_to_mpc(ctx, ct_paired, dtype=np.float64)
        for local_c in range(used):
            col = bi * c_per_block + local_c
            s0_f0[:, col] = sh_re.share0[local_c * m : (local_c + 1) * m]
            s1_f0[:, col] = sh_re.share1[local_c * m : (local_c + 1) * m]
            s0_f1[:, col] = sh_im.share0[local_c * m : (local_c + 1) * m]
            s1_f1[:, col] = sh_im.share1[local_c * m : (local_c + 1) * m]
    return ShareMatrix(s0_f0, s1_f0), ShareMatrix(s0_f1, s1_f1)


def _eval_f0_f1_matrices(ctx: CKKSContext, blocks: List[Cipher], *, m: int, d_out: int, use_cc: bool):

    from src.engines.mpc_gelu_secure import (
        build_plain_algorithm4_constants,
        load_secure_gelu_config,
    )
    from src.utils import to_mpc_mat

    cfg = load_secure_gelu_config()
    consts = build_plain_algorithm4_constants(cfg)
    x_mat, _ = to_mpc_mat(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=None)
    s = float(1 << int(cfg.scale_bits))
    a = float(consts.a) / s
    b = float(consts.b) / s
    c = float(consts.c) / s
    p_const = float(consts.p) / s
    m_const = float(consts.m) / s
    e = float(consts.e) / s
    x = np.asarray(x_mat, dtype=np.float64)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    f0 = a * x4 - b * x3 + c * x2 + m_const * x + e
    f1 = a * x4 + b * x3 + c * x2 + p_const * x + e
    return f0, f1


def _eval_f0_f1_blocks(ctx: CKKSContext, blocks: List[Cipher], *, m: int, d_out: int, use_cc: bool):

    from src.bridges.ckks_mpc_bridge import complex_mpc_to_ckks, real_mpc_to_ckks
    from src.engines.mpc_gelu_secure import (
        build_plain_algorithm4_constants,
        load_secure_gelu_config,
    )
    from src.utils import to_mpc_mat

    cfg = load_secure_gelu_config()
    consts = build_plain_algorithm4_constants(cfg)

    x_mat, _ = to_mpc_mat(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=None)

    s = float(1 << int(cfg.scale_bits))
    a = float(consts.a) / s
    b = float(consts.b) / s
    c = float(consts.c) / s
    p_const = float(consts.p) / s
    m_const = float(consts.m) / s
    e = float(consts.e) / s

    x = np.asarray(x_mat, dtype=np.float64)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    f0 = a * x4 - b * x3 + c * x2 + m_const * x + e
    f1 = a * x4 + b * x3 + c * x2 + p_const * x + e

    n = ctx.nslots
    c_per_block = n // m
    n_blocks = (d_out + c_per_block - 1) // c_per_block

    def _encode_local(arr):
        out_blocks = []
        for bi in range(n_blocks):
            v = np.zeros(n, dtype=np.complex128)
            used = min(c_per_block, max(0, d_out - bi * c_per_block))
            for local_c in range(used):
                col = bi * c_per_block + local_c
                v[local_c * m : (local_c + 1) * m] = arr[:, col].astype(np.complex128)
            if use_cc:
                v_re = v.real.astype(np.float64)
                v_im = np.zeros_like(v_re, dtype=np.float64)
                out_blocks.append(complex_mpc_to_ckks(ctx, v_re, v_im))
            else:
                out_blocks.append(real_mpc_to_ckks(ctx, v.real.astype(np.float64)))
        return out_blocks

    return _encode_local(f0), _encode_local(f1)


def roundC(
    ctx: CKKSContext,
    blocks: List[Cipher],
    *,
    m: int,
    d_out: int,
    return_meta: bool = False,
    use_cc: bool = True,
    mpc_engine=None,
    bridge=None,
) -> List[Cipher] | tuple[List[Cipher], dict]:
    engine = mpc_engine if mpc_engine is not None else get_mpc_engine()

    if _expanded_gelu_enabled() and hasattr(engine, "gelu_preeval_shares"):
        x_shares = to_mpc_mat_shares(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=bridge)
        if _expanded_gelu_pack_f0f1() and use_cc:
            f0_mat, f1_mat = _eval_f0_f1_matrices(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc)
            f0_shares, f1_shares = _bridge_paired_f0f1(
                ctx,
                f0_mat,
                f1_mat,
                m=m,
                d_out=d_out,
                bridge=bridge,
            )
            ct_in_complex = (len(pair_cc(blocks)) if use_cc else len(blocks)) + (
                (d_out + ctx.nslots // m - 1) // (ctx.nslots // m)
            )
        else:
            f0_blocks, f1_blocks = _eval_f0_f1_blocks(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc)
            f0_shares = to_mpc_mat_shares(ctx, f0_blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=bridge)
            f1_shares = to_mpc_mat_shares(ctx, f1_blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=bridge)
            ct_in_complex = 3 * (len(pair_cc(blocks)) if use_cc else len(blocks))
        shares_out = engine.gelu_preeval_shares(x_shares, f0_shares, f1_shares)
    else:
        shares_in = to_mpc_mat_shares(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=bridge)
        shares_out = engine.gelu_shares(shares_in)
        ct_in_complex = len(pair_cc(blocks)) if use_cc else len(blocks)

    out_blocks = shares_to_ckks_mat(ctx, shares_out, m=m, use_cc=use_cc, bridge=bridge)
    ct_out_complex = len(pair_cc(out_blocks)) if use_cc else len(out_blocks)
    if return_meta:
        return (out_blocks, {"ct_in": ct_in_complex, "ct_out": ct_out_complex})
    return out_blocks
