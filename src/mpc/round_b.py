from __future__ import annotations

from typing import List

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.engines.mpc_engine_factory import get_mpc_engine
from src.utils import pair_cc, shares_to_ckks_mat, to_mpc_mat_shares


def roundB(
    ctx: CKKSContext,
    blocks: List[Cipher],
    *,
    m: int,
    d_out: int,
    eps: float = 1e-05,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    return_meta: bool = False,
    use_cc: bool = True,
    mpc_engine=None,
    bridge=None,
) -> List[Cipher] | tuple[List[Cipher], dict]:
    engine = mpc_engine if mpc_engine is not None else get_mpc_engine()
    shares_in = to_mpc_mat_shares(ctx, blocks, m=m, d_out=d_out, use_cc=use_cc, bridge=bridge)
    shares_out = engine.layer_norm_shares(shares_in, eps=eps, gamma=gamma, beta=beta, ln_tag="ln1")
    out_blocks = shares_to_ckks_mat(ctx, shares_out, m=m, use_cc=use_cc, bridge=bridge)
    ct_in_complex = len(pair_cc(blocks)) if use_cc else len(blocks)
    ct_out_complex = len(pair_cc(out_blocks)) if use_cc else len(out_blocks)
    if return_meta:
        return (out_blocks, {"ct_in": ct_in_complex, "ct_out": ct_out_complex})
    return out_blocks
