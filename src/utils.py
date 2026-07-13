from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.bridges.ckks_mpc_bridge import (
    complex_ckks_to_mpc,
    complex_mpc_to_ckks,
    real_ckks_to_mpc,
    real_mpc_to_ckks,
)

CKKSContext = Any
Cipher = Any


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b)) + 1e-12
    return num / den


def stat_snap(ctx: CKKSContext) -> Dict[str, int]:
    s = ctx.stats
    g = lambda k: int(getattr(s, k, 0))
    return dict(
        ks_rots=g("ks_rots"),
        ks_relin=g("ks_relin"),
        ks_conj=g("ks_conj"),
        ks_total=g("ks_total"),
        ks_muls_ctct=g("ks_muls_ctct"),
        ops_rotate=g("ops_rotate"),
        ops_mul_ct=g("ops_mul_ct"),
        ops_mul_pt=g("ops_mul_pt"),
        ops_add=g("ops_add"),
        ops_conj=g("ops_conj"),
        cts_created=g("cts_created"),
    )


def stat_diff(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: int(b.get(k, 0) - a.get(k, 0)) for k in a}


def ct_real(ctx: CKKSContext, ct: Cipher) -> Cipher:
    return ct.add(ctx.conjugate(ct)).mul_scalar(0.5)


def print_stage(
    tag: str,
    *,
    ct_in: Optional[int],
    ct_out: Optional[int],
    ks_rots: Optional[int],
    ks_muls: Optional[int],
    ks_conj: Optional[int],
    rel_err: Optional[float],
) -> None:
    tag_col = f"[{tag}]".ljust(10)
    if ct_in is None and ct_out is None:
        ct_col = "".ljust(20)
    else:
        cin = "NA" if ct_in is None else str(ct_in)
        cout = "NA" if ct_out is None else str(ct_out)
        ct_col = f"ct_in={cin:<3} ct_out={cout:<3}"
    rot_s = "NA" if ks_rots is None else str(ks_rots)
    mul_s = "NA" if ks_muls is None else str(ks_muls)
    conj_s = "NA" if ks_conj is None else str(ks_conj)
    ks_col = f"ks=rot:{rot_s:>4} mul:{mul_s:>4} conj:{conj_s:>4}"
    err_col = f"rel_err={rel_err:.3e}" if rel_err is not None else "rel_err=NA"
    print(f"{tag_col} {ct_col} {ks_col}  {err_col}")


def pair_cc(blocks: List[Cipher]) -> List[Cipher]:
    out: List[Cipher] = []
    i = 0
    while i < len(blocks):
        ct = blocks[i]
        if i + 1 < len(blocks):
            ct = ct.add(blocks[i + 1].mul_scalar(1j))
        out.append(ct)
        i += 2
    return out


def to_mpc_mat(
    ctx: CKKSContext,
    blocks: List[Cipher],
    *,
    m: int,
    d_out: int,
    use_cc: bool,
    bridge: Any = None,
) -> Tuple[np.ndarray, int]:
    n = ctx.nslots
    c = n // m
    out = np.zeros((m, d_out), dtype=np.float64)
    in_blocks = pair_cc(blocks) if use_cc else blocks
    b = 0
    for ct in in_blocks:
        if use_cc:
            if bridge is not None:
                sh_re, sh_im = bridge.complex_ckks_to_mpc(ct, dtype=np.float64)
            else:
                sh_re, sh_im = complex_ckks_to_mpc(ctx, ct, dtype=np.float64)
            v_re = sh_re.reconstruct()
            v_im = sh_im.reconstruct()
            used_re = min(c, max(0, d_out - b * c))
            if used_re <= 0:
                break
            for local_c in range(used_re):
                col = b * c + local_c
                out[:, col] = v_re[local_c * m : (local_c + 1) * m]
            b += 1
            used_im = min(c, max(0, d_out - b * c))
            if used_im <= 0:
                break
            for local_c in range(used_im):
                col = b * c + local_c
                out[:, col] = v_im[local_c * m : (local_c + 1) * m]
            b += 1
        else:
            if bridge is not None:
                sh_re = bridge.ckks_to_mpc(ct, dtype=np.float64)
            else:
                sh_re = real_ckks_to_mpc(ctx, ct, dtype=np.float64)
            v_re = sh_re.reconstruct()
            used = min(c, max(0, d_out - b * c))
            if used <= 0:
                break
            for local_c in range(used):
                col = b * c + local_c
                out[:, col] = v_re[local_c * m : (local_c + 1) * m]
            b += 1
    return out, len(in_blocks)


def to_ckks_mat(ctx: CKKSContext, x: np.ndarray, *, m: int, use_cc: bool, bridge: Any = None) -> List[Cipher]:
    n = ctx.nslots
    c = n // m
    d_out = x.shape[1]
    blocks = (d_out + c - 1) // c
    out: List[Cipher] = []
    for b in range(blocks):
        v = np.zeros(n, dtype=np.complex128)
        used = min(c, max(0, d_out - b * c))
        for local_c in range(used):
            col = b * c + local_c
            v[local_c * m : (local_c + 1) * m] = x[:, col].astype(np.complex128)
        if use_cc:
            v_re = v.real.astype(np.float64)
            v_im = np.zeros_like(v_re, dtype=np.float64)
            if bridge is not None:
                out.append(bridge.complex_mpc_to_ckks(v_re, v_im))
            else:
                out.append(complex_mpc_to_ckks(ctx, v_re, v_im))
        else:
            v_re = v.real.astype(np.float64)
            if bridge is not None:
                out.append(bridge.mpc_to_ckks(v_re))
            else:
                out.append(real_mpc_to_ckks(ctx, v_re))
    return out


def to_mpc_mat_shares(
    ctx: CKKSContext,
    blocks: List[Cipher],
    *,
    m: int,
    d_out: int,
    use_cc: bool,
    bridge: Any = None,
) -> "ShareMatrix":

    from src.share_types import ShareMatrix

    n = ctx.nslots
    c = n // m
    s0 = np.zeros((m, d_out), dtype=np.float64)
    s1 = np.zeros((m, d_out), dtype=np.float64)
    in_blocks = pair_cc(blocks) if use_cc else blocks
    b = 0
    for ct in in_blocks:
        if use_cc:
            if bridge is not None:
                sh_re, sh_im = bridge.complex_ckks_to_mpc(ct, dtype=np.float64)
            else:
                sh_re, sh_im = complex_ckks_to_mpc(ctx, ct, dtype=np.float64)

            used_re = min(c, max(0, d_out - b * c))
            if used_re <= 0:
                break
            for local_c in range(used_re):
                col = b * c + local_c
                s0[:, col] = sh_re.share0[local_c * m : (local_c + 1) * m]
                s1[:, col] = sh_re.share1[local_c * m : (local_c + 1) * m]
            b += 1

            used_im = min(c, max(0, d_out - b * c))
            if used_im <= 0:
                break
            for local_c in range(used_im):
                col = b * c + local_c
                s0[:, col] = sh_im.share0[local_c * m : (local_c + 1) * m]
                s1[:, col] = sh_im.share1[local_c * m : (local_c + 1) * m]
            b += 1
        else:
            if bridge is not None:
                sh_re = bridge.ckks_to_mpc(ct, dtype=np.float64)
            else:
                sh_re = real_ckks_to_mpc(ctx, ct, dtype=np.float64)
            used = min(c, max(0, d_out - b * c))
            if used <= 0:
                break
            for local_c in range(used):
                col = b * c + local_c
                s0[:, col] = sh_re.share0[local_c * m : (local_c + 1) * m]
                s1[:, col] = sh_re.share1[local_c * m : (local_c + 1) * m]
            b += 1
    return ShareMatrix(s0, s1)


def shares_to_ckks_mat(
    ctx: CKKSContext,
    shares: "ShareMatrix",
    *,
    m: int,
    use_cc: bool,
    bridge: Any = None,
) -> List[Cipher]:

    return to_ckks_mat(ctx, shares.reconstruct(), m=m, use_cc=use_cc, bridge=bridge)


def ln_apply(
    x: np.ndarray,
    *,
    eps: float = 1e-5,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def smx_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    y = np.exp(x)
    return y / (y.sum(axis=1, keepdims=True) + 1e-12)


def score_unpack_vecs(p_dec: List[np.ndarray], *, H: int, m: int, mp_half) -> np.ndarray:
    n = p_dec[0].shape[0]
    half = m // 2
    blen = H * m
    rows = np.arange(m)
    S = np.zeros((H, m, m), dtype=np.float64)
    for t in range(half):
        start = t * blen
        vec = np.empty(blen, dtype=np.complex128)
        for i in range(blen):
            lin = start + i
            k = lin // n
            off = lin % n
            vec[i] = p_dec[k][off]
        _, s = mp_half[t]
        for h in range(H):
            seg = vec[h * m : (h + 1) * m]
            seg = np.roll(seg, +s)
            S[h, rows, (rows + t) % m] += seg.real
            S[h, rows, (rows + half + t) % m] += seg.imag
    return S


def pack_a_groups(
    ctx: CKKSContext,
    A_all: List[np.ndarray],
    *,
    H: int,
    m: int,
    d_h: int | None = None,
    pack_a_fn: Callable[..., Cipher],
) -> List[Cipher]:
    C = ctx.nslots // m
    Dh = d_h if d_h is not None else (m // 2)
    heads_per_ct = C // Dh
    groups = H // heads_per_ct
    A_cts: List[Cipher] = []
    for g in range(groups):
        ids = [g * heads_per_ct + i for i in range(heads_per_ct)]
        A_cts.append(pack_a_fn(ctx, [A_all[h] for h in ids], m=m, heads_per_ct=heads_per_ct))
    return A_cts


def to_mpc_vecs(
    ctx: CKKSContext, cts: List[Cipher], *, use_cc: bool, dtype=np.float64, bridge: Any = None
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for ct in cts:
        if use_cc:
            if bridge is not None:
                sh_re, sh_im = bridge.complex_ckks_to_mpc(ct, dtype=dtype)
            else:
                sh_re, sh_im = complex_ckks_to_mpc(ctx, ct, dtype=dtype)
        else:
            if bridge is not None:
                sh_re = bridge.ckks_to_mpc(ct, dtype=dtype)
                sh_im = bridge.ckks_to_mpc(ct.mul_scalar(-1j), dtype=dtype)
            else:
                sh_re = real_ckks_to_mpc(ctx, ct, dtype=dtype)
                sh_im = real_ckks_to_mpc(ctx, ct.mul_scalar(-1j), dtype=dtype)
        vec = sh_re.reconstruct().astype(np.complex128) + 1j * sh_im.reconstruct().astype(np.complex128)
        out.append(vec)
    return out
