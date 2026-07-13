from __future__ import annotations

import inspect
import os

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.engines.mpc_engine_factory import get_mpc_engine, resolve_pipeline
from src.engines.mpc_gelu_secure import secure_gelu_piecewise_reference
from src.inference_runtime import (
    inject_layer_running_denoms,
    prepare_layer_running_denoms,
)
from src.models.model_config import ModelConfig
from src.models.model_config import get_config as _get_config

_MODEL_CFG = _get_config(os.environ.get("ENCFORMER_MODEL", "bert-base"))


QKV_NSLOTS = _MODEL_CFG.nslots
QKV_M = _MODEL_CFG.m
QKV_H = _MODEL_CFG.H
QKV_D = _MODEL_CFG.d_model
QKV_D1 = _MODEL_CFG.d1
QKV_N1 = _MODEL_CFG.n1


def reconfigure(model_name: str) -> None:

    global _MODEL_CFG, QKV_NSLOTS, QKV_M, QKV_H, QKV_D, QKV_D1, QKV_N1
    _MODEL_CFG = _get_config(model_name)
    QKV_NSLOTS = _MODEL_CFG.nslots
    QKV_M = _MODEL_CFG.m
    QKV_H = _MODEL_CFG.H
    QKV_D = _MODEL_CFG.d_model
    QKV_D1 = _MODEL_CFG.d1
    QKV_N1 = _MODEL_CFG.n1


def _resolve_cfg(cfg) -> ModelConfig:

    return cfg if cfg is not None else _MODEL_CFG


def _effective_ckks_backend(backend: str) -> str:

    pipeline_ckks, _ = resolve_pipeline()
    if pipeline_ckks:
        return pipeline_ckks
    return backend


from src.bridges.ckks_mpc_bridge import complex_ckks_to_mpc, complex_mpc_to_ckks
from src.fhe.simulator.qkv import pa6, perm_fdp, perm_hd, pre_tbl_c, proj_mix, split_qk
from src.fhe.simulator.score import B_FOLD, aln_h, emit_f, mapf, mk_bank, pack_f, red_h, score_ref, unpack_f
from src.fhe.simulator.value import dec_grp, pack_a, smx, sv
from src.mpc.round_a import roundA
from src.mpc.round_b import roundB
from src.mpc.round_c import roundC
from src.mpc.round_d import roundD
from src.utils import (
    ct_real,
    ln_apply,
    pack_a_groups,
    pair_cc,
    print_stage,
    rel_err,
    score_unpack_vecs,
    smx_rows,
    stat_diff,
    stat_snap,
    to_ckks_mat,
    to_mpc_mat,
    to_mpc_vecs,
)

SEED = 42


_SEG_ENC_LEVEL: int | None = None


def cs(fn, /, **kwargs):
    sig = inspect.signature(fn)
    params = sig.parameters
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_secure_ref(x: np.ndarray) -> np.ndarray:
    return secure_gelu_piecewise_reference(np.asarray(x, dtype=np.float64))


def gelu(x: np.ndarray) -> np.ndarray:

    return gelu_tanh(x)


def ln(X: np.ndarray, eps: float = 1e-05) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def fgs(ctx: CKKSContext, grid):
    out = []
    for row in grid:
        acc = ctx.zeros()
        for ct in row:
            acc = acc.add(ct)
        out.append(acc)
    return out


def dec_fdp(ctx: CKKSContext, cts: list[Cipher], *, m: int, used: int) -> np.ndarray:
    out = np.zeros((m, len(cts) * used), dtype=np.float64)
    with ctx.decorypt_scope():
        for bid, ct in enumerate(cts):
            v = ct.decorypt(copy=True, readonly=True).real
            for c in range(used):
                out[:, bid * used + c] = v[c * m : (c + 1) * m]
    return out


def bb(ctx: CKKSContext, base_cts: list[Cipher], *, m: int, N1: int) -> list[list[Cipher]]:
    n = ctx.nslots
    babies: list[list[Cipher]] = []
    for base in base_cts:
        row: list[Cipher] = []
        for q in range(N1):
            shift = q * m % n
            row.append(base.rot(shift) if shift else base)
        babies.append(row)
    return babies


def linp(ctx: CKKSContext, Gpairs: list[list[Cipher]], W: np.ndarray, *, m: int, N1: int, G: int) -> list[list[Cipher]]:
    n = ctx.nslots
    C = n // m
    d_in, d_out = W.shape
    assert d_in == G * C
    assert d_out % C == 0
    assert C % N1 == 0
    H_pairs = (G + 1) // 2
    assert len(Gpairs) == H_pairs
    N2 = C // N1
    blocks = d_out // C
    slots = np.arange(n, dtype=np.int32)
    c = slots // m
    Cf = [[ctx.zeros() for _ in range(N2)] for _ in range(blocks)]
    for b in range(blocks):
        for p in range(N2):
            S = ctx.zeros()
            for q in range(N1):
                t_local = (c + q) % C
                out_c = (c - p * N1) % C
                ell = b * C + out_c
                Acc = ctx.zeros()
                for h in range(H_pairs):
                    ge = 2 * h
                    go = ge + 1
                    wr = W[ge * C + t_local, ell].astype(np.float64).astype(np.complex128)
                    if go < G:
                        wi = W[go * C + t_local, ell].astype(np.float64).astype(np.complex128)
                        P = wr - 1j * wi
                    else:
                        P = wr
                    Acc = Acc.add(Gpairs[h][q].mul_pt(P))
                S = S.add(Acc)
            Cf[b][p] = Cf[b][p].add(S)
    return Cf


def fg(ctx: CKKSContext, Cf_lazy: list[list[Cipher]], *, m: int, N1: int) -> list[Cipher]:
    n = ctx.nslots
    C = n // m
    assert C % N1 == 0
    N2 = C // N1
    folded: list[Cipher] = []
    for b in range(len(Cf_lazy)):
        acc = ctx.zeros()
        for p in range(N2):
            ct = Cf_lazy[b][p]
            shift = p * N1 * m % n
            if shift:
                ct = ct.rot(shift)
            acc = acc.add(ct)
        folded.append(acc)
    return folded


def decb(ctx: CKKSContext, folded_blocks: list[Cipher], *, m: int, d_out: int) -> np.ndarray:
    n = ctx.nslots
    C = n // m
    out = np.zeros((m, d_out), dtype=np.float64)
    with ctx.decorypt_scope():
        for b in range(len(folded_blocks)):
            used = min(C, max(0, d_out - b * C))
            if used <= 0:
                break
            v = folded_blocks[b].decorypt(copy=True, readonly=True).real
            for local_c in range(used):
                col = b * C + local_c
                out[:, col] = v[local_c * m : (local_c + 1) * m]
    return out


def pack_scp_blocks(ctx: CKKSContext, X: np.ndarray, *, m: int) -> list[Cipher]:
    n = ctx.nslots
    C = n // m
    d_out = X.shape[1]
    blocks = (d_out + C - 1) // C
    out: list[Cipher] = []
    for b in range(blocks):
        v = np.zeros(n, dtype=np.complex128)
        used = min(C, max(0, d_out - b * C))
        for local_c in range(used):
            col = b * C + local_c
            v[local_c * m : (local_c + 1) * m] = X[:, col].astype(np.complex128)
        out.append(ctx.encorypt(v))
    return out


def repack_id(cts: list[Cipher], *, m: int, iters: int) -> list[Cipher]:
    if not cts or iters <= 0:
        return cts
    out: list[Cipher] = []
    for ct in cts:
        t = ct
        for _ in range(iters):
            t = t.rot(m).rot(-m)
        out.append(t)
    return out


def addb(
    ctx: CKKSContext, blocks: list[Cipher], bias: np.ndarray, *, m: int, stride: int | None = None
) -> list[Cipher]:
    n = ctx.nslots
    C = n // m
    step = C if stride is None else int(stride)
    out: list[Cipher] = []
    for b, ct in enumerate(blocks):
        v = np.zeros(n, dtype=np.complex128)
        for local_c in range(step):
            col = b * step + local_c
            if col >= bias.size:
                break
            v[local_c * m : (local_c + 1) * m] = bias[col]
        out.append(ct.add(ctx.encorypt(v)))
    return out


def addl(a: list[Cipher], b: list[Cipher]) -> list[Cipher]:
    assert len(a) == len(b)
    return [a[i].add(b[i]) for i in range(len(a))]


def pack_outp(ctx: CKKSContext, Y_groups: list[Cipher]) -> list[Cipher]:
    G = len(Y_groups)
    Y_clean = [ct_real(ctx, ct) for ct in Y_groups]
    out: list[Cipher] = []
    for h in range((G + 1) // 2):
        ge = 2 * h
        go = ge + 1
        ct = Y_clean[ge]
        if go < G:
            ct = ct.add(Y_clean[go].mul_scalar(1j))
        out.append(ct)
    return out


def pack_pairs(ctx: CKKSContext, X_blocks: list[Cipher]) -> list[Cipher]:
    X_clean = [ct_real(ctx, ct) for ct in X_blocks]
    out: list[Cipher] = []
    for i in range(0, len(X_clean), 2):
        ct = X_clean[i]
        if i + 1 < len(X_clean):
            ct = ct.add(X_clean[i + 1].mul_scalar(1j))
        out.append(ct)
    return out


def run_lin(
    ctx: CKKSContext, *, X_blocks: list[Cipher], W: np.ndarray, N1: int = 16, force_real: bool = True
) -> tuple[list[Cipher], dict]:
    m = QKV_M
    n = ctx.nslots
    C = n // m
    d_in, d_out = W.shape
    G = d_in // C
    assert d_in == G * C
    assert d_out % C == 0
    assert len(X_blocks) == G
    assert C % N1 == 0
    base_cts = pack_pairs(ctx, X_blocks)
    babies = bb(ctx, base_cts, m=m, N1=N1)
    Cf_lazy = linp(ctx, babies, W, m=m, N1=N1, G=G)
    folded = fg(ctx, Cf_lazy, m=m, N1=N1)
    if force_real:
        Y_blocks = [ct_real(ctx, ct) for ct in folded]
    else:
        Y_blocks = folded
    return (Y_blocks, dict(ct_in=len(base_cts), ct_out=len(Y_blocks)))


def run_out(
    ctx: CKKSContext, *, Y_groups: list[Cipher], W_O: np.ndarray, N1: int = 16, force_real: bool = True
) -> tuple[list[Cipher], dict]:
    m = QKV_M
    n = ctx.nslots
    C = n // m
    d_in, d_out = W_O.shape
    assert d_in == QKV_D and d_out == QKV_D
    assert C % N1 == 0
    base_cts = pack_outp(ctx, Y_groups)
    babies = bb(ctx, base_cts, m=m, N1=N1)
    G = d_in // C
    Cf_lazy = linp(ctx, babies, W_O, m=m, N1=N1, G=G)
    folded = fg(ctx, Cf_lazy, m=m, N1=N1)
    if force_real:
        Z_blocks = [ct_real(ctx, ct) for ct in folded]
    else:
        Z_blocks = folded
    return (Z_blocks, dict(ct_in=len(base_cts), ct_out=len(Z_blocks)))


def run_q(
    ctx: CKKSContext,
    *,
    A: np.ndarray,
    WQ: np.ndarray,
    WK: np.ndarray,
    WV: np.ndarray,
    bQ: np.ndarray | None = None,
    bK: np.ndarray | None = None,
    bV: np.ndarray | None = None,
):
    n = ctx.nslots
    m = QKV_M
    C = n // m

    from src.fhe.simulator.score import CASE_REL_C_USED

    c_used_qk = min(C, CASE_REL_C_USED)
    blocks_qk = (QKV_D + c_used_qk - 1) // c_used_qk

    c_used_v = C
    blocks_v = (QKV_D + c_used_v - 1) // c_used_v
    Dh = QKV_D // QKV_H
    perm_fdp_cols = perm_fdp(QKV_H, Dh)
    perm_hm_cols = perm_hd(QKV_H, Dh)
    WQ_fdp = WQ[:, perm_fdp_cols]
    WK_fdp = WK[:, perm_fdp_cols]
    WV_hm = WV[:, perm_hm_cols]
    bQ_fdp = bQ[perm_fdp_cols] if bQ is not None else None
    bK_fdp = bK[perm_fdp_cols] if bK is not None else None
    bV_hm = bV[perm_hm_cols] if bV is not None else None

    target_d1 = 6 * C
    d_model_actual = A.shape[1]
    if d_model_actual < target_d1:
        A = np.pad(A, ((0, 0), (0, target_d1 - d_model_actual)))
        WQ_fdp = np.pad(WQ_fdp, ((0, target_d1 - d_model_actual), (0, 0)))
        WK_fdp = np.pad(WK_fdp, ((0, target_d1 - d_model_actual), (0, 0)))
        WV_hm = np.pad(WV_hm, ((0, target_d1 - d_model_actual), (0, 0)))
    A_cts = [ctx.encorypt(v) for v in pa6(A, m=m, n_slots=n)]
    A_pairs = [
        A_cts[0].add(A_cts[1].mul_scalar(1j)),
        A_cts[2].add(A_cts[3].mul_scalar(1j)),
        A_cts[4].add(A_cts[5].mul_scalar(1j)),
    ]
    ct_pairs_babies: list[list[Cipher]] = []
    for Ap in A_pairs:
        row: list[Cipher] = []
        for q in range(QKV_N1):
            shift = q * m % n
            row.append(Ap.rot(shift) if shift else Ap)
        ct_pairs_babies.append(row)
    QK_tab, _ = cs(
        pre_tbl_c,
        WQ=WQ_fdp,
        WK=WK_fdp,
        WV=WV_hm,
        C=C,
        c_used=c_used_qk,
        d2=QKV_D,
        blocks=blocks_qk,
        N1=QKV_N1,
        m=m,
        n_slots=n,
        nslots=n,
    )
    _QK2, V_tab = cs(
        pre_tbl_c,
        WQ=WQ_fdp,
        WK=WK_fdp,
        WV=WV_hm,
        C=C,
        c_used=c_used_v,
        d2=QKV_D,
        blocks=blocks_v,
        N1=QKV_N1,
        m=m,
        n_slots=n,
        nslots=n,
    )
    if proj_mix is None:
        raise RuntimeError("missing proj_mix")
    QK_grid, V_grid = cs(
        proj_mix,
        ctx=ctx,
        ct_pairs_babies=ct_pairs_babies,
        QK_tab=QK_tab,
        V_tab=V_tab,
        d2=QKV_D,
        qk_c_used=c_used_qk,
        qk_blocks=blocks_qk,
        v_c_used=c_used_v,
        v_blocks=blocks_v,
        c_used_qk=c_used_qk,
        blocks_qk=blocks_qk,
        c_used_v=c_used_v,
        blocks_v=blocks_v,
        m=m,
        N1=QKV_N1,
        C=C,
        n_slots=n,
        nslots=n,
    )
    QK_blocks = fgs(ctx, QK_grid)
    Q_blocks, K_blocks = split_qk(ctx, QK_blocks)
    Q_blocks = [ct_real(ctx, ct) for ct in Q_blocks]
    K_blocks = [ct_real(ctx, ct) for ct in K_blocks]
    V_blocks = fgs(ctx, V_grid)
    V_blocks = [ct_real(ctx, ct) for ct in V_blocks]
    if bQ_fdp is not None:
        Q_blocks = addb(ctx, Q_blocks, bQ_fdp, m=m, stride=c_used_qk)
    if bK_fdp is not None:
        K_blocks = addb(ctx, K_blocks, bK_fdp, m=m, stride=c_used_qk)
    if bV_hm is not None:
        V_blocks = addb(ctx, V_blocks, bV_hm, m=m)
    return (Q_blocks, K_blocks, V_blocks, blocks_qk, c_used_qk, blocks_v, c_used_v, (WQ_fdp, WK_fdp, WV_hm))


def run_s(ctx: CKKSContext, *, Q_cts: list[Cipher], K_cts: list[Cipher], blocks: int, c_used: int):
    m = QKV_M
    H = QKV_H
    b = B_FOLD
    mp_half, g = mapf(m, b)
    Q_used = [c_used] * blocks
    K_used = [c_used] * blocks
    Q_bank, K_bank, K_bank_h = mk_bank(ctx, Q_cts, Q_used, K_cts, K_used, m=m, b=b, g=g)
    raw_fold = emit_f(Q_bank, K_bank, K_bank_h, mp_half, m=m)
    if c_used % H == 0:
        r_per_bid = [0 for _ in range(blocks)]
    else:
        r_per_bid = [bid * c_used % H for bid in range(blocks)]
    half = m // 2
    D_fold: list[Cipher] = []
    for t in range(half):
        term_by_r = {}
        for bid, _, term in raw_fold[t]:
            r = r_per_bid[bid]
            term_by_r[r] = term if r not in term_by_r else term_by_r[r].add(term)
        acc = None
        for r, term_sum in term_by_r.items():
            red = red_h(ctx, term_sum, used_cols=c_used, H=H, m=m)
            if r != 0:
                red = aln_h(ctx, red, H=H, m=m, r=r)
            acc = red if acc is None else acc.add(red)
        D_fold.append(acc)
    return pack_f(ctx, D_fold, H=H, m=m)


def mpc_repack(ctx: CKKSContext, *, P: list[Cipher], H: int, m: int) -> tuple[list[Cipher], list[np.ndarray]]:
    mp_half, _ = mapf(m, B_FOLD)
    S_scores = unpack_f(ctx, P, H=H, m=m, mp_half=mp_half)
    A_all = [smx(S_scores[h]) for h in range(H)]
    C = ctx.nslots // m
    heads_per_ct = C // (QKV_D // H)
    groups = H // heads_per_ct
    A_cts: list[Cipher] = []
    for g in range(groups):
        ids = [g * heads_per_ct + i for i in range(heads_per_ct)]
        A_cts.append(pack_a(ctx, [A_all[h] for h in ids], m=m, heads_per_ct=heads_per_ct))
    return (A_cts, A_all)


def run_v(ctx: CKKSContext, *, A_cts: list[Cipher], V_cts: list[Cipher]) -> list[Cipher]:
    m = QKV_M
    C = ctx.nslots // m
    heads_per_ct = C // (QKV_D // QKV_H)
    groups = QKV_H // heads_per_ct
    assert len(A_cts) == groups and len(V_cts) == groups
    Y_groups: list[Cipher] = []
    for g in range(groups):
        Yg, _ = sv(ctx, A_packed=A_cts[g], V_ct=V_cts[g], m=m, d_h=QKV_D // QKV_H, heads_per_ct=heads_per_ct)
        Y_groups.append(Yg)
    return Y_groups


def _pack_a_diag_pair_ct(ctx, A_heads, *, t, m, d_h, heads_per_ct):

    n = ctx.nslots
    half = m // 2
    v = np.zeros(n, dtype=np.complex128)
    mat = v.reshape(n // m, m)
    rows = np.arange(m, dtype=np.int32)
    cols0 = (rows - (t % m)) % m
    cols1 = (rows - ((t + half) % m)) % m
    for h in range(heads_per_ct):
        A = A_heads[h]
        d0 = A[rows, cols0].astype(np.complex128, copy=False)
        d1 = A[rows, cols1].astype(np.complex128, copy=False)
        pair = d0 + 1j * d1
        seg_lo = h * d_h
        seg_hi = seg_lo + d_h
        mat[seg_lo:seg_hi, :] = pair[None, :]
    v.setflags(write=False)
    return ctx.encorypt(v)


def _sv_pair_direct(ctx, *, A_heads, V_ct, m, d_h, heads_per_ct, rot_within_fn):

    from collections import defaultdict

    from src.fhe.simulator.value import _add_assign
    from src.fhe.simulator.value import rot_within as _rot_within

    n = ctx.nslots
    C = n // m
    half = m // 2
    rot_break = defaultdict(int)
    mask_cache = {}
    rot_cache = {}
    _rwf = rot_within_fn or _rot_within

    A_pair_cts = [_pack_a_diag_pair_ct(ctx, A_heads, t=t, m=m, d_h=d_h, heads_per_ct=heads_per_ct) for t in range(half)]

    V_half = _rwf(
        ctx,
        V_ct,
        m=m,
        used_C=C,
        t=-half % m,
        mask_cache=mask_cache,
        rot_cache=rot_cache,
        rot_break=rot_break,
        tag="V_halfshift_pd",
    )
    U = _add_assign(V_ct, V_half.mul_scalar(-1j))

    out = None
    U_mask_cache = {}
    U_rot_cache = {}
    for t in range(half):
        if t == 0:
            U_t = U
        else:
            U_t = _rwf(
                ctx,
                U,
                m=m,
                used_C=C,
                t=-t % m,
                mask_cache=U_mask_cache,
                rot_cache=U_rot_cache,
                rot_break=rot_break,
                tag="U_shift_pd",
            )
        term = U_t.mul_ct(A_pair_cts[t], relin=True)
        out = _add_assign(out, term)
    return out


def run_v_pair_ct(ctx: CKKSContext, *, A_pair_cts: list[Cipher], V_cts: list[Cipher]) -> list[Cipher]:

    from collections import defaultdict

    from src.fhe.simulator.value import _add_assign
    from src.fhe.simulator.value import rot_within as _rot_within

    m = QKV_M
    C = ctx.nslots // m
    d_h = QKV_D // QKV_H
    heads_per_ct = C // d_h
    groups = QKV_H // heads_per_ct
    half = m // 2
    assert len(V_cts) == groups
    assert len(A_pair_cts) == groups * half

    Y_groups: list[Cipher] = []
    for g in range(groups):
        rot_break = defaultdict(int)
        mask_cache = {}
        rot_cache = {}

        a_cts_g = A_pair_cts[g * half : (g + 1) * half]

        V_half = _rot_within(
            ctx,
            V_cts[g],
            m=m,
            used_C=C,
            t=-half % m,
            mask_cache=mask_cache,
            rot_cache=rot_cache,
            rot_break=rot_break,
            tag="V_halfshift_pd",
        )
        U = _add_assign(V_cts[g], V_half.mul_scalar(-1j))

        out = None
        U_mask_cache = {}
        U_rot_cache = {}
        for t in range(half):
            if t == 0:
                U_t = U
            else:
                U_t = _rot_within(
                    ctx,
                    U,
                    m=m,
                    used_C=C,
                    t=-t % m,
                    mask_cache=U_mask_cache,
                    rot_cache=U_rot_cache,
                    rot_break=rot_break,
                    tag="U_shift_pd",
                )
            term = U_t.mul_ct(a_cts_g[t], relin=True)
            out = _add_assign(out, term)
        Y_groups.append(out)
    return Y_groups


def run_v_direct(ctx: CKKSContext, *, A_all: list[np.ndarray], V_cts: list[Cipher]) -> list[Cipher]:

    m = QKV_M
    C = ctx.nslots // m
    d_h = QKV_D // QKV_H
    heads_per_ct = C // d_h
    groups = QKV_H // heads_per_ct
    assert len(V_cts) == groups

    Y_groups: list[Cipher] = []
    for g in range(groups):
        ids = [g * heads_per_ct + i for i in range(heads_per_ct)]
        A_heads = [A_all[h] for h in ids]
        Yg = _sv_pair_direct(
            ctx, A_heads=A_heads, V_ct=V_cts[g], m=m, d_h=d_h, heads_per_ct=heads_per_ct, rot_within_fn=None
        )
        Y_groups.append(Yg)
    return Y_groups


def to_mpc_mat_complex_packed(ctx, blocks, *, m, d_out, bridge=None):

    n = ctx.nslots
    c = n // m
    out = np.zeros((m, d_out), dtype=np.float64)
    b = 0
    for ct in blocks:
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
    return out, len(blocks)


def run(*, use_cc: bool = True, use_scp: bool = True, mpc_engine: str | None = None):
    import time as _time

    _timings = {}
    np.random.seed(SEED)
    ctx = CKKSContext(QKV_NSLOTS)
    mpc_backend = get_mpc_engine(mpc_engine)
    print(f"[MPC] engine={mpc_backend.name} device={mpc_backend.device}")
    A = np.random.randn(QKV_M, QKV_D1).astype(np.float64)
    WQ = np.random.randn(QKV_D1, QKV_D).astype(np.float64)
    WK = np.random.randn(QKV_D1, QKV_D).astype(np.float64)
    WV = np.random.randn(QKV_D1, QKV_D).astype(np.float64)
    bQ = np.random.randn(QKV_D).astype(np.float64)
    bK = np.random.randn(QKV_D).astype(np.float64)
    bV = np.random.randn(QKV_D).astype(np.float64)
    bO = np.random.randn(QKV_D).astype(np.float64)
    b1 = np.random.randn(4 * QKV_D).astype(np.float64)
    b2 = np.random.randn(QKV_D).astype(np.float64)
    ln1_w = np.random.randn(QKV_D).astype(np.float64)
    ln1_b = np.random.randn(QKV_D).astype(np.float64)
    ln2_w = np.random.randn(QKV_D).astype(np.float64)
    ln2_b = np.random.randn(QKV_D).astype(np.float64)
    s0 = stat_snap(ctx)
    _t0 = _time.perf_counter()
    Q_cts, K_cts, V_cts, blocks_qk, c_used_qk, blocks_v, c_used_v, W_views = run_q(
        ctx, A=A, WQ=WQ, WK=WK, WV=WV, bQ=bQ, bK=bK, bV=bV
    )
    _timings["QKV"] = _time.perf_counter() - _t0
    s1 = stat_snap(ctx)
    d01 = stat_diff(s0, s1)
    repk = int(os.environ.get("SCP_REPACK_ITERS", "30"))
    if not use_scp:
        Q_cts = repack_id(Q_cts, m=QKV_M, iters=repk)
        K_cts = repack_id(K_cts, m=QKV_M, iters=repk)
    _t0 = _time.perf_counter()
    P = run_s(ctx, Q_cts=Q_cts, K_cts=K_cts, blocks=blocks_qk, c_used=c_used_qk)
    _timings["Score"] = _time.perf_counter() - _t0
    if not use_scp:
        P = repack_id(P, m=QKV_M, iters=repk)
    s2 = stat_snap(ctx)
    d12 = stat_diff(s1, s2)

    mp_half_pre, _ = mapf(QKV_M, B_FOLD)
    S_dec_pre = unpack_f(ctx, P, H=QKV_H, m=QKV_M, mp_half=mp_half_pre)
    _t0 = _time.perf_counter()
    A_cts, A_all = roundA(ctx, P, H=QKV_H, m=QKV_M, b_fold=B_FOLD, use_cc=use_cc, mpc_engine=mpc_backend)
    _timings["MPC-A"] = _time.perf_counter() - _t0
    s4 = stat_snap(ctx)
    d24 = stat_diff(s2, s4)
    if not use_scp:
        A_cts = repack_id(A_cts, m=QKV_M, iters=repk)
        V_cts = repack_id(V_cts, m=QKV_M, iters=repk)
    _t0 = _time.perf_counter()
    Y_groups = run_v(ctx, A_cts=A_cts, V_cts=V_cts)
    _timings["Value"] = _time.perf_counter() - _t0
    if not use_scp:
        Y_groups = repack_id(Y_groups, m=QKV_M, iters=repk)
    s5 = stat_snap(ctx)
    d45 = stat_diff(s4, s5)
    W_O = np.random.randn(QKV_D, QKV_D).astype(np.float64)
    s6 = stat_snap(ctx)
    _t0 = _time.perf_counter()
    Z_blocks, _ = run_out(ctx, Y_groups=Y_groups, W_O=W_O, N1=16)
    Z_blocks = addb(ctx, Z_blocks, bO, m=QKV_M)
    _timings["OUT"] = _time.perf_counter() - _t0
    s7 = stat_snap(ctx)
    d67 = stat_diff(s6, s7)
    s7a = stat_snap(ctx)
    _t0 = _time.perf_counter()
    A_blocks = pack_scp_blocks(ctx, A, m=QKV_M)
    Z_res_blocks = addl(Z_blocks, A_blocks)
    Z_ln_blocks, mpc_b_meta = roundB(
        ctx,
        Z_res_blocks,
        m=QKV_M,
        d_out=QKV_D,
        gamma=ln1_w,
        beta=ln1_b,
        return_meta=True,
        use_cc=use_cc,
        mpc_engine=mpc_backend,
    )
    _timings["MPC-B"] = _time.perf_counter() - _t0
    s7b = stat_snap(ctx)
    d7ab = stat_diff(s7a, s7b)
    W1 = np.random.randn(QKV_D, 4 * QKV_D).astype(np.float64)
    W2 = np.random.randn(4 * QKV_D, QKV_D).astype(np.float64)
    s8 = stat_snap(ctx)
    _t0 = _time.perf_counter()
    H1_blocks, _ = run_lin(ctx, X_blocks=Z_ln_blocks, W=W1, N1=32)
    H1_blocks = addb(ctx, H1_blocks, b1, m=QKV_M)
    _timings["FF1"] = _time.perf_counter() - _t0
    s9 = stat_snap(ctx)
    d89 = stat_diff(s8, s9)
    s9a = stat_snap(ctx)
    _t0 = _time.perf_counter()
    H1_gelu_blocks, mpc_c_meta = roundC(
        ctx, H1_blocks, m=QKV_M, d_out=4 * QKV_D, return_meta=True, use_cc=use_cc, mpc_engine=mpc_backend
    )
    _timings["MPC-C"] = _time.perf_counter() - _t0
    s9b = stat_snap(ctx)
    d9ab = stat_diff(s9a, s9b)
    s10 = stat_snap(ctx)
    _t0 = _time.perf_counter()
    H2_blocks, _ = run_lin(ctx, X_blocks=H1_gelu_blocks, W=W2, N1=8)
    H2_blocks = addb(ctx, H2_blocks, b2, m=QKV_M)
    _timings["FF2"] = _time.perf_counter() - _t0
    s11 = stat_snap(ctx)
    d1011 = stat_diff(s10, s11)
    s11a = stat_snap(ctx)
    _t0 = _time.perf_counter()
    H2_res_blocks = addl(H2_blocks, Z_ln_blocks)
    H2_ln_blocks, mpc_d_meta = roundD(
        ctx,
        H2_res_blocks,
        m=QKV_M,
        d_out=QKV_D,
        gamma=ln2_w,
        beta=ln2_b,
        return_meta=True,
        use_cc=use_cc,
        mpc_engine=mpc_backend,
    )
    _timings["MPC-D"] = _time.perf_counter() - _t0
    s11b = stat_snap(ctx)
    d11ab = stat_diff(s11a, s11b)
    WQ_fdp, WK_fdp, WV_hm = W_views
    Dh = QKV_D // QKV_H
    perm_fdp_cols = perm_fdp(QKV_H, Dh)
    perm_hm_cols = perm_hd(QKV_H, Dh)
    bQ_fdp = bQ[perm_fdp_cols]
    bK_fdp = bK[perm_fdp_cols]
    bV_hm = bV[perm_hm_cols]
    Q_plain = A @ WQ_fdp + bQ_fdp
    K_plain = A @ WK_fdp + bK_fdp
    V_plain = A @ WV_hm + bV_hm
    Q_dec = dec_fdp(ctx, Q_cts, m=QKV_M, used=c_used_qk)[:, :QKV_D]
    K_dec = dec_fdp(ctx, K_cts, m=QKV_M, used=c_used_qk)[:, :QKV_D]
    errQ = rel_err(Q_dec, Q_plain)
    errK = rel_err(K_dec, K_plain)
    mp_half, _ = mapf(QKV_M, B_FOLD)
    inv_fdp = np.empty_like(perm_fdp_cols)
    inv_fdp[perm_fdp_cols] = np.arange(len(perm_fdp_cols))
    Q_hm = Q_plain[:, inv_fdp]
    K_hm = K_plain[:, inv_fdp]
    S_ref = score_ref(Q_hm, K_hm, QKV_H)
    S_dec = unpack_f(ctx, P, H=QKV_H, m=QKV_M, mp_half=mp_half)
    errS = rel_err(S_dec, S_ref)
    V_dec = dec_fdp(ctx, V_cts, m=QKV_M, used=c_used_v)
    errV = rel_err(V_dec, V_plain)
    Y_truth_all = [A_all[h] @ V_plain[:, h * Dh : (h + 1) * Dh] for h in range(QKV_H)]
    Y_truth = np.concatenate(Y_truth_all, axis=1)
    Cseg = ctx.nslots // QKV_M
    Y_dec = np.zeros((QKV_M, QKV_D), dtype=np.float64)
    for g, ct in enumerate(Y_groups):
        blk = dec_grp(ctx, ct, m=QKV_M)
        Y_dec[:, g * Cseg : (g + 1) * Cseg] = blk
    errY = rel_err(Y_dec, Y_truth)
    Z_truth = Y_truth @ W_O + bO
    OUT_dec = decb(ctx, Z_blocks, m=QKV_M, d_out=QKV_D)
    errOUT = rel_err(OUT_dec, Z_truth)
    Z_ln_dec = decb(ctx, Z_ln_blocks, m=QKV_M, d_out=QKV_D)
    H1_truth = Z_ln_dec @ W1 + b1
    FF1_dec = decb(ctx, H1_blocks, m=QKV_M, d_out=4 * QKV_D)
    errFF1 = rel_err(FF1_dec, H1_truth)
    H1_gelu_dec = decb(ctx, H1_gelu_blocks, m=QKV_M, d_out=4 * QKV_D)
    H1_gelu_tanh_ref = gelu_tanh(H1_truth)
    H1_gelu_secure_ref = gelu_secure_ref(H1_truth)
    err_mpc_c_tanh_ref = rel_err(H1_gelu_dec, H1_gelu_tanh_ref)
    err_mpc_c_secure_ref = rel_err(H1_gelu_dec, H1_gelu_secure_ref)
    H2_truth = H1_gelu_dec @ W2 + b2
    FF2_dec = decb(ctx, H2_blocks, m=QKV_M, d_out=QKV_D)
    errFF2 = rel_err(FF2_dec, H2_truth)

    errMPC_A_num = 0.0
    errMPC_A_den = 0.0
    for h in range(QKV_H):
        smx_ref_h = smx_rows(S_dec_pre[h])
        errMPC_A_num += float(np.linalg.norm(A_all[h] - smx_ref_h))
        errMPC_A_den += float(np.linalg.norm(smx_ref_h)) + 1e-12
    errMPC_A = errMPC_A_num / errMPC_A_den

    Z_res_dec = decb(ctx, Z_res_blocks, m=QKV_M, d_out=QKV_D)
    Z_ln_ref = ln_apply(Z_res_dec, gamma=ln1_w, beta=ln1_b)
    errMPC_B = rel_err(Z_ln_dec, Z_ln_ref)

    H2_res_dec = decb(ctx, H2_res_blocks, m=QKV_M, d_out=QKV_D)
    H2_ln_dec = decb(ctx, H2_ln_blocks, m=QKV_M, d_out=QKV_D)
    H2_ln_ref = ln_apply(H2_res_dec, gamma=ln2_w, beta=ln2_b)
    errMPC_D = rel_err(H2_ln_dec, H2_ln_ref)
    err_qkv = (errQ + errK + errV) / 3.0
    print_stage(
        "QKV",
        ct_in=None,
        ct_out=None,
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=err_qkv,
    )
    print_stage(
        "Score",
        ct_in=None,
        ct_out=None,
        ks_rots=d12["ks_rots"],
        ks_muls=d12["ks_muls_ctct"],
        ks_conj=d12["ks_conj"],
        rel_err=errS,
    )
    mpc_a_in = len(P) if use_cc else 2 * len(P)
    mpc_a_out = len(A_cts) if use_cc else 2 * len(A_cts)
    print_stage(
        "MPC-A",
        ct_in=mpc_a_in,
        ct_out=mpc_a_out,
        ks_rots=d24["ks_rots"],
        ks_muls=d24["ks_muls_ctct"],
        ks_conj=d24["ks_conj"],
        rel_err=errMPC_A,
    )
    print_stage(
        "Value",
        ct_in=None,
        ct_out=None,
        ks_rots=d45["ks_rots"],
        ks_muls=d45["ks_muls_ctct"],
        ks_conj=d45["ks_conj"],
        rel_err=errY,
    )
    print_stage(
        "OUT",
        ct_in=None,
        ct_out=None,
        ks_rots=d67["ks_rots"],
        ks_muls=d67["ks_muls_ctct"],
        ks_conj=d67["ks_conj"],
        rel_err=errOUT,
    )
    print_stage(
        "MPC-B",
        ct_in=mpc_b_meta["ct_in"],
        ct_out=mpc_b_meta["ct_out"],
        ks_rots=d7ab["ks_rots"],
        ks_muls=d7ab["ks_muls_ctct"],
        ks_conj=d7ab["ks_conj"],
        rel_err=errMPC_B,
    )
    print_stage(
        "FF1",
        ct_in=None,
        ct_out=None,
        ks_rots=d89["ks_rots"],
        ks_muls=d89["ks_muls_ctct"],
        ks_conj=d89["ks_conj"],
        rel_err=errFF1,
    )
    print_stage(
        "MPC-C",
        ct_in=mpc_c_meta["ct_in"],
        ct_out=mpc_c_meta["ct_out"],
        ks_rots=d9ab["ks_rots"],
        ks_muls=d9ab["ks_muls_ctct"],
        ks_conj=d9ab["ks_conj"],
        rel_err=err_mpc_c_secure_ref,
    )
    print(f"[MPC-C-REF] rel_err_secure={err_mpc_c_secure_ref:.3e} rel_err_tanh={err_mpc_c_tanh_ref:.3e}")
    print_stage(
        "FF2",
        ct_in=None,
        ct_out=None,
        ks_rots=d1011["ks_rots"],
        ks_muls=d1011["ks_muls_ctct"],
        ks_conj=d1011["ks_conj"],
        rel_err=errFF2,
    )
    print_stage(
        "MPC-D",
        ct_in=mpc_d_meta["ct_in"],
        ct_out=mpc_d_meta["ct_out"],
        ks_rots=d11ab["ks_rots"],
        ks_muls=d11ab["ks_muls_ctct"],
        ks_conj=d11ab["ks_conj"],
        rel_err=errMPC_D,
    )
    _timings["Total"] = sum(v for k, v in _timings.items() if k != "Total")
    print("\n  [Timing]", " | ".join(f"{k}={v:.2f}s" for k, v in _timings.items()))
    bert_chk(seq_len=QKV_M)
    print("\nDone.")


def bert_chk(seq_len: int = 128) -> None:
    try:
        import torch
        from transformers import BertConfig, BertModel
    except Exception as exc:
        print(f"[BERT] transformers/torch not available: {exc}")
        return

    def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.linalg.norm(a - b))
        den = float(np.linalg.norm(b)) + 1e-12
        return num / den

    torch.set_grad_enabled(False)
    config = BertConfig()
    model = BertModel(config)
    model.eval()
    hidden = config.hidden_size
    heads = config.num_attention_heads
    dh = hidden // heads
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, seq_len))
    with torch.no_grad():
        emb = model.embeddings(input_ids=input_ids)
        layer = model.encoder.layer[0]
    dtype = np.float32
    A = emb.squeeze(0).cpu().numpy().astype(dtype, copy=False)
    WQ = layer.attention.self.query.weight.cpu().numpy().T.astype(dtype, copy=False)
    WK = layer.attention.self.key.weight.cpu().numpy().T.astype(dtype, copy=False)
    WV = layer.attention.self.value.weight.cpu().numpy().T.astype(dtype, copy=False)
    bQ = layer.attention.self.query.bias.cpu().numpy().astype(dtype, copy=False)
    bK = layer.attention.self.key.bias.cpu().numpy().astype(dtype, copy=False)
    bV = layer.attention.self.value.bias.cpu().numpy().astype(dtype, copy=False)
    WO = layer.attention.output.dense.weight.cpu().numpy().T.astype(dtype, copy=False)
    bO = layer.attention.output.dense.bias.cpu().numpy().astype(dtype, copy=False)
    ln1_w = layer.attention.output.LayerNorm.weight.cpu().numpy().astype(dtype, copy=False)
    ln1_b = layer.attention.output.LayerNorm.bias.cpu().numpy().astype(dtype, copy=False)
    W1 = layer.intermediate.dense.weight.cpu().numpy().T.astype(dtype, copy=False)
    b1 = layer.intermediate.dense.bias.cpu().numpy().astype(dtype, copy=False)
    W2 = layer.output.dense.weight.cpu().numpy().T.astype(dtype, copy=False)
    b2 = layer.output.dense.bias.cpu().numpy().astype(dtype, copy=False)
    ln2_w = layer.output.LayerNorm.weight.cpu().numpy().astype(dtype, copy=False)
    ln2_b = layer.output.LayerNorm.bias.cpu().numpy().astype(dtype, copy=False)
    Q = A @ WQ + bQ
    K = A @ WK + bK
    V = A @ WV + bV
    Qh = Q.reshape(seq_len, heads, dh)
    Kh = K.reshape(seq_len, heads, dh)
    Vh = V.reshape(seq_len, heads, dh)
    scale = dtype(np.sqrt(dh))
    scores = np.zeros((heads, seq_len, seq_len), dtype=dtype)
    for h in range(heads):
        scores[h] = Qh[:, h, :] @ Kh[:, h, :].T / scale
    eps = dtype(1e-12)
    eps_ln = dtype(1e-05)
    scores = scores - scores.max(axis=2, keepdims=True).astype(dtype, copy=False)
    P = np.exp(scores)
    P = P / (P.sum(axis=2, keepdims=True) + eps)
    ctx_np = np.zeros((seq_len, heads, dh), dtype=dtype)
    for h in range(heads):
        ctx_np[:, h, :] = P[h] @ Vh[:, h, :]
    ctx_np = ctx_np.reshape(seq_len, hidden)
    attn = ctx_np @ WO + bO

    def _ln_np(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=1, keepdims=True, dtype=dtype)
        var = np.var(x, axis=1, keepdims=True, dtype=dtype)
        return (x - mean) / np.sqrt(var + eps_ln)

    x1 = attn + A
    x1 = _ln_np(x1)
    x1 = x1 * ln1_w + ln1_b
    h1 = x1 @ W1 + b1
    h1_gelu = 0.5 * h1 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h1 + 0.044715 * h1**3)))
    h2 = h1_gelu @ W2 + b2
    x2 = h2 + x1
    x2 = _ln_np(x2)
    x2 = x2 * ln2_w + ln2_b
    A_t = emb.squeeze(0)
    WQ_t = torch.from_numpy(WQ).to(A_t)
    WK_t = torch.from_numpy(WK).to(A_t)
    WV_t = torch.from_numpy(WV).to(A_t)
    bQ_t = torch.from_numpy(bQ).to(A_t)
    bK_t = torch.from_numpy(bK).to(A_t)
    bV_t = torch.from_numpy(bV).to(A_t)
    WO_t = torch.from_numpy(WO).to(A_t)
    bO_t = torch.from_numpy(bO).to(A_t)
    ln1_w_t = torch.from_numpy(ln1_w).to(A_t)
    ln1_b_t = torch.from_numpy(ln1_b).to(A_t)
    W1_t = torch.from_numpy(W1).to(A_t)
    b1_t = torch.from_numpy(b1).to(A_t)
    W2_t = torch.from_numpy(W2).to(A_t)
    b2_t = torch.from_numpy(b2).to(A_t)
    ln2_w_t = torch.from_numpy(ln2_w).to(A_t)
    ln2_b_t = torch.from_numpy(ln2_b).to(A_t)
    Q_t = A_t @ WQ_t + bQ_t
    K_t = A_t @ WK_t + bK_t
    V_t = A_t @ WV_t + bV_t
    Qh_t = Q_t.reshape(seq_len, heads, dh)
    Kh_t = K_t.reshape(seq_len, heads, dh)
    Vh_t = V_t.reshape(seq_len, heads, dh)
    scale_t = torch.tensor(float(np.sqrt(dh)), dtype=Q_t.dtype, device=Q_t.device)
    scores_t = torch.zeros((heads, seq_len, seq_len), dtype=Q_t.dtype, device=Q_t.device)
    for h in range(heads):
        scores_t[h] = Qh_t[:, h, :] @ Kh_t[:, h, :].T / scale_t
    eps_t = torch.tensor(1e-12, dtype=Q_t.dtype, device=Q_t.device)
    eps_ln_t = torch.tensor(1e-05, dtype=Q_t.dtype, device=Q_t.device)
    scores_t = scores_t - scores_t.max(dim=2, keepdim=True).values
    P_t = torch.exp(scores_t)
    P_t = P_t / (P_t.sum(dim=2, keepdim=True) + eps_t)
    ctx_t = torch.zeros((seq_len, heads, dh), dtype=Q_t.dtype, device=Q_t.device)
    for h in range(heads):
        ctx_t[:, h, :] = P_t[h] @ Vh_t[:, h, :]
    ctx_t = ctx_t.reshape(seq_len, hidden)
    attn_t = ctx_t @ WO_t + bO_t
    x1_t = attn_t + A_t
    x1_t = (x1_t - x1_t.mean(dim=1, keepdim=True)) / torch.sqrt(
        x1_t.var(dim=1, keepdim=True, unbiased=False) + eps_ln_t
    )
    x1_t = x1_t * ln1_w_t + ln1_b_t
    h1_t = x1_t @ W1_t + b1_t
    h1_gelu_t = 0.5 * h1_t * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (h1_t + 0.044715 * h1_t**3)))
    h2_t = h1_gelu_t @ W2_t + b2_t
    x2_t = h2_t + x1_t
    x2_t = (x2_t - x2_t.mean(dim=1, keepdim=True)) / torch.sqrt(
        x2_t.var(dim=1, keepdim=True, unbiased=False) + eps_ln_t
    )
    x2_t = x2_t * ln2_w_t + ln2_b_t
    final_err = _rel_err(x2, x2_t.cpu().numpy())
    print(f"\n[BERT] rel_err={final_err:.3e}")


def _make_ckks_context(backend: str, nslots: int, gpu: str | None = None):

    if backend == "plain":
        return CKKSContext(nslots)
    elif backend == "phantom":
        from src.fhe.phantom.phantom_runtime import (
            LevelCKKSContext as PhantomLevelCtx,
        )
        from src.fhe.phantom.phantom_runtime import (
            set_visible_gpus,
        )

        if gpu is not None:
            set_visible_gpus(gpu)

        enc_level = _SEG_ENC_LEVEL if _SEG_ENC_LEVEL is not None else 1

        log_coeff = int(os.getenv("PHANTOM_LOG_COEFF", "15"))
        special_primes = int(os.getenv("PHANTOM_SPECIAL_PRIMES", "4"))
        return PhantomLevelCtx(
            default_enc_level=enc_level,
            nslots=nslots,
            log_coeff_count=log_coeff,
            special_prime_count=special_primes,
        )
    elif backend == "desilo":
        from src.fhe.desilo.desilo_runtime import (
            LevelCKKSContext as DesiloLevelCtx,
        )
        from src.fhe.desilo.desilo_runtime import (
            set_visible_gpus,
        )

        if gpu is not None:
            set_visible_gpus(gpu)
        mode = "gpu" if gpu is not None else "cpu"
        enc_level = _SEG_ENC_LEVEL if _SEG_ENC_LEVEL is not None else None
        return DesiloLevelCtx(
            default_enc_level=enc_level,
            log_coeff_count=15,
            special_prime_count=2,
            mode=mode,
        )
    else:
        raise ValueError(f"Unknown ckks_backend: {backend!r}. Choose plain/phantom/desilo.")


def run_with_weights(
    *,
    weights: dict,
    running_denoms: dict | None = None,
    input_embeds: np.ndarray,
    attention_mask: np.ndarray | None = None,
    layer_idx: int = 0,
    use_cc: bool = True,
    mpc_engine: str | None = None,
    ckks_backend: str = "plain",
    gpu: str | None = None,
    ctx=None,
    mpc_backend=None,
    prepared_denoms: dict[str, np.ndarray] | None = None,
    verbose: bool = True,
    pre_ln: bool = False,
    return_timings: bool = False,
    bridge=None,
    secure_bridge: bool = False,
    return_encrypted: bool = False,
    cfg: ModelConfig | None = None,
    progress=None,
    total_layers: int = 1,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:

    _cfg = _resolve_cfg(cfg)

    if cfg is not None and _cfg.name != _MODEL_CFG.name:
        reconfigure(_cfg.name)
    effective_ckks_backend = _effective_ckks_backend(ckks_backend)
    if effective_ckks_backend == "phantom_native":
        if pre_ln:
            raise NotImplementedError(
                "phantom_native weighted execution is currently implemented for post-LN BERT-style layers only."
            )
        if return_encrypted:
            raise NotImplementedError("phantom_native weighted execution does not yet support return_encrypted=True.")
        if bridge is not None or secure_bridge:
            raise NotImplementedError(
                "phantom_native weighted execution does not yet support the "
                "external secure bridge path. Use the local/native benchmark path "
                "instead."
            )

        from src.fhe.phantom.phantom_native_pipe import run_native_with_weights

        mpc_backend = mpc_backend if mpc_backend is not None else get_mpc_engine(mpc_engine)
        if prepared_denoms is None:
            prepared_denoms = prepare_layer_running_denoms(running_denoms, layer_idx)
        return run_native_with_weights(
            weights=weights,
            running_denoms=running_denoms,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            gpu="0" if gpu is None else str(gpu),
            mpc_engine=mpc_engine,
            mpc_backend=mpc_backend,
            prepared_denoms=prepared_denoms,
            verbose=verbose,
            return_timings=return_timings,
            model_config=_cfg.name,
        )

    ctx = ctx if ctx is not None else _make_ckks_context(effective_ckks_backend, _cfg.nslots, gpu=gpu)
    mpc_backend = mpc_backend if mpc_backend is not None else get_mpc_engine(mpc_engine)
    if prepared_denoms is None:
        prepared_denoms = prepare_layer_running_denoms(running_denoms, layer_idx)
    inject_layer_running_denoms(mpc_backend, prepared_denoms)

    _bridge = bridge
    if _bridge is None and secure_bridge:
        from src.bridges.in_process_bridge import InProcessBridge

        _bridge = InProcessBridge(ctx)

    import time as _time

    _timings = {}

    def _emit(stage: str):
        if progress is not None:
            progress.on_stage_complete(
                layer=layer_idx,
                total_layers=total_layers,
                stage=stage,
                elapsed_s=_timings[stage],
                timings_so_far=_timings,
            )

    def _emit_start(stage: str):
        if progress is not None:
            progress.on_stage_start(
                layer=layer_idx,
                total_layers=total_layers,
                stage=stage,
            )

    A = input_embeds.astype(np.float64)
    m = _cfg.m
    d = _cfg.d_model

    WQ = weights["WQ"]
    WK = weights["WK"]
    WV = weights["WV"]
    bQ = weights.get("bQ")
    bK = weights.get("bK")
    bV = weights.get("bV")
    WO = weights["WO"]
    bO = weights.get("bO")
    ln1_w = weights.get("ln1_w")
    ln1_b = weights.get("ln1_b")
    W1 = weights["W1"]
    b1 = weights.get("b1")
    W2 = weights["W2"]
    b2 = weights.get("b2")
    ln2_w = weights.get("ln2_w")
    ln2_b = weights.get("ln2_b")

    if pre_ln:
        _emit_start("RoundB (LN1)")
        _t0 = _time.perf_counter()
        A_blocks = pack_scp_blocks(ctx, A, m=m)
        A_ln1_blocks = roundB(
            ctx, A_blocks, m=m, d_out=d, gamma=ln1_w, beta=ln1_b, use_cc=use_cc, mpc_engine=mpc_backend, bridge=_bridge
        )
        A_ln1 = decb(ctx, A_ln1_blocks, m=m, d_out=d)
        _timings["RoundB (LN1)"] = _time.perf_counter() - _t0
        _emit("RoundB (LN1)")

        _emit_start("QKV")
        _t0 = _time.perf_counter()
        Q_cts, K_cts, V_cts, blocks_qk, c_used_qk, blocks_v, c_used_v, _ = run_q(
            ctx, A=A_ln1, WQ=WQ, WK=WK, WV=WV, bQ=bQ, bK=bK, bV=bV
        )
        _timings["QKV"] = _time.perf_counter() - _t0
        _emit("QKV")

        _emit_start("Score")
        _t0 = _time.perf_counter()
        P = run_s(ctx, Q_cts=Q_cts, K_cts=K_cts, blocks=blocks_qk, c_used=c_used_qk)
        _timings["Score"] = _time.perf_counter() - _t0
        _emit("Score")

        _emit_start("RoundA (Softmax)")
        _t0 = _time.perf_counter()
        d_k = d // _cfg.H
        A_cts, _ = roundA(
            ctx,
            P,
            H=_cfg.H,
            m=m,
            d_k=d_k,
            b_fold=B_FOLD,
            use_cc=use_cc,
            mpc_engine=mpc_backend,
            attn_mask=attention_mask,
            bridge=_bridge,
        )
        _timings["RoundA (Softmax)"] = _time.perf_counter() - _t0
        _emit("RoundA (Softmax)")

        _emit_start("Value")
        _t0 = _time.perf_counter()
        Y_groups = run_v(ctx, A_cts=A_cts, V_cts=V_cts)
        _timings["Value"] = _time.perf_counter() - _t0
        _emit("Value")

        _emit_start("Output Proj")
        _t0 = _time.perf_counter()
        Z_blocks, _ = run_out(ctx, Y_groups=Y_groups, W_O=WO, N1=16)
        if bO is not None:
            Z_blocks = addb(ctx, Z_blocks, bO, m=m)
        _timings["Output Proj"] = _time.perf_counter() - _t0
        _emit("Output Proj")

        _emit_start("Residual1")
        _t0 = _time.perf_counter()
        A_blocks_orig = pack_scp_blocks(ctx, A, m=m)
        x_blocks = addl(Z_blocks, A_blocks_orig)
        _timings["Residual1"] = _time.perf_counter() - _t0
        _emit("Residual1")

        _emit_start("RoundD (LN2)")
        _t0 = _time.perf_counter()
        x_ln2_blocks = roundD(
            ctx, x_blocks, m=m, d_out=d, gamma=ln2_w, beta=ln2_b, use_cc=use_cc, mpc_engine=mpc_backend, bridge=_bridge
        )
        _timings["RoundD (LN2)"] = _time.perf_counter() - _t0
        _emit("RoundD (LN2)")

        _emit_start("FF1")
        _t0 = _time.perf_counter()
        H1_blocks, _ = run_lin(ctx, X_blocks=x_ln2_blocks, W=W1, N1=32)
        if b1 is not None:
            H1_blocks = addb(ctx, H1_blocks, b1, m=m)
        _timings["FF1"] = _time.perf_counter() - _t0
        _emit("FF1")

        _emit_start("RoundC (GELU)")
        _t0 = _time.perf_counter()
        H1_gelu = roundC(ctx, H1_blocks, m=m, d_out=W1.shape[1], use_cc=use_cc, mpc_engine=mpc_backend, bridge=_bridge)
        _timings["RoundC (GELU)"] = _time.perf_counter() - _t0
        _emit("RoundC (GELU)")

        _emit_start("FF2")
        _t0 = _time.perf_counter()
        H2_blocks, _ = run_lin(ctx, X_blocks=H1_gelu, W=W2, N1=8)
        if b2 is not None:
            H2_blocks = addb(ctx, H2_blocks, b2, m=m)
        _timings["FF2"] = _time.perf_counter() - _t0
        _emit("FF2")

        _emit_start("Residual2")
        _t0 = _time.perf_counter()
        out_blocks = addl(H2_blocks, x_blocks)
        _timings["Residual2"] = _time.perf_counter() - _t0
        _emit("Residual2")

        _final_blocks = out_blocks

    else:
        _emit_start("QKV")
        _t0 = _time.perf_counter()
        Q_cts, K_cts, V_cts, blocks_qk, c_used_qk, blocks_v, c_used_v, _ = run_q(
            ctx, A=A, WQ=WQ, WK=WK, WV=WV, bQ=bQ, bK=bK, bV=bV
        )
        _timings["QKV"] = _time.perf_counter() - _t0
        _emit("QKV")

        _emit_start("Score")
        _t0 = _time.perf_counter()
        P = run_s(ctx, Q_cts=Q_cts, K_cts=K_cts, blocks=blocks_qk, c_used=c_used_qk)
        _timings["Score"] = _time.perf_counter() - _t0
        _emit("Score")

        _emit_start("RoundA (Softmax)")
        _t0 = _time.perf_counter()
        d_k = d // _cfg.H
        A_cts, _ = roundA(
            ctx,
            P,
            H=_cfg.H,
            m=m,
            d_k=d_k,
            b_fold=B_FOLD,
            use_cc=use_cc,
            mpc_engine=mpc_backend,
            attn_mask=attention_mask,
            bridge=_bridge,
        )
        _timings["RoundA (Softmax)"] = _time.perf_counter() - _t0
        _emit("RoundA (Softmax)")

        _emit_start("Value")
        _t0 = _time.perf_counter()
        Y_groups = run_v(ctx, A_cts=A_cts, V_cts=V_cts)
        _timings["Value"] = _time.perf_counter() - _t0
        _emit("Value")

        _emit_start("Output Proj")
        _t0 = _time.perf_counter()
        Z_blocks, _ = run_out(ctx, Y_groups=Y_groups, W_O=WO, N1=16)
        if bO is not None:
            Z_blocks = addb(ctx, Z_blocks, bO, m=m)
        _timings["Output Proj"] = _time.perf_counter() - _t0
        _emit("Output Proj")

        _emit_start("RoundB (LN1)")
        _t0 = _time.perf_counter()
        A_blocks = pack_scp_blocks(ctx, A, m=m)
        Z_res = addl(Z_blocks, A_blocks)
        Z_ln = roundB(
            ctx, Z_res, m=m, d_out=d, gamma=ln1_w, beta=ln1_b, use_cc=use_cc, mpc_engine=mpc_backend, bridge=_bridge
        )
        _timings["RoundB (LN1)"] = _time.perf_counter() - _t0
        _emit("RoundB (LN1)")

        _emit_start("FF1")
        _t0 = _time.perf_counter()
        H1_blocks, _ = run_lin(ctx, X_blocks=Z_ln, W=W1, N1=32)
        if b1 is not None:
            H1_blocks = addb(ctx, H1_blocks, b1, m=m)
        _timings["FF1"] = _time.perf_counter() - _t0
        _emit("FF1")

        _emit_start("RoundC (GELU)")
        _t0 = _time.perf_counter()
        H1_gelu = roundC(ctx, H1_blocks, m=m, d_out=W1.shape[1], use_cc=use_cc, mpc_engine=mpc_backend, bridge=_bridge)
        _timings["RoundC (GELU)"] = _time.perf_counter() - _t0
        _emit("RoundC (GELU)")

        _emit_start("FF2")
        _t0 = _time.perf_counter()
        H2_blocks, _ = run_lin(ctx, X_blocks=H1_gelu, W=W2, N1=8)
        if b2 is not None:
            H2_blocks = addb(ctx, H2_blocks, b2, m=m)
        _timings["FF2"] = _time.perf_counter() - _t0
        _emit("FF2")

        _emit_start("RoundD (LN2)")
        _t0 = _time.perf_counter()
        H2_res = addl(H2_blocks, Z_ln)
        H2_ln = roundD(
            ctx, H2_res, m=m, d_out=d, gamma=ln2_w, beta=ln2_b, use_cc=use_cc, mpc_engine=mpc_backend, bridge=_bridge
        )
        _timings["RoundD (LN2)"] = _time.perf_counter() - _t0
        _emit("RoundD (LN2)")

        _final_blocks = H2_ln

    if return_encrypted:
        _timings["Total"] = sum(v for k, v in _timings.items() if k != "Total")
        if verbose:
            print("  [Timing]", " | ".join(f"{k}={v:.2f}s" for k, v in _timings.items()))
        if return_timings:
            return _final_blocks, _timings
        return _final_blocks

    is_last_layer = layer_idx == total_layers - 1
    if is_last_layer:
        _emit_start("Decrypt")
    _t0 = _time.perf_counter()
    out = decb(ctx, _final_blocks, m=m, d_out=d)
    _timings["Decrypt"] = _time.perf_counter() - _t0
    if is_last_layer:
        _emit("Decrypt")

    _timings["Total"] = sum(v for k, v in _timings.items() if k != "Total")

    if verbose:
        print("  [Timing]", " | ".join(f"{k}={v:.2f}s" for k, v in _timings.items()))

    if return_timings:
        return out, _timings
    return out


def bridge_decrypt(
    ctx: CKKSContext, cts: list[Cipher], *, comm_stats=None, bridge=None
) -> list[tuple[np.ndarray, np.ndarray]]:

    shares: list[tuple[np.ndarray, np.ndarray]] = []
    for ct in cts:
        if bridge is not None:
            sh_re, sh_im = bridge.complex_ckks_to_mpc(ct, dtype=np.float64)
            shares.append((sh_re.reconstruct(), sh_im.reconstruct()))
        else:
            sh_re, sh_im = complex_ckks_to_mpc(ctx, ct, comm_stats=comm_stats)
            shares.append((sh_re.reconstruct(), sh_im.reconstruct()))
    return shares


def bridge_encrypt(
    ctx: CKKSContext, shares: list[tuple[np.ndarray, np.ndarray]], *, comm_stats=None, bridge=None
) -> list[Cipher]:

    if bridge is not None:
        return [bridge.complex_mpc_to_ckks(v_re, v_im) for v_re, v_im in shares]
    return [complex_mpc_to_ckks(ctx, v_re, v_im, comm_stats=comm_stats) for v_re, v_im in shares]


def bridge_transplant(old_ctx: CKKSContext, new_ctx: CKKSContext, cts: list[Cipher], *, bridge=None) -> list[Cipher]:

    shares = bridge_decrypt(old_ctx, cts, bridge=bridge)
    return bridge_encrypt(new_ctx, shares, bridge=bridge)


def secure_final_ln(
    ctx: CKKSContext,
    blocks: list[Cipher],
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    m: int | None = None,
    d: int | None = None,
    use_cc: bool = True,
    mpc_engine=None,
    bridge=None,
) -> list[Cipher]:

    _m = m if m is not None else QKV_M
    _d = d if d is not None else QKV_D
    mpc = mpc_engine if mpc_engine is not None else get_mpc_engine(None)
    return roundD(ctx, blocks, m=_m, d_out=_d, gamma=gamma, beta=beta, use_cc=use_cc, mpc_engine=mpc, bridge=bridge)


def secure_linear_head(
    ctx: CKKSContext,
    blocks: list[Cipher],
    W: np.ndarray,
    *,
    b: np.ndarray | None = None,
    m: int | None = None,
    N1: int = 16,
) -> list[Cipher]:

    _m = m if m is not None else QKV_M
    C = ctx.nslots // _m
    d_in, d_out = W.shape

    d_out_padded = ((d_out + C - 1) // C) * C
    if d_out_padded != d_out:
        W_padded = np.zeros((d_in, d_out_padded), dtype=np.float64)
        W_padded[:, :d_out] = W
    else:
        W_padded = W
    out_blocks, _ = run_lin(ctx, X_blocks=blocks, W=W_padded, N1=N1)
    if b is not None:
        b_padded = np.zeros(d_out_padded, dtype=np.float64)
        b_padded[: len(b)] = b
        out_blocks = addb(ctx, out_blocks, b_padded, m=_m)
    return out_blocks


def _free_ctx_gc():

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_segmented(
    *,
    use_cc: bool = True,
    use_scp: bool = True,
    mpc_engine: str | None = None,
    enc_levels: dict[str, int] | None = None,
    bridge=None,
):

    global _SEG_ENC_LEVEL
    import time as _time

    from src.utils_comm import CommStats, estimate_ct_bytes, print_comm_summary

    _timings: dict[str, float] = {}
    np.random.seed(SEED)
    mpc_backend = get_mpc_engine(mpc_engine)
    elvl = enc_levels or {}

    _comm = CommStats()
    _comm.ct_bytes = estimate_ct_bytes(QKV_NSLOTS)
    _comm.mpc_msg_bytes = QKV_M * QKV_D * 8
    mpc_backend.comm_stats = _comm

    print(f"[MPC] engine={mpc_backend.name} device={mpc_backend.device}")
    print(f"[MODE] segmented (per-stage fresh CKKS context + complex conversion)")
    if elvl:
        print(f"[ENC_LEVELS] {elvl}")

    m = QKV_M
    d = QKV_D
    H = QKV_H

    A = np.random.randn(m, QKV_D1).astype(np.float64)
    WQ = np.random.randn(QKV_D1, d).astype(np.float64)
    WK = np.random.randn(QKV_D1, d).astype(np.float64)
    WV = np.random.randn(QKV_D1, d).astype(np.float64)
    bQ = np.random.randn(d).astype(np.float64)
    bK = np.random.randn(d).astype(np.float64)
    bV = np.random.randn(d).astype(np.float64)
    bO = np.random.randn(d).astype(np.float64)
    b1 = np.random.randn(4 * d).astype(np.float64)
    b2 = np.random.randn(d).astype(np.float64)
    ln1_w = np.random.randn(d).astype(np.float64)
    ln1_b = np.random.randn(d).astype(np.float64)
    ln2_w = np.random.randn(d).astype(np.float64)
    ln2_b = np.random.randn(d).astype(np.float64)
    W_O = np.random.randn(d, d).astype(np.float64)
    W1 = np.random.randn(d, 4 * d).astype(np.float64)
    W2 = np.random.randn(4 * d, d).astype(np.float64)

    _t0 = _time.perf_counter()
    _SEG_ENC_LEVEL = elvl.get("qkv")
    ctx = CKKSContext(QKV_NSLOTS)
    Q_cts, K_cts, V_cts, blocks_qk, c_used_qk, blocks_v, c_used_v, _ = run_q(
        ctx, A=A, WQ=WQ, WK=WK, WV=WV, bQ=bQ, bK=bK, bV=bV
    )
    repk = int(os.environ.get("SCP_REPACK_ITERS", "30"))
    if not use_scp:
        Q_cts = repack_id(Q_cts, m=m, iters=repk)
        K_cts = repack_id(K_cts, m=m, iters=repk)
    _timings["QKV"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()
    Q_shares = bridge_decrypt(ctx, Q_cts, comm_stats=_comm, bridge=bridge)
    K_shares = bridge_decrypt(ctx, K_cts, comm_stats=_comm, bridge=bridge)
    V_shares = bridge_decrypt(ctx, V_cts, comm_stats=_comm, bridge=bridge)
    del Q_cts, K_cts, V_cts
    del ctx
    _free_ctx_gc()
    _SEG_ENC_LEVEL = elvl.get("score")
    ctx = CKKSContext(QKV_NSLOTS)
    Q_cts = bridge_encrypt(ctx, Q_shares, comm_stats=_comm, bridge=bridge)
    K_cts = bridge_encrypt(ctx, K_shares, comm_stats=_comm, bridge=bridge)
    del Q_shares, K_shares
    _timings["Conv:QKV->Score"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()
    P = run_s(ctx, Q_cts=Q_cts, K_cts=K_cts, blocks=blocks_qk, c_used=c_used_qk)
    if not use_scp:
        P = repack_id(P, m=m, iters=repk)
    _timings["Score"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()
    d_k = d // H
    A_cts, A_all = roundA(
        ctx, P, H=H, m=m, d_k=d_k, b_fold=B_FOLD, use_cc=use_cc, mpc_engine=mpc_backend, bridge=bridge
    )
    del Q_cts, K_cts, P
    del ctx
    _free_ctx_gc()

    _SEG_ENC_LEVEL = elvl.get("value")
    ctx = CKKSContext(QKV_NSLOTS)
    if bridge is not None:
        V_cts = [bridge.complex_mpc_to_ckks(v_re, v_im) for v_re, v_im in V_shares]
    else:
        V_cts = [complex_mpc_to_ckks(ctx, v_re, v_im) for v_re, v_im in V_shares]
    del V_shares
    if not use_scp:
        V_cts = repack_id(V_cts, m=m, iters=repk)

    C = ctx.nslots // m
    heads_per_ct = C // (d // H)
    groups = H // heads_per_ct
    A_cts = [
        pack_a(ctx, [A_all[g * heads_per_ct + i] for i in range(heads_per_ct)], m=m, heads_per_ct=heads_per_ct)
        for g in range(groups)
    ]
    del A_all
    _timings["MPC-A+Conv"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()

    Y_groups = run_v(ctx, A_cts=A_cts, V_cts=V_cts)
    if not use_scp:
        Y_groups = repack_id(Y_groups, m=m, iters=repk)

    Z_blocks, _ = run_out(ctx, Y_groups=Y_groups, W_O=W_O, N1=16, force_real=False)
    _timings["Value+OUT"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()

    Z_mat, _ = to_mpc_mat_complex_packed(ctx, Z_blocks, m=m, d_out=d, bridge=bridge)
    del V_cts, Y_groups, Z_blocks
    del ctx
    _free_ctx_gc()

    Z_mat = Z_mat + bO[np.newaxis, :d]
    Z_res_mat = Z_mat + A[:, :d]
    try:
        Z_ln_mat = mpc_backend.layer_norm(Z_res_mat, eps=1e-5, gamma=ln1_w, beta=ln1_b, ln_tag="ln1")
    except TypeError:
        Z_ln_mat = mpc_backend.layer_norm(Z_res_mat, eps=1e-5, gamma=ln1_w, beta=ln1_b)

    _SEG_ENC_LEVEL = elvl.get("ff1")
    ctx = CKKSContext(QKV_NSLOTS)
    Z_ln_blocks = to_ckks_mat(ctx, Z_ln_mat, m=m, use_cc=use_cc, bridge=bridge)
    _timings["MPC-B+Conv"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()

    H1_blocks, _ = run_lin(ctx, X_blocks=Z_ln_blocks, W=W1, N1=32, force_real=False)
    _timings["FF1"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()

    H1_mat, _ = to_mpc_mat_complex_packed(ctx, H1_blocks, m=m, d_out=4 * d, bridge=bridge)
    del Z_ln_blocks, H1_blocks
    del ctx
    _free_ctx_gc()
    H1_mat = H1_mat + b1[np.newaxis, : 4 * d]
    H1_gelu_mat = mpc_backend.gelu(H1_mat)

    _SEG_ENC_LEVEL = elvl.get("ff2")
    ctx = CKKSContext(QKV_NSLOTS)
    H1_gelu_blocks = to_ckks_mat(ctx, H1_gelu_mat, m=m, use_cc=use_cc, bridge=bridge)
    _timings["MPC-C+Conv"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()

    H2_blocks, _ = run_lin(ctx, X_blocks=H1_gelu_blocks, W=W2, N1=8, force_real=False)
    _timings["FF2"] = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()

    H2_mat, _ = to_mpc_mat_complex_packed(ctx, H2_blocks, m=m, d_out=d, bridge=bridge)
    del ctx
    _free_ctx_gc()
    H2_mat = H2_mat + b2[np.newaxis, :d]
    H2_res_mat = H2_mat + Z_ln_mat
    try:
        H2_ln_mat = mpc_backend.layer_norm(H2_res_mat, eps=1e-5, gamma=ln2_w, beta=ln2_b, ln_tag="ln2")
    except TypeError:
        H2_ln_mat = mpc_backend.layer_norm(H2_res_mat, eps=1e-5, gamma=ln2_w, beta=ln2_b)
    _timings["MPC-D"] = _time.perf_counter() - _t0

    fhe_keys = ["QKV", "Score", "Value+OUT", "FF1", "FF2"]
    conv_keys = [k for k in _timings if k not in fhe_keys]
    fhe_total = sum(_timings[k] for k in fhe_keys)
    conv_total = sum(_timings[k] for k in conv_keys)
    _timings["FHE_total"] = fhe_total
    _timings["Conv_total"] = conv_total
    _timings["Total"] = fhe_total + conv_total

    print("\n  [Timing — segmented]")
    for k, v in _timings.items():
        print(f"    {k:>20s} = {v:.3f}s")

    print_comm_summary(_comm, fhe_time_s=fhe_total)
    print("\nDone.")


def main(
    *,
    use_cc: bool = True,
    use_scp: bool = True,
    mpc_engine: str | None = None,
    segmented: bool = False,
    enc_levels: dict[str, int] | None = None,
):
    if segmented:
        run_segmented(use_cc=use_cc, use_scp=use_scp, mpc_engine=mpc_engine, enc_levels=enc_levels)
    else:
        run(use_cc=use_cc, use_scp=use_scp, mpc_engine=mpc_engine)


if __name__ == "__main__":
    main()
