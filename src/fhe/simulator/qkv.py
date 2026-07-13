from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.models.model_config import BERT_BASE
from src.utils import ct_real, print_stage, rel_err, stat_diff, stat_snap

SEED = 42


_cfg = BERT_BASE
NSLOTS = _cfg.nslots
M = _cfg.m
N1 = _cfg.n1
C = NSLOTS // M
N2 = C // N1
D2 = _cfg.d_model
H = _cfg.H
DH = _cfg.d_h


D1 = 6 * C


QK_C_USED = 128
QK_BLOCKS = 6
V_C_USED = 128
V_BLOCKS = 6


def _add_assign(acc, term):
    if acc is None:
        return term
    add_inplace = getattr(acc, "add_inplace", None)
    if callable(add_inplace):
        try:
            add_inplace(term)
            return acc
        except Exception:
            pass
    return acc.add(term)


def _roll_into(src: np.ndarray, shift: int, dst: np.ndarray) -> None:
    n = src.size
    s = int(shift) % n
    if s == 0:
        dst[:] = src
        return
    dst[:s] = src[-s:]
    dst[s:] = src[:-s]


def pa6(A: np.ndarray, m: int, n_slots: int) -> List[np.ndarray]:
    m_rows, d1 = A.shape
    assert m_rows == m
    C = n_slots // m
    assert d1 == 6 * C
    vs = [np.zeros(n_slots, dtype=np.complex128) for _ in range(6)]
    for local_c in range(C):
        sl = slice(local_c * m, (local_c + 1) * m)
        for i in range(6):
            vs[i][sl] = A[:, i * C + local_c].astype(np.complex128)
    return vs


def pre_tbl_c(
    WQ: np.ndarray, WK: np.ndarray, WV: np.ndarray, C: int, d2: int, *, c_used: int
) -> Tuple[np.ndarray, np.ndarray]:
    d1, d2_check = WQ.shape
    assert d2_check == d2 and WK.shape == (d1, d2) and (WV.shape == (d1, d2))
    assert d1 == 6 * C
    blocks = (d2 + c_used - 1) // c_used
    QK_tab = np.zeros((6, C, blocks, C), dtype=np.complex128)
    V_tab = np.zeros((6, C, blocks, C), dtype=np.float64)
    j = np.arange(C, dtype=np.int32)
    for i in range(6):
        iC = i * C
        for t in range(C):
            rows = iC + (j + t) % C
            for b in range(blocks):
                cols = b * c_used + j
                valid = (j < c_used) & (cols < d2)
                QK_tab[i, t, b, :] = 0
                V_tab[i, t, b, :] = 0
                if np.any(valid):
                    QK_tab[i, t, b, valid] = WQ[rows[valid], cols[valid]] + 1j * WK[rows[valid], cols[valid]]
                    V_tab[i, t, b, valid] = WV[rows[valid], cols[valid]]
    return (QK_tab, V_tab)


def fold_grid(C_grid: Sequence[Sequence[Cipher]], ctx: CKKSContext) -> List[Cipher]:
    out: List[Cipher] = []
    for b in range(len(C_grid)):
        acc = None
        for ct in C_grid[b]:
            acc = _add_assign(acc, ct)
        out.append(acc if acc is not None else ctx.zeros())
    return out


def split_qk(ctx: CKKSContext, QK_blocks: Sequence[Cipher]) -> Tuple[List[Cipher], List[Cipher]]:
    Q_out: List[Cipher] = []
    K_out: List[Cipher] = []
    for ct in QK_blocks:
        ctc = ctx.conjugate(ct)
        Q_ct = ct.add(ctc).mul_scalar(0.5)
        diff = ct.add(ctc.mul_scalar(-1.0))
        K_ct = diff.mul_scalar(-0.5j)
        Q_out.append(Q_ct)
        K_out.append(K_ct)
    return (Q_out, K_out)


def decrypt_blocks(blocks_ct: Sequence[Cipher], m: int, d2: int, ctx: CKKSContext, *, c_used: int) -> np.ndarray:
    out = np.zeros((m, d2), dtype=np.float64)
    with ctx.decorypt_scope():
        for b, S_b in enumerate(blocks_ct):
            used = min(c_used, max(0, d2 - b * c_used))
            if used <= 0:
                break
            v = S_b.decorypt(copy=True, readonly=True).real
            out[:, b * c_used : b * c_used + used] = v[: used * m].reshape(used, m).T
    return out


def perm_fdp(H: int, Dh: int) -> np.ndarray:
    return np.array([h * Dh + u for u in range(Dh) for h in range(H)], dtype=np.int64)


def perm_hd(H: int, Dh: int) -> np.ndarray:
    return np.array([h * Dh + u for h in range(H) for u in range(Dh)], dtype=np.int64)


def proj_mix(
    ctx: CKKSContext,
    *,
    ct_pairs_babies: List[List[Cipher]],
    QK_tab: np.ndarray,
    V_tab: np.ndarray,
    m: int,
    N1: int,
    d2: int,
    qk_c_used: int,
    qk_blocks: int,
    v_c_used: int,
    v_blocks: int,
) -> Tuple[List[List[Cipher]], List[List[Cipher]]]:
    n_slots = ctx.nslots
    C = n_slots // m
    assert C % N1 == 0
    N2 = C // N1
    qk_used_per_block = [min(qk_c_used, max(0, d2 - b * qk_c_used)) for b in range(qk_blocks)]
    v_used_per_block = [min(v_c_used, max(0, d2 - b * v_c_used)) for b in range(v_blocks)]
    ct_pairs_conj_babies = [[ctx.conjugate(ct) for ct in babies] for babies in ct_pairs_babies]
    QK_grid: List[List[Cipher | None]] = [[None for _ in range(N2)] for _ in range(qk_blocks)]
    V_grid: List[List[Cipher | None]] = [[None for _ in range(N2)] for _ in range(v_blocks)]
    base_buf = np.empty(n_slots, dtype=np.complex128)
    aux_buf = np.empty(n_slots, dtype=np.complex128)
    rot_base_buf = np.empty(n_slots, dtype=np.complex128)
    rot_aux_buf = np.empty(n_slots, dtype=np.complex128)
    wm_buf = np.empty(n_slots, dtype=np.complex128)
    wp_buf = np.empty(n_slots, dtype=np.complex128)
    base_mat = base_buf.reshape(C, m)
    aux_mat = aux_buf.reshape(C, m)
    for b in range(qk_blocks):
        used = qk_used_per_block[b]
        if used == 0:
            continue
        for p in range(N2):
            gshift = p * N1 * m % n_slots
            tmp_qk = None
            base = p * N1
            for q in range(N1):
                t = base + q
                term_qk = None
                for pair_idx in range(3):
                    g_even = 2 * pair_idx
                    g_odd = 2 * pair_idx + 1
                    base_mat[:] = QK_tab[g_even, t, b, :][:, None]
                    aux_mat[:] = QK_tab[g_odd, t, b, :][:, None]
                    base_buf[qk_c_used * m : C * m] = 0
                    aux_buf[qk_c_used * m : C * m] = 0
                    if used < qk_c_used:
                        base_buf[used * m : qk_c_used * m] = 0
                        aux_buf[used * m : qk_c_used * m] = 0
                    if gshift:
                        _roll_into(base_buf, +gshift, rot_base_buf)
                        _roll_into(aux_buf, +gshift, rot_aux_buf)
                    else:
                        rot_base_buf[:] = base_buf
                        rot_aux_buf[:] = aux_buf
                    wm_buf[:] = rot_base_buf
                    wm_buf += -1j * rot_aux_buf
                    wp_buf[:] = rot_base_buf
                    wp_buf += 1j * rot_aux_buf
                    ct_b = ct_pairs_babies[pair_idx][q]
                    ct_c = ct_pairs_conj_babies[pair_idx][q]
                    term_pair = _add_assign(ct_b.mul_pt(wm_buf), ct_c.mul_pt(wp_buf)).mul_scalar(0.5)
                    term_qk = _add_assign(term_qk, term_pair)
                tmp_qk = _add_assign(tmp_qk, term_qk)
            if tmp_qk is None:
                tmp_qk = ctx.zeros()
            QK_grid[b][p] = tmp_qk.rot(gshift) if gshift else tmp_qk
    for b in range(v_blocks):
        used = v_used_per_block[b]
        if used == 0:
            continue
        for p in range(N2):
            gshift = p * N1 * m % n_slots
            tmp_v = None
            base = p * N1
            for q in range(N1):
                t = base + q
                term_v = None
                for pair_idx in range(3):
                    g_even = 2 * pair_idx
                    g_odd = 2 * pair_idx + 1
                    base_mat[:] = V_tab[g_even, t, b, :][:, None]
                    aux_mat[:] = V_tab[g_odd, t, b, :][:, None]
                    base_buf[: used * m] += -1j * aux_buf[: used * m]
                    base_buf[v_c_used * m : C * m] = 0
                    if used < v_c_used:
                        base_buf[used * m : v_c_used * m] = 0
                    if gshift:
                        _roll_into(base_buf, +gshift, rot_base_buf)
                    else:
                        rot_base_buf[:] = base_buf
                    ct_b = ct_pairs_babies[pair_idx][q]
                    term_v = _add_assign(term_v, ct_b.mul_pt(rot_base_buf))
                tmp_v = _add_assign(tmp_v, term_v)
            if tmp_v is None:
                tmp_v = ctx.zeros()
            V_grid[b][p] = tmp_v.rot(gshift) if gshift else tmp_v
    out_qk: List[List[Cipher]] = [[cell if cell is not None else ctx.zeros() for cell in row] for row in QK_grid]
    out_v: List[List[Cipher]] = [[cell if cell is not None else ctx.zeros() for cell in row] for row in V_grid]
    return (out_qk, out_v)


def run() -> None:
    np.random.seed(SEED)
    C = NSLOTS // M
    d1 = 6 * C
    d2 = D2
    assert D2 == H * DH
    A = np.random.randn(M, d1).astype(np.float64)
    WQ = np.random.randn(d1, d2).astype(np.float64)
    WK = np.random.randn(d1, d2).astype(np.float64)
    WV = np.random.randn(d1, d2).astype(np.float64)
    perm_fdp_cols = perm_fdp(H, DH)
    perm_hm_cols = perm_hd(H, DH)
    WQ_fdp = WQ[:, perm_fdp_cols]
    WK_fdp = WK[:, perm_fdp_cols]
    WV_hm = WV[:, perm_hm_cols]
    ctx = CKKSContext(NSLOTS)
    n_slots = ctx.nslots
    qk_c_used = QK_C_USED
    qk_blocks = QK_BLOCKS
    v_c_used = V_C_USED
    v_blocks = V_BLOCKS
    s0 = stat_snap(ctx)
    vs = pa6(A, M, n_slots)
    vp = [vs[0] + 1j * vs[1], vs[2] + 1j * vs[3], vs[4] + 1j * vs[5]]
    A_pairs = [ctx.encorypt(v) for v in vp]
    ct_pairs_babies = []
    for Ap in A_pairs:
        row = []
        for q in range(N1):
            shift = q * M % n_slots
            row.append(Ap.rot(shift) if shift else Ap)
        ct_pairs_babies.append(row)
    QK_tab, _ = pre_tbl_c(WQ_fdp, WK_fdp, WV_hm, C, d2, c_used=qk_c_used)
    _QK2, V_tab = pre_tbl_c(WQ_fdp, WK_fdp, WV_hm, C, d2, c_used=v_c_used)
    QK_grid, V_grid = proj_mix(
        ctx,
        ct_pairs_babies=ct_pairs_babies,
        QK_tab=QK_tab,
        V_tab=V_tab,
        d2=d2,
        qk_c_used=qk_c_used,
        qk_blocks=qk_blocks,
        v_c_used=v_c_used,
        v_blocks=v_blocks,
        m=M,
        N1=N1,
    )
    QK_blocks = fold_grid(QK_grid, ctx)[:qk_blocks]
    V_raw_blocks = fold_grid(V_grid, ctx)[:v_blocks]
    Q_blocks, K_blocks = split_qk(ctx, QK_blocks)
    Q_blocks = [ct_real(ctx, ct) for ct in Q_blocks]
    K_blocks = [ct_real(ctx, ct) for ct in K_blocks]
    V_blocks = [ct_real(ctx, ct) for ct in V_raw_blocks]
    s1 = stat_snap(ctx)
    d01 = stat_diff(s0, s1)
    Qb = decrypt_blocks(Q_blocks, M, d2, ctx, c_used=qk_c_used)
    Kb = decrypt_blocks(K_blocks, M, d2, ctx, c_used=qk_c_used)
    Vb = decrypt_blocks(V_blocks, M, d2, ctx, c_used=v_c_used)
    Q_ref = A @ WQ_fdp
    K_ref = A @ WK_fdp
    V_ref = A @ WV_hm
    rel_err_avg = (rel_err(Qb, Q_ref) + rel_err(Kb, K_ref) + rel_err(Vb, V_ref)) / 3.0
    max_abs = max(
        float(np.max(np.abs(Qb - Q_ref))),
        float(np.max(np.abs(Kb - K_ref))),
        float(np.max(np.abs(Vb - V_ref))),
    )
    print_stage(
        "QKV",
        ct_in=3,
        ct_out=None,
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=rel_err_avg,
    )
    print(f"  max_abs_err = {max_abs:.6e}")
    assert rel_err_avg < 1e-09, f"FAIL: QKV rel_err={rel_err_avg}"


def main() -> None:
    run()


if __name__ == "__main__":
    main()
