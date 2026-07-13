from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.utils import ct_real, print_stage, rel_err, stat_diff, stat_snap

SEED = 42
M = 128
NSLOTS = 16384
NUM_HEADS = 12
D_H = 64
D2 = NUM_HEADS * D_H
OUT_N1 = 16


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


def _pre_tbl_pair(w: np.ndarray, *, c: int, blocks: int, g: int) -> np.ndarray:
    tab = np.zeros((g, c, blocks, c), dtype=np.complex128)
    j = np.arange(c, dtype=np.int32)
    for ge in range(g):
        off = ge * c
        for t in range(c):
            rows = off + (j + t) % c
            for b in range(blocks):
                cols = b * c + j
                tab[ge, t, b, :] = w[rows, cols]
    return tab


def smx(X: np.ndarray) -> np.ndarray:
    X = X - X.max(axis=1, keepdims=True)
    Y = np.exp(X)
    return Y / (Y.sum(axis=1, keepdims=True) + 1e-12)


def mk_data(seed: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    A_logits_all = [np.random.randn(M, M).astype(np.float64) for _ in range(NUM_HEADS)]
    A_all = [smx(L) for L in A_logits_all]
    V_all = [np.random.randn(M, D_H).astype(np.float64) for _ in range(NUM_HEADS)]
    W_O = np.random.randn(D2, D2).astype(np.float64)
    Y_truth_all = [A_all[h] @ V_all[h] for h in range(NUM_HEADS)]
    Y_truth = np.concatenate(Y_truth_all, axis=1)
    Z_truth = Y_truth @ W_O
    return (A_all, V_all, W_O, Y_truth, Z_truth)


def pack_v(ctx: CKKSContext, V_heads: List[np.ndarray], *, m: int, d_h: int, heads_per_ct: int) -> Cipher:
    assert len(V_heads) == heads_per_ct
    n = ctx.nslots
    C = n // m
    assert heads_per_ct * d_h <= C
    v = np.zeros(n, dtype=np.complex128)
    for h in range(heads_per_ct):
        V = V_heads[h]
        for c in range(d_h):
            seg = h * d_h + c
            v[seg * m : (seg + 1) * m] = V[:, c].astype(np.complex128)
    v.setflags(write=False)
    return ctx.encorypt(v)


def pack_a(ctx: CKKSContext, A_heads: List[np.ndarray], *, m: int, heads_per_ct: int) -> Cipher:
    assert len(A_heads) == heads_per_ct
    n = ctx.nslots
    C = n // m
    half = m // 2
    assert half > 0
    assert heads_per_ct * half <= C
    v = np.zeros(n, dtype=np.complex128)
    for h in range(heads_per_ct):
        A = A_heads[h]
        base_seg = h * half
        for t in range(half):
            tp = t + half
            d0 = np.empty(m, dtype=np.complex128)
            d1 = np.empty(m, dtype=np.complex128)
            for r in range(m):
                d0[r] = A[r, (r - t) % m]
                d1[r] = A[r, (r - tp) % m]
            seg = base_seg + t
            v[seg * m : (seg + 1) * m] = d0 + 1j * d1
    v.setflags(write=False)
    return ctx.encorypt(v)


def _make_masks_arrays(nslots: int, m: int, used_C: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    t %= m
    head_pat = np.zeros(m, dtype=np.complex128)
    tail_pat = np.zeros(m, dtype=np.complex128)
    if t == 0:
        head_pat[:] = 1.0
    else:
        head_pat[: m - t] = 1.0
        tail_pat[m - t :] = 1.0
    head = np.zeros(nslots, dtype=np.complex128)
    tail = np.zeros(nslots, dtype=np.complex128)
    for c in range(used_C):
        sl = slice(c * m, (c + 1) * m)
        head[sl] = head_pat
        tail[sl] = tail_pat
    head.setflags(write=False)
    tail.setflags(write=False)
    return (head, tail)


def rot_within(
    ctx: CKKSContext,
    Ct: Cipher,
    *,
    m: int,
    used_C: int,
    t: int,
    mask_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
    rot_cache: Dict[Tuple[int, int], Cipher],
    rot_break: Dict[str, int],
    tag: str,
) -> Cipher:
    n = ctx.nslots
    t %= m
    if t == 0:
        return Ct
    s1, s2 = (t % n, (t - m) % n)
    k1, k2 = ((id(Ct), s1), (id(Ct), s2))
    R1 = rot_cache.get(k1)
    if R1 is None:
        R1 = Ct.rot(s1)
        rot_cache[k1] = R1
        rot_break[tag] += 1
    R2 = rot_cache.get(k2)
    if R2 is None:
        R2 = Ct.rot(s2)
        rot_cache[k2] = R2
        rot_break[tag] += 1
    mt = mask_cache.get(t)
    if mt is None:
        mt = _make_masks_arrays(n, m, used_C, t)
        mask_cache[t] = mt
    head, tail = mt
    return _add_assign(R1.mul_pt(head), R2.mul_pt(tail))


def mk_col_masks(*, nslots: int, m: int, heads_per_ct: int, d_h: int) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for s in range(d_h):
        mask = np.zeros(nslots, dtype=np.complex128)
        for h in range(heads_per_ct):
            seg = h * d_h + s
            mask[seg * m : (seg + 1) * m] = 1.0
        masks.append(mask)
    return masks


def mk_a_bank(ctx: CKKSContext, A_packed: Cipher, *, m: int, d_h: int, rot_break: Dict[str, int]) -> Dict[int, Cipher]:
    n = ctx.nslots
    half = m // 2
    assert half > 0 and d_h == 64
    bank: Dict[int, Cipher] = {0: A_packed}
    for d in range(-63, 64):
        if d == 0:
            continue
        shift = d * m % n
        bank[d] = A_packed.rot(shift) if shift else A_packed
        rot_break["A_pre_rot"] += 1
    return bank


def ex_a_db(
    ctx: CKKSContext,
    A_bank: Dict[int, Cipher],
    col_masks: List[np.ndarray],
    *,
    m: int,
    d_h: int,
    t: int,
    rot_break: Dict[str, int],
) -> Cipher:
    half = m // 2
    assert half > 0 and d_h == 64
    t %= half
    acc = None
    for s in range(d_h):
        d = t - s
        ct = A_bank[d]
        acc = _add_assign(acc, ct.mul_pt(col_masks[s]))
        rot_break["A_maskadd"] += 1
    return acc if acc is not None else ctx.zeros()


def sv(
    ctx: CKKSContext, *, A_packed: Cipher, V_ct: Cipher, m: int, d_h: int, heads_per_ct: int
) -> Tuple[Cipher, Dict[str, int]]:
    n = ctx.nslots
    C = n // m
    half = m // 2
    assert half > 0 and d_h == 64
    assert heads_per_ct * d_h == C, "V must be dense"

    rot_break = defaultdict(int)
    col_masks = mk_col_masks(nslots=n, m=m, heads_per_ct=heads_per_ct, d_h=d_h)
    A_bank = mk_a_bank(ctx, A_packed, m=m, d_h=d_h, rot_break=rot_break)
    mask_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    rot_cache: Dict[Tuple[int, int], Cipher] = {}
    V_half = rot_within(
        ctx,
        V_ct,
        m=m,
        used_C=C,
        t=-half % m,
        mask_cache=mask_cache,
        rot_cache=rot_cache,
        rot_break=rot_break,
        tag="V_halfshift",
    )
    U = _add_assign(V_ct, V_half.mul_scalar(-1j))
    out: Optional[Cipher] = None
    U_mask_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    U_rot_cache: Dict[Tuple[int, int], Cipher] = {}
    for t in range(half):
        A_t = ex_a_db(ctx, A_bank, col_masks, m=m, d_h=d_h, t=t, rot_break=rot_break)
        if t == 0:
            U_t = U
        else:
            U_t = rot_within(
                ctx,
                U,
                m=m,
                used_C=C,
                t=-t % m,
                mask_cache=U_mask_cache,
                rot_cache=U_rot_cache,
                rot_break=rot_break,
                tag="U_shift",
            )
        term = U_t.mul_ct(A_t, relin=True)
        out = _add_assign(out, term)
    assert out is not None
    return (out, dict(rot_break))


def dec_grp(ctx: CKKSContext, Y_ct: Cipher, *, m: int) -> np.ndarray:
    n = ctx.nslots
    C = n // m
    with ctx.decorypt_scope():
        v = Y_ct.decorypt(copy=True, readonly=True).real
    return v[: C * m].reshape(C, m).T.copy()


def pack_out_sv(ctx: CKKSContext, Y_groups: List[Cipher], *, force_real: bool = True) -> List[Cipher]:
    G = len(Y_groups)
    assert G == 6, "Expected 6 SV group ciphertexts (2 heads per CT)."
    Y_clean: List[Cipher] = [ct_real(ctx, ct) for ct in Y_groups] if force_real else list(Y_groups)
    Hp = (G + 1) // 2
    out: List[Cipher] = []
    for h in range(Hp):
        ge = 2 * h
        go = ge + 1
        ct = Y_clean[ge]
        if go < G:
            ct = _add_assign(ct, Y_clean[go].mul_scalar(1j))
        out.append(ct)
    return out


def build_babies_min(ctx: CKKSContext, base_cts: List[Cipher], *, m: int, N1: int) -> List[List[Cipher]]:
    n = ctx.nslots
    babies: List[List[Cipher]] = []
    for base in base_cts:
        row: List[Cipher] = []
        for q in range(N1):
            shift = q * m % n
            row.append(base.rot(shift) if shift else base)
        babies.append(row)
    return babies


def linear_babies_pairs(
    ctx: CKKSContext, Gpairs: List[List[Cipher]], W: np.ndarray, *, m: int, N1: int, G: int
) -> List[List[Cipher]]:
    n = ctx.nslots
    C = n // m
    Wc = np.asarray(W, dtype=np.complex128)
    d_in, d_out = Wc.shape
    assert d_in == G * C
    assert d_out % C == 0
    assert C % N1 == 0
    H_pairs = (G + 1) // 2
    assert len(Gpairs) == H_pairs
    N2 = C // N1
    blocks = d_out // C
    w_tab = _pre_tbl_pair(Wc, c=C, blocks=blocks, g=G)
    wr_c = np.empty(C, dtype=np.complex128)
    wi_c = np.empty(C, dtype=np.complex128)
    pt_c = np.empty(C, dtype=np.complex128)
    pt_buf = np.empty(n, dtype=np.complex128)
    pt_mat = pt_buf.reshape(C, m)
    p_shifts = [(p * N1) % C for p in range(N2)]
    Cf: List[List[Cipher | None]] = [[None for _ in range(N2)] for _ in range(blocks)]
    for b in range(blocks):
        for p in range(N2):
            p_shift = p_shifts[p]
            S = None
            for q in range(N1):
                t_local = (p_shift + q) % C
                Acc = None
                for h in range(H_pairs):
                    ge = 2 * h
                    go = ge + 1
                    wr_src = w_tab[ge, t_local, b, :]
                    if p_shift:
                        _roll_into(wr_src, p_shift, wr_c)
                    else:
                        wr_c[:] = wr_src
                    if go < G:
                        wi_src = w_tab[go, t_local, b, :]
                        if p_shift:
                            _roll_into(wi_src, p_shift, wi_c)
                        else:
                            wi_c[:] = wi_src
                        pt_c[:] = wr_c
                        pt_c += -1j * wi_c
                    else:
                        pt_c[:] = wr_c
                    pt_mat[:] = pt_c[:, None]
                    Acc = _add_assign(Acc, Gpairs[h][q].mul_pt(pt_buf))
                S = _add_assign(S, Acc)
            Cf[b][p] = S if S is not None else ctx.zeros()
    return Cf


def fold_grid(ctx: CKKSContext, Cf_lazy: List[List[Cipher]], *, m: int, N1: int) -> List[Cipher]:
    n = ctx.nslots
    C = n // m
    assert C % N1 == 0
    N2 = C // N1
    folded: List[Cipher] = []
    for b in range(len(Cf_lazy)):
        acc = None
        for p in range(N2):
            ct = Cf_lazy[b][p]
            shift = p * N1 * m % n
            if shift:
                ct = ct.rot(shift)
            acc = _add_assign(acc, ct)
        folded.append(acc if acc is not None else ctx.zeros())
    return folded


def decrypt_blocks(ctx: CKKSContext, folded_blocks: Sequence[Cipher], *, m: int, d_out: int) -> np.ndarray:
    n = ctx.nslots
    C = n // m
    out = np.zeros((m, d_out), dtype=np.float64)
    with ctx.decorypt_scope():
        for b in range(len(folded_blocks)):
            used = min(C, max(0, d_out - b * C))
            if used <= 0:
                break
            v = folded_blocks[b].decorypt(copy=True, readonly=True).real
            out[:, b * C : b * C + used] = v[: used * m].reshape(used, m).T
    return out


def run() -> None:
    A_all, V_all, W_O, Y_truth, _ = mk_data(SEED)
    ctx = CKKSContext(NSLOTS)
    C = ctx.nslots // M
    heads_per_ct = C // D_H
    groups = NUM_HEADS // heads_per_ct
    s_sv0 = stat_snap(ctx)
    Y_groups: List[Cipher] = []
    for g in range(groups):
        ids = [g * heads_per_ct + i for i in range(heads_per_ct)]
        A_packed = pack_a(ctx, [A_all[h] for h in ids], m=M, heads_per_ct=heads_per_ct)
        V_ct = pack_v(ctx, [V_all[h] for h in ids], m=M, d_h=D_H, heads_per_ct=heads_per_ct)
        Yg, _ = sv(ctx, A_packed=A_packed, V_ct=V_ct, m=M, d_h=D_H, heads_per_ct=heads_per_ct)
        Y_groups.append(Yg)
    s_sv1 = stat_snap(ctx)
    d01 = stat_diff(s_sv0, s_sv1)
    Y_dec = np.zeros((M, groups * (heads_per_ct * D_H)), dtype=np.float64)
    for g, ct in enumerate(Y_groups):
        Y_blk = dec_grp(ctx, ct, m=M)
        Cseg = Y_blk.shape[1]
        Y_dec[:, g * Cseg : (g + 1) * Cseg] = Y_blk
    rel_err_main = rel_err(Y_dec, Y_truth)
    max_abs_sv = float(np.max(np.abs(Y_dec - Y_truth)))
    ct_in = groups
    print_stage(
        "Value",
        ct_in=ct_in,
        ct_out=None,
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=rel_err_main,
    )
    print(f"  max_abs_err = {max_abs_sv:.6e}")
    assert rel_err_main < 1e-09, f"FAIL: Value rel_err={rel_err_main}"
    s0 = stat_snap(ctx)
    base_cts = pack_out_sv(ctx, Y_groups)
    Y_plain = Y_truth
    babies = build_babies_min(ctx, base_cts, m=M, N1=OUT_N1)
    Cf_lazy = linear_babies_pairs(ctx, babies, W_O, m=M, N1=OUT_N1, G=groups)
    folded = fold_grid(ctx, Cf_lazy, m=M, N1=OUT_N1)
    folded_real = [ct_real(ctx, ct) for ct in folded]
    Z_dec = decrypt_blocks(ctx, folded_real, m=M, d_out=D2)
    Z_truth_use = Y_plain @ W_O
    err = rel_err(Z_dec, Z_truth_use)
    max_abs_out = float(np.max(np.abs(Z_dec - Z_truth_use)))
    s1 = stat_snap(ctx)
    d01 = stat_diff(s0, s1)
    ct_out_report = (len(folded_real) + 1) // 2
    print_stage(
        "OUT",
        ct_in=None,
        ct_out=ct_out_report,
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=err,
    )
    print(f"  max_abs_err = {max_abs_out:.6e}")
    assert err < 1e-09, f"FAIL: OUT rel_err={err}"


def main() -> None:
    run()


if __name__ == "__main__":
    main()
