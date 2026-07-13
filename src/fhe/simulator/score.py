from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.engines.ckks_engine_plain import Cipher, CKKSContext
from src.utils import print_stage, rel_err, stat_diff, stat_snap

SEED = 42
NSLOTS = 16384
M = 128
B_FOLD = 16
D1 = 768
D = 768
H = 12
CASE_REL_BLOCKS = 7
CASE_REL_C_USED = 120


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


def perm_fdp(H: int, Dh: int) -> np.ndarray:
    return np.array([h * Dh + u for u in range(Dh) for h in range(H)], dtype=np.int64)


def pack_fdp(
    ctx: CKKSContext, X: np.ndarray, *, m: int, H: int, c_used: int | None = None, blocks_override: int | None = None
) -> Tuple[List[Cipher], List[int]]:
    n = ctx.nslots
    assert n % m == 0
    C = n // m
    m0, d = X.shape
    assert m0 == m
    assert d % H == 0
    Dh = d // H
    perm = perm_fdp(H, Dh)
    X_fdp = X[:, perm]
    if c_used is None:
        c_used = C
    if c_used <= 0 or c_used > C:
        raise ValueError(f"c_used must be in [1..C], got {c_used} with C={C}")
    blocks = (d + c_used - 1) // c_used
    if blocks_override is not None:
        if blocks_override < blocks:
            raise ValueError(f"blocks_override={blocks_override} too small; need at least {blocks}")
        blocks = blocks_override
    out: List[Cipher] = []
    used_list: List[int] = []
    for bid in range(blocks):
        v = np.zeros(n, dtype=np.complex128)
        base = bid * c_used
        used = min(c_used, max(0, d - base))
        used_list.append(used)
        for c in range(used):
            col = base + c
            v[c * m : (c + 1) * m] = X_fdp[:, col].astype(np.complex128)
        if used > 0:
            out.append(ctx.encorypt(v))
    return (out, used_list)


_mask_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}


def mk_ht(nslots: int, seglen: int, used_segs: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    t = t % seglen
    key = (nslots, used_segs, (seglen, t))
    if key in _mask_cache:
        return _mask_cache[key]
    head_pat = np.zeros(seglen, dtype=np.complex128)
    tail_pat = np.zeros(seglen, dtype=np.complex128)
    head_pat[: seglen - t] = 1.0
    tail_pat[seglen - t :] = 1.0
    head = np.zeros(nslots, dtype=np.complex128)
    tail = np.zeros(nslots, dtype=np.complex128)
    for c in range(used_segs):
        sl = slice(c * seglen, (c + 1) * seglen)
        head[sl] = head_pat
        tail[sl] = tail_pat
    head.setflags(write=False)
    tail.setflags(write=False)
    _mask_cache[key] = (head, tail)
    return (head, tail)


def rot_within(
    ctx: CKKSContext, Ct: Cipher, *, m: int, used_C: int, t: int, cache: Dict[Tuple[int, int], Cipher]
) -> Cipher:
    n = ctx.nslots
    t %= m
    if t == 0:
        return Ct
    s1 = t % n
    s2 = (t - m) % n
    k1 = (id(Ct), s1)
    k2 = (id(Ct), s2)
    R1 = cache.get(k1)
    if R1 is None:
        R1 = Ct.rot(s1)
        cache[k1] = R1
    R2 = cache.get(k2)
    if R2 is None:
        R2 = Ct.rot(s2)
        cache[k2] = R2
    head, tail = mk_ht(n, m, used_C, t)
    return _add_assign(R1.mul_pt(head), R2.mul_pt(tail))


def rot_w1(ctx: CKKSContext, Ct: Cipher, *, L: int, t: int) -> Cipher:
    n = ctx.nslots
    t %= L
    if t == 0:
        return Ct
    R1 = Ct.rot(t % n)
    R2 = Ct.rot((t - L) % n)
    head, tail = mk_ht(n, L, 1, t)
    return _add_assign(R1.mul_pt(head), R2.mul_pt(tail))


def mapf(m: int, b: int) -> Tuple[Dict[int, Tuple[int, int]], int]:
    if m % b != 0:
        raise ValueError("For folding, require b | m")
    g = m // b
    if g % 2 != 0:
        raise ValueError("For folding, need g even (i.e., b | (m/2))")
    half = m // 2
    mp: Dict[int, Tuple[int, int]] = {}
    for j in range(g):
        for s in range(b):
            t = (j * b - s) % m
            if t < half and t not in mp:
                mp[t] = (j, s)
                if len(mp) == half:
                    return (mp, g)
    raise RuntimeError("Failed to cover all half diagonals")


def mk_bank(
    ctx: CKKSContext,
    Q_cts: List[Cipher],
    Q_used: List[int],
    K_cts: List[Cipher],
    K_used: List[int],
    *,
    m: int,
    b: int,
    g: int,
) -> Tuple[List[List[Cipher]], List[List[Cipher]], List[List[Cipher]]]:
    assert len(Q_cts) == len(K_cts) == len(Q_used) == len(K_used)
    Q_bank: List[List[Cipher]] = []
    K_bank: List[List[Cipher]] = []
    K_bank_h: List[List[Cipher]] = []
    for Qb, used in zip(Q_cts, Q_used):
        cache: Dict[Tuple[int, int], Cipher] = {}
        Q_bank.append([rot_within(ctx, Qb, m=m, used_C=used, t=s, cache=cache) for s in range(b)])
    for Kb, used in zip(K_cts, K_used):
        cache: Dict[Tuple[int, int], Cipher] = {}
        Kg = [rot_within(ctx, Kb, m=m, used_C=used, t=j * b, cache=cache) for j in range(g)]
        Kh = [rot_within(ctx, Kb, m=m, used_C=used, t=(j + g // 2) % g * b, cache=cache) for j in range(g)]
        K_bank.append(Kg)
        K_bank_h.append(Kh)
    return (Q_bank, K_bank, K_bank_h)


def emit_f(
    Q_bank: List[List[Cipher]],
    K_bank: List[List[Cipher]],
    K_bank_h: List[List[Cipher]],
    mp_half: Dict[int, Tuple[int, int]],
    *,
    m: int,
) -> List[List[Tuple[int, int, Cipher]]]:
    half = m // 2
    blocks = len(Q_bank)
    raw: List[List[Tuple[int, int, Cipher]]] = [[] for _ in range(half)]
    for t in range(half):
        j, s = mp_half[t]
        for bid in range(blocks):
            combo = K_bank[bid][j].add(K_bank_h[bid][j].mul_scalar(1j))
            term = Q_bank[bid][s].mul_ct(combo, relin=True)
            raw[t].append((bid, s, term))
    return raw


def msk_first(n: int, m: int, used_cols: int) -> np.ndarray:
    mask = np.zeros(n, dtype=np.complex128)
    mask[: used_cols * m] = 1.0
    return mask


def msk_rng(n: int, m: int, seg_lo: int, seg_hi: int) -> np.ndarray:
    mask = np.zeros(n, dtype=np.complex128)
    mask[seg_lo * m : seg_hi * m] = 1.0
    return mask


def red_h(ctx: CKKSContext, term: Cipher, *, used_cols: int, H: int, m: int) -> Cipher:
    n = ctx.nslots
    C = n // m
    Z = term
    if used_cols < C:
        Z = Z.mul_pt(msk_first(n, m, used_cols))
    U = (used_cols + H - 1) // H
    delta = 1
    while delta < U:
        src_seg_lo = delta * H
        if src_seg_lo >= used_cols:
            break
        src_mask = msk_rng(n, m, src_seg_lo, used_cols)
        Z = _add_assign(Z, Z.mul_pt(src_mask).rot(delta * H * m % n))
        delta <<= 1
    Z = Z.mul_pt(msk_rng(n, m, 0, H))
    return Z


def aln_h(ctx: CKKSContext, Z: Cipher, *, H: int, m: int, r: int) -> Cipher:
    if r == 0:
        return Z
    k_align = (H - r) % H * m
    if k_align == 0:
        return Z
    Z = rot_w1(ctx, Z, L=H * m, t=k_align)
    n = ctx.nslots
    Z = Z.mul_pt(msk_rng(n, m, 0, H))
    return Z


def pack_f(ctx: CKKSContext, D_fold: List[Cipher], *, H: int, m: int) -> List[Cipher]:
    n = ctx.nslots
    blen = H * m
    total_vals = len(D_fold) * blen
    out_cts = math.ceil(total_vals / n)
    P = [ctx.encorypt(np.zeros(n, dtype=np.complex128)) for _ in range(out_cts)]
    for t, Dt in enumerate(D_fold):
        L = t * blen
        k = L // n
        off = L % n
        if off + blen <= n:
            rot = -off % n
            P[k] = _add_assign(P[k], Dt.rot(rot))
        else:
            L1 = n - off
            mask1 = np.zeros(n, dtype=np.complex128)
            mask1[:L1] = 1.0
            mask2 = np.zeros(n, dtype=np.complex128)
            mask2[L1:blen] = 1.0
            rot = -off % n
            P[k] = _add_assign(P[k], Dt.mul_pt(mask1).rot(rot))
            P[k + 1] = _add_assign(P[k + 1], Dt.mul_pt(mask2).rot(rot))
    return P


def unpack_fold(
    ctx: CKKSContext, D_fold: List[Cipher], *, H: int, m: int, mp_half: Dict[int, Tuple[int, int]]
) -> np.ndarray:
    half = m // 2
    rows = np.arange(m)
    S = np.zeros((H, m, m), dtype=np.float64)
    with ctx.decorypt_scope():
        for t in range(half):
            vec = D_fold[t].decorypt(copy=True, readonly=True)
            _, s = mp_half[t]
            for h in range(H):
                seg = vec[h * m : (h + 1) * m]
                seg = np.roll(seg, +s)
                S[h, rows, (rows + t) % m] += seg.real
                S[h, rows, (rows + half + t) % m] += seg.imag
    return S


def unpack_f(ctx: CKKSContext, P: List[Cipher], *, H: int, m: int, mp_half: Dict[int, Tuple[int, int]]) -> np.ndarray:
    n = ctx.nslots
    half = m // 2
    blen = H * m
    rows = np.arange(m)
    S = np.zeros((H, m, m), dtype=np.float64)
    with ctx.decorypt_scope():
        P_dec = [ct.decorypt(copy=True, readonly=True) for ct in P]
    for t in range(half):
        start = t * blen
        vec = np.empty(blen, dtype=np.complex128)
        for i in range(blen):
            lin = start + i
            k = lin // n
            off = lin % n
            vec[i] = P_dec[k][off]
        _, s = mp_half[t]
        for h in range(H):
            seg = vec[h * m : (h + 1) * m]
            seg = np.roll(seg, +s)
            S[h, rows, (rows + t) % m] += seg.real
            S[h, rows, (rows + half + t) % m] += seg.imag
    return S


def score_ref(Q: np.ndarray, K: np.ndarray, H: int) -> np.ndarray:
    m, d = Q.shape
    Dh = d // H
    out = np.zeros((H, m, m), dtype=np.float64)
    for h in range(H):
        out[h] = Q[:, h * Dh : (h + 1) * Dh] @ K[:, h * Dh : (h + 1) * Dh].T
    return out


def run() -> None:
    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    m = M
    A = np.random.randn(m, D1).astype(np.float64)
    WQ = np.random.randn(D1, D).astype(np.float64)
    WK = np.random.randn(D1, D).astype(np.float64)
    Q = A @ WQ
    K = A @ WK
    S_ref = score_ref(Q, K, H)
    blocks = CASE_REL_BLOCKS
    c_used = CASE_REL_C_USED
    Q_cts, Q_used = pack_fdp(ctx, Q, m=m, H=H, c_used=c_used, blocks_override=blocks)
    K_cts, K_used = pack_fdp(ctx, K, m=m, H=H, c_used=c_used, blocks_override=blocks)
    mp_half, g = mapf(m, B_FOLD)
    s0 = stat_snap(ctx)
    Q_bank, K_bank, K_bank_h = mk_bank(ctx, Q_cts, Q_used, K_cts, K_used, m=m, b=B_FOLD, g=g)
    raw_fold = emit_f(Q_bank, K_bank, K_bank_h, mp_half, m=m)
    if c_used % H == 0:
        r_per_bid = [0 for _ in range(blocks)]
    else:
        r_per_bid = [bid * c_used % H for bid in range(blocks)]
    half = m // 2
    D_fold: List[Cipher] = []
    for t in range(half):
        term_by_r: Dict[int, Cipher] = {}
        for bid, _, term in raw_fold[t]:
            r = r_per_bid[bid]
            term_by_r[r] = term if r not in term_by_r else _add_assign(term_by_r[r], term)
        acc: Cipher | None = None
        for r, term_sum in term_by_r.items():
            red = red_h(ctx, term_sum, used_cols=c_used, H=H, m=m)
            if r != 0:
                red = aln_h(ctx, red, H=H, m=m, r=r)
            acc = _add_assign(acc, red)
        assert acc is not None
        D_fold.append(acc)
    P = pack_f(ctx, D_fold, H=H, m=m)
    s1 = stat_snap(ctx)
    d01 = stat_diff(s0, s1)
    S_out = unpack_f(ctx, P, H=H, m=m, mp_half=mp_half)
    err_out = rel_err(S_out, S_ref)
    max_abs = float(np.max(np.abs(S_out - S_ref)))
    print_stage(
        "Score",
        ct_in=None,
        ct_out=len(P),
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=err_out,
    )
    print(f"  max_abs_err = {max_abs:.6e}")
    assert err_out < 1e-09, f"FAIL: relerr={err_out}"


def main() -> None:
    run()


if __name__ == "__main__":
    main()
