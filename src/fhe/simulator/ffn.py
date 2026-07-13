from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.engines.ckks_engine_plain import CKKSContext
from src.utils import ct_real, print_stage, rel_err, stat_diff, stat_snap

SEED = 42
NSLOTS = 16384
M = 128
N1 = 16
D1 = 768
DMID = 3072
D2 = 768
B1 = None
B2 = None


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


def _sum_terms(ctx: CKKSContext, terms: List[Any]):
    if not terms:
        return None
    add_many = getattr(ctx, "add_many", None)
    if callable(add_many):
        try:
            return add_many(terms)
        except Exception:
            pass
    acc = terms[0]
    for t in terms[1:]:
        acc = _add_assign(acc, t)
    return acc


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


def _pre_tbl_real(w: np.ndarray, *, c: int, blocks: int, g: int) -> np.ndarray:
    tab = np.zeros((g, c, blocks, c), dtype=np.float64)
    j = np.arange(c, dtype=np.int32)
    for ge in range(g):
        off = ge * c
        for t in range(c):
            rows = off + (j + t) % c
            for b in range(blocks):
                cols = b * c + j
                tab[ge, t, b, :] = w[rows, cols]
    return tab


def ssum(*ds: Dict[str, int]) -> Dict[str, int]:
    o: Dict[str, int] = {}
    for d in ds:
        for k, v in d.items():
            o[k] = o.get(k, 0) + int(v)
    return o


def pack_real(x: np.ndarray, m: int, n: int, g: int) -> List[np.ndarray]:
    c = n // m
    out = [np.zeros(n, dtype=np.complex128) for _ in range(g)]
    for i in range(g):
        for j in range(c):
            sl = slice(j * m, (j + 1) * m)
            out[i][sl] = x[:, i * c + j].astype(np.complex128)
    return out


def cap_b(b: int, base: int, n1: int) -> int:
    return max(base, min(int(b), base * n1))


def alloc_q(base: int, n1: int, b: int) -> List[List[int]]:
    b = cap_b(b, base, n1)
    cnt = [1] * base
    rem = b - base
    q = 1
    while rem > 0 and q < n1:
        for i in range(base):
            if rem <= 0:
                break
            if cnt[i] < n1:
                cnt[i] += 1
                rem -= 1
        q += 1
    return [list(range(cnt[i])) for i in range(base)]


def mk_seed(ctx: CKKSContext, base_cts: List[Any], *, m: int, n1: int, b: int) -> List[Dict[int, Any]]:
    qs = alloc_q(len(base_cts), n1, b)
    bank: List[Dict[int, Any]] = [dict() for _ in range(len(base_cts))]
    for i, ct0 in enumerate(base_cts):
        bank[i][0] = ct0
        for q in qs[i]:
            if q == 0:
                continue
            bank[i][q] = ct0.rot((q * m) % ctx.nslots)
    return bank


def mk_baby(ctx: CKKSContext, base_cts: List[Any], bank: List[Dict[int, Any]], *, m: int, n1: int) -> List[List[Any]]:
    out: List[List[Any]] = []
    for i, ct0 in enumerate(base_cts):
        row: List[Any] = [None] * n1
        for q in range(n1):
            if q in bank[i]:
                row[q] = bank[i][q]
            else:
                sh = (q * m) % ctx.nslots
                row[q] = ct0.rot(sh) if sh else ct0
        out.append(row)
    return out


def lin_pair(ctx: CKKSContext, gp: List[List[Any]], w: np.ndarray, *, m: int, n1: int, g: int):
    n = ctx.nslots
    c = n // m
    w_c = np.asarray(w, dtype=np.complex128)
    d_out = w_c.shape[1]
    if d_out % c != 0:
        raise ValueError(f"d_out={d_out} must be divisible by c={c}")
    hp = (g + 1) // 2
    n2 = c // n1
    blk = d_out // c
    w_tab = _pre_tbl_pair(w_c, c=c, blocks=blk, g=g)
    wr_c = np.empty(c, dtype=np.complex128)
    wi_c = np.empty(c, dtype=np.complex128)
    pt_c = np.empty(c, dtype=np.complex128)
    pt_buf = np.empty(n, dtype=np.complex128)
    pt_mat = pt_buf.reshape(c, m)
    p_shifts = [(p * n1) % c for p in range(n2)]
    cf: List[List[Any]] = [[None for _ in range(n2)] for _ in range(blk)]
    for b in range(blk):
        for p in range(n2):
            p_shift = p_shifts[p]
            q_terms: List[Any] = []
            for q in range(n1):
                t = (p_shift + q) % c
                h_terms: List[Any] = []
                for h in range(hp):
                    ge = 2 * h
                    go = ge + 1
                    wr_src = w_tab[ge, t, b, :]
                    if p_shift:
                        _roll_into(wr_src, p_shift, wr_c)
                    else:
                        wr_c[:] = wr_src
                    if go < g:
                        wi_src = w_tab[go, t, b, :]
                        if p_shift:
                            _roll_into(wi_src, p_shift, wi_c)
                        else:
                            wi_c[:] = wi_src
                        pt_c[:] = wr_c
                        pt_c += -1j * wi_c
                    else:
                        pt_c[:] = wr_c
                    pt_mat[:] = pt_c[:, None]
                    h_terms.append(gp[h][q].mul_pt(pt_buf))
                acc = _sum_terms(ctx, h_terms)
                if acc is not None:
                    q_terms.append(acc)
            s = _sum_terms(ctx, q_terms)
            cf[b][p] = s if s is not None else ctx.zeros()
    return cf


def lin_real(ctx: CKKSContext, gp: List[List[Any]], w: np.ndarray, *, m: int, n1: int, g: int):
    n = ctx.nslots
    c = n // m
    w_r = np.asarray(w, dtype=np.float64)
    d_out = w_r.shape[1]
    if d_out % c != 0:
        raise ValueError(f"d_out={d_out} must be divisible by c={c}")
    n2 = c // n1
    blk = d_out // c
    w_tab = _pre_tbl_real(w_r, c=c, blocks=blk, g=g)
    wr_c = np.empty(c, dtype=np.float64)
    pt_buf = np.empty(n, dtype=np.complex128)
    pt_mat = pt_buf.reshape(c, m)
    p_shifts = [(p * n1) % c for p in range(n2)]
    cf: List[List[Any]] = [[None for _ in range(n2)] for _ in range(blk)]
    for b in range(blk):
        for p in range(n2):
            p_shift = p_shifts[p]
            q_terms: List[Any] = []
            for q in range(n1):
                t = (p_shift + q) % c
                h_terms: List[Any] = []
                for ge in range(g):
                    wr_src = w_tab[ge, t, b, :]
                    if p_shift:
                        _roll_into(wr_src, p_shift, wr_c)
                    else:
                        wr_c[:] = wr_src
                    pt_mat[:] = wr_c[:, None]
                    h_terms.append(gp[ge][q].mul_pt(pt_buf))
                acc = _sum_terms(ctx, h_terms)
                if acc is not None:
                    q_terms.append(acc)
            s = _sum_terms(ctx, q_terms)
            cf[b][p] = s if s is not None else ctx.zeros()
    return cf


def fold(ctx: CKKSContext, cf, *, m: int, n1: int) -> List[Any]:
    n = ctx.nslots
    c = n // m
    n2 = c // n1
    out: List[Any] = []
    for b in range(len(cf)):
        terms: List[Any] = []
        for p in range(n2):
            ct = cf[b][p]
            sh = (p * n1 * m) % n
            if sh:
                ct = ct.rot(sh)
            terms.append(ct)
        acc = _sum_terms(ctx, terms)
        out.append(acc if acc is not None else ctx.zeros())
    return out


def dec(ctx: CKKSContext, cts: Sequence[Any], *, m: int, d_out: int) -> np.ndarray:
    n = ctx.nslots
    c = n // m
    out = np.zeros((m, d_out), dtype=np.float64)
    with ctx.decorypt_scope():
        for b in range(len(cts)):
            used = min(c, max(0, d_out - b * c))
            if used <= 0:
                break
            v = cts[b].decorypt(copy=True, readonly=True).real
            out[:, b * c : b * c + used] = v[: used * m].reshape(used, m).T
    return out


def run1(
    *, x: np.ndarray, w: np.ndarray, m: int, n1: int, b_over: Optional[int], kernel: str = "pair"
) -> Dict[str, Any]:
    ctx = CKKSContext(NSLOTS)
    n = ctx.nslots
    c = n // m
    d_in, d_out = w.shape
    g = d_in // c
    s0 = stat_snap(ctx)
    r = pack_real(x, m, n, g)
    rc = [ctx.encorypt(v) for v in r]
    if kernel == "pair":
        base: List[Any] = []
        i = 0
        while i < len(rc):
            ct = ct_real(ctx, rc[i])
            if i + 1 < len(rc):
                ct = ct.add(ct_real(ctx, rc[i + 1]).mul_scalar(1j))
            base.append(ct)
            i += 2
    elif kernel == "real":
        base = [ct_real(ctx, ct) for ct in rc]
    else:
        raise ValueError(f"Unknown FFN kernel: {kernel}")
    s1 = stat_snap(ctx)
    k1 = stat_diff(s0, s1)
    b = len(base) if b_over is None else cap_b(b_over, len(base), n1)
    s2 = stat_snap(ctx)
    bank = mk_seed(ctx, base, m=m, n1=n1, b=b)
    s3 = stat_snap(ctx)
    k2 = stat_diff(s2, s3)
    s4 = stat_snap(ctx)
    baby = mk_baby(ctx, base, bank, m=m, n1=n1)
    if kernel == "pair":
        cf = lin_pair(ctx, baby, w, m=m, n1=n1, g=g)
    else:
        cf = lin_real(ctx, baby, w, m=m, n1=n1, g=g)
    s5 = stat_snap(ctx)
    k3 = stat_diff(s4, s5)
    s6 = stat_snap(ctx)
    fd = fold(ctx, cf, m=m, n1=n1)
    out = [ct_real(ctx, ct) for ct in fd]
    s7 = stat_snap(ctx)
    k4 = stat_diff(s6, s7)
    y = dec(ctx, out, m=m, d_out=d_out)
    ref = x @ w
    return dict(
        ct_in=len(base),
        ct_out=((len(out) + 1) // 2) if kernel == "pair" else len(out),
        ks=ssum(k1, k2, k3, k4),
        err=rel_err(y, ref),
        max_abs=float(np.max(np.abs(y - ref))),
    )


def pl(tag: str, r: Dict[str, Any]) -> None:
    print_stage(
        tag,
        ct_in=r["ct_in"],
        ct_out=r["ct_out"],
        ks_rots=r["ks"]["ks_rots"],
        ks_muls=r["ks"]["ks_muls_ctct"],
        ks_conj=r["ks"]["ks_conj"],
        rel_err=r["err"],
    )


def run() -> None:
    np.random.seed(SEED)
    x = np.random.randn(M, D1).astype(np.float64)
    w1 = np.random.randn(D1, DMID).astype(np.float64)
    w2 = np.random.randn(DMID, D2).astype(np.float64)
    r1 = run1(x=x, w=w1, m=M, n1=N1, b_over=B1, kernel="pair")
    pl("FF1", r1)
    print(f"  max_abs_err = {r1['max_abs']:.6e}")
    assert r1["err"] < 1e-09, f"FAIL: FF1 rel_err={r1['err']}"
    h1 = x @ w1
    r2 = run1(x=h1, w=w2, m=M, n1=N1, b_over=B2, kernel="pair")
    pl("FF2", r2)
    print(f"  max_abs_err = {r2['max_abs']:.6e}")
    assert r2["err"] < 1e-09, f"FAIL: FF2 rel_err={r2['err']}"


def main() -> None:
    run()


if __name__ == "__main__":
    main()
