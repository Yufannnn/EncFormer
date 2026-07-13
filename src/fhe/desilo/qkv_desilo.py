from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import numpy as np

import src.fhe.simulator.qkv as sim
from src.fhe.desilo.desilo_runtime import (
    LevelCKKSContext,
    attempts_summary,
    probe_max_level,
    search_lowest_working_level,
    set_visible_gpus,
)
from src.models.model_config import get_config
from src.utils import ct_real, print_stage, rel_err, stat_diff, stat_snap


def _real_if_needed(ctx: LevelCKKSContext, ct: Any):
    if hasattr(ct, "_ct_i") and getattr(ct, "_ct_i") is None:
        return ct
    return ct_real(ctx, ct)


def run_once(
    *,
    enc_level: int,
    mode: str,
    thread_count: int,
    log_coeff_count: int,
    special_prime_count: int,
    seed: int,
    d2: int,
    qk_c_used: int,
    v_c_used: int,
    n1: int,
    model_config: str = "bert-base",
) -> Dict[str, Any]:
    cfg = get_config(model_config)
    m = cfg.m
    d1 = cfg.d1
    d_model = cfg.d_model
    H = cfg.H
    DH = cfg.d_h
    nslots = cfg.nslots

    np.random.seed(seed)

    A = np.random.randn(m, d1).astype(np.float64)
    WQ = np.random.randn(d1, d2).astype(np.float64)
    WK = np.random.randn(d1, d2).astype(np.float64)
    WV = np.random.randn(d1, d2).astype(np.float64)

    if d2 == d_model:
        perm_fdp_cols = sim.perm_fdp(H, DH)
        perm_hm_cols = sim.perm_hd(H, DH)
        WQ_fdp = WQ[:, perm_fdp_cols]
        WK_fdp = WK[:, perm_fdp_cols]
        WV_hm = WV[:, perm_hm_cols]
    else:
        WQ_fdp = WQ
        WK_fdp = WK
        WV_hm = WV

    ctx = LevelCKKSContext(
        default_enc_level=enc_level,
        mode=mode,
        thread_count=thread_count,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )
    if ctx.nslots != nslots:
        raise ValueError(f"QKV expects nslots={nslots}, got {ctx.nslots}")

    s0 = stat_snap(ctx)
    t0 = time.perf_counter()

    vs = sim.pa6(A, m, ctx.nslots)
    vp = [vs[0] + 1j * vs[1], vs[2] + 1j * vs[3], vs[4] + 1j * vs[5]]
    A_pairs = [ctx.encorypt(v) for v in vp]

    ct_pairs_babies = []
    for Ap in A_pairs:
        row = []
        for q in range(n1):
            shift = q * m % ctx.nslots
            row.append(Ap.rot(shift) if shift else Ap)
        ct_pairs_babies.append(row)

    C = ctx.nslots // m
    QK_tab, _ = sim.pre_tbl_c(WQ_fdp, WK_fdp, WV_hm, C, d2, c_used=qk_c_used)
    _QK2, V_tab = sim.pre_tbl_c(WQ_fdp, WK_fdp, WV_hm, C, d2, c_used=v_c_used)

    QK_grid, V_grid = sim.proj_mix(
        ctx,
        ct_pairs_babies=ct_pairs_babies,
        QK_tab=QK_tab,
        V_tab=V_tab,
        d2=d2,
        qk_c_used=qk_c_used,
        qk_blocks=(d2 + qk_c_used - 1) // qk_c_used,
        v_c_used=v_c_used,
        v_blocks=(d2 + v_c_used - 1) // v_c_used,
        m=m,
        N1=n1,
    )

    qk_blocks = (d2 + qk_c_used - 1) // qk_c_used
    v_blocks = (d2 + v_c_used - 1) // v_c_used
    QK_blocks = sim.fold_grid(QK_grid, ctx)[:qk_blocks]
    V_raw_blocks = sim.fold_grid(V_grid, ctx)[:v_blocks]
    Q_blocks, K_blocks = sim.split_qk(ctx, QK_blocks)
    Q_blocks = [_real_if_needed(ctx, ct) for ct in Q_blocks]
    K_blocks = [_real_if_needed(ctx, ct) for ct in K_blocks]
    V_blocks = [_real_if_needed(ctx, ct) for ct in V_raw_blocks]

    s1 = stat_snap(ctx)
    d01 = stat_diff(s0, s1)

    Qb = sim.decrypt_blocks(Q_blocks, m, d2, ctx, c_used=qk_c_used)
    Kb = sim.decrypt_blocks(K_blocks, m, d2, ctx, c_used=qk_c_used)
    Vb = sim.decrypt_blocks(V_blocks, m, d2, ctx, c_used=v_c_used)

    rel_err_avg = (rel_err(Qb, A @ WQ_fdp) + rel_err(Kb, A @ WK_fdp) + rel_err(Vb, A @ WV_hm)) / 3.0
    elapsed_sec = time.perf_counter() - t0

    print_stage(
        "QKV-ds",
        ct_in=3,
        ct_out=None,
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=rel_err_avg,
    )

    return {
        "elapsed_sec": elapsed_sec,
        "rel_err": float(rel_err_avg),
        "stats": d01,
        "enc_level": enc_level,
        "mode": mode,
        "nslots": ctx.nslots,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run QKV module on Desilo CKKS engine")
    p.add_argument("--gpu", type=str, default="2", help="CUDA_VISIBLE_DEVICES value, e.g. '2' or '3'")
    p.add_argument("--mode", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--thread-count", type=int, default=0)
    p.add_argument(
        "--log-coeff-count",
        type=int,
        default=None,
        help="Polynomial degree log2(N); auto-derived from model config if omitted",
    )
    p.add_argument("--special-prime-count", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-config",
        type=str,
        default="bert-base",
        choices=["bert-base", "bert-large", "gpt2-base"],
        help="Model config for geometry (default: bert-base)",
    )
    p.add_argument("--quick", action="store_true", help="Use reduced dimensions for fast benchmarking")
    p.add_argument("--qk-c-used", type=int, default=None, help="Q/K columns used per block (<= C)")
    p.add_argument("--v-c-used", type=int, default=None, help="V columns used per block (<= C)")
    p.add_argument("--n1", type=int, default=None, help="Baby-step N1 (must divide C)")
    p.add_argument("--start-level", type=int, default=8)
    p.add_argument("--fixed-level", type=int, default=None)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.model_config)
    log_coeff_count = args.log_coeff_count or (int(np.log2(cfg.nslots)) + 1)

    set_visible_gpus(args.gpu if args.mode == "gpu" else None)

    if args.quick:
        d2 = 64
        qk_c_used = 32 if args.qk_c_used is None else int(args.qk_c_used)
        v_c_used = 32 if args.v_c_used is None else int(args.v_c_used)
        n1 = 8 if args.n1 is None else int(args.n1)
    else:
        d2 = cfg.d_model
        C = cfg.C
        qk_c_used = C if args.qk_c_used is None else int(args.qk_c_used)
        v_c_used = C if args.v_c_used is None else int(args.v_c_used)
        n1 = cfg.n1 if args.n1 is None else int(args.n1)

    if args.fixed_level is not None:
        out = run_once(
            enc_level=args.fixed_level,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=log_coeff_count,
            special_prime_count=args.special_prime_count,
            seed=args.seed,
            d2=d2,
            qk_c_used=qk_c_used,
            v_c_used=v_c_used,
            n1=n1,
            model_config=args.model_config,
        )
        print(
            f"[QKV-ds] gpu={args.gpu} model={args.model_config} level={args.fixed_level} d2={d2} "
            f"qk_c_used={qk_c_used} v_c_used={v_c_used} n1={n1} "
            f"elapsed={out['elapsed_sec']:.3f}s rel_err={out['rel_err']:.3e}"
        )
        return

    max_level = args.max_level
    if max_level is None:
        max_level = probe_max_level(
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=log_coeff_count,
            special_prime_count=args.special_prime_count,
        )

    best_level, out, attempts = search_lowest_working_level(
        lambda lv: run_once(
            enc_level=lv,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=log_coeff_count,
            special_prime_count=args.special_prime_count,
            seed=args.seed,
            d2=d2,
            qk_c_used=qk_c_used,
            v_c_used=v_c_used,
            n1=n1,
            model_config=args.model_config,
        ),
        start_level=args.start_level,
        min_level=args.min_level,
        max_level=max_level,
    )

    print(f"[QKV-ds] attempts: {attempts_summary(attempts)}")
    print(
        f"[QKV-ds] gpu={args.gpu} model={args.model_config} best_level={best_level} d2={d2} "
        f"qk_c_used={qk_c_used} v_c_used={v_c_used} n1={n1} "
        f"elapsed={out['elapsed_sec']:.3f}s rel_err={out['rel_err']:.3e}"
    )


if __name__ == "__main__":
    main()
