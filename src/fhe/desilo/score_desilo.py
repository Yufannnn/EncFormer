from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import numpy as np

import src.fhe.simulator.score as sim
from src.fhe.desilo.desilo_runtime import (
    LevelCKKSContext,
    attempts_summary,
    probe_max_level,
    search_lowest_working_level,
    set_visible_gpus,
)
from src.models.model_config import get_config
from src.utils import print_stage, rel_err, stat_diff, stat_snap


def run_once(
    *,
    enc_level: int,
    mode: str,
    thread_count: int,
    log_coeff_count: int,
    special_prime_count: int,
    seed: int,
    d: int,
    h: int,
    case_rel_blocks: int,
    case_rel_c_used: int,
    model_config: str = "bert-base",
) -> Dict[str, Any]:
    cfg = get_config(model_config)
    m = cfg.m
    d1 = cfg.d_model
    nslots = cfg.nslots

    np.random.seed(seed)

    ctx = LevelCKKSContext(
        default_enc_level=enc_level,
        mode=mode,
        thread_count=thread_count,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )
    if ctx.nslots != nslots:
        raise ValueError(f"Score expects nslots={nslots}, got {ctx.nslots}")

    A = np.random.randn(m, d1).astype(np.float64)
    WQ = np.random.randn(d1, d).astype(np.float64)
    WK = np.random.randn(d1, d).astype(np.float64)

    Q = A @ WQ
    K = A @ WK
    S_ref = sim.score_ref(Q, K, h)

    Q_cts, Q_used = sim.pack_fdp(
        ctx,
        Q,
        m=m,
        H=h,
        c_used=case_rel_c_used,
        blocks_override=case_rel_blocks,
    )
    K_cts, K_used = sim.pack_fdp(
        ctx,
        K,
        m=m,
        H=h,
        c_used=case_rel_c_used,
        blocks_override=case_rel_blocks,
    )

    t0 = time.perf_counter()
    s0 = stat_snap(ctx)

    mp_half, g = sim.mapf(m, sim.B_FOLD)
    Q_bank, K_bank, K_bank_h = sim.mk_bank(
        ctx,
        Q_cts,
        Q_used,
        K_cts,
        K_used,
        m=m,
        b=sim.B_FOLD,
        g=g,
    )
    raw_fold = sim.emit_f(Q_bank, K_bank, K_bank_h, mp_half, m=m)

    if case_rel_c_used % h == 0:
        r_per_bid = [0 for _ in range(case_rel_blocks)]
    else:
        r_per_bid = [bid * case_rel_c_used % h for bid in range(case_rel_blocks)]

    half = m // 2
    D_fold = []
    for t in range(half):
        term_by_r: Dict[int, Any] = {}
        for bid, _, term in raw_fold[t]:
            r = r_per_bid[bid]
            term_by_r[r] = term if r not in term_by_r else term_by_r[r].add(term)

        acc = None
        for r, term_sum in term_by_r.items():
            red = sim.red_h(ctx, term_sum, used_cols=case_rel_c_used, H=h, m=m)
            if r != 0:
                red = sim.aln_h(ctx, red, H=h, m=m, r=r)
            acc = red if acc is None else acc.add(red)
        if acc is None:
            raise RuntimeError("Unexpected empty accumulator in Score fold")
        D_fold.append(acc)

    P = sim.pack_f(ctx, D_fold, H=h, m=m)

    s1 = stat_snap(ctx)
    d01 = stat_diff(s0, s1)

    S_out = sim.unpack_f(ctx, P, H=h, m=m, mp_half=mp_half)
    err_out = rel_err(S_out, S_ref)
    elapsed_sec = time.perf_counter() - t0

    print_stage(
        "Score-ds",
        ct_in=None,
        ct_out=len(P),
        ks_rots=d01["ks_rots"],
        ks_muls=d01["ks_muls_ctct"],
        ks_conj=d01["ks_conj"],
        rel_err=err_out,
    )

    return {
        "elapsed_sec": elapsed_sec,
        "rel_err": float(err_out),
        "stats": d01,
        "ct_out": len(P),
        "enc_level": enc_level,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run Score module on Desilo CKKS engine")
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
    p.add_argument("--case-rel-blocks", type=int, default=None, help="Number of relation blocks for packed Score")
    p.add_argument("--case-rel-c-used", type=int, default=None, help="Columns used per relation block")
    p.add_argument("--start-level", type=int, default=8)
    p.add_argument("--fixed-level", type=int, default=None)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.model_config)
    log_coeff_count = args.log_coeff_count or (int(np.log2(cfg.nslots)) + 1)

    set_visible_gpus(args.gpu if args.mode == "gpu" else None)

    if args.quick:
        d = 64
        h = 4
        case_rel_blocks = 2 if args.case_rel_blocks is None else int(args.case_rel_blocks)
        case_rel_c_used = 32 if args.case_rel_c_used is None else int(args.case_rel_c_used)
    else:
        d = cfg.d_model
        h = cfg.H
        C = cfg.C
        case_rel_c_used = C if args.case_rel_c_used is None else int(args.case_rel_c_used)
        case_rel_blocks = -(-d // case_rel_c_used) if args.case_rel_blocks is None else int(args.case_rel_blocks)

    if args.fixed_level is not None:
        out = run_once(
            enc_level=args.fixed_level,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=log_coeff_count,
            special_prime_count=args.special_prime_count,
            seed=args.seed,
            d=d,
            h=h,
            case_rel_blocks=case_rel_blocks,
            case_rel_c_used=case_rel_c_used,
            model_config=args.model_config,
        )
        print(
            f"[Score-ds] gpu={args.gpu} model={args.model_config} level={args.fixed_level} d={d} h={h} "
            f"case_rel_blocks={case_rel_blocks} case_rel_c_used={case_rel_c_used} "
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
            d=d,
            h=h,
            case_rel_blocks=case_rel_blocks,
            case_rel_c_used=case_rel_c_used,
            model_config=args.model_config,
        ),
        start_level=args.start_level,
        min_level=args.min_level,
        max_level=max_level,
    )

    print(f"[Score-ds] attempts: {attempts_summary(attempts)}")
    print(
        f"[Score-ds] gpu={args.gpu} model={args.model_config} best_level={best_level} d={d} h={h} "
        f"case_rel_blocks={case_rel_blocks} case_rel_c_used={case_rel_c_used} "
        f"elapsed={out['elapsed_sec']:.3f}s rel_err={out['rel_err']:.3e}"
    )


if __name__ == "__main__":
    main()
