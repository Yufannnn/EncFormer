from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Optional

import numpy as np

import src.fhe.simulator.ffn as sim
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


def _run1_with_ctx(
    *,
    ctx: LevelCKKSContext,
    x: np.ndarray,
    w: np.ndarray,
    m: int,
    n1: int,
    b_over: Optional[int],
    out_force_real: bool,
    kernel: str,
    expected_nslots: int | None = None,
) -> Dict[str, Any]:
    if expected_nslots is not None and ctx.nslots != expected_nslots:
        raise ValueError(f"FFN expects nslots={expected_nslots}, got {ctx.nslots}")

    n = ctx.nslots
    c = n // m
    d_in, d_out = w.shape
    g = d_in // c

    s0 = stat_snap(ctx)
    r = sim.pack_real(x, m, n, g)
    rc = [ctx.encorypt(v) for v in r]

    if kernel == "pair":
        base = []
        i = 0
        while i < len(rc):
            ct = _real_if_needed(ctx, rc[i])
            if i + 1 < len(rc):
                ct = ct.add(_real_if_needed(ctx, rc[i + 1]).mul_scalar(1j))
            base.append(ct)
            i += 2
    elif kernel == "real":
        base = [_real_if_needed(ctx, ct) for ct in rc]
    else:
        raise ValueError(f"Unknown FFN kernel: {kernel}")

    s1 = stat_snap(ctx)
    k1 = stat_diff(s0, s1)

    b = len(base) if b_over is None else sim.cap_b(b_over, len(base), n1)

    s2 = stat_snap(ctx)
    bank = sim.mk_seed(ctx, base, m=m, n1=n1, b=b)
    s3 = stat_snap(ctx)
    k2 = stat_diff(s2, s3)

    s4 = stat_snap(ctx)
    baby = sim.mk_baby(ctx, base, bank, m=m, n1=n1)
    if kernel == "pair":
        cf = sim.lin_pair(ctx, baby, w, m=m, n1=n1, g=g)
    else:
        cf = sim.lin_real(ctx, baby, w, m=m, n1=n1, g=g)
    s5 = stat_snap(ctx)
    k3 = stat_diff(s4, s5)

    s6 = stat_snap(ctx)
    fd = sim.fold(ctx, cf, m=m, n1=n1)
    out = [ct_real(ctx, ct) for ct in fd] if out_force_real else fd
    s7 = stat_snap(ctx)
    k4 = stat_diff(s6, s7)

    y = sim.dec(ctx, out, m=m, d_out=d_out)

    return {
        "ct_in": len(base),
        "ct_out": ((len(out) + 1) // 2) if kernel == "pair" else len(out),
        "ks": sim.ssum(k1, k2, k3, k4),
        "err": rel_err(y, x @ w),
    }


def run1_desilo(
    *,
    x: np.ndarray,
    w: np.ndarray,
    m: int,
    n1: int,
    b_over: Optional[int],
    enc_level: int,
    mode: str,
    thread_count: int,
    log_coeff_count: int,
    special_prime_count: int,
    out_force_real: bool,
    kernel: str,
    expected_nslots: int | None = None,
) -> Dict[str, Any]:
    ctx = LevelCKKSContext(
        default_enc_level=enc_level,
        mode=mode,
        thread_count=thread_count,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )
    return _run1_with_ctx(
        ctx=ctx,
        x=x,
        w=w,
        m=m,
        n1=n1,
        b_over=b_over,
        out_force_real=out_force_real,
        kernel=kernel,
        expected_nslots=expected_nslots,
    )


def run_once(
    *,
    enc_level: int,
    mode: str,
    thread_count: int,
    log_coeff_count: int,
    special_prime_count: int,
    seed: int,
    d1: int,
    dmid: int,
    d2: int,
    out_force_real: bool,
    n1_ff1: Optional[int] = None,
    n1_ff2: Optional[int] = None,
    kernel: str = "pair",
    model_config: str = "bert-base",
) -> Dict[str, Any]:
    cfg = get_config(model_config)
    m = cfg.m
    nslots = cfg.nslots
    default_n1 = cfg.n1

    np.random.seed(seed)
    x = np.random.randn(m, d1).astype(np.float64)
    w1 = np.random.randn(d1, dmid).astype(np.float64)
    w2 = np.random.randn(dmid, d2).astype(np.float64)

    ctx = LevelCKKSContext(
        default_enc_level=enc_level,
        mode=mode,
        thread_count=thread_count,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )

    t0 = time.perf_counter()
    use_n1_ff1 = default_n1 if n1_ff1 is None else int(n1_ff1)
    use_n1_ff2 = default_n1 if n1_ff2 is None else int(n1_ff2)

    r1 = _run1_with_ctx(
        ctx=ctx,
        x=x,
        w=w1,
        m=m,
        n1=use_n1_ff1,
        b_over=None,
        out_force_real=out_force_real,
        kernel=kernel,
        expected_nslots=nslots,
    )
    print_stage(
        "FF1-ds",
        ct_in=r1["ct_in"],
        ct_out=r1["ct_out"],
        ks_rots=r1["ks"]["ks_rots"],
        ks_muls=r1["ks"]["ks_muls_ctct"],
        ks_conj=r1["ks"]["ks_conj"],
        rel_err=r1["err"],
    )

    h1 = x @ w1
    r2 = _run1_with_ctx(
        ctx=ctx,
        x=h1,
        w=w2,
        m=m,
        n1=use_n1_ff2,
        b_over=None,
        out_force_real=out_force_real,
        kernel=kernel,
        expected_nslots=nslots,
    )
    print_stage(
        "FF2-ds",
        ct_in=r2["ct_in"],
        ct_out=r2["ct_out"],
        ks_rots=r2["ks"]["ks_rots"],
        ks_muls=r2["ks"]["ks_muls_ctct"],
        ks_conj=r2["ks"]["ks_conj"],
        rel_err=r2["err"],
    )

    elapsed_sec = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed_sec,
        "ff1_err": float(r1["err"]),
        "ff2_err": float(r2["err"]),
        "ff1_stats": r1["ks"],
        "ff2_stats": r2["ks"],
        "ff1_ct_in": int(r1["ct_in"]),
        "ff1_ct_out": int(r1["ct_out"]),
        "ff2_ct_in": int(r2["ct_in"]),
        "ff2_ct_out": int(r2["ct_out"]),
        "enc_level": enc_level,
        "out_force_real": out_force_real,
        "n1_ff1": int(use_n1_ff1),
        "n1_ff2": int(use_n1_ff2),
        "kernel": str(kernel),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run FFN module on Desilo CKKS engine")
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
    p.add_argument(
        "--out-force-real",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to force ct_real on folded FFN outputs before decrypt/eval.",
    )
    p.add_argument("--kernel", type=str, default="pair", choices=["pair", "real"], help="FFN kernel variant.")
    p.add_argument("--n1-ff1", type=int, default=None, help="Baby-step N1 for FFN layer 1.")
    p.add_argument("--n1-ff2", type=int, default=None, help="Baby-step N1 for FFN layer 2.")
    p.add_argument("--start-level", type=int, default=8)
    p.add_argument("--fixed-level", type=int, default=None)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.model_config)
    log_coeff_count = args.log_coeff_count or (int(np.log2(cfg.nslots)) + 1)

    if args.quick:
        d1 = 256
        dmid = 512
        d2 = 256
    else:
        d1 = cfg.d_model
        dmid = cfg.d_ff
        d2 = cfg.d_model

    set_visible_gpus(args.gpu if args.mode == "gpu" else None)

    if args.fixed_level is not None:
        out = run_once(
            enc_level=args.fixed_level,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=log_coeff_count,
            special_prime_count=args.special_prime_count,
            seed=args.seed,
            d1=d1,
            dmid=dmid,
            d2=d2,
            out_force_real=args.out_force_real,
            n1_ff1=args.n1_ff1,
            n1_ff2=args.n1_ff2,
            kernel=args.kernel,
            model_config=args.model_config,
        )
        print(
            f"[FFN-ds] gpu={args.gpu} model={args.model_config} level={args.fixed_level} "
            f"d1={d1} dmid={dmid} d2={d2} "
            f"kernel={out['kernel']} n1_ff1={out['n1_ff1']} n1_ff2={out['n1_ff2']} "
            f"out_force_real={args.out_force_real} "
            f"elapsed={out['elapsed_sec']:.3f}s "
            f"ff1_err={out['ff1_err']:.3e} ff2_err={out['ff2_err']:.3e}"
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
            d1=d1,
            dmid=dmid,
            d2=d2,
            out_force_real=args.out_force_real,
            n1_ff1=args.n1_ff1,
            n1_ff2=args.n1_ff2,
            kernel=args.kernel,
            model_config=args.model_config,
        ),
        start_level=args.start_level,
        min_level=args.min_level,
        max_level=max_level,
    )

    print(f"[FFN-ds] attempts: {attempts_summary(attempts)}")
    print(
        f"[FFN-ds] gpu={args.gpu} model={args.model_config} best_level={best_level} "
        f"d1={d1} dmid={dmid} d2={d2} "
        f"kernel={out['kernel']} n1_ff1={out['n1_ff1']} n1_ff2={out['n1_ff2']} "
        f"out_force_real={args.out_force_real} "
        f"elapsed={out['elapsed_sec']:.3f}s "
        f"ff1_err={out['ff1_err']:.3e} ff2_err={out['ff2_err']:.3e}"
    )


if __name__ == "__main__":
    main()
