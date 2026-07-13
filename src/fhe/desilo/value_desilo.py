from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

import src.fhe.simulator.value as sim
from src.fhe.desilo.desilo_runtime import (
    LevelCKKSContext,
    attempts_summary,
    probe_max_level,
    search_lowest_working_level,
    set_visible_gpus,
)
from src.models.model_config import get_config
from src.utils import ct_real, print_stage, rel_err, stat_diff, stat_snap


def _smx(x: np.ndarray) -> np.ndarray:

    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _mk_diag_vec(A: np.ndarray, t: int, m: int) -> np.ndarray:
    rows = np.arange(m, dtype=np.int32)
    cols = (rows - (t % m)) % m
    return A[rows, cols].astype(np.complex128, copy=False)


def _pack_a_diag_ct(
    ctx: LevelCKKSContext,
    A_heads: List[np.ndarray],
    *,
    t: int,
    m: int,
    d_h: int,
    heads_per_ct: int,
) -> Any:
    n = ctx.nslots
    v = np.zeros(n, dtype=np.complex128)
    mat = v.reshape(n // m, m)
    for h in range(heads_per_ct):
        d = _mk_diag_vec(A_heads[h], t=t, m=m)
        seg_lo = h * d_h
        seg_hi = seg_lo + d_h
        mat[seg_lo:seg_hi, :] = d[None, :]
    v.setflags(write=False)
    return ctx.encorypt(v)


def _pack_a_diag_pair_ct(
    ctx: LevelCKKSContext,
    A_heads: List[np.ndarray],
    *,
    t: int,
    m: int,
    d_h: int,
    heads_per_ct: int,
) -> Any:

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


def _sv_baseline_noncomplex(
    ctx: LevelCKKSContext,
    *,
    A_heads: List[np.ndarray],
    V_ct: Any,
    m: int,
    d_h: int,
    heads_per_ct: int,
    sv_relin: bool,
    sv_final_relin: bool,
) -> tuple[Any, Dict[str, int]]:
    n = ctx.nslots
    C = n // m
    rot_break = defaultdict(int)
    mask_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rot_cache: Dict[tuple[int, int], Any] = {}

    A_diag_cts = [
        _pack_a_diag_ct(
            ctx,
            A_heads,
            t=t,
            m=m,
            d_h=d_h,
            heads_per_ct=heads_per_ct,
        )
        for t in range(m)
    ]

    out = None
    for t in range(m):
        if t == 0:
            V_t = V_ct
        else:
            V_t = sim.rot_within(
                ctx,
                V_ct,
                m=m,
                used_C=C,
                t=-t % m,
                mask_cache=mask_cache,
                rot_cache=rot_cache,
                rot_break=rot_break,
                tag="V_shift_nc",
            )
        term = V_t.mul_ct(A_diag_cts[t], relin=sv_relin)
        out = sim._add_assign(out, term)
    if out is None:
        raise RuntimeError("non-complex baseline SV produced empty output")
    if (not sv_relin) and sv_final_relin:
        relin_fn = getattr(out, "relinearize", None)
        if callable(relin_fn):
            out = relin_fn()
    return out, dict(rot_break)


def _sv_pair_direct(
    ctx: LevelCKKSContext,
    *,
    A_heads: List[np.ndarray],
    V_ct: Any,
    m: int,
    d_h: int,
    heads_per_ct: int,
    sv_relin: bool,
    sv_final_relin: bool,
) -> tuple[Any, Dict[str, int]]:
    n = ctx.nslots
    C = n // m
    half = m // 2
    rot_break = defaultdict(int)
    mask_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rot_cache: Dict[tuple[int, int], Any] = {}

    A_pair_cts = [
        _pack_a_diag_pair_ct(
            ctx,
            A_heads,
            t=t,
            m=m,
            d_h=d_h,
            heads_per_ct=heads_per_ct,
        )
        for t in range(half)
    ]

    V_half = sim.rot_within(
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
    U = sim._add_assign(V_ct, V_half.mul_scalar(-1j))

    out = None
    U_mask_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    U_rot_cache: Dict[tuple[int, int], Any] = {}
    for t in range(half):
        if t == 0:
            U_t = U
        else:
            U_t = sim.rot_within(
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
        term = U_t.mul_ct(A_pair_cts[t], relin=sv_relin)
        out = sim._add_assign(out, term)
    if out is None:
        raise RuntimeError("pair-direct SV produced empty output")
    if (not sv_relin) and sv_final_relin:
        relin_fn = getattr(out, "relinearize", None)
        if callable(relin_fn):
            out = relin_fn()
    return out, dict(rot_break)


def run_once(
    *,
    enc_level: int,
    mode: str,
    thread_count: int,
    log_coeff_count: int,
    special_prime_count: int,
    seed: int,
    run_groups: int | None,
    run_out: bool,
    value_kernel: str,
    sv_relin: bool,
    sv_final_relin: bool,
    out_pre_force_real: bool,
    out_post_force_real: bool,
    out_base_relin: bool,
    model_config: str = "bert-base",
) -> Dict[str, Any]:
    cfg = get_config(model_config)
    m = cfg.m
    num_heads = cfg.H
    d_h = cfg.d_h
    d_model = cfg.d_model
    nslots = cfg.nslots
    out_n1 = cfg.n1

    np.random.seed(seed)
    A_logits_all = [np.random.randn(m, m).astype(np.float64) for _ in range(num_heads)]
    A_all = [_smx(L) for L in A_logits_all]
    V_all = [np.random.randn(m, d_h).astype(np.float64) for _ in range(num_heads)]
    W_O = np.random.randn(d_model, d_model).astype(np.float64)
    Y_truth_all = [A_all[h] @ V_all[h] for h in range(num_heads)]
    Y_truth = np.concatenate(Y_truth_all, axis=1)

    ctx = LevelCKKSContext(
        default_enc_level=enc_level,
        mode=mode,
        thread_count=thread_count,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )
    if ctx.nslots != nslots:
        raise ValueError(f"Value expects nslots={nslots}, got {ctx.nslots}")

    C = ctx.nslots // m
    heads_per_ct = C // d_h
    groups = num_heads // heads_per_ct
    if run_groups is None:
        used_groups = groups
    else:
        used_groups = int(run_groups)
        if used_groups <= 0 or used_groups > groups:
            raise ValueError(f"run_groups must be in [1, {groups}], got {used_groups}")

    t0 = time.perf_counter()

    s_sv0 = stat_snap(ctx)
    Y_groups: List[Any] = []
    for g in range(used_groups):
        ids = [g * heads_per_ct + i for i in range(heads_per_ct)]
        A_heads = [A_all[h] for h in ids]
        V_ct = sim.pack_v(ctx, [V_all[h] for h in ids], m=m, d_h=d_h, heads_per_ct=heads_per_ct)
        if value_kernel == "baseline-nc":
            Yg, _ = _sv_baseline_noncomplex(
                ctx,
                A_heads=A_heads,
                V_ct=V_ct,
                m=m,
                d_h=d_h,
                heads_per_ct=heads_per_ct,
                sv_relin=sv_relin,
                sv_final_relin=sv_final_relin,
            )
        elif value_kernel == "pair-direct":
            Yg, _ = _sv_pair_direct(
                ctx,
                A_heads=A_heads,
                V_ct=V_ct,
                m=m,
                d_h=d_h,
                heads_per_ct=heads_per_ct,
                sv_relin=sv_relin,
                sv_final_relin=sv_final_relin,
            )
        elif value_kernel == "complex":
            A_packed = sim.pack_a(ctx, A_heads, m=m, heads_per_ct=heads_per_ct)
            Yg, _ = sim.sv(ctx, A_packed=A_packed, V_ct=V_ct, m=m, d_h=d_h, heads_per_ct=heads_per_ct)
        else:
            raise ValueError(f"Unknown value kernel: {value_kernel}")
        Y_groups.append(Yg)
    s_sv1 = stat_snap(ctx)
    d_sv = stat_diff(s_sv0, s_sv1)

    group_width = heads_per_ct * d_h
    Y_dec = np.zeros((m, used_groups * group_width), dtype=np.float64)
    for g, ct in enumerate(Y_groups):
        Y_blk = sim.dec_grp(ctx, ct, m=m)
        Cseg = Y_blk.shape[1]
        Y_dec[:, g * Cseg : (g + 1) * Cseg] = Y_blk
    rel_err_main = rel_err(Y_dec, Y_truth[:, : used_groups * group_width])

    print_stage(
        "Value-ds",
        ct_in=used_groups,
        ct_out=None,
        ks_rots=d_sv["ks_rots"],
        ks_muls=d_sv["ks_muls_ctct"],
        ks_conj=d_sv["ks_conj"],
        rel_err=rel_err_main,
    )

    err_out = None
    d_out = {"ks_rots": 0, "ks_muls_ctct": 0, "ks_conj": 0}
    if run_out:
        if used_groups != groups:
            raise ValueError("OUT stage requires all value groups. Use run_groups=None or run_groups=6.")
        s_out0 = stat_snap(ctx)
        base_cts = sim.pack_out_sv(ctx, Y_groups, force_real=out_pre_force_real)
        if out_base_relin:
            relined = []
            for ct in base_cts:
                relin_fn = getattr(ct, "relinearize", None)
                relined.append(relin_fn() if callable(relin_fn) else ct)
            base_cts = relined
        babies = sim.build_babies_min(ctx, base_cts, m=m, N1=out_n1)
        Cf_lazy = sim.linear_babies_pairs(ctx, babies, W_O, m=m, N1=out_n1, G=groups)
        folded = sim.fold_grid(ctx, Cf_lazy, m=m, N1=out_n1)
        folded_use = [ct_real(ctx, ct) for ct in folded] if out_post_force_real else folded
        Z_dec = sim.decrypt_blocks(ctx, folded_use, m=m, d_out=d_model)
        Z_truth_use = Y_truth @ W_O
        err_out = rel_err(Z_dec, Z_truth_use)
        s_out1 = stat_snap(ctx)
        d_out = stat_diff(s_out0, s_out1)

        ct_out_report = (len(folded_use) + 1) // 2
        print_stage(
            "OUT-ds",
            ct_in=None,
            ct_out=ct_out_report,
            ks_rots=d_out["ks_rots"],
            ks_muls=d_out["ks_muls_ctct"],
            ks_conj=d_out["ks_conj"],
            rel_err=err_out,
        )

    elapsed_sec = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed_sec,
        "value_rel_err": float(rel_err_main),
        "out_rel_err": None if err_out is None else float(err_out),
        "value_stats": d_sv,
        "out_stats": d_out,
        "enc_level": enc_level,
        "value_kernel": value_kernel,
        "sv_relin": sv_relin,
        "sv_final_relin": sv_final_relin,
        "out_pre_force_real": out_pre_force_real,
        "out_post_force_real": out_post_force_real,
        "out_base_relin": out_base_relin,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run Value module on Desilo CKKS engine")
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
    p.add_argument("--quick", action="store_true", help="Run only first two value groups and skip OUT stage")
    p.add_argument("--run-groups", type=int, default=None, help="Number of value groups to run (max 6)")
    p.add_argument("--skip-out", action="store_true", help="Skip OUT projection stage")
    p.add_argument(
        "--value-kernel",
        type=str,
        default="pair-direct",
        choices=["baseline-nc", "pair-direct", "complex"],
        help="Value kernel variant: baseline non-complex or packed complex",
    )
    p.add_argument(
        "--sv-relin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to relinearize each SV ct*ct multiply (baseline-nc kernel only).",
    )
    p.add_argument(
        "--sv-final-relin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to relinearize the accumulated SV output when --no-sv-relin is used.",
    )
    p.add_argument(
        "--out-force-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy switch: sets both pre and post OUT ct_real behavior unless split flags are set.",
    )
    p.add_argument(
        "--out-pre-force-real",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to force ct_real before OUT pairing (overrides --out-force-real if set).",
    )
    p.add_argument(
        "--out-post-force-real",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to force ct_real after OUT fold (overrides --out-force-real if set).",
    )
    p.add_argument(
        "--out-base-relin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to relinearize base OUT ciphertexts after pairing.",
    )
    p.add_argument("--start-level", type=int, default=8)
    p.add_argument("--fixed-level", type=int, default=None)
    p.add_argument("--min-level", type=int, default=0)
    p.add_argument("--max-level", type=int, default=None)
    args = p.parse_args()

    cfg = get_config(args.model_config)
    log_coeff_count = args.log_coeff_count or (int(np.log2(cfg.nslots)) + 1)

    run_groups = args.run_groups
    run_out = not args.skip_out
    if args.quick:
        run_groups = 2
        run_out = False

    out_pre_force_real = args.out_force_real if args.out_pre_force_real is None else bool(args.out_pre_force_real)
    out_post_force_real = args.out_force_real if args.out_post_force_real is None else bool(args.out_post_force_real)

    set_visible_gpus(args.gpu if args.mode == "gpu" else None)

    if args.fixed_level is not None:
        out = run_once(
            enc_level=args.fixed_level,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=log_coeff_count,
            special_prime_count=args.special_prime_count,
            seed=args.seed,
            run_groups=run_groups,
            run_out=run_out,
            value_kernel=args.value_kernel,
            sv_relin=args.sv_relin,
            sv_final_relin=args.sv_final_relin,
            out_pre_force_real=out_pre_force_real,
            out_post_force_real=out_post_force_real,
            out_base_relin=args.out_base_relin,
            model_config=args.model_config,
        )
        print(
            f"[Value-ds] gpu={args.gpu} model={args.model_config} kernel={args.value_kernel} "
            f"level={args.fixed_level} sv_relin={args.sv_relin} "
            f"sv_final_relin={args.sv_final_relin} out_pre_force_real={out_pre_force_real} "
            f"out_post_force_real={out_post_force_real} out_base_relin={args.out_base_relin} "
            f"elapsed={out['elapsed_sec']:.3f}s value_err={out['value_rel_err']:.3e} out_err={out['out_rel_err']}"
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
            run_groups=run_groups,
            run_out=run_out,
            value_kernel=args.value_kernel,
            sv_relin=args.sv_relin,
            sv_final_relin=args.sv_final_relin,
            out_pre_force_real=out_pre_force_real,
            out_post_force_real=out_post_force_real,
            out_base_relin=args.out_base_relin,
            model_config=args.model_config,
        ),
        start_level=args.start_level,
        min_level=args.min_level,
        max_level=max_level,
    )

    print(f"[Value-ds] attempts: {attempts_summary(attempts)}")
    print(
        f"[Value-ds] gpu={args.gpu} model={args.model_config} kernel={args.value_kernel} "
        f"best_level={best_level} sv_relin={args.sv_relin} "
        f"sv_final_relin={args.sv_final_relin} out_pre_force_real={out_pre_force_real} "
        f"out_post_force_real={out_post_force_real} out_base_relin={args.out_base_relin} "
        f"elapsed={out['elapsed_sec']:.3f}s value_err={out['value_rel_err']:.3e} out_err={out['out_rel_err']}"
    )


if __name__ == "__main__":
    main()
