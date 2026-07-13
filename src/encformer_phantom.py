from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import src.fhe.simulator.qkv as qkv_sim
import src.fhe.simulator.score as score_sim
from src.fhe.phantom.phantom_native_bench import (
    resolve_native_binaries,
    run_ffn_full_native,
    run_qkv_full_native,
    run_score_full_native,
    run_value_full_native,
)
from src.fhe.phantom.phantom_runtime import LevelCKKSContext, set_visible_gpus


def _sum_ks(*stats: Dict[str, int]) -> Dict[str, int]:
    out = {"ks_rots": 0, "ks_muls_ctct": 0, "ks_conj": 0}
    for s in stats:
        out["ks_rots"] += int(s.get("ks_rots", 0))
        out["ks_muls_ctct"] += int(s.get("ks_muls_ctct", 0))
        out["ks_conj"] += int(s.get("ks_conj", 0))
    return out


def _stage_hdr(name: str, elapsed: float) -> None:
    print(f"\n[{name}] elapsed={elapsed:.3f}s")


@contextlib.contextmanager
def _temp_env(var: str, value: str | None):
    old = os.environ.get(var)
    if value is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = str(value)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


@contextlib.contextmanager
def _patch_core_ctx(*, seed: int, ctx_factory):
    import src.encformer as core

    old_ctx = core.CKKSContext
    old_seed = core.SEED
    core.CKKSContext = ctx_factory
    core.SEED = int(seed)
    try:
        yield core
    finally:
        core.CKKSContext = old_ctx
        core.SEED = old_seed


def _run_bridge_pipeline(
    args,
    *,
    log_coeff_count: int,
    special_prime_count: int,
    galois_mode: str,
    segmented: bool = False,
) -> None:
    enc_levels = (
        {
            "qkv": args.qkv_level,
            "score": args.score_level,
            "value": args.value_level,
            "ff1": args.ffn_level,
            "ff2": args.ffn_level,
        }
        if segmented
        else None
    )

    def _ctx_factory(nslots: int):
        import src.encformer as core

        seg_level = getattr(core, "_SEG_ENC_LEVEL", None)
        enc_level = (
            seg_level if seg_level is not None else (None if args.bridge_level is None else int(args.bridge_level))
        )
        return LevelCKKSContext(
            default_enc_level=enc_level,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=int(log_coeff_count),
            special_prime_count=int(special_prime_count),
            nslots=int(nslots),
            auto_rescale_mul_pt=True,
            auto_rescale_mul_ct=True,
        )

    with _temp_env("PHANTOM_GALOIS_MODE", galois_mode):
        use_cc = not getattr(args, "no_cc", False)
        use_scp = not getattr(args, "no_scp", False)
        with _patch_core_ctx(seed=args.seed, ctx_factory=_ctx_factory) as core:
            core.main(
                use_cc=use_cc, use_scp=use_scp, mpc_engine=args.mpc_engine, segmented=segmented, enc_levels=enc_levels
            )


def _gpu_key(gpu: str) -> str:
    g = str(gpu).strip()
    return g if g else "__none__"


def main() -> None:
    p = argparse.ArgumentParser(description="Integrated EncFormer runner on Phantom engine")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--gpu-qkv", type=str, default=None)
    p.add_argument("--gpu-score", type=str, default=None)
    p.add_argument("--gpu-value", type=str, default=None)
    p.add_argument("--gpu-ffn", type=str, default=None)
    p.add_argument("--schedule", type=str, default="per-gpu", choices=["sequential", "per-gpu", "parallel"])
    p.add_argument("--max-workers", type=int, default=0, help="0 = auto")
    p.add_argument("--mode", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--thread-count", type=int, default=0)
    p.add_argument("--log-coeff-count", type=int, default=8)
    p.add_argument("--special-prime-count", type=int, default=2)
    p.add_argument("--qkv-log-coeff-count", type=int, default=None)
    p.add_argument("--score-log-coeff-count", type=int, default=None)
    p.add_argument("--value-log-coeff-count", type=int, default=None)
    p.add_argument("--ffn-log-coeff-count", type=int, default=None)
    p.add_argument("--qkv-special-prime-count", type=int, default=None)
    p.add_argument("--score-special-prime-count", type=int, default=None)
    p.add_argument("--value-special-prime-count", type=int, default=None)
    p.add_argument("--ffn-special-prime-count", type=int, default=None)
    p.add_argument("--galois-mode", type=str, default="")
    p.add_argument("--qkv-galois-mode", type=str, default=None)
    p.add_argument("--score-galois-mode", type=str, default=None)
    p.add_argument("--value-galois-mode", type=str, default=None)
    p.add_argument("--ffn-galois-mode", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--qkv-level", type=int, default=8)
    p.add_argument("--qkv-backend", type=str, default="native", choices=["python", "native"])
    p.add_argument("--score-backend", type=str, default="native", choices=["python", "native"])
    p.add_argument("--value-backend", type=str, default="native", choices=["python", "native"])
    p.add_argument("--ffn-backend", type=str, default="native", choices=["python", "native"])
    p.add_argument("--native-root", type=str, default=None)
    p.add_argument("--native-build-dir", type=str, default="build")
    p.add_argument("--native-no-build", action="store_true")
    p.add_argument("--qkv-native-root", type=str, default=None)
    p.add_argument("--qkv-native-build-dir", type=str, default="build")
    p.add_argument("--qkv-native-no-build", action="store_true")
    p.add_argument("--score-level", type=int, default=8)
    p.add_argument("--value-level", type=int, default=8)
    p.add_argument("--ffn-level", type=int, default=8)

    p.add_argument("--qkv-qk-c-used", type=int, default=qkv_sim.QK_C_USED)
    p.add_argument("--qkv-v-c-used", type=int, default=qkv_sim.V_C_USED)
    p.add_argument("--qkv-n1", type=int, default=qkv_sim.N1)
    p.add_argument("--score-case-rel-blocks", type=int, default=score_sim.CASE_REL_BLOCKS)
    p.add_argument("--score-case-rel-c-used", type=int, default=score_sim.CASE_REL_C_USED)

    p.add_argument("--value-kernel", type=str, default="baseline-nc", choices=["baseline-nc", "pair-direct", "complex"])
    p.add_argument("--value-sv-relin", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--value-sv-final-relin", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-out-force-real", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-out-pre-force-real", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--value-out-post-force-real", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--value-out-base-relin", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ffn-out-force-real", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ffn-kernel", type=str, default="pair", choices=["pair", "real"])
    p.add_argument("--ffn-n1-ff1", type=int, default=None)
    p.add_argument("--ffn-n1-ff2", type=int, default=None)

    p.add_argument(
        "--bridge-conversion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run end-to-end EncFormer with simulator-equivalent CKKS<->MPC bridge conversion logic.",
    )
    p.add_argument(
        "--bridge-level",
        type=int,
        default=None,
        help="Default CKKS level for bridge-conversion mode (default: backend default).",
    )
    p.add_argument(
        "--bridge-log-coeff-count", type=int, default=None, help="Coeff chain log count for bridge-conversion mode."
    )
    p.add_argument(
        "--bridge-special-prime-count", type=int, default=None, help="Special-prime count for bridge-conversion mode."
    )
    p.add_argument(
        "--bridge-galois-mode", type=str, default=None, help="PHANTOM_GALOIS_MODE override for bridge-conversion mode."
    )
    p.add_argument(
        "--mpc-engine",
        type=str,
        default=None,
        help="MPC engine override for bridge-conversion mode (e.g. plain, crypten).",
    )

    p.add_argument(
        "--segmented",
        action="store_true",
        help="Use per-stage fresh CKKS contexts with complex conversion at boundaries.",
    )
    p.add_argument("--no-cc", action="store_true", help="Disable complex-conjugate packing (wo_CC ablation)")
    p.add_argument("--no-scp", action="store_true", help="Disable stage-compatible patterns (wo_SCP ablation)")
    p.add_argument("--quick", action="store_true", help="Reduced-size smoke test")
    p.add_argument("--json", action="store_true", help="Emit a JSON summary block")
    args = p.parse_args()

    value_out_pre_force_real = (
        args.value_out_force_real if args.value_out_pre_force_real is None else bool(args.value_out_pre_force_real)
    )
    value_out_post_force_real = (
        args.value_out_force_real if args.value_out_post_force_real is None else bool(args.value_out_post_force_real)
    )

    qkv_log = args.log_coeff_count if args.qkv_log_coeff_count is None else int(args.qkv_log_coeff_count)
    score_log = args.log_coeff_count if args.score_log_coeff_count is None else int(args.score_log_coeff_count)
    value_log = args.log_coeff_count if args.value_log_coeff_count is None else int(args.value_log_coeff_count)
    ffn_log = args.log_coeff_count if args.ffn_log_coeff_count is None else int(args.ffn_log_coeff_count)

    qkv_sp = args.special_prime_count if args.qkv_special_prime_count is None else int(args.qkv_special_prime_count)
    score_sp = (
        args.special_prime_count if args.score_special_prime_count is None else int(args.score_special_prime_count)
    )
    value_sp = (
        args.special_prime_count if args.value_special_prime_count is None else int(args.value_special_prime_count)
    )
    ffn_sp = args.special_prime_count if args.ffn_special_prime_count is None else int(args.ffn_special_prime_count)

    qkv_galois = args.galois_mode if args.qkv_galois_mode is None else args.qkv_galois_mode
    score_galois = args.galois_mode if args.score_galois_mode is None else args.score_galois_mode
    value_galois = args.galois_mode if args.value_galois_mode is None else args.value_galois_mode
    ffn_galois = args.galois_mode if args.ffn_galois_mode is None else args.ffn_galois_mode

    shared_native_root = args.native_root
    shared_native_build_dir = args.native_build_dir
    shared_native_no_build = bool(args.native_no_build)

    qkv_native_root = args.qkv_native_root if args.qkv_native_root is not None else shared_native_root
    qkv_native_build_dir = args.qkv_native_build_dir if args.qkv_native_build_dir else shared_native_build_dir
    qkv_native_no_build = bool(args.qkv_native_no_build) or shared_native_no_build

    if args.bridge_conversion:
        if args.quick:
            raise ValueError("--quick is not supported with --bridge-conversion.")
        set_visible_gpus(args.gpu if args.mode == "gpu" else None)
        bridge_log = args.log_coeff_count if args.bridge_log_coeff_count is None else int(args.bridge_log_coeff_count)
        bridge_sp = (
            args.special_prime_count
            if args.bridge_special_prime_count is None
            else int(args.bridge_special_prime_count)
        )
        bridge_galois = args.galois_mode if args.bridge_galois_mode is None else str(args.bridge_galois_mode)
        _run_bridge_pipeline(
            args,
            log_coeff_count=bridge_log,
            special_prime_count=bridge_sp,
            galois_mode=bridge_galois,
            segmented=args.segmented,
        )
        return

    if any(b != "native" for b in (args.qkv_backend, args.score_backend, args.value_backend, args.ffn_backend)):
        raise ValueError("EncFormer_phantom is configured for native CUDA-only pipeline. Set all backends to 'native'.")

    if args.quick:
        score_blocks = 2
        score_c_used = 32
    else:
        score_blocks = args.score_case_rel_blocks
        score_c_used = args.score_case_rel_c_used

    gpu_qkv = args.gpu if args.gpu_qkv is None else args.gpu_qkv
    gpu_score = args.gpu if args.gpu_score is None else args.gpu_score
    gpu_value = args.gpu if args.gpu_value is None else args.gpu_value
    gpu_ffn = args.gpu if args.gpu_ffn is None else args.gpu_ffn

    qkv_same_as_shared = (
        str(qkv_native_root) == str(shared_native_root)
        and str(qkv_native_build_dir) == str(shared_native_build_dir)
        and bool(qkv_native_no_build) == bool(shared_native_no_build)
    )
    if qkv_same_as_shared:
        bin_map = resolve_native_binaries(
            targets=[
                "bench_ckks_qkv_full",
                "bench_ckks_score_full",
                "bench_ckks_value_full",
                "bench_ckks_ffn_full",
            ],
            phantom_root=shared_native_root,
            build_dir=shared_native_build_dir,
            build_if_needed=not shared_native_no_build,
        )
    else:
        bin_map = {}
        bin_map.update(
            resolve_native_binaries(
                targets=["bench_ckks_qkv_full"],
                phantom_root=qkv_native_root,
                build_dir=qkv_native_build_dir,
                build_if_needed=not qkv_native_no_build,
            )
        )
        bin_map.update(
            resolve_native_binaries(
                targets=["bench_ckks_score_full", "bench_ckks_value_full", "bench_ckks_ffn_full"],
                phantom_root=shared_native_root,
                build_dir=shared_native_build_dir,
                build_if_needed=not shared_native_no_build,
            )
        )

    def run_qkv() -> Dict[str, Any]:
        return run_qkv_full_native(
            gpu=gpu_qkv if args.mode == "gpu" else None,
            binary_path=bin_map["bench_ckks_qkv_full"],
        )

    def run_score() -> Dict[str, Any]:
        return run_score_full_native(
            gpu=gpu_score if args.mode == "gpu" else None,
            binary_path=bin_map["bench_ckks_score_full"],
        )

    def run_value() -> Dict[str, Any]:
        return run_value_full_native(
            gpu=gpu_value if args.mode == "gpu" else None,
            binary_path=bin_map["bench_ckks_value_full"],
        )

    def run_ffn() -> Dict[str, Any]:
        return run_ffn_full_native(
            gpu=gpu_ffn if args.mode == "gpu" else None,
            binary_path=bin_map["bench_ckks_ffn_full"],
        )

    stage_specs = [
        ("qkv", "QKV", _gpu_key(gpu_qkv), run_qkv),
        ("score", "Score", _gpu_key(gpu_score), run_score),
        ("value", "Value", _gpu_key(gpu_value), run_value),
        ("ffn", "FFN", _gpu_key(gpu_ffn), run_ffn),
    ]

    t0 = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}

    if args.schedule == "sequential":
        for key, _label, _gpu, fn in stage_specs:
            results[key] = fn()
    elif args.schedule == "parallel":
        max_workers = int(args.max_workers) if args.max_workers > 0 else len(stage_specs)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(stage_specs)))) as ex:
            fut_map = {ex.submit(fn): key for key, _label, _gpu, fn in stage_specs}
            for fut in as_completed(fut_map):
                results[fut_map[fut]] = fut.result()
    else:
        groups: Dict[str, List[tuple[str, str, Any]]] = {}
        for key, label, gpu_key, fn in stage_specs:
            groups.setdefault(gpu_key, []).append((key, label, fn))
        max_workers = int(args.max_workers) if args.max_workers > 0 else len(groups)

        def run_group(items: List[tuple[str, str, Any]]) -> List[tuple[str, Dict[str, Any]]]:
            out: List[tuple[str, Dict[str, Any]]] = []
            for key, _label, fn in items:
                out.append((key, fn()))
            return out

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(groups)))) as ex:
            futs = [ex.submit(run_group, items) for items in groups.values()]
            for fut in as_completed(futs):
                for key, out in fut.result():
                    results[key] = out

    t1 = time.perf_counter()

    qkv = results["qkv"]
    score = results["score"]
    value = results["value"]
    ffn = results["ffn"]

    _stage_hdr("QKV", float(qkv["elapsed_sec"]))
    _stage_hdr("Score", float(score["elapsed_sec"]))
    _stage_hdr("Value", float(value["elapsed_sec"]))
    _stage_hdr("FFN", float(ffn["elapsed_sec"]))

    ffn_ks = _sum_ks(ffn["ff1_stats"], ffn["ff2_stats"])
    total_ks = _sum_ks(
        qkv["stats"],
        score["stats"],
        value["value_stats"],
        value["out_stats"],
        ffn_ks,
    )

    sum_stage_elapsed = (
        float(qkv["elapsed_sec"])
        + float(score["elapsed_sec"])
        + float(value["elapsed_sec"])
        + float(ffn["elapsed_sec"])
    )
    total_wall = t1 - t0

    print("\n=== EncFormer_phantom ===")
    print("status: PASS")
    print(f"schedule: {args.schedule}")
    print(f"total_wall_elapsed: {total_wall:.3f}s")
    print(f"sum_stage_elapsed: {sum_stage_elapsed:.3f}s")
    print(f"overhead: {total_wall - sum_stage_elapsed:.3f}s")
    print(f"ks_total: rot={total_ks['ks_rots']} mul={total_ks['ks_muls_ctct']} conj={total_ks['ks_conj']}")

    if args.json:
        report: Dict[str, Any] = {
            "config": {
                "gpu": args.gpu,
                "gpu_qkv": gpu_qkv,
                "gpu_score": gpu_score,
                "gpu_value": gpu_value,
                "gpu_ffn": gpu_ffn,
                "schedule": args.schedule,
                "mode": args.mode,
                "seed": args.seed,
                "qkv_level": args.qkv_level,
                "qkv_backend": args.qkv_backend,
                "score_backend": args.score_backend,
                "value_backend": args.value_backend,
                "ffn_backend": args.ffn_backend,
                "qkv_native_root": args.qkv_native_root,
                "qkv_native_build_dir": args.qkv_native_build_dir,
                "native_root": args.native_root,
                "native_build_dir": args.native_build_dir,
                "native_no_build": args.native_no_build,
                "score_level": args.score_level,
                "value_level": args.value_level,
                "ffn_level": args.ffn_level,
                "qkv_log_coeff_count": qkv_log,
                "score_log_coeff_count": score_log,
                "value_log_coeff_count": value_log,
                "ffn_log_coeff_count": ffn_log,
                "qkv_special_prime_count": qkv_sp,
                "score_special_prime_count": score_sp,
                "value_special_prime_count": value_sp,
                "ffn_special_prime_count": ffn_sp,
                "qkv_galois_mode": qkv_galois,
                "score_galois_mode": score_galois,
                "value_galois_mode": value_galois,
                "ffn_galois_mode": ffn_galois,
                "qkv_qk_c_used": args.qkv_qk_c_used,
                "qkv_v_c_used": args.qkv_v_c_used,
                "qkv_n1": args.qkv_n1,
                "score_case_rel_blocks": score_blocks,
                "score_case_rel_c_used": score_c_used,
                "value_kernel": args.value_kernel,
                "value_sv_relin": args.value_sv_relin,
                "value_sv_final_relin": args.value_sv_final_relin,
                "value_out_force_real": args.value_out_force_real,
                "value_out_pre_force_real": value_out_pre_force_real,
                "value_out_post_force_real": value_out_post_force_real,
                "value_out_base_relin": args.value_out_base_relin,
                "ffn_out_force_real": args.ffn_out_force_real,
                "ffn_kernel": args.ffn_kernel,
                "ffn_n1_ff1": args.ffn_n1_ff1,
                "ffn_n1_ff2": args.ffn_n1_ff2,
                "quick": args.quick,
            },
            "stage_elapsed": {
                "qkv": float(qkv["elapsed_sec"]),
                "score": float(score["elapsed_sec"]),
                "value": float(value["elapsed_sec"]),
                "ffn": float(ffn["elapsed_sec"]),
            },
            "stage_rel_err": {
                "qkv": float(qkv["rel_err"]),
                "score": float(score["rel_err"]),
                "value": float(value["value_rel_err"]),
                "out": None if value["out_rel_err"] is None else float(value["out_rel_err"]),
                "ff1": float(ffn["ff1_err"]),
                "ff2": float(ffn["ff2_err"]),
            },
            "ks": {
                "qkv": qkv["stats"],
                "score": score["stats"],
                "value": value["value_stats"],
                "out": value["out_stats"],
                "ff1": ffn["ff1_stats"],
                "ff2": ffn["ff2_stats"],
                "total": total_ks,
            },
            "totals": {
                "total_wall_elapsed": total_wall,
                "sum_stage_elapsed": sum_stage_elapsed,
                "overhead": total_wall - sum_stage_elapsed,
            },
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
