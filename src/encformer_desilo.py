from __future__ import annotations

import argparse
import contextlib
import json
import time
from typing import Any, Dict

import src.fhe.simulator.ffn as ffn_sim
import src.fhe.simulator.qkv as qkv_sim
import src.fhe.simulator.score as score_sim
import src.fhe.simulator.value as value_sim
from src.fhe.desilo.desilo_runtime import LevelCKKSContext, set_visible_gpus
from src.fhe.desilo.ffn_desilo import run_once as run_ffn_once
from src.fhe.desilo.qkv_desilo import run_once as run_qkv_once
from src.fhe.desilo.score_desilo import run_once as run_score_once
from src.fhe.desilo.value_desilo import run_once as run_value_once


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


def _run_bridge_pipeline(args, *, segmented: bool = False) -> None:
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
        del nslots
        import src.encformer as core

        seg_level = getattr(core, "_SEG_ENC_LEVEL", None)
        enc_level = (
            seg_level if seg_level is not None else (None if args.bridge_level is None else int(args.bridge_level))
        )
        return LevelCKKSContext(
            default_enc_level=enc_level,
            mode=args.mode,
            thread_count=args.thread_count,
            log_coeff_count=args.log_coeff_count,
            special_prime_count=args.special_prime_count,
        )

    use_cc = not getattr(args, "no_cc", False)
    use_scp = not getattr(args, "no_scp", False)
    with _patch_core_ctx(seed=args.seed, ctx_factory=_ctx_factory) as core:
        try:
            core.main(
                use_cc=use_cc, use_scp=use_scp, mpc_engine=args.mpc_engine, segmented=segmented, enc_levels=enc_levels
            )
        except Exception as exc:
            msg = str(exc)
            if "positive level" in msg.lower():
                raise RuntimeError(
                    "Desilo bridge-conversion mode exhausted CKKS levels in full end-to-end run. "
                    "Increase chain capacity if your build allows it, or use the staged runner mode."
                ) from exc
            raise


def main() -> None:
    p = argparse.ArgumentParser(description="Integrated EncFormer runner on Desilo engine")
    p.add_argument("--gpu", type=str, default="2")
    p.add_argument("--mode", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--thread-count", type=int, default=0)
    p.add_argument("--log-coeff-count", type=int, default=15)
    p.add_argument("--special-prime-count", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--qkv-level", type=int, default=5)
    p.add_argument("--score-level", type=int, default=10)
    p.add_argument("--value-level", type=int, default=8)
    p.add_argument("--ffn-level", type=int, default=4)

    p.add_argument("--qkv-qk-c-used", type=int, default=qkv_sim.QK_C_USED)
    p.add_argument("--qkv-v-c-used", type=int, default=qkv_sim.V_C_USED)
    p.add_argument("--qkv-n1", type=int, default=qkv_sim.N1)
    p.add_argument("--score-case-rel-blocks", type=int, default=score_sim.CASE_REL_BLOCKS)
    p.add_argument("--score-case-rel-c-used", type=int, default=score_sim.CASE_REL_C_USED)

    p.add_argument("--value-kernel", type=str, default="pair-direct", choices=["baseline-nc", "pair-direct", "complex"])
    p.add_argument("--value-sv-relin", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--value-sv-final-relin", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-out-force-real", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-out-pre-force-real", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--value-out-post-force-real", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--value-out-base-relin", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ffn-out-force-real", action=argparse.BooleanOptionalAction, default=False)

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

    set_visible_gpus(args.gpu if args.mode == "gpu" else None)

    if args.bridge_conversion:
        if args.quick:
            raise ValueError("--quick is not supported with --bridge-conversion.")
        _run_bridge_pipeline(args, segmented=args.segmented)
        return

    if args.quick:
        qkv_d2 = 64
        score_d = 64
        score_h = 4
        score_blocks = 2
        score_c_used = 32
        ffn_d1 = 256
        ffn_dmid = 512
        ffn_d2 = 256
        value_run_groups = 2
        value_run_out = False
    else:
        qkv_d2 = qkv_sim.D2
        score_d = score_sim.D
        score_h = score_sim.H
        score_blocks = args.score_case_rel_blocks
        score_c_used = args.score_case_rel_c_used
        ffn_d1 = ffn_sim.D1
        ffn_dmid = ffn_sim.DMID
        ffn_d2 = ffn_sim.D2
        value_run_groups = None
        value_run_out = True

    t0 = time.perf_counter()

    qkv = run_qkv_once(
        enc_level=args.qkv_level,
        mode=args.mode,
        thread_count=args.thread_count,
        log_coeff_count=args.log_coeff_count,
        special_prime_count=args.special_prime_count,
        seed=args.seed,
        d2=qkv_d2,
        qk_c_used=args.qkv_qk_c_used,
        v_c_used=args.qkv_v_c_used,
        n1=args.qkv_n1,
    )
    _stage_hdr("QKV", float(qkv["elapsed_sec"]))

    score = run_score_once(
        enc_level=args.score_level,
        mode=args.mode,
        thread_count=args.thread_count,
        log_coeff_count=args.log_coeff_count,
        special_prime_count=args.special_prime_count,
        seed=args.seed,
        d=score_d,
        h=score_h,
        case_rel_blocks=score_blocks,
        case_rel_c_used=score_c_used,
    )
    _stage_hdr("Score", float(score["elapsed_sec"]))

    value = run_value_once(
        enc_level=args.value_level,
        mode=args.mode,
        thread_count=args.thread_count,
        log_coeff_count=args.log_coeff_count,
        special_prime_count=args.special_prime_count,
        seed=args.seed,
        run_groups=value_run_groups,
        run_out=value_run_out,
        value_kernel=args.value_kernel,
        sv_relin=args.value_sv_relin,
        sv_final_relin=args.value_sv_final_relin,
        out_pre_force_real=value_out_pre_force_real,
        out_post_force_real=value_out_post_force_real,
        out_base_relin=args.value_out_base_relin,
    )
    _stage_hdr("Value", float(value["elapsed_sec"]))

    ffn = run_ffn_once(
        enc_level=args.ffn_level,
        mode=args.mode,
        thread_count=args.thread_count,
        log_coeff_count=args.log_coeff_count,
        special_prime_count=args.special_prime_count,
        seed=args.seed,
        d1=ffn_d1,
        dmid=ffn_dmid,
        d2=ffn_d2,
        out_force_real=args.ffn_out_force_real,
    )
    _stage_hdr("FFN", float(ffn["elapsed_sec"]))

    t1 = time.perf_counter()

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

    print("\n=== EncFormer_desilo ===")
    print(f"status: PASS")
    print(f"total_wall_elapsed: {total_wall:.3f}s")
    print(f"sum_stage_elapsed: {sum_stage_elapsed:.3f}s")
    print(f"overhead: {total_wall - sum_stage_elapsed:.3f}s")
    print(f"ks_total: rot={total_ks['ks_rots']} mul={total_ks['ks_muls_ctct']} conj={total_ks['ks_conj']}")

    if args.json:
        report: Dict[str, Any] = {
            "config": {
                "gpu": args.gpu,
                "mode": args.mode,
                "seed": args.seed,
                "qkv_level": args.qkv_level,
                "score_level": args.score_level,
                "value_level": args.value_level,
                "ffn_level": args.ffn_level,
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
