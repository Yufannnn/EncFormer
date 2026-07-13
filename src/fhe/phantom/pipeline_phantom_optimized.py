from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.fhe.phantom.phantom_native_bench import resolve_native_binaries

_ELAPSED_S_RE = re.compile(r"elapsed=([0-9]+(?:\.[0-9]+)?)s")
_ELAPSED_MS_RE = re.compile(r"(?m)^\s*elapsed_ms\s+([-+]?[0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*$")


@dataclass
class RunSpec:
    name: str
    cmd: List[str]
    env: Optional[Dict[str, str]]
    order: int
    gpu: str


@dataclass
class RunResult:
    name: str
    cmd: List[str]
    returncode: int
    wall_elapsed: float
    stdout: str
    stderr: str
    order: int
    gpu: str


def _parse_num(s: str) -> float:
    return float(s.replace(",", "").strip())


def _run_env(spec: RunSpec) -> RunResult:
    base_env = os.environ.copy()
    if spec.env:
        base_env.update(spec.env)
    t0 = time.perf_counter()
    p = subprocess.run(spec.cmd, capture_output=True, text=True, env=base_env)
    t1 = time.perf_counter()
    return RunResult(
        name=spec.name,
        cmd=list(spec.cmd),
        returncode=p.returncode,
        wall_elapsed=t1 - t0,
        stdout=p.stdout,
        stderr=p.stderr,
        order=spec.order,
        gpu=spec.gpu,
    )


def _run_group(specs: Sequence[RunSpec]) -> List[RunResult]:
    out: List[RunResult] = []
    for spec in specs:
        rr = _run_env(spec)
        out.append(rr)
        if rr.returncode != 0:
            break
    return out


def _parse_elapsed(stdout: str) -> float | None:
    m = _ELAPSED_S_RE.search(stdout)
    if m:
        return float(m.group(1))
    m = _ELAPSED_MS_RE.search(stdout)
    if m:
        return _parse_num(m.group(1)) / 1000.0
    return None


def _gpu_key(gpu: str) -> str:
    g = str(gpu).strip()
    return g if g else "__none__"


def main() -> None:
    p = argparse.ArgumentParser(description="Optimized multi-GPU Phantom pipeline runner")
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
    p.add_argument("--qkv-qk-c-used", type=int, default=128)
    p.add_argument("--qkv-v-c-used", type=int, default=128)
    p.add_argument("--qkv-n1", type=int, default=16)
    p.add_argument("--score-case-rel-blocks", type=int, default=7)
    p.add_argument("--score-case-rel-c-used", type=int, default=120)
    p.add_argument(
        "--ffn-out-force-real",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to force ct_real on FFN folded outputs before decrypt/eval.",
    )
    p.add_argument("--ffn-kernel", type=str, default="pair", choices=["pair", "real"])
    p.add_argument("--ffn-n1-ff1", type=int, default=None)
    p.add_argument("--ffn-n1-ff2", type=int, default=None)
    p.add_argument("--value-kernel", type=str, default="baseline-nc", choices=["baseline-nc", "pair-direct", "complex"])
    p.add_argument(
        "--value-sv-relin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to relinearize each SV ct*ct multiply in Value baseline-nc.",
    )
    p.add_argument(
        "--value-sv-final-relin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to relinearize accumulated SV output when --no-value-sv-relin.",
    )
    p.add_argument(
        "--value-out-force-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to force ct_real before/after OUT projection in Value.",
    )
    p.add_argument(
        "--value-out-pre-force-real",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to force ct_real before OUT pairing (overrides value-out-force-real).",
    )
    p.add_argument(
        "--value-out-post-force-real",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to force ct_real after OUT fold (overrides value-out-force-real).",
    )
    p.add_argument(
        "--value-out-base-relin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to relinearize base OUT ciphertexts after pairing in Value.",
    )
    args = p.parse_args()

    value_out_pre_force_real = (
        args.value_out_force_real if args.value_out_pre_force_real is None else bool(args.value_out_pre_force_real)
    )
    value_out_post_force_real = (
        args.value_out_force_real if args.value_out_post_force_real is None else bool(args.value_out_post_force_real)
    )

    gpu_qkv = args.gpu if args.gpu_qkv is None else args.gpu_qkv
    gpu_score = args.gpu if args.gpu_score is None else args.gpu_score
    gpu_value = args.gpu if args.gpu_value is None else args.gpu_value
    gpu_ffn = args.gpu if args.gpu_ffn is None else args.gpu_ffn

    py = sys.executable

    def base_cmd(mod: str, level: int, gpu: str, log_coeff_count: int, special_prime_count: int) -> List[str]:
        return [
            py,
            "-m",
            mod,
            "--gpu",
            gpu,
            "--mode",
            args.mode,
            "--thread-count",
            str(args.thread_count),
            "--log-coeff-count",
            str(log_coeff_count),
            "--special-prime-count",
            str(special_prime_count),
            "--seed",
            str(args.seed),
            "--fixed-level",
            str(level),
        ]

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

    if any(b != "native" for b in (args.qkv_backend, args.score_backend, args.value_backend, args.ffn_backend)):
        raise ValueError(
            "pipeline_phantom_optimized is configured for native CUDA-only execution. Set all backends to 'native'."
        )

    runs: List[RunSpec] = []

    if args.qkv_backend == "native":
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

        def native_env(gpu: str) -> Optional[Dict[str, str]]:
            if args.mode != "gpu":
                return None
            return {"CUDA_VISIBLE_DEVICES": str(gpu)}

        runs = [
            RunSpec("QKV", [bin_map["bench_ckks_qkv_full"]], native_env(gpu_qkv), 0, _gpu_key(gpu_qkv)),
            RunSpec("Score", [bin_map["bench_ckks_score_full"]], native_env(gpu_score), 1, _gpu_key(gpu_score)),
            RunSpec("Value", [bin_map["bench_ckks_value_full"]], native_env(gpu_value), 2, _gpu_key(gpu_value)),
            RunSpec("FFN", [bin_map["bench_ckks_ffn_full"]], native_env(gpu_ffn), 3, _gpu_key(gpu_ffn)),
        ]
    else:
        qkv_cmd = base_cmd("src.fhe.phantom.qkv_phantom", args.qkv_level, gpu_qkv, qkv_log, qkv_sp) + [
            "--qk-c-used",
            str(args.qkv_qk_c_used),
            "--v-c-used",
            str(args.qkv_v_c_used),
            "--n1",
            str(args.qkv_n1),
        ]
        score_cmd = base_cmd("src.fhe.phantom.score_phantom", args.score_level, gpu_score, score_log, score_sp) + [
            "--case-rel-blocks",
            str(args.score_case_rel_blocks),
            "--case-rel-c-used",
            str(args.score_case_rel_c_used),
        ]
        value_cmd = (
            base_cmd("src.fhe.phantom.value_phantom", args.value_level, gpu_value, value_log, value_sp)
            + ["--value-kernel", args.value_kernel]
            + (["--sv-relin"] if args.value_sv_relin else ["--no-sv-relin"])
            + (["--sv-final-relin"] if args.value_sv_final_relin else ["--no-sv-final-relin"])
            + (["--out-force-real"] if args.value_out_force_real else ["--no-out-force-real"])
            + (["--out-pre-force-real"] if value_out_pre_force_real else ["--no-out-pre-force-real"])
            + (["--out-post-force-real"] if value_out_post_force_real else ["--no-out-post-force-real"])
            + (["--out-base-relin"] if args.value_out_base_relin else ["--no-out-base-relin"])
        )
        ffn_cmd = (
            base_cmd("src.fhe.phantom.ffn_phantom", args.ffn_level, gpu_ffn, ffn_log, ffn_sp)
            + (["--out-force-real"] if args.ffn_out_force_real else ["--no-out-force-real"])
            + ["--kernel", args.ffn_kernel]
            + ([] if args.ffn_n1_ff1 is None else ["--n1-ff1", str(args.ffn_n1_ff1)])
            + ([] if args.ffn_n1_ff2 is None else ["--n1-ff2", str(args.ffn_n1_ff2)])
        )
        qkv_env = {"PHANTOM_GALOIS_MODE": str(qkv_galois)} if qkv_galois else None
        score_env = {"PHANTOM_GALOIS_MODE": str(score_galois)} if score_galois else None
        value_env = {"PHANTOM_GALOIS_MODE": str(value_galois)} if value_galois else None
        ffn_env = {"PHANTOM_GALOIS_MODE": str(ffn_galois)} if ffn_galois else None
        runs = [
            RunSpec("QKV", qkv_cmd, qkv_env, 0, _gpu_key(gpu_qkv)),
            RunSpec("Score", score_cmd, score_env, 1, _gpu_key(gpu_score)),
            RunSpec("Value", value_cmd, value_env, 2, _gpu_key(gpu_value)),
            RunSpec("FFN", ffn_cmd, ffn_env, 3, _gpu_key(gpu_ffn)),
        ]

    all_t0 = time.perf_counter()
    results: List[RunResult] = []

    if args.schedule == "sequential":
        results = _run_group(runs)
    elif args.schedule == "parallel":
        max_workers = int(args.max_workers) if args.max_workers > 0 else len(runs)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(runs)))) as ex:
            fut_map = {ex.submit(_run_env, spec): spec for spec in runs}
            for fut in as_completed(fut_map):
                results.append(fut.result())
        results.sort(key=lambda r: r.order)
    else:
        groups: Dict[str, List[RunSpec]] = {}
        for spec in runs:
            groups.setdefault(spec.gpu, []).append(spec)
        max_workers = int(args.max_workers) if args.max_workers > 0 else len(groups)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(groups)))) as ex:
            futs = [ex.submit(_run_group, specs) for specs in groups.values()]
            for fut in as_completed(futs):
                results.extend(fut.result())
        results.sort(key=lambda r: r.order)

    all_t1 = time.perf_counter()

    print("=== Optimized Phantom Pipeline ===")
    print(f"schedule: {args.schedule}")
    ok = True
    mod_elapsed_sum = 0.0
    for rr in results:
        m_elapsed = _parse_elapsed(rr.stdout)
        if m_elapsed is not None:
            mod_elapsed_sum += m_elapsed
        ok = ok and rr.returncode == 0
        print(f"\n[{rr.name}] gpu={rr.gpu} rc={rr.returncode} wall={rr.wall_elapsed:.3f}s module_elapsed={m_elapsed}")
        if rr.stdout.strip():
            print(rr.stdout.rstrip())
        if rr.stderr.strip():
            print(rr.stderr.rstrip())

    print("\n=== Summary ===")
    print(f"status: {'PASS' if ok else 'FAIL'}")
    print(f"total_wall_elapsed: {all_t1 - all_t0:.3f}s")
    print(f"sum_module_elapsed: {mod_elapsed_sum:.3f}s")


if __name__ == "__main__":
    main()
