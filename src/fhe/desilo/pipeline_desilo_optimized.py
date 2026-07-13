from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Sequence

_ELAPSED_RE = re.compile(r"elapsed=([0-9]+(?:\.[0-9]+)?)s")


@dataclass
class RunSpec:
    name: str
    cmd: List[str]
    order: int
    gpu: str


@dataclass
class RunResult:
    name: str
    cmd: List[str]
    returncode: int
    elapsed_sec: float
    stdout: str
    stderr: str
    order: int
    gpu: str


def _run_and_capture(spec: RunSpec) -> RunResult:
    t0 = time.perf_counter()
    p = subprocess.run(spec.cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    return RunResult(
        name=spec.name,
        cmd=list(spec.cmd),
        returncode=p.returncode,
        elapsed_sec=t1 - t0,
        stdout=p.stdout,
        stderr=p.stderr,
        order=spec.order,
        gpu=spec.gpu,
    )


def _run_group(specs: Sequence[RunSpec]) -> List[RunResult]:
    out: List[RunResult] = []
    for spec in specs:
        rr = _run_and_capture(spec)
        out.append(rr)
        if rr.returncode != 0:
            break
    return out


def _module_elapsed(stdout: str) -> float | None:
    m = _ELAPSED_RE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


def _print_result(rr: RunResult) -> None:
    print(f"\n=== {rr.name} ===")
    print("cmd:", " ".join(rr.cmd))
    print(f"gpu: {rr.gpu}")
    print(f"returncode: {rr.returncode}")
    print(f"wall_elapsed: {rr.elapsed_sec:.3f}s")
    m_elapsed = _module_elapsed(rr.stdout)
    if m_elapsed is not None:
        print(f"module_elapsed: {m_elapsed:.3f}s")
    if rr.stdout.strip():
        print("stdout:")
        print(rr.stdout.rstrip())
    if rr.stderr.strip():
        print("stderr:")
        print(rr.stderr.rstrip())


def _gpu_key(gpu: str) -> str:
    g = str(gpu).strip()
    return g if g else "__none__"


def main() -> None:
    p = argparse.ArgumentParser(description="Optimized multi-GPU Desilo pipeline runner")
    p.add_argument("--gpu-main", type=str, default="2", help="Default GPU for QKV+Score+Value")
    p.add_argument("--gpu-aux", type=str, default="3", help="Default GPU for FFN")
    p.add_argument("--gpu-qkv", type=str, default=None)
    p.add_argument("--gpu-score", type=str, default=None)
    p.add_argument("--gpu-value", type=str, default=None)
    p.add_argument("--gpu-ffn", type=str, default=None)
    p.add_argument("--schedule", type=str, default="per-gpu", choices=["sequential", "per-gpu", "parallel"])
    p.add_argument("--max-workers", type=int, default=0, help="0 = auto")
    p.add_argument("--mode", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--thread-count", type=int, default=0)
    p.add_argument("--log-coeff-count", type=int, default=15)
    p.add_argument("--special-prime-count", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qkv-level", type=int, default=5)
    p.add_argument("--score-level", type=int, default=10)
    p.add_argument("--value-level", type=int, default=8)
    p.add_argument("--ffn-level", type=int, default=4)
    p.add_argument("--qkv-qk-c-used", type=int, default=128)
    p.add_argument("--qkv-v-c-used", type=int, default=128)
    p.add_argument("--qkv-n1", type=int, default=16)
    p.add_argument("--score-case-rel-blocks", type=int, default=7)
    p.add_argument("--score-case-rel-c-used", type=int, default=120)
    p.add_argument("--value-kernel", type=str, default="pair-direct", choices=["baseline-nc", "pair-direct", "complex"])
    p.add_argument("--value-sv-relin", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--value-sv-final-relin", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-out-force-real", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-out-pre-force-real", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--value-out-post-force-real", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--value-out-base-relin", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ffn-out-force-real", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    value_out_pre_force_real = (
        args.value_out_force_real if args.value_out_pre_force_real is None else bool(args.value_out_pre_force_real)
    )
    value_out_post_force_real = (
        args.value_out_force_real if args.value_out_post_force_real is None else bool(args.value_out_post_force_real)
    )

    gpu_qkv = args.gpu_main if args.gpu_qkv is None else args.gpu_qkv
    gpu_score = args.gpu_main if args.gpu_score is None else args.gpu_score
    gpu_value = args.gpu_main if args.gpu_value is None else args.gpu_value
    gpu_ffn = args.gpu_aux if args.gpu_ffn is None else args.gpu_ffn

    py = sys.executable

    def base_cmd(mod: str, gpu: str, level: int) -> List[str]:
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
            str(args.log_coeff_count),
            "--special-prime-count",
            str(args.special_prime_count),
            "--seed",
            str(args.seed),
            "--fixed-level",
            str(level),
        ]

    runs: List[RunSpec] = [
        RunSpec(
            name="QKV",
            cmd=base_cmd("src.fhe.desilo.qkv_desilo", gpu_qkv, args.qkv_level)
            + ["--qk-c-used", str(args.qkv_qk_c_used), "--v-c-used", str(args.qkv_v_c_used), "--n1", str(args.qkv_n1)],
            order=0,
            gpu=_gpu_key(gpu_qkv),
        ),
        RunSpec(
            name="Score",
            cmd=base_cmd("src.fhe.desilo.score_desilo", gpu_score, args.score_level)
            + [
                "--case-rel-blocks",
                str(args.score_case_rel_blocks),
                "--case-rel-c-used",
                str(args.score_case_rel_c_used),
            ],
            order=1,
            gpu=_gpu_key(gpu_score),
        ),
        RunSpec(
            name="Value",
            cmd=base_cmd("src.fhe.desilo.value_desilo", gpu_value, args.value_level)
            + ["--value-kernel", args.value_kernel]
            + (["--sv-relin"] if args.value_sv_relin else ["--no-sv-relin"])
            + (["--sv-final-relin"] if args.value_sv_final_relin else ["--no-sv-final-relin"])
            + (["--out-force-real"] if args.value_out_force_real else ["--no-out-force-real"])
            + (["--out-pre-force-real"] if value_out_pre_force_real else ["--no-out-pre-force-real"])
            + (["--out-post-force-real"] if value_out_post_force_real else ["--no-out-post-force-real"])
            + (["--out-base-relin"] if args.value_out_base_relin else ["--no-out-base-relin"]),
            order=2,
            gpu=_gpu_key(gpu_value),
        ),
        RunSpec(
            name="FFN",
            cmd=base_cmd("src.fhe.desilo.ffn_desilo", gpu_ffn, args.ffn_level)
            + (["--out-force-real"] if args.ffn_out_force_real else ["--no-out-force-real"]),
            order=3,
            gpu=_gpu_key(gpu_ffn),
        ),
    ]

    t_all0 = time.perf_counter()
    results: List[RunResult] = []

    if args.schedule == "sequential":
        results = _run_group(runs)
    elif args.schedule == "parallel":
        max_workers = int(args.max_workers) if args.max_workers > 0 else len(runs)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(runs)))) as ex:
            fut_map = {ex.submit(_run_and_capture, spec): spec for spec in runs}
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

    total_wall = time.perf_counter() - t_all0

    for rr in results:
        _print_result(rr)

    ok = all(r.returncode == 0 for r in results)
    print("\n=== Summary ===")
    print(f"overall_status: {'PASS' if ok else 'FAIL'}")
    print(f"schedule: {args.schedule}")
    print(f"total_wall_elapsed: {total_wall:.3f}s")

    if results:
        seq_sum = 0.0
        for rr in results:
            m_elapsed = _module_elapsed(rr.stdout)
            if m_elapsed is not None:
                seq_sum += m_elapsed
        if seq_sum > 0.0:
            print(f"sum_of_module_elapsed (non-overlapped): {seq_sum:.3f}s")


if __name__ == "__main__":
    main()
