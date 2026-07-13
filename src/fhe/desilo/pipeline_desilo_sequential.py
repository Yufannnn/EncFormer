from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RunResult:
    name: str
    cmd: List[str]
    returncode: int
    wall_elapsed: float
    stdout: str
    stderr: str


def _run(name: str, cmd: Sequence[str]) -> RunResult:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    return RunResult(
        name=name, cmd=list(cmd), returncode=p.returncode, wall_elapsed=t1 - t0, stdout=p.stdout, stderr=p.stderr
    )


def _parse_elapsed(stdout: str) -> float | None:
    m = re.search(r"elapsed=([0-9]+(?:\.[0-9]+)?)s", stdout)
    if not m:
        return None
    return float(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description="Sequential single-GPU Desilo pipeline runner")
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
    p.add_argument("--value-kernel", type=str, default="pair-direct", choices=["baseline-nc", "pair-direct", "complex"])
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

    py = sys.executable

    def base_cmd(mod: str, level: int) -> List[str]:
        return [
            py,
            "-m",
            mod,
            "--gpu",
            args.gpu,
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

    runs = [
        (
            "QKV",
            base_cmd("src.fhe.desilo.qkv_desilo", args.qkv_level)
            + ["--qk-c-used", str(args.qkv_qk_c_used), "--v-c-used", str(args.qkv_v_c_used), "--n1", str(args.qkv_n1)],
        ),
        (
            "Score",
            base_cmd("src.fhe.desilo.score_desilo", args.score_level)
            + [
                "--case-rel-blocks",
                str(args.score_case_rel_blocks),
                "--case-rel-c-used",
                str(args.score_case_rel_c_used),
            ],
        ),
        (
            "Value",
            base_cmd("src.fhe.desilo.value_desilo", args.value_level)
            + ["--value-kernel", args.value_kernel]
            + (["--sv-relin"] if args.value_sv_relin else ["--no-sv-relin"])
            + (["--sv-final-relin"] if args.value_sv_final_relin else ["--no-sv-final-relin"])
            + (["--out-force-real"] if args.value_out_force_real else ["--no-out-force-real"])
            + (["--out-pre-force-real"] if value_out_pre_force_real else ["--no-out-pre-force-real"])
            + (["--out-post-force-real"] if value_out_post_force_real else ["--no-out-post-force-real"])
            + (["--out-base-relin"] if args.value_out_base_relin else ["--no-out-base-relin"]),
        ),
        (
            "FFN",
            base_cmd("src.fhe.desilo.ffn_desilo", args.ffn_level)
            + (["--out-force-real"] if args.ffn_out_force_real else ["--no-out-force-real"]),
        ),
    ]

    all_t0 = time.perf_counter()
    results: List[RunResult] = []
    for name, cmd in runs:
        rr = _run(name, cmd)
        results.append(rr)
        if rr.returncode != 0:
            break
    all_t1 = time.perf_counter()

    print("=== Sequential Desilo ===")
    ok = True
    mod_elapsed_sum = 0.0
    for rr in results:
        m_elapsed = _parse_elapsed(rr.stdout)
        if m_elapsed is not None:
            mod_elapsed_sum += m_elapsed
        ok = ok and rr.returncode == 0
        print(f"\\n[{rr.name}] rc={rr.returncode} wall={rr.wall_elapsed:.3f}s module_elapsed={m_elapsed}")
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
