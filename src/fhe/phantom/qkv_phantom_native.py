from __future__ import annotations

import argparse

from src.fhe.phantom.phantom_native_bench import run_qkv_full_native
from src.utils import print_stage


def main() -> None:
    p = argparse.ArgumentParser(description="Run native C++ CUDA Phantom QKV full benchmark")
    p.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--phantom-root", type=str, default=None, help="Path to Phantom C++ source root")
    p.add_argument("--build-dir", type=str, default="build", help="Build directory under Phantom root")
    p.add_argument("--no-build", action="store_true", help="Do not build binary if missing")
    args = p.parse_args()

    out = run_qkv_full_native(
        gpu=args.gpu,
        phantom_root=args.phantom_root,
        build_dir=args.build_dir,
        build_if_needed=not args.no_build,
    )

    stats = out["stats"]
    print_stage(
        "QKV-nat",
        ct_in=6,
        ct_out=None,
        ks_rots=stats.get("ks_rots"),
        ks_muls=stats.get("ks_muls_ctct"),
        ks_conj=stats.get("ks_conj"),
        rel_err=out.get("rel_err"),
    )
    print(
        f"[QKV-nat] gpu={args.gpu} elapsed={out['elapsed_sec']:.3f}s "
        f"rel_err={out['rel_err']:.3e} q={out['rel_err_q']:.3e} "
        f"k={out['rel_err_k']:.3e} v={out['rel_err_v']:.3e} binary={out['binary']}"
    )


if __name__ == "__main__":
    main()
