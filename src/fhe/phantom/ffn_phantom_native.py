from __future__ import annotations

import argparse

from src.fhe.phantom.phantom_native_bench import run_ffn_full_native
from src.utils import print_stage


def main() -> None:
    p = argparse.ArgumentParser(description="Run native C++ CUDA Phantom FFN full benchmark")
    p.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--phantom-root", type=str, default=None, help="Path to Phantom C++ source root")
    p.add_argument("--build-dir", type=str, default="build", help="Build directory under Phantom root")
    p.add_argument("--no-build", action="store_true", help="Do not build binary if missing")
    args = p.parse_args()

    out = run_ffn_full_native(
        gpu=args.gpu,
        phantom_root=args.phantom_root,
        build_dir=args.build_dir,
        build_if_needed=not args.no_build,
    )

    ff1 = out["ff1_stats"]
    ff2 = out["ff2_stats"]
    print_stage(
        "FF1-nat",
        ct_in=None,
        ct_out=None,
        ks_rots=ff1.get("ks_rots"),
        ks_muls=ff1.get("ks_muls_ctct"),
        ks_conj=ff1.get("ks_conj"),
        rel_err=out.get("ff1_err"),
    )
    print_stage(
        "FF2-nat",
        ct_in=None,
        ct_out=None,
        ks_rots=ff2.get("ks_rots"),
        ks_muls=ff2.get("ks_muls_ctct"),
        ks_conj=ff2.get("ks_conj"),
        rel_err=out.get("ff2_err"),
    )
    print(
        f"[FFN-nat] gpu={args.gpu} elapsed={out['elapsed_sec']:.3f}s "
        f"ff1={out['ff1_elapsed_ms']:.2f}ms ff2={out['ff2_elapsed_ms']:.2f}ms "
        f"rel_err(ff1)={out['ff1_err']:.3e} rel_err(ff2)={out['ff2_err']:.3e} "
        f"binary={out['binary']}"
    )


if __name__ == "__main__":
    main()
