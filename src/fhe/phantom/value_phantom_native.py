from __future__ import annotations

import argparse

from src.fhe.phantom.phantom_native_bench import run_value_full_native
from src.utils import print_stage


def main() -> None:
    p = argparse.ArgumentParser(description="Run native C++ CUDA Phantom Value+Out full benchmark")
    p.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--phantom-root", type=str, default=None, help="Path to Phantom C++ source root")
    p.add_argument("--build-dir", type=str, default="build", help="Build directory under Phantom root")
    p.add_argument("--no-build", action="store_true", help="Do not build binary if missing")
    args = p.parse_args()

    out = run_value_full_native(
        gpu=args.gpu,
        phantom_root=args.phantom_root,
        build_dir=args.build_dir,
        build_if_needed=not args.no_build,
    )

    vstats = out["value_stats"]
    ostats = out["out_stats"]
    print_stage(
        "Value-nat",
        ct_in=None,
        ct_out=None,
        ks_rots=vstats.get("ks_rots"),
        ks_muls=vstats.get("ks_muls_ctct"),
        ks_conj=vstats.get("ks_conj"),
        rel_err=out.get("value_rel_err"),
    )
    print_stage(
        "OUT-nat",
        ct_in=None,
        ct_out=None,
        ks_rots=ostats.get("ks_rots"),
        ks_muls=ostats.get("ks_muls_ctct"),
        ks_conj=ostats.get("ks_conj"),
        rel_err=out.get("out_rel_err"),
    )
    print(
        f"[Value-nat] gpu={args.gpu} elapsed={out['elapsed_sec']:.3f}s "
        f"value={out['value_elapsed_ms']:.2f}ms out={out['out_elapsed_ms']:.2f}ms "
        f"rel_err(value)={out['value_rel_err']:.3e} rel_err(out)={out['out_rel_err']:.3e} "
        f"binary={out['binary']}"
    )


if __name__ == "__main__":
    main()
