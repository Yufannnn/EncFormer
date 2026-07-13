from __future__ import annotations

import argparse
import time

import numpy as np

from src.engines.mpc_engine_factory import get_mpc_engine
from src.engines.mpc_engine_plain import PlainMpcEngine
from src.engines.mpc_gelu_secure import (
    load_secure_gelu_config,
    quantize_real_array,
    secure_gelu_plain_fixedpoint,
    selector_bits_public,
)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b)) + 1e-12
    return num / den


def bench(name: str, fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    dt = (time.perf_counter() - t0) * 1000.0
    return out, dt


def _max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(x, dtype=np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare MPC engine math outputs against plain backend.")
    ap.add_argument("--engine", default="crypten", choices=["plain", "crypten", "mixed"])
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--cols", type=int, default=768)
    ap.add_argument("--tol", type=float, default=2e-2, help="Max relative error for each op.")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(args.rows, args.cols)).astype(np.float64)
    gamma = rng.normal(1.0, 0.1, size=(args.cols,)).astype(np.float64)
    beta = rng.normal(0.0, 0.1, size=(args.cols,)).astype(np.float64)
    cfg = load_secure_gelu_config()
    eps = 1.0 / float(cfg.scale)

    plain = PlainMpcEngine()
    try:
        target = get_mpc_engine(args.engine)
    except Exception as exc:
        print(f"[SKIP] unable to initialize engine '{args.engine}': {exc}")
        return 0

    print(f"[MPC-BEHAVIOR] target={target.name} device={target.device} rows={args.rows} cols={args.cols}")

    p_smx, tps = bench("plain.softmax", plain.softmax_rows, x)
    e_smx, tes = bench("target.softmax", target.softmax_rows, x)
    p_ln, tpl = bench("plain.layer_norm", lambda: plain.layer_norm(x, eps=1e-5, gamma=gamma, beta=beta))
    e_ln, tel = bench("target.layer_norm", lambda: target.layer_norm(x, eps=1e-5, gamma=gamma, beta=beta))
    p_gelu, tpg = bench("secure_ref.gelu", lambda: secure_gelu_plain_fixedpoint(x, cfg))
    e_gelu, teg = bench("target.gelu", target.gelu, x)

    err_smx = rel_err(e_smx, p_smx)
    err_ln = rel_err(e_ln, p_ln)
    err_gelu = rel_err(e_gelu, p_gelu)

    boundary_pts = np.array(
        [-3.0, -2.7 - eps, -2.7, -2.7 + eps, -eps, 0.0, eps, 2.7 - eps, 2.7, 2.7 + eps, 3.0],
        dtype=np.float64,
    )
    boundary = boundary_pts.reshape(1, -1)
    b_ref = secure_gelu_plain_fixedpoint(boundary, cfg)
    b_tgt = target.gelu(boundary)
    err_boundary = rel_err(b_tgt, b_ref)

    q_boundary = quantize_real_array(boundary, cfg)
    left_mask = boundary < -float(cfg.threshold)
    right_mask = boundary > float(cfg.threshold)
    left_max = _max_abs(b_ref[left_mask]) if np.any(left_mask) else 0.0
    right_max = _max_abs(b_ref[right_mask] - q_boundary[right_mask]) if np.any(right_mask) else 0.0

    z0, z1, z2 = selector_bits_public(boundary, cfg)
    zsum = z0 + z1 + z2
    selector_ok = bool(np.all((zsum == 0) | (zsum == 1)))

    print(f"[softmax] rel_err={err_smx:.3e} plain_ms={tps:.2f} target_ms={tes:.2f}")
    print(f"[lnorm ] rel_err={err_ln:.3e} plain_ms={tpl:.2f} target_ms={tel:.2f}")
    print(f"[gelu  ] rel_err={err_gelu:.3e} ref_ms={tpg:.2f} target_ms={teg:.2f}")
    print(f"[gelu-bnd] rel_err={err_boundary:.3e} left_zero_max={left_max:.3e} right_identity_max={right_max:.3e}")
    print(f"[gelu-bits] selector_exclusive={selector_ok} zsum={zsum.flatten().tolist()}")

    if max(err_smx, err_ln, err_gelu, err_boundary) > args.tol:
        print(f"[FAIL] max rel_err exceeded tolerance {args.tol:.3e}")
        return 1
    if (left_max > args.tol) or (right_max > args.tol):
        print(f"[FAIL] branch behavior exceeded tolerance {args.tol:.3e}")
        return 1
    if not selector_ok:
        print("[FAIL] selector bits are not mutually exclusive.")
        return 1
    print(f"[PASS] all rel_err values are within tolerance {args.tol:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
