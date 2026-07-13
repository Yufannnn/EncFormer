from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.engines.mpc_engine_factory import get_mpc_engine
from src.fhe.phantom.phantom_native_bench import (
    _parse_kv_map,
    resolve_native_binaries,
)
from src.inference_runtime import (
    inject_layer_running_denoms,
    prepare_layer_running_denoms,
)
from src.models.model_config import get_config

SEED = 42
_SUPPORTED_CONFIGS = ("bert-base", "bert-large", "gpt2-base")


def _get_native_config(model_config_name: str | None = None):

    name = model_config_name or "bert-base"
    return get_config(name)


def _pipe_targets(model_config_name: str) -> tuple[str, ...]:

    from src.fhe.phantom.phantom_native_bench import _config_target

    return tuple(_config_target(t, model_config_name) for t in ("pipe_ckks_attn", "pipe_ckks_ff1", "pipe_ckks_ff2"))


def _layer_target(model_config_name: str) -> str:

    from src.fhe.phantom.phantom_native_bench import _config_target

    return _config_target("pipe_ckks_layer", model_config_name)


def _run_native(binary: str, pipe_dir: str, gpu: str) -> Dict[str, float]:
    env = os.environ.copy()
    env["PIPE_DIR"] = pipe_dir
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    r = subprocess.run(
        [binary],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Native binary {Path(binary).name} failed (rc={r.returncode}):\n"
            f"stdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}"
        )
    return _parse_kv_map(r.stdout)


def _start_native(binary: str, pipe_dir: str, gpu: str) -> subprocess.Popen:

    env = os.environ.copy()
    env["PIPE_DIR"] = pipe_dir
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout_f = open(os.path.join(pipe_dir, "_native_stdout.log"), "w")
    stderr_f = open(os.path.join(pipe_dir, "_native_stderr.log"), "w")
    return subprocess.Popen(
        [binary],
        stdout=stdout_f,
        stderr=stderr_f,
        env=env,
    )


def _wait_for_file(
    path: str,
    timeout: float = 600.0,
    *,
    proc: subprocess.Popen | None = None,
    name: str | None = None,
) -> None:

    t0 = time.monotonic()
    while not os.path.exists(path):
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                stdout, stderr = proc.communicate()
                proc_name = name or Path(getattr(proc, "args", ["<native>"])[0]).name
                raise RuntimeError(
                    f"Native binary {proc_name} exited before producing {Path(path).name} "
                    f"(rc={rc}).\n"
                    f"stdout: {(stdout or '')[-500:]}\n"
                    f"stderr: {(stderr or '')[-500:]}"
                )
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def _finish_native(proc: subprocess.Popen, name: str, timeout: float = 600.0) -> Dict[str, float]:

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    pipe_dir = proc._pipe_dir if hasattr(proc, "_pipe_dir") else None
    stdout = ""
    stderr = ""
    if proc.stdout and hasattr(proc.stdout, "name"):
        stdout_path = proc.stdout.name
        proc.stdout.close()
        try:
            with open(stdout_path) as f:
                stdout = f.read()
        except Exception:
            pass
    if proc.stderr and hasattr(proc.stderr, "name"):
        stderr_path = proc.stderr.name
        proc.stderr.close()
        try:
            with open(stderr_path) as f:
                stderr = f.read()
        except Exception:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"Native binary {name} failed (rc={proc.returncode}):\n"
            f"stdout: {(stdout or '')[-500:]}\nstderr: {(stderr or '')[-500:]}"
        )
    return _parse_kv_map(stdout)


def _write_f64(path: str, *arrays: np.ndarray) -> None:
    buf = np.concatenate([a.ravel() for a in arrays])
    buf.astype(np.float64).tofile(path)


def _write_f64_atomic(path: str, *arrays: np.ndarray) -> None:

    tmp = path + ".tmp"
    buf = np.concatenate([a.ravel() for a in arrays])
    buf.astype(np.float64).tofile(tmp)
    os.rename(tmp, path)


def _read_f64(path: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64).reshape(shape)


def native_supported_model_configs() -> tuple[str, ...]:

    return _SUPPORTED_CONFIGS


def validate_native_model_config(model_config_name: str) -> None:

    if model_config_name not in _SUPPORTED_CONFIGS:
        raise ValueError(f"phantom_native supports {_SUPPORTED_CONFIGS}. Received {model_config_name!r}.")


def _expect_shape(name: str, arr: np.ndarray, shape: tuple[int, ...], cfg_name: str = "") -> None:
    if arr.shape != shape:
        raise ValueError(
            f"phantom_native expects {name} shape {shape}, got {arr.shape}."
            + (f" (config={cfg_name})" if cfg_name else "")
        )


def validate_native_inputs(
    weights: Dict[str, np.ndarray],
    input_embeds: np.ndarray,
    attention_mask: np.ndarray | None = None,
    model_config_name: str = "bert-base",
) -> None:

    cfg = _get_native_config(model_config_name)
    M, D, D_MID = cfg.m, cfg.d_model, cfg.d_ff

    _expect_shape("input_embeds", input_embeds, (M, D), model_config_name)
    expected = {
        "WQ": (D, D),
        "WK": (D, D),
        "WV": (D, D),
        "WO": (D, D),
        "W1": (D, D_MID),
        "W2": (D_MID, D),
    }
    for name, shape in expected.items():
        _expect_shape(name, np.asarray(weights[name]), shape, model_config_name)

    optional = {
        "bQ": (D,),
        "bK": (D,),
        "bV": (D,),
        "bO": (D,),
        "b1": (D_MID,),
        "b2": (D,),
        "ln1_w": (D,),
        "ln1_b": (D,),
        "ln2_w": (D,),
        "ln2_b": (D,),
    }
    for name, shape in optional.items():
        if name in weights and weights[name] is not None:
            _expect_shape(name, np.asarray(weights[name]), shape, model_config_name)

    if attention_mask is None:
        return
    mask = np.asarray(attention_mask)
    valid_shapes = {(M,), (M, M)}
    if mask.shape not in valid_shapes:
        raise ValueError(f"phantom_native expects attention_mask shape {(M,)} or {(M, M)}, got {mask.shape}.")


def run_native_segmented(
    *,
    gpu: str = "2",
    mpc_engine: str | None = None,
    pipe_dir: str | None = None,
    model_config_name: str = "bert-base",
) -> Dict[str, Any]:

    validate_native_model_config(model_config_name)
    cfg = _get_native_config(model_config_name)
    M, D, D_MID, H, DH = cfg.m, cfg.d_model, cfg.d_ff, cfg.H, cfg.d_model // cfg.H

    targets = list(_pipe_targets(model_config_name))
    bins = resolve_native_binaries(targets=targets)

    mpc = get_mpc_engine(mpc_engine)
    print(f"[MPC] engine={mpc.name} device={mpc.device}")
    print(f"[GPU] {gpu}  model={model_config_name}")

    td = pipe_dir or tempfile.mkdtemp(prefix="encformer_pipe_")
    os.makedirs(td, exist_ok=True)
    print(f"[PIPE_DIR] {td}")

    for f in ("phase1_done", "a_heads_in.bin", "a_heads_in.bin.tmp"):
        p = os.path.join(td, f)
        if os.path.exists(p):
            os.remove(p)

    timings: Dict[str, float] = {}
    np.random.seed(SEED)

    A_in = np.random.randn(M, D).astype(np.float64)
    WQ = np.random.randn(D, D).astype(np.float64)
    WK = np.random.randn(D, D).astype(np.float64)
    WV = np.random.randn(D, D).astype(np.float64)
    bQ = np.random.randn(D).astype(np.float64)
    bK = np.random.randn(D).astype(np.float64)
    bV = np.random.randn(D).astype(np.float64)
    bO = np.random.randn(D).astype(np.float64)
    b1 = np.random.randn(D_MID).astype(np.float64)
    b2 = np.random.randn(D).astype(np.float64)
    ln1_w = np.random.randn(D).astype(np.float64)
    ln1_b = np.random.randn(D).astype(np.float64)
    ln2_w = np.random.randn(D).astype(np.float64)
    ln2_b = np.random.randn(D).astype(np.float64)
    W_O = np.random.randn(D, D).astype(np.float64)
    W1 = np.random.randn(D, D_MID).astype(np.float64)
    W2 = np.random.randn(D_MID, D).astype(np.float64)

    fhe_metrics: Dict[str, Dict[str, float]] = {}

    from src.fhe.phantom.phantom_native_bench import _config_target

    def _bin(base: str) -> str:
        return bins[_config_target(base, model_config_name)]

    t0 = time.perf_counter()
    _write_f64(
        os.path.join(td, "attn_in.bin"),
        A_in,
        WQ,
        WK,
        WV,
        bQ,
        bK,
        bV,
        W_O,
        bO,
    )

    proc = _start_native(_bin("pipe_ckks_attn"), td, gpu)

    _wait_for_file(
        os.path.join(td, "phase1_done"),
        proc=proc,
        name=Path(_bin("pipe_ckks_attn")).name,
    )
    t_phase1 = time.perf_counter() - t0

    S_heads_flat = _read_f64(os.path.join(td, "score_out.bin"), (H, M, M))
    _score_scale = 1.0 / np.sqrt(DH)
    S_heads = [S_heads_flat[h] * _score_scale for h in range(H)]

    t_mpc0 = time.perf_counter()
    A_heads = []
    for h in range(H):
        try:
            A_heads.append(mpc.softmax_rows(S_heads[h], head_index=h))
        except TypeError:
            A_heads.append(mpc.softmax_rows(S_heads[h]))
    A_heads_arr = np.array(A_heads, dtype=np.float64)
    timings["MPC-Softmax"] = time.perf_counter() - t_mpc0

    _write_f64_atomic(os.path.join(td, "a_heads_in.bin"), A_heads_arr)

    kv = _finish_native(proc, "pipe_ckks_attn")
    fhe_metrics["Attention"] = kv
    timings["Attention"] = time.perf_counter() - t0

    Z = _read_f64(os.path.join(td, "value_out.bin"), (M, D))

    t0 = time.perf_counter()
    Z_res = Z + A_in[:, :D]
    try:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b, ln_tag="ln1")
    except TypeError:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b)
    timings["MPC-LN1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _write_f64(os.path.join(td, "ff1_in.bin"), Z_ln, W1)
    kv = _run_native(_bin("pipe_ckks_ff1"), td, gpu)
    fhe_metrics["FF1"] = kv

    H1 = _read_f64(os.path.join(td, "ff1_out.bin"), (M, D_MID))
    timings["FF1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    H1_bias = H1 + b1[np.newaxis, :]
    H1_gelu = mpc.gelu(H1_bias)
    timings["MPC-GELU"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _write_f64(os.path.join(td, "ff2_in.bin"), H1_gelu, W2)
    kv = _run_native(_bin("pipe_ckks_ff2"), td, gpu)
    fhe_metrics["FF2"] = kv

    H2 = _read_f64(os.path.join(td, "ff2_out.bin"), (M, D))
    timings["FF2"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    H2_bias = H2 + b2[np.newaxis, :]
    H2_res = H2_bias + Z_ln
    try:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b, ln_tag="ln2")
    except TypeError:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b)
    timings["MPC-LN2"] = time.perf_counter() - t0

    fhe_keys = ["Attention", "FF1", "FF2"]
    mpc_keys = [k for k in timings if k not in fhe_keys]
    fhe_total = sum(timings[k] for k in fhe_keys)
    mpc_total = sum(timings[k] for k in mpc_keys)

    print("\n  [Timing — native segmented pipeline]")
    for k, v in timings.items():
        print(f"    {k:>20s} = {v:.3f}s")
    print(f"    {'FHE_total':>20s} = {fhe_total:.3f}s")
    print(f"    {'MPC_total':>20s} = {mpc_total:.3f}s")
    print(f"    {'Total':>20s} = {fhe_total + mpc_total:.3f}s")

    print("\n  [FHE breakdown (from native binaries)]")
    for stage, kv in fhe_metrics.items():
        setup = kv.get("setup_ms", 0)
        fhe_ms = kv.get("fhe_ms", 0)
        qkv_fhe = kv.get("qkv_fhe_ms", 0)
        score_fhe = kv.get("score_fhe_ms", 0)
        sv_fhe = kv.get("sv_fhe_ms", 0)
        out_fhe = kv.get("out_fhe_ms", 0)
        enc = kv.get("encrypt_ms", 0)
        dec = kv.get("decrypt_ms", 0)
        wait = kv.get("wait_ms", 0)
        rd = kv.get("read_ms", 0)
        wr = kv.get("write_ms", 0)
        err_keys = [k for k in kv if k.startswith("rel_err")]
        errs = {k: kv[k] for k in err_keys}
        parts = [f"setup={setup:.0f}"]
        if qkv_fhe:
            parts.append(f"qkv={qkv_fhe:.0f}")
        if score_fhe:
            parts.append(f"score={score_fhe:.0f}")
        if sv_fhe:
            parts.append(f"sv={sv_fhe:.0f}")
        if out_fhe:
            parts.append(f"out={out_fhe:.0f}")
        if fhe_ms:
            parts.append(f"fhe={fhe_ms:.0f}")
        parts.extend([f"enc={enc:.0f}", f"dec={dec:.0f}"])
        if wait:
            parts.append(f"wait={wait:.0f}")
        parts.append(f"io={rd + wr:.0f}")
        err_str = ", ".join(f"{k}={v:.1e}" for k, v in errs.items())
        print(f"    {stage:>12s}: {', '.join(parts)} ms | {err_str}")

    print("\nDone.")
    return {"timings": timings, "fhe_metrics": fhe_metrics}


def run_native_with_weights(
    *,
    weights: Dict[str, np.ndarray],
    running_denoms: Dict[str, np.ndarray] | None = None,
    input_embeds: np.ndarray,
    attention_mask: np.ndarray | None = None,
    layer_idx: int = 0,
    gpu: str = "2",
    mpc_engine: str | None = None,
    pipe_dir: str | None = None,
    bins: Dict[str, str] | None = None,
    mpc_backend=None,
    prepared_denoms: dict[str, np.ndarray] | None = None,
    verbose: bool = True,
    return_timings: bool = False,
    model_config: str = "bert-base",
) -> np.ndarray | tuple[np.ndarray, Dict[str, float]]:

    validate_native_model_config(model_config)
    cfg = _get_native_config(model_config)
    validate_native_inputs(weights, np.asarray(input_embeds), attention_mask, model_config_name=model_config)
    targets = list(_pipe_targets(model_config))
    bins = bins if bins is not None else resolve_native_binaries(targets=targets)

    from src.fhe.phantom.phantom_native_bench import _config_target

    def _bin(base: str) -> str:
        return bins[_config_target(base, model_config)]

    mpc = mpc_backend if mpc_backend is not None else get_mpc_engine(mpc_engine)
    if prepared_denoms is None:
        prepared_denoms = prepare_layer_running_denoms(running_denoms, layer_idx)
    inject_layer_running_denoms(mpc, prepared_denoms)

    td = pipe_dir or tempfile.mkdtemp(prefix="encformer_pipe_")
    os.makedirs(td, exist_ok=True)
    for f in ("phase1_done", "a_heads_in.bin", "a_heads_in.bin.tmp"):
        p = os.path.join(td, f)
        if os.path.exists(p):
            os.remove(p)

    import time as _time

    _timings: Dict[str, float] = {}

    A_in = input_embeds.astype(np.float64)
    d = A_in.shape[1]
    WQ = weights["WQ"]
    WK = weights["WK"]
    WV = weights["WV"]
    bQ = weights.get("bQ", np.zeros(d, dtype=np.float64))
    bK = weights.get("bK", np.zeros(d, dtype=np.float64))
    bV = weights.get("bV", np.zeros(d, dtype=np.float64))
    W_O = weights["WO"]
    bO = weights.get("bO", np.zeros(d, dtype=np.float64))
    W1 = weights["W1"]
    b1 = weights.get("b1", np.zeros(W1.shape[1], dtype=np.float64))
    W2 = weights["W2"]
    b2 = weights.get("b2", np.zeros(d, dtype=np.float64))
    ln1_w = weights.get("ln1_w")
    ln1_b = weights.get("ln1_b")
    ln2_w = weights.get("ln2_w")
    ln2_b = weights.get("ln2_b")

    m = A_in.shape[0]
    d_mid = W1.shape[1]
    H = cfg.H

    t0 = _time.perf_counter()
    _write_f64(os.path.join(td, "attn_in.bin"), A_in, WQ, WK, WV, bQ, bK, bV, W_O, bO)
    proc = _start_native(_bin("pipe_ckks_attn"), td, gpu)
    _wait_for_file(
        os.path.join(td, "phase1_done"),
        proc=proc,
        name=Path(_bin("pipe_ckks_attn")).name,
    )

    S_heads_flat = _read_f64(os.path.join(td, "score_out.bin"), (H, m, m))

    _dh = d // H
    _score_scale = 1.0 / np.sqrt(_dh)
    if attention_mask is not None:
        _mask_1d = np.asarray(attention_mask, dtype=np.float64).reshape(-1)
        _mask_add = (1.0 - _mask_1d) * -1e9 if _mask_1d.shape[0] == m else np.zeros(m)
    else:
        _mask_add = np.zeros(m)
    S_heads = [S_heads_flat[h] * _score_scale + _mask_add[None, :] for h in range(H)]

    t_mpc0 = _time.perf_counter()
    A_heads = []
    for h in range(H):
        try:
            A_heads.append(mpc.softmax_rows(S_heads[h], head_index=h))
        except TypeError:
            A_heads.append(mpc.softmax_rows(S_heads[h]))
    A_heads_arr = np.array(A_heads, dtype=np.float64)
    _timings["MPC-Softmax"] = _time.perf_counter() - t_mpc0

    _write_f64_atomic(os.path.join(td, "a_heads_in.bin"), A_heads_arr)
    _finish_native(proc, "pipe_ckks_attn")
    _timings["Attention"] = _time.perf_counter() - t0

    Z = _read_f64(os.path.join(td, "value_out.bin"), (m, d))

    t0 = _time.perf_counter()
    Z_res = Z + A_in[:, :d]
    try:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b, ln_tag="ln1")
    except TypeError:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b)
    _timings["MPC-LN1"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    _write_f64(os.path.join(td, "ff1_in.bin"), Z_ln, W1)
    _run_native(_bin("pipe_ckks_ff1"), td, gpu)
    H1 = _read_f64(os.path.join(td, "ff1_out.bin"), (m, d_mid))
    _timings["FF1"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    H1_bias = H1 + b1[np.newaxis, :d_mid]
    H1_gelu = mpc.gelu(H1_bias)
    _timings["MPC-GELU"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    _write_f64(os.path.join(td, "ff2_in.bin"), H1_gelu, W2)
    _run_native(_bin("pipe_ckks_ff2"), td, gpu)
    H2 = _read_f64(os.path.join(td, "ff2_out.bin"), (m, d))
    _timings["FF2"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    H2_bias = H2 + b2[np.newaxis, :d]
    H2_res = H2_bias + Z_ln
    try:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b, ln_tag="ln2")
    except TypeError:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b)
    _timings["MPC-LN2"] = _time.perf_counter() - t0

    _timings["Total"] = sum(v for k, v in _timings.items() if k != "Total")
    if verbose:
        print("  [Timing]", " | ".join(f"{k}={v:.2f}s" for k, v in _timings.items()))
    if return_timings:
        return H2_ln, _timings
    return H2_ln


def run_native_continuous_with_weights(
    *,
    weights: Dict[str, np.ndarray],
    running_denoms: Dict[str, np.ndarray] | None = None,
    input_embeds: np.ndarray,
    attention_mask: np.ndarray | None = None,
    layer_idx: int = 0,
    gpu: str = "2",
    mpc_engine: str | None = None,
    pipe_dir: str | None = None,
    bins: Dict[str, str] | None = None,
    mpc_backend=None,
    prepared_denoms: dict[str, np.ndarray] | None = None,
    verbose: bool = True,
    return_timings: bool = False,
    model_config: str = "bert-base",
) -> np.ndarray | tuple[np.ndarray, Dict[str, float]]:

    validate_native_model_config(model_config)
    cfg = _get_native_config(model_config)
    validate_native_inputs(weights, np.asarray(input_embeds), attention_mask, model_config_name=model_config)

    target = _layer_target(model_config)
    if bins is None:
        bins = resolve_native_binaries(targets=[target])
    layer_bin = bins[target]

    mpc = mpc_backend if mpc_backend is not None else get_mpc_engine(mpc_engine)
    if prepared_denoms is None:
        prepared_denoms = prepare_layer_running_denoms(running_denoms, layer_idx)
    inject_layer_running_denoms(mpc, prepared_denoms)

    td = pipe_dir or tempfile.mkdtemp(prefix="encformer_layer_")
    os.makedirs(td, exist_ok=True)
    for f in (
        "phase1_done",
        "phase2_done",
        "phase3_done",
        "phase4_done",
        "softmax_done",
        "ln1_done",
        "gelu_done",
    ):
        p = os.path.join(td, f)
        if os.path.exists(p):
            os.remove(p)

    timings: Dict[str, float] = {}
    bridge_rng = np.random.default_rng(67890 + int(layer_idx))

    M, D, D_MID, H = cfg.m, cfg.d_model, cfg.d_ff, cfg.H
    A_in = input_embeds.astype(np.float64)
    WQ = np.asarray(weights["WQ"], dtype=np.float64)
    WK = np.asarray(weights["WK"], dtype=np.float64)
    WV = np.asarray(weights["WV"], dtype=np.float64)
    bQ = np.asarray(weights.get("bQ", np.zeros(D, dtype=np.float64)), dtype=np.float64)
    bK = np.asarray(weights.get("bK", np.zeros(D, dtype=np.float64)), dtype=np.float64)
    bV = np.asarray(weights.get("bV", np.zeros(D, dtype=np.float64)), dtype=np.float64)
    W_O = np.asarray(weights["WO"], dtype=np.float64)
    bO = np.asarray(weights.get("bO", np.zeros(D, dtype=np.float64)), dtype=np.float64)
    W1 = np.asarray(weights["W1"], dtype=np.float64)
    b1 = np.asarray(weights.get("b1", np.zeros(D_MID, dtype=np.float64)), dtype=np.float64)
    W2 = np.asarray(weights["W2"], dtype=np.float64)
    b2 = np.asarray(weights.get("b2", np.zeros(D, dtype=np.float64)), dtype=np.float64)
    ln1_w = np.asarray(weights.get("ln1_w"), dtype=np.float64)
    ln1_b = np.asarray(weights.get("ln1_b"), dtype=np.float64)
    ln2_w = np.asarray(weights.get("ln2_w"), dtype=np.float64)
    ln2_b = np.asarray(weights.get("ln2_b"), dtype=np.float64)

    t0 = time.perf_counter()
    _write_f64(
        os.path.join(td, "layer_in.bin"),
        A_in,
        WQ,
        WK,
        WV,
        bQ,
        bK,
        bV,
        W_O,
        bO,
        W1,
        W2,
    )
    proc = _start_native(layer_bin, td, gpu)

    _wait_for_file(
        os.path.join(td, "phase1_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase1-QKV+Score"] = time.perf_counter() - t0

    S_heads = _bridge_reconstruct(
        os.path.join(td, "score_masked.bin"),
        os.path.join(td, "score_server_share.bin"),
        (H, M, M),
    )

    t_mpc0 = time.perf_counter()
    A_heads_list = []
    for h in range(H):
        try:
            A_heads_list.append(mpc.softmax_rows(S_heads[h], head_index=h))
        except TypeError:
            A_heads_list.append(mpc.softmax_rows(S_heads[h]))
    A_heads_arr = np.array(A_heads_list, dtype=np.float64)
    timings["MPC-Softmax"] = time.perf_counter() - t_mpc0

    _bridge_split_shares(
        A_heads_arr,
        bridge_rng,
        os.path.join(td, "softmax_client_share.bin"),
        os.path.join(td, "softmax_server_share.bin"),
    )
    _touch_file(os.path.join(td, "softmax_done"))

    t_p2 = time.perf_counter()
    _wait_for_file(
        os.path.join(td, "phase2_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase2-SV+OUT"] = time.perf_counter() - t_p2

    Z = _bridge_reconstruct(
        os.path.join(td, "z_masked.bin"),
        os.path.join(td, "z_server_share.bin"),
        (M, D),
    )

    t_mpc1 = time.perf_counter()
    Z_res = Z + A_in[:, :D]
    try:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b, ln_tag="ln1")
    except TypeError:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b)
    timings["MPC-LN1"] = time.perf_counter() - t_mpc1

    _bridge_split_shares(
        Z_ln,
        bridge_rng,
        os.path.join(td, "ln1_client_share.bin"),
        os.path.join(td, "ln1_server_share.bin"),
    )
    _touch_file(os.path.join(td, "ln1_done"))

    t_p3 = time.perf_counter()
    _wait_for_file(
        os.path.join(td, "phase3_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase3-FF1"] = time.perf_counter() - t_p3

    H1 = _bridge_reconstruct(
        os.path.join(td, "ff1_masked.bin"),
        os.path.join(td, "ff1_server_share.bin"),
        (M, D_MID),
    )

    t_mpc2 = time.perf_counter()
    H1_bias = H1 + b1[np.newaxis, :]
    H1_gelu = mpc.gelu(H1_bias)
    timings["MPC-GELU"] = time.perf_counter() - t_mpc2

    _bridge_split_shares(
        H1_gelu,
        bridge_rng,
        os.path.join(td, "gelu_client_share.bin"),
        os.path.join(td, "gelu_server_share.bin"),
    )
    _touch_file(os.path.join(td, "gelu_done"))

    t_p4 = time.perf_counter()
    _wait_for_file(
        os.path.join(td, "phase4_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase4-FF2"] = time.perf_counter() - t_p4

    H2 = _bridge_reconstruct(
        os.path.join(td, "ff2_masked.bin"),
        os.path.join(td, "ff2_server_share.bin"),
        (M, D),
    )
    _finish_native(proc, Path(layer_bin).name)

    t_mpc3 = time.perf_counter()
    H2_bias = H2 + b2[np.newaxis, :]
    H2_res = H2_bias + Z_ln
    try:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b, ln_tag="ln2")
    except TypeError:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b)
    timings["MPC-LN2"] = time.perf_counter() - t_mpc3

    timings["Total"] = sum(v for k, v in timings.items() if k != "Total")
    if verbose:
        print("  [Timing]", " | ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
    if return_timings:
        return H2_ln, timings
    return H2_ln


def _bridge_params() -> tuple[int, int, int]:

    q_bits = int(os.environ.get("CKKS_Q_CONV_BITS", "52"))
    r_bits = int(os.environ.get("CKKS_RING_BITS", "43"))
    p_bits = int(os.environ.get("CKKS_MPC_PREC_BITS", "13"))
    return (1 << q_bits, 1 << r_bits, 1 << p_bits)


def _bridge_reduce(v: np.ndarray) -> np.ndarray:

    q_conv, ring_mod, scale = _bridge_params()
    x_int = np.round(v * scale).astype(np.int64)
    x_ring = np.mod(x_int, ring_mod)

    x_ring = np.where(x_ring >= ring_mod // 2, x_ring - ring_mod, x_ring)
    t_q = np.mod(x_ring, q_conv)
    t_q = np.where(t_q >= q_conv // 2, t_q - q_conv, t_q)
    return t_q.astype(np.float64) / scale


def _bridge_reconstruct(masked_path: str, server_path: str, shape: tuple) -> np.ndarray:

    masked = _read_f64(masked_path, shape)
    server = _read_f64(server_path, shape)
    return _bridge_reduce(masked) + _bridge_reduce(server)


def _bridge_split_shares(x: np.ndarray, rng: np.random.Generator, client_path: str, server_path: str) -> None:

    q_conv, ring_mod, scale = _bridge_params()
    x_int = np.round(x.ravel() * scale).astype(np.int64)
    r = rng.integers(0, q_conv, size=x_int.shape, dtype=np.int64)

    masked_q = np.mod(x_int + r, q_conv)
    masked_q = np.where(masked_q >= q_conv // 2, masked_q - q_conv, masked_q)
    masked_ring = np.mod(masked_q, ring_mod)
    masked_ring = np.where(masked_ring >= ring_mod // 2, masked_ring - ring_mod, masked_ring)
    client = masked_ring.astype(np.float64) / scale

    neg_r_q = np.mod(-r, q_conv)
    neg_r_q = np.where(neg_r_q >= q_conv // 2, neg_r_q - q_conv, neg_r_q)
    neg_r_ring = np.mod(neg_r_q, ring_mod)
    neg_r_ring = np.where(neg_r_ring >= ring_mod // 2, neg_r_ring - ring_mod, neg_r_ring)
    server = neg_r_ring.astype(np.float64) / scale

    _write_f64_atomic(client_path, client.reshape(x.shape))
    _write_f64_atomic(server_path, server.reshape(x.shape))


def run_native_continuous(
    *,
    gpu: str = "2",
    mpc_engine: str | None = None,
    pipe_dir: str | None = None,
    model_config_name: str = "bert-base",
) -> Dict[str, Any]:

    validate_native_model_config(model_config_name)
    cfg = _get_native_config(model_config_name)
    M, D, D_MID, H, DH = cfg.m, cfg.d_model, cfg.d_ff, cfg.H, cfg.d_model // cfg.H

    target = _layer_target(model_config_name)
    bins = resolve_native_binaries(targets=[target])
    layer_bin = bins[target]

    mpc = get_mpc_engine(mpc_engine)
    print(f"[MPC] engine={mpc.name} device={mpc.device}")
    print(f"[GPU] {gpu}  model={model_config_name}")

    td = pipe_dir or tempfile.mkdtemp(prefix="encformer_layer_")
    os.makedirs(td, exist_ok=True)
    print(f"[PIPE_DIR] {td}")

    for f in ("phase1_done", "phase2_done", "phase3_done", "phase4_done", "softmax_done", "ln1_done", "gelu_done"):
        p = os.path.join(td, f)
        if os.path.exists(p):
            os.remove(p)

    timings: Dict[str, float] = {}
    np.random.seed(SEED)
    bridge_rng = np.random.default_rng(67890)

    A_in = np.random.randn(M, D).astype(np.float64)
    WQ = np.random.randn(D, D).astype(np.float64)
    WK = np.random.randn(D, D).astype(np.float64)
    WV = np.random.randn(D, D).astype(np.float64)
    bQ = np.random.randn(D).astype(np.float64)
    bK = np.random.randn(D).astype(np.float64)
    bV = np.random.randn(D).astype(np.float64)
    bO = np.random.randn(D).astype(np.float64)
    b1 = np.random.randn(D_MID).astype(np.float64)
    b2 = np.random.randn(D).astype(np.float64)
    ln1_w = np.random.randn(D).astype(np.float64)
    ln1_b = np.random.randn(D).astype(np.float64)
    ln2_w = np.random.randn(D).astype(np.float64)
    ln2_b = np.random.randn(D).astype(np.float64)
    W_O = np.random.randn(D, D).astype(np.float64)
    W1 = np.random.randn(D, D_MID).astype(np.float64)
    W2 = np.random.randn(D_MID, D).astype(np.float64)

    t0 = time.perf_counter()
    _write_f64(
        os.path.join(td, "layer_in.bin"),
        A_in,
        WQ,
        WK,
        WV,
        bQ,
        bK,
        bV,
        W_O,
        bO,
        W1,
        W2,
    )

    proc = _start_native(layer_bin, td, gpu)

    _wait_for_file(
        os.path.join(td, "phase1_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    t_phase1 = time.perf_counter() - t0

    S_heads = _bridge_reconstruct(
        os.path.join(td, "score_masked.bin"),
        os.path.join(td, "score_server_share.bin"),
        (H, M, M),
    )
    timings["Phase1-QKV+Score"] = t_phase1

    t_mpc0 = time.perf_counter()
    A_heads_list = []
    for h in range(H):
        try:
            A_heads_list.append(mpc.softmax_rows(S_heads[h], head_index=h))
        except TypeError:
            A_heads_list.append(mpc.softmax_rows(S_heads[h]))
    A_heads_arr = np.array(A_heads_list, dtype=np.float64)
    timings["MPC-Softmax"] = time.perf_counter() - t_mpc0

    _bridge_split_shares(
        A_heads_arr,
        bridge_rng,
        os.path.join(td, "softmax_client_share.bin"),
        os.path.join(td, "softmax_server_share.bin"),
    )
    _touch_file(os.path.join(td, "softmax_done"))

    t_p2 = time.perf_counter()
    _wait_for_file(
        os.path.join(td, "phase2_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase2-SV+OUT"] = time.perf_counter() - t_p2

    Z = _bridge_reconstruct(
        os.path.join(td, "z_masked.bin"),
        os.path.join(td, "z_server_share.bin"),
        (M, D),
    )

    t_mpc1 = time.perf_counter()
    Z_res = Z + A_in[:, :D]
    try:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b, ln_tag="ln1")
    except TypeError:
        Z_ln = mpc.layer_norm(Z_res, eps=1e-5, gamma=ln1_w, beta=ln1_b)
    timings["MPC-LN1"] = time.perf_counter() - t_mpc1

    _bridge_split_shares(
        Z_ln,
        bridge_rng,
        os.path.join(td, "ln1_client_share.bin"),
        os.path.join(td, "ln1_server_share.bin"),
    )
    _touch_file(os.path.join(td, "ln1_done"))

    t_p3 = time.perf_counter()
    _wait_for_file(
        os.path.join(td, "phase3_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase3-FF1"] = time.perf_counter() - t_p3

    H1 = _bridge_reconstruct(
        os.path.join(td, "ff1_masked.bin"),
        os.path.join(td, "ff1_server_share.bin"),
        (M, D_MID),
    )

    t_mpc2 = time.perf_counter()
    H1_bias = H1 + b1[np.newaxis, :]
    expanded_gelu = os.environ.get("ENCFORMER_EXPANDED_GELU", "0") in ("1", "true", "True")
    if expanded_gelu and hasattr(mpc, "gelu_preeval"):
        f0_path = os.path.join(td, "ff1_f0_share.bin")
        f1_path = os.path.join(td, "ff1_f1_share.bin")
        if os.path.exists(f0_path) and os.path.exists(f1_path):
            F0 = _bridge_reconstruct(
                f0_path,
                os.path.join(td, "ff1_f0_server_share.bin"),
                (M, D_MID),
            )
            F1 = _bridge_reconstruct(
                f1_path,
                os.path.join(td, "ff1_f1_server_share.bin"),
                (M, D_MID),
            )
            H1_gelu = mpc.gelu_preeval(H1_bias, f0=F0, f1=F1)
        else:
            H1_gelu = mpc.gelu(H1_bias)
    else:
        H1_gelu = mpc.gelu(H1_bias)
    timings["MPC-GELU"] = time.perf_counter() - t_mpc2

    _bridge_split_shares(
        H1_gelu,
        bridge_rng,
        os.path.join(td, "gelu_client_share.bin"),
        os.path.join(td, "gelu_server_share.bin"),
    )
    _touch_file(os.path.join(td, "gelu_done"))

    t_p4 = time.perf_counter()
    _wait_for_file(
        os.path.join(td, "phase4_done"),
        proc=proc,
        name=Path(layer_bin).name,
    )
    timings["Phase4-FF2"] = time.perf_counter() - t_p4

    H2 = _bridge_reconstruct(
        os.path.join(td, "ff2_masked.bin"),
        os.path.join(td, "ff2_server_share.bin"),
        (M, D),
    )

    kv = _finish_native(proc, "pipe_ckks_layer")

    t_mpc3 = time.perf_counter()
    H2_bias = H2 + b2[np.newaxis, :]
    H2_res = H2_bias + Z_ln
    try:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b, ln_tag="ln2")
    except TypeError:
        H2_ln = mpc.layer_norm(H2_res, eps=1e-5, gamma=ln2_w, beta=ln2_b)
    timings["MPC-LN2"] = time.perf_counter() - t_mpc3

    fhe_keys = [k for k in timings if k.startswith("Phase")]
    mpc_keys = [k for k in timings if k.startswith("MPC")]
    fhe_total = sum(timings[k] for k in fhe_keys)
    mpc_total = sum(timings[k] for k in mpc_keys)

    print("\n  [Timing — native continuous pipeline (bridge)]")
    for k, v in timings.items():
        print(f"    {k:>20s} = {v:.3f}s")
    print(f"    {'FHE_total':>20s} = {fhe_total:.3f}s")
    print(f"    {'MPC_total':>20s} = {mpc_total:.3f}s")
    print(f"    {'Total':>20s} = {fhe_total + mpc_total:.3f}s")

    print("\n  [FHE breakdown (from native binary)]")
    for key in sorted(kv.keys()):
        print(f"    {key} = {kv[key]}")

    print("\nDone.")
    return {"timings": timings, "fhe_metrics": kv}


def _touch_file(path: str) -> None:

    with open(path, "w"):
        pass


def main():
    parser = argparse.ArgumentParser(description="EncFormer native CKKS pipeline")
    parser.add_argument("--gpu", default="2", help="CUDA_VISIBLE_DEVICES (default: 2)")
    parser.add_argument("--mpc-engine", default=None, help="MPC engine: plain, crypten, mixed")
    parser.add_argument("--pipe-dir", default=None, help="Directory for binary I/O files")
    parser.add_argument(
        "--model-config", default="bert-base", choices=list(_SUPPORTED_CONFIGS), help="Model config for native binaries"
    )
    parser.add_argument("--continuous", action="store_true", help="Use continuous pipeline (single binary with bridge)")
    args = parser.parse_args()
    validate_native_model_config(args.model_config)
    if args.continuous:
        run_native_continuous(
            gpu=args.gpu,
            mpc_engine=args.mpc_engine,
            pipe_dir=args.pipe_dir,
            model_config_name=args.model_config,
        )
    else:
        run_native_segmented(
            gpu=args.gpu,
            mpc_engine=args.mpc_engine,
            pipe_dir=args.pipe_dir,
            model_config_name=args.model_config,
        )


if __name__ == "__main__":
    main()
