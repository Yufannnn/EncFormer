from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_KV_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s+([-+]?[0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*$")

_LOCAL_NATIVE_EXAMPLES = Path(__file__).resolve().parent.parent / "native" / "phantom" / "examples"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PHANTOM_ROOT = _PROJECT_ROOT / "third_party" / "phantom-fhe"
_LOCAL_NATIVE_FILES = (
    "bench_ckks_qkv_full.cu",
    "bench_ckks_score_full.cu",
    "bench_ckks_value_full.cu",
    "bench_ckks_ffn_full.cu",
    "bench_f0f1.cu",
    "model_config.h",
    "pipe_io.h",
    "pipe_ckks_common.h",
    "pipe_ckks_ff1.cu",
    "pipe_ckks_ff2.cu",
    "pipe_ckks_qkv.cu",
    "pipe_ckks_value_out.cu",
    "pipe_ckks_qkv_score.cu",
    "pipe_ckks_attn.cu",
    "pipe_ckks_layer.cu",
    "pipe_bridge.h",
)


_MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "bert-base": {
        "ENCFORMER_M": 128,
        "ENCFORMER_D": 768,
        "ENCFORMER_H": 12,
        "ENCFORMER_D_FF": 3072,
        "ENCFORMER_NSLOTS": 16384,
    },
    "bert-large": {
        "ENCFORMER_M": 128,
        "ENCFORMER_D": 1024,
        "ENCFORMER_H": 16,
        "ENCFORMER_D_FF": 4096,
        "ENCFORMER_NSLOTS": 32768,
    },
    "gpt2-base": {
        "ENCFORMER_M": 64,
        "ENCFORMER_D": 768,
        "ENCFORMER_H": 12,
        "ENCFORMER_D_FF": 3072,
        "ENCFORMER_NSLOTS": 16384,
    },
}


_BASE_TARGETS = (
    ("bench_ckks_qkv_full", "bench_ckks_qkv_full.cu"),
    ("bench_ckks_score_full", "bench_ckks_score_full.cu"),
    ("bench_ckks_value_full", "bench_ckks_value_full.cu"),
    ("bench_ckks_ffn_full", "bench_ckks_ffn_full.cu"),
    ("bench_f0f1", "bench_f0f1.cu"),
    ("pipe_ckks_ff1", "pipe_ckks_ff1.cu"),
    ("pipe_ckks_ff2", "pipe_ckks_ff2.cu"),
    ("pipe_ckks_qkv", "pipe_ckks_qkv.cu"),
    ("pipe_ckks_value_out", "pipe_ckks_value_out.cu"),
    ("pipe_ckks_qkv_score", "pipe_ckks_qkv_score.cu"),
    ("pipe_ckks_attn", "pipe_ckks_attn.cu"),
    ("pipe_ckks_layer", "pipe_ckks_layer.cu"),
)


def _config_suffix(model_config: str) -> str:

    return "_" + model_config.replace("-", "_")


def _config_target(base_target: str, model_config: str) -> str:

    return base_target + _config_suffix(model_config)


def _extra_targets_for_config(model_config: str) -> Sequence[tuple[str, str]]:

    return tuple((_config_target(t, model_config), src) for t, src in _BASE_TARGETS)


_EXTRA_TARGETS = _BASE_TARGETS + tuple(
    (_config_target(t, cfg), src) for cfg in _MODEL_CONFIGS for t, src in _BASE_TARGETS
)


def _parse_int(s: str) -> int:
    return int(s.replace(",", "").strip())


def _parse_num(s: str) -> float:
    return float(s.replace(",", "").strip())


def _parse_kv_map(stdout: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ln in stdout.splitlines():
        m = _KV_RE.match(ln)
        if not m:
            continue
        out[m.group(1)] = _parse_num(m.group(2))
    return out


def _is_phantom_root(path: Path) -> bool:
    return (path / "examples").is_dir() and (path / "CMakeLists.txt").is_file()


def _find_phantom_root(explicit_root: Optional[str] = None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if _is_phantom_root(root):
            return root
        raise FileNotFoundError(
            f"Phantom root is invalid or missing required files (examples/, CMakeLists.txt): {root}"
        )

    env_root = os.getenv("PHANTOM_NATIVE_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if _is_phantom_root(root):
            return root
        raise FileNotFoundError(
            f"PHANTOM_NATIVE_ROOT is invalid or missing required files (examples/, CMakeLists.txt): {root}"
        )

    local_root = _DEFAULT_PHANTOM_ROOT.resolve()
    if _is_phantom_root(local_root):
        return local_root

    try:
        import pyPhantom  # type: ignore

        so_path = Path(pyPhantom.__file__).resolve()
        for parent in so_path.parents:
            if not (_is_phantom_root(parent) and (parent / "include").is_dir()):
                continue

            try:
                parent.relative_to(_PROJECT_ROOT)
            except ValueError:
                continue
            return parent
    except Exception:
        pass

    raise RuntimeError(
        "Could not locate Phantom C++ root. "
        f"Default expected path: {_DEFAULT_PHANTOM_ROOT}. "
        "Set PHANTOM_NATIVE_ROOT or pass --phantom-root."
    )


def _run(
    cmd: list[str], *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None
) -> subprocess.CompletedProcess[str]:
    use_env = os.environ.copy()
    if env:
        use_env.update(env)
    return subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, env=use_env, capture_output=True, text=True)


def _sync_local_native_sources(root: Path) -> None:
    if not _LOCAL_NATIVE_EXAMPLES.is_dir():
        return

    examples_dir = root / "examples"
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"Phantom examples dir not found: {examples_dir}")

    for name in _LOCAL_NATIVE_FILES:
        src = _LOCAL_NATIVE_EXAMPLES / name
        if src.is_file():
            dst = examples_dir / name

            if dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst)

    cmake_path = examples_dir / "CMakeLists.txt"
    if not cmake_path.is_file():
        return

    txt = cmake_path.read_text(encoding="utf-8")
    changed = False

    for target, source in _BASE_TARGETS:
        token = f"add_executable({target}"
        if token in txt:
            continue
        txt += (
            f"\nadd_executable({target}\n"
            f"        {source}\n"
            ")\n"
            f"target_link_libraries({target} PRIVATE Phantom)\n"
            f"target_include_directories({target} PUBLIC ${{CMAKE_SOURCE_DIR}}/include)\n"
            f"set_target_properties({target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/bin)\n"
        )
        changed = True

    for cfg_name, defs in _MODEL_CONFIGS.items():
        defs_str = " ".join(f"{k}={v}" for k, v in defs.items())
        for base_target, source in _BASE_TARGETS:
            target = _config_target(base_target, cfg_name)
            token = f"add_executable({target}"
            if token in txt:
                continue
            txt += (
                f"\nadd_executable({target}\n"
                f"        {source}\n"
                ")\n"
                f"target_link_libraries({target} PRIVATE Phantom)\n"
                f"target_include_directories({target} PUBLIC ${{CMAKE_SOURCE_DIR}}/include)\n"
                f"target_compile_definitions({target} PRIVATE {defs_str})\n"
                f"set_target_properties({target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/bin)\n"
            )
            changed = True

    if changed:
        cmake_path.write_text(txt, encoding="utf-8")


def _ensure_configured(root: Path, bdir: Path) -> None:
    cache_path = bdir / "CMakeCache.txt"
    if cache_path.is_file():
        txt = cache_path.read_text(encoding="utf-8", errors="ignore")
        home_key = "CMAKE_HOME_DIRECTORY:INTERNAL="
        cache_home: str | None = None
        for ln in txt.splitlines():
            if ln.startswith(home_key):
                cache_home = ln.split("=", 1)[1].strip()
                break
        if cache_home == str(root):
            return

        cache_path.unlink(missing_ok=True)
        shutil.rmtree(bdir / "CMakeFiles", ignore_errors=True)

    cfg = _run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(bdir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=root,
    )
    if cfg.returncode != 0:
        raise RuntimeError(f"cmake configure failed:\n{cfg.stdout}\n{cfg.stderr}")


def ensure_native_binary(
    *,
    target: str,
    phantom_root: Optional[str] = None,
    build_dir: str = "build",
    force_build: bool = False,
) -> Path:
    root = _find_phantom_root(phantom_root)
    _sync_local_native_sources(root)
    bdir = (root / build_dir).resolve()
    bin_path = bdir / "bin" / target

    _ensure_configured(root, bdir)

    build = _run(
        [
            "cmake",
            "--build",
            str(bdir),
            "--target",
            target,
            "-j",
            str(max(4, (os.cpu_count() or 8) // 2)),
        ],
        cwd=root,
    )
    if build.returncode != 0:
        no_rule = ("No rule to make target" in build.stderr) or ("No rule to make target" in build.stdout)
        if no_rule:
            cfg = _run(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(bdir),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=root,
            )
            if cfg.returncode != 0:
                raise RuntimeError(f"cmake configure failed:\n{cfg.stdout}\n{cfg.stderr}")
            build = _run(
                [
                    "cmake",
                    "--build",
                    str(bdir),
                    "--target",
                    target,
                    "-j",
                    str(max(4, (os.cpu_count() or 8) // 2)),
                ],
                cwd=root,
            )
        if build.returncode != 0:
            raise RuntimeError(f"cmake build failed for {target}:\n{build.stdout}\n{build.stderr}")
    if not bin_path.is_file():
        raise FileNotFoundError(f"Binary not found after build: {bin_path}")
    return bin_path


def _build_targets_once(*, root: Path, bdir: Path, targets: Sequence[str]) -> None:
    uniq_targets: list[str] = []
    for t in targets:
        tt = str(t).strip()
        if not tt or tt in uniq_targets:
            continue
        uniq_targets.append(tt)
    if not uniq_targets:
        return

    cmd = [
        "cmake",
        "--build",
        str(bdir),
        "--target",
        *uniq_targets,
        "-j",
        str(max(4, (os.cpu_count() or 8) // 2)),
    ]
    build = _run(cmd, cwd=root)
    if build.returncode != 0:
        no_rule = ("No rule to make target" in build.stderr) or ("No rule to make target" in build.stdout)
        if no_rule:
            cfg = _run(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(bdir),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                cwd=root,
            )
            if cfg.returncode != 0:
                raise RuntimeError(f"cmake configure failed:\n{cfg.stdout}\n{cfg.stderr}")
            build = _run(cmd, cwd=root)
        if build.returncode != 0:
            tnames = ", ".join(uniq_targets)
            raise RuntimeError(f"cmake build failed for [{tnames}]:\n{build.stdout}\n{build.stderr}")


def resolve_native_binaries(
    *,
    targets: Sequence[str],
    phantom_root: Optional[str] = None,
    build_dir: str = "build",
    build_if_needed: bool = True,
) -> Dict[str, str]:
    root = _find_phantom_root(phantom_root)
    _sync_local_native_sources(root)
    bdir = (root / build_dir).resolve()
    _ensure_configured(root, bdir)

    uniq_targets: list[str] = []
    for t in targets:
        tt = str(t).strip()
        if not tt or tt in uniq_targets:
            continue
        uniq_targets.append(tt)

    if build_if_needed:
        _build_targets_once(root=root, bdir=bdir, targets=uniq_targets)

    out: Dict[str, str] = {}
    missing: list[str] = []
    for target in uniq_targets:
        p = bdir / "bin" / target
        if not p.is_file():
            missing.append(f"{target} ({p})")
            continue
        out[target] = str(p)

    if missing:
        raise FileNotFoundError(
            "Native binary not found. "
            + ("Build was disabled. " if not build_if_needed else "")
            + "Missing: "
            + ", ".join(missing)
        )
    return out


def _run_native_target(
    *,
    target: str,
    gpu: Optional[str],
    phantom_root: Optional[str],
    build_dir: str,
    build_if_needed: bool,
    binary_path: Optional[str] = None,
    model_config: Optional[str] = None,
) -> Dict[str, Any]:

    if model_config is not None and binary_path is None:
        if model_config not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown model config {model_config!r}. Available: {list(_MODEL_CONFIGS.keys())}")
        target = _config_target(target, model_config)
    if binary_path is not None:
        bin_path = Path(binary_path).expanduser().resolve()
        if not bin_path.is_file():
            raise FileNotFoundError(f"Native binary does not exist: {bin_path}")
    else:
        root = _find_phantom_root(phantom_root)
        _sync_local_native_sources(root)
        bdir = (root / build_dir).resolve()
        bin_path = bdir / "bin" / target
        if build_if_needed:
            bin_path = ensure_native_binary(
                target=target,
                phantom_root=str(root),
                build_dir=build_dir,
                force_build=False,
            )
        elif not bin_path.is_file():
            raise FileNotFoundError(
                f"Native binary not found and build disabled: {bin_path}. "
                "Run without --no-build or set --build-dir correctly."
            )

    env: Dict[str, str] = {}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    p = _run([str(bin_path)], env=env if env else None)
    if p.returncode != 0:
        raise RuntimeError(f"{target} failed:\n{p.stdout}\n{p.stderr}")

    return {
        "stdout": p.stdout,
        "stderr": p.stderr,
        "binary": str(bin_path),
    }


def run_qkv_full_native(
    *,
    gpu: Optional[str] = None,
    phantom_root: Optional[str] = None,
    build_dir: str = "build",
    build_if_needed: bool = True,
    binary_path: Optional[str] = None,
    model_config: Optional[str] = None,
) -> Dict[str, Any]:
    out = _run_native_target(
        target="bench_ckks_qkv_full",
        gpu=gpu,
        phantom_root=phantom_root,
        build_dir=build_dir,
        build_if_needed=build_if_needed,
        binary_path=binary_path,
        model_config=model_config,
    )

    kv = _parse_kv_map(out["stdout"])
    required = (
        "elapsed_ms",
        "rel_err_q",
        "rel_err_k",
        "rel_err_v",
        "rel_err",
        "ks_rots",
        "ks_muls_ctct",
        "ks_conj",
    )
    for k in required:
        if k not in kv:
            raise RuntimeError(f"bench_ckks_qkv_full missing key '{k}' in output")

    return {
        **out,
        "elapsed_sec": float(kv["elapsed_ms"]) / 1000.0,
        "elapsed_ms": float(kv["elapsed_ms"]),
        "rel_err_q": float(kv["rel_err_q"]),
        "rel_err_k": float(kv["rel_err_k"]),
        "rel_err_v": float(kv["rel_err_v"]),
        "rel_err": float(kv["rel_err"]),
        "stats": {
            "ks_rots": int(kv["ks_rots"]),
            "ks_muls_ctct": int(kv["ks_muls_ctct"]),
            "ks_conj": int(kv["ks_conj"]),
        },
    }


def run_score_full_native(
    *,
    gpu: Optional[str] = None,
    phantom_root: Optional[str] = None,
    build_dir: str = "build",
    build_if_needed: bool = True,
    binary_path: Optional[str] = None,
    model_config: Optional[str] = None,
) -> Dict[str, Any]:
    out = _run_native_target(
        target="bench_ckks_score_full",
        gpu=gpu,
        phantom_root=phantom_root,
        build_dir=build_dir,
        build_if_needed=build_if_needed,
        binary_path=binary_path,
        model_config=model_config,
    )
    kv = _parse_kv_map(out["stdout"])
    for k in ("elapsed_ms", "rel_err", "ks_rots", "ks_muls_ctct", "ks_conj"):
        if k not in kv:
            raise RuntimeError(f"bench_ckks_score_full missing key '{k}' in output")
    return {
        **out,
        "elapsed_ms": float(kv["elapsed_ms"]),
        "elapsed_sec": float(kv["elapsed_ms"]) / 1000.0,
        "rel_err": float(kv["rel_err"]),
        "stats": {
            "ks_rots": int(kv["ks_rots"]),
            "ks_muls_ctct": int(kv["ks_muls_ctct"]),
            "ks_conj": int(kv["ks_conj"]),
        },
    }


def run_value_full_native(
    *,
    gpu: Optional[str] = None,
    phantom_root: Optional[str] = None,
    build_dir: str = "build",
    build_if_needed: bool = True,
    binary_path: Optional[str] = None,
    model_config: Optional[str] = None,
) -> Dict[str, Any]:
    out = _run_native_target(
        target="bench_ckks_value_full",
        gpu=gpu,
        phantom_root=phantom_root,
        build_dir=build_dir,
        build_if_needed=build_if_needed,
        binary_path=binary_path,
        model_config=model_config,
    )
    kv = _parse_kv_map(out["stdout"])
    required = (
        "elapsed_ms",
        "elapsed_ms_value",
        "elapsed_ms_out",
        "rel_err_value",
        "rel_err_out",
        "ks_rots_value",
        "ks_muls_ctct_value",
        "ks_conj_value",
        "ks_rots_out",
        "ks_muls_ctct_out",
        "ks_conj_out",
    )
    for k in required:
        if k not in kv:
            raise RuntimeError(f"bench_ckks_value_full missing key '{k}' in output")

    return {
        **out,
        "elapsed_ms": float(kv["elapsed_ms"]),
        "elapsed_sec": float(kv["elapsed_ms"]) / 1000.0,
        "value_elapsed_ms": float(kv["elapsed_ms_value"]),
        "out_elapsed_ms": float(kv["elapsed_ms_out"]),
        "value_rel_err": float(kv["rel_err_value"]),
        "out_rel_err": float(kv["rel_err_out"]),
        "value_stats": {
            "ks_rots": int(kv["ks_rots_value"]),
            "ks_muls_ctct": int(kv["ks_muls_ctct_value"]),
            "ks_conj": int(kv["ks_conj_value"]),
        },
        "out_stats": {
            "ks_rots": int(kv["ks_rots_out"]),
            "ks_muls_ctct": int(kv["ks_muls_ctct_out"]),
            "ks_conj": int(kv["ks_conj_out"]),
        },
    }


def run_ffn_full_native(
    *,
    gpu: Optional[str] = None,
    phantom_root: Optional[str] = None,
    build_dir: str = "build",
    build_if_needed: bool = True,
    binary_path: Optional[str] = None,
    model_config: Optional[str] = None,
) -> Dict[str, Any]:
    out = _run_native_target(
        target="bench_ckks_ffn_full",
        gpu=gpu,
        phantom_root=phantom_root,
        build_dir=build_dir,
        build_if_needed=build_if_needed,
        binary_path=binary_path,
        model_config=model_config,
    )
    kv = _parse_kv_map(out["stdout"])
    required = (
        "elapsed_ms",
        "elapsed_ms_ff1",
        "elapsed_ms_ff2",
        "rel_err_ff1",
        "rel_err_ff2",
        "ks_rots_ff1",
        "ks_muls_ctct_ff1",
        "ks_conj_ff1",
        "ks_rots_ff2",
        "ks_muls_ctct_ff2",
        "ks_conj_ff2",
    )
    for k in required:
        if k not in kv:
            raise RuntimeError(f"bench_ckks_ffn_full missing key '{k}' in output")

    return {
        **out,
        "elapsed_ms": float(kv["elapsed_ms"]),
        "elapsed_sec": float(kv["elapsed_ms"]) / 1000.0,
        "ff1_elapsed_ms": float(kv["elapsed_ms_ff1"]),
        "ff2_elapsed_ms": float(kv["elapsed_ms_ff2"]),
        "ff1_err": float(kv["rel_err_ff1"]),
        "ff2_err": float(kv["rel_err_ff2"]),
        "ff1_stats": {
            "ks_rots": int(kv["ks_rots_ff1"]),
            "ks_muls_ctct": int(kv["ks_muls_ctct_ff1"]),
            "ks_conj": int(kv["ks_conj_ff1"]),
        },
        "ff2_stats": {
            "ks_rots": int(kv["ks_rots_ff2"]),
            "ks_muls_ctct": int(kv["ks_muls_ctct_ff2"]),
            "ks_conj": int(kv["ks_conj_ff2"]),
        },
    }
