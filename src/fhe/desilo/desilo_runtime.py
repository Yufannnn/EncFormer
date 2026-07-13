from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.engines.ckks_engine_desilo import CKKSContext as DesiloCKKSContext


@dataclass
class LevelAttempt:
    level: int
    ok: bool
    elapsed_sec: float
    error: Optional[str] = None


class LevelCKKSContext(DesiloCKKSContext):
    __slots__ = ("_default_enc_level",)

    def __init__(self, *, default_enc_level: Optional[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._default_enc_level = default_enc_level

    @property
    def default_enc_level(self) -> Optional[int]:
        return self._default_enc_level

    def _resolve_level(self, level: Optional[int]) -> Optional[int]:
        if level is not None:
            return level
        if self._default_enc_level is not None:
            return self._default_enc_level

        if hasattr(self.engine, "max_level"):
            return int(self.engine.max_level)
        return None

    def zeros(self, *, level: Optional[int] = None):
        return super().zeros(level=self._resolve_level(level))

    def encorypt(self, arr, *, level: Optional[int] = None):
        return super().encorypt(arr, level=self._resolve_level(level))


def set_visible_gpus(gpu_ids: Optional[str]) -> None:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def probe_max_level(
    *,
    mode: str,
    thread_count: int,
    log_coeff_count: int,
    special_prime_count: int,
) -> int:
    ctx = DesiloCKKSContext(
        mode=mode,
        thread_count=thread_count,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )
    return int(ctx.engine.max_level)


def _try_level(
    fn: Callable[[int], Dict[str, Any]],
    level: int,
) -> Tuple[bool, Optional[Dict[str, Any]], LevelAttempt]:
    t0 = time.perf_counter()
    try:
        out = fn(level)
        t1 = time.perf_counter()
        return True, out, LevelAttempt(level=level, ok=True, elapsed_sec=t1 - t0)
    except Exception as exc:
        t1 = time.perf_counter()
        err = f"{type(exc).__name__}: {exc}"
        return False, None, LevelAttempt(level=level, ok=False, elapsed_sec=t1 - t0, error=err)


def search_lowest_working_level(
    fn: Callable[[int], Dict[str, Any]],
    *,
    start_level: int,
    min_level: int,
    max_level: int,
) -> Tuple[int, Dict[str, Any], List[LevelAttempt]]:
    attempts: List[LevelAttempt] = []

    ok, out, att = _try_level(fn, start_level)
    attempts.append(att)

    if ok and out is not None:
        best_level = start_level
        best_out = out
        for level in range(start_level - 1, min_level - 1, -1):
            ok2, out2, att2 = _try_level(fn, level)
            attempts.append(att2)
            if ok2 and out2 is not None:
                best_level = level
                best_out = out2
            else:
                break
        return best_level, best_out, attempts

    first_pass_level: Optional[int] = None
    first_pass_out: Optional[Dict[str, Any]] = None
    for level in range(start_level + 1, max_level + 1):
        ok2, out2, att2 = _try_level(fn, level)
        attempts.append(att2)
        if ok2 and out2 is not None:
            first_pass_level = level
            first_pass_out = out2
            break

    if first_pass_level is None or first_pass_out is None:
        msgs = "; ".join(f"L{a.level}:{'ok' if a.ok else a.error}" for a in attempts)
        raise RuntimeError(f"No working encryption level in [{min_level}, {max_level}]. Attempts: {msgs}")

    best_level = first_pass_level
    best_out = first_pass_out
    for level in range(first_pass_level - 1, min_level - 1, -1):
        ok3, out3, att3 = _try_level(fn, level)
        attempts.append(att3)
        if ok3 and out3 is not None:
            best_level = level
            best_out = out3
        else:
            break

    return best_level, best_out, attempts


def attempts_summary(attempts: List[LevelAttempt]) -> str:
    parts = []
    for a in attempts:
        if a.ok:
            parts.append(f"L{a.level}:ok({a.elapsed_sec:.2f}s)")
        else:
            parts.append(f"L{a.level}:fail({a.error})")
    return " | ".join(parts)
