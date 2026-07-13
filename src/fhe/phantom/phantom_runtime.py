from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.engines.ckks_engine_phantom import CKKSContext as PhantomCKKSContext


@dataclass
class LevelAttempt:
    level: int
    ok: bool
    elapsed_sec: float
    error: Optional[str] = None


def _coeff_chain_from_counts(log_coeff_count: int, special_prime_count: int) -> List[int]:

    import os as _os

    body = max(2, int(log_coeff_count))
    special = max(1, int(special_prime_count))
    body_bits = int(_os.getenv("ENCFORMER_BODY_PRIME_BITS", "40"))
    return [60] + [body_bits] * body + [60] * special


def _sparse_pow2_steps(nslots: int) -> List[int]:
    steps = {0}
    bit = 1
    half = nslots // 2
    while bit <= half:
        steps.add(bit)
        steps.add(-bit)
        bit <<= 1
    return sorted(steps)


def _token128_steps(nslots: int) -> List[int]:

    steps = set(_sparse_pow2_steps(nslots))
    half = nslots // 2

    local = min(127, half)
    for s in range(1, local + 1):
        steps.add(s)
        steps.add(-s)

    seg = 128
    if seg <= half:
        for k in range(1, (half // seg) + 1):
            v = k * seg
            steps.add(v)
            steps.add(-v)
    return sorted(steps)


def _seg128_steps(nslots: int) -> List[int]:

    steps = set(_sparse_pow2_steps(nslots))
    half = nslots // 2
    seg = 128
    if seg <= half:
        for k in range(1, (half // seg) + 1):
            v = k * seg
            steps.add(v)
            steps.add(-v)
    return sorted(steps)


class LevelCKKSContext(PhantomCKKSContext):
    __slots__ = ("_default_enc_level",)

    def __init__(
        self,
        *,
        default_enc_level: Optional[int],
        mode: str = "gpu",
        thread_count: int = 0,
        log_coeff_count: int = 15,
        special_prime_count: int = 4,
        nslots: int = 16384,
        poly_modulus_degree: Optional[int] = None,
        coeff_modulus_bits: Optional[Sequence[int]] = None,
        special_modulus_size: Optional[int] = None,
        scale: float = 2.0**40,
        galois_steps: Optional[Sequence[int]] = None,
        auto_rescale_mul_pt: bool = False,
        auto_rescale_mul_ct: bool = False,
    ) -> None:

        del mode, thread_count
        env_scale = os.getenv("PHANTOM_CKKS_SCALE")
        if env_scale:
            scale = float(env_scale)
        mul_tech = os.getenv("PHANTOM_MUL_TECH")
        if mul_tech is not None:
            mul_tech = mul_tech.strip().lower()
            if not mul_tech:
                mul_tech = None
        enc_mode = os.getenv("PHANTOM_ENCRYPT_MODE", "").strip().lower()
        encrypt_symmetric = enc_mode in {"sym", "symmetric", "sk"}
        gmode = os.getenv("PHANTOM_GALOIS_MODE", "").lower().strip()
        if galois_steps is None and gmode in {"sparse", "pow2", "sparse_pow2"}:
            galois_steps = _sparse_pow2_steps(int(nslots))
        elif galois_steps is None and gmode in {"token128", "dense128", "bert128"}:
            galois_steps = _token128_steps(int(nslots))
        elif galois_steps is None and gmode in {"seg128", "m128"}:
            galois_steps = _seg128_steps(int(nslots))
        if coeff_modulus_bits is None:
            coeff_modulus_bits = _coeff_chain_from_counts(
                log_coeff_count=log_coeff_count,
                special_prime_count=special_prime_count,
            )

        if special_modulus_size is None:
            special_modulus_size = int(special_prime_count)
        use_level = 1 if default_enc_level is None else int(default_enc_level)
        super().__init__(
            nslots=nslots,
            poly_modulus_degree=poly_modulus_degree,
            coeff_modulus_bits=coeff_modulus_bits,
            special_modulus_size=special_modulus_size,
            scale=scale,
            mul_tech=mul_tech,
            default_chain_index=use_level,
            galois_steps=galois_steps,
            auto_rescale_mul_pt=auto_rescale_mul_pt,
            auto_rescale_mul_ct=auto_rescale_mul_ct,
            encrypt_symmetric=encrypt_symmetric,
        )
        self._default_enc_level = default_enc_level

    @property
    def default_enc_level(self) -> Optional[int]:
        return self._default_enc_level

    def zeros(self, *, level: Optional[int] = None):
        use_level = self._default_enc_level if level is None else level
        return super().zeros(level=use_level)

    def encorypt(self, arr, *, level: Optional[int] = None):
        use_level = self._default_enc_level if level is None else level
        return super().encorypt(arr, level=use_level)


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
    del mode, thread_count
    ctx = LevelCKKSContext(
        default_enc_level=1,
        log_coeff_count=log_coeff_count,
        special_prime_count=special_prime_count,
    )
    return int(ctx._max_chain_index)


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
