from __future__ import annotations

import contextlib
import os
import sys
import weakref
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np

from src.engines.ckks_engine_abc import CipherBase, CKKSEngine, CKKSStats, Number, Plain

_phantom_env_path = os.getenv("PHANTOM_PYTHONPATH", "").strip()
if _phantom_env_path and _phantom_env_path not in sys.path:
    sys.path.insert(0, _phantom_env_path)

try:
    import pyPhantom as phantom
except Exception as e:  # pragma: no cover
    phantom = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

__all__ = ["CKKSContext", "Cipher", "CKKSStats", "Number", "Plain"]

if TYPE_CHECKING:
    from typing import Iterable


def _require_phantom() -> None:
    if phantom is None:
        raise ImportError(
            "pyPhantom is not importable. Build PhantomFHE with "
            "PHANTOM_ENABLE_PYTHON_BINDING=ON and add its build/lib to PYTHONPATH."
        ) from _IMPORT_ERR


def _norm_steps(steps: Optional[Sequence[int]], *, nslots: int) -> list[int]:
    out = set(int(s) for s in (steps or []))
    out.add(0)
    norm: set[int] = set()
    half = nslots // 2
    for s in out:
        d = int(s) % nslots
        if d > half:
            d -= nslots
        norm.add(d)
    return sorted(norm)


def _default_galois_steps(nslots: int) -> list[int]:

    steps = {0}

    bit = 1
    while bit <= (nslots // 2):
        steps.add(bit)
        steps.add(-bit)
        bit <<= 1

    local = min(127, nslots // 2)
    for s in range(1, local + 1):
        steps.add(s)
        steps.add(-s)
    return sorted(steps)


class CKKSContext(CKKSEngine):
    __slots__ = (
        "_context",
        "_encoder",
        "_sk",
        "_pk",
        "_rlk",
        "_glk",
        "_scale",
        "_default_chain_index",
        "_max_chain_index",
        "_nslots",
        "_stats",
        "_decorypt_enabled",
        "_scalar_pt_cache",
        "_vec_pt_cache",
        "_rot_decomp_cache",
        "_auto_rescale_mul_pt",
        "_auto_rescale_mul_ct",
        "_encrypt_symmetric",
    )

    def __init__(
        self,
        nslots: int = 16384,
        *,
        poly_modulus_degree: Optional[int] = None,
        coeff_modulus_bits: Optional[Sequence[int]] = None,
        special_modulus_size: int = 1,
        scale: float = 2.0**40,
        mul_tech: Optional[str | Any] = None,
        default_chain_index: int = 1,
        galois_steps: Optional[Sequence[int]] = None,
        auto_rescale_mul_pt: bool = False,
        auto_rescale_mul_ct: bool = False,
        encrypt_symmetric: bool = False,
    ) -> None:
        _require_phantom()

        nslots = int(nslots)
        poly_degree = int(poly_modulus_degree) if poly_modulus_degree is not None else int(2 * nslots)
        if coeff_modulus_bits is None:
            coeff_modulus_bits = [60, 40, 40, 40, 40, 40, 40, 40, 40, 60]
        bits = [int(x) for x in coeff_modulus_bits]
        chain_index = int(default_chain_index)
        max_chain_index = max(1, len(bits) - 1)
        if chain_index < 1 or chain_index > max_chain_index:
            raise ValueError(f"default_chain_index must be in [1, {max_chain_index}]")

        params = phantom.params(phantom.scheme_type.ckks)
        params.set_poly_modulus_degree(poly_degree)
        params.set_coeff_modulus(phantom.create_coeff_modulus(poly_degree, bits))
        params.set_special_modulus_size(int(special_modulus_size))
        if mul_tech is not None:
            tech = mul_tech
            if isinstance(tech, str):
                name = tech.strip().lower()
                if not name:
                    tech = None
                else:
                    tech = getattr(phantom.mul_tech_type, name, None)
                    if tech is None:
                        raise ValueError("mul_tech must be one of: none, behz, hps, hps_overq, hps_overq_leveled")
            if tech is not None:
                try:
                    params.set_mul_tech(tech)
                except Exception as exc:
                    if "only supported for BFV" not in str(exc):
                        raise
        if galois_steps is None:
            galois_steps = _default_galois_steps(nslots)
        steps = _norm_steps(galois_steps, nslots=nslots)
        params.set_galois_elts(phantom.get_elts_from_steps(steps, poly_degree))

        ctx = phantom.context(params)
        enc = phantom.ckks_encoder(ctx)
        slot_count = int(enc.slot_count())
        if slot_count != nslots:
            raise ValueError(
                f"Requested nslots={nslots}, but Phantom CKKS slot_count={slot_count}. "
                f"Use poly_modulus_degree={2 * nslots} or adjust nslots."
            )

        super().__init__(nslots=slot_count)
        self._context = ctx
        self._encoder = enc
        self._sk = phantom.secret_key(ctx)
        self._pk = self._sk.gen_publickey(ctx)
        self._rlk = self._sk.gen_relinkey(ctx)
        self._glk = self._sk.create_galois_keys(ctx)
        self._scale = float(scale)
        self._default_chain_index = chain_index
        self._max_chain_index = max_chain_index
        self._encrypt_symmetric = bool(encrypt_symmetric)
        self._scalar_pt_cache: Dict[tuple[float, int, float], Any] = {}
        self._vec_pt_cache: Dict[tuple[int, int, float], tuple[weakref.ref[np.ndarray], Any]] = {}
        self._rot_decomp_cache: Dict[int, tuple[int, ...]] = {}
        self._auto_rescale_mul_pt = bool(auto_rescale_mul_pt)
        self._auto_rescale_mul_ct = bool(auto_rescale_mul_ct)

    @property
    def nslots(self) -> int:
        return self._nslots

    @property
    def stats(self) -> CKKSStats:
        return self._stats

    @property
    def context(self):
        return self._context

    def _check_decorypt_allowed(self) -> None:
        if self._decorypt_enabled <= 0:
            raise PermissionError("decorypt() is disabled; use 'with engine.decorypt_scope(): ...'")

    @contextlib.contextmanager
    def decorypt_scope(self):
        self._decorypt_enabled += 1
        try:
            yield
        finally:
            self._decorypt_enabled -= 1

    def _pad_complex(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.complex128).ravel()
        if a.size > self.nslots:
            raise ValueError(f"Input length {a.size} exceeds nslots {self.nslots}")
        if a.size == self.nslots:
            return a
        out = np.zeros(self.nslots, dtype=np.complex128)
        out[: a.size] = a
        return out

    def _encode_real_plain(self, vals: np.ndarray, *, chain_index: int, scale: Optional[float] = None):
        ci = int(chain_index)
        if ci < 1 or ci > self._max_chain_index:
            raise ValueError(f"chain_index must be in [1, {self._max_chain_index}]")
        arr = np.asarray(vals, dtype=np.float64).ravel()
        if arr.size > self.nslots:
            raise ValueError(f"Plain length {arr.size} exceeds nslots {self.nslots}")
        if arr.size < self.nslots:
            pad = np.zeros(self.nslots, dtype=np.float64)
            pad[: arr.size] = arr
            arr = pad
        sc = float(self._scale if scale is None else scale)

        try:
            return self._encoder.encode_double_vector(
                self._context,
                arr,
                sc,
                chain_index=ci,
            )
        except Exception:
            pass
        pt = self._encoder.encode_double_vector(
            self._context,
            arr,
            sc,
            chain_index=1,
        )
        if ci > 1:
            pt = phantom.mod_switch_to(self._context, pt, ci)
        return pt

    def _encode_scalar_plain(
        self,
        value: float,
        *,
        chain_index: int,
        scale: Optional[float] = None,
    ):
        sc = float(self._scale if scale is None else scale)
        key = (float(value), int(chain_index), sc)
        if key in self._scalar_pt_cache:
            return self._scalar_pt_cache[key]
        vals = np.full(self.nslots, float(value), dtype=np.float64)
        pt = self._encode_real_plain(vals, chain_index=chain_index, scale=sc)
        if len(self._scalar_pt_cache) < 32:
            self._scalar_pt_cache[key] = pt
        return pt

    def _encode_real_plain_cached(self, vals: np.ndarray, *, chain_index: int, scale: Optional[float] = None):
        arr = np.asarray(vals, dtype=np.float64).ravel()
        ci = int(chain_index)
        sc = float(self._scale if scale is None else scale)
        if arr.size != self.nslots or arr.flags.writeable:
            return self._encode_real_plain(arr, chain_index=ci, scale=sc)
        key = (id(arr), ci, sc)
        hit = self._vec_pt_cache.get(key)
        if hit is not None:
            aref, pt = hit
            if aref() is arr:
                return pt
            self._vec_pt_cache.pop(key, None)
        pt = self._encode_real_plain(arr, chain_index=ci, scale=sc)
        try:
            aref = weakref.ref(arr)
        except TypeError:
            return pt
        if len(self._vec_pt_cache) >= 512:
            self._vec_pt_cache.clear()
        self._vec_pt_cache[key] = (aref, pt)
        return pt

    def zeros(self, *, level: Optional[int] = None) -> "Cipher":
        return self.encorypt(0.0, level=level)

    def encorypt(self, arr: Plain, *, level: Optional[int] = None) -> "Cipher":
        target_idx = self._default_chain_index if level is None else int(level)
        if target_idx < 1 or target_idx > self._max_chain_index:
            raise ValueError(f"encorypt level/chain_index must be in [1, {self._max_chain_index}]")

        if np.isscalar(arr):
            z = complex(arr)
            pt_r = self._encode_scalar_plain(float(z.real), chain_index=1)
            if self._encrypt_symmetric:
                ct_r = self._sk.encrypt_symmetric(self._context, pt_r)
            else:
                ct_r = self._pk.encrypt_asymmetric(self._context, pt_r)
            if abs(z.imag) > 0.0:
                pt_i = self._encode_scalar_plain(float(z.imag), chain_index=1)
                if self._encrypt_symmetric:
                    ct_i = self._sk.encrypt_symmetric(self._context, pt_i)
                else:
                    ct_i = self._pk.encrypt_asymmetric(self._context, pt_i)
            else:
                ct_i = None
            if target_idx > 1:
                ct_r = phantom.mod_switch_to(self._context, ct_r, target_idx)
                ct_i = phantom.mod_switch_to(self._context, ct_i, target_idx) if ct_i is not None else None
            return Cipher(self, ct_r, ct_i, chain_index=target_idx)

        v = self._pad_complex(np.asarray(arr, dtype=np.complex128))
        pt_r = self._encode_real_plain(v.real, chain_index=1)
        if self._encrypt_symmetric:
            ct_r = self._sk.encrypt_symmetric(self._context, pt_r)
        else:
            ct_r = self._pk.encrypt_asymmetric(self._context, pt_r)
        if np.any(v.imag):
            pt_i = self._encode_real_plain(v.imag, chain_index=1)
            if self._encrypt_symmetric:
                ct_i = self._sk.encrypt_symmetric(self._context, pt_i)
            else:
                ct_i = self._pk.encrypt_asymmetric(self._context, pt_i)
        else:
            ct_i = None
        if target_idx > 1:
            ct_r = phantom.mod_switch_to(self._context, ct_r, target_idx)
            ct_i = phantom.mod_switch_to(self._context, ct_i, target_idx) if ct_i is not None else None
        return Cipher(self, ct_r, ct_i, chain_index=target_idx)

    def conjugate(self, ct: "Cipher") -> "Cipher":
        if not isinstance(ct, Cipher) or ct.engine is not self:
            raise TypeError("ct must be a ciphertext from this context")
        return ct.conjugate()

    def _same_engine(self, other: "CKKSContext") -> bool:
        return self is other

    def _decode_real_ct(self, ct_obj) -> np.ndarray:
        cur = ct_obj
        last_scale_exc: Exception | None = None
        scale_reset_done = False

        for _ in range(self._max_chain_index + 2):
            try:
                pt = self._sk.decrypt(self._context, cur)
                out = self._encoder.decode_double_vector(self._context, pt)
                arr = np.asarray(out, dtype=np.float64)
                if arr.size != self.nslots:
                    raise RuntimeError(f"Decoded slot count mismatch: got {arr.size}, expected {self.nslots}")
                return arr
            except Exception as exc:
                if "scale out of bounds" not in str(exc):
                    raise
                last_scale_exc = exc

                if not scale_reset_done:
                    try:
                        cur.set_scale(float(self._scale))
                        scale_reset_done = True
                        continue
                    except Exception:
                        pass
                try:
                    cur = phantom.rescale_to_next(self._context, cur)
                except Exception:
                    break
        if last_scale_exc is not None:
            raise last_scale_exc
        raise RuntimeError("Failed to decode ciphertext")

    def add_many(self, cts: "Iterable[Cipher]") -> "Cipher":
        seq = list(cts)
        if not seq:
            return self.zeros()
        first = seq[0]
        if not isinstance(first, Cipher) or first.engine is not self:
            raise TypeError("add_many expects ciphertexts from this context")
        for ct in seq[1:]:
            if not isinstance(ct, Cipher) or ct.engine is not self:
                raise TypeError("add_many expects ciphertexts from this context")

        same_chain = all(ct._chain_index == first._chain_index for ct in seq)
        same_scale = all(Cipher._scale_close(ct._scale, first._scale) for ct in seq)
        if not same_chain or not same_scale:
            acc = first
            for ct in seq[1:]:
                acc = acc.add(ct)
            return acc

        raw_r = [ct._ct_r for ct in seq if ct._ct_r is not None]
        if not raw_r:
            raise ValueError("add_many received ciphertexts without real components")
        if len(raw_r) == 1:
            out_r = raw_r[0]
        else:
            out_r = phantom.ciphertext()
            phantom.add_many(self._context, raw_r, out_r)

        raw_i = [ct._ct_i for ct in seq if ct._ct_i is not None]
        if not raw_i:
            out_i = None
        elif len(raw_i) == 1:
            out_i = raw_i[0]
        else:
            out_i = phantom.ciphertext()
            phantom.add_many(self._context, raw_i, out_i)

        if len(seq) > 1:
            self.stats.ops_add += len(seq) - 1
        return Cipher(
            self,
            out_r,
            out_i,
            chain_index=first._chain_index,
            scale=first._scale,
        )


class Cipher(CipherBase):
    __slots__ = ("_engine", "_ct_r", "_ct_i", "_chain_index", "_scale")

    def __init__(
        self,
        engine: CKKSContext,
        ct_r: Any,
        ct_i: Any = None,
        *,
        chain_index: int = 1,
        scale: Optional[float] = None,
    ):
        if not isinstance(engine, CKKSContext):
            raise TypeError("engine must be CKKSContext")
        self._engine = engine
        self._ct_r = ct_r
        self._ct_i = ct_i
        self._chain_index = int(chain_index)
        self._scale = float(engine._scale if scale is None else scale)
        self._engine.stats.cts_created += 1

    @property
    def engine(self) -> CKKSContext:
        return self._engine

    def _check_same_engine(self, other: "Cipher") -> None:
        if not isinstance(other, Cipher) or self.engine is not other.engine:
            raise TypeError("ciphertexts must share the same engine")

    def _add_ct(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        return phantom.add(self.engine.context, a, b)

    def _sub_ct(self, a, b):
        if a is None and b is None:
            return None
        if a is None:
            return phantom.negate(self.engine.context, b)
        if b is None:
            return a
        return phantom.sub(self.engine.context, a, b)

    def _mul_plain(self, ct, vec: np.ndarray):
        if ct is None:
            return None
        pt = self.engine._encode_real_plain(vec, chain_index=self._chain_index)
        return self._mul_plain_encoded(ct, pt)

    def _mul_plain_encoded(self, ct, pt):
        if ct is None:
            return None
        return phantom.multiply_plain(self.engine.context, ct, pt)

    @staticmethod
    def _is_mask_like(arr: np.ndarray) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        if a.size == 0:
            return False
        r = np.rint(a)
        if np.max(np.abs(a - r)) > 1e-12:
            return False
        return np.max(np.abs(r)) <= 1.0

    def _mod_switch_ct(self, ct, from_chain: int, to_chain: int):
        if ct is None or from_chain == to_chain:
            return ct
        return phantom.mod_switch_to(self.engine.context, ct, int(to_chain))

    @staticmethod
    def _scale_close(a: float, b: float, *, rtol: float = 1e-3) -> bool:
        aa = abs(float(a))
        bb = abs(float(b))
        return abs(aa - bb) <= rtol * max(aa, bb, 1.0)

    def _set_pair_scale(self, ct_r, ct_i, scale: float) -> None:
        s = float(scale)
        if ct_r is not None:
            ct_r.set_scale(s)
        if ct_i is not None:
            ct_i.set_scale(s)

    def _mod_switch_pair_to(self, ct_r, ct_i, from_chain: int, to_chain: int):
        if from_chain == to_chain:
            return ct_r, ct_i, int(to_chain)
        if from_chain > to_chain:
            raise ValueError(f"cannot mod-switch upwards from chain {from_chain} to {to_chain} in Phantom backend")
        out_r = phantom.mod_switch_to(self.engine.context, ct_r, int(to_chain)) if ct_r is not None else None
        out_i = phantom.mod_switch_to(self.engine.context, ct_i, int(to_chain)) if ct_i is not None else None
        return out_r, out_i, int(to_chain)

    def _rescale_pair_once(self, ct_r, ct_i, chain: int, scale: float):
        if chain >= self.engine._max_chain_index:
            raise ValueError(f"cannot rescale at chain {chain}; max chain is {self.engine._max_chain_index}")
        out_r = phantom.rescale_to_next(self.engine.context, ct_r) if ct_r is not None else None
        out_i = phantom.rescale_to_next(self.engine.context, ct_i) if ct_i is not None else None
        out_chain = int(chain) + 1
        out_scale = float(scale) / float(self.engine._scale)
        self._set_pair_scale(out_r, out_i, out_scale)
        return out_r, out_i, out_chain, out_scale

    def _raise_pair_scale_once(self, ct_r, ct_i, chain: int, scale: float):
        pt_one = self.engine._encode_scalar_plain(1.0, chain_index=int(chain))
        out_r = self._mul_plain_encoded(ct_r, pt_one) if ct_r is not None else None
        out_i = self._mul_plain_encoded(ct_i, pt_one) if ct_i is not None else None
        out_scale = float(scale) * float(self.engine._scale)
        self._set_pair_scale(out_r, out_i, out_scale)
        return out_r, out_i, out_scale

    def _align_for_binary_op(
        self,
        ar,
        ai,
        ac: int,
        ascale: float,
        br,
        bi,
        bc: int,
        bscale: float,
        *,
        op_name: str,
    ):
        ar2, ai2, ac2, as2 = ar, ai, int(ac), float(ascale)
        br2, bi2, bc2, bs2 = br, bi, int(bc), float(bscale)
        base = max(float(self.engine._scale), 2.0)
        hi_ratio = base / 2.0
        lo_ratio = 1.0 / hi_ratio

        max_iters = 64
        for _ in range(max_iters):
            if bs2 == 0.0 or as2 == 0.0:
                break
            ratio = as2 / bs2
            if ratio > hi_ratio:
                if ac2 >= self.engine._max_chain_index:
                    break
                ar2, ai2, ac2, as2 = self._rescale_pair_once(ar2, ai2, ac2, as2)
                continue
            if ratio < lo_ratio:
                if bc2 >= self.engine._max_chain_index:
                    break
                br2, bi2, bc2, bs2 = self._rescale_pair_once(br2, bi2, bc2, bs2)
                continue
            break

        tc = max(ac2, bc2)
        ar2, ai2, ac2 = self._mod_switch_pair_to(ar2, ai2, ac2, tc)
        br2, bi2, bc2 = self._mod_switch_pair_to(br2, bi2, bc2, tc)
        del ac2, bc2

        for _ in range(128):
            min_scale = max(min(abs(as2), abs(bs2)), 1e-30)
            ratio = max(abs(as2), abs(bs2)) / min_scale
            if ratio <= 1.5:
                break
            if abs(as2) < abs(bs2):
                ar2, ai2, as2 = self._raise_pair_scale_once(ar2, ai2, tc, as2)
            else:
                br2, bi2, bs2 = self._raise_pair_scale_once(br2, bi2, tc, bs2)

        min_scale = max(min(abs(as2), abs(bs2)), 1e-30)
        ratio = max(abs(as2), abs(bs2)) / min_scale
        if ratio > 1.5:
            raise ValueError(f"scale mismatch in {op_name}(). unable to align safely (ratio={ratio:.3e})")

        target_scale = min(abs(as2), abs(bs2))
        if target_scale <= 0.0:
            target_scale = float(self.engine._scale)
        self._set_pair_scale(ar2, ai2, target_scale)
        self._set_pair_scale(br2, bi2, target_scale)
        return ar2, ai2, br2, bi2, tc, float(target_scale)

    def _auto_rescale_pair(self, ct_r, ct_i, chain: int, scale: float, *, enabled: bool):
        if not enabled:
            return ct_r, ct_i, int(chain), float(scale)
        if int(chain) >= self.engine._max_chain_index:
            return ct_r, ct_i, int(chain), float(scale)
        out_r = phantom.rescale_to_next(self.engine.context, ct_r) if ct_r is not None else None
        out_i = phantom.rescale_to_next(self.engine.context, ct_i) if ct_i is not None else None
        out_chain = int(chain) + 1
        out_scale = float(scale) / float(self.engine._scale)
        self._set_pair_scale(out_r, out_i, out_scale)
        return out_r, out_i, out_chain, out_scale

    def _add_plain(self, other: Plain) -> "Cipher":
        if np.isscalar(other):
            z = complex(other)
            v = np.full(self.nslots, z, dtype=np.complex128)
        else:
            v = self.engine._pad_complex(np.asarray(other, dtype=np.complex128))
        pt_r = self.engine._encode_real_plain_cached(
            v.real.astype(np.float64, copy=False),
            chain_index=self._chain_index,
            scale=self._scale,
        )
        out_r = phantom.add_plain(self.engine.context, self._ct_r, pt_r)

        out_i = self._ct_i
        if np.any(v.imag):
            pt_i = self.engine._encode_real_plain_cached(
                v.imag.astype(np.float64, copy=False),
                chain_index=self._chain_index,
                scale=self._scale,
            )
            if out_i is None:
                out_i = phantom.sub(self.engine.context, self._ct_r, self._ct_r)
            out_i = phantom.add_plain(self.engine.context, out_i, pt_i)
        return Cipher(
            self.engine,
            out_r,
            out_i,
            chain_index=self._chain_index,
            scale=self._scale,
        )

    def mul_pt_parts(
        self,
        pr: np.ndarray,
        pi: np.ndarray | None = None,
        *,
        has_pi: bool | None = None,
        scale: Optional[float] = None,
    ) -> "Cipher":
        self.engine.stats.ops_mul_pt += 1
        ar = self._ct_r
        ai = self._ct_i
        pt_scale = float(self.engine._scale if scale is None else scale)

        if has_pi is None:
            has_pi = bool(pi is not None and np.any(np.asarray(pi)))

        pr_arr = np.asarray(pr, dtype=np.float64).ravel()
        pt_pr = self.engine._encode_real_plain_cached(pr_arr, chain_index=self._chain_index, scale=scale)

        if not has_pi:
            out_r = self._mul_plain_encoded(ar, pt_pr)
            out_i = self._mul_plain_encoded(ai, pt_pr) if ai is not None else None
            out_r, out_i, out_chain, out_scale = self._auto_rescale_pair(
                out_r,
                out_i,
                self._chain_index,
                self._scale * pt_scale,
                enabled=self.engine._auto_rescale_mul_pt,
            )
            return Cipher(
                self.engine,
                out_r,
                out_i,
                chain_index=out_chain,
                scale=out_scale,
            )

        if pi is None:
            raise ValueError("pi must be provided when has_pi=True")
        pi_arr = np.asarray(pi, dtype=np.float64).ravel()
        pt_pi = self.engine._encode_real_plain_cached(pi_arr, chain_index=self._chain_index, scale=scale)

        ar_pr = self._mul_plain_encoded(ar, pt_pr)
        ai_pi = self._mul_plain_encoded(ai, pt_pi) if ai is not None else None
        out_r = self._sub_ct(ar_pr, ai_pi)

        ar_pi = self._mul_plain_encoded(ar, pt_pi)
        ai_pr = self._mul_plain_encoded(ai, pt_pr) if ai is not None else None
        out_i = self._add_ct(ar_pi, ai_pr)
        out_r, out_i, out_chain, out_scale = self._auto_rescale_pair(
            out_r,
            out_i,
            self._chain_index,
            self._scale * pt_scale,
            enabled=self.engine._auto_rescale_mul_pt,
        )
        return Cipher(
            self.engine,
            out_r,
            out_i,
            chain_index=out_chain,
            scale=out_scale,
        )

    def add(self, other) -> "Cipher":
        if not isinstance(other, Cipher):
            if isinstance(other, (np.ndarray, list, tuple, int, float, complex, np.number)):
                self.engine.stats.ops_add += 1
                return self._add_plain(other)
            else:
                raise TypeError("add expects Cipher or plain scalar/vector")
        self._check_same_engine(other)
        self.engine.stats.ops_add += 1

        def _try_add(ar, ai, ac, br, bi, bc):
            if ac == bc and self._scale_close(self._scale, other._scale):
                try:
                    out_r = self._add_ct(ar, br)
                    out_i = self._add_ct(ai, bi)
                    return out_r, out_i, ac, float(min(abs(self._scale), abs(other._scale)))
                except Exception as exc:
                    if "scale mismatch" not in str(exc):
                        raise
            aar, aai, bbr, bbi, tc, aligned_scale = self._align_for_binary_op(
                ar,
                ai,
                ac,
                self._scale,
                br,
                bi,
                bc,
                other._scale,
                op_name="add",
            )
            out_r = self._add_ct(aar, bbr)
            out_i = self._add_ct(aai, bbi)
            return out_r, out_i, tc, aligned_scale

        ar, ai, ac = self._ct_r, self._ct_i, self._chain_index
        br, bi, bc = other._ct_r, other._ct_i, other._chain_index
        try:
            out_r, out_i, tc, aligned_scale = _try_add(ar, ai, ac, br, bi, bc)
            return Cipher(self.engine, out_r, out_i, chain_index=tc, scale=aligned_scale)
        except Exception as exc:
            if "scale mismatch" in str(exc):
                raise ValueError(
                    "scale mismatch in add(). Phantom backend could not safely align CKKS scales."
                ) from exc
            raise

    def add_inplace(self, other: "Cipher") -> "Cipher":
        if not isinstance(other, Cipher):
            raise TypeError("add_inplace expects Cipher rhs")
        self._check_same_engine(other)
        self.engine.stats.ops_add += 1
        if self._chain_index == other._chain_index and self._scale_close(self._scale, other._scale):
            try:
                self._ct_r = self._add_ct(self._ct_r, other._ct_r)
                self._ct_i = self._add_ct(self._ct_i, other._ct_i)
                self._scale = float(min(abs(self._scale), abs(other._scale)))
                return self
            except Exception as exc:
                if "scale mismatch" not in str(exc):
                    raise
        aar, aai, bbr, bbi, tc, aligned_scale = self._align_for_binary_op(
            self._ct_r,
            self._ct_i,
            self._chain_index,
            self._scale,
            other._ct_r,
            other._ct_i,
            other._chain_index,
            other._scale,
            op_name="add",
        )
        self._ct_r = self._add_ct(aar, bbr)
        self._ct_i = self._add_ct(aai, bbi)
        self._chain_index = int(tc)
        self._scale = float(aligned_scale)
        return self

    def mul_scalar(self, s: Number) -> "Cipher":
        z = complex(s)

        if z == 1:
            return self
        if z == -1:
            self.engine.stats.ops_mul_pt += 1
            out_r = phantom.negate(self.engine.context, self._ct_r)
            out_i = phantom.negate(self.engine.context, self._ct_i) if self._ct_i is not None else None
            return Cipher(self.engine, out_r, out_i, chain_index=self._chain_index, scale=self._scale)
        if z == 1j:
            self.engine.stats.ops_mul_pt += 1
            if self._ct_i is not None:
                out_r = phantom.negate(self.engine.context, self._ct_i)
            else:
                out_r = phantom.sub(self.engine.context, self._ct_r, self._ct_r)
            out_i = self._ct_r
            return Cipher(self.engine, out_r, out_i, chain_index=self._chain_index, scale=self._scale)
        if z == -1j:
            self.engine.stats.ops_mul_pt += 1
            if self._ct_i is not None:
                out_r = self._ct_i
            else:
                out_r = phantom.sub(self.engine.context, self._ct_r, self._ct_r)
            out_i = phantom.negate(self.engine.context, self._ct_r)
            return Cipher(self.engine, out_r, out_i, chain_index=self._chain_index, scale=self._scale)
        return self.mul_pt(z)

    def mul_pt(self, other: Plain) -> "Cipher":
        if np.isscalar(other):
            z = complex(other)
            self.engine.stats.ops_mul_pt += 1
            pt_pr = self.engine._encode_scalar_plain(float(z.real), chain_index=self._chain_index)
            out_scale = self._scale * float(self.engine._scale)
            if abs(z.imag) > 0.0:
                pt_pi = self.engine._encode_scalar_plain(float(z.imag), chain_index=self._chain_index)
                ar = self._ct_r
                ai = self._ct_i
                ar_pr = self._mul_plain_encoded(ar, pt_pr)
                ai_pi = self._mul_plain_encoded(ai, pt_pi) if ai is not None else None
                out_r = self._sub_ct(ar_pr, ai_pi)
                ar_pi = self._mul_plain_encoded(ar, pt_pi)
                ai_pr = self._mul_plain_encoded(ai, pt_pr) if ai is not None else None
                out_i = self._add_ct(ar_pi, ai_pr)
                out_r, out_i, out_chain, out_scale = self._auto_rescale_pair(
                    out_r,
                    out_i,
                    self._chain_index,
                    out_scale,
                    enabled=self.engine._auto_rescale_mul_pt,
                )
                return Cipher(
                    self.engine,
                    out_r,
                    out_i,
                    chain_index=out_chain,
                    scale=out_scale,
                )

            out_r = self._mul_plain_encoded(self._ct_r, pt_pr)
            out_i = self._mul_plain_encoded(self._ct_i, pt_pr) if self._ct_i is not None else None
            out_r, out_i, out_chain, out_scale = self._auto_rescale_pair(
                out_r,
                out_i,
                self._chain_index,
                out_scale,
                enabled=self.engine._auto_rescale_mul_pt,
            )
            return Cipher(
                self.engine,
                out_r,
                out_i,
                chain_index=out_chain,
                scale=out_scale,
            )

        v = self.engine._pad_complex(np.asarray(other, dtype=np.complex128))
        pr = v.real.astype(np.float64, copy=False)
        pi = v.imag.astype(np.float64, copy=False)
        has_pi = bool(np.any(pi))
        return self.mul_pt_parts(pr, pi if has_pi else None, has_pi=has_pi, scale=None)

    def _mul_ct_base(self, a, b, *, relin: bool):
        if a is None or b is None:
            return None
        if relin:
            return phantom.multiply_and_relin(self.engine.context, a, b, self.engine._rlk)
        return phantom.multiply(self.engine.context, a, b)

    def mul_ct(self, other: "Cipher", relin: bool = True) -> "Cipher":
        self._check_same_engine(other)
        st = self.engine.stats
        st.ops_mul_ct += 1
        st.ks_muls_ctct += 1
        if relin:
            st.ks_relin += 1

        def _try_mul(ar, ai, ac, br, bi, bc):
            if ac == bc and self._scale_close(self._scale, other._scale):
                try:
                    acm = self._mul_ct_base(ar, br, relin=relin)
                    bdm = self._mul_ct_base(ai, bi, relin=relin)
                    adm = self._mul_ct_base(ar, bi, relin=relin)
                    bcm = self._mul_ct_base(ai, br, relin=relin)
                    out_r = self._sub_ct(acm, bdm)
                    out_i = self._add_ct(adm, bcm)
                    aligned_scale = float(min(abs(self._scale), abs(other._scale)))
                    return out_r, out_i, ac, aligned_scale
                except Exception as exc:
                    if "scale mismatch" not in str(exc):
                        raise
            a, b, c, d, tc, aligned_scale = self._align_for_binary_op(
                ar,
                ai,
                ac,
                self._scale,
                br,
                bi,
                bc,
                other._scale,
                op_name="mul_ct",
            )
            acm = self._mul_ct_base(a, c, relin=relin)
            bdm = self._mul_ct_base(b, d, relin=relin)
            adm = self._mul_ct_base(a, d, relin=relin)
            bcm = self._mul_ct_base(b, c, relin=relin)
            out_r = self._sub_ct(acm, bdm)
            out_i = self._add_ct(adm, bcm)
            return out_r, out_i, tc, aligned_scale

        ar, ai, ac = self._ct_r, self._ct_i, self._chain_index
        br, bi, bc = other._ct_r, other._ct_i, other._chain_index
        try:
            out_r, out_i, tc, aligned_scale = _try_mul(ar, ai, ac, br, bi, bc)
            out_r, out_i, tc, out_scale = self._auto_rescale_pair(
                out_r,
                out_i,
                tc,
                aligned_scale * aligned_scale,
                enabled=self.engine._auto_rescale_mul_ct,
            )
            return Cipher(
                self.engine,
                out_r,
                out_i,
                chain_index=tc,
                scale=out_scale,
            )
        except Exception as exc:
            if "scale mismatch" in str(exc):
                raise ValueError(
                    "scale mismatch in mul_ct(). Phantom backend could not safely align CKKS scales."
                ) from exc
            raise

    def rot(self, k: int) -> "Cipher":
        if not isinstance(k, (int, np.integer)):
            raise TypeError("k must be int")
        n = self.nslots
        d = int(k) % n
        if d == 0:
            return self
        step = d if d <= (n // 2) else d - n
        ks_count = 0
        try:
            out_r = phantom.rotate(self.engine.context, self._ct_r, int(step), self.engine._glk)
            out_i = (
                phantom.rotate(self.engine.context, self._ct_i, int(step), self.engine._glk)
                if self._ct_i is not None
                else None
            )
            ks_count = 1
        except Exception as exc:
            if "Galois key not present" not in str(exc):
                raise
            substeps = self.engine._rot_decomp_cache.get(int(step))
            if substeps is None:
                rem = abs(int(step))
                if rem == 0:
                    return self
                sign = 1 if step > 0 else -1
                parts: list[int] = []
                bit = 1
                while rem:
                    if rem & 1:
                        parts.append(sign * bit)
                    rem >>= 1
                    bit <<= 1
                substeps = tuple(parts)
                if len(self.engine._rot_decomp_cache) < 512:
                    self.engine._rot_decomp_cache[int(step)] = substeps
            cur_r = self._ct_r
            cur_i = self._ct_i
            for s in substeps:
                cur_r = phantom.rotate(self.engine.context, cur_r, int(s), self.engine._glk)
                if cur_i is not None:
                    cur_i = phantom.rotate(self.engine.context, cur_i, int(s), self.engine._glk)
            out_r = cur_r
            out_i = cur_i
            ks_count = len(substeps)
        st = self.engine.stats
        st.ops_rotate += ks_count
        st.ks_rots += ks_count
        return Cipher(self.engine, out_r, out_i, chain_index=self._chain_index, scale=self._scale)

    def conjugate(self) -> "Cipher":
        st = self.engine.stats
        st.ops_conj += 1
        st.ks_conj += 1
        if self._ct_i is None:
            return self
        out_i = phantom.negate(self.engine.context, self._ct_i)
        return Cipher(
            self.engine,
            self._ct_r,
            out_i,
            chain_index=self._chain_index,
            scale=self._scale,
        )

    def relinearize(self) -> "Cipher":
        fn = getattr(phantom, "relinearize", None)
        if fn is None:
            return self
        out_r = fn(self.engine.context, self._ct_r, self.engine._rlk) if self._ct_r is not None else None
        out_i = fn(self.engine.context, self._ct_i, self.engine._rlk) if self._ct_i is not None else None
        self.engine.stats.ks_relin += 1
        return Cipher(
            self.engine,
            out_r,
            out_i,
            chain_index=self._chain_index,
            scale=self._scale,
        )

    def rescale_to_next(self) -> "Cipher":
        if self._chain_index >= self.engine._max_chain_index:
            return self
        out_r = phantom.rescale_to_next(self.engine.context, self._ct_r) if self._ct_r is not None else None
        out_i = phantom.rescale_to_next(self.engine.context, self._ct_i) if self._ct_i is not None else None
        out_chain = self._chain_index + 1
        out_scale = float(self._scale) / float(self.engine._scale)
        self._set_pair_scale(out_r, out_i, out_scale)
        return Cipher(
            self.engine,
            out_r,
            out_i,
            chain_index=out_chain,
            scale=out_scale,
        )

    def decorypt(self, *, copy: bool = True, readonly: bool = True) -> np.ndarray:
        self.engine._check_decorypt_allowed()
        r = self.engine._decode_real_ct(self._ct_r)
        if self._ct_i is None:
            out = r.astype(np.complex128, copy=False)
        else:
            i = self.engine._decode_real_ct(self._ct_i)
            out = r + 1j * i
        if copy:
            out = np.array(out, dtype=np.complex128, copy=True)
        if readonly:
            out.setflags(write=False)
        return out

    def __repr__(self) -> str:
        return (
            f"PhantomCipher(nslots={self.nslots}, chain_index={self._chain_index}, "
            f"scale={self._scale:.3e}, imag={self._ct_i is not None})"
        )
