from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.engines.ckks_engine_abc import CipherBase, CKKSEngine, CKKSStats, Number, Plain

try:
    from desilofhe import Engine as DFEngine
except ImportError:
    from liberate.fhe import ckks_engine as _ckks_engine_cls

    DFEngine = _ckks_engine_cls


@dataclass
class _MPState:
    engine_common: Any = None
    engine_local: Any = None
    pk_a: Any = None
    pk_b_local: Any = None
    public_key: Any = None
    sk_local: Any = None
    ready: bool = False


class CKKSContext(CKKSEngine):
    __slots__ = (
        "_engine",
        "_sk",
        "_pk",
        "_evk_key",
        "_rotk",
        "_conj_key",
        "_stats",
        "_nslots",
        "_decorypt_enabled",
        "_use_binary_rotate",
        "backend",
        "_pt_cache",
        "_mp_enabled",
        "_mp",
    )

    def __init__(
        self,
        *,
        mode: str = "cpu",
        thread_count: int = 0,
        use_multiparty: bool = False,
        use_binary_rotate: bool = False,
        log_coeff_count: Optional[int] = None,
        special_prime_count: Optional[int] = None,
    ) -> None:

        try:
            eng = DFEngine(
                int(log_coeff_count),
                int(special_prime_count),
                mode=mode,
                thread_count=thread_count,
                use_multiparty=bool(use_multiparty),
            )
        except TypeError:
            devices = [0] if mode == "gpu" else None
            eng = DFEngine(
                devices=devices,
                verbose=False,
                logN=int(log_coeff_count),
                num_special_primes=int(special_prime_count),
            )

        self._engine = eng
        self._nslots = (
            self._engine.num_slots if hasattr(self._engine, "num_slots") else (1 << (int(log_coeff_count) - 1))
        )
        self.backend = mode
        self._stats = CKKSStats()
        self._decorypt_enabled = 0
        self._sk = self._engine.create_secret_key()
        self._pk = self._engine.create_public_key(self._sk)
        self._evk_key = None
        self._rotk: Dict[str, Any] = {}
        self._conj_key = None
        self._use_binary_rotate = bool(use_binary_rotate)
        self._pt_cache: Dict[complex, Any] = {}
        self._mp_enabled = bool(use_multiparty)
        self._mp: Optional[_MPState] = None

    @property
    def nslots(self) -> int:
        return self._nslots

    @property
    def stats(self) -> CKKSStats:
        return self._stats

    @property
    def engine(self):
        return self._engine

    @property
    def _secret_key(self):
        return self._sk

    @property
    def _public_key(self):
        return self._pk

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

    def _ensure_evk(self) -> None:
        if self._evk_key is None:
            try:
                self._evk_key = self._engine.create_relinearization_key(self._sk)
            except TypeError:
                self._evk_key = self._engine.create_evk(self._sk)

    def _ensure_rotk(self, step: int = 0) -> None:
        if self._is_liberate:
            if step not in self._rotk:
                self._rotk[step] = self._engine.create_rotation_key(self._sk, step)
        elif "gen" not in self._rotk:
            self._rotk["gen"] = self._engine.create_rotation_key(self._sk)

    def _ensure_conj(self) -> None:
        if self._conj_key is None:
            self._conj_key = self._engine.create_conjugation_key(self._sk)

    def conjugate(self, ct: "Cipher") -> "Cipher":
        if not isinstance(ct, Cipher) or ct.engine is not self:
            raise TypeError("ct must be a ciphertext from this context")
        return ct.conjugate()

    def rescale(self, obj: Any) -> Any:

        if isinstance(obj, Cipher):
            return Cipher(self, self._engine.rescale(obj._ct))
        return self._engine.rescale(obj)

    @property
    def _is_liberate(self) -> bool:
        return hasattr(self._engine, "encorypt")

    def encode_pt(self, arr: Plain) -> Any:
        return self._engine.encode(arr)

    def encode_pt_torch(self, tensor: "torch.Tensor", *, level: Optional[int] = None) -> Any:
        if torch is None:
            raise RuntimeError("torch is required for encode_pytorch_tensor")
        if level is None:
            return self._engine.encode_pytorch_tensor(tensor)
        return self._engine.encode_pytorch_tensor(tensor, level=level)

    def _get_const_pt(self, c: complex) -> Any:
        if c in self._pt_cache:
            return self._pt_cache[c]
        v = np.full(self.nslots, complex(c), dtype=np.complex128)
        pt = self._engine.encode(v)
        if len(self._pt_cache) < 8:
            self._pt_cache[c] = pt
        return pt

    def zeros(self, *, level: Optional[int] = None) -> "Cipher":
        v = np.zeros(self.nslots, dtype=np.complex128)
        v.setflags(write=False)
        if self._is_liberate:
            ct = self._engine.encorypt(v, self._pk, level=level or 0)
        elif level is not None:
            ct = self._engine.encrypt(v, self._pk, level=level)
        else:
            ct = self._engine.encrypt(v, self._pk)
        return Cipher(self, ct)

    def _pad_to_slots(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.complex128).ravel()
        if a.size == self.nslots:
            return a
        if a.size > self.nslots:
            raise ValueError(f"Input length {a.size} exceeds nslots {self.nslots}")
        out = np.zeros(self.nslots, dtype=np.complex128)
        out[: a.size] = a
        return out

    def encorypt(self, arr: Plain, *, level: Optional[int] = None) -> "Cipher":
        if np.isscalar(arr):
            v = np.full(self.nslots, complex(arr), dtype=np.complex128)
        else:
            v = self._pad_to_slots(np.asarray(arr, dtype=np.complex128))
        v.setflags(write=False)
        if self._is_liberate:
            ct = self._engine.encorypt(v, self._pk, level=level or 0)
        elif level is not None:
            ct = self._engine.encrypt(v, self._pk, level=level)
        else:
            ct = self._engine.encrypt(v, self._pk)
        return Cipher(self, ct)

    def decorypt(self, c: "Cipher", *, copy: bool = True, readonly: bool = True) -> np.ndarray:
        self._check_decorypt_allowed()
        if self._is_liberate:
            arr = self._engine.decrode(c._ct, self._sk)
        else:
            arr = self._engine.decrypt(c._ct, self._sk)
        if torch is not None and isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        out = np.array(arr, dtype=np.complex128, copy=True if copy else False)
        if readonly:
            out.setflags(False)
        return out

    def _same_engine(self, other: "CKKSContext") -> bool:
        return self is other

    def mp_begin(self) -> Any:
        if self._mp is None:
            self._mp = _MPState()
        self._mp.engine_common = self._engine
        self._mp.engine_local = self._engine
        if self._mp.pk_a is None:
            self._mp.pk_a = self._mp.engine_common.create_public_key_a()
        self._mp.sk_local = self._mp.engine_local.create_secret_key()
        self._mp.pk_b_local = self._mp.engine_local.create_public_key_b(self._mp.sk_local, self._mp.pk_a)
        self._mp.ready = False
        return self._mp.pk_b_local

    def mp_finalize_public_key(self, pk_b_all: List[Any], *, set_default: bool = False) -> Any:
        if self._mp is None or self._mp.pk_a is None:
            raise RuntimeError("mp_begin() or mp_adopt_public_key_a() must be called before mp_finalize_public_key().")
        self._mp.public_key = self._mp.engine_common.create_multiparty_public_key(pk_b_all, self._mp.pk_a)
        self._mp.ready = True
        if set_default:
            self._pk = self._mp.public_key
        return self._mp.public_key

    def mp_use_as_default_encryption_key(self, enable: bool = True) -> None:
        if self._mp is None or not self._mp.ready or self._mp.public_key is None:
            raise RuntimeError("Multiparty public key not ready. Call mp_begin()/mp_finalize_public_key().")
        self._pk = self._mp.public_key if enable else self._engine.create_public_key(self._sk)

    def mp_is_ready(self) -> bool:
        return bool(self._mp and self._mp.ready and self._mp.public_key is not None)

    def mp_info(self) -> Dict[str, Any]:
        return {
            "enabled": self._mp_enabled or (self._mp is not None),
            "ready": self.mp_is_ready(),
            "has_pk_a": bool(self._mp and self._mp.pk_a is not None),
            "has_pk_b_local": bool(self._mp and self._mp.pk_b_local is not None),
            "has_sk_local": bool(self._mp and self._mp.sk_local is not None),
            "using_default_pk_is_mp": (
                self._pk is not None and self._mp is not None and self._mp.public_key is self._pk
            ),
        }

    def mp_adopt_public_key_a(self, pk_a: Any) -> None:
        if self._mp is None:
            self._mp = _MPState()
        self._mp.engine_common = self._engine
        self._mp.engine_local = self._engine
        self._mp.pk_a = pk_a

    def mp_public_key_a(self) -> Any:
        if self._mp is None or self._mp.pk_a is None:
            raise RuntimeError("No pk_a stored. Call mp_begin() or mp_adopt_public_key_a().")
        return self._mp.pk_a

    def mp_public_key(self) -> Any:
        if not self.mp_is_ready():
            raise RuntimeError("Multiparty public key not ready. Call mp_finalize_public_key().")
        return self._mp.public_key

    def mp_public_key_b_local(self) -> Any:
        if self._mp is None or self._mp.pk_b_local is None:
            raise RuntimeError("No local pk_b. Call mp_begin().")
        return self._mp.pk_b_local

    def mp_secret_key_local(self) -> Any:
        if self._mp is None or self._mp.sk_local is None:
            raise RuntimeError("No local secret key. Call mp_begin().")
        return self._mp.sk_local

    def mp_set_public_key(self, mp_public_key: Any, *, set_default: bool = False) -> None:
        if self._mp is None:
            self._mp = _MPState()
        self._mp.engine_common = self._engine
        self._mp.engine_local = self._engine
        self._mp.public_key = mp_public_key
        self._mp.ready = True
        if set_default:
            self._pk = mp_public_key

    def mp_encrypt(self, arr: Plain, *, level: Optional[int] = None) -> "Cipher":
        if not self.mp_is_ready():
            raise RuntimeError("Multiparty public key not ready. Call mp_finalize_public_key() or mp_set_public_key().")
        if np.isscalar(arr):
            v = np.full(self.nslots, complex(arr), dtype=np.complex128)
        else:
            v = self._pad_to_slots(np.asarray(arr, dtype=np.complex128))
        v.setflags(False)
        eng = self._engine
        try:
            ct = (
                eng.encrypt(v, self._mp.public_key, level=level)
                if level is not None
                else eng.encrypt(v, self._mp.public_key)
            )
        except TypeError:
            ct = eng.encrypt(v, self._mp.public_key)
        return Cipher(self, ct)

    def mp_individual_decrypt(self, c: Union["Cipher", Any]) -> Any:
        if self._mp is None or self._mp.sk_local is None:
            raise RuntimeError("mp_begin() must be called before mp_individual_decrypt().")
        raw_ct = c._ct if isinstance(c, Cipher) else c
        return self._mp.engine_local.individual_decrypt(raw_ct, self._mp.sk_local)

    def mp_multiparty_decrypt(self, c: Union["Cipher", Any], shares: List[Any]) -> np.ndarray:
        if self._mp is None or not self._mp.ready or self._mp.public_key is None:
            raise RuntimeError("Multiparty public key not ready. Call mp_begin()/mp_finalize_public_key().")
        raw_ct = c._ct if isinstance(c, Cipher) else c
        arr = self._mp.engine_common.multiparty_decrypt(raw_ct, shares)
        return np.array(arr, dtype=np.complex128)

    def _get_engine(self):
        return self._engine

    def _get_relin_key(self):
        self._ensure_evk()
        return self._evk_key


class Cipher(CipherBase):
    __slots__ = ("_engine", "_ct")

    def __init__(self, engine: CKKSContext, ct: Any):
        if not isinstance(engine, CKKSContext):
            raise TypeError("engine must be CKKSContext")
        self._engine = engine
        self._ct = ct
        self._engine.stats.cts_created += 1

    @property
    def engine(self) -> CKKSContext:
        return self._engine

    def clone(self) -> "Cipher":
        return self.engine.engine.clone()

    def add(self, other) -> "Cipher":
        eng = self.engine.engine
        if isinstance(other, Cipher):
            self.engine.stats.ops_add += 1
            if self.engine._is_liberate:
                new_ct = eng.cc_add(self._ct, other._ct)
            else:
                new_ct = eng.add(self._ct, other._ct)
            return Cipher(self.engine, new_ct)
        elif isinstance(other, (np.ndarray, list, tuple, int, float, complex, np.number)):
            arr = np.asarray(other)
            if np.iscomplexobj(arr):
                rhs = self.engine.encode_pt(arr.astype(np.complex128, copy=False))
            else:
                rhs = arr.astype(np.float64, copy=False)
        else:
            rhs = other
        self.engine.stats.ops_add += 1
        if self.engine._is_liberate:
            new_ct = eng.cm_add(self._ct, rhs)
        else:
            new_ct = eng.add(self._ct, rhs)
        return Cipher(self.engine, new_ct)

    def add_inplace(self, other) -> "Cipher":
        if not isinstance(other, Cipher):
            raise TypeError("add_inplace currently supports Cipher rhs only")
        if other.engine is not self.engine:
            raise TypeError("add_inplace requires ciphertexts from the same engine/context")
        self.engine.stats.ops_add += 1
        eng = self.engine.engine
        try:
            eng.add(self._ct, other._ct, self._ct)
        except Exception:
            self._ct = eng.add(self._ct, other._ct)
        return self

    def sub(self, other) -> "Cipher":
        if isinstance(other, Cipher):
            if other.engine._same_engine(self.engine) is False:
                raise TypeError("sub requires ciphertexts from the same engine/context")
            rhs = other._ct
        else:
            rhs = other
        self.engine.stats.ops_add += 1
        eng = self.engine.engine
        try:
            new_ct = eng.subtract(self._ct, rhs)
        except AttributeError:
            new_ct = eng.sub(self._ct, rhs)
        return Cipher(self.engine, new_ct)

    def _cm_mult(self, ct, msg):

        eng = self.engine.engine
        if self.engine._is_liberate:
            return eng.cm_mult(ct, msg)
        return eng.multiply(ct, msg)

    def _cc_mult(self, ct_a, ct_b):

        eng = self.engine.engine
        if self.engine._is_liberate:
            return eng.cc_mult(ct_a, ct_b)
        return eng.multiply(ct_a, ct_b)

    def mul_scalar(self, s: Number) -> "Cipher":
        self.engine.stats.ops_mul_pt += 1
        z = complex(s)
        eng = self.engine.engine
        if self.engine._is_liberate:
            vec = np.full(self.engine.nslots, z, dtype=np.complex128)
            new_ct = eng.cm_mult(self._ct, vec)
            return Cipher(self.engine, new_ct)
        if abs(z.imag) == 0.0:
            try:
                new_ct = eng.multiply(self._ct, float(z.real))
                return Cipher(self.engine, new_ct)
            except Exception:
                pass
        pt = self.engine._get_const_pt(z)
        new_ct = eng.multiply(self._ct, pt)
        return Cipher(self.engine, new_ct)

    def mul_pt(self, other: Plain) -> "Cipher":
        eng = self.engine.engine
        if not np.isscalar(other) and not isinstance(other, np.ndarray):
            self.engine.stats.ops_mul_pt += 1
            new_ct = self._cm_mult(self._ct, other)
            return Cipher(self.engine, new_ct)
        if np.isscalar(other):
            return self.mul_scalar(other)
        self.engine.stats.ops_mul_pt += 1
        arr = np.asarray(other)
        if np.iscomplexobj(arr):
            vec = arr.astype(np.complex128, copy=False).ravel()
        else:
            vec = arr.astype(np.float64, copy=False).ravel()
        if self.engine._is_liberate:
            new_ct = eng.cm_mult(self._ct, vec)
        else:
            try:
                new_ct = eng.multiply(self._ct, vec)
            except Exception:
                pt = eng.encode(vec)
                new_ct = eng.multiply(self._ct, pt)
        return Cipher(self.engine, new_ct)

    def mul_ct(self, other: "Cipher", relin: bool = True) -> "Cipher":
        if not isinstance(other, Cipher) or other.engine is not self.engine:
            raise TypeError("mul_ct requires ciphertexts from the same engine/context")
        st = self.engine.stats
        st.ops_mul_ct += 1
        st.ks_muls_ctct += 1
        new_ct = self._cc_mult(self._ct, other._ct)
        if relin:
            self.engine._ensure_evk()
            st.ks_relin += 1
            new_ct = self.engine.engine.relinearize(new_ct, self.engine._evk_key)
        return Cipher(self.engine, new_ct)

    def relinearize(self) -> "Cipher":
        self.engine._ensure_evk()
        self.engine.stats.ks_relin += 1
        new_ct = self.engine.engine.relinearize(self._ct, self.engine._evk_key)
        return Cipher(self.engine, new_ct)

    def rot(self, k: int) -> "Cipher":
        if not isinstance(k, (int, np.integer)):
            raise TypeError("rot expects an integer step")
        n = self.engine.nslots
        d = int(k) % n
        if d == 0:
            return self
        self.engine.stats.ops_rotate += 1
        self.engine.stats.ks_rots += 1
        eng = self.engine.engine
        if self.engine._is_liberate:
            self.engine._ensure_rotk(-d)
            rk = self.engine._rotk[-d]
            ct2 = eng.rotate_single(self._ct, rk)
        else:
            self.engine._ensure_rotk()
            rk = self.engine._rotk["gen"]
            ct2 = eng.rotate(self._ct, rk, -d)
        return Cipher(self.engine, ct2)

    def conjugate(self) -> "Cipher":
        self.engine._ensure_conj()
        self.engine.stats.ops_conj += 1
        self.engine.stats.ks_conj += 1
        new_ct = self.engine.engine.conjugate(self._ct, self.engine._conj_key)
        return Cipher(self.engine, new_ct)

    def decorypt(self, *, copy: bool = True, readonly: bool = True) -> np.ndarray:
        self.engine._check_decorypt_allowed()
        eng = self.engine.engine
        if self.engine._is_liberate:
            arr = eng.decrode(self._ct, self.engine._secret_key)
        else:
            arr = eng.decrypt(self._ct, self.engine._secret_key)
        if torch is not None and isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if copy:
            out = np.array(arr, dtype=np.complex128, copy=True)
        else:
            out = np.asarray(arr, dtype=np.complex128)
        if readonly:
            out.setflags(write=False)
        return out

    def to_numpy(self, *, readonly: bool = True) -> np.ndarray:
        return self.decorypt(copy=True, readonly=readonly)

    def nslots(self) -> int:
        return self.engine.nslots

    def __repr__(self) -> str:
        return f"Cipher(nslots={self.engine.nslots})"
