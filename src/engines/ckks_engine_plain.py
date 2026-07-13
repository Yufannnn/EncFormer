from __future__ import annotations

import numpy as np

from .ckks_engine_abc import CipherBase, CKKSEngine, CKKSStats, Number, Plain

__all__ = ["CKKSContext", "CKKSCiphertext", "Cipher", "CKKSStats", "Number", "Plain"]


class CKKSContext(CKKSEngine):
    def __init__(self, n_slots: int):
        super().__init__(nslots=int(n_slots))
        self.depth = 0

    def zeros(self) -> "CKKSCiphertext":
        v = np.zeros(self.nslots, dtype=np.complex128)
        v.setflags(write=False)
        return CKKSCiphertext(self, v, size=2, _trusted=True)

    def encorypt(self, arr: Plain) -> "CKKSCiphertext":
        if np.isscalar(arr):
            v = np.full(self.nslots, complex(arr), dtype=np.complex128)
        else:
            a = np.asarray(arr, dtype=np.complex128).ravel()
            if a.ndim != 1:
                raise ValueError("encorypt: need scalar/1d")
            if a.size > self.nslots:
                raise ValueError(f"encorypt: len {a.size}>{self.nslots}")
            v = np.zeros(self.nslots, dtype=np.complex128)
            v[: a.size] = a
        v.setflags(write=False)
        return CKKSCiphertext(self, v, size=2, _trusted=True)

    def relinearize(self, ct: "CKKSCiphertext") -> "CKKSCiphertext":
        if not isinstance(ct, CKKSCiphertext) or ct.engine is not self:
            raise TypeError("ct: wrong ctx")
        return ct.relinearize()

    def conjugate(self, ct: "CKKSCiphertext") -> "CKKSCiphertext":
        if not isinstance(ct, CKKSCiphertext) or ct.engine is not self:
            raise TypeError("ct: wrong ctx")

        s = self.stats
        s.ks_conj += max(0, ct.size - 1)
        s.ops_conj += 1

        out = np.conjugate(ct._expose_vec_readonly())
        out.setflags(write=False)
        return CKKSCiphertext(self, out, size=ct.size, _trusted=True)

    def increment_depth(self, new_depth: int) -> None:
        if new_depth > self.depth:
            self.depth = new_depth


class CKKSCiphertext(CipherBase):
    __slots__ = ("_ctx", "_v", "_size", "depth")

    def __init__(
        self,
        engine: CKKSContext,
        vec: np.ndarray,
        size: int = 2,
        _trusted: bool = False,
    ):
        if not isinstance(engine, CKKSContext):
            raise TypeError("ctx: bad")
        if not isinstance(vec, np.ndarray) or vec.dtype != np.complex128:
            raise TypeError("vec: need np.complex128")
        if vec.ndim != 1:
            raise ValueError("vec: need 1d")
        if vec.size != engine.nslots:
            raise ValueError(f"vec: len {vec.size}!={engine.nslots}")
        if not isinstance(size, int) or size < 2:
            raise ValueError("size: <2")
        if vec.flags.writeable and not _trusted:
            vec = vec.copy()
            vec.setflags(write=False)

        self._engine = engine
        super().__init__(engine)

        self._v = vec
        self._size = int(size)
        self.depth = 0

    @property
    def size(self) -> int:
        return self._size

    def _expose_vec_readonly(self) -> np.ndarray:
        return self._v

    def rot(self, k: int) -> "CKKSCiphertext":
        if not isinstance(k, int):
            raise TypeError("k: not int")
        d = k % self.nslots
        if d == 0:
            return self

        s = self.engine.stats
        s.ks_rots += max(0, self._size - 1)
        s.ops_rotate += 1

        out = np.roll(self._v, -d)
        out.setflags(write=False)
        return CKKSCiphertext(self.engine, out, size=self._size, _trusted=True)

    def conjugate(self) -> "CKKSCiphertext":
        s = self.engine.stats
        s.ks_conj += max(0, self._size - 1)
        s.ops_conj += 1
        out = np.conjugate(self._v)
        out.setflags(write=False)
        return CKKSCiphertext(self.engine, out, size=self._size, _trusted=True)

    def relinearize(self) -> "CKKSCiphertext":
        if self._size <= 2:
            return self
        s = self.engine.stats
        s.ks_relin += self._size - 2
        return CKKSCiphertext(self.engine, self._v, size=2, _trusted=True)

    def add(self, other: "CKKSCiphertext") -> "CKKSCiphertext":
        self._check_same_engine(other)
        self.engine.stats.ops_add += 1
        out = self._v + other._v
        out.setflags(write=False)
        ct = CKKSCiphertext(
            self.engine,
            out,
            size=max(self._size, other._size),
            _trusted=True,
        )
        ct.depth = max(self.depth, other.depth)
        return ct

    def mul_pt(self, other: Plain) -> "CKKSCiphertext":
        s = self.engine.stats

        if np.isscalar(other):
            pt = np.full(self.nslots, complex(other), dtype=np.complex128)
        else:
            pt = np.asarray(other, dtype=np.complex128).ravel()
            if pt.ndim != 1:
                raise ValueError("pt: need scalar/1d")
            if pt.size != self.nslots:
                raise ValueError(f"pt: len {pt.size}!={self.nslots}")

        s.ops_mul_pt += 1
        out = self._v * pt
        out.setflags(write=False)

        new_depth = self.depth + 1
        self.engine.increment_depth(new_depth)

        ct = CKKSCiphertext(self.engine, out, size=self._size, _trusted=True)
        ct.depth = new_depth
        return ct

    def mul_ct(self, other: "CKKSCiphertext", relin: bool = True) -> "CKKSCiphertext":
        self._check_same_engine(other)
        s = self.engine.stats
        s.ks_muls_ctct += 1
        s.ops_mul_ct += 1

        out = self._v * other._v
        out.setflags(write=False)

        res_size = self._size + other._size - 1
        if relin:
            s.ks_relin += max(0, res_size - 2)
            res_size = 2

        new_depth = max(self.depth, other.depth) + 1
        self.engine.increment_depth(new_depth)

        ct = CKKSCiphertext(self.engine, out, size=res_size, _trusted=True)
        ct.depth = new_depth
        return ct

    def mul_scalar(self, s_val: Number) -> "CKKSCiphertext":
        self.engine.stats.ops_mul_pt += 1
        out = self._v * complex(s_val)
        out.setflags(write=False)

        new_depth = self.depth + 1
        self.engine.increment_depth(new_depth)

        ct = CKKSCiphertext(self.engine, out, size=self._size, _trusted=True)
        ct.depth = new_depth
        return ct

    def decorypt(self, *, copy: bool = True, readonly: bool = True) -> np.ndarray:
        self.engine._check_decorypt_allowed()
        arr = self._v.copy() if copy else self._v
        if readonly:
            arr.setflags(write=False)
        self.depth = 0
        return arr

    def _check_same_engine(self, other: "CKKSCiphertext") -> None:
        if not isinstance(other, CKKSCiphertext) or self.engine is not other.engine:
            raise TypeError("ctx: mismatch")

    def __repr__(self) -> str:
        return f"CT(n={self.nslots},sz={self._size},d={self.depth})"


Cipher = CKKSCiphertext
