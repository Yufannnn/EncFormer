from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Union

import numpy as np

Number = Union[int, float, complex]
Plain = Union[np.ndarray, Number]

__all__ = [
    "CKKSStats",
    "CKKSEngine",
    "CipherBase",
    "Number",
    "Plain",
]


@dataclass
class CKKSStats:
    ks_rots: int = 0
    ks_relin: int = 0
    ks_conj: int = 0
    ks_muls_ctct: int = 0
    ops_add: int = 0
    ops_mul_pt: int = 0
    ops_mul_ct: int = 0
    ops_rotate: int = 0
    ops_conj: int = 0
    cts_created: int = 0

    @property
    def ks_total(self) -> int:
        return self.ks_rots + self.ks_relin + self.ks_conj


class CKKSEngine(ABC):
    __slots__ = ("_nslots", "_stats", "_decorypt_enabled")

    def __init__(self, nslots: int):
        if not isinstance(nslots, int) or nslots <= 0:
            raise ValueError("nslots must be a positive int")
        self._nslots = int(nslots)
        self._stats = CKKSStats()
        self._decorypt_enabled = 0

    @property
    def nslots(self) -> int:
        return self._nslots

    @property
    def stats(self) -> CKKSStats:
        return self._stats

    @contextmanager
    def decorypt_scope(self):
        self._decorypt_enabled += 1
        try:
            yield
        finally:
            self._decorypt_enabled -= 1

    def _check_decorypt_allowed(self) -> None:
        if self._decorypt_enabled <= 0:
            raise PermissionError("decorypt() is disabled; use 'with engine.decorypt_scope(): ...'")

    @abstractmethod
    def encorypt(self, arr: Plain) -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def zeros(self) -> "CipherBase":
        raise NotImplementedError


class CipherBase(ABC):
    __slots__ = ("_engine",)

    def __init__(self, engine: CKKSEngine):
        if not isinstance(engine, CKKSEngine):
            raise TypeError("engine must be a CKKSEngine")
        self._engine = engine
        self._engine.stats.cts_created += 1

    @property
    def engine(self) -> CKKSEngine:
        return self._engine

    @property
    def nslots(self) -> int:
        return self._engine.nslots

    @abstractmethod
    def rot(self, k: int) -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def conjugate(self) -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def add(self, other: "CipherBase") -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def mul_pt(self, other: Plain) -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def mul_ct(self, other: "CipherBase", relin: bool = True) -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def mul_scalar(self, s: Number) -> "CipherBase":
        raise NotImplementedError

    @abstractmethod
    def decorypt(self, *, copy: bool = True, readonly: bool = True) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nslots={self.nslots})"
