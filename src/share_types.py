from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ShareMatrix:
    share0: np.ndarray
    share1: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.share0.shape

    def reconstruct(self) -> np.ndarray:

        return self.share0 + self.share1

    def reshare(self, rng: Optional[np.random.Generator] = None) -> "ShareMatrix":

        if rng is None:
            rng = np.random.default_rng()
        r = rng.standard_normal(self.share0.shape).astype(self.share0.dtype)
        full = self.share0 + self.share1
        return ShareMatrix(full - r, r)
