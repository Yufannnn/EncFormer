import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fhe.phantom import two_party_native as tp  # noqa: E402

G = tp.Geom()
RNG = np.random.default_rng(3)


def _u64(shape):
    return RNG.integers(0, 2**64, size=shape, dtype=np.uint64)


def test_fixed_point_roundtrip():
    x = RNG.normal(0, 10, (64, 32))
    r = _u64(x.shape)
    assert np.allclose(tp.from_fixed(r + (tp.to_fixed(x) - r)), x, atol=2.0 ** -13)


def test_pack_pairs_then_unpack_is_identity():
    x = _u64((G.m, G.d))
    p = tp.pack_pairs(x, G, G.c)
    assert p.shape == (G.d // (2 * G.c), 2, G.nslots)
    assert np.array_equal(tp.unpack_blocks(tp.split_paired(p), G, G.d, G.c), x)


def test_pack_pfd_layout():
    a = _u64((G.h, G.m, G.m))
    p = tp.pack_pfd(a, G)
    half, m = G.m // 2, G.m
    for b, hl, t, r in ((0, 0, 0, 0), (2, 1, 5, 77), (5, 1, 63, 127)):
        h = b * (G.c // G.d_h) + hl
        seg = (hl * half + t) * m
        assert p[b, 0, seg + r] == a[h, r, (r - t) % m]
        assert p[b, 1, seg + r] == a[h, r, (r - t - half) % m]
