// fixedpoint.h — Fixed-point ring arithmetic for 2PC protocols.
//
// Mirrors the conventions in mpc_gelu_secure.py (PlainFixedPointOps) and
// EzPC SCI's FixedPoint library so that emulated and native paths produce
// identical numerical results.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "ezpc_sci/config.h"

namespace ezpc_sci {
namespace fp {

inline int64_t wrap(int64_t x, int64_t mod, int64_t half) {
    int64_t r = x % mod;
    if (r < 0) r += mod;
    return (r >= half) ? r - mod : r;
}

inline int64_t encode(double x, const ProtocolConfig &cfg) {
    return wrap(static_cast<int64_t>(std::llround(x * cfg.scale())),
                cfg.ring_mod(), cfg.ring_half());
}

inline double decode(int64_t x, const ProtocolConfig &cfg) {
    return static_cast<double>(wrap(x, cfg.ring_mod(), cfg.ring_half()))
           / static_cast<double>(cfg.scale());
}

// Truncation after multiplication (probabilistic truncation in real SCI;
// deterministic rounding here for reproducibility).
inline int64_t trunc(int64_t x, const ProtocolConfig &cfg) {
    int64_t s = cfg.scale();
    int64_t mod = cfg.ring_mod();
    int64_t half = cfg.ring_half();
    int64_t out;
    if (x >= 0)
        out = (x + s / 2) / s;
    else
        out = -((-x + s / 2) / s);
    return wrap(out, mod, half);
}

inline int64_t mul(int64_t a, int64_t b, const ProtocolConfig &cfg) {
    int64_t mod = cfg.ring_mod();
    int64_t half = cfg.ring_half();
    return trunc(wrap(a, mod, half) * wrap(b, mod, half), cfg);
}

inline int64_t mul_const(int64_t a, int64_t c, const ProtocolConfig &cfg) {
    return trunc(wrap(a, cfg.ring_mod(), cfg.ring_half()) * c, cfg);
}

inline int64_t add(int64_t a, int64_t b, const ProtocolConfig &cfg) {
    return wrap(wrap(a, cfg.ring_mod(), cfg.ring_half()) +
                wrap(b, cfg.ring_mod(), cfg.ring_half()),
                cfg.ring_mod(), cfg.ring_half());
}

inline int64_t sub(int64_t a, int64_t b, const ProtocolConfig &cfg) {
    return wrap(wrap(a, cfg.ring_mod(), cfg.ring_half()) -
                wrap(b, cfg.ring_mod(), cfg.ring_half()),
                cfg.ring_mod(), cfg.ring_half());
}

inline int64_t neg(int64_t a, const ProtocolConfig &cfg) {
    return wrap(-wrap(a, cfg.ring_mod(), cfg.ring_half()),
                cfg.ring_mod(), cfg.ring_half());
}

// Comparison: returns 1 if a < b (Millionaire's protocol in real SCI).
inline int64_t cmp_lt(int64_t a, int64_t b, const ProtocolConfig &cfg) {
    return (wrap(a, cfg.ring_mod(), cfg.ring_half()) <
            wrap(b, cfg.ring_mod(), cfg.ring_half())) ? 1 : 0;
}

// Multiplexer (OT-based in real SCI): val * bit
inline int64_t mux(int64_t val, int64_t bit, const ProtocolConfig &cfg) {
    return wrap(wrap(val, cfg.ring_mod(), cfg.ring_half()) * bit,
                cfg.ring_mod(), cfg.ring_half());
}

// XOR for single-bit values
inline int64_t xor_bits(int64_t a, int64_t b) {
    return a + b - 2 * a * b;
}

// Vectorised helpers operating on contiguous buffers.
inline void encode_vec(const double *in, int64_t *out, size_t n, const ProtocolConfig &cfg) {
    for (size_t i = 0; i < n; i++) out[i] = encode(in[i], cfg);
}

inline void decode_vec(const int64_t *in, double *out, size_t n, const ProtocolConfig &cfg) {
    for (size_t i = 0; i < n; i++) out[i] = decode(in[i], cfg);
}

} // namespace fp
} // namespace ezpc_sci
