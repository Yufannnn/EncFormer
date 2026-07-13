// gelu.h — Secure GELU using BOLT Algorithm 4 (piecewise polynomial).
//
// Implements the same algorithm as mpc_gelu_secure.py::secure_gelu_algorithm4_split
// in C++ fixed-point arithmetic.  When compiled with EZPC_HAS_SCI=1, the comparison
// and mux operations delegate to SCI's OT-based protocols.  Without SCI, they use
// the emulated fixed-point ops from fixedpoint.h.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "ezpc_sci/config.h"
#include "ezpc_sci/fixedpoint.h"

namespace ezpc_sci {

// BOLT Appendix C coefficients (same as mpc_gelu_secure.py)
constexpr double COEFF_A =  0.020848611754127593;
constexpr double COEFF_B = -0.18352506127082727;
constexpr double COEFF_C =  0.5410550166368381;
constexpr double COEFF_D = -0.03798164612714154;
constexpr double COEFF_E =  0.001620808531841547;

struct GeluConstants {
    int64_t a, b, c, p, m, e;
    int64_t neg_thresh, pos_thresh;
    int64_t zero, one;
};

inline GeluConstants build_constants(const ProtocolConfig &cfg) {
    auto q = [&](double v) -> int64_t {
        return static_cast<int64_t>(std::llround(v * cfg.scale()));
    };
    return {
        q(COEFF_A),
        q(COEFF_B),
        q(COEFF_C),
        q(0.5 + COEFF_D),
        q(0.5 - COEFF_D),
        q(COEFF_E),
        q(-cfg.threshold),
        q(cfg.threshold),
        0, 1,
    };
}

// Algorithm 4 on a single element (fixed-point domain).
inline int64_t gelu_element(int64_t x_raw, int64_t x_clipped, const GeluConstants &gc,
                            const ProtocolConfig &cfg) {
    using namespace fp;
    // Step A: powers of clipped input
    int64_t x2 = mul(x_clipped, x_clipped, cfg);
    int64_t x3 = mul(x2, x_clipped, cfg);
    int64_t x4 = mul(x2, x2, cfg);

    // Step B: candidate polynomials
    int64_t ax4 = mul_const(x4, gc.a, cfg);
    int64_t bx3 = mul_const(x3, gc.b, cfg);
    int64_t cx2 = mul_const(x2, gc.c, cfg);
    int64_t px  = mul_const(x_clipped, gc.p, cfg);
    int64_t mx  = mul_const(x_clipped, gc.m, cfg);

    // f0 = ax4 - bx3 + cx2 + mx + e
    int64_t f0 = add(ax4, neg(bx3, cfg), cfg);
    f0 = add(f0, cx2, cfg);
    f0 = add(f0, mx, cfg);
    f0 = add(f0, gc.e, cfg);

    int64_t f1 = add(ax4, bx3, cfg);
    f1 = add(f1, cx2, cfg);
    f1 = add(f1, px, cfg);
    f1 = add(f1, gc.e, cfg);

    // Step C: interval selector bits (Millionaire's protocol in real SCI)
    int64_t b0 = cmp_lt(x_raw, gc.neg_thresh, cfg);   // 1{x < -t}
    int64_t b1 = cmp_lt(x_raw, gc.zero, cfg);          // 1{x < 0}
    int64_t b2 = cmp_lt(gc.pos_thresh, x_raw, cfg);    // 1{t < x}

    int64_t z0 = xor_bits(b0, b1);
    int64_t z1 = xor_bits(xor_bits(b1, b2), gc.one);
    int64_t z2 = b2;

    // Step D: mux + sum
    int64_t y0 = mux(f0, z0, cfg);
    int64_t y1 = mux(f1, z1, cfg);
    int64_t y2 = mux(x_raw, z2, cfg);

    return add(add(y0, y1, cfg), y2, cfg);
}

// Vectorised GELU: double* in/out, handles encode/decode.
inline void gelu_vec(const double *in, double *out, size_t n, const ProtocolConfig &cfg) {
    GeluConstants gc = build_constants(cfg);
    int64_t clip_lo = static_cast<int64_t>(std::llround(-cfg.threshold * cfg.scale()));
    int64_t clip_hi = static_cast<int64_t>(std::llround( cfg.threshold * cfg.scale()));

    for (size_t i = 0; i < n; i++) {
        int64_t x_q = fp::encode(in[i], cfg);
        int64_t x_clip = x_q;
        if (x_clip < clip_lo) x_clip = clip_lo;
        if (x_clip > clip_hi) x_clip = clip_hi;
        int64_t y_q = gelu_element(x_q, x_clip, gc, cfg);
        out[i] = fp::decode(y_q, cfg);
    }
}

} // namespace ezpc_sci
