// bpmax.h — Paper BPMax attention weights in 2PC.
//   out[r,j] = clamp(x[r,j] + c, 0)^p * inv_rd[r]
// where inv_rd[r] = 1 / (R_d[r] + eps) is a PUBLIC frozen running denominator
// (per query row). No exp, no shared division — only clamp (GT + if_else),
// integer power (mul + truncate_reduce), and a public scalar multiply.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "ezpc_sci/config.h"

#ifdef EZPC_HAS_SCI
#include "ezpc_sci/context.h"
#endif

namespace ezpc_sci {

#ifdef EZPC_HAS_SCI
inline void bpmax_rows_sci(SCIContext &ctx, const double *in, const double *inv_rd,
                           double *out, size_t rows, size_t cols,
                           double c, int p, const ProtocolConfig &cfg) {
    FixOp *fix = ctx.fixop();
    int ell = cfg.ring_bits;
    int s = cfg.scale_bits;
    size_t n = rows * cols;
    uint64_t maskv = (ell == 64) ? ~0ULL : ((1ULL << ell) - 1);
    auto enc = [&](double v) -> uint64_t {
        int64_t q = static_cast<int64_t>(std::llround(v * static_cast<double>(1LL << s)));
        return static_cast<uint64_t>(q) & maskv;
    };
    // scale-s product then truncate back to scale s, staying inside ell bits.
    auto mul_tr = [&](const FixArray &a, const FixArray &b) -> FixArray {
        FixArray prod = fix->mul(a, b, ell + s);        // scale 2s, ell+s bits
        FixArray t = fix->truncate_reduce(prod, s);     // scale s, ell bits
        return fix->reduce(t, ell);                     // ensure ell bits for next op
    };

    // Secret input x (server owns; client passes zeros).
    std::vector<uint64_t> xd(n);
    for (size_t i = 0; i < n; i++) xd[i] = enc(in[i]);
    FixArray X = fix->input(SCI_SERVER, static_cast<int>(n), xd.data(), true, ell, s);

    // z = clamp(x + c, 0)
    FixArray Z = fix->add(X, enc(c));
    BoolArray gt = fix->GT(Z, static_cast<uint64_t>(0));
    FixArray z = fix->if_else(gt, Z, static_cast<uint64_t>(0));

    // pw = z^p by repeated scale-preserving multiply
    FixArray pw = z;
    for (int k = 1; k < p; k++) pw = mul_tr(pw, z);

    // Reveal pw = clamp(x+c,0)^p to PUBLIC. The public per-row 1/R_d multiply
    // is applied by the caller (no leakage: R_d is public and the BPMax weights
    // are revealed to PUBLIC anyway). inv_rd is applied here on the cleartext.
    FixArray pub = fix->output(sci::PUBLIC, pw);
    for (size_t r = 0; r < rows; r++) {
        for (size_t j = 0; j < cols; j++) {
            int64_t v = static_cast<int64_t>(pub.data[r * cols + j]);
            if (v >= (1LL << (ell - 1))) v -= (1LL << ell);
            out[r * cols + j] = (static_cast<double>(v) / static_cast<double>(1LL << s)) * inv_rd[r];
        }
    }
}
#endif

// Emulated fallback (identical arithmetic in double precision).
inline void bpmax_rows(const double *in, const double *inv_rd, double *out,
                       size_t rows, size_t cols, double c, int p, const ProtocolConfig &cfg) {
    (void)cfg;
    for (size_t r = 0; r < rows; r++) {
        for (size_t j = 0; j < cols; j++) {
            double z = in[r * cols + j] + c;
            if (z < 0) z = 0;
            double pw = 1.0;
            for (int k = 0; k < p; k++) pw *= z;
            out[r * cols + j] = pw * inv_rd[r];
        }
    }
}

}  // namespace ezpc_sci
