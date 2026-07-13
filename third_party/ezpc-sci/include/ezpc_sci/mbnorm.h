// mbnorm.h — Paper MBNorm (BatchLayerNorm) in 2PC.
//   y[r,j] = ((x[r,j] - mean_r) / R_d[r]) * gamma[j] + beta[j]
// mean_r = sum_j x[r,j] / cols. R_d[r], gamma, beta are PUBLIC. The op is
// linear in x with public coefficients, so only a row-sum + reveal are needed;
// the ÷cols ÷R_d ×gamma +beta affine is applied on the revealed cleartext.
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
inline void mbnorm_rows_sci(SCIContext &ctx, const double *in, const double *inv_rd,
                            const double *gamma, const double *beta, double *out,
                            size_t rows, size_t cols, const ProtocolConfig &cfg) {
    FixOp *fix = ctx.fixop();
    int ell = cfg.ring_bits;
    int s = cfg.scale_bits;
    int party = ctx.sci_party();
    size_t n = rows * cols;
    uint64_t maskv = (ell == 64) ? ~0ULL : ((1ULL << ell) - 1);
    auto enc = [&](double v) -> uint64_t {
        int64_t q = static_cast<int64_t>(std::llround(v * static_cast<double>(1LL << s)));
        return static_cast<uint64_t>(q) & maskv;
    };
    std::vector<uint64_t> xd(n);
    for (size_t i = 0; i < n; i++) xd[i] = enc(in[i]);
    FixArray X = fix->input(SCI_SERVER, static_cast<int>(n), xd.data(), true, ell, s);

    for (size_t r = 0; r < rows; r++) {
        FixArray row = X.subset(r * cols, (r + 1) * cols);
        // row_sum (local additions)
        FixArray rowsum = row.subset(0, 1);
        for (size_t j = 1; j < cols; j++) rowsum = fix->add(rowsum, row.subset(j, j + 1));
        // u = cols * x - rowsum   (integer scalar mul, then broadcast-subtract)
        FixArray colsX = fix->mul(row, static_cast<uint64_t>(cols), ell);
        FixArray sumB(party, cols, true, ell, s);
        for (size_t j = 0; j < cols; j++) sumB.data[j] = rowsum.data[0];
        FixArray u = fix->sub(colsX, sumB);
        FixArray pub = fix->output(sci::PUBLIC, u);
        double invc = 1.0 / static_cast<double>(cols);
        for (size_t j = 0; j < cols; j++) {
            int64_t v = static_cast<int64_t>(pub.data[j]);
            if (v >= (1LL << (ell - 1))) v -= (1LL << ell);
            double centered = (static_cast<double>(v) / static_cast<double>(1LL << s)) * invc;  // x - mean
            out[r * cols + j] = centered * inv_rd[r] * gamma[j] + beta[j];
        }
    }
}
#endif

inline void mbnorm_rows(const double *in, const double *inv_rd, const double *gamma,
                        const double *beta, double *out, size_t rows, size_t cols,
                        const ProtocolConfig &cfg) {
    (void)cfg;
    for (size_t r = 0; r < rows; r++) {
        double mean = 0.0;
        for (size_t j = 0; j < cols; j++) mean += in[r * cols + j];
        mean /= static_cast<double>(cols);
        for (size_t j = 0; j < cols; j++)
            out[r * cols + j] = (in[r * cols + j] - mean) * inv_rd[r] * gamma[j] + beta[j];
    }
}

}  // namespace ezpc_sci
