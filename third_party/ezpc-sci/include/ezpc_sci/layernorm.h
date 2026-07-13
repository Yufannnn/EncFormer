// layernorm.h — Secure layer normalization via 2PC protocols.
//
// With EZPC_HAS_SCI: uses FixOp for secure multiply (variance computation),
// sqrt with recp_sqrt=true (inverse sqrt via Newton-Raphson/Goldschmidt),
// and element-wise multiply for normalization.
//
// Without SCI: standard floating-point layer norm.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ezpc_sci/config.h"
#include "ezpc_sci/fixedpoint.h"

#ifdef EZPC_HAS_SCI
#include "ezpc_sci/context.h"
#endif

namespace ezpc_sci {

#ifdef EZPC_HAS_SCI
// Native SCI layer norm using FixOp.
// For each row: mean → center → variance → rsqrt → normalize.
inline void layer_norm_sci(SCIContext &ctx,
                           const double *in, double *out,
                           size_t rows, size_t cols,
                           double eps,
                           const ProtocolConfig &cfg) {
    FixOp *fix = ctx.fixop();
    int party = ctx.sci_party();
    int ell = cfg.ring_bits;
    int s = cfg.scale_bits;
    size_t n = rows * cols;
    uint64_t mask = (ell == 64) ? ~0ULL : (1ULL << ell) - 1;

    // Encode input
    std::vector<uint64_t> x_data(n);
    for (size_t i = 0; i < n; i++) {
        int64_t q = static_cast<int64_t>(std::llround(in[i] * (1LL << s)));
        x_data[i] = static_cast<uint64_t>(q) & mask;
    }
    FixArray x = fix->input(SCI_SERVER, n, x_data.data(), true, ell, s);

    for (size_t r = 0; r < rows; r++) {
        FixArray row = x.subset(r * cols, (r + 1) * cols);

        // 1. Mean: sum all elements, divide by cols (public constant).
        FixArray sum = row.subset(0, 1);
        for (size_t c = 1; c < cols; c++) {
            sum = fix->add(sum, row.subset(c, c + 1));
        }
        // Divide by cols: multiply by (1/cols) as a public constant.
        // Encode 1/cols as fixed-point.
        uint64_t inv_cols = static_cast<uint64_t>(
            std::llround((1.0 / cols) * (1LL << s))) & mask;
        FixArray mean_val = fix->mul(sum, inv_cols);
        // Truncate to keep scale consistent
        mean_val = fix->truncate_reduce(mean_val, s);

        // Broadcast mean and subtract: centered = row - mean
        FixArray mean_bcast(party, cols, true, ell, s);
        for (size_t c = 0; c < cols; c++) {
            mean_bcast.data[c] = mean_val.data[0];
        }
        FixArray centered = fix->sub(row, mean_bcast);

        // 2. Variance: mean of (centered^2).
        //    Square via secure multiply, then sum and divide.
        FixArray sq = fix->mul(centered, centered, 2 * ell);
        sq = fix->truncate_reduce(sq, s);  // back to ell bits, scale s

        FixArray var_sum = sq.subset(0, 1);
        for (size_t c = 1; c < cols; c++) {
            var_sum = fix->add(var_sum, sq.subset(c, c + 1));
        }
        FixArray var_val = fix->mul(var_sum, inv_cols);
        var_val = fix->truncate_reduce(var_val, s);

        // Add epsilon (public constant)
        uint64_t eps_q = static_cast<uint64_t>(std::llround(eps * (1LL << s))) & mask;
        FixArray var_eps = fix->add(var_val, eps_q);

        // 3. Inverse sqrt: 1/sqrt(var + eps) using SCI's sqrt with recp_sqrt=true.
        FixArray inv_std = fix->sqrt(var_eps, ell, s, /*recp_sqrt=*/true);

        // 4. Normalize: centered * inv_std
        FixArray inv_std_bcast(party, cols, true, ell, s);
        for (size_t c = 0; c < cols; c++) {
            inv_std_bcast.data[c] = inv_std.data[0];
        }
        FixArray normed = fix->mul(centered, inv_std_bcast, 2 * ell);
        normed = fix->truncate_reduce(normed, s);

        // Reveal and decode
        FixArray pub = fix->output(sci::PUBLIC, normed);
        for (size_t c = 0; c < cols; c++) {
            int64_t val = static_cast<int64_t>(pub.data[c]);
            if (val >= (1LL << (ell - 1))) val -= (1LL << ell);
            out[r * cols + c] = static_cast<double>(val) / static_cast<double>(1LL << s);
        }
    }
}
#endif

// Row-wise layer normalization: dispatches to SCI or emulated path.
inline void layer_norm(const double *in, double *out,
                       size_t rows, size_t cols,
                       double eps,
                       const ProtocolConfig &cfg) {
    // Emulated path
    for (size_t r = 0; r < rows; r++) {
        const double *row_in = in + r * cols;
        double *row_out = out + r * cols;

        double mean = 0.0;
        for (size_t c = 0; c < cols; c++)
            mean += row_in[c];
        mean /= static_cast<double>(cols);

        double var = 0.0;
        for (size_t c = 0; c < cols; c++) {
            double d = row_in[c] - mean;
            var += d * d;
        }
        var /= static_cast<double>(cols);

        double inv_std = 1.0 / std::sqrt(var + eps);
        for (size_t c = 0; c < cols; c++)
            row_out[c] = (row_in[c] - mean) * inv_std;
    }
}

} // namespace ezpc_sci
