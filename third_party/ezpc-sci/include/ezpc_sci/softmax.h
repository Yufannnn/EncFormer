// softmax.h — Secure softmax via 2PC protocols.
//
// With EZPC_HAS_SCI: uses FixOp for secure comparison (max-finding),
// exp (LUT-based), and div (Goldschmidt reciprocal).
//
// Without SCI: standard floating-point softmax.
#pragma once

#include <algorithm>
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
// Native SCI softmax using FixOp high-level API.
// For each row: max → subtract → exp → sum → divide.
inline void softmax_rows_sci(SCIContext &ctx,
                             const double *in, double *out,
                             size_t rows, size_t cols,
                             const ProtocolConfig &cfg) {
    FixOp *fix = ctx.fixop();
    int party = ctx.sci_party();
    int ell = cfg.ring_bits;
    int s = cfg.scale_bits;
    size_t n = rows * cols;

    // Encode input into fixed-point shares.
    // Server provides the actual data; client provides zeros.
    std::vector<uint64_t> x_data(n);
    uint64_t mask = (ell == 64) ? ~0ULL : (1ULL << ell) - 1;
    for (size_t i = 0; i < n; i++) {
        int64_t q = static_cast<int64_t>(std::llround(in[i] * (1LL << s)));
        x_data[i] = static_cast<uint64_t>(q) & mask;
    }
    FixArray x = fix->input(SCI_SERVER, n, x_data.data(), true, ell, s);

    // Process each row
    for (size_t r = 0; r < rows; r++) {
        FixArray row = x.subset(r * cols, (r + 1) * cols);

        // 1. Find row max via pairwise comparison (log2(cols) rounds).
        FixArray row_max = row.subset(0, 1);
        for (size_t c = 1; c < cols; c++) {
            FixArray elem = row.subset(c, c + 1);
            BoolArray gt = fix->GT(elem, row_max);
            row_max = fix->if_else(gt, elem, row_max);
        }

        // Broadcast max and subtract: shifted = row - max
        FixArray max_broadcast(party, cols, true, ell, s);
        for (size_t c = 0; c < cols; c++) {
            max_broadcast.data[c] = row_max.data[0];
        }
        FixArray shifted = fix->sub(row, max_broadcast);

        // 2. Exponentiate: exp(shifted) using SCI's LUT-based exp.
        //    SCI exp assumes negative input, which shifted should be (x - max <= 0).
        FixArray exp_vals = fix->exp(shifted, ell, s);

        // 3. Sum the exponentials.
        //    Accumulate pairwise to reduce depth.
        FixArray sum_val = exp_vals.subset(0, 1);
        for (size_t c = 1; c < cols; c++) {
            sum_val = fix->add(sum_val, exp_vals.subset(c, c + 1));
        }

        // 4. Divide each exp by sum: out = exp_vals / sum
        for (size_t c = 0; c < cols; c++) {
            FixArray num = exp_vals.subset(c, c + 1);
            FixArray den_broadcast(party, 1, true, ell, s);
            den_broadcast.data[0] = sum_val.data[0];
            FixArray result = fix->div(num, den_broadcast, ell, s);

            // Reveal and decode
            FixArray pub = fix->output(sci::PUBLIC, result);
            int64_t val = static_cast<int64_t>(pub.data[0]);
            if (val >= (1LL << (ell - 1))) val -= (1LL << ell);
            out[r * cols + c] = static_cast<double>(val) / static_cast<double>(1LL << s);
        }
    }
}
#endif

// Row-wise softmax: dispatches to SCI or emulated path.
inline void softmax_rows(const double *in, double *out,
                         size_t rows, size_t cols,
                         const ProtocolConfig &cfg) {
    // Emulated path: standard numerically-stable softmax.
    for (size_t r = 0; r < rows; r++) {
        const double *row_in = in + r * cols;
        double *row_out = out + r * cols;

        double mx = row_in[0];
        for (size_t c = 1; c < cols; c++)
            mx = std::max(mx, row_in[c]);

        double sum = 0.0;
        for (size_t c = 0; c < cols; c++) {
            row_out[c] = std::exp(row_in[c] - mx);
            sum += row_out[c];
        }
        double inv = 1.0 / (sum + 1e-12);
        for (size_t c = 0; c < cols; c++)
            row_out[c] *= inv;
    }
}

} // namespace ezpc_sci
