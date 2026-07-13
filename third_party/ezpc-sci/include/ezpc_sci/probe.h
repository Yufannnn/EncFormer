// probe.h — minimal 2PC round-trip probe to validate SCI FixOp sharing.
// input(SERVER) -> output(PUBLIC): if this returns x, secret-sharing works and
// bpmax_2pc (clamp+power+public-divide) is viable.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ezpc_sci/config.h"

#ifdef EZPC_HAS_SCI
#include "ezpc_sci/context.h"
#endif

namespace ezpc_sci {

#ifdef EZPC_HAS_SCI
inline void probe_roundtrip_sci(SCIContext &ctx, const double *in, double *out,
                                size_t n, const ProtocolConfig &cfg) {
    FixOp *fix = ctx.fixop();
    int ell = cfg.ring_bits;
    int s = cfg.scale_bits;
    std::vector<uint64_t> xd(n);
    uint64_t maskv = (ell == 64) ? ~0ULL : ((1ULL << ell) - 1);
    for (size_t i = 0; i < n; i++) {
        int64_t q = static_cast<int64_t>(std::llround(in[i] * static_cast<double>(1LL << s)));
        xd[i] = static_cast<uint64_t>(q) & maskv;
    }
    FixArray X = fix->input(SCI_SERVER, static_cast<int>(n), xd.data(), true, ell, s);
    FixArray pub = fix->output(sci::PUBLIC, X);
    for (size_t i = 0; i < n; i++) {
        int64_t v = static_cast<int64_t>(pub.data[i]);
        if (v >= (1LL << (ell - 1))) v -= (1LL << ell);
        out[i] = static_cast<double>(v) / static_cast<double>(1LL << s);
    }
}
#endif

inline void probe_roundtrip(const double *in, double *out, size_t n, const ProtocolConfig &cfg) {
    (void)cfg;
    std::memcpy(out, in, n * sizeof(double));
}

}  // namespace ezpc_sci
