#pragma once

#ifdef EZPC_HAS_SCI

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ezpc_sci/context.h"
#include "utils/prg.h"
#include <memory>

namespace ezpc_sci {
namespace shares {

constexpr int BOUNDARY_BITS = 64;

enum class TruncMode { LOCAL = 0, FAITHFUL = 1 };

inline uint64_t ring_mask(int ell) { return ell >= 64 ? ~0ULL : ((1ULL << ell) - 1); }

inline int64_t quantize(double v, int scale, int bound_log2 = 62) {
    double q = std::nearbyint(std::ldexp(v, scale));
    if (!(std::fabs(q) < std::ldexp(1.0, bound_log2)))
        throw std::overflow_error("quantize: constant does not fit its bound");
    return static_cast<int64_t>(q);
}

inline FixArray from_share(FixOp *fix, const uint64_t *d, size_t n, int ell, int s) {
    FixArray a(fix->party, static_cast<int>(n),  true, ell, s);
    const uint64_t m = ring_mask(ell);
    for (size_t i = 0; i < n; i++) a.data[i] = d[i] & m;
    return a;
}

inline void add_public(FixOp *fix, FixArray &a, uint64_t c) {
    if (fix->party != sci::ALICE) return;
    const uint64_t m = a.ell_mask();
    for (int i = 0; i < a.size; i++) a.data[i] = (a.data[i] + c) & m;
}

inline void ltrunc(bool alice, uint64_t *v, size_t m, int s) {
    if (s == 0) return;
    for (size_t i = 0; i < m; i++) v[i] = alice ? (v[i] >> s) : (0 - ((0 - v[i]) >> s));
}

inline void ftrunc_nonneg(FixOp *fix, uint64_t *v, size_t m, int s) {
    FixArray V(fix->party, static_cast<int>(m), true, BOUNDARY_BITS, 0);
    std::memcpy(V.data, v, m * sizeof(uint64_t));
    add_public(fix, V, 1ULL << (s - 1));
    std::vector<uint8_t> z(m, 0);
    FixArray Y = fix->right_shift(V, s, z.data());
    std::memcpy(v, Y.data, m * sizeof(uint64_t));
}

inline void ftrunc_bounded(FixOp *fix, uint64_t *v, size_t m, int s) {
    const bool alice = fix->party == sci::ALICE;
    if (alice) for (size_t i = 0; i < m; i++) v[i] += (1ULL << 62);
    ftrunc_nonneg(fix, v, m, s);
    if (alice) for (size_t i = 0; i < m; i++) v[i] -= (1ULL << (62 - s));
}

inline void trunc(FixOp *fix, TruncMode mode, uint64_t *v, size_t m, int s) {
    if (s == 0) return;
    if (mode == TruncMode::LOCAL) ltrunc(fix->party == sci::ALICE, v, m, s);
    else ftrunc_bounded(fix, v, m, s);
}

inline void random_u64(uint64_t *d, size_t m) {
    sci::PRG128 prg;
    prg.random_data(d, static_cast<int>(m * sizeof(uint64_t)));
}

inline void gen_square_triples(FixOp *fix, uint64_t *a, uint64_t *a2, size_t m) {
    random_u64(a, m);
    std::vector<uint64_t> zeros(m, 0), cross(m, 0);
    const bool alice = fix->party == sci::ALICE;
    fix->mult->hadamard_cross_terms(static_cast<int32_t>(m), alice ? a : zeros.data(),
                                    alice ? zeros.data() : a, cross.data(), 64, 64, 64,
                                    MultMode::Alice_has_A);
    for (size_t i = 0; i < m; i++) a2[i] = a[i] * a[i] + 2 * cross[i];
}

inline void gen_triples(FixOp *fix, uint64_t *a, uint64_t *b, uint64_t *c, size_t m) {
    random_u64(a, m);
    random_u64(b, m);
    std::vector<uint64_t> cross(m, 0);
    fix->mult->hadamard_cross_terms(static_cast<int32_t>(m), a, b, cross.data(), 64, 64, 64, MultMode::None);
    for (size_t i = 0; i < m; i++) c[i] = a[i] * b[i] + cross[i];
}

inline void exchange(FixOp *fix, const uint64_t *mine, uint64_t *theirs, size_t m) {
    const bool alice = fix->party == sci::ALICE;
    sci::NetIO *tx = alice ? fix->iopack->io : fix->iopack->io_rev;
    sci::NetIO *rx = alice ? fix->iopack->io_rev : fix->iopack->io;
    std::thread t([&] {
        tx->send_data(mine, static_cast<int>(m * sizeof(uint64_t)));
        tx->flush();
    });
    rx->recv_data(theirs, static_cast<int>(m * sizeof(uint64_t)));
    t.join();
}

inline void beaver_square(FixOp *fix, const uint64_t *z, const uint64_t *a, const uint64_t *a2,
                          uint64_t *out, size_t m) {
    const bool alice = fix->party == sci::ALICE;
    std::vector<uint64_t> e(m), et(m);
    for (size_t i = 0; i < m; i++) e[i] = z[i] - a[i];
    exchange(fix, e.data(), et.data(), m);
    for (size_t i = 0; i < m; i++) {
        uint64_t ev = e[i] + et[i];
        out[i] = a2[i] + 2 * ev * a[i] + (alice ? ev * ev : 0);
    }
}

inline void beaver_mul(FixOp *fix, const uint64_t *x, const uint64_t *y, const uint64_t *a, const uint64_t *b,
                       const uint64_t *c, uint64_t *out, size_t m) {
    const bool alice = fix->party == sci::ALICE;
    std::vector<uint64_t> ef(2 * m), eft(2 * m);
    for (size_t i = 0; i < m; i++) {
        ef[i] = x[i] - a[i];
        ef[m + i] = y[i] - b[i];
    }
    exchange(fix, ef.data(), eft.data(), 2 * m);
    for (size_t i = 0; i < m; i++) {
        uint64_t e = ef[i] + eft[i], f = ef[m + i] + eft[m + i];
        out[i] = c[i] + e * b[i] + f * a[i] + (alice ? e * f : 0);
    }
}

inline void relu64(FixOp *fix, const uint64_t *y, uint64_t *out, size_t m, int cmp_bits, int F) {
    FixArray Yk = from_share(fix, y, m, cmp_bits, F);
    BoolArray neg = fix->LT(Yk, static_cast<uint64_t>(0));
    BoolArray pos = fix->bool_op->NOT(neg);
    std::vector<uint64_t> yy(y, y + m);
    fix->aux->multiplexer(pos.data, yy.data(), out, static_cast<int32_t>(m), 64, 64);
}

struct StepProfiler {
    FixOp *fix;
    bool on;
    uint64_t last = 0, last_ext = 0;
    std::chrono::steady_clock::time_point t;
    static uint64_t ext(FixOp *f) {
#ifdef EZPC_SCI_HAS_EXT_COUNTER
        return f->iopack->io->ext_bytes + f->iopack->io_rev->ext_bytes + f->iopack->io_GC->ext_bytes;
#else
        (void)f;
        return 0;
#endif
    }
    StepProfiler(FixOp *f, size_t b) : fix(f), on(b == 0 && std::getenv("EZPC_SCI_PROFILE") != nullptr) {
        last = f->iopack->get_comm();
        last_ext = ext(f);
        t = std::chrono::steady_clock::now();
    }
    void mark(const char *what) {
        if (!on) return;
        uint64_t c = fix->iopack->get_comm(), x = ext(fix);
        auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[sci-profile party=%d] %-30s sent %12llu B (ot-ext %12llu B) %8.3f s\n", fix->party,
                     what, static_cast<unsigned long long>(c - last), static_cast<unsigned long long>(x - last_ext),
                     std::chrono::duration<double>(now - t).count());
        last = c;
        last_ext = x;
        t = now;
    }
};

template <class Body>
inline int run_chunks(SCIContext &ctx, size_t n, Body body) {
    int T = ctx.threads();
    if (n < static_cast<size_t>(T)) T = n > 0 ? static_cast<int>(n) : 1;
    if (n == 0) return 0;
    if (T == 1) {
        body(ctx.fixop(0), size_t(0), n);
        return 1;
    }
    std::vector<std::thread> ths;
    std::vector<std::exception_ptr> errs(T);
    for (int t = 0; t < T; t++) {
        size_t b = n * t / T, e = n * (t + 1) / T;
        ths.emplace_back([&, t, b, e] {
            try {
                body(ctx.fixop(t), b, e);
            } catch (...) {
                errs[t] = std::current_exception();
            }
        });
    }
    for (auto &th : ths) th.join();
    for (auto &er : errs)
        if (er) std::rethrow_exception(er);
    return T;
}

template <class Fn>
inline PhaseStats measure_phase(SCIContext &ctx, Fn fn, int *used = nullptr) {
    const int T = ctx.threads();
    std::vector<CommSnapshot> c0(T);
    for (int t = 0; t < T; t++) c0[t] = ctx.comm_thread(t);
#ifdef EZPC_SCI_HAS_OT_POOL
    std::vector<uint64_t> ot0 = ctx.ot_counts();
#endif
    auto t0 = std::chrono::steady_clock::now();
    int u = fn();
    auto t1 = std::chrono::steady_clock::now();
    if (used) *used = u;
    PhaseStats ps;
    ps.seconds = std::chrono::duration<double>(t1 - t0).count();
    for (int t = 0; t < T; t++) {
        CommSnapshot c1 = ctx.comm_thread(t);
        ps.bytes_sent += c1.bytes_sent - c0[t].bytes_sent;
        ps.bytes_recv += c1.bytes_recv - c0[t].bytes_recv;
        ps.rounds = std::max<uint64_t>(ps.rounds, c1.rounds - c0[t].rounds);
        ps.ext_bytes_sent += c1.ext_bytes_sent - c0[t].ext_bytes_sent;
        ps.ext_choice_bits += c1.ext_choice_bits - c0[t].ext_choice_bits;
        ps.ext_ots += c1.ext_ots - c0[t].ext_ots;
        ps.ext_seconds = std::max(ps.ext_seconds, c1.ext_seconds - c0[t].ext_seconds);
        ps.derand_bytes += c1.derand_bytes - c0[t].derand_bytes;
    }
#ifdef EZPC_SCI_HAS_OT_POOL
    std::vector<uint64_t> ot1 = ctx.ot_counts();
    ps.ot_counts.resize(ot1.size());
    for (size_t i = 0; i < ot1.size(); i++) ps.ot_counts[i] = ot1[i] - ot0[i];
#endif
    return ps;
}

template <class Off, class On>
inline CallStats two_phase(SCIContext &ctx, size_t n, Off offline, On online) {
    CallStats st;
    st.n = n;
    st.planned = ctx.planned;
    st.offline = measure_phase(ctx, offline);
    st.online = measure_phase(ctx, online, &st.threads);
    ctx.last_stats = st;
    return st;
}

inline void check_F(int F, const char *who) {
    if (F < 1 || F > 24) throw std::invalid_argument(std::string(who) + ": F must be in [1, 24]");
}

inline std::shared_ptr<Material> take_material(SCIContext &ctx, const std::string &sig) {
    if (ctx.plan_queue.empty())
        throw std::runtime_error("planned mode: offline material exhausted (call not in the plan): " + sig);
    auto m = ctx.plan_queue.front();
    if (m->sig != sig)
        throw std::runtime_error("planned mode: call does not match the plan: got '" + sig + "', planned '" +
                                 m->sig + "'");
    ctx.plan_queue.pop_front();
    return m;
}

template <class Gen>
inline std::shared_ptr<Material> obtain_material(SCIContext &ctx, const std::string &sig, Gen gen) {
    if (ctx.planned) return take_material(ctx, sig);
    auto m = std::make_shared<Material>();
    m->sig = sig;
    gen(*m);
    return m;
}

struct BPMaxOpts {
    double c = 5.0;
    int p = 5;
    int F = 13;
    int a = 3;
    int cmp_bits = 21;
    int scale_frac = 16;
    int scale_width = 21;
    bool prescaled = false;
    TruncMode scale_trunc = TruncMode::FAITHFUL;
};

inline std::string bpmax_sig(size_t n, const BPMaxOpts &o) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "bpmax n=%zu p=%d c=%.17g F=%d a=%d cmp=%d sf=%d sw=%d prescaled=%d tr=%d",
                  n, o.p, o.c, o.F, o.a, o.cmp_bits, o.scale_frac, o.scale_width, int(o.prescaled), int(o.scale_trunc));
    return buf;
}

inline std::vector<int64_t> bpmax_scales(size_t rows, size_t cols, const double *rd, const BPMaxOpts &o) {
    std::vector<int64_t> sq;
    if (o.prescaled) return sq;
    if (rd == nullptr) throw std::invalid_argument("bpmax_rows_shares: R_d required");
    sq.resize(rows * cols);
    for (size_t r = 0; r < rows; r++) {
        double v = rd[r];
        if (!std::isfinite(v) || v <= 0.0) throw std::invalid_argument("bpmax_rows_shares: R_d must be > 0");
        int64_t q = quantize(std::pow(v, -1.0 / o.p) * std::ldexp(1.0, o.a), o.scale_frac, o.scale_width);
        for (size_t j = 0; j < cols; j++) sq[r * cols + j] = q;
    }
    return sq;
}

inline void gen_bpmax_material(SCIContext &ctx, size_t n, const BPMaxOpts &o, Material &M) {
    const int top = 31 - __builtin_clz(static_cast<unsigned>(o.p));
    const int nsq = top, nmul = __builtin_popcount(static_cast<unsigned>(o.p)) - 1;
    M.A.resize(nsq * n); M.A2.resize(nsq * n);
    M.TA.resize(nmul * n); M.TB.resize(nmul * n); M.TC.resize(nmul * n);
    run_chunks(ctx, n, [&](FixOp *fix, size_t b, size_t e) {
        const size_t m = e - b;
        for (int k = 0; k < nsq; k++) gen_square_triples(fix, &M.A[k * n + b], &M.A2[k * n + b], m);
        for (int k = 0; k < nmul; k++) gen_triples(fix, &M.TA[k * n + b], &M.TB[k * n + b], &M.TC[k * n + b], m);
    });
}

inline CallStats bpmax_rows_shares(SCIContext &ctx, const uint64_t *x, const double *rd, uint64_t *out,
                                   size_t rows, size_t cols, const BPMaxOpts &o) {
    check_F(o.F, "bpmax_rows_shares");
    if (o.p < 1 || o.p > 16) throw std::invalid_argument("bpmax_rows_shares: p must be in [1, 16]");
    if (o.a < 0 || o.a * o.p > 40) throw std::invalid_argument("bpmax_rows_shares: a out of range");
    if (o.cmp_bits < o.F + 3 || o.cmp_bits > 64) throw std::invalid_argument("bpmax_rows_shares: cmp_bits out of range");
    const bool alice = ctx.sci_party() == SCI_SERVER;
    const size_t n = rows * cols;
    const int F = o.F, p = o.p, pa = o.p * o.a, Ks = o.scale_frac;

    const std::vector<int64_t> sq = bpmax_scales(rows, cols, rd, o);
    const uint64_t cq = static_cast<uint64_t>(quantize(o.c, F));
    const int top = 31 - __builtin_clz(static_cast<unsigned>(p));
    const std::string sig = bpmax_sig(n, o);
    std::shared_ptr<Material> mat;

    auto offline = [&] {
        mat = obtain_material(ctx, sig, [&](Material &m) { gen_bpmax_material(ctx, n, o, m); });
        return 0;
    };
    auto online = [&] {
        const Material &M = *mat;
        const auto &A = M.A, &A2 = M.A2, &TA = M.TA, &TB = M.TB, &TC = M.TC;
        return run_chunks(ctx, n, [&](FixOp *fix, size_t b, size_t e) {
            const size_t m = e - b;
            StepProfiler prof(fix, b);
            std::vector<uint64_t> y(x + b, x + e);
            if (!o.prescaled) {
                if (alice) for (size_t i = 0; i < m; i++) y[i] += cq;
                for (size_t i = 0; i < m; i++) y[i] *= static_cast<uint64_t>(sq[b + i]);
                prof.mark("bpmax: scale (x+c)*s");
                trunc(fix, o.scale_trunc, y.data(), m, Ks);
                prof.mark("bpmax: scale truncation");
            }
            std::vector<uint64_t> z(m), cur(m), tmp(m);
            relu64(fix, y.data(), z.data(), m, o.cmp_bits, F);
            prof.mark("bpmax: relu (cmp + mux64)");
            if (p == 5) {
                beaver_square(fix, z.data(), &A[b], &A2[b], tmp.data(), m);
                prof.mark("bpmax: Pi_mult z^2 (1 open)");
                ftrunc_nonneg(fix, tmp.data(), m, F);
                prof.mark("bpmax: faithful trunc F");
                beaver_square(fix, tmp.data(), &A[n + b], &A2[n + b], cur.data(), m);
                prof.mark("bpmax: Pi_mult z^4 (1 open)");
                beaver_mul(fix, cur.data(), z.data(), &TA[b], &TB[b], &TC[b], tmp.data(), m);
                prof.mark("bpmax: Pi_mult z^5 (2 open)");
                ftrunc_nonneg(fix, tmp.data(), m, 2 * F + pa);
                prof.mark("bpmax: faithful trunc 2F+pa");
                std::memcpy(out + b, tmp.data(), m * sizeof(uint64_t));
            } else {
                cur = z;
                int ks = 0, km = 0;
                for (int bit = top - 1; bit >= 0; --bit) {
                    beaver_square(fix, cur.data(), &A[ks * n + b], &A2[ks * n + b], tmp.data(), m);
                    ++ks;
                    ftrunc_nonneg(fix, tmp.data(), m, F);
                    cur.swap(tmp);
                    if ((p >> bit) & 1) {
                        beaver_mul(fix, cur.data(), z.data(), &TA[km * n + b], &TB[km * n + b], &TC[km * n + b],
                                   tmp.data(), m);
                        ++km;
                        ftrunc_nonneg(fix, tmp.data(), m, F);
                        cur.swap(tmp);
                    }
                }
                if (pa > 0) ftrunc_nonneg(fix, cur.data(), m, pa);
                std::memcpy(out + b, cur.data(), m * sizeof(uint64_t));
            }
        });
    };
    return two_phase(ctx, n, offline, online);
}

struct LNOpts {
    int F = 13;
    int coef_bits = 16;
    int coef_width = 24;
    int mean_bits = 17;
    TruncMode trunc = TruncMode::LOCAL;
};

inline std::string ln_sig(size_t rows, size_t cols, const LNOpts &o) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "mbln rows=%zu cols=%zu F=%d cb=%d cw=%d mb=%d tr=%d", rows, cols, o.F,
                  o.coef_bits, o.coef_width, o.mean_bits, int(o.trunc));
    return buf;
}

inline void ln_coefs(size_t rows, size_t cols, const double *rd, size_t rd_len, const double *gamma,
                     const double *beta, double eps, double ln_l, const LNOpts &o, std::vector<int64_t> &g,
                     std::vector<uint64_t> &bq) {
    if (rd == nullptr || (rd_len != rows && rd_len != 1))
        throw std::invalid_argument("layernorm_shares: R_d required (length rows or 1)");
    g.resize(rows * cols);
    for (size_t r = 0; r < rows; r++) {
        double den = ln_l * rd[rd_len == 1 ? 0 : r] + eps;
        if (!std::isfinite(den) || den <= 0.0) throw std::invalid_argument("layernorm_shares: ln_l R_d + eps must be > 0");
        for (size_t j = 0; j < cols; j++)
            g[r * cols + j] = quantize((gamma ? gamma[j] : 1.0) / den, o.coef_bits, o.coef_width - 1);
    }
    bq.clear();
    if (beta) {
        bq.resize(cols);
        for (size_t j = 0; j < cols; j++) bq[j] = static_cast<uint64_t>(quantize(beta[j], o.F + o.coef_bits));
    }
}

inline CallStats layernorm_shares(SCIContext &ctx, const uint64_t *x, const double *rd, size_t rd_len,
                                  const double *gamma, const double *beta, uint64_t *out, size_t rows,
                                  size_t cols, double eps, double ln_l, const LNOpts &o) {
    check_F(o.F, "layernorm_shares");
    if (o.coef_bits < 1 || o.coef_width < 2 || o.coef_width > 48 || o.F + o.coef_bits > 40)
        throw std::invalid_argument("layernorm_shares: coef_bits / coef_width out of range");
    const bool alice = ctx.sci_party() == SCI_SERVER;
    const int F = o.F, K = o.coef_bits;
    const size_t n = rows * cols;
    std::vector<int64_t> g;
    std::vector<uint64_t> bq;
    ln_coefs(rows, cols, rd, rd_len, gamma, beta, eps, ln_l, o, g, bq);
    int t = 0;
    size_t odd = cols;
    while (odd > 1 && (odd & 1) == 0) { odd >>= 1; ++t; }
    int K0 = 0;
    uint64_t qo = 1;
    if (odd > 1) {
        int lg = 0;
        while ((size_t(1) << lg) < odd) ++lg;
        K0 = o.mean_bits + lg;
        qo = static_cast<uint64_t>(std::nearbyint(std::ldexp(1.0, K0) / static_cast<double>(odd)));
    }
    std::vector<uint64_t> c(n);
    const std::string sig = ln_sig(rows, cols, o);
    std::shared_ptr<Material> mat;

    auto offline = [&] {
        mat = obtain_material(ctx, sig, [](Material &) {});
        return 0;
    };
    auto online = [&] {
        for (size_t r = 0; r < rows; r++) {
            const uint64_t *xr = x + r * cols;
            uint64_t rs = 0;
            for (size_t j = 0; j < cols; j++) rs += xr[j];
            ltrunc(alice, &rs, 1, t);
            if (odd > 1) {
                rs *= qo;
                ltrunc(alice, &rs, 1, K0);
            }
            for (size_t j = 0; j < cols; j++) c[r * cols + j] = xr[j] - rs;
        }
        if (o.trunc == TruncMode::LOCAL) {
            for (size_t i = 0; i < n; i++)
                out[i] = c[i] * static_cast<uint64_t>(g[i]) + (alice && !bq.empty() ? bq[i % cols] : 0);
            ltrunc(alice, out, n, K);
            return 1;
        }
        return run_chunks(ctx, n, [&](FixOp *fix, size_t b, size_t e) {
            const size_t m = e - b;
            std::vector<uint64_t> v(c.begin() + b, c.begin() + e);
            for (size_t i = 0; i < m; i++) v[i] *= static_cast<uint64_t>(g[b + i]);
            if (alice && !bq.empty()) for (size_t i = 0; i < m; i++) v[i] += bq[(b + i) % cols];
            trunc(fix, o.trunc, v.data(), m, K);
            std::memcpy(out + b, v.data(), m * sizeof(uint64_t));
        });
    };
    return two_phase(ctx, n, offline, online);
}

struct GeluSelectOpts {
    int F = 13;
    int cmp_bits = 22;
    double threshold = 2.7;
};

inline std::string gelu_sig(size_t n, const GeluSelectOpts &o) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "gelu_select n=%zu F=%d cmp=%d t=%.17g", n, o.F, o.cmp_bits, o.threshold);
    return buf;
}

inline CallStats gelu_select_shares(SCIContext &ctx, const uint64_t *x, const uint64_t *f0, const uint64_t *f1,
                                    uint64_t *out, size_t n, const GeluSelectOpts &o) {
    check_F(o.F, "gelu_select_shares");
    if (o.cmp_bits < o.F + 4 || o.cmp_bits > 64) throw std::invalid_argument("gelu_select_shares: cmp_bits out of range");
    const int k = o.cmp_bits, F = o.F;
    const uint64_t mask = ring_mask(k);
    const uint64_t tq = static_cast<uint64_t>(quantize(o.threshold, F)) & mask;
    const std::string sig = gelu_sig(n, o);
    auto offline = [&] {
        obtain_material(ctx, sig, [](Material &) {});
        return 0;
    };
    auto online = [&] {
        return run_chunks(ctx, n, [&](FixOp *fix, size_t b, size_t e) {
            const size_t m = e - b;
            const int party = fix->party;
            const bool alice = (party == sci::ALICE);
            StepProfiler prof(fix, b);
            FixArray D(party, static_cast<int>(3 * m), true, k, F);
            const uint64_t ta = alice ? tq : 0;
            for (size_t i = 0; i < m; i++) {
                uint64_t xi = x[b + i] & mask;
                D.data[i] = (xi + ta) & mask;
                D.data[m + i] = xi;
                D.data[2 * m + i] = (ta - xi) & mask;
            }
            BoolArray B = fix->LT(D, static_cast<uint64_t>(0));
            prof.mark("gelu: 3n compare (MSB)");
            std::vector<uint8_t> S(3 * m);
            for (size_t i = 0; i < m; i++) {
                uint8_t b0 = B.data[i], b1 = B.data[m + i], b2 = B.data[2 * m + i];
                S[i] = b0 ^ b1;
                S[m + i] = b1 ^ b2 ^ static_cast<uint8_t>(alice);
                S[2 * m + i] = b2;
            }
            std::vector<uint64_t> V(3 * m), Y(3 * m);
            std::memcpy(V.data(), f0 + b, m * sizeof(uint64_t));
            std::memcpy(V.data() + m, f1 + b, m * sizeof(uint64_t));
            std::memcpy(V.data() + 2 * m, x + b, m * sizeof(uint64_t));
            fix->aux->multiplexer(S.data(), V.data(), Y.data(), static_cast<int32_t>(3 * m), 64, 64);
            prof.mark("gelu: 3n mux (64-bit)");
            for (size_t i = 0; i < m; i++) out[b + i] = Y[i] + Y[m + i] + Y[2 * m + i];
        });
    };
    return two_phase(ctx, n, offline, online);
}

inline void prepare_bpmax(SCIContext &ctx, size_t rows, size_t cols, const double *rd, const BPMaxOpts &o) {
    if (ctx.planned) throw std::runtime_error("prepare_*: call before entering planned mode");
    (void)rd;
    auto m = std::make_shared<Material>();
    m->sig = bpmax_sig(rows * cols, o);
    gen_bpmax_material(ctx, rows * cols, o, *m);
    ctx.plan_queue.push_back(m);
}

inline void prepare_layernorm(SCIContext &ctx, size_t rows, size_t cols, const LNOpts &o) {
    if (ctx.planned) throw std::runtime_error("prepare_*: call before entering planned mode");
    auto m = std::make_shared<Material>();
    m->sig = ln_sig(rows, cols, o);
    ctx.plan_queue.push_back(m);
}

inline void prepare_gelu_select(SCIContext &ctx, size_t n, const GeluSelectOpts &o) {
    if (ctx.planned) throw std::runtime_error("prepare_*: call before entering planned mode");
    auto m = std::make_shared<Material>();
    m->sig = gelu_sig(n, o);
    ctx.plan_queue.push_back(m);
}

}
}

#endif
