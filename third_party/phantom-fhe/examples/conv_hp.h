#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <sys/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "phantom.h"
#include "evaluate.cuh"
#include "ntt.cuh"

namespace conv_hp {

using i128 = __int128;
using u128 = unsigned __int128;

constexpr int kF = 13;
constexpr int kSigmaDefault = 40;
constexpr int kKRMax = 104;

struct ConvConfig {
    int F = kF;
    size_t conv_primes = 3;
    int slot_bound_log2 = 6;
};

struct dd {
    double hi, lo;
};
struct cdd {
    dd re, im;
};

static inline dd dd_two_sum(double a, double b) {
    double s = a + b;
    double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
static inline dd dd_quick(double a, double b) {
    double s = a + b;
    return {s, b - (s - a)};
}
static inline dd dd_two_prod(double a, double b) {
    double p = a * b;
#if defined(__FMA__)
    return {p, std::fma(a, b, -p)};
#else
    constexpr double split = 134217729.0;
    double t = split * a;
    double ahi = t - (t - a), alo = a - ahi;
    t = split * b;
    double bhi = t - (t - b), blo = b - bhi;
    return {p, ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo};
#endif
}
static inline dd dd_add(dd a, dd b) {
    dd s = dd_two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return dd_quick(s.hi, s.lo);
}
static inline dd dd_sub(dd a, dd b) { return dd_add(a, {-b.hi, -b.lo}); }
static inline dd dd_mul(dd a, dd b) {
    dd p = dd_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return dd_quick(p.hi, p.lo);
}
static inline dd dd_mul_d(dd a, double b) {
    dd p = dd_two_prod(a.hi, b);
    p.lo += a.lo * b;
    return dd_quick(p.hi, p.lo);
}
static inline dd dd_div_d(dd a, double b) {
    double q1 = a.hi / b;
    dd p = dd_two_prod(q1, b);
    dd s = dd_two_sum(a.hi, -p.hi);
    s.lo -= p.lo;
    s.lo += a.lo;
    double q2 = (s.hi + s.lo) / b;
    return dd_quick(q1, q2);
}
static inline cdd cdd_add(const cdd &a, const cdd &b) { return {dd_add(a.re, b.re), dd_add(a.im, b.im)}; }
static inline cdd cdd_sub(const cdd &a, const cdd &b) { return {dd_sub(a.re, b.re), dd_sub(a.im, b.im)}; }
static inline cdd cdd_mul(const cdd &a, const cdd &b) {
    return {dd_sub(dd_mul(a.re, b.re), dd_mul(a.im, b.im)), dd_add(dd_mul(a.re, b.im), dd_mul(a.im, b.re))};
}
static inline cdd cdd_scale(const cdd &a, dd s) { return {dd_mul(a.re, s), dd_mul(a.im, s)}; }

static inline dd dd_from_i128(i128 t) {
    double hi = static_cast<double>(t);
    double lo = static_cast<double>(t - static_cast<i128>(hi));
    return dd_quick(hi, lo);
}
static inline dd dd_from_i64(int64_t t) {
    double hi = static_cast<double>(t);
    double lo = static_cast<double>(t - static_cast<int64_t>(hi));
    return dd_quick(hi, lo);
}

static inline uint64_t conv_d2u64_mod(double r) {
    if (std::fabs(r) < 9.0e18) return static_cast<uint64_t>(static_cast<int64_t>(r));
    double m = std::fmod(r, 18446744073709551616.0);
    if (m >= 0) return static_cast<uint64_t>(m);
    return uint64_t(0) - static_cast<uint64_t>(-m);
}

static inline uint64_t dd_round_mod64(dd x) {
    double r = std::nearbyint(x.hi);
    double f = (x.hi - r) + x.lo;
    double r2 = std::nearbyint(f);
    return conv_d2u64_mod(r) + static_cast<uint64_t>(static_cast<int64_t>(r2));
}

static inline i128 dd_round_i128(dd x) {
    double r = std::nearbyint(x.hi);
    double f = (x.hi - r) + x.lo;
    double r2 = std::nearbyint(f);
    return static_cast<i128>(r) + static_cast<i128>(static_cast<int64_t>(r2));
}

static inline void dd_sincos_small(dd x, dd &s, dd &c) {
    dd x2 = dd_mul(x, x);
    dd term = x;
    s = x;
    for (int n = 1; n < 18; n++) {
        term = dd_div_d(dd_mul(term, x2), -static_cast<double>((2 * n) * (2 * n + 1)));
        s = dd_add(s, term);
    }
    term = {1.0, 0.0};
    c = {1.0, 0.0};
    for (int n = 1; n < 18; n++) {
        term = dd_div_d(dd_mul(term, x2), -static_cast<double>((2 * n - 1) * (2 * n)));
        c = dd_add(c, term);
    }
}

class Rng {
public:

    Rng() {
        uint8_t seed[44];
        size_t got = 0;
        while (got < sizeof(seed)) {
            ssize_t r = getrandom(seed + got, sizeof(seed) - got, 0);
            if (r <= 0) throw std::runtime_error("conv_hp::Rng: getrandom failed");
            got += static_cast<size_t>(r);
        }
        init(seed, seed + 32, 0);
        std::memset(seed, 0, sizeof(seed));
    }

    Rng(const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) { init(key, nonce, counter); }
    static Rng from_test_seed(uint64_t s) {
        uint8_t key[32] = {0}, nonce[12] = {0};
        std::memcpy(key, &s, sizeof(s));
        return Rng(key, nonce, 0);
    }

    uint32_t next_u32() {
        if (pos_ >= 16) refill();
        return buf_[pos_++];
    }
    uint64_t next_u64() {
        uint64_t lo = next_u32();
        uint64_t hi = next_u32();
        return lo | (hi << 32);
    }

    uint64_t uniform_bits(int bits) {
        if (bits <= 0) return 0;
        uint64_t v = next_u64();
        return bits >= 64 ? v : (v & ((uint64_t(1) << bits) - 1));
    }

    i128 uniform_signed_pow2(int K) {
        int bits = K + 1;
        u128 v;
        if (bits <= 64) {
            v = uniform_bits(bits);
        } else {
            uint64_t lo = next_u64();
            uint64_t hi = uniform_bits(bits - 64);
            v = (static_cast<u128>(hi) << 64) | lo;
        }
        return static_cast<i128>(v) - (static_cast<i128>(1) << K);
    }

    static void chacha20_block(const uint32_t in[16], uint32_t out[16]) {
        uint32_t x[16];
        std::memcpy(x, in, sizeof(x));
        for (int i = 0; i < 10; i++) {
            qr(x[0], x[4], x[8], x[12]);
            qr(x[1], x[5], x[9], x[13]);
            qr(x[2], x[6], x[10], x[14]);
            qr(x[3], x[7], x[11], x[15]);
            qr(x[0], x[5], x[10], x[15]);
            qr(x[1], x[6], x[11], x[12]);
            qr(x[2], x[7], x[8], x[13]);
            qr(x[3], x[4], x[9], x[14]);
        }
        for (int i = 0; i < 16; i++) out[i] = x[i] + in[i];
    }
    const uint32_t *state() const { return st_; }

private:
    uint32_t st_[16];
    uint32_t buf_[16];
    int pos_ = 16;

    static inline uint32_t rotl(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
    static inline void qr(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
        a += b; d ^= a; d = rotl(d, 16);
        c += d; b ^= c; b = rotl(b, 12);
        a += b; d ^= a; d = rotl(d, 8);
        c += d; b ^= c; b = rotl(b, 7);
    }
    static inline uint32_t le32(const uint8_t *p) {
        return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    }
    void init(const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) {
        st_[0] = 0x61707865; st_[1] = 0x3320646e; st_[2] = 0x79622d32; st_[3] = 0x6b206574;
        for (int i = 0; i < 8; i++) st_[4 + i] = le32(key + 4 * i);
        st_[12] = counter;
        for (int i = 0; i < 3; i++) st_[13 + i] = le32(nonce + 4 * i);
        pos_ = 16;
    }
    void refill() {
        chacha20_block(st_, buf_);
        if (++st_[12] == 0) {
            if (++st_[13] == 0)
                if (++st_[14] == 0) ++st_[15];
        }
        pos_ = 0;
    }
};

class HpFFT {
public:
    explicit HpFFT(size_t poly_degree) : N_(poly_degree), n_(poly_degree / 2), M_(2 * poly_degree) {
        if (N_ < 8 || (N_ & (N_ - 1))) throw std::invalid_argument("HpFFT: N must be a power of two >= 8");
        logn_ = 0;
        while ((size_t(1) << logn_) < n_) logn_++;

        const dd two_pi = {6.283185307179586232e+00, 2.449293598294706414e-16};
        std::vector<cdd> base(M_ / 8 + 1);
        for (size_t k = 0; k <= M_ / 8; k++) {
            dd th = dd_mul_d(two_pi, static_cast<double>(k) / static_cast<double>(M_));
            dd s, c;
            dd_sincos_small(th, s, c);
            base[k] = {c, s};
        }
        roots_.resize(M_);
        for (size_t k = 0; k < M_; k++) roots_[k] = root_from_base(base, k);
        rot_group_.resize(n_);
        uint64_t pos = 1;
        for (size_t j = 0; j < n_; j++) {
            rot_group_[j] = static_cast<uint32_t>(pos);
            pos = (pos * 5) & (M_ - 1);
        }
        bitrev_.resize(n_);
        for (size_t i = 0; i < n_; i++) {
            size_t r = 0;
            for (int b = 0; b < logn_; b++)
                if (i & (size_t(1) << b)) r |= size_t(1) << (logn_ - 1 - b);
            bitrev_[i] = static_cast<uint32_t>(r);
        }

        for (size_t len = 2; len <= n_; len <<= 1) {
            size_t lenh = len >> 1, lenq = len << 2, gap = M_ / lenq;
            for (size_t j = 0; j < lenh; j++) tw_fwd_.push_back(roots_[(rot_group_[j] % lenq) * gap]);
        }
        for (size_t len = n_; len >= 2; len >>= 1) {
            size_t lenh = len >> 1, lenq = len << 2, gap = M_ / lenq;
            for (size_t j = 0; j < lenh; j++) tw_inv_.push_back(roots_[(lenq - (rot_group_[j] % lenq)) * gap]);
        }
    }

    size_t poly_degree() const { return N_; }
    size_t slots() const { return n_; }
    const cdd &root(size_t k) const { return roots_[k & (M_ - 1)]; }
    uint32_t rot_group(size_t j) const { return rot_group_[j]; }

    void forward(cdd *v) const {
        bit_reverse(v);
        size_t off = 0;
        for (size_t len = 2; len <= n_; len <<= 1) {
            size_t lenh = len >> 1;
            const cdd *tw = tw_fwd_.data() + off;
            off += lenh;
            for (size_t i = 0; i < n_; i += len) {
                cdd *a = v + i, *b = v + i + lenh;
                for (size_t j = 0; j < lenh; j++) {
                    cdd u = a[j];
                    cdd w = cdd_mul(b[j], tw[j]);
                    a[j] = cdd_add(u, w);
                    b[j] = cdd_sub(u, w);
                }
            }
        }
    }

    void inverse(cdd *v) const {
        size_t off = 0;
        for (size_t len = n_; len >= 2; len >>= 1) {
            size_t lenh = len >> 1;
            const cdd *tw = tw_inv_.data() + off;
            off += lenh;
            for (size_t i = 0; i < n_; i += len) {
                cdd *a = v + i, *b = v + i + lenh;
                for (size_t j = 0; j < lenh; j++) {
                    cdd u = cdd_add(a[j], b[j]);
                    cdd w = cdd_mul(cdd_sub(a[j], b[j]), tw[j]);
                    a[j] = u;
                    b[j] = w;
                }
            }
        }
        bit_reverse(v);
        const double inv_n = 1.0 / static_cast<double>(n_);
        for (size_t i = 0; i < n_; i++) {
            v[i].re.hi *= inv_n; v[i].re.lo *= inv_n;
            v[i].im.hi *= inv_n; v[i].im.lo *= inv_n;
        }
    }

    void decode_dd(const i128 *coeffs, dd mult, cdd *out) const {
        for (size_t k = 0; k < n_; k++) out[k] = {dd_from_i128(coeffs[k]), dd_from_i128(coeffs[k + n_])};
        forward(out);
        for (size_t k = 0; k < n_; k++) out[k] = cdd_scale(out[k], mult);
    }

    void decode_round_mod64(const i128 *coeffs, double scale, int F, bool negate, uint64_t *re,
                            uint64_t *im) const {
        std::vector<cdd> buf(n_);
        decode_dd(coeffs, dd_div_d({std::ldexp(1.0, F), 0.0}, scale), buf.data());
        for (size_t j = 0; j < n_; j++) {
            uint64_t r = dd_round_mod64(buf[j].re), i = dd_round_mod64(buf[j].im);
            re[j] = negate ? uint64_t(0) - r : r;
            im[j] = negate ? uint64_t(0) - i : i;
        }
    }

    void encode_fixed(const int64_t *y_re, const int64_t *y_im, size_t n, double scale, int F,
                      i128 *coeffs_out, cdd *dd_out = nullptr) const {
        if (n > n_) throw std::invalid_argument("HpFFT::encode_fixed: too many slots");
        std::vector<cdd> buf(n_);
        dd mult = dd_mul_d({scale, 0.0}, std::ldexp(1.0, -F));
        for (size_t j = 0; j < n_; j++) {
            if (j < n) buf[j] = {dd_mul(dd_from_i64(y_re[j]), mult), dd_mul(dd_from_i64(y_im[j]), mult)};
            else buf[j] = {{0, 0}, {0, 0}};
        }
        inverse(buf.data());
        for (size_t k = 0; k < n_; k++) {
            coeffs_out[k] = dd_round_i128(buf[k].re);
            coeffs_out[k + n_] = dd_round_i128(buf[k].im);
        }
        if (dd_out) std::copy(buf.begin(), buf.end(), dd_out);
    }

private:
    size_t N_, n_, M_;
    int logn_;
    std::vector<cdd> roots_, tw_fwd_, tw_inv_;
    std::vector<uint32_t> rot_group_, bitrev_;

    cdd root_from_base(const std::vector<cdd> &base, size_t index) const {
        index &= M_ - 1;
        auto neg = [](dd a) { return dd{-a.hi, -a.lo}; };
        if (index <= M_ / 8) return base[index];
        if (index <= M_ / 4) {
            const cdd &a = base[M_ / 4 - index];
            return {a.im, a.re};
        }
        if (index <= M_ / 2) {
            cdd a = root_from_base(base, M_ / 2 - index);
            return {neg(a.re), a.im};
        }
        if (index <= 3 * M_ / 4) {
            cdd a = root_from_base(base, index - M_ / 2);
            return {neg(a.re), neg(a.im)};
        }
        cdd a = root_from_base(base, M_ - index);
        return {a.re, neg(a.im)};
    }
    void bit_reverse(cdd *v) const {
        for (size_t i = 0; i < n_; i++) {
            size_t r = bitrev_[i];
            if (i < r) std::swap(v[i], v[r]);
        }
    }
};

inline const HpFFT &hp_fft(size_t N) {
    static std::mutex mu;
    static std::map<size_t, std::unique_ptr<HpFFT>> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto &p = cache[N];
    if (!p) p = std::make_unique<HpFFT>(N);
    return *p;
}

struct ShoupConst {
    uint64_t w, wp;
};
static inline ShoupConst shoup_const(uint64_t w, uint64_t q) {
    return {w, static_cast<uint64_t>((static_cast<u128>(w) << 64) / q)};
}

static inline uint64_t mul_shoup(uint64_t x, ShoupConst c, uint64_t q) {
    uint64_t qhat = static_cast<uint64_t>((static_cast<u128>(x) * c.wp) >> 64);
    uint64_t r = x * c.w - qhat * q;
    return r >= q ? r - q : r;
}
static inline uint64_t conv_powmod(uint64_t a, uint64_t e, uint64_t q) {
    u128 r = 1, b = a % q;
    while (e) {
        if (e & 1) r = (r * b) % q;
        b = (b * b) % q;
        e >>= 1;
    }
    return static_cast<uint64_t>(r);
}

class RnsLevel {
public:
    RnsLevel() = default;
    RnsLevel(const PhantomContext &ctx, size_t chain_index) : chain_index_(chain_index) {
        const auto &cm = ctx.get_context_data(chain_index).parms().coeff_modulus();
        for (const auto &m : cm) q_.push_back(m.value());
        N_ = ctx.get_context_data(chain_index).parms().poly_modulus_degree();
        const size_t k = q_.size();
        for (auto q : q_)
            if (q >= (uint64_t(1) << 62)) throw std::invalid_argument("RnsLevel: prime >= 2^62");
        one_.resize(k);
        r64_.resize(k);
        inv_.resize(k);
        pm_.assign(k * k, {0, 0});
        for (size_t i = 0; i < k; i++) {
            one_[i] = shoup_const(1, q_[i]);
            uint64_t r64 = static_cast<uint64_t>((static_cast<u128>(1) << 64) % q_[i]);
            r64_[i] = shoup_const(r64, q_[i]);

            uint64_t p = 1 % q_[i];
            for (size_t j = 0; j < i; j++) {
                pm_[i * k + j] = shoup_const(p, q_[i]);
                p = static_cast<uint64_t>((static_cast<u128>(p) * (q_[j] % q_[i])) % q_[i]);
            }
            inv_[i] = shoup_const(i == 0 ? 1 : conv_powmod(p, q_[i] - 2, q_[i]), q_[i]);
        }

        prefix_.resize(k);
        limit_.resize(k);
        fits_.resize(k);
        i128 P = 1;
        bool fits = true;
        log2_q_ = 0;
        for (size_t j = 0; j < k; j++) {
            fits_[j] = fits;
            prefix_[j] = fits ? P : 0;
            limit_[j] = fits ? ((static_cast<i128>(1) << 125) / P) : 0;
            if (fits) {
                double lp = std::log2(static_cast<double>(P)) + std::log2(static_cast<double>(q_[j]));
                if (lp < 125.5) P *= static_cast<i128>(q_[j]);
                else fits = false;
            }
            log2_q_ += std::log2(static_cast<double>(q_[j]));
        }
    }

    size_t size() const { return q_.size(); }
    size_t poly_degree() const { return N_; }
    size_t chain_index() const { return chain_index_; }
    double log2_q() const { return log2_q_; }
    const std::vector<uint64_t> &primes() const { return q_; }

    void to_int128(const uint64_t *res, i128 *out) const {
        const size_t k = q_.size();
        if (k > 8) throw std::invalid_argument("RnsLevel::to_int128: at most 8 primes");
        int64_t d[8];
        for (size_t c = 0; c < N_; c++) {
            uint64_t a0 = res[c];
            d[0] = a0 > q_[0] / 2 ? static_cast<int64_t>(a0) - static_cast<int64_t>(q_[0]) : static_cast<int64_t>(a0);
            for (size_t i = 1; i < k; i++) {
                const uint64_t qi = q_[i];
                uint64_t acc = 0;
                for (size_t j = 0; j < i; j++) {
                    uint64_t dj = d[j] >= 0 ? mul_shoup(static_cast<uint64_t>(d[j]), one_[i], qi)
                                            : neg_mod(mul_shoup(static_cast<uint64_t>(-d[j]), one_[i], qi), qi);
                    acc += mul_shoup(dj, pm_[i * k + j], qi);
                    if (acc >= qi) acc -= qi;
                }
                uint64_t ai = res[i * N_ + c];
                uint64_t diff = ai >= acc ? ai - acc : ai + qi - acc;
                uint64_t di = mul_shoup(diff, inv_[i], qi);
                d[i] = di > qi / 2 ? static_cast<int64_t>(di) - static_cast<int64_t>(qi) : static_cast<int64_t>(di);
            }
            i128 v = 0;
            for (size_t j = 0; j < k; j++) {
                if (d[j] == 0) continue;
                i128 dj = d[j];
                if (!fits_[j] || (dj < 0 ? -dj : dj) > limit_[j])
                    throw std::runtime_error("conv_hp: centred coefficient exceeds 2^126 (decryption out of range)");
                v += prefix_[j] * dj;
            }
            out[c] = v;
        }
    }

    void from_int128(const i128 *t, uint64_t *res) const {
        const size_t k = q_.size();
        for (size_t c = 0; c < N_; c++) {
            i128 v = t[c];
            bool neg = v < 0;
            u128 u = neg ? static_cast<u128>(-v) : static_cast<u128>(v);
            uint64_t lo = static_cast<uint64_t>(u), hi = static_cast<uint64_t>(u >> 64);
            for (size_t i = 0; i < k; i++) {
                uint64_t qi = q_[i];
                uint64_t r = mul_shoup(hi, r64_[i], qi) + mul_shoup(lo, one_[i], qi);
                if (r >= qi) r -= qi;
                res[i * N_ + c] = neg ? neg_mod(r, qi) : r;
            }
        }
    }

private:
    size_t chain_index_ = 0, N_ = 0;
    double log2_q_ = 0;
    std::vector<uint64_t> q_;
    std::vector<ShoupConst> one_, r64_, inv_, pm_;
    std::vector<i128> prefix_, limit_;
    std::vector<bool> fits_;
    static inline uint64_t neg_mod(uint64_t r, uint64_t q) { return r == 0 ? 0 : q - r; }
};

static __global__ void conv_addsub_mod_kernel(uint64_t *dst, const uint64_t *src, const DModulus *modulus,
                                              size_t N, size_t k, int subtract) {
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N * k; tid += blockDim.x * gridDim.x) {
        uint64_t q = modulus[tid / N].value();
        uint64_t a = dst[tid], b = src[tid];
        uint64_t r;
        if (subtract) r = a >= b ? a - b : a + (q - b);
        else {
            r = a + b;
            if (r >= q) r -= q;
        }
        dst[tid] = r;
    }
}

inline void conv_check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("conv_hp CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

inline phantom::util::cuda_auto_ptr<uint64_t> conv_int_poly_to_ntt(const PhantomContext &ctx, const RnsLevel &lvl,
                                                                    const i128 *coeffs) {
    const auto &s = cudaStreamPerThread;
    const size_t N = lvl.poly_degree(), k = lvl.size();
    std::vector<uint64_t> res(k * N);
    lvl.from_int128(coeffs, res.data());
    auto d = phantom::util::make_cuda_auto_ptr<uint64_t>(k * N, s);
    conv_check_cuda(cudaMemcpyAsync(d.get(), res.data(), k * N * sizeof(uint64_t), cudaMemcpyHostToDevice, s), "H2D");
    nwt_2d_radix8_forward_inplace(d.get(), ctx.gpu_rns_tables(), k, 0, s);
    conv_check_cuda(cudaStreamSynchronize(s), "int_poly_to_ntt");
    return d;
}

inline void conv_ntt_poly_to_int(const PhantomContext &ctx, const RnsLevel &lvl, const uint64_t *d_ntt,
                                 i128 *coeffs_out) {
    const auto &s = cudaStreamPerThread;
    const size_t N = lvl.poly_degree(), k = lvl.size();
    auto tmp = phantom::util::make_cuda_auto_ptr<uint64_t>(k * N, s);
    conv_check_cuda(cudaMemcpyAsync(tmp.get(), d_ntt, k * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s), "D2D");
    nwt_2d_radix8_backward_inplace(tmp.get(), ctx.gpu_rns_tables(), k, 0, s);
    std::vector<uint64_t> res(k * N);
    conv_check_cuda(cudaMemcpyAsync(res.data(), tmp.get(), k * N * sizeof(uint64_t), cudaMemcpyDeviceToHost, s), "D2H");
    conv_check_cuda(cudaStreamSynchronize(s), "ntt_poly_to_int");
    lvl.to_int128(res.data(), coeffs_out);
}

inline void conv_add_to_c0(const PhantomContext &ctx, PhantomCiphertext &ct, const uint64_t *d_ntt, bool subtract) {
    const auto &s = cudaStreamPerThread;
    const size_t N = ct.poly_modulus_degree(), k = ct.coeff_modulus_size();
    if (!ct.is_ntt_form()) throw std::invalid_argument("conv_hp: ciphertext must be in NTT form");
    const size_t total = N * k;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((total + threads - 1) / threads, 65535));
    conv_addsub_mod_kernel<<<blocks, threads, 0, s>>>(ct.data(), d_ntt, ctx.gpu_rns_tables().modulus(), N, k,
                                                      subtract ? 1 : 0);
    conv_check_cuda(cudaGetLastError(), "conv_addsub_mod_kernel");
}

struct ConvBudget {
    size_t chain_index = 0;
    size_t primes = 0;
    double log2_q = 0;
    double log2_bcoef = 0;
    int K_R = 0;
    int sigma = 0;
    int sigma_max = 0;
    bool ok = false;
};

inline ConvBudget conv_budget(const PhantomContext &ctx, size_t chain_index, double scale, int sigma,
                              const ConvConfig &cfg = {}) {
    ConvBudget b;
    const auto &parms = ctx.get_context_data(chain_index).parms();
    b.chain_index = chain_index;
    b.primes = parms.coeff_modulus().size();
    for (const auto &m : parms.coeff_modulus()) b.log2_q += std::log2(static_cast<double>(m.value()));
    const double log2_N = std::log2(static_cast<double>(parms.poly_modulus_degree()));
    b.log2_bcoef = std::log2(scale) + cfg.slot_bound_log2 + 1.0;
    const int base = static_cast<int>(std::ceil(log2_N + b.log2_bcoef));
    b.sigma = sigma;
    b.K_R = base + sigma;

    b.ok = (b.K_R + 2 < b.log2_q) && b.K_R <= kKRMax;
    int kmax = std::min(static_cast<int>(std::ceil(b.log2_q - 2.0)) - 1, kKRMax);
    b.sigma_max = kmax - base;
    return b;
}

inline size_t conv_chain_for_primes(const PhantomContext &ctx, size_t primes) {
    for (size_t c = ctx.get_first_index(); c < ctx.total_parm_size(); c++)
        if (ctx.get_context_data(c).parms().coeff_modulus().size() == primes) return c;
    throw std::invalid_argument("conv_hp: no chain level with the requested number of primes");
}

struct C2MServerOut {
    PhantomCiphertext ct_masked;
    std::vector<uint64_t> s1_re, s1_im;
    ConvBudget budget;
};

struct C2MPrecomp {
    size_t chain_index = 0;
    double scale = 0;
    ConvBudget budget;
    PhantomCiphertext mask_ct;
    std::vector<uint64_t> s1_re, s1_im;
};

inline PhantomCiphertext conv_enc_pk_zero(const PhantomContext &ctx, const PhantomPublicKey &pk, size_t chain_index,
                                          double scale) {

    PhantomCiphertext z = const_cast<PhantomPublicKey &>(pk).encrypt_zero_asymmetric(ctx);
    if (z.chain_index() != chain_index) phantom::mod_switch_to_inplace(ctx, z, chain_index);
    z.set_scale(scale);
    return z;
}

inline C2MPrecomp c2m_server_precompute(const PhantomContext &ctx, const PhantomPublicKey &pk, size_t chain_index,
                                        double scale, Rng &rng, int sigma = kSigmaDefault,
                                        const ConvConfig &cfg = {}) {
    C2MPrecomp p;
    p.chain_index = chain_index;
    p.scale = scale;
    p.budget = conv_budget(ctx, chain_index, scale, sigma, cfg);
    if (!p.budget.ok)
        throw std::invalid_argument("c2m_server: level " + std::to_string(chain_index) + " (" +
                                    std::to_string(p.budget.primes) + " primes, " +
                                    std::to_string(p.budget.log2_q) + " bits) cannot hold K_R = " +
                                    std::to_string(p.budget.K_R) + " (sigma " + std::to_string(sigma) +
                                    "); max sigma here is " + std::to_string(p.budget.sigma_max));
    RnsLevel lvl(ctx, chain_index);
    const size_t N = lvl.poly_degree();
    const HpFFT &fft = hp_fft(N);
    std::vector<i128> R(N);
    for (size_t c = 0; c < N; c++) R[c] = rng.uniform_signed_pow2(p.budget.K_R);
    p.s1_re.resize(N / 2);
    p.s1_im.resize(N / 2);
    fft.decode_round_mod64(R.data(), scale, cfg.F,  true, p.s1_re.data(), p.s1_im.data());
    p.mask_ct = conv_enc_pk_zero(ctx, pk, chain_index, scale);
    auto dR = conv_int_poly_to_ntt(ctx, lvl, R.data());
    conv_add_to_c0(ctx, p.mask_ct, dR.get(), false);
    conv_check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "c2m_server_precompute");
    std::fill(R.begin(), R.end(), i128(0));
    return p;
}

inline C2MServerOut c2m_server_apply(const PhantomContext &ctx, const PhantomCiphertext &ct, C2MPrecomp &&p) {
    if (ct.size() != 2) throw std::invalid_argument("c2m_server: ciphertext must have size 2 (relinearise first)");
    if (ct.chain_index() > p.chain_index)
        throw std::invalid_argument("c2m_server: ciphertext is below the conversion level");
    if (ct.scale() != p.scale) throw std::invalid_argument("c2m_server: ciphertext scale differs from precomputed");
    C2MServerOut out;
    out.ct_masked = ct;
    if (out.ct_masked.chain_index() != p.chain_index) phantom::mod_switch_to_inplace(ctx, out.ct_masked, p.chain_index);
    phantom::add_inplace(ctx, out.ct_masked, p.mask_ct);
    conv_check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "c2m_server_apply");
    out.s1_re = std::move(p.s1_re);
    out.s1_im = std::move(p.s1_im);
    out.budget = p.budget;
    return out;
}

inline size_t c2m_conv_chain(const PhantomContext &ctx, const PhantomCiphertext &ct, const ConvConfig &cfg) {
    size_t target = conv_chain_for_primes(ctx, cfg.conv_primes);
    return std::max(target, ct.chain_index());
}

inline C2MServerOut c2m_server(const PhantomContext &ctx, const PhantomPublicKey &pk, const PhantomCiphertext &ct,
                               Rng &rng, int sigma = kSigmaDefault, const ConvConfig &cfg = {}) {
    size_t chain = c2m_conv_chain(ctx, ct, cfg);
    return c2m_server_apply(ctx, ct, c2m_server_precompute(ctx, pk, chain, ct.scale(), rng, sigma, cfg));
}

struct M2CServerMsg {
    struct ToClient {
        std::vector<uint64_t> re, im;
    } to_client;
    struct Kept {
        std::vector<uint64_t> r_re, r_im;
        int F = kF;
    } kept;
};

inline M2CServerMsg m2c_server_mask(const uint64_t *s1_re, const uint64_t *s1_im, size_t n, Rng &rng,
                                    int sigma = kSigmaDefault, const ConvConfig &cfg = {}) {
    const int b = cfg.slot_bound_log2 + cfg.F;
    const int rbits = b + sigma;
    if (rbits + 2 > 64 || rbits < 0)
        throw std::invalid_argument("m2c_server_mask: need b + sigma + 2 <= 64 (b = " + std::to_string(b) +
                                    ", sigma = " + std::to_string(sigma) + ")");
    M2CServerMsg m;
    m.kept.F = cfg.F;
    m.to_client.re.resize(n);
    m.to_client.im.resize(n);
    m.kept.r_re.resize(n);
    m.kept.r_im.resize(n);
    for (size_t j = 0; j < n; j++) {
        uint64_t rr = rng.uniform_bits(rbits), ri = rng.uniform_bits(rbits);
        m.kept.r_re[j] = rr;
        m.kept.r_im[j] = ri;
        m.to_client.re[j] = s1_re[j] + rr;
        m.to_client.im[j] = s1_im[j] + ri;
    }
    return m;
}

inline void m2c_server_finish(const PhantomContext &ctx, PhantomCiphertext &ct, const M2CServerMsg::Kept &kept) {
    RnsLevel lvl(ctx, ct.chain_index());
    const size_t N = lvl.poly_degree();
    const HpFFT &fft = hp_fft(N);
    const size_t n = kept.r_re.size();
    std::vector<int64_t> rr(n), ri(n);
    for (size_t j = 0; j < n; j++) {
        rr[j] = static_cast<int64_t>(kept.r_re[j]);
        ri[j] = static_cast<int64_t>(kept.r_im[j]);
    }
    std::vector<i128> coeffs(N);
    fft.encode_fixed(rr.data(), ri.data(), n, ct.scale(), kept.F, coeffs.data());
    auto d = conv_int_poly_to_ntt(ctx, lvl, coeffs.data());
    conv_add_to_c0(ctx, ct, d.get(), true);
    conv_check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "m2c_server_finish");
}

}
