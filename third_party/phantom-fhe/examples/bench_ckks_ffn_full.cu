#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "example.h"
#include "phantom.h"
#include "util.cuh"
#include "evaluate.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

using pc64 = cuDoubleComplex;

namespace {

constexpr int SEED = 114514;

#include "model_config.h"
constexpr int NSLOTS  = EF_NSLOTS;
constexpr int M       = EF_M;
constexpr int C       = EF_C;
constexpr int N1_FF1  = EF_N1_DEFAULT;
constexpr int N2_FF1  = EF_N2_DEFAULT;
constexpr int N1_FF2  = EF_N1_FF2;
constexpr int N2_FF2  = EF_N2_FF2;

constexpr int D1      = EF_D;
constexpr int DMID    = EF_D_FF;
constexpr int D2      = EF_D;

constexpr int G1      = EF_G_FF1;
constexpr int B1      = EF_B_FF1;
constexpr int G2      = EF_G_FF2;
constexpr int B2      = EF_B_FF2;
constexpr int HP1     = EF_HP_FF1;
constexpr int HP2     = EF_HP_FF2;
constexpr int C_IN_FF1 = EF_C_USED_FF1_IN;
constexpr int C_IN_FF2 = EF_C_USED_FF2_IN;

inline pc64 pcplx(double r, double i = 0.0) { return make_cuDoubleComplex(r, i); }
inline double preal(const pc64 &z) { return cuCreal(z); }

inline void cuda_sync() {
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

inline double ms_since(std::chrono::high_resolution_clock::time_point t0,
                       std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count() * 1e3;
}

static uint64_t modinv_u64(uint64_t a, uint64_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(m), newr = static_cast<int64_t>(a % m);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp_t = t - q * newt;
        t = newt;
        newt = tmp_t;
        int64_t tmp_r = r - q * newr;
        r = newr;
        newr = tmp_r;
    }
    if (r != 1) throw std::runtime_error("modinv failed");
    if (t < 0) t += static_cast<int64_t>(m);
    return static_cast<uint64_t>(t);
}

static uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    uint64_t x = a % m;
    while (e) {
        if (e & 1) r = static_cast<uint64_t>((__uint128_t)r * x % m);
        x = static_cast<uint64_t>((__uint128_t)x * x % m);
        e >>= 1;
    }
    return r;
}

static int norm_step(int step, int slots) {
    step %= slots;
    if (step > slots / 2) step -= slots;
    if (step < -slots / 2) step += slots;
    return step;
}

static uint32_t galois_elt_from_step(int step, uint32_t gen, uint64_t m) {
    if (step == 0) return 1u;
    bool neg = (step < 0);
    uint64_t e = static_cast<uint64_t>(neg ? -(int64_t)step : (int64_t)step);
    uint64_t g = powmod_u64(static_cast<uint64_t>(gen), e, m);
    if (neg) g = modinv_u64(g, m);
    return static_cast<uint32_t>(g);
}

static double rel_err_real(const std::vector<double> &a, const std::vector<double> &b) {
    long double num = 0.0L;
    long double den = 0.0L;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        const long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        num += d * d;
        const long double v = static_cast<long double>(b[i]);
        den += v * v;
    }
    return static_cast<double>(std::sqrt(static_cast<double>(num)) /
                               (std::sqrt(static_cast<double>(den)) + 1e-18));
}

static void roll_right(const double *src, int n, int shift, std::vector<double> &dst) {
    const int s = ((shift % n) + n) % n;
    if (s == 0) {
        for (int i = 0; i < n; i++) dst[i] = src[i];
        return;
    }
    for (int i = 0; i < n; i++) {
        const int j = (i + s) % n;
        dst[j] = src[i];
    }
}

static void roll_right_c(const pc64 *src, int n, int shift, std::vector<pc64> &dst) {
    const int s = ((shift % n) + n) % n;
    if (s == 0) {
        for (int i = 0; i < n; i++) dst[i] = src[i];
        return;
    }
    for (int i = 0; i < n; i++) {
        const int j = (i + s) % n;
        dst[j] = src[i];
    }
}

static inline size_t tab_idx(int ge, int t, int b, int j, int c, int blocks) {
    return ((((size_t)ge * (size_t)c + (size_t)t) * (size_t)blocks + (size_t)b) * (size_t)c + (size_t)j);
}

static std::vector<double> build_wtab_real(const std::vector<double> &w, int d_in, int d_out, int c) {
    const int g = d_in / c;
    const int blocks = d_out / c;
    std::vector<double> tab((size_t)g * (size_t)c * (size_t)blocks * (size_t)c, 0.0);
    for (int ge = 0; ge < g; ge++) {
        const int off = ge * c;
        for (int t = 0; t < c; t++) {
            for (int b = 0; b < blocks; b++) {
                for (int j = 0; j < c; j++) {
                    const int row = off + ((j + t) % c);
                    const int col = b * c + j;
                    tab[tab_idx(ge, t, b, j, c, blocks)] = w[(size_t)row * (size_t)d_out + (size_t)col];
                }
            }
        }
    }
    return tab;
}

static std::vector<pc64> build_wtab_paired(const std::vector<double> &w, int d_in, int d_out, int c,
                                            int c_in = -1) {
    if (c_in < 0) c_in = c;
    const int g = d_in / c_in;
    const int hp = g / 2;
    const int blocks = d_out / c;
    std::vector<pc64> tab((size_t)hp * (size_t)c * (size_t)blocks * (size_t)c, pcplx(0.0, 0.0));
    for (int h = 0; h < hp; h++) {
        const int off_e = (2 * h) * c_in;
        const int off_o = (2 * h + 1) * c_in;
        for (int t = 0; t < c; t++) {
            for (int b = 0; b < blocks; b++) {
                for (int j = 0; j < c; j++) {
                    const int input_seg = ((j + t) % c);
                    if (input_seg >= c_in) continue;
                    const int row_e = off_e + input_seg;
                    const int row_o = off_o + input_seg;
                    const int col = b * c + j;
                    double we = w[(size_t)row_e * (size_t)d_out + (size_t)col];
                    double wo = w[(size_t)row_o * (size_t)d_out + (size_t)col];
                    tab[tab_idx(h, t, b, j, c, blocks)] = pcplx(0.5 * we, -0.5 * wo);
                }
            }
        }
    }
    return tab;
}

static std::vector<double> matmul_ref(const std::vector<double> &a, int m, int k, const std::vector<double> &w, int n) {
    std::vector<double> y((size_t)m * (size_t)n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int t = 0; t < k; t++) {
                acc += a[(size_t)i * (size_t)k + (size_t)t] * w[(size_t)t * (size_t)n + (size_t)j];
            }
            y[(size_t)i * (size_t)n + (size_t)j] = acc;
        }
    }
    return y;
}

static std::vector<PhantomCiphertext> encrypt_packed_groups(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    PhantomSecretKey &sk,
    const std::vector<double> &x,
    int m,
    int d,
    int g,
    double scale
) {
    const int c = NSLOTS / m;
    const size_t slots = encoder.slot_count();
    std::vector<PhantomCiphertext> out(g);
    for (int ge = 0; ge < g; ge++) {
        std::vector<pc64> msg(slots, pcplx(0.0, 0.0));
        for (int col = 0; col < c; col++) {
            const int global_col = ge * c + col;
            for (int i = 0; i < m; i++) {
                msg[(size_t)col * (size_t)m + (size_t)i] = pcplx(x[(size_t)i * (size_t)d + (size_t)global_col], 0.0);
            }
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, 1);
        sk.encrypt_symmetric(ctx, pt, out[ge]);
    }
    return out;
}

static std::vector<PhantomCiphertext> encrypt_packed_pairs(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    PhantomSecretKey &sk,
    const std::vector<double> &x,
    int m, int d, int hp,
    double scale,
    int c_in = -1
) {
    const int c = NSLOTS / m;
    if (c_in < 0) c_in = c;
    const size_t slots = encoder.slot_count();
    std::vector<PhantomCiphertext> out(hp);
    for (int h = 0; h < hp; h++) {
        std::vector<pc64> msg(slots, pcplx(0.0, 0.0));
        for (int col = 0; col < c_in; col++) {
            const int col_e = (2 * h) * c_in + col;
            const int col_o = (2 * h + 1) * c_in + col;
            for (int i = 0; i < m; i++) {
                double re = (col_e < d) ? x[(size_t)i * (size_t)d + (size_t)col_e] : 0.0;
                double im = (col_o < d) ? x[(size_t)i * (size_t)d + (size_t)col_o] : 0.0;
                msg[(size_t)col * (size_t)m + (size_t)i] = pcplx(re, im);
            }
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, 1);
        sk.encrypt_symmetric(ctx, pt, out[h]);
    }
    return out;
}

struct KSCounters {
    long long rots = 0;
    long long muls_ctct = 0;
    long long conj = 0;
};

static std::vector<std::vector<PhantomCiphertext>> build_babies(
    const PhantomContext &ctx,
    const PhantomGaloisKey &gk,
    const std::vector<PhantomCiphertext> &base,
    int n1,
    int m,
    KSCounters &ks
) {
    const int g = static_cast<int>(base.size());
    std::vector<std::vector<PhantomCiphertext>> gp(g, std::vector<PhantomCiphertext>(n1));
    for (int ge = 0; ge < g; ge++) {
        gp[ge][0] = base[ge];
        for (int q = 1; q < n1; q++) {
            PhantomCiphertext ct = base[ge];
            rotate_inplace(ctx, ct, norm_step(q * m, NSLOTS), gk);
            gp[ge][q] = ct;
            ks.rots += 1;
        }
    }
    return gp;
}

static std::vector<PhantomCiphertext> linear_real_dense(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    const PhantomGaloisKey &gk,
    const std::vector<std::vector<PhantomCiphertext>> &gp,
    const std::vector<double> &wtab,
    int d_out,
    int m,
    int n1,
    int n2,
    int c,
    double scale_w,
    const PhantomCiphertext &ct_zero_mul,
    KSCounters &ks
) {
    const int g = static_cast<int>(gp.size());
    const int blocks = d_out / c;
    const size_t slots = encoder.slot_count();

    std::vector<double> wr(c, 0.0);
    std::vector<pc64> pt_msg(slots, pcplx(0.0, 0.0));
    std::vector<std::vector<PhantomCiphertext>> cf(blocks, std::vector<PhantomCiphertext>(n2, ct_zero_mul));

    for (int b = 0; b < blocks; b++) {
        for (int p = 0; p < n2; p++) {
            const int p_shift = (p * n1) % c;
            for (int q = 0; q < n1; q++) {
                const int t = (p_shift + q) % c;
                bool has_h = false;
                PhantomCiphertext h_acc;
                for (int ge = 0; ge < g; ge++) {
                    const double *src = &wtab[tab_idx(ge, t, b, 0, c, blocks)];
                    if (p_shift == 0) {
                        for (int j = 0; j < c; j++) wr[j] = src[j];
                    } else {
                        roll_right(src, c, p_shift, wr);
                    }

                    for (int seg = 0; seg < c; seg++) {
                        const pc64 vv = pcplx(wr[seg], 0.0);
                        const int base = seg * m;
                        for (int i = 0; i < m; i++) {
                            pt_msg[(size_t)base + (size_t)i] = vv;
                        }
                    }

                    PhantomPlaintext pt;
                    encoder.encode(ctx, pt_msg, scale_w, pt, 1);

                    PhantomCiphertext term = gp[ge][q];
                    multiply_plain_inplace(ctx, term, pt);
                    if (!has_h) {
                        h_acc = term;
                        has_h = true;
                    } else {
                        add_inplace(ctx, h_acc, term);
                    }
                }

                if (!has_h) continue;
                if (q == 0) {
                    cf[b][p] = h_acc;
                } else {
                    add_inplace(ctx, cf[b][p], h_acc);
                }
            }
        }
    }

    std::vector<PhantomCiphertext> out(blocks, ct_zero_mul);
    for (int b = 0; b < blocks; b++) {
        bool inited = false;
        PhantomCiphertext acc;
        for (int p = 0; p < n2; p++) {
            PhantomCiphertext term = cf[b][p];
            if (p != 0) {
                const int sh = norm_step(p * n1 * m, NSLOTS);
                rotate_inplace(ctx, term, sh, gk);
                ks.rots += 1;
            }
            if (!inited) {
                acc = term;
                inited = true;
            } else {
                add_inplace(ctx, acc, term);
            }
        }
        if (inited) out[b] = acc;
    }

    return out;
}

static std::vector<PhantomCiphertext> linear_complex_paired(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    const PhantomGaloisKey &gk,
    const std::vector<std::vector<PhantomCiphertext>> &gp,
    const std::vector<pc64> &wtab,
    int d_out, int m, int n1, int n2, int c,
    double scale_w,
    const PhantomCiphertext &ct_zero_mul,
    KSCounters &ks
) {
    const int hp = static_cast<int>(gp.size());
    const int blocks = d_out / c;
    const size_t slots = encoder.slot_count();

    std::vector<pc64> wr(c, pcplx(0.0, 0.0));
    std::vector<pc64> pt_msg(slots, pcplx(0.0, 0.0));
    std::vector<std::vector<PhantomCiphertext>> cf(blocks, std::vector<PhantomCiphertext>(n2, ct_zero_mul));

    for (int b = 0; b < blocks; b++) {
        for (int p = 0; p < n2; p++) {
            const int p_shift = (p * n1) % c;
            for (int q = 0; q < n1; q++) {
                const int t = (p_shift + q) % c;
                bool has_h = false;
                PhantomCiphertext h_acc;
                for (int h = 0; h < hp; h++) {
                    const pc64 *src = &wtab[tab_idx(h, t, b, 0, c, blocks)];
                    if (p_shift == 0) {
                        for (int j = 0; j < c; j++) wr[j] = src[j];
                    } else {
                        roll_right_c(src, c, p_shift, wr);
                    }
                    for (int seg = 0; seg < c; seg++) {
                        const pc64 vv = wr[seg];
                        const int base = seg * m;
                        for (int i = 0; i < m; i++) {
                            pt_msg[(size_t)base + (size_t)i] = vv;
                        }
                    }

                    PhantomPlaintext pt;
                    encoder.encode(ctx, pt_msg, scale_w, pt, 1);

                    PhantomCiphertext term = gp[h][q];
                    multiply_plain_inplace(ctx, term, pt);
                    if (!has_h) {
                        h_acc = term;
                        has_h = true;
                    } else {
                        add_inplace(ctx, h_acc, term);
                    }
                }
                if (!has_h) continue;
                if (q == 0) {
                    cf[b][p] = h_acc;
                } else {
                    add_inplace(ctx, cf[b][p], h_acc);
                }
            }
        }
    }

    std::vector<PhantomCiphertext> out(blocks, ct_zero_mul);
    for (int b = 0; b < blocks; b++) {
        bool inited = false;
        PhantomCiphertext acc;
        for (int p = 0; p < n2; p++) {
            PhantomCiphertext term = cf[b][p];
            if (p != 0) {
                const int sh = norm_step(p * n1 * m, NSLOTS);
                rotate_inplace(ctx, term, sh, gk);
                ks.rots += 1;
            }
            if (!inited) {
                acc = term;
                inited = true;
            } else {
                add_inplace(ctx, acc, term);
            }
        }
        if (inited) out[b] = acc;
    }
    return out;
}

static void ct_real_blocks(
    const PhantomContext &ctx,
    const PhantomGaloisKey &gk,
    std::vector<PhantomCiphertext> &cts,
    size_t conj_elt,
    KSCounters &ks
) {
    for (auto &ct : cts) {
        PhantomCiphertext ct_conj = ct;
        apply_galois_inplace(ctx, ct_conj, conj_elt, gk);
        ks.conj += 1;
        add_inplace(ctx, ct, ct_conj);
    }
}

static std::vector<double> decrypt_blocks(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts,
    int m,
    int d_out
) {
    const int c = NSLOTS / m;
    std::vector<double> out((size_t)m * (size_t)d_out, 0.0);
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        const int used = std::min(c, std::max(0, d_out - b * c));
        if (used <= 0) break;
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);
        for (int col = 0; col < used; col++) {
            for (int i = 0; i < m; i++) {
                out[(size_t)i * (size_t)d_out + (size_t)(b * c + col)] = preal(dec[(size_t)col * (size_t)m + (size_t)i]);
            }
        }
    }
    return out;
}

} // namespace

int main() {
    constexpr size_t NPOLY = static_cast<size_t>(NSLOTS) * 2;
    constexpr uint64_t MMOD = 2ull * static_cast<uint64_t>(NPOLY);
    constexpr uint32_t GEN_ROT = 5u;

    const double scale_in = std::pow(2.0, 40);
    const double scale_w = std::pow(2.0, 40);
    const double scale_mul = scale_in * scale_w;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "failed to set cuda device\n";
        return 1;
    }

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(NPOLY);
    parms.set_special_modulus_size(4);
    // Chain: 15 body limbs at scale 2^40, 4 special primes. Total 900 bits at N=32768.
    // Security: lambda>=128 under HE Std v1.1 SPARSE ternary distribution (cap log Q<=1747).
    parms.set_coeff_modulus(CoeffModulus::Create(NPOLY, {60, 40,40,40,40,40,40,40,40,40,40,40,40,40,40,40, 60,60,60,60}));

    std::set<int> needed_steps;
    for (int s = 1; s < C; s++) {
        needed_steps.insert(norm_step(s * M, NSLOTS));
    }

    std::vector<uint32_t> galois_elts;
    galois_elts.reserve(3 + needed_steps.size());
    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);
    galois_elts.push_back(GEN_ROT);
    galois_elts.push_back(static_cast<uint32_t>(modinv_u64(static_cast<uint64_t>(GEN_ROT), MMOD)));
    galois_elts.push_back(conj_elt);
    for (int s : needed_steps) {
        galois_elts.push_back(galois_elt_from_step(s, GEN_ROT, MMOD));
    }
    std::sort(galois_elts.begin(), galois_elts.end());
    galois_elts.erase(std::unique(galois_elts.begin(), galois_elts.end()), galois_elts.end());
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    print_parameters(ctx);

    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    (void)pk;
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomCKKSEncoder encoder(ctx);

    const size_t slots = encoder.slot_count();
    if (static_cast<int>(slots) != NSLOTS) {
        std::cerr << "slot_count mismatch\n";
        return 1;
    }

    std::mt19937_64 rng(SEED);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<double> x((size_t)M * (size_t)D1, 0.0);
    for (size_t i = 0; i < x.size(); i++) x[i] = nd(rng);

    std::vector<double> W1((size_t)D1 * (size_t)DMID, 0.0);
    std::vector<double> W2((size_t)DMID * (size_t)D2, 0.0);
    for (size_t i = 0; i < W1.size(); i++) W1[i] = nd(rng);
    for (size_t i = 0; i < W2.size(); i++) W2[i] = nd(rng);

    std::vector<double> h_ref = matmul_ref(x, M, D1, W1, DMID);
    std::vector<double> y_ref = matmul_ref(h_ref, M, DMID, W2, D2);

    std::vector<PhantomCiphertext> x_ct = encrypt_packed_pairs(ctx, encoder, sk, x, M, D1, HP1, scale_in, C_IN_FF1);

    std::vector<pc64> zmsg(slots, pcplx(0.0, 0.0));
    PhantomPlaintext pt_zero_mul;
    encoder.encode(ctx, zmsg, scale_mul, pt_zero_mul, 1);
    PhantomCiphertext ct_zero_mul;
    sk.encrypt_symmetric(ctx, pt_zero_mul, ct_zero_mul);

    std::vector<pc64> W1_tab = build_wtab_paired(W1, D1, DMID, C, C_IN_FF1);
    std::vector<pc64> W2_tab = build_wtab_paired(W2, DMID, D2, C, C_IN_FF2);

    const size_t ce = static_cast<size_t>(conj_elt);

    KSCounters ks_ff1;
    auto t10 = std::chrono::high_resolution_clock::now();
    auto babies1 = build_babies(ctx, gk, x_ct, N1_FF1, M, ks_ff1);
    auto h_ct = linear_complex_paired(ctx, encoder, gk, babies1, W1_tab, DMID, M, N1_FF1, N2_FF1, C, scale_w, ct_zero_mul, ks_ff1);
    ct_real_blocks(ctx, gk, h_ct, ce, ks_ff1);
    cuda_sync();
    auto t11 = std::chrono::high_resolution_clock::now();

    // Decrypt FF1 output for error checking and re-encrypt as complex pairs for FF2
    std::vector<double> h_got = decrypt_blocks(ctx, encoder, sk, h_ct, M, DMID);
    auto h_ct_paired = encrypt_packed_pairs(ctx, encoder, sk, h_got, M, DMID, HP2, scale_in, C_IN_FF2);
    cuda_sync();

    KSCounters ks_ff2;
    auto t20 = std::chrono::high_resolution_clock::now();
    auto babies2 = build_babies(ctx, gk, h_ct_paired, N1_FF2, M, ks_ff2);
    auto y_ct = linear_complex_paired(ctx, encoder, gk, babies2, W2_tab, D2, M, N1_FF2, N2_FF2, C, scale_w, ct_zero_mul, ks_ff2);
    ct_real_blocks(ctx, gk, y_ct, ce, ks_ff2);
    cuda_sync();
    auto t21 = std::chrono::high_resolution_clock::now();

    const double t_ff1_ms = ms_since(t10, t11);
    const double t_ff2_ms = ms_since(t20, t21);
    const double t_total_ms = t_ff1_ms + t_ff2_ms;

    std::vector<double> y_got = decrypt_blocks(ctx, encoder, sk, y_ct, M, D2);

    const double err_ff1 = rel_err_real(h_got, h_ref);
    const double err_ff2 = rel_err_real(y_got, y_ref);

    std::cout << std::setprecision(10);
    std::cout << "stage ffn_full\n";
    std::cout << "elapsed_ms " << t_total_ms << "\n";
    std::cout << "elapsed_ms_ff1 " << t_ff1_ms << "\n";
    std::cout << "elapsed_ms_ff2 " << t_ff2_ms << "\n";
    std::cout << "rel_err_ff1 " << std::scientific << err_ff1 << std::defaultfloat << "\n";
    std::cout << "rel_err_ff2 " << std::scientific << err_ff2 << std::defaultfloat << "\n";
    std::cout << "ks_rots_ff1 " << ks_ff1.rots << "\n";
    std::cout << "ks_muls_ctct_ff1 " << ks_ff1.muls_ctct << "\n";
    std::cout << "ks_conj_ff1 " << ks_ff1.conj << "\n";
    std::cout << "ks_rots_ff2 " << ks_ff2.rots << "\n";
    std::cout << "ks_muls_ctct_ff2 " << ks_ff2.muls_ctct << "\n";
    std::cout << "ks_conj_ff2 " << ks_ff2.conj << "\n";

    return 0;
}
