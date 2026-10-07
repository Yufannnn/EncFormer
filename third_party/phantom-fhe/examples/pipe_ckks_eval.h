#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "phantom.h"
#include "evaluate.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

using pc64 = cuDoubleComplex;

namespace pipe_ckks {

inline pc64 pcplx(double r, double i = 0.0) { return make_cuDoubleComplex(r, i); }
inline double preal(const pc64 &z) { return cuCreal(z); }
inline double pimag(const pc64 &z) { return cuCimag(z); }

inline void cuda_sync() {
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

inline double ms_since(std::chrono::high_resolution_clock::time_point t0,
                       std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count() * 1e3;
}

inline int ckks_body_limbs_from_env(int fallback) {
    const char *e = std::getenv("ENCFORMER_CKKS_BODY_LIMBS");
    if (!e || !*e) return fallback;
    char *end = nullptr;
    long v = std::strtol(e, &end, 10);
    if (end == e || v < 1 || v > 32) return fallback;
    return static_cast<int>(v);
}

inline std::vector<int> ckks_modulus_bits(int body_limbs) {
    std::vector<int> bits;
    bits.reserve(static_cast<size_t>(body_limbs) + 2);
    bits.push_back(60);
    for (int i = 0; i < body_limbs; ++i) bits.push_back(40);
    bits.push_back(60);
    return bits;
}

inline void set_lite_ckks_pipeline_params(EncryptionParameters &parms, size_t npoly) {
    const int body_limbs = ckks_body_limbs_from_env(7);
    parms.set_poly_modulus_degree(npoly);
    parms.set_special_modulus_size(1);
    parms.set_coeff_modulus(CoeffModulus::Create(npoly, ckks_modulus_bits(body_limbs)));
}

inline size_t data_primes_at(const PhantomContext &ctx, size_t ci) {
    return ctx.get_context_data(ci).parms().coeff_modulus().size();
}

inline double dropped_prime_at(const PhantomContext &ctx, size_t ci) {
    const auto &q = ctx.get_context_data(ci).parms().coeff_modulus();
    return static_cast<double>(q.back().value());
}

inline void require_chain(const PhantomCiphertext &ct, size_t ci, const char *where) {
    if (ct.chain_index() != ci)
        throw std::runtime_error(std::string(where) + ": expected chain index " + std::to_string(ci) +
                                 ", got " + std::to_string(ct.chain_index()));
}

inline void require_scale(const PhantomCiphertext &ct, double scale, const char *where) {
    if (std::fabs(ct.scale() / scale - 1.0) > 1e-9)
        throw std::runtime_error(std::string(where) + ": scale " + std::to_string(std::log2(ct.scale())) +
                                 " bits, expected " + std::to_string(std::log2(scale)));
}

inline uint64_t modinv_u64(uint64_t a, uint64_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(m), newr = static_cast<int64_t>(a % m);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp_t = t - q * newt;
        t = newt; newt = tmp_t;
        int64_t tmp_r = r - q * newr;
        r = newr; newr = tmp_r;
    }
    if (r != 1) throw std::runtime_error("modinv failed");
    if (t < 0) t += static_cast<int64_t>(m);
    return static_cast<uint64_t>(t);
}

inline uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m, x = a % m;
    while (e) {
        if (e & 1) r = static_cast<uint64_t>((__uint128_t)r * x % m);
        x = static_cast<uint64_t>((__uint128_t)x * x % m);
        e >>= 1;
    }
    return r;
}

inline int norm_step(int step, int slots) {
    step %= slots;
    if (step > slots / 2) step -= slots;
    if (step < -slots / 2) step += slots;
    return step;
}

inline uint32_t galois_elt_from_step(int step, uint32_t gen, uint64_t m) {
    if (step == 0) return 1u;
    bool neg = (step < 0);
    uint64_t e = static_cast<uint64_t>(neg ? -(int64_t)step : (int64_t)step);
    uint64_t g = powmod_u64(static_cast<uint64_t>(gen), e, m);
    if (neg) g = modinv_u64(g, m);
    return static_cast<uint32_t>(g);
}

inline std::vector<uint32_t> build_galois_elts_linear(
    int nslots, int m, int n1, int n2,
    uint32_t gen_rot, uint64_t mmod)
{
    std::set<int> needed;

    for (int q = 1; q < n1; q++)
        needed.insert(norm_step(q * m, nslots));

    for (int p = 1; p < n2; p++)
        needed.insert(norm_step(p * n1 * m, nslots));

    const uint32_t conj_elt = static_cast<uint32_t>(mmod - 1);
    std::vector<uint32_t> elts;
    elts.reserve(3 + needed.size());
    elts.push_back(gen_rot);
    elts.push_back(static_cast<uint32_t>(modinv_u64(static_cast<uint64_t>(gen_rot), mmod)));
    elts.push_back(conj_elt);
    for (int s : needed)
        elts.push_back(galois_elt_from_step(s, gen_rot, mmod));
    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

inline std::vector<uint32_t> build_galois_elts_full_columns(
    int nslots, int m, uint32_t gen_rot, uint64_t mmod)
{
    const int c = nslots / m;
    std::set<int> needed;
    for (int s = 1; s < c; s++)
        needed.insert(norm_step(s * m, nslots));
    const uint32_t conj_elt = static_cast<uint32_t>(mmod - 1);
    std::vector<uint32_t> elts;
    elts.reserve(3 + needed.size());
    elts.push_back(gen_rot);
    elts.push_back(static_cast<uint32_t>(modinv_u64(static_cast<uint64_t>(gen_rot), mmod)));
    elts.push_back(conj_elt);
    for (int s : needed)
        elts.push_back(galois_elt_from_step(s, gen_rot, mmod));
    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

inline std::vector<uint32_t> build_galois_elts_within_row(
    int nslots, int m, uint32_t gen_rot, uint64_t mmod)
{
    std::set<int> needed;
    for (int s = 1; s < m; s++) {
        needed.insert(norm_step(s, nslots));
        needed.insert(norm_step(-s, nslots));
    }
    const uint32_t conj_elt = static_cast<uint32_t>(mmod - 1);
    std::vector<uint32_t> elts;
    elts.reserve(3 + needed.size());
    elts.push_back(gen_rot);
    elts.push_back(static_cast<uint32_t>(modinv_u64(static_cast<uint64_t>(gen_rot), mmod)));
    elts.push_back(conj_elt);
    for (int s : needed)
        elts.push_back(galois_elt_from_step(s, gen_rot, mmod));
    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

inline std::vector<uint32_t> merge_galois_elts(
    const std::vector<uint32_t> &a, const std::vector<uint32_t> &b)
{
    std::vector<uint32_t> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct KSCounters {
    long long rots = 0;
    long long muls_ctct = 0;
    long long conj = 0;
};

inline double rel_err_real(const std::vector<double> &a, const std::vector<double> &b) {
    long double num = 0.0L, den = 0.0L;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        num += d * d;
        long double v = static_cast<long double>(b[i]);
        den += v * v;
    }
    return static_cast<double>(std::sqrt(static_cast<double>(num)) /
                               (std::sqrt(static_cast<double>(den)) + 1e-18));
}

inline double mse_real(const std::vector<double> &a, const std::vector<double> &b) {
    long double num = 0.0L;
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    for (size_t i = 0; i < n; i++) {
        long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        num += d * d;
    }
    return static_cast<double>(num / static_cast<long double>(n));
}

inline double mse_complex_slots(const std::vector<pc64> &a, const std::vector<pc64> &b) {
    long double num = 0.0L;
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    for (size_t i = 0; i < n; i++) {
        long double dr = static_cast<long double>(preal(a[i])) - static_cast<long double>(preal(b[i]));
        long double di = static_cast<long double>(pimag(a[i])) - static_cast<long double>(pimag(b[i]));
        num += dr * dr + di * di;
    }
    return static_cast<double>(num / static_cast<long double>(n));
}

inline void roll_right(const double *src, int n, int shift, std::vector<double> &dst) {
    const int s = ((shift % n) + n) % n;
    if (s == 0) { for (int i = 0; i < n; i++) dst[i] = src[i]; return; }
    for (int i = 0; i < n; i++) dst[(i + s) % n] = src[i];
}

inline void roll_right_c(const pc64 *src, int n, int shift, std::vector<pc64> &dst) {
    const int s = ((shift % n) + n) % n;
    if (s == 0) { for (int i = 0; i < n; i++) dst[i] = src[i]; return; }
    for (int i = 0; i < n; i++) dst[(i + s) % n] = src[i];
}

inline size_t tab_idx(int ge, int t, int b, int j, int c, int blocks) {
    return ((((size_t)ge * (size_t)c + (size_t)t) * (size_t)blocks + (size_t)b) * (size_t)c + (size_t)j);
}

inline std::vector<pc64> build_wtab_paired(const double *w, int d_in, int d_out, int c,
                                            int c_used = -1, int blocks_override = -1,
                                            int c_in = -1) {
    if (c_used < 0) c_used = c;
    if (c_in < 0) c_in = c;
    const int g = d_in / c_in;
    const int hp = g / 2;
    const int blocks = (blocks_override > 0) ? blocks_override : (d_out / c);
    std::vector<pc64> tab((size_t)hp * c * blocks * c, pcplx(0.0));
    for (int h = 0; h < hp; h++) {
        const int off_e = (2 * h) * c_in;
        const int off_o = (2 * h + 1) * c_in;
        for (int t = 0; t < c; t++)
            for (int b = 0; b < blocks; b++)
                for (int j = 0; j < c; j++) {
                    if (j >= c_used) continue;
                    const int col = b * c_used + j;
                    if (col >= d_out) continue;
                    const int input_seg = ((j + t) % c);
                    if (input_seg >= c_in) continue;
                    const int row_e = off_e + input_seg;
                    const int row_o = off_o + input_seg;
                    double we = w[(size_t)row_e * d_out + col];
                    double wo = w[(size_t)row_o * d_out + col];
                    tab[tab_idx(h, t, b, j, c, blocks)] = pcplx(0.5 * we, -0.5 * wo);
                }
    }
    return tab;
}

inline std::vector<double> matmul_ref(const double *a, int m, int k, const double *w, int n) {
    std::vector<double> y((size_t)m * n, 0.0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int t = 0; t < k; t++)
                acc += a[(size_t)i * k + t] * w[(size_t)t * n + j];
            y[(size_t)i * n + j] = acc;
        }
    return y;
}

inline std::vector<std::vector<PhantomCiphertext>> build_babies(
    const PhantomContext &ctx, const PhantomGaloisKey &gk,
    const std::vector<PhantomCiphertext> &base,
    int n1, int m, int nslots, KSCounters &ks)
{
    const int g = static_cast<int>(base.size());
    std::vector<std::vector<PhantomCiphertext>> gp(g, std::vector<PhantomCiphertext>(n1));
    for (int ge = 0; ge < g; ge++) {
        gp[ge][0] = base[ge];
        for (int q = 1; q < n1; q++) {
            PhantomCiphertext ct = base[ge];
            rotate_inplace(ctx, ct, norm_step(q * m, nslots), gk);
            gp[ge][q] = ct;
            ks.rots += 1;
        }
    }
    return gp;
}

inline std::vector<PhantomCiphertext> linear_complex_paired(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, const PhantomGaloisKey &gk,
    const std::vector<std::vector<PhantomCiphertext>> &gp,
    const std::vector<pc64> &wtab,
    int d_out, int m, int n1, int n2, int c, int nslots,
    double scale_w, KSCounters &ks,
    int blocks_override = -1, size_t chain_idx = 1)
{
    const int hp = static_cast<int>(gp.size());
    const int blocks = (blocks_override > 0) ? blocks_override : (d_out / c);
    const size_t slots = encoder.slot_count();

    std::vector<pc64> wr(c, pcplx(0.0));
    std::vector<pc64> pt_msg(slots, pcplx(0.0));
    std::vector<std::vector<PhantomCiphertext>> cf(blocks, std::vector<PhantomCiphertext>(n2));
    std::vector<std::vector<char>> cf_set(blocks, std::vector<char>(n2, 0));

    for (int b = 0; b < blocks; b++) {
        for (int p = 0; p < n2; p++) {
            const int p_shift = (p * n1) % c;
            for (int q = 0; q < n1; q++) {
                const int t = (p_shift + q) % c;
                bool has_h = false;
                PhantomCiphertext h_acc;
                for (int h = 0; h < hp; h++) {
                    const pc64 *src = &wtab[tab_idx(h, t, b, 0, c, blocks)];
                    if (p_shift == 0)
                        for (int j = 0; j < c; j++) wr[j] = src[j];
                    else
                        roll_right_c(src, c, p_shift, wr);
                    for (int seg = 0; seg < c; seg++) {
                        const pc64 vv = wr[seg];
                        const int base_idx = seg * m;
                        for (int i = 0; i < m; i++)
                            pt_msg[(size_t)base_idx + i] = vv;
                    }
                    PhantomPlaintext pt;
                    encoder.encode(ctx, pt_msg, scale_w, pt, chain_idx);
                    PhantomCiphertext term = gp[h][q];
                    multiply_plain_inplace(ctx, term, pt);
                    if (!has_h) { h_acc = term; has_h = true; }
                    else add_inplace(ctx, h_acc, term);
                }
                if (!has_h) continue;
                if (!cf_set[b][p]) { cf[b][p] = std::move(h_acc); cf_set[b][p] = 1; }
                else add_inplace(ctx, cf[b][p], h_acc);
            }
        }
    }

    std::vector<PhantomCiphertext> out(blocks);
    for (int b = 0; b < blocks; b++) {
        bool inited = false;
        PhantomCiphertext acc;
        for (int p = 0; p < n2; p++) {
            if (!cf_set[b][p]) continue;
            PhantomCiphertext term = cf[b][p];
            if (p != 0) {
                rotate_inplace(ctx, term, norm_step(p * n1 * m, nslots), gk);
                ks.rots += 1;
            }
            if (!inited) { acc = term; inited = true; }
            else add_inplace(ctx, acc, term);
        }
        if (inited) out[b] = acc;
    }
    return out;
}

inline std::vector<PhantomCiphertext> linear_complex_paired(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, const PhantomGaloisKey &gk,
    const std::vector<std::vector<PhantomCiphertext>> &gp,
    const std::vector<pc64> &wtab,
    int d_out, int m, int n1, int n2, int c, int nslots,
    double scale_w, const PhantomCiphertext &  , KSCounters &ks,
    int blocks_override = -1, size_t chain_idx = 1)
{
    return linear_complex_paired(ctx, encoder, gk, gp, wtab, d_out, m, n1, n2, c, nslots, scale_w, ks,
                                 blocks_override, chain_idx);
}

inline void ct_real_blocks(
    const PhantomContext &ctx, const PhantomGaloisKey &gk,
    std::vector<PhantomCiphertext> &cts, size_t conj_elt, KSCounters &ks)
{
    for (auto &ct : cts) {
        PhantomCiphertext ct_conj = ct;
        apply_galois_inplace(ctx, ct_conj, conj_elt, gk);
        ks.conj += 1;
        add_inplace(ctx, ct, ct_conj);
    }
}

inline bool mto_enabled() {
    const char *e = std::getenv("ENCFORMER_MTO");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}

inline size_t mto_target_chain_index() {
    const char *e = std::getenv("ENCFORMER_MTO_TARGET_CI");
    if (!e || !*e) return 2;
    long v = std::strtol(e, nullptr, 10);
    if (v < 1) v = 1;
    return static_cast<size_t>(v);
}

inline std::vector<PhantomCiphertext> trim_for_c2m(
    const PhantomContext &ctx,
    const std::vector<PhantomCiphertext> &in,
    size_t target_chain_index = 0)
{
    if (!mto_enabled()) return in;
    if (target_chain_index == 0) target_chain_index = mto_target_chain_index();

    std::vector<PhantomCiphertext> out;
    out.reserve(in.size());
    for (const auto &ct : in) {
        PhantomCiphertext c = ct;

        while (c.chain_index() < target_chain_index) {
            mod_switch_to_next_inplace(ctx, c);
        }
        out.push_back(std::move(c));
    }
    return out;
}

inline bool expanded_gelu_enabled() {
    const char *e = std::getenv("ENCFORMER_EXPANDED_GELU");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}

namespace bolt_gelu {
    constexpr double A = 0.020848611754127593;
    constexpr double B = -0.18352506127082727;
    constexpr double C_COEF = 0.5410550166368381;
    constexpr double D = -0.03798164612714154;
    constexpr double E = 0.001620808531841547;
}

struct PreEvalF0F1 {
    std::vector<PhantomCiphertext> f0;
    std::vector<PhantomCiphertext> f1;
};

inline PreEvalF0F1 compute_f0_f1_in_ckks(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    PhantomRelinKey &rlk,
    const std::vector<PhantomCiphertext> &x_blocks,
    double scale_in,
    int  )
{
    PreEvalF0F1 out;
    out.f0.reserve(x_blocks.size());
    out.f1.reserve(x_blocks.size());

    const size_t slots = encoder.slot_count();
    auto encode_const = [&](double v, double scale, size_t chain_idx) {
        std::vector<pc64> msg(slots, pcplx(v));
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, chain_idx);
        return pt;
    };

    auto rescale_and_snap = [&](PhantomCiphertext &ct) {
        rescale_to_next_inplace(ctx, ct);
        ct.set_scale(scale_in);
    };

    for (const auto &x_in : x_blocks) {

        PhantomCiphertext x = x_in;

        PhantomCiphertext x2 = multiply_and_relin(ctx, x, x, rlk);
        rescale_and_snap(x2);

        PhantomCiphertext x_at_l1 = x;
        mod_switch_to_next_inplace(ctx, x_at_l1);

        PhantomCiphertext x3 = multiply_and_relin(ctx, x2, x_at_l1, rlk);
        rescale_and_snap(x3);

        PhantomCiphertext x4 = multiply_and_relin(ctx, x2, x2, rlk);
        rescale_and_snap(x4);

        const size_t chain_x4 = x4.chain_index();
        const size_t chain_x3 = x3.chain_index();
        const size_t chain_x2 = x2.chain_index();
        const size_t chain_x  = x.chain_index();

        PhantomCiphertext A_x4 = x4;
        {
            PhantomPlaintext pt_A = encode_const(bolt_gelu::A, scale_in, chain_x4);
            multiply_plain_inplace(ctx, A_x4, pt_A);
            rescale_and_snap(A_x4);
        }

        PhantomCiphertext B_x3 = x3;
        {
            PhantomPlaintext pt_B = encode_const(bolt_gelu::B, scale_in, chain_x3);
            multiply_plain_inplace(ctx, B_x3, pt_B);
            rescale_and_snap(B_x3);
        }

        PhantomCiphertext C_x2 = x2;
        {
            PhantomPlaintext pt_C = encode_const(bolt_gelu::C_COEF, scale_in, chain_x2);
            multiply_plain_inplace(ctx, C_x2, pt_C);
            rescale_and_snap(C_x2);
            mod_switch_to_next_inplace(ctx, C_x2);
        }

        PhantomCiphertext m_x = x;
        {
            PhantomPlaintext pt_m = encode_const(0.5 - bolt_gelu::D, scale_in, chain_x);
            multiply_plain_inplace(ctx, m_x, pt_m);
            rescale_and_snap(m_x);
            mod_switch_to_next_inplace(ctx, m_x);
            mod_switch_to_next_inplace(ctx, m_x);
        }
        PhantomCiphertext p_x = x;
        {
            PhantomPlaintext pt_p = encode_const(0.5 + bolt_gelu::D, scale_in, chain_x);
            multiply_plain_inplace(ctx, p_x, pt_p);
            rescale_and_snap(p_x);
            mod_switch_to_next_inplace(ctx, p_x);
            mod_switch_to_next_inplace(ctx, p_x);
        }

        PhantomCiphertext F0 = A_x4;
        sub_inplace(ctx, F0, B_x3);
        add_inplace(ctx, F0, C_x2);
        add_inplace(ctx, F0, m_x);
        {
            PhantomPlaintext pt_E = encode_const(bolt_gelu::E, F0.scale(), F0.chain_index());
            add_plain_inplace(ctx, F0, pt_E);
        }

        PhantomCiphertext F1 = A_x4;
        add_inplace(ctx, F1, B_x3);
        add_inplace(ctx, F1, C_x2);
        add_inplace(ctx, F1, p_x);
        {
            PhantomPlaintext pt_E = encode_const(bolt_gelu::E, F1.scale(), F1.chain_index());
            add_plain_inplace(ctx, F1, pt_E);
        }

        out.f0.push_back(std::move(F0));
        out.f1.push_back(std::move(F1));
    }
    return out;
}

inline void encode_const_vec(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
    pc64 c_val, double scale, PhantomPlaintext &pt,
    size_t chain_index = 1)
{
    const size_t slots = encoder.slot_count();
    std::vector<pc64> v(slots, c_val);
    encoder.encode(ctx, v, scale, pt, chain_index);
}

inline std::vector<int> perm_fdp(int H, int Dh) {
    int D = H * Dh;
    std::vector<int> perm(D);
    for (int u = 0; u < Dh; u++)
        for (int h = 0; h < H; h++)
            perm[u * H + h] = h * Dh + u;
    return perm;
}

inline std::vector<double> permute_cols(const double *w, int d_in, int d_out,
                                         const std::vector<int> &perm) {
    std::vector<double> out((size_t)d_in * d_out);
    for (int i = 0; i < d_in; i++)
        for (int j = 0; j < d_out; j++)
            out[(size_t)i * d_out + j] = w[(size_t)i * d_out + perm[j]];
    return out;
}

inline void add_bias_blocks(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
    std::vector<PhantomCiphertext> &cts,
    const double *bias, int d, int m, int nslots, int c_used,
    double scale_mul, size_t chain_idx = 1)
{
    const size_t slots = encoder.slot_count();
    const int blocks = static_cast<int>(cts.size());
    for (int bid = 0; bid < blocks; bid++) {
        std::vector<pc64> msg(slots, pcplx(0.0));
        for (int seg = 0; seg < c_used; seg++) {
            int col = bid * c_used + seg;
            double bval = (col < d) ? bias[col] : 0.0;
            for (int i = 0; i < m; i++)
                msg[(size_t)seg * m + i] = pcplx(bval, 0.0);
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale_mul, pt, chain_idx);
        add_plain_inplace(ctx, cts[bid], pt);
    }
}

inline std::vector<uint32_t> build_galois_elts_score(
    int nslots, int m, int b_fold, int g, int c_used, int H,
    uint32_t gen_rot, uint64_t mmod)
{
    std::set<int> needed;

    for (int s = 1; s < b_fold; s++) {
        needed.insert(norm_step(s, nslots));
        needed.insert(norm_step(s - m, nslots));
    }
    for (int j = 1; j < g; j++) {
        int rot = j * b_fold;
        needed.insert(norm_step(rot, nslots));
        needed.insert(norm_step(rot - m, nslots));
    }
    for (int j = 0; j < g; j++) {
        int rot = ((j + g / 2) % g) * b_fold;
        if (rot > 0) {
            needed.insert(norm_step(rot, nslots));
            needed.insert(norm_step(rot - m, nslots));
        }
    }

    int U = (c_used + H - 1) / H;
    int delta = 1;
    while (delta < U) {
        int src_seg_lo = delta * H;
        if (src_seg_lo >= c_used) break;
        needed.insert(norm_step(delta * H * m, nslots));
        delta <<= 1;
    }

    int blen = H * m;
    int half = m / 2;
    for (int t = 0; t < half; t++) {
        int L = t * blen;
        int off = L % nslots;
        if (off > 0) {
            int rot = (nslots - off) % nslots;
            needed.insert(norm_step(rot, nslots));
        }
    }

    const uint32_t conj_elt = static_cast<uint32_t>(mmod - 1);
    std::vector<uint32_t> elts;
    elts.reserve(3 + needed.size());
    elts.push_back(gen_rot);
    elts.push_back(static_cast<uint32_t>(modinv_u64(static_cast<uint64_t>(gen_rot), mmod)));
    elts.push_back(conj_elt);
    for (int s : needed)
        elts.push_back(galois_elt_from_step(s, gen_rot, mmod));
    std::sort(elts.begin(), elts.end());
    elts.erase(std::unique(elts.begin(), elts.end()), elts.end());
    return elts;
}

}
