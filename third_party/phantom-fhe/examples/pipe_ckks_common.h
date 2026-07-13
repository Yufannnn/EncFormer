#pragma once
// pipe_ckks_common.h — Shared CKKS helper functions for the native pipe pipeline.
// Extracted from bench_ckks_*.cu to avoid duplication.

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

#include "phantom.h"
#include "evaluate.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

using pc64 = cuDoubleComplex;

namespace pipe_ckks {

// ---------------------------------------------------------------------------
// Basic helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Modular arithmetic + Galois helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Galois element set builders
// ---------------------------------------------------------------------------

// Build the minimal set of Galois elements for column rotations (baby-step/giant-step).
// Baby steps: q*M for q=1..N1-1.  Giant steps: p*N1*M for p=1..N2-1.  Plus conjugate.
inline std::vector<uint32_t> build_galois_elts_linear(
    int nslots, int m, int n1, int n2,
    uint32_t gen_rot, uint64_t mmod)
{
    std::set<int> needed;
    // Baby steps
    for (int q = 1; q < n1; q++)
        needed.insert(norm_step(q * m, nslots));
    // Giant steps
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

// Build full column rotation set (s*M for s=1..C-1) — for benchmarks / Score / Value stages.
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

// Build within-row rotation set (s for s=1..M-1) — for Score / Value stages.
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

// Merge multiple Galois element sets.
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

// ---------------------------------------------------------------------------
// KS operation counter
// ---------------------------------------------------------------------------

struct KSCounters {
    long long rots = 0;
    long long muls_ctct = 0;
    long long conj = 0;
};

// ---------------------------------------------------------------------------
// Error measurement
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Rolling helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Weight table indexing and construction
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Reference matmul (for verification)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Encrypt / Decrypt
// ---------------------------------------------------------------------------

inline std::vector<PhantomCiphertext> encrypt_packed_pairs(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const double *x, int m, int d, int hp, int nslots, double scale,
    size_t chain_idx = 1, int c_in = -1)
{
    const int c = nslots / m;
    if (c_in < 0) c_in = c;
    const size_t slots = encoder.slot_count();
    std::vector<PhantomCiphertext> out(hp);
    for (int h = 0; h < hp; h++) {
        std::vector<pc64> msg(slots, pcplx(0.0));
        for (int col = 0; col < c_in; col++) {
            const int col_e = (2 * h) * c_in + col;
            const int col_o = (2 * h + 1) * c_in + col;
            for (int i = 0; i < m; i++) {
                double re = (col_e < d) ? x[(size_t)i * d + col_e] : 0.0;
                double im = (col_o < d) ? x[(size_t)i * d + col_o] : 0.0;
                msg[(size_t)col * m + i] = pcplx(re, im);
            }
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, chain_idx);
        sk.encrypt_symmetric(ctx, pt, out[h]);
    }
    return out;
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
    double scale_w, const PhantomCiphertext &ct_zero_mul, KSCounters &ks,
    int blocks_override = -1, size_t chain_idx = 1)
{
    const int hp = static_cast<int>(gp.size());
    const int blocks = (blocks_override > 0) ? blocks_override : (d_out / c);
    const size_t slots = encoder.slot_count();

    std::vector<pc64> wr(c, pcplx(0.0));
    std::vector<pc64> pt_msg(slots, pcplx(0.0));
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
                if (q == 0) cf[b][p] = h_acc;
                else add_inplace(ctx, cf[b][p], h_acc);
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

// ---------------------------------------------------------------------------
// Modulus-Trimming Optimisation (MTO) — paper §IV.B.
//
// At a CKKS-to-MPC boundary, the ciphertext only needs enough modulus to
// satisfy the conversion-safety threshold log2(q) >= ell + sigma + 1.
// Default: ell=43 (MPC ring Z_{2^43}), sigma=40 (CKKS scale exponent),
// threshold = 84 bits. Trimming the chain to ~2 ordinary limbs (60+40 = 100
// bits, or 40+40 = 80 -> needs 3 limbs) before decrypt reduces P1's CKKS
// inverse-NTT + base-extension cost roughly linearly in remaining limbs.
//
// Behaviour: in-place mod-switch on a *copy* of the input vector so that
// any callers holding the original ciphertext are not affected. No-op if
// the ciphertexts are already at or below `target_chain_index` or if the
// MTO env disable flag is set.
//
// Activated by env var ENCFORMER_MTO=1 (default off until correctness
// is verified on the target GPU build). Target level can be overridden with
// ENCFORMER_MTO_TARGET_CI; default 2 (= keep 2 data limbs).
// ---------------------------------------------------------------------------

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
        PhantomCiphertext c = ct;  // deep copy preserves caller's state
        // Phantom's chain_index() typically counts remaining moduli; rescaling
        // moves the index in the opposite direction depending on the build.
        // We trim only when the current index is strictly above the target.
        // The exact API name (mod_switch_to_next_inplace vs
        // mod_switch_to_inplace) may vary across Phantom revisions; adjust if
        // the build fails.
        while (c.chain_index() < target_chain_index) {
            mod_switch_to_next_inplace(ctx, c);
        }
        out.push_back(std::move(c));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Expanded-GELU CKKS-side pre-evaluation (paper §IV.B Eq. 4).
//
// Given input ciphertexts x (post-FF1, pre-GELU boundary), compute:
//   F0(x) = a*x^4 - b*x^3 + c*x^2 + (0.5-d)*x + e
//   F1(x) = a*x^4 + b*x^3 + c*x^2 + (0.5+d)*x + e
// where (a, b, c, d, e) are the BOLT Algorithm 4 coefficients.
//
// Cost: ~3 ct-ct mults (x^2, x^3, x^4) + ~5 plaintext mults + adds per CT.
// Depth budget: 2 multiplicative levels for the powers.
//
// The result is shipped to MPC alongside x; the MPC side then runs only
// the Step C (interval bits) + Step D (mux) of the algorithm via
// `gelu_preeval` instead of the full 4-round secure GELU.
//
// Activated by env var ENCFORMER_EXPANDED_GELU=1 (default off).
// NOTE: structural scaffold; verified Python equivalent exists at
// src/engines/mpc_gelu_secure.py:precompute_f0_f1_fixedpoint.
// CUDA chain-level/scale plumbing TBD on the GPU build.
// ---------------------------------------------------------------------------

inline bool expanded_gelu_enabled() {
    const char *e = std::getenv("ENCFORMER_EXPANDED_GELU");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}

// BOLT Algorithm 4 coefficients (kept in sync with mpc_gelu_secure.py:10-14).
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

// Pre-evaluate BOLT-Algorithm-4 GELU polynomial candidates F_0, F_1 in CKKS
// so the MPC side only needs cmp + mux (1 round vs the 4-round full Step A+B+C+D).
//
// F_0(x) = A x^4 - B x^3 + C x^2 + (0.5 - D) x + E
// F_1(x) = A x^4 + B x^3 + C x^2 + (0.5 + D) x + E
//
// Chain-level management: caller passes x at level L. Powers are computed via
// multiply_and_relin + rescale_to_next; lower-degree powers are mod-switched
// down so all six terms align before accumulation. Mirrors the Python in
// src/engines/mpc_gelu_secure.py:secure_gelu_algorithm4_split Step B exactly.
inline PreEvalF0F1 compute_f0_f1_in_ckks(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    PhantomRelinKey &rlk,
    const std::vector<PhantomCiphertext> &x_blocks,
    double scale_in,
    int /*nslots*/)
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

    // Helper: rescale and snap the scale to scale_in exactly (post-rescale
    // scales drift by q_dropped/2^40 which trips Phantom's strict scale check).
    auto rescale_and_snap = [&](PhantomCiphertext &ct) {
        rescale_to_next_inplace(ctx, ct);
        ct.set_scale(scale_in);
    };

    for (const auto &x_in : x_blocks) {
        // ---- Step A: powers x^2, x^3, x^4 ----------------------------------
        PhantomCiphertext x = x_in;                         // L, scale s

        PhantomCiphertext x2 = multiply_and_relin(ctx, x, x, rlk);
        rescale_and_snap(x2);                               // L-1, scale = s

        PhantomCiphertext x_at_l1 = x;
        mod_switch_to_next_inplace(ctx, x_at_l1);           // L-1

        PhantomCiphertext x3 = multiply_and_relin(ctx, x2, x_at_l1, rlk);
        rescale_and_snap(x3);                               // L-2, scale = s

        PhantomCiphertext x4 = multiply_and_relin(ctx, x2, x2, rlk);
        rescale_and_snap(x4);                               // L-2, scale = s

        // ---- Step B: assemble F0, F1 via plaintext multiply + accumulate --
        const size_t chain_x4 = x4.chain_index();           // L-2
        const size_t chain_x3 = x3.chain_index();           // L-2
        const size_t chain_x2 = x2.chain_index();           // L-1
        const size_t chain_x  = x.chain_index();            // L

        // A * x^4 (rescaled to L-3)
        PhantomCiphertext A_x4 = x4;
        {
            PhantomPlaintext pt_A = encode_const(bolt_gelu::A, scale_in, chain_x4);
            multiply_plain_inplace(ctx, A_x4, pt_A);
            rescale_and_snap(A_x4);                         // L-3, scale = s
        }

        // B * x^3 (signed: bolt_gelu::B is negative)
        PhantomCiphertext B_x3 = x3;
        {
            PhantomPlaintext pt_B = encode_const(bolt_gelu::B, scale_in, chain_x3);
            multiply_plain_inplace(ctx, B_x3, pt_B);
            rescale_and_snap(B_x3);                         // L-3, scale = s
        }

        // C * x^2 (mod-switched down to L-3 to match A_x4 / B_x3)
        PhantomCiphertext C_x2 = x2;
        {
            PhantomPlaintext pt_C = encode_const(bolt_gelu::C_COEF, scale_in, chain_x2);
            multiply_plain_inplace(ctx, C_x2, pt_C);
            rescale_and_snap(C_x2);                         // L-2
            mod_switch_to_next_inplace(ctx, C_x2);          // L-3
        }

        // (0.5 - D) * x and (0.5 + D) * x (two variants; mod-switch down to L-3)
        PhantomCiphertext m_x = x;  // for F0
        {
            PhantomPlaintext pt_m = encode_const(0.5 - bolt_gelu::D, scale_in, chain_x);
            multiply_plain_inplace(ctx, m_x, pt_m);
            rescale_and_snap(m_x);                          // L-1
            mod_switch_to_next_inplace(ctx, m_x);           // L-2
            mod_switch_to_next_inplace(ctx, m_x);           // L-3
        }
        PhantomCiphertext p_x = x;  // for F1
        {
            PhantomPlaintext pt_p = encode_const(0.5 + bolt_gelu::D, scale_in, chain_x);
            multiply_plain_inplace(ctx, p_x, pt_p);
            rescale_and_snap(p_x);
            mod_switch_to_next_inplace(ctx, p_x);
            mod_switch_to_next_inplace(ctx, p_x);           // L-3
        }

        // ---- Assemble F0 = A_x4 - B_x3 + C_x2 + (0.5-D)*x + E -------------
        PhantomCiphertext F0 = A_x4;          // deep-copy ctor (see existing usage)
        sub_inplace(ctx, F0, B_x3);
        add_inplace(ctx, F0, C_x2);
        add_inplace(ctx, F0, m_x);
        {
            PhantomPlaintext pt_E = encode_const(bolt_gelu::E, F0.scale(), F0.chain_index());
            add_plain_inplace(ctx, F0, pt_E);
        }

        // ---- Assemble F1 = A_x4 + B_x3 + C_x2 + (0.5+D)*x + E -------------
        PhantomCiphertext F1 = A_x4;          // A_x4 is still intact after F0 build
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

// Decrypt blocks to row-major matrix (real parts only — after ct_real_blocks).
inline std::vector<double> decrypt_blocks(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts, int m, int d_out, int nslots)
{
    const int c = nslots / m;
    std::vector<double> out((size_t)m * d_out, 0.0);
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        const int used = std::min(c, std::max(0, d_out - b * c));
        if (used <= 0) break;
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);
        for (int col = 0; col < used; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + (b * c + col)] = preal(dec[(size_t)col * m + i]);
    }
    return out;
}

// Decrypt complex-packed blocks (no ct_real — even cols from real, odd cols from imag).
inline std::vector<double> decrypt_blocks_complex_packed(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts, int m, int d_out, int nslots)
{
    const int c = nslots / m;
    std::vector<double> out((size_t)m * d_out, 0.0);
    int col_offset = 0;
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);

        // Real part → even block columns
        int used_re = std::min(c, std::max(0, d_out - col_offset));
        for (int col = 0; col < used_re; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + col_offset + col] = preal(dec[(size_t)col * m + i]);
        col_offset += used_re;

        // Imag part → odd block columns
        int used_im = std::min(c, std::max(0, d_out - col_offset));
        for (int col = 0; col < used_im; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + col_offset + col] = pimag(dec[(size_t)col * m + i]);
        col_offset += used_im;
    }
    return out;
}

// Encode a constant complex value into all slots.
inline void encode_const_vec(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder,
    pc64 c_val, double scale, PhantomPlaintext &pt,
    size_t chain_index = 1)
{
    const size_t slots = encoder.slot_count();
    std::vector<pc64> v(slots, c_val);
    encoder.encode(ctx, v, scale, pt, chain_index);
}

// Create an encryption of zero at multiply scale (for accumulator initialization).
inline PhantomCiphertext make_ct_zero_mul(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    double scale_mul, size_t chain_idx = 1)
{
    const size_t slots = encoder.slot_count();
    std::vector<pc64> zmsg(slots, pcplx(0.0));
    PhantomPlaintext pt;
    encoder.encode(ctx, zmsg, scale_mul, pt, chain_idx);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(ctx, pt, ct);
    return ct;
}

// Encrypt M×D matrix as G=D/C real-valued block ciphertexts.
// slot[c*M + i] = X[i, g*C + c]  (real only, imag = 0).
inline std::vector<PhantomCiphertext> encrypt_real_blocks(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const double *x, int m, int d, int nslots, double scale,
    size_t chain_idx = 1)
{
    const int c = nslots / m;
    const int g_count = (d + c - 1) / c;
    const size_t slots = encoder.slot_count();
    std::vector<PhantomCiphertext> out(g_count);
    for (int g = 0; g < g_count; g++) {
        std::vector<pc64> msg(slots, pcplx(0.0));
        for (int col = 0; col < c; col++) {
            const int abs_col = g * c + col;
            for (int i = 0; i < m; i++)
                msg[(size_t)col * m + i] =
                    pcplx((abs_col < d) ? x[(size_t)i * d + abs_col] : 0.0, 0.0);
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, chain_idx);
        sk.encrypt_symmetric(ctx, pt, out[g]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// FDP column permutation: perm[u*H + h] = h*Dh + u  (interleaves heads)
// ---------------------------------------------------------------------------

inline std::vector<int> perm_fdp(int H, int Dh) {
    int D = H * Dh;
    std::vector<int> perm(D);
    for (int u = 0; u < Dh; u++)
        for (int h = 0; h < H; h++)
            perm[u * H + h] = h * Dh + u;
    return perm;
}

// Apply column permutation to weight matrix W[d_in × d_out] → W_perm[d_in × d_out]
inline std::vector<double> permute_cols(const double *w, int d_in, int d_out,
                                         const std::vector<int> &perm) {
    std::vector<double> out((size_t)d_in * d_out);
    for (int i = 0; i < d_in; i++)
        for (int j = 0; j < d_out; j++)
            out[(size_t)i * d_out + j] = w[(size_t)i * d_out + perm[j]];
    return out;
}

// ---------------------------------------------------------------------------
// Add plaintext bias to encrypted blocks (for QKV bias in FHE domain)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Decrypt blocks with c_used stride (for FDP-packed blocks)
// ---------------------------------------------------------------------------

inline std::vector<double> decrypt_blocks_cusued(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts, int m, int d_out, int nslots, int c_used)
{
    const int c = nslots / m;
    std::vector<double> out((size_t)m * d_out, 0.0);
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        const int used = std::min(c_used, std::max(0, d_out - b * c_used));
        if (used <= 0) break;
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);
        for (int col = 0; col < used; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + (b * c_used + col)] = preal(dec[(size_t)col * m + i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Galois element builders for Score (FDP)
// ---------------------------------------------------------------------------

inline std::vector<uint32_t> build_galois_elts_score(
    int nslots, int m, int b_fold, int g, int c_used, int H,
    uint32_t gen_rot, uint64_t mmod)
{
    std::set<int> needed;

    // Within-row rotations for mk_bank (Q_bank: s=1..b-1, K_bank: j*b, K_bank_h: ((j+g/2)%g)*b)
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

    // Column rotations for red_h
    int U = (c_used + H - 1) / H;
    int delta = 1;
    while (delta < U) {
        int src_seg_lo = delta * H;
        if (src_seg_lo >= c_used) break;
        needed.insert(norm_step(delta * H * m, nslots));
        delta <<= 1;
    }

    // Column rotations for pack_f
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

} // namespace pipe_ckks
