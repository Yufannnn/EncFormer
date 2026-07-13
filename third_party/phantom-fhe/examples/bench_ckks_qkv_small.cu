#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "example.h"     // print_parameters(...)
#include "phantom.h"
#include "util.cuh"
#include "evaluate.cuh"  // rotate_inplace, apply_galois_inplace, add_inplace, multiply_plain_inplace, etc.

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

// Phantom encoder supports cuDoubleComplex vectors in this repo
using pc64 = cuDoubleComplex;
static inline pc64 pcplx(double r, double i = 0.0) { return make_cuDoubleComplex(r, i); }
static inline double preal(const pc64 &z) { return cuCreal(z); }
static inline double pimag(const pc64 &z) { return cuCimag(z); }

static inline void cuda_sync() {
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

static inline double ms_since(std::chrono::high_resolution_clock::time_point t0,
                              std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count() * 1e3;
}

static void banner(const std::string &title) {
    std::cout << "\n" << title << "\n" << std::string(94, '-') << "\n";
}

static void row(const std::string &k, const std::string &v, int w = 34) {
    std::cout << "  " << std::left << std::setw(w) << k << v << "\n";
}

static void row_bc(const std::string &k, const std::string &vb, const std::string &vc,
                   int kw = 34, int cw = 18) {
    std::cout << "  " << std::left << std::setw(kw) << k
              << std::right << std::setw(cw) << vb
              << std::right << std::setw(cw) << vc << "\n";
}

// Extended Euclid modular inverse
static uint64_t modinv_u64(uint64_t a, uint64_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)(a % m);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp_t = t - q * newt; t = newt; newt = tmp_t;
        int64_t tmp_r = r - q * newr; r = newr; newr = tmp_r;
    }
    if (r != 1) throw std::runtime_error("modinv: a is not invertible mod m");
    if (t < 0) t += (int64_t)m;
    return (uint64_t)t;
}

static uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    uint64_t x = a % m;
    while (e) {
        if (e & 1) r = (uint64_t)((__uint128_t)r * x % m);
        x = (uint64_t)((__uint128_t)x * x % m);
        e >>= 1;
    }
    return r;
}

static int norm_step(int step, int slots) {
    step %= slots;
    if (step >  slots / 2) step -= slots;
    if (step < -slots / 2) step += slots;
    return step;
}

static uint32_t galois_elt_from_step(int step, uint32_t gen, uint64_t m) {
    if (step == 0) return 1u;
    bool neg = (step < 0);
    uint64_t e = (uint64_t)(neg ? -(int64_t)step : (int64_t)step);
    uint64_t g = powmod_u64((uint64_t)gen, e, m);
    if (neg) g = modinv_u64(g, m);
    return (uint32_t)g;
}

static string fmt_int(int64_t x) {
    std::ostringstream oss;
    oss.imbue(std::locale(""));
    oss << x;
    return oss.str();
}

static double rel_err_complex(const std::vector<pc64> &a, const std::vector<pc64> &b) {
    long double num = 0.0L, den = 0.0L;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        long double dr = (long double)(preal(a[i]) - preal(b[i]));
        long double di = (long double)(pimag(a[i]) - pimag(b[i]));
        num += dr*dr + di*di;
        long double br = (long double)preal(b[i]);
        long double bi = (long double)pimag(b[i]);
        den += br*br + bi*bi;
    }
    return (double)(std::sqrt((double)num) / (std::sqrt((double)den) + 1e-18));
}

static void encode_const_vec(const PhantomContext &ctx,
                             PhantomCKKSEncoder &encoder,
                             pc64 c,
                             double scale,
                             PhantomPlaintext &pt,
                             size_t chain_index = 1) {
    const size_t slots = encoder.slot_count();
    std::vector<pc64> v(slots, c);
    encoder.encode(ctx, v, scale, pt, chain_index);
}

struct Times {
    double precomplex = 0, babyrots = 0, prep = 0, proj = 0, fold = 0, split = 0, vreal = 0, vpack = 0, total = 0;
};

struct Out {
    Times t;
    std::vector<PhantomCiphertext> Q_blocks; // len=6
    std::vector<PhantomCiphertext> K_blocks; // len=6
    std::vector<PhantomCiphertext> V_packed; // len=3
};

int main() {
    // -----------------------------
    // small ring config
    // -----------------------------
    const int SEED   = 114514;
    const int NSLOTS = 16384;
    const int MSEG   = 128;
    const int C      = NSLOTS / MSEG; // 128
    const int D2     = 768;
    const int N1     = 16;
    const int N2     = C / N1;        // 8
    const int BLOCKS = D2 / C;        // 6

    const size_t   Npoly = (size_t)NSLOTS * 2;      // 32768
    const uint64_t m     = 2ull * (uint64_t)Npoly;  // 65536

    const double scale_in  = std::pow(2.0, 40);
    const double scale_w   = std::pow(2.0, 30);
    const double scale_mul = scale_in * scale_w;    // accumulators use this

    cudaSetDevice(0);

    // -----------------------------
    // Parameters
    // -----------------------------
    EncryptionParameters parms(phantom::scheme_type::ckks);
    parms.set_poly_modulus_degree(Npoly);
    parms.set_special_modulus_size(2);
    parms.set_coeff_modulus(CoeffModulus::Create(Npoly, {60, 40, 40, 60, 60, 60}));

    const uint32_t gen_rot  = 5u;
    const uint32_t conj_elt = (uint32_t)(m - 1);

    std::set<int> needed_steps;
    for (int t = 1; t < C; t++) needed_steps.insert(norm_step(t * MSEG, NSLOTS));
    for (int p = 1; p < N2; p++) needed_steps.insert(norm_step(p * N1 * MSEG, NSLOTS));

    std::vector<uint32_t> galois_elts;
    galois_elts.reserve(3 + needed_steps.size());
    galois_elts.push_back(gen_rot);
    galois_elts.push_back((uint32_t)modinv_u64((uint64_t)gen_rot, m));
    galois_elts.push_back(conj_elt);
    for (int s : needed_steps) galois_elts.push_back(galois_elt_from_step(s, gen_rot, m));
    std::sort(galois_elts.begin(), galois_elts.end());
    galois_elts.erase(std::unique(galois_elts.begin(), galois_elts.end()), galois_elts.end());
    parms.set_galois_elts(galois_elts);

    std::cout << "[PHANTOM] small-ring QKV bench"
              << "  (slots=" << NSLOTS << ", Npoly=" << Npoly << ", m=" << m << ")\n";
    std::cout << "[PHANTOM] galois_elts count = " << galois_elts.size()
              << "  (includes conj=" << conj_elt << ")\n";

    PhantomContext context(parms);
    print_parameters(context);

    // -----------------------------
    // Keygen
    // -----------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    PhantomSecretKey sk(context);
    cuda_sync();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(18) << "keygen(sk)" << "  " << ms_since(t0, t1) << " ms\n";

    t0 = std::chrono::high_resolution_clock::now();
    PhantomPublicKey pk = sk.gen_publickey(context);
    cuda_sync();
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(18) << "keygen(pk)" << "  " << ms_since(t0, t1) << " ms\n";

    t0 = std::chrono::high_resolution_clock::now();
    PhantomRelinKey rlk = sk.gen_relinkey(context);
    cuda_sync();
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(18) << "keygen(relin)" << "  " << ms_since(t0, t1) << " ms\n";

    t0 = std::chrono::high_resolution_clock::now();
    PhantomGaloisKey gk = sk.create_galois_keys(context);
    cuda_sync();
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(18) << "keygen(galois)" << "  " << ms_since(t0, t1) << " ms\n";

    // -----------------------------
    // Encoder + constants
    // -----------------------------
    PhantomCKKSEncoder encoder(context);
    const size_t slots = encoder.slot_count();
    if ((int)slots != NSLOTS) throw std::runtime_error("slot_count mismatch");

    PhantomPlaintext pt_zero_in, pt_zero_mul, pt_i, pt_neg_i, pt_minus_one;
    encode_const_vec(context, encoder, pcplx(0.0, 0.0), scale_in,  pt_zero_in,  1);
    encode_const_vec(context, encoder, pcplx(0.0, 0.0), scale_mul, pt_zero_mul, 1);
    encode_const_vec(context, encoder, pcplx(0.0, 1.0), 1.0,       pt_i,        1);
    encode_const_vec(context, encoder, pcplx(0.0,-1.0), 1.0,       pt_neg_i,    1);
    encode_const_vec(context, encoder, pcplx(-1.0,0.0), 1.0,       pt_minus_one,1);

    // Inputs: 6 ciphertexts (random real) at scale_in
    std::mt19937_64 rng(SEED);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<PhantomCiphertext> ct_in(6);
    for (int g = 0; g < 6; g++) {
        std::vector<pc64> msg(slots);
        for (size_t i = 0; i < slots; i++) msg[i] = pcplx(nd(rng), 0.0);
        PhantomPlaintext pt;
        encoder.encode(context, msg, scale_in, pt, 1);
        sk.encrypt_symmetric(context, pt, ct_in[g]);
    }
    cuda_sync();

    PhantomCiphertext ct_zero_in, ct_zero_mul;
    sk.encrypt_symmetric(context, pt_zero_in,  ct_zero_in);
    sk.encrypt_symmetric(context, pt_zero_mul, ct_zero_mul);
    cuda_sync();

    // Weights: scalar per slot
    std::vector<PhantomPlaintext> wq_pt(6), wk_pt(6), wv_pt(6);
    std::vector<double> wq_s(6), wk_s(6), wv_s(6);
    for (int g = 0; g < 6; g++) {
        wq_s[g] = nd(rng);
        wk_s[g] = nd(rng);
        wv_s[g] = nd(rng);
        encode_const_vec(context, encoder, pcplx(wq_s[g], 0.0), scale_w, wq_pt[g], 1);
        encode_const_vec(context, encoder, pcplx(wk_s[g], 0.0), scale_w, wk_pt[g], 1);
        encode_const_vec(context, encoder, pcplx(wv_s[g], 0.0), scale_w, wv_pt[g], 1);
    }

    // Complex pair combine (wm/wp include the 0.5 factor)
    std::vector<PhantomPlaintext> wm_pt(3), wp_pt(3), wv_mix_pt(3);
    for (int p = 0; p < 3; p++) {
        int ge = 2*p, go = 2*p + 1;
        std::complex<double> we(wq_s[ge], wk_s[ge]);
        std::complex<double> wo(wq_s[go], wk_s[go]);

        std::complex<double> wm = 0.5 * (we + std::complex<double>(0.0, -1.0) * wo);
        std::complex<double> wp = 0.5 * (we + std::complex<double>(0.0,  1.0) * wo);

        encode_const_vec(context, encoder, pcplx(wm.real(), wm.imag()), scale_w, wm_pt[p], 1);
        encode_const_vec(context, encoder, pcplx(wp.real(), wp.imag()), scale_w, wp_pt[p], 1);

        std::complex<double> vmix(wv_s[ge], 0.0);
        vmix += std::complex<double>(0.0, -1.0) * std::complex<double>(wv_s[go], 0.0);
        encode_const_vec(context, encoder, pcplx(vmix.real(), vmix.imag()), scale_w, wv_mix_pt[p], 1);
    }

    auto conjugate_inplace = [&](PhantomCiphertext &ct) {
        apply_galois_inplace(context, ct, (size_t)conj_elt, gk);
    };
    auto rotate_step_inplace = [&](PhantomCiphertext &ct, int step) {
        if (step == 0) return;
        rotate_inplace(context, ct, step, gk);
    };

    auto f2 = [](double x){ std::ostringstream o; o<<std::fixed<<std::setprecision(2)<<x; return o.str(); };

    // -----------------------------
    // BASELINE
    // -----------------------------
    auto run_baseline = [&]() -> Out {
        Out out;
        auto T0 = std::chrono::high_resolution_clock::now();

        auto t_p0 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<PhantomCiphertext>> rots(6, std::vector<PhantomCiphertext>(C));
        for (int g = 0; g < 6; g++) {
            rots[g][0] = ct_in[g];
            for (int t = 1; t < C; t++) {
                rots[g][t] = ct_in[g];
                rotate_step_inplace(rots[g][t], norm_step(t * MSEG, NSLOTS));
            }
        }
        cuda_sync();
        auto t_p1 = std::chrono::high_resolution_clock::now();
        out.t.prep = ms_since(t_p0, t_p1);

        auto t_j0 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<PhantomCiphertext>> Qg(BLOCKS, std::vector<PhantomCiphertext>(N2));
        std::vector<std::vector<PhantomCiphertext>> Kg(BLOCKS, std::vector<PhantomCiphertext>(N2));
        std::vector<std::vector<PhantomCiphertext>> Vg(BLOCKS, std::vector<PhantomCiphertext>(N2));
        for (int b = 0; b < BLOCKS; b++) for (int p = 0; p < N2; p++) {
            Qg[b][p]=ct_zero_mul; Kg[b][p]=ct_zero_mul; Vg[b][p]=ct_zero_mul;
        }

        for (int p = 0; p < N2; p++) {
            int base = p * N1;
            for (int b = 0; b < BLOCKS; b++) {
                PhantomCiphertext Sq = ct_zero_mul, Sk = ct_zero_mul, Sv = ct_zero_mul;
                for (int q = 0; q < N1; q++) {
                    int t = base + q;
                    for (int g = 0; g < 6; g++) {
                        PhantomCiphertext tq = rots[g][t];
                        multiply_plain_inplace(context, tq, wq_pt[g]);
                        add_inplace(context, Sq, tq);

                        PhantomCiphertext tk = rots[g][t];
                        multiply_plain_inplace(context, tk, wk_pt[g]);
                        add_inplace(context, Sk, tk);

                        PhantomCiphertext tv = rots[g][t];
                        multiply_plain_inplace(context, tv, wv_pt[g]);
                        add_inplace(context, Sv, tv);
                    }
                }
                add_inplace(context, Qg[b][p], Sq);
                add_inplace(context, Kg[b][p], Sk);
                add_inplace(context, Vg[b][p], Sv);
            }
        }
        cuda_sync();
        auto t_j1 = std::chrono::high_resolution_clock::now();
        out.t.proj = ms_since(t_j0, t_j1);

        auto t_f0 = std::chrono::high_resolution_clock::now();
        out.Q_blocks.assign(BLOCKS, ct_zero_mul);
        out.K_blocks.assign(BLOCKS, ct_zero_mul);
        std::vector<PhantomCiphertext> V_blocks(BLOCKS, ct_zero_mul);

        for (int b = 0; b < BLOCKS; b++) {
            PhantomCiphertext qsum = ct_zero_mul, ksum = ct_zero_mul, vsum = ct_zero_mul;
            for (int p = 0; p < N2; p++) {
                add_inplace(context, qsum, Qg[b][p]);
                add_inplace(context, ksum, Kg[b][p]);
                add_inplace(context, vsum, Vg[b][p]);
            }
            out.Q_blocks[b] = qsum;
            out.K_blocks[b] = ksum;
            V_blocks[b]      = vsum;
        }

        // match complex split outputs (2Q,2K,2V)
        for (int b = 0; b < BLOCKS; b++) {
            add_inplace(context, out.Q_blocks[b], out.Q_blocks[b]);
            add_inplace(context, out.K_blocks[b], out.K_blocks[b]);
            add_inplace(context, V_blocks[b],     V_blocks[b]);
        }
        cuda_sync();
        auto t_f1 = std::chrono::high_resolution_clock::now();
        out.t.fold = ms_since(t_f0, t_f1);

        auto t_v0 = std::chrono::high_resolution_clock::now();
        out.V_packed.clear();
        out.V_packed.reserve(3);
        for (int p = 0; p < 3; p++) {
            PhantomCiphertext a = V_blocks[2*p];
            PhantomCiphertext b = V_blocks[2*p + 1];
            multiply_plain_inplace(context, b, pt_i);
            add_inplace(context, a, b);
            out.V_packed.push_back(a);
        }
        cuda_sync();
        auto t_v1 = std::chrono::high_resolution_clock::now();
        out.t.vpack = ms_since(t_v0, t_v1);

        auto T1 = std::chrono::high_resolution_clock::now();
        out.t.total = ms_since(T0, T1);
        return out;
    };

    // -----------------------------
    // COMPLEX (BSGS)
    // -----------------------------
    auto run_complex = [&]() -> Out {
        Out out;
        auto T0 = std::chrono::high_resolution_clock::now();

        auto t_c0 = std::chrono::high_resolution_clock::now();
        PhantomCiphertext ct01 = ct_in[0];
        { PhantomCiphertext tmp = ct_in[1]; multiply_plain_inplace(context, tmp, pt_i); add_inplace(context, ct01, tmp); }
        PhantomCiphertext ct23 = ct_in[2];
        { PhantomCiphertext tmp = ct_in[3]; multiply_plain_inplace(context, tmp, pt_i); add_inplace(context, ct23, tmp); }
        PhantomCiphertext ct45 = ct_in[4];
        { PhantomCiphertext tmp = ct_in[5]; multiply_plain_inplace(context, tmp, pt_i); add_inplace(context, ct45, tmp); }
        cuda_sync();
        auto t_c1 = std::chrono::high_resolution_clock::now();
        out.t.precomplex = ms_since(t_c0, t_c1);

        auto t_b0 = std::chrono::high_resolution_clock::now();
        std::vector<PhantomCiphertext> b01(N1), b23(N1), b45(N1);
        b01[0]=ct01; b23[0]=ct23; b45[0]=ct45;
        for (int q = 1; q < N1; q++) {
            int step = norm_step(q * MSEG, NSLOTS);
            b01[q]=ct01; rotate_step_inplace(b01[q], step);
            b23[q]=ct23; rotate_step_inplace(b23[q], step);
            b45[q]=ct45; rotate_step_inplace(b45[q], step);
        }
        std::vector<PhantomCiphertext> c01(N1), c23(N1), c45(N1);
        for (int q = 0; q < N1; q++) {
            c01[q]=b01[q]; conjugate_inplace(c01[q]);
            c23[q]=b23[q]; conjugate_inplace(c23[q]);
            c45[q]=b45[q]; conjugate_inplace(c45[q]);
        }
        cuda_sync();
        auto t_b1 = std::chrono::high_resolution_clock::now();
        out.t.babyrots = ms_since(t_b0, t_b1);

        auto t_j0 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<PhantomCiphertext>> QKg(BLOCKS, std::vector<PhantomCiphertext>(N2));
        std::vector<std::vector<PhantomCiphertext>> Vraw(BLOCKS, std::vector<PhantomCiphertext>(N2));
        for (int b = 0; b < BLOCKS; b++) for (int p = 0; p < N2; p++) {
            QKg[b][p]=ct_zero_mul; Vraw[b][p]=ct_zero_mul;
        }

        for (int b = 0; b < BLOCKS; b++) {
            for (int p = 0; p < N2; p++) {
                int gshift = norm_step(p * N1 * MSEG, NSLOTS);
                PhantomCiphertext tmp_qk = ct_zero_mul;
                PhantomCiphertext tmp_v  = ct_zero_mul;

                for (int q = 0; q < N1; q++) {
                    PhantomCiphertext term_qk = ct_zero_mul;

                    { PhantomCiphertext t1=b01[q]; multiply_plain_inplace(context, t1, wm_pt[0]);
                      PhantomCiphertext t2=c01[q]; multiply_plain_inplace(context, t2, wp_pt[0]);
                      add_inplace(context, t1, t2); add_inplace(context, term_qk, t1); }
                    { PhantomCiphertext t1=b23[q]; multiply_plain_inplace(context, t1, wm_pt[1]);
                      PhantomCiphertext t2=c23[q]; multiply_plain_inplace(context, t2, wp_pt[1]);
                      add_inplace(context, t1, t2); add_inplace(context, term_qk, t1); }
                    { PhantomCiphertext t1=b45[q]; multiply_plain_inplace(context, t1, wm_pt[2]);
                      PhantomCiphertext t2=c45[q]; multiply_plain_inplace(context, t2, wp_pt[2]);
                      add_inplace(context, t1, t2); add_inplace(context, term_qk, t1); }

                    add_inplace(context, tmp_qk, term_qk);

                    PhantomCiphertext term_v = ct_zero_mul;
                    { PhantomCiphertext t1=b01[q]; multiply_plain_inplace(context, t1, wv_mix_pt[0]); add_inplace(context, term_v, t1); }
                    { PhantomCiphertext t1=b23[q]; multiply_plain_inplace(context, t1, wv_mix_pt[1]); add_inplace(context, term_v, t1); }
                    { PhantomCiphertext t1=b45[q]; multiply_plain_inplace(context, t1, wv_mix_pt[2]); add_inplace(context, term_v, t1); }
                    add_inplace(context, tmp_v, term_v);
                }

                if (gshift != 0) {
                    rotate_step_inplace(tmp_qk, gshift);
                    rotate_step_inplace(tmp_v,  gshift);
                }
                QKg[b][p]=tmp_qk;
                Vraw[b][p]=tmp_v;
            }
        }
        cuda_sync();
        auto t_j1 = std::chrono::high_resolution_clock::now();
        out.t.proj = ms_since(t_j0, t_j1);

        auto t_f0 = std::chrono::high_resolution_clock::now();
        std::vector<PhantomCiphertext> QK_blocks(BLOCKS, ct_zero_mul);
        std::vector<PhantomCiphertext> Vraw_blocks(BLOCKS, ct_zero_mul);
        for (int b = 0; b < BLOCKS; b++) {
            PhantomCiphertext qks=ct_zero_mul, vs=ct_zero_mul;
            for (int p = 0; p < N2; p++) { add_inplace(context, qks, QKg[b][p]); add_inplace(context, vs, Vraw[b][p]); }
            QK_blocks[b]=qks; Vraw_blocks[b]=vs;
        }
        cuda_sync();
        auto t_f1 = std::chrono::high_resolution_clock::now();
        out.t.fold = ms_since(t_f0, t_f1);

        auto t_s0 = std::chrono::high_resolution_clock::now();
        out.Q_blocks.assign(BLOCKS, ct_zero_mul);
        out.K_blocks.assign(BLOCKS, ct_zero_mul);
        for (int b = 0; b < BLOCKS; b++) {
            PhantomCiphertext conj = QK_blocks[b];
            conjugate_inplace(conj);

            PhantomCiphertext Q = QK_blocks[b];
            add_inplace(context, Q, conj); // 2Q

            PhantomCiphertext diff = conj;
            multiply_plain_inplace(context, diff, pt_minus_one);
            add_inplace(context, diff, QK_blocks[b]);
            multiply_plain_inplace(context, diff, pt_neg_i); // 2K

            out.Q_blocks[b]=Q;
            out.K_blocks[b]=diff;
        }
        cuda_sync();
        auto t_s1 = std::chrono::high_resolution_clock::now();
        out.t.split = ms_since(t_s0, t_s1);

        auto t_r0 = std::chrono::high_resolution_clock::now();
        std::vector<PhantomCiphertext> V_blocks(BLOCKS, ct_zero_mul);
        for (int b = 0; b < BLOCKS; b++) {
            PhantomCiphertext conj = Vraw_blocks[b];
            conjugate_inplace(conj);
            PhantomCiphertext Vr = Vraw_blocks[b];
            add_inplace(context, Vr, conj); // 2V
            V_blocks[b]=Vr;
        }
        cuda_sync();
        auto t_r1 = std::chrono::high_resolution_clock::now();
        out.t.vreal = ms_since(t_r0, t_r1);

        auto t_v0 = std::chrono::high_resolution_clock::now();
        out.V_packed.clear();
        out.V_packed.reserve(3);
        for (int p = 0; p < 3; p++) {
            PhantomCiphertext a = V_blocks[2*p];
            PhantomCiphertext b = V_blocks[2*p + 1];
            multiply_plain_inplace(context, b, pt_i);
            add_inplace(context, a, b);
            out.V_packed.push_back(a);
        }
        cuda_sync();
        auto t_v1 = std::chrono::high_resolution_clock::now();
        out.t.vpack = ms_since(t_v0, t_v1);

        auto T1 = std::chrono::high_resolution_clock::now();
        out.t.total = ms_since(T0, T1);
        return out;
    };

    // Warmup + run
    (void)run_baseline();
    (void)run_complex();
    cuda_sync();

    Out B  = run_baseline();
    Out Cx = run_complex();
    cuda_sync();

    // -----------------------------
    // Print compare (FF-style)
    // -----------------------------
    banner("QKV (PHANTOM CKKS) — SMALL ring — dense-in/dense-out (structure microbench)");
    row("SEED", std::to_string(SEED));
    row("Dims", "slots=16,384, m=128, C=128, d1=768, d2=768");
    row("BSGS", "N1=16, N2=8, blocks=6");

    banner("QKV: A -> (Q, K, V) — Ciphertext I/O");
    std::cout << "  " << std::left << std::setw(34) << "" << std::right << std::setw(18) << "baseline" << std::right << std::setw(18) << "complex" << "\n";
    row_bc("Input CTs (dense A)",  "6", "6");
    row_bc("Core input CTs",       "6", "3");
    row_bc("Output CTs: Q blocks", "6", "6");
    row_bc("Output CTs: K blocks", "6", "6");
    row_bc("Output CTs: V packed", "3", "3");

    banner("QKV — Time (ms)");
    row_bc("pre-complexify",            f2(B.t.precomplex), f2(Cx.t.precomplex));
    row_bc("baby rots (+conj babies)",  f2(B.t.babyrots),   f2(Cx.t.babyrots));
    row_bc("prep rotbank",              f2(B.t.prep),       f2(Cx.t.prep));
    row_bc("proj (QKV)",                f2(B.t.proj),       f2(Cx.t.proj));
    row_bc("fold (to blocks)",          f2(B.t.fold),       f2(Cx.t.fold));
    row_bc("split QK -> Q,K",           f2(B.t.split),      f2(Cx.t.split));
    row_bc("V real-clean",              f2(B.t.vreal),      f2(Cx.t.vreal));
    row_bc("V pack (to 3 CTs)",         f2(B.t.vpack),      f2(Cx.t.vpack));
    row_bc("TOTAL",                     f2(B.t.total),      f2(Cx.t.total));

    banner("QKV — Theoretical keyswitch counts (same as SIM bookkeeping)");
    row_bc("ks_rots",
           fmt_int(6 * (C - 1)),
           fmt_int(3 * (N1 - 1) + 2 * BLOCKS * (N2 - 1)));
    row_bc("ks_conj",
           "0",
           fmt_int(3 * N1 + 2 * BLOCKS));

    // Optional: keep ONE sanity number (V only), remove if you want absolutely no decrypt
    banner("Sanity (decrypt) — baseline vs complex (optional)");
    PhantomPlaintext ptB, ptC;
    std::vector<pc64> dB, dC;
    sk.decrypt(context, B.V_packed[0], ptB);
    sk.decrypt(context, Cx.V_packed[0], ptC);
    encoder.decode(context, ptB, dB);
    encoder.decode(context, ptC, dC);
    row("RelErr(V_packed0.complex)", (std::ostringstream() << std::scientific << std::setprecision(3)
        << rel_err_complex(dC, dB)).str());

    return 0;
}
