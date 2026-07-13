// bench_ckks_micro.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "example.h"     // print_parameters(...)
#include "phantom.h"
#include "util.cuh"
#include "evaluate.cuh"  // apply_galois_inplace

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

static inline void cuda_sync() {
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    double idx = (p / 100.0) * double(v.size() - 1);
    size_t i0 = (size_t)std::floor(idx);
    size_t i1 = std::min(i0 + 1, v.size() - 1);
    double t = idx - double(i0);
    return v[i0] * (1.0 - t) + v[i1] * t;
}

struct BenchStats {
    double mean_ms = 0.0;
    double p50_ms  = 0.0;
    double p90_ms  = 0.0;
    double p99_ms  = 0.0;
};

template <class F>
static BenchStats bench_op_ret(const std::string &label, F &&op, int warmup = 20, int iters = 200) {
    for (int i = 0; i < warmup; i++) { op(); cuda_sync(); }

    std::vector<double> ms;
    ms.reserve(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        op();
        cuda_sync();
        auto t1 = std::chrono::high_resolution_clock::now();
        ms.push_back(std::chrono::duration<double>(t1 - t0).count() * 1e3);
    }

    BenchStats s;
    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / double(ms.size());
    s.p50_ms  = percentile(ms, 50);
    s.p90_ms  = percentile(ms, 90);
    s.p99_ms  = percentile(ms, 99);

    std::cout << std::setw(18) << label
              << "  mean=" << std::setw(8) << std::fixed << std::setprecision(3) << s.mean_ms << " ms"
              << "   p50=" << std::setw(8) << s.p50_ms
              << "  p90=" << std::setw(8) << s.p90_ms
              << "  p99=" << std::setw(8) << s.p99_ms
              << "\n";
    return s;
}

template <class F>
static bool bench_op_safe_ret(const std::string &label, F &&op, BenchStats &out, int warmup = 20, int iters = 200) {
    try {
        out = bench_op_ret(label, std::forward<F>(op), warmup, iters);
        return true;
    } catch (const std::exception &e) {
        std::cout << std::setw(18) << label << "  SKIPPED (" << e.what() << ")\n";
        out = BenchStats{};
        return false;
    }
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

static std::vector<int> build_coeff_bits(int L, int SPECIAL) {
    // Main chain: 60 + L*40 + 60
    // Special primes: SPECIAL*60 (appended)
    std::vector<int> bits;
    bits.reserve(2 + L + SPECIAL);
    bits.push_back(60);
    bits.insert(bits.end(), L, 40);
    bits.push_back(60);
    bits.insert(bits.end(), SPECIAL, 60);
    return bits;
}

struct CMView {
    std::vector<int> bits;
    std::vector<uint64_t> primes;
};

// NOTE: in your build, coeff_modulus() is vector<phantom::arith::Modulus>
static CMView cmview_from_coeff_modulus(const std::vector<phantom::arith::Modulus> &cm) {
    CMView v;
    v.bits.reserve(cm.size());
    v.primes.reserve(cm.size());
    for (size_t i = 0; i < cm.size(); i++) {
        v.bits.push_back((int)cm[i].bit_count());
        v.primes.push_back((uint64_t)cm[i].value());
    }
    return v;
}

static void print_cmview(const std::string &tag, const CMView &v) {
    long long sum_bits = 0;
    for (int b : v.bits) sum_bits += b;

    std::cout << tag << " coeff_modulus bits = {";
    for (size_t i = 0; i < v.bits.size(); i++) {
        std::cout << v.bits[i] << (i + 1 == v.bits.size() ? "" : ", ");
    }
    std::cout << "}  (count=" << v.bits.size() << ", sum_bits=" << sum_bits << ")\n";

    std::cout << tag << " primes = ";
    for (size_t i = 0; i < v.primes.size(); i++) {
        std::cout << v.primes[i] << (i + 1 == v.primes.size() ? "" : " , ");
    }
    std::cout << "\n";
}

static CMView drop_last(const CMView &v) {
    CMView o = v;
    if (!o.bits.empty())   o.bits.pop_back();
    if (!o.primes.empty()) o.primes.pop_back();
    return o;
}

static void dump_scale(const std::string &tag, const PhantomCiphertext &ct) {
    double s = (double)ct.scale();
    std::cout << tag << " scale=" << std::setprecision(6) << std::scientific << s
              << " (log2=" << std::fixed << std::setprecision(3) << std::log2(s) << ")\n";
}

struct Case { int L; size_t N; };

static void run_case(const Case &tc,
                     int SPECIAL,
                     double SCALE,
                     double P99_DIV,
                     std::ofstream *csv) {
    const int L = tc.L;
    const size_t N = tc.N;
    const uint64_t m = 2ull * (uint64_t)N;

    const uint32_t gen_rot     = 5u;
    const uint32_t gen_rot_inv = (uint32_t)modinv_u64((uint64_t)gen_rot, m);
    const uint32_t conj_elt    = (uint32_t)(m - 1);

    std::vector<int> bits = build_coeff_bits(L, SPECIAL);
    const int total_primes = (int)bits.size();

    std::cout << "\n============================================================\n";
    std::cout << "[BENCH] L=" << L << "  N=" << N
              << "  (bits: 60 + " << L << "*40 + 60, special=" << SPECIAL
              << ", total_primes=" << total_primes << ")\n";
    std::cout << "============================================================\n";

    std::cout << "[BITS] {";
    for (size_t i = 0; i < bits.size(); i++) std::cout << bits[i] << (i + 1 == bits.size() ? "" : ", ");
    std::cout << "}\n";

    EncryptionParameters parms(phantom::scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_special_modulus_size(SPECIAL);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
    parms.set_galois_elts({ gen_rot, gen_rot_inv, conj_elt });

    std::cout << "[PHANTOM] galois_elts={ " << gen_rot << ", " << gen_rot_inv << ", " << conj_elt
              << " }  (m=" << m << ")\n";

    PhantomContext context(parms);

    // Phantom's built-in print shows the "full" mod chain (includes appended special prime(s))
    print_parameters(context);

    // Ciphertext-chain primes at creation (your prior output shows this excludes special prime)
    auto *cd0 = &context.first_context_data();
    auto chain0 = cmview_from_coeff_modulus(cd0->parms().coeff_modulus());
    print_cmview("[CT chain / created]", chain0);

    // "After 1 rescale": your build has no next_context_data(), so derive expected by dropping last.
    auto chain1_expected = drop_last(chain0);
    print_cmview("[CT chain / after 1 rescale (expected)]", chain1_expected);
    std::cout << "[NOTE] ContextData has no next_context_data() in this Phantom build; "
                 "so level-1 chain is derived by dropping the last ciphertext-chain prime.\n";

    // ---- keygen ----
    PhantomSecretKey sk(context); cuda_sync();
    PhantomPublicKey pk = sk.gen_publickey(context); (void)pk; cuda_sync();
    PhantomRelinKey rlk = sk.gen_relinkey(context); cuda_sync();
    PhantomGaloisKey gk = sk.create_galois_keys(context); cuda_sync();

    // ---- encode/encrypt ----
    PhantomCKKSEncoder encoder(context);
    const size_t slots = encoder.slot_count();     // N/2
    const size_t used_slots = std::min<size_t>(slots, (size_t)(1ull << 14));

    std::vector<double> a(slots, 0.0), b(slots, 0.0);
    std::mt19937_64 rng(123);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (size_t i = 0; i < used_slots; i++) { a[i] = nd(rng); b[i] = nd(rng); }

    PhantomPlaintext pt_a, pt_b;
    encoder.encode(context, a, SCALE, pt_a, 1);
    encoder.encode(context, b, SCALE, pt_b, 1);
    cuda_sync();

    PhantomCiphertext ct_a, ct_b;
    sk.encrypt_symmetric(context, pt_a, ct_a);
    sk.encrypt_symmetric(context, pt_b, ct_b);
    cuda_sync();

    dump_scale("[CT_A before]", ct_a);

    // One mul + relin + rescale
    PhantomCiphertext ct_mul = multiply(context, ct_a, ct_b);
    relinearize_inplace(context, ct_mul, rlk);
    rescale_to_next_inplace(context, ct_mul);
    cuda_sync();

    dump_scale("[CT_MUL after 1x mul+relin+rescale]", ct_mul);

    // ---- benches ----
    BenchStats st_ct{}, st_rot{}, st_conj{};

    bool ok_ct = bench_op_safe_ret("ct mult", [&]() {
        PhantomCiphertext tmp = multiply(context, ct_a, ct_b);
        relinearize_inplace(context, tmp, rlk);
        rescale_to_next_inplace(context, tmp);
    }, st_ct);

    bool ok_rot = bench_op_safe_ret("rotate(1)", [&]() {
        auto tmp = ct_a;
        rotate_inplace(context, tmp, 1, gk);
    }, st_rot);

    bool ok_conj = bench_op_safe_ret("conjugate", [&]() {
        auto tmp = ct_a;
        apply_galois_inplace(context, tmp, (size_t)conj_elt, gk);
    }, st_conj);

    double p99_ct   = ok_ct   ? (st_ct.p99_ms   / P99_DIV) : std::numeric_limits<double>::quiet_NaN();
    double p99_rot  = ok_rot  ? (st_rot.p99_ms  / P99_DIV) : std::numeric_limits<double>::quiet_NaN();
    double p99_conj = ok_conj ? (st_conj.p99_ms / P99_DIV) : std::numeric_limits<double>::quiet_NaN();

    if (csv && csv->is_open()) {
        (*csv) << N << "," << L << "," << total_primes << ","
               << std::setprecision(10) << p99_ct << ","
               << std::setprecision(10) << p99_rot << ","
               << std::setprecision(10) << p99_conj << "\n";
        csv->flush();
    }
}

int main() {
    cudaSetDevice(0);

    // Requested cases:
    //   L=8  -> N=32768
    //   L=3  -> N=32768 and N=65536
    const std::vector<Case> CASES = {
        {8, 1ull << 15},   // 32768
        {3, 1ull << 15},   // 32768
        {3, 1ull << 16},   // 65536
    };

    // You can set this (not preset)
    const int SPECIAL = 1;

    const double SCALE = std::pow(2.0, 40);
    const double P99_DIV = 1.5;

    const char *csv_path_env = std::getenv("CSV_PATH");
    const std::string csv_path = (csv_path_env && csv_path_env[0]) ? std::string(csv_path_env)
                                                                    : std::string("bench_phantom_cases_p99.csv");

    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "[WARN] Could not open CSV for write: " << csv_path << "\n";
    } else {
        csv << "N,L,total_primes,ct_mult_p99_ms,rot_p99_ms,conj_p99_ms\n";
        csv.flush();
    }

    for (const auto &tc : CASES) {
        run_case(tc, SPECIAL, SCALE, P99_DIV, csv.is_open() ? &csv : nullptr);
    }

    if (csv.is_open()) {
        std::cout << "\n[wrote] " << csv_path
                  << "  (ct_mult/rot/conj p99 divided by 1.5)\n";
    }
    return 0;
}
