// bench_f0f1.cu — measure compute_f0_f1_in_ckks (BOLT Algorithm-4 Step B
// pre-evaluation in CKKS) per FF1-output block under PhantomFHE on A100.
//
// Reports per-block wallclock so the per-inference cost can be reconstructed
// as: per-block × (D_FF / C_paired) × L_layers.
//
// Build target: bench_f0f1 (see examples/CMakeLists.txt).
// Run: ./bench_f0f1 [num_warmup] [num_iter]   (defaults 1, 5)

#include <algorithm>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "example.h"
#include "pipe_ckks_common.h"

using namespace pipe_ckks;

namespace {

#include "model_config.h"
constexpr int NSLOTS = EF_NSLOTS;
constexpr int M      = EF_M;
constexpr int C      = EF_C;
constexpr int D_FF   = EF_D_FF;
constexpr int HP_FF1 = EF_HP_FF1;

static pc64 eval_poly_slot(const pc64 &x, bool f1) {
    const std::complex<double> z(preal(x), pimag(x));
    const std::complex<double> z2 = z * z;
    const std::complex<double> z3 = z2 * z;
    const std::complex<double> z4 = z2 * z2;
    const double sign_b = f1 ? bolt_gelu::B : -bolt_gelu::B;
    const double lin = f1 ? (0.5 + bolt_gelu::D) : (0.5 - bolt_gelu::D);
    const std::complex<double> y =
        bolt_gelu::A * z4 + sign_b * z3 + bolt_gelu::C_COEF * z2 +
        lin * z + bolt_gelu::E;
    return pcplx(y.real(), y.imag());
}

static std::vector<pc64> decrypt_slots(
    const PhantomContext &ctx,
    PhantomCKKSEncoder &encoder,
    PhantomSecretKey &sk,
    const PhantomCiphertext &ct)
{
    PhantomPlaintext pt;
    std::vector<pc64> dec;
    sk.decrypt(ctx, ct, pt);
    encoder.decode(ctx, pt, dec);
    return dec;
}

}  // anonymous namespace

int main(int argc, char **argv) {
    const int num_warmup = (argc > 1) ? std::atoi(argv[1]) : 1;
    const int num_iter   = (argc > 2) ? std::atoi(argv[2]) : 5;
    const char *std_env = std::getenv("ENCFORMER_F0F1_INPUT_STD");
    const double input_std = (std_env && *std_env) ? std::atof(std_env) : 0.02;

    // --- CKKS context: matches the reportable native pipeline profile. ---
    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NSLOTS * 2);
    const int body_limbs = ckks_body_limbs_from_env(7);
    const int log2q = 120 + 40 * body_limbs;
    const double scale_in = std::pow(2.0, 40);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rlk = sk.gen_relinkey(ctx);
    PhantomCKKSEncoder encoder(ctx);

    const size_t slots = encoder.slot_count();
    const int blocks_per_layer = D_FF / C;  // number of FF1-output ciphertext blocks per layer

    std::cout << "Block 3 params: N=" << (NSLOTS * 2)
              << " slots=" << slots
              << " body_limbs=" << body_limbs
              << " log2Q=" << log2q
              << " input_std=" << input_std
              << " D_FF=" << D_FF
              << " C=" << C
              << " blocks_per_layer=" << blocks_per_layer
              << std::endl;

    // --- Random input vector (representing one FF1-output block) ---
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<double> dist(0.0, input_std);

    auto build_block = [&](std::vector<pc64> *plain = nullptr) -> PhantomCiphertext {
        std::vector<pc64> msg(slots, pcplx(0.0));
        for (size_t i = 0; i < slots; ++i) {
            msg[i] = pcplx(dist(rng), dist(rng));
        }
        if (plain) *plain = msg;
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale_in, pt, 1);
        PhantomCiphertext ct;
        sk.encrypt_symmetric(ctx, pt, ct);
        return ct;
    };

    // --- Warmup ---
    for (int w = 0; w < num_warmup; ++w) {
        std::vector<PhantomCiphertext> x_blocks;
        for (int b = 0; b < blocks_per_layer; ++b) x_blocks.push_back(build_block());
        cudaDeviceSynchronize();
        auto out = compute_f0_f1_in_ckks(ctx, encoder, rlk, x_blocks, scale_in, NSLOTS);
        cudaDeviceSynchronize();
    }

    // --- Timing iterations ---
    std::vector<double> per_layer_ms;
    double mse_f0 = 0.0;
    double mse_f1 = 0.0;
    for (int it = 0; it < num_iter; ++it) {
        std::vector<PhantomCiphertext> x_blocks;
        std::vector<std::vector<pc64>> plain_blocks;
        if (it == 0) plain_blocks.reserve(static_cast<size_t>(blocks_per_layer));
        for (int b = 0; b < blocks_per_layer; ++b) {
            if (it == 0) {
                plain_blocks.emplace_back();
                x_blocks.push_back(build_block(&plain_blocks.back()));
            } else {
                x_blocks.push_back(build_block());
            }
        }
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = compute_f0_f1_in_ckks(ctx, encoder, rlk, x_blocks, scale_in, NSLOTS);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        const double layer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        per_layer_ms.push_back(layer_ms);
        std::cout << "iter " << it << " per_layer_ms " << std::fixed
                  << std::setprecision(3) << layer_ms << std::endl;

        if (it == 0) {
            double max_mse_f0 = 0.0;
            double max_mse_f1 = 0.0;
            for (int b = 0; b < blocks_per_layer; ++b) {
                std::vector<pc64> ref_f0(slots);
                std::vector<pc64> ref_f1(slots);
                for (size_t s = 0; s < slots; ++s) {
                    ref_f0[s] = eval_poly_slot(plain_blocks[static_cast<size_t>(b)][s], false);
                    ref_f1[s] = eval_poly_slot(plain_blocks[static_cast<size_t>(b)][s], true);
                }
                auto got_f0 = decrypt_slots(ctx, encoder, sk, out.f0[static_cast<size_t>(b)]);
                auto got_f1 = decrypt_slots(ctx, encoder, sk, out.f1[static_cast<size_t>(b)]);
                max_mse_f0 = std::max(max_mse_f0, mse_complex_slots(got_f0, ref_f0));
                max_mse_f1 = std::max(max_mse_f1, mse_complex_slots(got_f1, ref_f1));
            }
            mse_f0 = max_mse_f0;
            mse_f1 = max_mse_f1;
        }
    }

    // --- Summary ---
    double mean = 0;
    for (double v : per_layer_ms) mean += v;
    mean /= per_layer_ms.size();
    double var = 0;
    for (double v : per_layer_ms) var += (v - mean) * (v - mean);
    var /= per_layer_ms.size();
    const double stdev = std::sqrt(var);

    std::cout << "\n=== F0/F1 pre-evaluation timing (PhantomFHE, A100) ===\n";
    std::cout << "Lite native chain (log2Q=" << log2q << ", scale=2^40), "
              << blocks_per_layer << " FF1-output blocks per layer\n";
    std::cout << "Per-layer wallclock: " << std::fixed << std::setprecision(3)
              << mean << " ms (stdev " << stdev << ")\n";
    std::cout << "Projected per-inference (L=12): " << (mean * 12) << " ms = "
              << (mean * 12 / 1000.0) << " s\n";
    std::cout << "mse_f0 " << std::scientific << mse_f0 << std::defaultfloat << "\n";
    std::cout << "mse_f1 " << std::scientific << mse_f1 << std::defaultfloat << "\n";
    std::cout << "mse " << std::scientific << std::max(mse_f0, mse_f1) << std::defaultfloat << "\n";

    return 0;
}
