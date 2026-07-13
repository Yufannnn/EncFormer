// pipe_ckks_qkv.cu — Native CKKS QKV projection with file I/O.
//
// Reads:  PIPE_DIR/qkv_in.bin = X[M×D1] ++ WQ[D1×D2] ++ WK[D1×D2] ++ WV[D1×D2]
//                                ++ bQ[D2] ++ bK[D2] ++ bV[D2]    (row-major float64)
// Writes: PIPE_DIR/qkv_out.bin = Q[M×D2] ++ K[M×D2] ++ V[M×D2]  (row-major float64)
//
// Uses 3 separate linear_complex_paired operations for Q, K, V (same as standalone QKV bench).
// Output is standard column order (head-major: Q[:, h*d_k:(h+1)*d_k] = head h's query).

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "example.h"
#include "pipe_io.h"
#include "pipe_ckks_common.h"

using namespace pipe_ckks;

namespace {
#include "model_config.h"
constexpr int NSLOTS = EF_NSLOTS;
constexpr int M      = EF_M;
constexpr int C      = EF_C;
constexpr int N1     = EF_N1_DEFAULT;
constexpr int N2     = EF_N2_DEFAULT;
constexpr int D1     = EF_D;
constexpr int D2     = EF_D;
constexpr int HP         = EF_HP_QKV;
constexpr int BLOCKS     = EF_BLOCKS;
constexpr int C_USED_LIN = EF_C_USED_LIN;
} // namespace

int main() {
    constexpr size_t NPOLY = static_cast<size_t>(NSLOTS) * 2;
    constexpr uint64_t MMOD = 2ull * static_cast<uint64_t>(NPOLY);
    constexpr uint32_t GEN_ROT = 5u;

    const double scale_in = std::pow(2.0, 40);
    const double scale_w  = std::pow(2.0, 40);
    const double scale_mul = scale_in * scale_w;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "failed to set cuda device\n";
        return 1;
    }

    // --- Read input data ---
    auto t_io0 = std::chrono::high_resolution_clock::now();
    std::string pd = pipe_io::get_pipe_dir();
    std::string in_path = pd + "/qkv_in.bin";

    constexpr size_t N_X  = (size_t)M * D1;
    constexpr size_t N_W  = (size_t)D1 * D2;
    constexpr size_t N_b  = D2;
    constexpr size_t TOTAL = N_X + 3 * N_W + 3 * N_b;
    std::vector<double> buf(TOTAL);
    pipe_io::read_f64(in_path.c_str(), buf.data(), buf.size());

    const double *X  = buf.data();
    const double *WQ = X + N_X;
    const double *WK = WQ + N_W;
    const double *WV = WK + N_W;
    const double *bQ = WV + N_W;
    const double *bK = bQ + N_b;
    const double *bV = bK + N_b;
    auto t_io1 = std::chrono::high_resolution_clock::now();

    // --- CKKS context setup ---
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);

    auto galois_elts = build_galois_elts_full_columns(NSLOTS, M, GEN_ROT, MMOD);
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    (void)pk;
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomCKKSEncoder encoder(ctx);
    cuda_sync();
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    // --- Encrypt input ---
    auto t_enc0 = std::chrono::high_resolution_clock::now();
    auto x_ct = encrypt_packed_pairs(ctx, encoder, sk, X, M, D1, HP, NSLOTS,
                                      scale_in, 1, C_USED_LIN);
    auto ct_zero_mul = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();
    auto t_enc1 = std::chrono::high_resolution_clock::now();

    // --- Build weight tables ---
    auto WQ_tab = build_wtab_paired(WQ, D1, D2, C, -1, -1, C_USED_LIN);
    auto WK_tab = build_wtab_paired(WK, D1, D2, C, -1, -1, C_USED_LIN);
    auto WV_tab = build_wtab_paired(WV, D1, D2, C, -1, -1, C_USED_LIN);

    // --- FHE computation: 3 linear projections ---
    KSCounters ks;
    auto t_fhe0 = std::chrono::high_resolution_clock::now();

    auto babies = build_babies(ctx, gk, x_ct, N1, M, NSLOTS, ks);

    auto Q_ct = linear_complex_paired(ctx, encoder, gk, babies, WQ_tab,
                                       D2, M, N1, N2, C, NSLOTS, scale_w, ct_zero_mul, ks);
    auto K_ct = linear_complex_paired(ctx, encoder, gk, babies, WK_tab,
                                       D2, M, N1, N2, C, NSLOTS, scale_w, ct_zero_mul, ks);
    auto V_ct = linear_complex_paired(ctx, encoder, gk, babies, WV_tab,
                                       D2, M, N1, N2, C, NSLOTS, scale_w, ct_zero_mul, ks);
    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);
    ct_real_blocks(ctx, gk, Q_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, K_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, V_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();
    auto t_fhe1 = std::chrono::high_resolution_clock::now();

    // --- Decrypt ---
    auto t_dec0 = std::chrono::high_resolution_clock::now();
    auto Q_out = decrypt_blocks(ctx, encoder, sk, Q_ct, M, D2, NSLOTS);
    auto K_out = decrypt_blocks(ctx, encoder, sk, K_ct, M, D2, NSLOTS);
    auto V_out = decrypt_blocks(ctx, encoder, sk, V_ct, M, D2, NSLOTS);

    // Add bias in plaintext
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D2; j++) {
            Q_out[(size_t)i * D2 + j] += bQ[j];
            K_out[(size_t)i * D2 + j] += bK[j];
            V_out[(size_t)i * D2 + j] += bV[j];
        }
    auto t_dec1 = std::chrono::high_resolution_clock::now();

    // --- Write output ---
    auto t_wrt0 = std::chrono::high_resolution_clock::now();
    constexpr size_t N_OUT = (size_t)M * D2;
    std::vector<double> out_buf(3 * N_OUT);
    std::copy(Q_out.begin(), Q_out.end(), out_buf.begin());
    std::copy(K_out.begin(), K_out.end(), out_buf.begin() + N_OUT);
    std::copy(V_out.begin(), V_out.end(), out_buf.begin() + 2 * N_OUT);
    std::string out_path = pd + "/qkv_out.bin";
    pipe_io::write_f64(out_path.c_str(), out_buf.data(), out_buf.size());
    auto t_wrt1 = std::chrono::high_resolution_clock::now();

    // --- Verify ---
    auto Q_ref = matmul_ref(X, M, D1, WQ, D2);
    auto K_ref = matmul_ref(X, M, D1, WK, D2);
    auto V_ref = matmul_ref(X, M, D1, WV, D2);
    // Add bias to references
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D2; j++) {
            Q_ref[(size_t)i * D2 + j] += bQ[j];
            K_ref[(size_t)i * D2 + j] += bK[j];
            V_ref[(size_t)i * D2 + j] += bV[j];
        }
    double err_q = rel_err_real(Q_out, Q_ref);
    double err_k = rel_err_real(K_out, K_ref);
    double err_v = rel_err_real(V_out, V_ref);
    double mse_q = mse_real(Q_out, Q_ref);
    double mse_k = mse_real(K_out, K_ref);
    double mse_v = mse_real(V_out, V_ref);

    // --- Print metrics ---
    std::cout << std::setprecision(10);
    std::cout << "stage pipe_qkv\n";
    std::cout << "setup_ms "     << ms_since(t_setup0, t_setup1) << "\n";
    std::cout << "encrypt_ms "   << ms_since(t_enc0, t_enc1) << "\n";
    std::cout << "fhe_ms "       << ms_since(t_fhe0, t_fhe1) << "\n";
    std::cout << "decrypt_ms "   << ms_since(t_dec0, t_dec1) << "\n";
    std::cout << "read_ms "      << ms_since(t_io0, t_io1) << "\n";
    std::cout << "write_ms "     << ms_since(t_wrt0, t_wrt1) << "\n";
    std::cout << "rel_err_q "    << std::scientific << err_q << std::defaultfloat << "\n";
    std::cout << "rel_err_k "    << std::scientific << err_k << std::defaultfloat << "\n";
    std::cout << "rel_err_v "    << std::scientific << err_v << std::defaultfloat << "\n";
    std::cout << "mse_q "        << std::scientific << mse_q << std::defaultfloat << "\n";
    std::cout << "mse_k "        << std::scientific << mse_k << std::defaultfloat << "\n";
    std::cout << "mse_v "        << std::scientific << mse_v << std::defaultfloat << "\n";
    std::cout << "ks_rots "      << ks.rots << "\n";
    std::cout << "ks_conj "      << ks.conj << "\n";
    std::cout << "galois_elts "  << galois_elts.size() << "\n";

    return 0;
}
