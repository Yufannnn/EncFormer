// pipe_ckks_value_out.cu — Native CKKS Value (SV multiply) + OUT projection.
//
// Reads:  PIPE_DIR/value_in.bin = A_heads[H×M×M] ++ V[M×D] ++ W_O[D×D] ++ bO[D]
// Writes: PIPE_DIR/value_out.bin = Z[M×D]  (row-major float64)
//
// FHE SV: encrypt V → diagonal decomposition (rot_within + A_diag pt multiply)
//         → decrypt Y.
// FHE OUT: encrypt Y (complex-paired) → linear_complex_paired → decrypt Z.
//
// Both passes share a single CKKS context with merged Galois elements
// (within-row for SV + column rotations for OUT).

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

#include "example.h"
#include "pipe_io.h"
#include "pipe_ckks_common.h"

using namespace pipe_ckks;

namespace {
#include "model_config.h"
constexpr int NSLOTS = EF_NSLOTS;
constexpr int M      = EF_M;
constexpr int C      = EF_C;
constexpr int N1_OUT = EF_N1_DEFAULT;
constexpr int N2_OUT = EF_N2_DEFAULT;
constexpr int D      = EF_D;
constexpr int H      = EF_H;
constexpr int DH     = EF_DH;
constexpr int HP         = EF_HP_OUT;
constexpr int BLOCKS     = EF_BLOCKS;
constexpr int G_SV       = EF_G_SV;
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
    std::string in_path = pd + "/value_in.bin";

    constexpr size_t N_A  = (size_t)H * M * M;
    constexpr size_t N_V  = (size_t)M * D;
    constexpr size_t N_WO = (size_t)D * D;
    constexpr size_t N_bO = D;
    constexpr size_t TOTAL = N_A + N_V + N_WO + N_bO;
    std::vector<double> buf(TOTAL);
    pipe_io::read_f64(in_path.c_str(), buf.data(), buf.size());

    const double *A_flat = buf.data();
    const double *V      = A_flat + N_A;
    const double *WO     = V + N_V;
    const double *bO     = WO + N_WO;
    auto t_io1 = std::chrono::high_resolution_clock::now();

    // --- CKKS context setup (merged Galois elements for SV + OUT) ---
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);

    // SV needs within-row rotations; OUT needs column rotations
    auto galois_sv  = build_galois_elts_within_row(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_out = build_galois_elts_full_columns(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_elts = merge_galois_elts(galois_sv, galois_out);
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    (void)pk;
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomCKKSEncoder encoder(ctx);
    cuda_sync();
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    const size_t slots = encoder.slot_count();
    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);
    KSCounters ks;

    // ==================== Pass 1: FHE SV multiply ====================
    // Encrypt V into G_SV=6 real-valued block ciphertexts
    auto t_sv0 = std::chrono::high_resolution_clock::now();
    auto v_ct = encrypt_real_blocks(ctx, encoder, sk, V, M, D, NSLOTS, scale_in);
    cuda_sync();

    // Precompute rot_within masks (head/tail) as plaintexts
    // rot_within(-t): t_param = (-t % M + M) % M
    //   s1 = t_param,  s2 = t_param - M
    //   head_mask: 1 for pos [0, M-t_param),  tail_mask: 1 for pos [M-t_param, M)
    // Identity mask for t_param=0 (all 1s) to ensure uniform scale across all rotations
    PhantomPlaintext identity_mask_pt;
    {
        std::vector<pc64> ones(slots, pcplx(1.0));
        encoder.encode(ctx, ones, scale_w, identity_mask_pt, 1);
    }

    std::vector<PhantomPlaintext> head_mask_pt(M), tail_mask_pt(M);
    for (int tp = 1; tp < M; tp++) {
        std::vector<pc64> hm(slots, pcplx(0.0)), tm(slots, pcplx(0.0));
        for (int c_seg = 0; c_seg < C; c_seg++)
            for (int i = 0; i < M; i++) {
                size_t idx = (size_t)c_seg * M + i;
                if (i < M - tp) hm[idx] = pcplx(1.0);
                else            tm[idx] = pcplx(1.0);
            }
        encoder.encode(ctx, hm, scale_w, head_mask_pt[tp], 1);
        encoder.encode(ctx, tm, scale_w, tail_mask_pt[tp], 1);
    }

    // SV computation: Y_g[i,c] = sum_t A_h[i,(i-t)%M] * V[(i-t)%M, g*C+c]
    // For each group g, iterate over diagonals t=0..M-1
    std::vector<double> Y_sv((size_t)M * D, 0.0);

    for (int g = 0; g < G_SV; g++) {
        // Precompute all rot_within results for V_ct[g]
        std::vector<PhantomCiphertext> v_rot(M);
        v_rot[0] = v_ct[g];
        // Apply identity mask so v_rot[0] has same scale as masked entries
        multiply_plain_inplace(ctx, v_rot[0], identity_mask_pt);
        for (int orig_t = 1; orig_t < M; orig_t++) {
            int tp = (M - orig_t) % M;  // t_param for rot_within(-orig_t)
            int s1 = tp;
            int s2 = tp - M;

            PhantomCiphertext R1 = v_ct[g];
            rotate_inplace(ctx, R1, norm_step(s1, NSLOTS), gk);
            ks.rots++;

            PhantomCiphertext R2 = v_ct[g];
            rotate_inplace(ctx, R2, norm_step(s2, NSLOTS), gk);
            ks.rots++;

            multiply_plain_inplace(ctx, R1, head_mask_pt[tp]);
            multiply_plain_inplace(ctx, R2, tail_mask_pt[tp]);
            add_inplace(ctx, R1, R2);
            v_rot[orig_t] = R1;
        }

        // Accumulate: for each diagonal t, encode A_diag and multiply
        PhantomCiphertext acc;
        bool acc_inited = false;

        for (int t = 0; t < M; t++) {
            // Encode A diagonal t for this group's heads
            std::vector<pc64> a_msg(slots, pcplx(0.0));
            for (int c_seg = 0; c_seg < C; c_seg++) {
                int head_idx = (g * C + c_seg) / DH;
                if (head_idx >= H) continue;
                const double *Ah = A_flat + (size_t)head_idx * M * M;
                for (int i = 0; i < M; i++) {
                    int k = ((i - t) % M + M) % M;
                    a_msg[(size_t)c_seg * M + i] = pcplx(Ah[(size_t)i * M + k], 0.0);
                }
            }
            PhantomPlaintext a_pt;
            encoder.encode(ctx, a_msg, scale_w, a_pt, 1);

            PhantomCiphertext term = v_rot[t];
            multiply_plain_inplace(ctx, term, a_pt);

            if (!acc_inited) { acc = term; acc_inited = true; }
            else add_inplace(ctx, acc, term);
        }

        // Decrypt this group's SV result (scale = scale_in * scale_w^2 from
        // two chained multiply_plain: mask + A_diag for t>0, or just A_diag for t=0).
        // For t=0 (no mask), scale = scale_in * scale_w (one multiply).
        // For t>0, scale = scale_in * scale_w * scale_w (two multiplies).
        // Since t=0 is added to t>0 terms which have higher scale, Phantom may
        // handle the mismatch. If not, we'll rescale uniformly.
        // Actually, v_rot[0] is unmasked (scale_in), while v_rot[t>0] is masked
        // (scale_in * scale_w). After A_diag multiply, t=0 → scale_in*scale_w,
        // t>0 → scale_in*scale_w*scale_w. These have different scales.

        // To fix: apply an identity mask to v_rot[0] so all have same scale.
        // Alternatively, just decrypt and verify. Phantom may auto-manage.

        // For now, decrypt the accumulated result
        PhantomPlaintext dec_pt;
        std::vector<pc64> dec_vec;
        sk.decrypt(ctx, acc, dec_pt);
        encoder.decode(ctx, dec_pt, dec_vec);

        for (int c_seg = 0; c_seg < C; c_seg++) {
            int abs_col = g * C + c_seg;
            if (abs_col >= D) break;
            for (int i = 0; i < M; i++)
                Y_sv[(size_t)i * D + abs_col] = preal(dec_vec[(size_t)c_seg * M + i]);
        }
    }
    cuda_sync();
    auto t_sv1 = std::chrono::high_resolution_clock::now();

    // ==================== Pass 2: FHE OUT projection ====================
    auto t_enc0 = std::chrono::high_resolution_clock::now();
    auto y_ct = encrypt_packed_pairs(ctx, encoder, sk, Y_sv.data(), M, D, HP, NSLOTS,
                                      scale_in, 1, C_USED_LIN);
    auto ct_zero_mul = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();
    auto t_enc1 = std::chrono::high_resolution_clock::now();

    auto WO_tab = build_wtab_paired(WO, D, D, C, -1, -1, C_USED_LIN);

    auto t_fhe0 = std::chrono::high_resolution_clock::now();
    auto babies = build_babies(ctx, gk, y_ct, N1_OUT, M, NSLOTS, ks);
    auto Z_ct = linear_complex_paired(ctx, encoder, gk, babies, WO_tab,
                                       D, M, N1_OUT, N2_OUT, C, NSLOTS,
                                       scale_w, ct_zero_mul, ks);
    ct_real_blocks(ctx, gk, Z_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();
    auto t_fhe1 = std::chrono::high_resolution_clock::now();

    // --- Decrypt OUT ---
    auto t_dec0 = std::chrono::high_resolution_clock::now();
    auto Z_out = decrypt_blocks(ctx, encoder, sk, Z_ct, M, D, NSLOTS);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            Z_out[(size_t)i * D + j] += bO[j];
    auto t_dec1 = std::chrono::high_resolution_clock::now();

    // --- Write output ---
    auto t_wrt0 = std::chrono::high_resolution_clock::now();
    std::string out_path = pd + "/value_out.bin";
    pipe_io::write_f64(out_path.c_str(), Z_out.data(), Z_out.size());
    auto t_wrt1 = std::chrono::high_resolution_clock::now();

    // --- Verify against plaintext reference ---
    // SV reference
    std::vector<double> Y_ref((size_t)M * D, 0.0);
    for (int h = 0; h < H; h++) {
        const double *Ah = A_flat + (size_t)h * M * M;
        for (int i = 0; i < M; i++)
            for (int u = 0; u < DH; u++) {
                double a = 0.0;
                for (int k = 0; k < M; k++)
                    a += Ah[(size_t)i * M + k] * V[(size_t)k * D + h * DH + u];
                Y_ref[(size_t)i * D + h * DH + u] = a;
            }
    }
    double err_sv = rel_err_real(Y_sv, Y_ref);
    double mse_sv = mse_real(Y_sv, Y_ref);

    // OUT reference
    auto Z_ref = matmul_ref(Y_sv.data(), M, D, WO, D);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            Z_ref[(size_t)i * D + j] += bO[j];
    double err_out = rel_err_real(Z_out, Z_ref);
    double mse_out = mse_real(Z_out, Z_ref);

    // --- Print metrics ---
    std::cout << std::setprecision(10);
    std::cout << "stage pipe_value_out\n";
    std::cout << "setup_ms "     << ms_since(t_setup0, t_setup1) << "\n";
    std::cout << "sv_fhe_ms "    << ms_since(t_sv0, t_sv1) << "\n";
    std::cout << "encrypt_ms "   << ms_since(t_enc0, t_enc1) << "\n";
    std::cout << "out_fhe_ms "   << ms_since(t_fhe0, t_fhe1) << "\n";
    std::cout << "decrypt_ms "   << ms_since(t_dec0, t_dec1) << "\n";
    std::cout << "read_ms "      << ms_since(t_io0, t_io1) << "\n";
    std::cout << "write_ms "     << ms_since(t_wrt0, t_wrt1) << "\n";
    std::cout << "rel_err_sv "   << std::scientific << err_sv << std::defaultfloat << "\n";
    std::cout << "rel_err_out "  << std::scientific << err_out << std::defaultfloat << "\n";
    std::cout << "mse_sv "       << std::scientific << mse_sv << std::defaultfloat << "\n";
    std::cout << "mse_out "      << std::scientific << mse_out << std::defaultfloat << "\n";
    std::cout << "ks_rots "      << ks.rots << "\n";
    std::cout << "ks_conj "      << ks.conj << "\n";
    std::cout << "galois_elts "  << galois_elts.size() << "\n";

    return 0;
}
