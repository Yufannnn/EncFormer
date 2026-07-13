// pipe_ckks_qkv_score.cu — Native CKKS QKV projection + FDP Score (combined).
//
// Q and K stay encrypted between QKV projection and Score computation —
// no decrypt/re-encrypt boundary.  V is decrypted and output for the Value stage.
//
// Reads:  PIPE_DIR/qkv_in.bin = X[M×D] ++ WQ[D×D] ++ WK[D×D] ++ WV[D×D]
//                                ++ bQ[D] ++ bK[D] ++ bV[D]    (row-major float64)
// Writes: PIPE_DIR/score_out.bin = S_heads[H×M×M]              (row-major float64)
//         PIPE_DIR/v_out.bin     = V[M×D]                       (row-major float64)
//
// QKV uses FDP column permutation for Q,K (c_used=96, blocks=8) and standard for V
// (c_used=128, blocks=6).  Score implements the folded dot product (FDP) algorithm:
//   mapf → mk_bank → emit_f → red_h → pack_f → unpack_f.
//
// Scale management:
//   encrypt at chain_index=2 → linear multiply at scale_w → scale=2^80
//   → rescale Q,K to chain_index=1 (scale≈2^40) before Score
//   → all Score masks at scale 1.0 (no level consumed)
//   → ct-ct multiply in emit_f gives scale≈2^80 at chain_index=1
//   → decrypt

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
constexpr int NSLOTS     = EF_NSLOTS;
constexpr int M          = EF_M;
constexpr int C          = EF_C;
constexpr int D          = EF_D;
constexpr int H          = EF_H;
constexpr int DH         = EF_DH;
constexpr int N1         = EF_N1_DEFAULT;
constexpr int N2         = EF_N2_DEFAULT;
constexpr int HP         = EF_HP_QKV;
constexpr int C_USED_LIN = EF_C_USED_LIN;

constexpr int C_USED_QK  = EF_C_USED_QK;
constexpr int BLOCKS_QK  = EF_BLOCKS_QK;
constexpr int C_USED_V   = EF_C_USED_V;
constexpr int BLOCKS_V   = EF_BLOCKS_V;

constexpr int B_FOLD     = EF_B_FOLD;
constexpr int G_FOLD     = EF_G_FOLD;
constexpr int HALF_M     = EF_HALF_M;
constexpr int BLEN       = EF_BLEN;

constexpr size_t CHAIN_IDX_TOP = 2;     // encrypt here

// =========================================================================
// Score helpers
// =========================================================================

struct MapEntry { int j; int s; };

void mapf_compute(MapEntry mp[HALF_M]) {
    bool seen[HALF_M] = {};
    int count = 0;
    for (int j = 0; j < G_FOLD && count < HALF_M; j++)
        for (int s = 0; s < B_FOLD && count < HALF_M; s++) {
            int t = ((j * B_FOLD - s) % M + M) % M;
            if (t < HALF_M && !seen[t]) {
                seen[t] = true;
                mp[t] = {j, s};
                count++;
            }
        }
}

// rot_within: rotate within M-length segments with proper-scale masks.
// mask_value: pcplx(1,0) for real masks, pcplx(0,1) for imaginary masks.
// All entries (including t=0) get a mask multiply for uniform scale.
PhantomCiphertext rot_within_ct(
    const PhantomContext &ctx, const PhantomGaloisKey &gk,
    PhantomCKKSEncoder &encoder,
    const PhantomCiphertext &ct,
    int t, int c_used_segs, double scale_mask,
    std::map<int, PhantomCiphertext> &rot_cache,
    KSCounters &ks,
    pc64 mask_value = pcplx(1.0, 0.0))
{
    t = ((t % M) + M) % M;
    const size_t slots = encoder.slot_count();
    size_t ci = ct.chain_index();

    if (t == 0) {
        // Identity mask at scale_mask for uniform scale
        std::vector<pc64> id(slots, pcplx(0.0));
        for (int seg = 0; seg < c_used_segs; seg++)
            for (int i = 0; i < M; i++)
                id[(size_t)seg * M + i] = mask_value;
        PhantomPlaintext id_pt;
        encoder.encode(ctx, id, scale_mask, id_pt, ci);
        PhantomCiphertext out = ct;
        multiply_plain_inplace(ctx, out, id_pt);
        return out;
    }

    // Get or create rotations
    auto get_rot = [&](int step) -> const PhantomCiphertext & {
        auto it = rot_cache.find(step);
        if (it == rot_cache.end()) {
            PhantomCiphertext R = ct;
            rotate_inplace(ctx, R, norm_step(step, NSLOTS), gk);
            ks.rots++;
            it = rot_cache.emplace(step, std::move(R)).first;
        }
        return it->second;
    };
    const PhantomCiphertext &R1 = get_rot(t);
    const PhantomCiphertext &R2 = get_rot(t - M);

    // Masks at scale_mask for accurate encoding
    std::vector<pc64> hm(slots, pcplx(0.0)), tm(slots, pcplx(0.0));
    for (int seg = 0; seg < c_used_segs; seg++)
        for (int i = 0; i < M; i++) {
            size_t idx = (size_t)seg * M + i;
            if (i < M - t) hm[idx] = mask_value;
            else           tm[idx] = mask_value;
        }

    PhantomPlaintext h_pt, t_pt;
    encoder.encode(ctx, hm, scale_mask, h_pt, ci);
    encoder.encode(ctx, tm, scale_mask, t_pt, ci);

    PhantomCiphertext r1 = R1, r2 = R2;
    multiply_plain_inplace(ctx, r1, h_pt);
    multiply_plain_inplace(ctx, r2, t_pt);
    add_inplace(ctx, r1, r2);
    return r1;
}

// mk_bank: build Q_bank[bid][s], K_bank[bid][j], K_bank_h[bid][j]
// K_bank_h absorbs the imaginary unit into its masks, so
// combo = K_bank[j] + K_bank_h[j] (no separate i_pt multiply needed).
void mk_bank(
    const PhantomContext &ctx, const PhantomGaloisKey &gk,
    PhantomCKKSEncoder &encoder,
    const std::vector<PhantomCiphertext> &Q_cts,
    const std::vector<PhantomCiphertext> &K_cts,
    int c_used, double scale_mask,
    std::vector<std::vector<PhantomCiphertext>> &Q_bank,
    std::vector<std::vector<PhantomCiphertext>> &K_bank,
    std::vector<std::vector<PhantomCiphertext>> &K_bank_h,
    KSCounters &ks)
{
    const int blocks = static_cast<int>(Q_cts.size());
    Q_bank.resize(blocks);
    K_bank.resize(blocks);
    K_bank_h.resize(blocks);

    const pc64 real_one = pcplx(1.0, 0.0);
    const pc64 imag_one = pcplx(0.0, 1.0);

    for (int bid = 0; bid < blocks; bid++) {
        Q_bank[bid].resize(B_FOLD);
        std::map<int, PhantomCiphertext> qcache;
        for (int s = 0; s < B_FOLD; s++)
            Q_bank[bid][s] = rot_within_ct(ctx, gk, encoder, Q_cts[bid],
                                            s, c_used, scale_mask, qcache, ks, real_one);

        K_bank[bid].resize(G_FOLD);
        K_bank_h[bid].resize(G_FOLD);
        std::map<int, PhantomCiphertext> kcache;
        for (int j = 0; j < G_FOLD; j++) {
            int rot_k  = j * B_FOLD;
            int rot_kh = ((j + G_FOLD / 2) % G_FOLD) * B_FOLD;
            K_bank[bid][j] = rot_within_ct(ctx, gk, encoder, K_cts[bid],
                                            rot_k, c_used, scale_mask, kcache, ks, real_one);
            K_bank_h[bid][j] = rot_within_ct(ctx, gk, encoder, K_cts[bid],
                                              rot_kh, c_used, scale_mask, kcache, ks, imag_one);
        }
    }
}

// emit_f: for each half-diagonal t, compute ct-ct multiply across blocks.
// K_bank_h already has imaginary masks, so combo = K_bank + K_bank_h directly.
void emit_f(
    const PhantomContext &ctx, const PhantomRelinKey &rk,
    const std::vector<std::vector<PhantomCiphertext>> &Q_bank,
    const std::vector<std::vector<PhantomCiphertext>> &K_bank,
    const std::vector<std::vector<PhantomCiphertext>> &K_bank_h,
    const MapEntry mp_half[HALF_M],
    std::vector<PhantomCiphertext> &D_fold,
    KSCounters &ks)
{
    const int blocks = static_cast<int>(Q_bank.size());
    D_fold.resize(HALF_M);

    for (int t = 0; t < HALF_M; t++) {
        int j = mp_half[t].j;
        int s = mp_half[t].s;
        PhantomCiphertext acc;
        bool inited = false;

        for (int bid = 0; bid < blocks; bid++) {
            // combo = K_bank[j] + K_bank_h[j]  (imaginary already in K_bank_h masks)
            PhantomCiphertext combo = K_bank[bid][j];
            add_inplace(ctx, combo, K_bank_h[bid][j]);

            // term = Q_bank[s] * combo  (ct-ct multiply with relin)
            PhantomCiphertext term = Q_bank[bid][s];
            multiply_and_relin_inplace(ctx, term, combo, rk);
            ks.muls_ctct++;

            if (!inited) { acc = std::move(term); inited = true; }
            else add_inplace(ctx, acc, term);
        }
        D_fold[t] = std::move(acc);
    }
}

// Segment mask helpers
inline std::vector<pc64> msk_rng(size_t slots, int m, int seg_lo, int seg_hi) {
    std::vector<pc64> mask(slots, pcplx(0.0));
    for (int c = seg_lo; c < seg_hi; c++)
        for (int i = 0; i < m; i++)
            mask[(size_t)c * m + i] = pcplx(1.0);
    return mask;
}

// red_h: sum-of-rotations reduction across head-dimension columns → first H segments.
// Uses (U-1) rotations + 1 final mask (at scale_mask) instead of tree with intermediate masks.
PhantomCiphertext red_h(
    const PhantomContext &ctx, const PhantomGaloisKey &gk,
    PhantomCKKSEncoder &encoder,
    PhantomCiphertext Z, int c_used, double scale_mask,
    KSCounters &ks)
{
    const size_t slots = encoder.slot_count();
    size_t ci = Z.chain_index();

    // Sum all U rotations of the ORIGINAL Z (not the accumulated version)
    int U = (c_used + H - 1) / H;  // 8 for c_used=96, H=12
    PhantomCiphertext Z_orig = Z;
    for (int delta = 1; delta < U; delta++) {
        int shift = delta * H * M;
        PhantomCiphertext Rd = Z_orig;
        rotate_inplace(ctx, Rd, norm_step(shift, NSLOTS), gk);
        ks.rots++;
        add_inplace(ctx, Z, Rd);
    }

    // Final mask to keep only first H segments
    auto h_mask = msk_rng(slots, M, 0, H);
    PhantomPlaintext hpt;
    encoder.encode(ctx, h_mask, scale_mask, hpt, ci);
    multiply_plain_inplace(ctx, Z, hpt);

    return Z;
}

// unpack_f: decrypt D_fold entries directly → S[H][M][M].
// Skips pack_f entirely — decrypts 64 cts instead of packing into 6.
// Eliminates scale mismatch issues and saves ~60 rotations.
void unpack_f(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &D_fold,
    const MapEntry mp_half[HALF_M],
    std::vector<double> &S_out)
{
    S_out.assign((size_t)H * M * M, 0.0);

    for (int t = 0; t < HALF_M; t++) {
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, D_fold[t], pt);
        encoder.decode(ctx, pt, dec);

        int s = mp_half[t].s;

        // D_fold[t] has useful data in the first H segments (after red_h)
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < M; i++) {
                size_t slot_idx = (size_t)h * M + i;
                double re = preal(dec[slot_idx]);
                double im = pimag(dec[slot_idx]);

                int dst_i = (i + s) % M;
                int col_t  = (dst_i + t) % M;
                int col_th = (dst_i + HALF_M + t) % M;
                S_out[(size_t)h * M * M + dst_i * M + col_t]  += re;
                S_out[(size_t)h * M * M + dst_i * M + col_th] += im;
            }
        }
    }
}

} // anonymous namespace

// =========================================================================
// Main
// =========================================================================

int main() {
    constexpr size_t NPOLY = static_cast<size_t>(NSLOTS) * 2;
    constexpr uint64_t MMOD = 2ull * static_cast<uint64_t>(NPOLY);
    constexpr uint32_t GEN_ROT = 5u;

    const double scale_in  = std::pow(2.0, 40);
    const double scale_w   = std::pow(2.0, 40);
    const double scale_mul = scale_in * scale_w;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "failed to set cuda device\n";
        return 1;
    }

    // --- Read input data ---
    auto t_io0 = std::chrono::high_resolution_clock::now();
    std::string pd = pipe_io::get_pipe_dir();
    std::string in_path = pd + "/qkv_in.bin";

    constexpr size_t N_X = (size_t)M * D;
    constexpr size_t N_W = (size_t)D * D;
    constexpr size_t N_b = D;
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

    // --- FDP permutation ---
    auto perm = perm_fdp(H, DH);
    auto WQ_fdp = permute_cols(WQ, D, D, perm);
    auto WK_fdp = permute_cols(WK, D, D, perm);
    std::vector<double> bQ_fdp(D), bK_fdp(D);
    for (int i = 0; i < D; i++) {
        bQ_fdp[i] = bQ[perm[i]];
        bK_fdp[i] = bK[perm[i]];
    }

    // --- CKKS context setup ---
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);

    auto galois_cols  = build_galois_elts_full_columns(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_score = build_galois_elts_score(NSLOTS, M, B_FOLD, G_FOLD,
                                                 C_USED_QK, H, GEN_ROT, MMOD);
    auto galois_elts = merge_galois_elts(galois_cols, galois_score);
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    (void)pk;
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);
    PhantomCKKSEncoder encoder(ctx);
    cuda_sync();
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);
    KSCounters ks;

    // --- Encrypt input at chain_index=2 ---
    auto t_enc0 = std::chrono::high_resolution_clock::now();
    auto x_ct = encrypt_packed_pairs(ctx, encoder, sk, X, M, D, HP, NSLOTS,
                                      scale_in, CHAIN_IDX_TOP, C_USED_LIN);
    auto ct_zero_mul = make_ct_zero_mul(ctx, encoder, sk, scale_mul, CHAIN_IDX_TOP);
    cuda_sync();
    auto t_enc1 = std::chrono::high_resolution_clock::now();

    // --- Weight tables ---
    auto WQ_tab = build_wtab_paired(WQ_fdp.data(), D, D, C, C_USED_QK, BLOCKS_QK, C_USED_LIN);
    auto WK_tab = build_wtab_paired(WK_fdp.data(), D, D, C, C_USED_QK, BLOCKS_QK, C_USED_LIN);
    auto WV_tab = build_wtab_paired(WV, D, D, C, -1, -1, C_USED_LIN);

    // --- QKV projection at chain_index=2 ---
    auto t_qkv0 = std::chrono::high_resolution_clock::now();

    auto babies = build_babies(ctx, gk, x_ct, N1, M, NSLOTS, ks);

    auto Q_ct = linear_complex_paired(ctx, encoder, gk, babies, WQ_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul, ks, BLOCKS_QK, CHAIN_IDX_TOP);
    auto K_ct = linear_complex_paired(ctx, encoder, gk, babies, WK_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul, ks, BLOCKS_QK, CHAIN_IDX_TOP);
    auto V_ct = linear_complex_paired(ctx, encoder, gk, babies, WV_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul, ks, -1, CHAIN_IDX_TOP);

    // ct_real (conjugate trick)
    ct_real_blocks(ctx, gk, Q_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, K_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, V_ct, static_cast<size_t>(conj_elt), ks);

    // Add bias to Q, K in encrypted domain (at chain_index=2, scale=2^80)
    add_bias_blocks(ctx, encoder, Q_ct, bQ_fdp.data(), D, M, NSLOTS, C_USED_QK,
                    scale_mul, CHAIN_IDX_TOP);
    add_bias_blocks(ctx, encoder, K_ct, bK_fdp.data(), D, M, NSLOTS, C_USED_QK,
                    scale_mul, CHAIN_IDX_TOP);

    // Rescale Q, K from chain_index=2 → chain_index=1 (scale ≈ 2^40)
    for (auto &ct : Q_ct) rescale_to_next_inplace(ctx, ct);
    for (auto &ct : K_ct) rescale_to_next_inplace(ctx, ct);

    cuda_sync();
    auto t_qkv1 = std::chrono::high_resolution_clock::now();

    // --- Decrypt V (stays at chain_index=2, no Score needed) ---
    auto t_vdec0 = std::chrono::high_resolution_clock::now();
    auto V_out = decrypt_blocks(ctx, encoder, sk, V_ct, M, D, NSLOTS);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            V_out[(size_t)i * D + j] += bV[j];
    auto t_vdec1 = std::chrono::high_resolution_clock::now();

    // ===================== Score (FDP) =====================
    // Q, K are now at chain_index=1, scale ≈ 2^40
    auto t_score0 = std::chrono::high_resolution_clock::now();

    MapEntry mp_half[HALF_M];
    mapf_compute(mp_half);

    // Build rotation banks (masks at scale_w for accurate encoding)
    std::vector<std::vector<PhantomCiphertext>> Q_bank, K_bank, K_bank_h;
    mk_bank(ctx, gk, encoder, Q_ct, K_ct, C_USED_QK, scale_w,
            Q_bank, K_bank, K_bank_h, ks);

    // emit_f: ct-ct multiply (K_bank_h has imaginary absorbed in masks)
    std::vector<PhantomCiphertext> D_fold;
    emit_f(ctx, rk, Q_bank, K_bank, K_bank_h, mp_half, D_fold, ks);

    // red_h per fold entry (sum-of-rotations + final mask at scale_w)
    for (int t = 0; t < HALF_M; t++)
        D_fold[t] = red_h(ctx, gk, encoder, std::move(D_fold[t]), C_USED_QK, scale_w, ks);

    cuda_sync();
    auto t_score1 = std::chrono::high_resolution_clock::now();

    // --- Decrypt and unpack Score (directly from D_fold, no pack_f) ---
    auto t_sdec0 = std::chrono::high_resolution_clock::now();
    std::vector<double> S_out;
    unpack_f(ctx, encoder, sk, D_fold, mp_half, S_out);
    auto t_sdec1 = std::chrono::high_resolution_clock::now();

    // --- Write outputs ---
    auto t_wrt0 = std::chrono::high_resolution_clock::now();
    pipe_io::write_f64((pd + "/score_out.bin").c_str(), S_out.data(), S_out.size());
    pipe_io::write_f64((pd + "/v_out.bin").c_str(), V_out.data(), V_out.size());
    auto t_wrt1 = std::chrono::high_resolution_clock::now();

    // --- Verify ---
    auto Q_ref = matmul_ref(X, M, D, WQ_fdp.data(), D);
    auto K_ref = matmul_ref(X, M, D, WK_fdp.data(), D);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++) {
            Q_ref[(size_t)i * D + j] += bQ_fdp[j];
            K_ref[(size_t)i * D + j] += bK_fdp[j];
        }

    auto Q_dec = decrypt_blocks_cusued(ctx, encoder, sk, Q_ct, M, D, NSLOTS, C_USED_QK);
    auto K_dec = decrypt_blocks_cusued(ctx, encoder, sk, K_ct, M, D, NSLOTS, C_USED_QK);
    double err_q = rel_err_real(Q_dec, Q_ref);
    double err_k = rel_err_real(K_dec, K_ref);
    double mse_q = mse_real(Q_dec, Q_ref);
    double mse_k = mse_real(K_dec, K_ref);

    auto V_ref = matmul_ref(X, M, D, WV, D);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            V_ref[(size_t)i * D + j] += bV[j];
    double err_v = rel_err_real(V_out, V_ref);
    double mse_v = mse_real(V_out, V_ref);

    // Score reference: in FDP order, head h uses columns h, h+H, h+2H, ...
    std::vector<double> S_ref((size_t)H * M * M, 0.0);
    for (int h = 0; h < H; h++)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++) {
                double acc = 0.0;
                for (int u = 0; u < DH; u++) {
                    int col = u * H + h;
                    acc += Q_ref[(size_t)i * D + col] * K_ref[(size_t)j * D + col];
                }
                S_ref[(size_t)h * M * M + i * M + j] = acc;
            }
    double err_s = rel_err_real(S_out, S_ref);
    double mse_s = mse_real(S_out, S_ref);

    // --- Print metrics ---
    std::cout << std::setprecision(10);
    std::cout << "stage pipe_qkv_score\n";
    std::cout << "setup_ms "     << ms_since(t_setup0, t_setup1) << "\n";
    std::cout << "encrypt_ms "   << ms_since(t_enc0, t_enc1) << "\n";
    std::cout << "qkv_fhe_ms "   << ms_since(t_qkv0, t_qkv1) << "\n";
    std::cout << "score_fhe_ms " << ms_since(t_score0, t_score1) << "\n";
    std::cout << "decrypt_ms "   << ms_since(t_vdec0, t_vdec1) + ms_since(t_sdec0, t_sdec1) << "\n";
    std::cout << "read_ms "      << ms_since(t_io0, t_io1) << "\n";
    std::cout << "write_ms "     << ms_since(t_wrt0, t_wrt1) << "\n";
    std::cout << "rel_err_q "    << std::scientific << err_q << std::defaultfloat << "\n";
    std::cout << "rel_err_k "    << std::scientific << err_k << std::defaultfloat << "\n";
    std::cout << "rel_err_v "    << std::scientific << err_v << std::defaultfloat << "\n";
    std::cout << "rel_err_s "    << std::scientific << err_s << std::defaultfloat << "\n";
    std::cout << "mse_q "        << std::scientific << mse_q << std::defaultfloat << "\n";
    std::cout << "mse_k "        << std::scientific << mse_k << std::defaultfloat << "\n";
    std::cout << "mse_v "        << std::scientific << mse_v << std::defaultfloat << "\n";
    std::cout << "mse_s "        << std::scientific << mse_s << std::defaultfloat << "\n";
    std::cout << "ks_rots "      << ks.rots << "\n";
    std::cout << "ks_muls "      << ks.muls_ctct << "\n";
    std::cout << "ks_conj "      << ks.conj << "\n";
    std::cout << "galois_elts "  << galois_elts.size() << "\n";

    return 0;
}
