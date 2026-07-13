// pipe_ckks_layer.cu — Full transformer layer in a single binary with bridge protocol.
//
// Single CKKS context + keygen covering all stages (QKV, Score, Value+OUT, FF1, FF2).
// Uses the server-side CKKS<->MPC bridge protocol for transitions to/from Python MPC.
// V stays encrypted in GPU memory between QKV and Value (never decrypted).
//
// Reads:  PIPE_DIR/layer_in.bin = X[M×D] ++ WQ[D×D] ++ WK[D×D] ++ WV[D×D]
//                                  ++ bQ[D] ++ bK[D] ++ bV[D] ++ W_O[D×D] ++ bO[D]
//                                  ++ W1[D×D_FF] ++ W2[D_FF×D]
//
// Phase 1: QKV + Score
//   Writes: score_masked.bin, score_server_share.bin, touches phase1_done
//   Waits:  softmax_client_share.bin, softmax_server_share.bin, softmax_done
//
// Phase 2: Value + OUT
//   Writes: z_masked.bin, z_server_share.bin, touches phase2_done
//   Waits:  ln1_client_share.bin, ln1_server_share.bin, ln1_done
//
// Phase 3: FF1
//   Writes: ff1_masked.bin, ff1_server_share.bin, touches phase3_done
//   Waits:  gelu_client_share.bin, gelu_server_share.bin, gelu_done
//
// Phase 4: FF2
//   Writes: ff2_masked.bin, ff2_server_share.bin, touches phase4_done

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sys/stat.h>
#include <unistd.h>

#include "example.h"
#include "pipe_io.h"
#include "pipe_ckks_common.h"
#include "pipe_bridge.h"

using namespace pipe_ckks;

namespace {

#include "model_config.h"
constexpr int NSLOTS     = EF_NSLOTS;
constexpr int M          = EF_M;
constexpr int C          = EF_C;
constexpr int D          = EF_D;
constexpr int H          = EF_H;
constexpr int DH         = EF_DH;
constexpr int D_FF       = EF_D_FF;
constexpr int HP         = EF_HP_QKV;
constexpr int C_USED_LIN = EF_C_USED_LIN;

// QKV BSGS
constexpr int N1         = EF_N1_DEFAULT;
constexpr int N2         = EF_N2_DEFAULT;
constexpr int C_USED_QK  = EF_C_USED_QK;
constexpr int BLOCKS_QK  = EF_BLOCKS_QK;

// Score FDP
constexpr int B_FOLD     = EF_B_FOLD;
constexpr int G_FOLD     = EF_G_FOLD;
constexpr int HALF_M     = EF_HALF_M;

// Value+OUT
constexpr int G_SV       = EF_G_SV;

// FF1/FF2
constexpr int HP_FF1     = EF_HP_FF1;
constexpr int HP_FF2     = EF_HP_FF2;
constexpr int C_IN_FF1   = EF_C_USED_FF1_IN;
constexpr int C_IN_FF2   = EF_C_USED_FF2_IN;
constexpr int N1_FF2     = EF_N1_FF2;
constexpr int N2_FF2     = EF_N2_FF2;

constexpr size_t CHAIN_IDX_TOP = 2;  // QKV encrypt level

// =========================================================================
// Score helpers (same as pipe_ckks_attn.cu)
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
            PhantomCiphertext combo = K_bank[bid][j];
            add_inplace(ctx, combo, K_bank_h[bid][j]);

            PhantomCiphertext term = Q_bank[bid][s];
            multiply_and_relin_inplace(ctx, term, combo, rk);
            ks.muls_ctct++;

            if (!inited) { acc = std::move(term); inited = true; }
            else add_inplace(ctx, acc, term);
        }
        D_fold[t] = std::move(acc);
    }
}

inline std::vector<pc64> msk_rng(size_t slots, int m, int seg_lo, int seg_hi) {
    std::vector<pc64> mask(slots, pcplx(0.0));
    for (int c = seg_lo; c < seg_hi; c++)
        for (int i = 0; i < m; i++)
            mask[(size_t)c * m + i] = pcplx(1.0);
    return mask;
}

PhantomCiphertext red_h(
    const PhantomContext &ctx, const PhantomGaloisKey &gk,
    PhantomCKKSEncoder &encoder,
    PhantomCiphertext Z, int c_used, double scale_mask,
    KSCounters &ks)
{
    const size_t slots = encoder.slot_count();
    size_t ci = Z.chain_index();

    int U = (c_used + H - 1) / H;
    PhantomCiphertext Z_orig = Z;
    for (int delta = 1; delta < U; delta++) {
        int shift = delta * H * M;
        PhantomCiphertext Rd = Z_orig;
        rotate_inplace(ctx, Rd, norm_step(shift, NSLOTS), gk);
        ks.rots++;
        add_inplace(ctx, Z, Rd);
    }

    auto h_mask = msk_rng(slots, M, 0, H);
    PhantomPlaintext hpt;
    encoder.encode(ctx, h_mask, scale_mask, hpt, ci);
    multiply_plain_inplace(ctx, Z, hpt);

    return Z;
}

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

// =========================================================================
// Filesystem helpers
// =========================================================================

bool file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

void wait_for_file(const std::string &path) {
    while (!file_exists(path.c_str()))
        usleep(10000);  // 10ms poll
}

void touch_file(const std::string &path) {
    FILE *f = fopen(path.c_str(), "w");
    if (f) fclose(f);
}

} // anonymous namespace

// =========================================================================
// Main — Full transformer layer
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

    // Bridge RNG (deterministic for reproducibility)
    int bridge_seed = 12345;
    const char *seed_env = std::getenv("BRIDGE_SEED");
    if (seed_env) bridge_seed = std::atoi(seed_env);
    std::mt19937_64 bridge_rng(bridge_seed);

    // --- Read ALL layer inputs ---
    auto t_io0 = std::chrono::high_resolution_clock::now();
    std::string pd = pipe_io::get_pipe_dir();

    constexpr size_t N_X  = (size_t)M * D;
    constexpr size_t N_W  = (size_t)D * D;
    constexpr size_t N_b  = D;
    constexpr size_t N_W1 = (size_t)D * D_FF;
    constexpr size_t N_W2 = (size_t)D_FF * D;
    // X + WQ + WK + WV + bQ + bK + bV + W_O + bO + W1 + W2
    constexpr size_t TOTAL_IN = N_X + 3 * N_W + 3 * N_b + N_W + N_b + N_W1 + N_W2;
    std::vector<double> buf(TOTAL_IN);
    pipe_io::read_f64((pd + "/layer_in.bin").c_str(), buf.data(), buf.size());

    const double *X   = buf.data();
    const double *WQ  = X + N_X;
    const double *WK  = WQ + N_W;
    const double *WV  = WK + N_W;
    const double *bQ  = WV + N_W;
    const double *bK  = bQ + N_b;
    const double *bV  = bK + N_b;
    const double *WO  = bV + N_b;
    const double *bO  = WO + N_W;
    const double *W1  = bO + N_b;
    const double *W2  = W1 + N_W1;
    auto t_io1 = std::chrono::high_resolution_clock::now();

    // --- FDP permutation for Q, K ---
    auto perm = perm_fdp(H, DH);
    auto WQ_fdp = permute_cols(WQ, D, D, perm);
    auto WK_fdp = permute_cols(WK, D, D, perm);
    std::vector<double> bQ_fdp(D), bK_fdp(D);
    for (int i = 0; i < D; i++) {
        bQ_fdp[i] = bQ[perm[i]];
        bK_fdp[i] = bK[perm[i]];
    }

    // =====================================================================
    // Single CKKS context + keygen (covers ALL stages)
    // =====================================================================
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);

    // Merge Galois elements for all stages
    auto galois_cols  = build_galois_elts_full_columns(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_score = build_galois_elts_score(NSLOTS, M, B_FOLD, G_FOLD,
                                                 C_USED_QK, H, GEN_ROT, MMOD);
    auto galois_sv    = build_galois_elts_within_row(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_elts  = merge_galois_elts(
        merge_galois_elts(galois_cols, galois_score),
        galois_sv);
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomRelinKey  rk = sk.gen_relinkey(ctx);
    PhantomCKKSEncoder encoder(ctx);
    cuda_sync();
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    const size_t slots = encoder.slot_count();
    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);
    KSCounters ks;

    std::cout << "stage pipe_layer\n";
    std::cout << "setup_ms " << ms_since(t_setup0, t_setup1) << "\n";
    std::cout << std::flush;

    // =====================================================================
    // PHASE 1: QKV + Score
    // =====================================================================

    auto t_p1_0 = std::chrono::high_resolution_clock::now();

    // Encrypt input at chain_index=2
    auto x_ct = encrypt_packed_pairs(ctx, encoder, sk, X, M, D, HP, NSLOTS,
                                      scale_in, CHAIN_IDX_TOP, C_USED_LIN);
    auto ct_zero_mul_qkv = make_ct_zero_mul(ctx, encoder, sk, scale_mul, CHAIN_IDX_TOP);
    cuda_sync();

    // Weight tables for Q, K, V
    auto WQ_tab = build_wtab_paired(WQ_fdp.data(), D, D, C, C_USED_QK, BLOCKS_QK, C_USED_LIN);
    auto WK_tab = build_wtab_paired(WK_fdp.data(), D, D, C, C_USED_QK, BLOCKS_QK, C_USED_LIN);
    auto WV_tab = build_wtab_paired(WV, D, D, C, -1, -1, C_USED_LIN);

    // QKV projection
    auto babies = build_babies(ctx, gk, x_ct, N1, M, NSLOTS, ks);

    auto Q_ct = linear_complex_paired(ctx, encoder, gk, babies, WQ_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul_qkv, ks, BLOCKS_QK, CHAIN_IDX_TOP);
    auto K_ct = linear_complex_paired(ctx, encoder, gk, babies, WK_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul_qkv, ks, BLOCKS_QK, CHAIN_IDX_TOP);
    auto V_ct = linear_complex_paired(ctx, encoder, gk, babies, WV_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul_qkv, ks, -1, CHAIN_IDX_TOP);

    // ct_real for Q, K, V
    ct_real_blocks(ctx, gk, Q_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, K_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, V_ct, static_cast<size_t>(conj_elt), ks);

    // Add bias
    add_bias_blocks(ctx, encoder, Q_ct, bQ_fdp.data(), D, M, NSLOTS, C_USED_QK,
                    scale_mul, CHAIN_IDX_TOP);
    add_bias_blocks(ctx, encoder, K_ct, bK_fdp.data(), D, M, NSLOTS, C_USED_QK,
                    scale_mul, CHAIN_IDX_TOP);
    add_bias_blocks(ctx, encoder, V_ct, bV, D, M, NSLOTS, C,
                    scale_mul, CHAIN_IDX_TOP);

    // Rescale Q, K, V
    for (auto &ct : Q_ct) rescale_to_next_inplace(ctx, ct);
    for (auto &ct : K_ct) rescale_to_next_inplace(ctx, ct);
    for (auto &ct : V_ct) rescale_to_next_inplace(ctx, ct);
    cuda_sync();

    // Score FDP
    MapEntry mp_half[HALF_M];
    mapf_compute(mp_half);

    std::vector<std::vector<PhantomCiphertext>> Q_bank, K_bank, K_bank_h;
    mk_bank(ctx, gk, encoder, Q_ct, K_ct, C_USED_QK, scale_w,
            Q_bank, K_bank, K_bank_h, ks);

    std::vector<PhantomCiphertext> D_fold;
    emit_f(ctx, rk, Q_bank, K_bank, K_bank_h, mp_half, D_fold, ks);

    for (int t = 0; t < HALF_M; t++)
        D_fold[t] = red_h(ctx, gk, encoder, std::move(D_fold[t]), C_USED_QK, scale_w, ks);
    cuda_sync();

    // MTO (paper §IV.B) before Score decrypt
    auto D_fold_trimmed = pipe_ckks::trim_for_c2m(ctx, D_fold);
    // Decrypt and unpack Score
    std::vector<double> S_out;
    unpack_f(ctx, encoder, sk, D_fold_trimmed, mp_half, S_out);

    // Free Score memory
    Q_bank.clear(); K_bank.clear(); K_bank_h.clear(); D_fold.clear();
    Q_ct.clear(); K_ct.clear();
    // babies no longer needed (will rebuild for OUT/FF)
    babies.clear();

    auto t_p1_1 = std::chrono::high_resolution_clock::now();

    // Bridge: CKKS → MPC (Score)
    pipe_bridge::ckks_to_mpc(S_out.data(), S_out.size(), bridge_rng,
                              pd + "/score_masked.bin",
                              pd + "/score_server_share.bin");
    touch_file(pd + "/phase1_done");
    std::cout << "qkv_score_ms " << ms_since(t_p1_0, t_p1_1) << "\n" << std::flush;

    // =====================================================================
    // Wait for MPC softmax result
    // =====================================================================
    auto t_wait0 = std::chrono::high_resolution_clock::now();
    wait_for_file(pd + "/softmax_done");
    auto t_wait1 = std::chrono::high_resolution_clock::now();

    // Bridge: MPC → plaintext (A_heads for Value SV multiply)
    constexpr size_t N_A = (size_t)H * M * M;
    auto A_heads = pipe_bridge::mpc_to_plain(
        pd + "/softmax_client_share.bin",
        pd + "/softmax_server_share.bin",
        N_A);
    const double *A_flat = A_heads.data();

    std::cout << "wait_softmax_ms " << ms_since(t_wait0, t_wait1) << "\n" << std::flush;

    // =====================================================================
    // PHASE 2: Value SV + OUT projection
    // =====================================================================

    auto t_p2_0 = std::chrono::high_resolution_clock::now();

    // V_ct is at chain_idx=3, scale≈2^40 (after rescale from Phase 1)
    size_t v_ci = V_ct[0].chain_index();

    // Precompute SV masks
    PhantomPlaintext identity_mask_pt;
    {
        std::vector<pc64> ones(slots, pcplx(1.0));
        encoder.encode(ctx, ones, scale_w, identity_mask_pt, v_ci);
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
        encoder.encode(ctx, hm, scale_w, head_mask_pt[tp], v_ci);
        encoder.encode(ctx, tm, scale_w, tail_mask_pt[tp], v_ci);
    }

    // SV computation
    std::vector<double> Y_sv((size_t)M * D, 0.0);

    for (int g = 0; g < G_SV; g++) {
        std::vector<PhantomCiphertext> v_rot(M);
        v_rot[0] = V_ct[g];
        multiply_plain_inplace(ctx, v_rot[0], identity_mask_pt);
        for (int orig_t = 1; orig_t < M; orig_t++) {
            int tp = (M - orig_t) % M;
            PhantomCiphertext R1 = V_ct[g];
            rotate_inplace(ctx, R1, norm_step(tp, NSLOTS), gk);
            ks.rots++;
            PhantomCiphertext R2 = V_ct[g];
            rotate_inplace(ctx, R2, norm_step(tp - M, NSLOTS), gk);
            ks.rots++;
            multiply_plain_inplace(ctx, R1, head_mask_pt[tp]);
            multiply_plain_inplace(ctx, R2, tail_mask_pt[tp]);
            add_inplace(ctx, R1, R2);
            v_rot[orig_t] = R1;
        }

        PhantomCiphertext acc;
        bool acc_inited = false;

        for (int t = 0; t < M; t++) {
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
            encoder.encode(ctx, a_msg, scale_w, a_pt, v_ci);

            PhantomCiphertext term = v_rot[t];
            multiply_plain_inplace(ctx, term, a_pt);

            if (!acc_inited) { acc = term; acc_inited = true; }
            else add_inplace(ctx, acc, term);
        }

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

    // Free V_ct
    V_ct.clear();

    // OUT projection: fresh encrypt Y_sv
    auto y_ct = encrypt_packed_pairs(ctx, encoder, sk, Y_sv.data(), M, D, HP, NSLOTS,
                                      scale_in, 1, C_USED_LIN);
    auto ct_zero_mul_out = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();

    auto WO_tab = build_wtab_paired(WO, D, D, C, -1, -1, C_USED_LIN);

    auto babies_out = build_babies(ctx, gk, y_ct, N1, M, NSLOTS, ks);
    auto Z_ct = linear_complex_paired(ctx, encoder, gk, babies_out, WO_tab,
                                       D, M, N1, N2, C, NSLOTS,
                                       scale_w, ct_zero_mul_out, ks);
    ct_real_blocks(ctx, gk, Z_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();

    // MTO (paper §IV.B) before OUT decrypt (Z -> LN1 boundary)
    auto Z_ct_trimmed = pipe_ckks::trim_for_c2m(ctx, Z_ct);
    // Decrypt OUT
    auto Z_out = decrypt_blocks(ctx, encoder, sk, Z_ct_trimmed, M, D, NSLOTS);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            Z_out[(size_t)i * D + j] += bO[j];

    auto t_p2_1 = std::chrono::high_resolution_clock::now();

    // Free OUT memory
    Z_ct.clear(); babies_out.clear(); y_ct.clear();

    // Bridge: CKKS → MPC (Z_out)
    pipe_bridge::ckks_to_mpc(Z_out.data(), Z_out.size(), bridge_rng,
                              pd + "/z_masked.bin",
                              pd + "/z_server_share.bin");
    touch_file(pd + "/phase2_done");
    std::cout << "sv_out_ms " << ms_since(t_p2_0, t_p2_1) << "\n" << std::flush;

    // =====================================================================
    // Wait for MPC LN1 result
    // =====================================================================
    auto t_wait2 = std::chrono::high_resolution_clock::now();
    wait_for_file(pd + "/ln1_done");
    auto t_wait3 = std::chrono::high_resolution_clock::now();

    // Bridge: MPC → CKKS (re-encrypt LN1 output for FF1)
    auto ln1_plain = pipe_bridge::mpc_to_plain(
        pd + "/ln1_client_share.bin",
        pd + "/ln1_server_share.bin",
        (size_t)M * D);

    std::cout << "wait_ln1_ms " << ms_since(t_wait2, t_wait3) << "\n" << std::flush;

    // =====================================================================
    // PHASE 3: FF1
    // =====================================================================

    auto t_p3_0 = std::chrono::high_resolution_clock::now();

    auto ff1_ct = encrypt_packed_pairs(ctx, encoder, sk, ln1_plain.data(), M, D, HP_FF1,
                                        NSLOTS, scale_in, 1, C_IN_FF1);
    auto ct_zero_mul_ff1 = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();

    auto W1_tab = build_wtab_paired(W1, D, D_FF, C, -1, -1, C_IN_FF1);

    auto babies_ff1 = build_babies(ctx, gk, ff1_ct, N1, M, NSLOTS, ks);
    auto h1_ct = linear_complex_paired(ctx, encoder, gk, babies_ff1, W1_tab,
                                        D_FF, M, N1, N2, C, NSLOTS,
                                        scale_w, ct_zero_mul_ff1, ks);
    ct_real_blocks(ctx, gk, h1_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();

    // MTO (paper §IV.B): trim modulus chain to conversion-safe minimum
    // before decrypt to reduce P1's CKKS-side decode cost.
    auto h1_ct_trimmed = pipe_ckks::trim_for_c2m(ctx, h1_ct);
    auto H1_out = decrypt_blocks(ctx, encoder, sk, h1_ct_trimmed, M, D_FF, NSLOTS);

    auto t_p3_1 = std::chrono::high_resolution_clock::now();

    // Free FF1 memory
    h1_ct.clear(); babies_ff1.clear(); ff1_ct.clear();

    // Bridge: CKKS → MPC (H1)
    pipe_bridge::ckks_to_mpc(H1_out.data(), H1_out.size(), bridge_rng,
                              pd + "/ff1_masked.bin",
                              pd + "/ff1_server_share.bin");
    touch_file(pd + "/phase3_done");
    std::cout << "ff1_ms " << ms_since(t_p3_0, t_p3_1) << "\n" << std::flush;

    // =====================================================================
    // Wait for MPC GELU result
    // =====================================================================
    auto t_wait4 = std::chrono::high_resolution_clock::now();
    wait_for_file(pd + "/gelu_done");
    auto t_wait5 = std::chrono::high_resolution_clock::now();

    auto gelu_plain = pipe_bridge::mpc_to_plain(
        pd + "/gelu_client_share.bin",
        pd + "/gelu_server_share.bin",
        (size_t)M * D_FF);

    std::cout << "wait_gelu_ms " << ms_since(t_wait4, t_wait5) << "\n" << std::flush;

    // =====================================================================
    // PHASE 4: FF2
    // =====================================================================

    auto t_p4_0 = std::chrono::high_resolution_clock::now();

    auto ff2_ct = encrypt_packed_pairs(ctx, encoder, sk, gelu_plain.data(), M, D_FF,
                                        HP_FF2, NSLOTS, scale_in, 1, C_IN_FF2);
    auto ct_zero_mul_ff2 = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();

    auto W2_tab = build_wtab_paired(W2, D_FF, D, C, -1, -1, C_IN_FF2);

    auto babies_ff2 = build_babies(ctx, gk, ff2_ct, N1_FF2, M, NSLOTS, ks);
    auto h2_ct = linear_complex_paired(ctx, encoder, gk, babies_ff2, W2_tab,
                                        D, M, N1_FF2, N2_FF2, C, NSLOTS,
                                        scale_w, ct_zero_mul_ff2, ks);
    ct_real_blocks(ctx, gk, h2_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();

    // MTO (paper §IV.B) before FF2 decrypt (H2 -> LN2 boundary)
    auto h2_ct_trimmed = pipe_ckks::trim_for_c2m(ctx, h2_ct);
    auto H2_out = decrypt_blocks(ctx, encoder, sk, h2_ct_trimmed, M, D, NSLOTS);

    auto t_p4_1 = std::chrono::high_resolution_clock::now();

    // Free FF2 memory
    h2_ct.clear(); babies_ff2.clear(); ff2_ct.clear();

    // Bridge: CKKS → MPC (H2 — final FHE output)
    pipe_bridge::ckks_to_mpc(H2_out.data(), H2_out.size(), bridge_rng,
                              pd + "/ff2_masked.bin",
                              pd + "/ff2_server_share.bin");
    touch_file(pd + "/phase4_done");

    // =====================================================================
    // Print metrics
    // =====================================================================
    std::cout << std::setprecision(10);
    std::cout << "sv_out_ms "     << ms_since(t_p2_0, t_p2_1) << "\n";
    std::cout << "ff1_ms "        << ms_since(t_p3_0, t_p3_1) << "\n";
    std::cout << "ff2_ms "        << ms_since(t_p4_0, t_p4_1) << "\n";
    std::cout << "read_ms "       << ms_since(t_io0, t_io1) << "\n";
    std::cout << "ks_rots "       << ks.rots << "\n";
    std::cout << "ks_muls "       << ks.muls_ctct << "\n";
    std::cout << "ks_conj "       << ks.conj << "\n";
    std::cout << "galois_elts "   << galois_elts.size() << "\n";

    return 0;
}
