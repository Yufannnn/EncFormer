// pipe_ckks_attn.cu — Full attention layer: QKV + Score + Value (SV) + OUT.
//
// V stays encrypted in GPU memory throughout — NEVER decrypted between QKV and
// Value.  Two-phase execution with filesystem signaling for MPC softmax.
//
// Phase 1 (QKV + Score):
//   Reads:  PIPE_DIR/attn_in.bin  = X[M×D] ++ WQ[D×D] ++ WK[D×D] ++ WV[D×D]
//                                    ++ bQ[D] ++ bK[D] ++ bV[D] ++ W_O[D×D] ++ bO[D]
//   Writes: PIPE_DIR/score_out.bin  = S_heads[H×M×M]
//           PIPE_DIR/phase1_done    (empty signal file)
//
// Phase 2 (Value + OUT):
//   Reads:  PIPE_DIR/a_heads_in.bin = A_heads[H×M×M]
//   Writes: PIPE_DIR/value_out.bin  = Z[M×D]
//
// Scale management:
//   QKV: encrypt at chain_idx=2, linear multiply scale_w → scale=2^80
//   Q,K: rescale to chain_idx=3 (scale≈2^40) for Score
//   V:   add bias at scale=2^80, rescale to chain_idx=3 (scale≈2^40) for SV
//   SV:  two plain multiplies (mask + A_diag) at scale_w → scale≈2^120, decrypt
//   OUT: fresh encrypt at chain_idx=1, linear_complex_paired, ct_real, decrypt

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

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
constexpr int HP         = EF_HP_QKV;
constexpr int C_USED_LIN = EF_C_USED_LIN;

// QKV constants
constexpr int N1         = EF_N1_DEFAULT;
constexpr int N2         = EF_N2_DEFAULT;
constexpr int C_USED_QK  = EF_C_USED_QK;
constexpr int BLOCKS_QK  = EF_BLOCKS_QK;
constexpr int C_USED_V   = EF_C_USED_V;
constexpr int BLOCKS_V   = EF_BLOCKS_V;

// Score constants
constexpr int B_FOLD     = EF_B_FOLD;
constexpr int G_FOLD     = EF_G_FOLD;
constexpr int HALF_M     = EF_HALF_M;
constexpr int BLEN       = EF_BLEN;

// Value+OUT constants
constexpr int N1_OUT     = EF_N1_DEFAULT;
constexpr int N2_OUT     = EF_N2_DEFAULT;
constexpr int G_SV       = EF_G_SV;

constexpr size_t CHAIN_IDX_TOP = 2;     // encrypt here

// =========================================================================
// Score helpers (same as pipe_ckks_qkv_score.cu)
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
// Main
// =========================================================================

int main(int argc, char **argv) {
    bool server_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0) server_mode = true;
    }
    if (!server_mode) {
        const char *envv = std::getenv("EF_SERVER_MODE");
        if (envv && envv[0] == '1') server_mode = true;
    }

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

    std::string pd = pipe_io::get_pipe_dir();

    // ---- One-time CKKS setup (shared across all per-layer iterations) ----
    auto t_setup0 = std::chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);

    auto galois_cols   = build_galois_elts_full_columns(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_score  = build_galois_elts_score(NSLOTS, M, B_FOLD, G_FOLD,
                                                  C_USED_QK, H, GEN_ROT, MMOD);
    auto galois_sv     = build_galois_elts_within_row(NSLOTS, M, GEN_ROT, MMOD);
    auto galois_elts   = merge_galois_elts(
        merge_galois_elts(galois_cols, galois_score),
        galois_sv);
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);
    PhantomCKKSEncoder encoder(ctx);
    cuda_sync();
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    const size_t slots = encoder.slot_count();
    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);

    if (server_mode) {
        std::cerr << "[attn_server] setup_ms " << ms_since(t_setup0, t_setup1)
                  << " galois_elts " << galois_elts.size() << " ready\n";
        std::ofstream rf(pd + "/attn_server_ready"); rf.close();
    }

    int iter = 0;
    while (true) {
        // Per-layer file paths (server mode uses indexed names, single-shot uses legacy names).
        std::string in_path, score_out_path, p1_done_path, ahead_in_path,
                    value_out_path, p1_req_path, p1_resp_path, p2_req_path, p2_resp_path;
        if (server_mode) {
            std::string sfx = "_" + std::to_string(iter);
            in_path        = pd + "/attn_in"      + sfx + ".bin";
            score_out_path = pd + "/score_out"    + sfx + ".bin";
            p1_done_path   = pd + "/phase1_done"  + sfx;
            ahead_in_path  = pd + "/a_heads_in"   + sfx + ".bin";
            value_out_path = pd + "/value_out"    + sfx + ".bin";
            p1_req_path    = pd + "/attn_phase1_request_"  + std::to_string(iter);
            p1_resp_path   = pd + "/attn_phase1_response_" + std::to_string(iter);
            p2_req_path    = pd + "/attn_phase2_request_"  + std::to_string(iter);
            p2_resp_path   = pd + "/attn_phase2_response_" + std::to_string(iter);

            // Wait for Phase 1 request from python.
            std::string shutdown_path = pd + "/attn_shutdown";
            while (!file_exists(p1_req_path.c_str())) {
                if (file_exists(shutdown_path.c_str())) {
                    std::cerr << "[attn_server] shutdown after " << iter << " iters\n";
                    return 0;
                }
                usleep(500);
            }
        } else {
            in_path        = pd + "/attn_in.bin";
            score_out_path = pd + "/score_out.bin";
            p1_done_path   = pd + "/phase1_done";
            ahead_in_path  = pd + "/a_heads_in.bin";
            value_out_path = pd + "/value_out.bin";
        }

        // --- Read input (per-layer) ---
        auto t_io0 = std::chrono::high_resolution_clock::now();
        constexpr size_t N_X = (size_t)M * D;
        constexpr size_t N_W = (size_t)D * D;
        constexpr size_t N_b = D;
        // X + WQ + WK + WV + bQ + bK + bV + W_O + bO
        constexpr size_t TOTAL_IN = N_X + 3 * N_W + 3 * N_b + N_W + N_b;
        std::vector<double> buf(TOTAL_IN);
        pipe_io::read_f64(in_path.c_str(), buf.data(), buf.size());

        const double *X  = buf.data();
        const double *WQ = X + N_X;
        const double *WK = WQ + N_W;
        const double *WV = WK + N_W;
        const double *bQ = WV + N_W;
        const double *bK = bQ + N_b;
        const double *bV = bK + N_b;
        const double *WO = bV + N_b;
        const double *bO = WO + N_W;
        auto t_io1 = std::chrono::high_resolution_clock::now();

        // --- FDP permutation for Q, K (per-layer; depends on weights) ---
        auto perm = perm_fdp(H, DH);
        auto WQ_fdp = permute_cols(WQ, D, D, perm);
        auto WK_fdp = permute_cols(WK, D, D, perm);
        std::vector<double> bQ_fdp(D), bK_fdp(D);
        for (int i = 0; i < D; i++) {
            bQ_fdp[i] = bQ[perm[i]];
            bK_fdp[i] = bK[perm[i]];
        }

        // Reset per-layer counters.
        KSCounters ks;

    // =====================================================================
    // Phase 1: QKV + Score
    // =====================================================================

    // --- Encrypt input at chain_index=2 ---
    auto t_enc0 = std::chrono::high_resolution_clock::now();
    auto x_ct = encrypt_packed_pairs(ctx, encoder, sk, X, M, D, HP, NSLOTS,
                                      scale_in, CHAIN_IDX_TOP, C_USED_LIN);
    auto ct_zero_mul_qkv = make_ct_zero_mul(ctx, encoder, sk, scale_mul, CHAIN_IDX_TOP);
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
                                       ct_zero_mul_qkv, ks, BLOCKS_QK, CHAIN_IDX_TOP);
    auto K_ct = linear_complex_paired(ctx, encoder, gk, babies, WK_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul_qkv, ks, BLOCKS_QK, CHAIN_IDX_TOP);
    auto V_ct = linear_complex_paired(ctx, encoder, gk, babies, WV_tab,
                                       D, M, N1, N2, C, NSLOTS, scale_w,
                                       ct_zero_mul_qkv, ks, -1, CHAIN_IDX_TOP);

    // ct_real (conjugate trick) — all three
    ct_real_blocks(ctx, gk, Q_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, K_ct, static_cast<size_t>(conj_elt), ks);
    ct_real_blocks(ctx, gk, V_ct, static_cast<size_t>(conj_elt), ks);

    // Add bias to Q, K in encrypted domain (chain_idx=2, scale=2^80)
    add_bias_blocks(ctx, encoder, Q_ct, bQ_fdp.data(), D, M, NSLOTS, C_USED_QK,
                    scale_mul, CHAIN_IDX_TOP);
    add_bias_blocks(ctx, encoder, K_ct, bK_fdp.data(), D, M, NSLOTS, C_USED_QK,
                    scale_mul, CHAIN_IDX_TOP);

    // Add bias to V in encrypted domain (chain_idx=2, scale=2^80, c_used=C)
    add_bias_blocks(ctx, encoder, V_ct, bV, D, M, NSLOTS, C,
                    scale_mul, CHAIN_IDX_TOP);

    // Rescale Q, K from chain_idx=2 → 3 (scale ≈ 2^40) for Score
    for (auto &ct : Q_ct) rescale_to_next_inplace(ctx, ct);
    for (auto &ct : K_ct) rescale_to_next_inplace(ctx, ct);

    // Rescale V from chain_idx=2 → 3 (scale ≈ 2^40) for SV
    for (auto &ct : V_ct) rescale_to_next_inplace(ctx, ct);

    cuda_sync();
    auto t_qkv1 = std::chrono::high_resolution_clock::now();

    // ===================== Score (FDP) =====================
    auto t_score0 = std::chrono::high_resolution_clock::now();

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
    auto t_score1 = std::chrono::high_resolution_clock::now();

    // --- Decrypt and unpack Score ---
    auto t_sdec0 = std::chrono::high_resolution_clock::now();
    std::vector<double> S_out;
    unpack_f(ctx, encoder, sk, D_fold, mp_half, S_out);
    auto t_sdec1 = std::chrono::high_resolution_clock::now();

    // Free Score memory (Q_bank, K_bank, K_bank_h, D_fold no longer needed)
    Q_bank.clear(); K_bank.clear(); K_bank_h.clear(); D_fold.clear();
    // Q_ct and K_ct also no longer needed
    Q_ct.clear(); K_ct.clear();

    // --- Write Score output and signal Phase 1 done ---
    pipe_io::write_f64(score_out_path.c_str(), S_out.data(), S_out.size());
    touch_file(p1_done_path);
    if (server_mode) {
        // Consume Phase 1 request, signal Phase 1 response.
        ::unlink(p1_req_path.c_str());
        touch_file(p1_resp_path);
    }
    std::cout << "[phase1 iter=" << iter << "] score written, waiting for MPC softmax...\n" << std::flush;

    // =====================================================================
    // Wait for Phase 2 input (A_heads from MPC softmax)
    // =====================================================================
    auto t_wait0 = std::chrono::high_resolution_clock::now();
    if (server_mode) {
        std::string shutdown_path = pd + "/attn_shutdown";
        while (!file_exists(p2_req_path.c_str())) {
            if (file_exists(shutdown_path.c_str())) {
                std::cerr << "[attn_server] shutdown mid-layer (phase2 wait) iter=" << iter << "\n";
                return 0;
            }
            usleep(500);
        }
    } else {
        wait_for_file(ahead_in_path);
    }
    auto t_wait1 = std::chrono::high_resolution_clock::now();

    constexpr size_t N_A = (size_t)H * M * M;
    std::vector<double> A_buf(N_A);
    pipe_io::read_f64(ahead_in_path.c_str(), A_buf.data(), N_A);
    const double *A_flat = A_buf.data();

    std::cout << "[phase2 iter=" << iter << "] A_heads received, starting Value+OUT\n" << std::flush;

    // =====================================================================
    // Phase 2: SV (Value) + OUT — V_ct still encrypted in GPU memory!
    // =====================================================================

    // V_ct is at chain_idx=3, scale≈2^40 (after rescale)
    size_t v_ci = V_ct[0].chain_index();

    auto t_sv0 = std::chrono::high_resolution_clock::now();

    // Precompute rot_within masks at V_ct's chain_index
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

    // SV computation: Y[i, g*C+c] = sum_t A_h[i, (i-t)%M] * V[(i-t)%M, g*C+c]
    std::vector<double> Y_sv((size_t)M * D, 0.0);

    for (int g = 0; g < G_SV; g++) {
        // Precompute all rot_within results for V_ct[g]
        std::vector<PhantomCiphertext> v_rot(M);
        v_rot[0] = V_ct[g];
        multiply_plain_inplace(ctx, v_rot[0], identity_mask_pt);
        for (int orig_t = 1; orig_t < M; orig_t++) {
            int tp = (M - orig_t) % M;
            int s1 = tp;
            int s2 = tp - M;

            PhantomCiphertext R1 = V_ct[g];
            rotate_inplace(ctx, R1, norm_step(s1, NSLOTS), gk);
            ks.rots++;

            PhantomCiphertext R2 = V_ct[g];
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
            // After mask multiply, v_rot[t] is at chain_idx = v_ci, but chain_index
            // doesn't change from plain multiply, so A_diag also at v_ci.
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

        // Decrypt this group's SV result
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

    // Free V_ct memory (no longer needed)
    V_ct.clear();

    // ==================== OUT projection ====================
    auto t_enc2 = std::chrono::high_resolution_clock::now();
    auto y_ct = encrypt_packed_pairs(ctx, encoder, sk, Y_sv.data(), M, D, HP, NSLOTS,
                                      scale_in, 1, C_USED_LIN);
    auto ct_zero_mul_out = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();
    auto t_enc3 = std::chrono::high_resolution_clock::now();

    auto WO_tab = build_wtab_paired(WO, D, D, C, -1, -1, C_USED_LIN);

    auto t_out0 = std::chrono::high_resolution_clock::now();
    auto babies_out = build_babies(ctx, gk, y_ct, N1_OUT, M, NSLOTS, ks);
    auto Z_ct = linear_complex_paired(ctx, encoder, gk, babies_out, WO_tab,
                                       D, M, N1_OUT, N2_OUT, C, NSLOTS,
                                       scale_w, ct_zero_mul_out, ks);
    ct_real_blocks(ctx, gk, Z_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();
    auto t_out1 = std::chrono::high_resolution_clock::now();

    // --- Decrypt OUT ---
    auto t_dec0 = std::chrono::high_resolution_clock::now();
    auto Z_out = decrypt_blocks(ctx, encoder, sk, Z_ct, M, D, NSLOTS);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            Z_out[(size_t)i * D + j] += bO[j];
    auto t_dec1 = std::chrono::high_resolution_clock::now();

    // --- Write output ---
    auto t_wrt0 = std::chrono::high_resolution_clock::now();
    pipe_io::write_f64(value_out_path.c_str(), Z_out.data(), Z_out.size());
    auto t_wrt1 = std::chrono::high_resolution_clock::now();
    if (server_mode) {
        ::unlink(p2_req_path.c_str());
        touch_file(p2_resp_path);
    }

    // =====================================================================
    // Verification
    // =====================================================================

    // V reference (for verification — decrypt V_ct internally; in production V is NEVER decrypted here)
    auto V_ref = matmul_ref(X, M, D, WV, D);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            V_ref[(size_t)i * D + j] += bV[j];

    // Q, K references
    auto Q_ref = matmul_ref(X, M, D, WQ_fdp.data(), D);
    auto K_ref = matmul_ref(X, M, D, WK_fdp.data(), D);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++) {
            Q_ref[(size_t)i * D + j] += bQ_fdp[j];
            K_ref[(size_t)i * D + j] += bK_fdp[j];
        }

    // Score reference
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

    // SV reference
    std::vector<double> Y_ref((size_t)M * D, 0.0);
    for (int h = 0; h < H; h++) {
        const double *Ah = A_flat + (size_t)h * M * M;
        for (int i = 0; i < M; i++)
            for (int u = 0; u < DH; u++) {
                double a = 0.0;
                for (int k = 0; k < M; k++)
                    a += Ah[(size_t)i * M + k] * V_ref[(size_t)k * D + h * DH + u];
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

    // =====================================================================
    // Print metrics
    // =====================================================================
    double wait_ms = ms_since(t_wait0, t_wait1);

    // In server mode, write per-iter metrics to a log file (not stdout, so
    // python doesn't have to drain stdout); single-shot writes to stdout.
    if (server_mode) {
        std::ofstream log(pd + "/attn_log_" + std::to_string(iter) + ".txt");
        log << std::setprecision(10);
        log << "stage pipe_attn\n";
        log << "encrypt_ms "   << ms_since(t_enc0, t_enc1) + ms_since(t_enc2, t_enc3) << "\n";
        log << "qkv_fhe_ms "   << ms_since(t_qkv0, t_qkv1) << "\n";
        log << "score_fhe_ms " << ms_since(t_score0, t_score1) << "\n";
        log << "sv_fhe_ms "    << ms_since(t_sv0, t_sv1) << "\n";
        log << "out_fhe_ms "   << ms_since(t_out0, t_out1) << "\n";
        log << "decrypt_ms "   << ms_since(t_sdec0, t_sdec1) + ms_since(t_dec0, t_dec1) << "\n";
        log << "wait_ms "      << wait_ms << "\n";
        log << "read_ms "      << ms_since(t_io0, t_io1) << "\n";
        log << "write_ms "     << ms_since(t_wrt0, t_wrt1) << "\n";
        log << "rel_err_s "    << std::scientific << err_s << std::defaultfloat << "\n";
        log << "rel_err_sv "   << std::scientific << err_sv << std::defaultfloat << "\n";
        log << "rel_err_out "  << std::scientific << err_out << std::defaultfloat << "\n";
        log << "mse_s "        << std::scientific << mse_s << std::defaultfloat << "\n";
        log << "mse_sv "       << std::scientific << mse_sv << std::defaultfloat << "\n";
        log << "mse_out "      << std::scientific << mse_out << std::defaultfloat << "\n";
        log << "ks_rots "      << ks.rots << "\n";
        log << "ks_muls "      << ks.muls_ctct << "\n";
        log << "ks_conj "      << ks.conj << "\n";
    } else {
        std::cout << std::setprecision(10);
        std::cout << "stage pipe_attn\n";
        std::cout << "setup_ms "     << ms_since(t_setup0, t_setup1) << "\n";
        std::cout << "encrypt_ms "   << ms_since(t_enc0, t_enc1) + ms_since(t_enc2, t_enc3) << "\n";
        std::cout << "qkv_fhe_ms "   << ms_since(t_qkv0, t_qkv1) << "\n";
        std::cout << "score_fhe_ms " << ms_since(t_score0, t_score1) << "\n";
        std::cout << "sv_fhe_ms "    << ms_since(t_sv0, t_sv1) << "\n";
        std::cout << "out_fhe_ms "   << ms_since(t_out0, t_out1) << "\n";
        std::cout << "decrypt_ms "   << ms_since(t_sdec0, t_sdec1) + ms_since(t_dec0, t_dec1) << "\n";
        std::cout << "wait_ms "      << wait_ms << "\n";
        std::cout << "read_ms "      << ms_since(t_io0, t_io1) << "\n";
        std::cout << "write_ms "     << ms_since(t_wrt0, t_wrt1) << "\n";
        std::cout << "rel_err_s "    << std::scientific << err_s << std::defaultfloat << "\n";
        std::cout << "rel_err_sv "   << std::scientific << err_sv << std::defaultfloat << "\n";
        std::cout << "rel_err_out "  << std::scientific << err_out << std::defaultfloat << "\n";
        std::cout << "mse_s "        << std::scientific << mse_s << std::defaultfloat << "\n";
        std::cout << "mse_sv "       << std::scientific << mse_sv << std::defaultfloat << "\n";
        std::cout << "mse_out "      << std::scientific << mse_out << std::defaultfloat << "\n";
        std::cout << "ks_rots "      << ks.rots << "\n";
        std::cout << "ks_muls "      << ks.muls_ctct << "\n";
        std::cout << "ks_conj "      << ks.conj << "\n";
        std::cout << "galois_elts "  << galois_elts.size() << "\n";
    }

    if (!server_mode) return 0;
    ++iter;
    } // end while server-mode loop
    return 0;
}
