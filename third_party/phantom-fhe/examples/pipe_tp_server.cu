#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "pipe_io.h"
#include "pipe_ckks_eval.h"
#include "pipe_ckks_keys.h"
#include "attn_kernels.h"
#include "tp_conv.h"
#include "ef_role_guard.h"

using namespace pipe_ckks;

namespace {
#include "model_config.h"
constexpr int NSLOTS = EF_NSLOTS, M = EF_M, C = EF_C, D = EF_D, H = EF_H, DH = EF_DH, DFF = EF_D_FF;
constexpr int C_USED_LIN = EF_C_USED_LIN, N1 = EF_N1_DEFAULT, N2 = EF_N2_DEFAULT;
constexpr int C_USED_QK = EF_C_USED_QK, BLOCKS_QK = EF_BLOCKS_QK;
using clk = std::chrono::high_resolution_clock;

size_t env_size(const char *k, size_t dflt) {
    const char *v = std::getenv(k);
    return (v && *v) ? std::strtoul(v, nullptr, 10) : dflt;
}

struct Layer {
    std::vector<double> WQf, WKf, WV, bQf, bKf, bV, WOh, W1, W2, b1;
    std::vector<double> row_scale;
    double add_c = 0.0;
};

Layer load_layer(const std::string &path) {
    const size_t nW = (size_t)D * D, nF = (size_t)D * DFF;
    std::vector<double> buf(4 * nW + 3 * D + 2 * nF + DFF + (size_t)H * M + 2);
    pipe_io::read_f64(path.c_str(), buf.data(), buf.size());
    const double *p = buf.data();
    const double *WQ = p, *WK = p + nW, *WV = p + 2 * nW, *bQ = p + 3 * nW, *bK = bQ + D, *bV = bK + D;
    const double *WO = bV + D, *W1 = WO + nW, *W2 = W1 + nF, *b1 = W2 + nF;
    Layer L;
    auto perm = perm_fdp(H, DH);
    const double qs = 1.0 / std::sqrt((double)DH);
    L.WQf = permute_cols(WQ, D, D, perm);
    for (auto &w : L.WQf) w *= qs;
    L.WKf = permute_cols(WK, D, D, perm);
    L.WV.assign(WV, WV + nW);
    L.bQf.resize(D); L.bKf.resize(D);
    for (int i = 0; i < D; i++) { L.bQf[i] = bQ[perm[i]] * qs; L.bKf[i] = bK[perm[i]]; }
    L.bV.assign(bV, bV + D);
    L.WOh.assign(WO, WO + nW);
    for (auto &w : L.WOh) w *= 0.5;
    L.W1.assign(W1, W1 + nF);
    L.W2.assign(W2, W2 + nF);
    L.b1.assign(b1, b1 + DFF);
    const double *rs = b1 + DFF, *tail = rs + (size_t)H * M;
    if (tail[1] != 0.0) { L.row_scale.assign(rs, rs + (size_t)H * M); L.add_c = tail[0]; }
    return L;
}

void write_score_index(const ScoreGeom &g, const std::string &path) {
    auto idx = score_unpack_index(g, NSLOTS);
    std::ofstream f(path, std::ios::binary);
    for (const auto &s : idx) {
        int32_t rec[6] = {s.ct, s.slot, s.h, s.row, s.col_re, s.col_im};
        f.write(reinterpret_cast<const char *>(rec), sizeof rec);
    }
}

}

int main() {
    const std::string dir = pipe_io::get_pipe_dir_or("TP_DIR");
    const std::string wire = dir + "/wire", srv = dir + "/srv";
    const size_t c0 = env_size("TP_C0", 2), cf1 = env_size("TP_CF1", 3), cf2 = env_size("TP_CF2", 6);
    if (cudaSetDevice(0) != cudaSuccess) { std::cerr << "no GPU\n"; return 1; }

    auto t_s0 = clk::now();
    pipe_io::wait_for_file(wire + "/keys/ready");
    auto prm = CkksParams::load(wire + "/keys/params.txt");
    EncryptionParameters parms(scheme_type::ckks);
    prm.to_parms(parms);
    PhantomContext ctx(parms);
    PhantomPublicKey pk;
    PhantomRelinKey rk;
    PhantomGaloisKey gk;
    load_obj(pk, wire + "/keys/pk.bin");
    load_obj(rk, wire + "/keys/rlk.bin");
    load_obj(gk, wire + "/keys/gk.bin");
    PhantomCKKSEncoder enc(ctx);
    cuda_sync();
    const double setup_ms = ms_since(t_s0, clk::now());
    Env e{ctx, enc, gk, rk, NSLOTS, M, static_cast<uint32_t>(4ull * NSLOTS - 1)};
    MaskCache mc;
    ScoreGeom sg{H, M, EF_B_FOLD, EF_G_FOLD, C_USED_QK};
    ValueGeom vg{M, DH};
    write_score_index(sg, srv + "/score_index.i32");
    {
        std::ofstream r(srv + "/server_ready");
        r << "setup_ms " << setup_ms << "\nconversion " << tp_conv::backend_name() << "\nc0 " << c0 << "\ncf1 " << cf1
          << "\ncf2 " << cf2
          << "\nscore_cts " << score_packed_cts(sg, NSLOTS) << "\n";
    }

    conv_hp::Rng rng;
    const double s_conv = std::pow(2.0, 30);
    const int b_score = (int)env_size("TP_BOUND_SCORE", 7), b_z = (int)env_size("TP_BOUND_Z", 7);
    const int b_h1 = (int)env_size("TP_BOUND_H1", 7), b_h2 = (int)env_size("TP_BOUND_H2", 7);
    const int b_f = (int)env_size("TP_BOUND_F", 15);
    Layer L;
    std::vector<PhantomCiphertext> V;
    std::map<std::string, tp_conv::M2CKept> kept;

    for (int k = 0;; k++) {
        const std::string req = srv + "/req_" + std::to_string(k);
        while (!pipe_io::file_present(req)) { pipe_io::exit_if_orphaned(req); usleep(200); }
        std::string cmd, a1, a2, a3;
        { std::ifstream f(req); f >> cmd >> a1 >> a2 >> a3; }
        std::ostringstream out;
        out << std::setprecision(10) << "cmd " << cmd << "\n";
        const KSCounters ks0 = e.ks;
        const long long pm0 = e.pt_mults, rs0 = e.rescales;
        cuda_sync();
        auto t0 = clk::now();
        size_t wire_bytes = 0;

        if (cmd == "shutdown") {
            std::ofstream(srv + "/resp_" + std::to_string(k)) << out.str();
            return 0;
        } else if (cmd == "load_layer") {
            L = load_layer(a1);
        } else if (cmd == "m2c_mask") {
            kept[a1] = tp_conv::m2c_server_mask(enc, rng, srv + "/" + a1 + "_s1.u64", std::stoul(a2), std::stoi(a3),
                                                wire + "/" + a1 + "_mask.u64");
            wire_bytes = kept[a1].ncts * 2 * enc.slot_count() * 8;
        } else if (cmd == "qkv_score") {
            auto x = tp_conv::m2c_server_finish(ctx, enc, wire + "/x_ct.bin", kept.at("x"));
            require_chain(x.at(0), c0, "qkv_score input");
            auto t1 = clk::now();
            auto tq = build_wtab_paired(L.WQf.data(), D, D, C, C_USED_QK, BLOCKS_QK, C_USED_LIN);
            auto tk = build_wtab_paired(L.WKf.data(), D, D, C, C_USED_QK, BLOCKS_QK, C_USED_LIN);
            auto tv = build_wtab_paired(L.WV.data(), D, D, C, -1, -1, C_USED_LIN);
            auto t2 = clk::now();
            auto xb = babies_of(e, x, N1);
            auto Q = linear_layer(e, xb, tq, D, N1, N2, C_USED_QK, BLOCKS_QK, L.bQf.data(), C_USED_QK);
            auto Kc = linear_layer(e, xb, tk, D, N1, N2, C_USED_QK, BLOCKS_QK, L.bKf.data(), C_USED_QK);
            V = linear_layer(e, xb, tv, D, N1, N2, C, D / C, L.bV.data(), C);
            xb.clear();
            cuda_sync();
            auto t3 = clk::now();
            auto packed = score_kernel(e, mc, sg, Q, Kc, s_conv, L.row_scale.empty() ? nullptr : L.row_scale.data(),
                                       L.add_c);
            cuda_sync();
            auto t4 = clk::now();
            wire_bytes = tp_conv::c2m_server(ctx, pk, enc, rng, packed, b_score, wire + "/score_c2m.bin",
                                             srv + "/score_s1.u64");
            auto t5 = clk::now();
            out << "m2c_finish_ms " << ms_since(t0, t1) << "\nwtab_ms " << ms_since(t1, t2) << "\nqkv_fhe_ms "
                << ms_since(t2, t3) << "\nscore_fhe_ms " << ms_since(t3, t4) << "\nc2m_ms " << ms_since(t4, t5)
                << "\nscore_chain " << packed[0].chain_index() << "\n";
        } else if (cmd == "value_out") {
            auto P = tp_conv::m2c_server_finish(ctx, enc, wire + "/pfd_ct.bin", kept.at("pfd"));
            require_chain(P.at(0), V.at(0).chain_index() + 1, "value_out input");
            auto t1 = clk::now();
            std::vector<PhantomCiphertext> o(V.size());
            for (size_t b = 0; b < V.size(); b++) o[b] = value_block(e, mc, vg, V[b], P[b], b % 2 == 1);
            auto z = pair_value_blocks(e, o);
            o.clear(); V.clear();
            cuda_sync();
            auto t2 = clk::now();
            auto to = build_wtab_paired(L.WOh.data(), D, D, C, -1, -1, C_USED_LIN);
            auto t3 = clk::now();
            auto Z = linear_layer_paired(e, z, to, D, N1, N2, D / C, s_conv);
            cuda_sync();
            auto t4 = clk::now();
            wire_bytes = tp_conv::c2m_server(ctx, pk, enc, rng, Z, b_z, wire + "/z_c2m.bin", srv + "/z_s1.u64");
            auto t5 = clk::now();
            out << "m2c_finish_ms " << ms_since(t0, t1) << "\nvalue_fhe_ms " << ms_since(t1, t2) << "\nwtab_ms "
                << ms_since(t2, t3) << "\nout_fhe_ms " << ms_since(t3, t4) << "\nc2m_ms " << ms_since(t4, t5)
                << "\nout_chain " << Z[0].chain_index() << "\n";
        } else if (cmd == "ff1") {
            auto x = tp_conv::m2c_server_finish(ctx, enc, wire + "/ff1_ct.bin", kept.at("ff1"));
            require_chain(x.at(0), cf1, "ff1 input");
            auto t1 = clk::now();
            auto tab = build_wtab_paired(L.W1.data(), D, DFF, C, -1, -1, EF_C_USED_FF1_IN);
            auto t2 = clk::now();
            auto y = linear_layer(e, x, tab, DFF, N1, N2, C, EF_B_FF1, L.b1.data(), C);
            cuda_sync();
            auto t3 = clk::now();
            auto g = gelu_candidates(e, y, s_conv, std::pow(2.0, (double)env_size("TP_F_SCALE_LOG2", 26)));
            y.clear();
            cuda_sync();
            auto t4 = clk::now();
            wire_bytes = tp_conv::c2m_server(ctx, pk, enc, rng, g.x_paired, b_h1, wire + "/h1x_c2m.bin", srv + "/h1x_s1.u64");
            wire_bytes += tp_conv::c2m_server(ctx, pk, enc, rng, g.f01, b_f, wire + "/h1f_c2m.bin", srv + "/h1f_s1.u64");
            auto t5 = clk::now();
            out << "m2c_finish_ms " << ms_since(t0, t1) << "\nwtab_ms " << ms_since(t1, t2) << "\nfhe_ms "
                << ms_since(t2, t3) << "\ngelu_poly_ms " << ms_since(t3, t4) << "\nc2m_ms " << ms_since(t4, t5)
                << "\nx_chain " << g.x_paired[0].chain_index() << "\nf_chain " << g.f01[0].chain_index() << "\n";
        } else if (cmd == "ff2") {
            auto x = tp_conv::m2c_server_finish(ctx, enc, wire + "/ff2_ct.bin", kept.at("ff2"));
            require_chain(x.at(0), cf2, "ff2 input");
            auto t1 = clk::now();
            auto tab = build_wtab_paired(L.W2.data(), DFF, D, C, -1, -1, EF_C_USED_FF2_IN);
            auto t2 = clk::now();
            auto y = linear_layer_paired(e, x, tab, D, EF_N1_FF2, EF_N2_FF2, EF_B_FF2, s_conv);
            cuda_sync();
            auto t3 = clk::now();
            wire_bytes = tp_conv::c2m_server(ctx, pk, enc, rng, y, b_h2, wire + "/h2_c2m.bin", srv + "/h2_s1.u64");
            auto t4 = clk::now();
            out << "m2c_finish_ms " << ms_since(t0, t1) << "\nwtab_ms " << ms_since(t1, t2) << "\nfhe_ms "
                << ms_since(t2, t3) << "\nc2m_ms " << ms_since(t3, t4) << "\nout_chain " << y[0].chain_index() << "\n";
        } else {
            throw std::runtime_error("unknown command " + cmd);
        }

        cuda_sync();
        out << "total_ms " << ms_since(t0, clk::now()) << "\nwire_bytes " << wire_bytes << "\nks_rots "
            << e.ks.rots - ks0.rots << "\nks_muls " << e.ks.muls_ctct - ks0.muls_ctct << "\nks_conj "
            << e.ks.conj - ks0.conj << "\npt_mults " << e.pt_mults - pm0 << "\nrescales " << e.rescales - rs0
            << "\nconversion " << tp_conv::backend_name() << "\n";
        const std::string resp = srv + "/resp_" + std::to_string(k);
        { std::ofstream f(resp + ".tmp"); f << out.str(); }
        std::rename((resp + ".tmp").c_str(), resp.c_str());
        ::unlink(req.c_str());
    }
}
