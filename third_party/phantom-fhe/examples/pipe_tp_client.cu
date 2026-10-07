#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "pipe_io.h"
#include "pipe_ckks_client.h"
#include "pipe_ckks_keys.h"
#include "tp_conv_client.h"

using namespace pipe_ckks;

namespace {
#include "model_config.h"
constexpr int NSLOTS = EF_NSLOTS, M = EF_M, H = EF_H;
using clk = std::chrono::high_resolution_clock;
}

int main() {
    const std::string dir = pipe_io::get_pipe_dir_or("TP_DIR");
    const std::string wire = dir + "/wire", cli = dir + "/cli";
    if (cudaSetDevice(0) != cudaSuccess) { std::cerr << "no GPU\n"; return 1; }
    constexpr size_t NPOLY = (size_t)NSLOTS * 2;
    constexpr uint64_t MMOD = 2ull * NPOLY;

    auto t0 = clk::now();
    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);
    parms.set_galois_elts(merge_galois_elts(
        merge_galois_elts(build_galois_elts_full_columns(NSLOTS, M, 5, MMOD),
                          build_galois_elts_score(NSLOTS, M, EF_B_FOLD, EF_G_FOLD, EF_C_USED_QK, H, 5, MMOD)),
        build_galois_elts_within_row(NSLOTS, M, 5, MMOD)));
    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomCKKSEncoder enc(ctx);
    size_t key_bytes = 0;
    ::mkdir((wire + "/keys").c_str(), 0700);
    params_from(parms).save(wire + "/keys/params.txt");
    {
        PhantomPublicKey pk = sk.gen_publickey(ctx);
        key_bytes += save_obj(pk, wire + "/keys/pk.bin");
    }
    {
        PhantomRelinKey rk = sk.gen_relinkey(ctx);
        key_bytes += save_obj(rk, wire + "/keys/rlk.bin");
    }
    {
        PhantomGaloisKey gk = sk.create_galois_keys(ctx);
        key_bytes += save_obj(gk, wire + "/keys/gk.bin");
    }
    cuda_sync();
    const double setup_ms = ms_since(t0, clk::now());
    std::ofstream(wire + "/keys/ready") << "key_bytes " << key_bytes << "\n";
    std::ofstream(cli + "/client_ready") << "setup_ms " << setup_ms << "\nkey_bytes " << key_bytes
                                         << "\nconversion " << tp_conv::backend_name() << "\n";

    for (int k = 0;; k++) {
        const std::string req = cli + "/req_" + std::to_string(k);
        while (!pipe_io::file_present(req)) { pipe_io::exit_if_orphaned(req); usleep(200); }
        std::string cmd, name;
        size_t ncts = 0, chain = 0;
        { std::ifstream f(req); f >> cmd >> name >> ncts >> chain; }
        std::ostringstream out;
        out << std::setprecision(10) << "cmd " << cmd << "\n";
        auto t1 = clk::now();
        size_t wire_bytes = 0;
        if (cmd == "shutdown") {
            std::ofstream(cli + "/resp_" + std::to_string(k)) << out.str();
            return 0;
        } else if (cmd == "c2m") {
            tp_conv::c2m_client(ctx, sk, enc, wire + "/" + name + "_c2m.bin", cli + "/" + name + "_s0.u64");
        } else if (cmd == "m2c") {
            wire_bytes = tp_conv::m2c_client(ctx, sk, enc, cli + "/" + name + "_s0.u64", wire + "/" + name + "_mask.u64",
                                             ncts, chain, std::pow(2.0, 40), wire + "/" + name + "_ct.bin");
        } else {
            throw std::runtime_error("unknown command " + cmd);
        }
        cuda_sync();
        out << "total_ms " << ms_since(t1, clk::now()) << "\nwire_bytes " << wire_bytes << "\n";
        const std::string resp = cli + "/resp_" + std::to_string(k);
        { std::ofstream f(resp + ".tmp"); f << out.str(); }
        std::rename((resp + ".tmp").c_str(), resp.c_str());
        ::unlink(req.c_str());
    }
}
