// pipe_ckks_ff1.cu — Native CKKS FF1 (feed-forward layer 1) with file I/O.
//
// Reads:  PIPE_DIR/ff1_in.bin  = X[M×D1] ++ W1[D1×DMID]  (row-major float64)
// Writes: PIPE_DIR/ff1_out.bin = H[M×DMID]                (row-major float64)
//
// FHE: encrypt_packed_pairs → build_babies → linear_complex_paired → (complex-packed decrypt)
// Skips ct_real_blocks for force_real=False optimization (saves 1 CKKS level).
// Output uses complex-packed decryption: real parts → even columns, imag parts → odd columns.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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
constexpr int N1         = EF_N1_DEFAULT;
constexpr int N2         = EF_N2_DEFAULT;
constexpr int D1         = EF_D;
constexpr int DMID       = EF_D_FF;
constexpr int HP         = EF_HP_FF1;
constexpr int BLOCKS_OUT = EF_B_FF1;
constexpr int C_IN_FF1   = EF_C_USED_FF1_IN;
} // namespace

// Tiny file-existence helper; uses POSIX stat so we don't pull <filesystem>.
static bool file_exists(const std::string &p) {
    struct stat s; return ::stat(p.c_str(), &s) == 0;
}

static void touch_empty(const std::string &p) {
    std::ofstream f(p); /* close on dtor */
}

// --- One iteration of FF1: read input, encrypt, compute, decrypt, write.
// Returns metrics; throws on I/O error.
template <class Ctx, class Enc, class GK, class SK>
static void run_ff1_once(
    Ctx &ctx, Enc &encoder, GK &gk, SK &sk,
    const PhantomCiphertext &ct_zero_mul,
    int NSLOTS_, int M_, int D1_, int DMID_, int HP_, int N1_, int N2_, int C_,
    int C_IN_FF1_, double scale_in, double scale_w, uint32_t conj_elt,
    const std::string &in_path, const std::string &out_path,
    std::ostream &log
) {
    auto t_io0 = std::chrono::high_resolution_clock::now();
    const size_t N_X  = (size_t)M_  * D1_;
    const size_t N_W1 = (size_t)D1_ * DMID_;
    std::vector<double> buf(N_X + N_W1);
    pipe_io::read_f64(in_path.c_str(), buf.data(), buf.size());
    const double *X  = buf.data();
    const double *W1 = buf.data() + N_X;
    auto t_io1 = std::chrono::high_resolution_clock::now();

    auto t_enc0 = std::chrono::high_resolution_clock::now();
    auto x_ct = encrypt_packed_pairs(ctx, encoder, sk, X, M_, D1_, HP_, NSLOTS_,
                                      scale_in, 1, C_IN_FF1_);
    cuda_sync();
    auto t_enc1 = std::chrono::high_resolution_clock::now();

    auto W1_tab = build_wtab_paired(W1, D1_, DMID_, C_, -1, -1, C_IN_FF1_);

    KSCounters ks;
    auto t_fhe0 = std::chrono::high_resolution_clock::now();
    auto babies = build_babies(ctx, gk, x_ct, N1_, M_, NSLOTS_, ks);
    auto h_ct = linear_complex_paired(ctx, encoder, gk, babies, W1_tab,
                                      DMID_, M_, N1_, N2_, C_, NSLOTS_,
                                      scale_w, ct_zero_mul, ks);
    ct_real_blocks(ctx, gk, h_ct, static_cast<size_t>(conj_elt), ks);
    cuda_sync();
    auto t_fhe1 = std::chrono::high_resolution_clock::now();

    auto t_dec0 = std::chrono::high_resolution_clock::now();
    auto h_out = decrypt_blocks(ctx, encoder, sk, h_ct, M_, DMID_, NSLOTS_);
    auto t_dec1 = std::chrono::high_resolution_clock::now();

    auto t_wrt0 = std::chrono::high_resolution_clock::now();
    pipe_io::write_f64(out_path.c_str(), h_out.data(), h_out.size());
    auto t_wrt1 = std::chrono::high_resolution_clock::now();

    auto h_ref = matmul_ref(X, M_, D1_, W1, DMID_);
    double err = rel_err_real(h_out, h_ref);
    double mse = mse_real(h_out, h_ref);

    log << std::setprecision(10);
    log << "stage pipe_ff1\n";
    log << "encrypt_ms "  << ms_since(t_enc0, t_enc1) << "\n";
    log << "fhe_ms "      << ms_since(t_fhe0, t_fhe1) << "\n";
    log << "decrypt_ms "  << ms_since(t_dec0, t_dec1) << "\n";
    log << "read_ms "     << ms_since(t_io0, t_io1) << "\n";
    log << "write_ms "    << ms_since(t_wrt0, t_wrt1) << "\n";
    log << "rel_err "     << std::scientific << err << std::defaultfloat << "\n";
    log << "mse "         << std::scientific << mse << std::defaultfloat << "\n";
    log << "ks_rots "     << ks.rots << "\n";
    log << "ks_conj "     << ks.conj << "\n";
}

int main(int argc, char **argv) {
    bool server_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0) server_mode = true;
    }
    if (!server_mode) {
        const char *env = std::getenv("EF_SERVER_MODE");
        if (env && env[0] == '1') server_mode = true;
    }

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

    // --- One-time CKKS setup (shared across all iterations in server mode) ---
    auto t_setup0 = std::chrono::high_resolution_clock::now();
    EncryptionParameters parms(scheme_type::ckks);
    set_lite_ckks_pipeline_params(parms, NPOLY);
    auto galois_elts = build_galois_elts_full_columns(NSLOTS, M, GEN_ROT, MMOD);
    parms.set_galois_elts(galois_elts);

    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    PhantomCKKSEncoder encoder(ctx);
    auto ct_zero_mul = make_ct_zero_mul(ctx, encoder, sk, scale_mul);
    cuda_sync();
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    const uint32_t conj_elt = static_cast<uint32_t>(MMOD - 1);
    std::string pd = pipe_io::get_pipe_dir();

    if (!server_mode) {
        // ---- legacy single-shot ----
        std::string in_path  = pd + "/ff1_in.bin";
        std::string out_path = pd + "/ff1_out.bin";
        std::cout << "setup_ms "     << ms_since(t_setup0, t_setup1) << "\n";
        std::cout << "galois_elts "  << galois_elts.size() << "\n";
        run_ff1_once(ctx, encoder, gk, sk, ct_zero_mul,
                     NSLOTS, M, D1, DMID, HP, N1, N2, C, C_IN_FF1,
                     scale_in, scale_w, conj_elt,
                     in_path, out_path, std::cout);
        return 0;
    }

    // ---- server mode ----
    // Print setup metrics once on stderr so python can capture them.
    std::cerr << "[ff1_server] setup_ms " << ms_since(t_setup0, t_setup1)
              << " galois_elts " << galois_elts.size() << " ready\n";
    std::string ready_path = pd + "/ff1_server_ready";
    touch_empty(ready_path);

    std::string shutdown_path = pd + "/ff1_shutdown";
    int iter = 0;
    while (true) {
        // Wait for next request.  Python writes ff1_in_<n>.bin then touches
        // ff1_request_<n>; server consumes both and writes ff1_out_<n>.bin
        // then touches ff1_response_<n>.  Iter id is monotonically increasing.
        std::string req = pd + "/ff1_request_" + std::to_string(iter);
        while (!file_exists(req)) {
            if (file_exists(shutdown_path)) {
                std::cerr << "[ff1_server] shutdown after " << iter << " iters\n";
                return 0;
            }
            usleep(500); // 0.5 ms poll
        }

        std::string in_path  = pd + "/ff1_in_"  + std::to_string(iter) + ".bin";
        std::string out_path = pd + "/ff1_out_" + std::to_string(iter) + ".bin";
        std::string log_path = pd + "/ff1_log_" + std::to_string(iter) + ".txt";

        std::ofstream log(log_path);
        run_ff1_once(ctx, encoder, gk, sk, ct_zero_mul,
                     NSLOTS, M, D1, DMID, HP, N1, N2, C, C_IN_FF1,
                     scale_in, scale_w, conj_elt,
                     in_path, out_path, log);
        log.close();
        ::unlink(req.c_str());
        touch_empty(pd + "/ff1_response_" + std::to_string(iter));
        ++iter;
    }
}
