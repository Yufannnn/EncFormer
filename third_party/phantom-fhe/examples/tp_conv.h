#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "pipe_ckks_eval.h"
#include "pipe_ckks_keys.h"
#include "conv_hp.h"

namespace tp_conv {

using namespace pipe_ckks;

constexpr int F_BITS = 13;
constexpr int SIGMA = 40;
inline const char *backend_name() { return "c2m_m2c_rerandomized"; }

inline conv_hp::ConvConfig config(int slot_bound_log2) {
    conv_hp::ConvConfig c;
    c.F = F_BITS;
    c.conv_primes = 2;
    c.slot_bound_log2 = slot_bound_log2;
    return c;
}

inline void write_u64(const std::string &path, const std::vector<uint64_t> &v) {
    std::ofstream f(path + ".tmp", std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), (std::streamsize)(v.size() * 8));
    f.close();
    if (!f) throw std::runtime_error("cannot write " + path);
    std::rename((path + ".tmp").c_str(), path.c_str());
}

inline std::vector<uint64_t> read_u64(const std::string &path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<uint64_t> v(n);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize)(n * 8));
    if ((size_t)f.gcount() != n * 8) throw std::runtime_error("short read " + path);
    return v;
}

struct M2CKept { std::vector<conv_hp::M2CServerMsg::Kept> kept; size_t ncts = 0; };

inline size_t c2m_server(const PhantomContext &ctx, const PhantomPublicKey &pk, PhantomCKKSEncoder &enc,
                         conv_hp::Rng &rng, const std::vector<PhantomCiphertext> &cts, int slot_bound_log2,
                         const std::string &wire_path, const std::string &share_path) {
    const size_t n = enc.slot_count();
    std::vector<PhantomCiphertext> masked;
    masked.reserve(cts.size());
    std::vector<uint64_t> s1(cts.size() * 2 * n);
    for (size_t k = 0; k < cts.size(); k++) {
        auto o = conv_hp::c2m_server(ctx, pk, cts[k], rng, SIGMA, config(slot_bound_log2));
        std::copy(o.s1_re.begin(), o.s1_re.end(), s1.begin() + k * 2 * n);
        std::copy(o.s1_im.begin(), o.s1_im.end(), s1.begin() + k * 2 * n + n);
        masked.push_back(std::move(o.ct_masked));
    }
    size_t bytes = save_cts_wire(ctx, masked, wire_path + ".tmp", false);
    std::rename((wire_path + ".tmp").c_str(), wire_path.c_str());
    write_u64(share_path, s1);
    return bytes;
}

inline M2CKept m2c_server_mask(PhantomCKKSEncoder &enc, conv_hp::Rng &rng, const std::string &share_path,
                               size_t ncts, int slot_bound_log2, const std::string &wire_mask_path) {
    const size_t n = enc.slot_count();
    auto s1 = read_u64(share_path, ncts * 2 * n);
    std::vector<uint64_t> to_client(ncts * 2 * n);
    M2CKept kept;
    kept.ncts = ncts;
    for (size_t k = 0; k < ncts; k++) {
        auto m = conv_hp::m2c_server_mask(&s1[k * 2 * n], &s1[k * 2 * n + n], n, rng, SIGMA, config(slot_bound_log2));
        std::copy(m.to_client.re.begin(), m.to_client.re.end(), to_client.begin() + k * 2 * n);
        std::copy(m.to_client.im.begin(), m.to_client.im.end(), to_client.begin() + k * 2 * n + n);
        kept.kept.push_back(std::move(m.kept));
    }
    write_u64(wire_mask_path, to_client);
    return kept;
}

inline std::vector<PhantomCiphertext> m2c_server_finish(const PhantomContext &ctx, PhantomCKKSEncoder &,
                                                        const std::string &wire_ct_path, const M2CKept &kept) {
    auto cts = load_cts_wire(ctx, wire_ct_path, true);
    if (cts.size() != kept.ncts) throw std::runtime_error("m2c_server_finish: ciphertext count mismatch");
    for (size_t k = 0; k < cts.size(); k++) conv_hp::m2c_server_finish(ctx, cts[k], kept.kept[k]);
    return cts;
}

}
