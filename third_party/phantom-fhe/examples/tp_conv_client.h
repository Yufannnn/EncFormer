#pragma once

#include "pipe_ckks_client.h"
#include "tp_conv.h"
#include "conv_hp_client.h"

namespace tp_conv {

inline void c2m_client(const PhantomContext &ctx, PhantomSecretKey &sk, PhantomCKKSEncoder &enc,
                       const std::string &wire_path, const std::string &share_path) {
    auto cts = load_cts_wire(ctx, wire_path, false);
    const size_t n = enc.slot_count();
    std::vector<uint64_t> s0(cts.size() * 2 * n);
    for (size_t k = 0; k < cts.size(); k++) {
        auto o = conv_hp::c2m_client(ctx, sk, cts[k], F_BITS);
        std::copy(o.s0_re.begin(), o.s0_re.end(), s0.begin() + k * 2 * n);
        std::copy(o.s0_im.begin(), o.s0_im.end(), s0.begin() + k * 2 * n + n);
    }
    write_u64(share_path, s0);
}

inline size_t m2c_client(const PhantomContext &ctx, PhantomSecretKey &sk, PhantomCKKSEncoder &enc,
                         const std::string &share_path, const std::string &wire_mask_path, size_t ncts,
                         size_t chain, double scale, const std::string &wire_ct_path) {
    const size_t n = enc.slot_count();
    auto s0 = read_u64(share_path, ncts * 2 * n);
    auto mk = read_u64(wire_mask_path, ncts * 2 * n);
    std::vector<PhantomCiphertext> cts;
    cts.reserve(ncts);
    for (size_t k = 0; k < ncts; k++) {
        conv_hp::M2CServerMsg::ToClient msg;
        msg.re.assign(mk.begin() + k * 2 * n, mk.begin() + k * 2 * n + n);
        msg.im.assign(mk.begin() + k * 2 * n + n, mk.begin() + (k + 1) * 2 * n);
        cts.push_back(conv_hp::m2c_client(ctx, sk, &s0[k * 2 * n], &s0[k * 2 * n + n], msg, chain, scale, F_BITS));
    }
    size_t bytes = save_cts_wire(ctx, cts, wire_ct_path + ".tmp", true);
    std::rename((wire_ct_path + ".tmp").c_str(), wire_ct_path.c_str());
    return bytes;
}

}
