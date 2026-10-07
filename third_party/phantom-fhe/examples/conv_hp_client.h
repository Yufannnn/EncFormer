#pragma once

#include "conv_hp.h"

namespace conv_hp {

inline std::vector<i128> c2m_client_lift(const PhantomContext &ctx, PhantomSecretKey &sk,
                                         const PhantomCiphertext &ct_masked) {
    PhantomPlaintext pt;
    sk.decrypt(ctx, ct_masked, pt);
    RnsLevel lvl(ctx, ct_masked.chain_index());
    std::vector<i128> t(lvl.poly_degree());
    conv_ntt_poly_to_int(ctx, lvl, pt.data(), t.data());
    return t;
}

struct C2MClientOut {
    std::vector<uint64_t> s0_re, s0_im;
};

inline C2MClientOut c2m_client(const PhantomContext &ctx, PhantomSecretKey &sk, const PhantomCiphertext &ct_masked,
                               int F = kF) {
    std::vector<i128> t = c2m_client_lift(ctx, sk, ct_masked);
    const HpFFT &fft = hp_fft(t.size());
    C2MClientOut out;
    out.s0_re.resize(t.size() / 2);
    out.s0_im.resize(t.size() / 2);
    fft.decode_round_mod64(t.data(), ct_masked.scale(), F,  false, out.s0_re.data(), out.s0_im.data());
    return out;
}

inline PhantomCiphertext m2c_client(const PhantomContext &ctx, PhantomSecretKey &sk, const uint64_t *s0_re,
                                    const uint64_t *s0_im, const M2CServerMsg::ToClient &msg, size_t chain_index,
                                    double scale, int F = kF) {
    RnsLevel lvl(ctx, chain_index);
    const size_t N = lvl.poly_degree();
    const size_t n = msg.re.size();
    if (msg.im.size() != n || n > N / 2) throw std::invalid_argument("m2c_client: bad message size");
    std::vector<int64_t> yr(n), yi(n);
    for (size_t j = 0; j < n; j++) {
        yr[j] = static_cast<int64_t>(s0_re[j] + msg.re[j]);
        yi[j] = static_cast<int64_t>(s0_im[j] + msg.im[j]);
    }
    std::vector<i128> coeffs(N);
    hp_fft(N).encode_fixed(yr.data(), yi.data(), n, scale, F, coeffs.data());

    const double lim = std::ldexp(1.0, static_cast<int>(lvl.log2_q()) - 2);
    for (size_t c = 0; c < N; c++)
        if (std::fabs(static_cast<double>(coeffs[c])) >= lim)
            throw std::runtime_error("m2c_client: encoded coefficients exceed q/4 at the target level");
    auto d = conv_int_poly_to_ntt(ctx, lvl, coeffs.data());
    PhantomPlaintext pt;
    pt.resize(lvl.size(), N, cudaStreamPerThread);
    pt.set_chain_index(chain_index);
    conv_check_cuda(cudaMemcpyAsync(pt.data(), d.get(), lvl.size() * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                                    cudaStreamPerThread),
                    "m2c_client D2D");
    PhantomCiphertext ct;
    sk.encrypt_symmetric(ctx, pt, ct);
    ct.set_scale(scale);
    conv_check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "m2c_client");
    return ct;
}

}
