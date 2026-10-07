#pragma once

#include "pipe_ckks_eval.h"

#define EF_PIPE_CKKS_CLIENT_INCLUDED 1
#ifdef EF_SERVER_ROLE
#error "pipe_ckks_client.h included in a server-role translation unit"
#endif

namespace pipe_ckks {

inline std::vector<PhantomCiphertext> encrypt_packed_pairs(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const double *x, int m, int d, int hp, int nslots, double scale,
    size_t chain_idx = 1, int c_in = -1)
{
    const int c = nslots / m;
    if (c_in < 0) c_in = c;
    const size_t slots = encoder.slot_count();
    std::vector<PhantomCiphertext> out(hp);
    for (int h = 0; h < hp; h++) {
        std::vector<pc64> msg(slots, pcplx(0.0));
        for (int col = 0; col < c_in; col++) {
            const int col_e = (2 * h) * c_in + col;
            const int col_o = (2 * h + 1) * c_in + col;
            for (int i = 0; i < m; i++) {
                double re = (col_e < d) ? x[(size_t)i * d + col_e] : 0.0;
                double im = (col_o < d) ? x[(size_t)i * d + col_o] : 0.0;
                msg[(size_t)col * m + i] = pcplx(re, im);
            }
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, chain_idx);
        sk.encrypt_symmetric(ctx, pt, out[h]);
    }
    return out;
}

inline std::vector<double> decrypt_blocks(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts, int m, int d_out, int nslots)
{
    const int c = nslots / m;
    std::vector<double> out((size_t)m * d_out, 0.0);
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        const int used = std::min(c, std::max(0, d_out - b * c));
        if (used <= 0) break;
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);
        for (int col = 0; col < used; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + (b * c + col)] = preal(dec[(size_t)col * m + i]);
    }
    return out;
}

inline std::vector<double> decrypt_blocks_complex_packed(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts, int m, int d_out, int nslots)
{
    const int c = nslots / m;
    std::vector<double> out((size_t)m * d_out, 0.0);
    int col_offset = 0;
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);

        int used_re = std::min(c, std::max(0, d_out - col_offset));
        for (int col = 0; col < used_re; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + col_offset + col] = preal(dec[(size_t)col * m + i]);
        col_offset += used_re;

        int used_im = std::min(c, std::max(0, d_out - col_offset));
        for (int col = 0; col < used_im; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + col_offset + col] = pimag(dec[(size_t)col * m + i]);
        col_offset += used_im;
    }
    return out;
}

inline PhantomCiphertext make_ct_zero_mul(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    double scale_mul, size_t chain_idx = 1)
{
    const size_t slots = encoder.slot_count();
    std::vector<pc64> zmsg(slots, pcplx(0.0));
    PhantomPlaintext pt;
    encoder.encode(ctx, zmsg, scale_mul, pt, chain_idx);
    PhantomCiphertext ct;
    sk.encrypt_symmetric(ctx, pt, ct);
    return ct;
}

inline std::vector<PhantomCiphertext> encrypt_real_blocks(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const double *x, int m, int d, int nslots, double scale,
    size_t chain_idx = 1)
{
    const int c = nslots / m;
    const int g_count = (d + c - 1) / c;
    const size_t slots = encoder.slot_count();
    std::vector<PhantomCiphertext> out(g_count);
    for (int g = 0; g < g_count; g++) {
        std::vector<pc64> msg(slots, pcplx(0.0));
        for (int col = 0; col < c; col++) {
            const int abs_col = g * c + col;
            for (int i = 0; i < m; i++)
                msg[(size_t)col * m + i] =
                    pcplx((abs_col < d) ? x[(size_t)i * d + abs_col] : 0.0, 0.0);
        }
        PhantomPlaintext pt;
        encoder.encode(ctx, msg, scale, pt, chain_idx);
        sk.encrypt_symmetric(ctx, pt, out[g]);
    }
    return out;
}

inline std::vector<double> decrypt_blocks_cusued(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, PhantomSecretKey &sk,
    const std::vector<PhantomCiphertext> &cts, int m, int d_out, int nslots, int c_used)
{
    const int c = nslots / m;
    std::vector<double> out((size_t)m * d_out, 0.0);
    for (int b = 0; b < static_cast<int>(cts.size()); b++) {
        const int used = std::min(c_used, std::max(0, d_out - b * c_used));
        if (used <= 0) break;
        PhantomPlaintext pt;
        std::vector<pc64> dec;
        sk.decrypt(ctx, cts[b], pt);
        encoder.decode(ctx, pt, dec);
        for (int col = 0; col < used; col++)
            for (int i = 0; i < m; i++)
                out[(size_t)i * d_out + (b * c_used + col)] = preal(dec[(size_t)col * m + i]);
    }
    return out;
}

}
