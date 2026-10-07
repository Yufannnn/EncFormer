#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "phantom.h"
#include "uintmodmath.cuh"
#include "pipe_ckks_eval.h"

namespace pipe_ckks {

__global__ void fused_mul_plain_acc_kernel(const uint64_t *const *cts, const uint64_t *const *pts, int K,
                                           const DModulus *mod, uint64_t *out, size_t N, size_t L) {
    const size_t total = N * L;
    for (size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x; tid < total; tid += (size_t)blockDim.x * gridDim.x) {
        const size_t l = tid / N;
        const uint64_t q = mod[l].value();
        const uint64_t *ratio = mod[l].const_ratio();
        uint64_t a0 = 0, a1 = 0;
        for (int k = 0; k < K; k++) {
            const uint64_t p = pts[k][tid];
            uint64_t m0 = phantom::arith::multiply_and_barrett_reduce_uint64(cts[k][tid], p, q, ratio);
            uint64_t m1 = phantom::arith::multiply_and_barrett_reduce_uint64(cts[k][total + tid], p, q, ratio);
            a0 += m0; a0 -= (a0 >= q) ? q : 0;
            a1 += m1; a1 -= (a1 >= q) ? q : 0;
        }
        out[tid] = a0;
        out[total + tid] = a1;
    }
}

class FusedMulPlainAcc {
public:

    void run(const PhantomContext &ctx, const std::vector<const PhantomCiphertext *> &cts,
             const std::vector<const PhantomPlaintext *> &pts, PhantomCiphertext &out) {
        const int K = static_cast<int>(cts.size());
        if (K == 0 || pts.size() != cts.size()) throw std::invalid_argument("fused_mul_plain_acc: bad sizes");
        const auto &c0 = *cts[0];
        if (c0.size() != 2) throw std::invalid_argument("fused_mul_plain_acc: ciphertexts must have 2 polys");
        const size_t ci = c0.chain_index();
        const auto &parms = ctx.get_context_data(ci).parms();
        const size_t N = parms.poly_modulus_degree(), L = parms.coeff_modulus().size();
        std::vector<const uint64_t *> hc(K), hp(K);
        for (int k = 0; k < K; k++) {
            if (cts[k]->chain_index() != ci || pts[k]->chain_index() != ci)
                throw std::invalid_argument("fused_mul_plain_acc: chain index mismatch");
            if (cts[k]->scale() != c0.scale() || pts[k]->scale() != pts[0]->scale())
                throw std::invalid_argument("fused_mul_plain_acc: scale mismatch");
            if (!cts[k]->is_ntt_form()) throw std::invalid_argument("fused_mul_plain_acc: NTT form expected");
            hc[k] = cts[k]->data();
            hp[k] = pts[k]->data();
        }
        ensure(K);
        cudaMemcpyAsync(d_cts_, hc.data(), K * sizeof(void *), cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(d_pts_, hp.data(), K * sizeof(void *), cudaMemcpyHostToDevice, cudaStreamPerThread);
        out = c0;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((N * L + threads - 1) / threads);
        fused_mul_plain_acc_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
            d_cts_, d_pts_, K, ctx.gpu_rns_tables().modulus(), out.data(), N, L);
        auto e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        out.set_scale(c0.scale() * pts[0]->scale());
    }

    ~FusedMulPlainAcc() {
        if (d_cts_) cudaFree(d_cts_);
        if (d_pts_) cudaFree(d_pts_);
    }

private:
    void ensure(int K) {
        if (K <= cap_) return;
        if (d_cts_) cudaFree(d_cts_);
        if (d_pts_) cudaFree(d_pts_);
        cudaMalloc(&d_cts_, K * sizeof(void *));
        cudaMalloc(&d_pts_, K * sizeof(void *));
        cap_ = K;
    }
    const uint64_t **d_cts_ = nullptr;
    const uint64_t **d_pts_ = nullptr;
    int cap_ = 0;
};

__global__ void fill_seg_const_kernel(cuDoubleComplex *v, const cuDoubleComplex *row, int c, int m, int shift) {
    const size_t total = (size_t)c * m;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        const int seg = static_cast<int>(idx / m);
        v[idx] = row[((seg - shift) % c + c) % c];
    }
}

inline std::vector<PhantomCiphertext> linear_complex_paired_dev(
    const PhantomContext &ctx, PhantomCKKSEncoder &encoder, const PhantomGaloisKey &gk,
    const std::vector<std::vector<PhantomCiphertext>> &gp, const std::vector<cuDoubleComplex> &wtab,
    int d_out, int m, int n1, int n2, int c, int nslots, double scale_w, KSCounters &ks,
    int blocks_override, size_t chain_idx)
{
    const int hp = static_cast<int>(gp.size());
    const int blocks = (blocks_override > 0) ? blocks_override : (d_out / c);
    const size_t slots = encoder.slot_count();
    if ((size_t)c * m != slots) throw std::invalid_argument("linear_complex_paired_dev: c*m must equal slot count");

    double maxmod = 0.0;
    for (const auto &w : wtab) maxmod = std::max(maxmod, std::hypot(w.x, w.y));
    const int max_bits = std::max(1, static_cast<int>(std::ceil(std::log2(std::max(scale_w * maxmod, 2.0)))) + 1);

    cuDoubleComplex *d_tab = nullptr, *d_vec = nullptr;
    cudaMallocAsync(&d_tab, wtab.size() * sizeof(cuDoubleComplex), cudaStreamPerThread);
    cudaMallocAsync(&d_vec, slots * sizeof(cuDoubleComplex), cudaStreamPerThread);
    cudaMemcpyAsync(d_tab, wtab.data(), wtab.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, cudaStreamPerThread);
    const unsigned th = 256, bl = static_cast<unsigned>((slots + th - 1) / th);

    std::vector<std::vector<PhantomCiphertext>> cf(blocks, std::vector<PhantomCiphertext>(n2));
    std::vector<std::vector<char>> cf_set(blocks, std::vector<char>(n2, 0));
    PhantomPlaintext pt;
    for (int b = 0; b < blocks; b++)
        for (int p = 0; p < n2; p++) {
            const int p_shift = (p * n1) % c;
            for (int q = 0; q < n1; q++) {
                const int t = (p_shift + q) % c;
                bool has_h = false;
                PhantomCiphertext h_acc;
                for (int h = 0; h < hp; h++) {
                    fill_seg_const_kernel<<<bl, th, 0, cudaStreamPerThread>>>(d_vec, d_tab + tab_idx(h, t, b, 0, c, blocks),
                                                                            c, m, p_shift);
                    encoder.encode_device(ctx, d_vec, slots, scale_w, pt, chain_idx, max_bits);
                    PhantomCiphertext term = gp[h][q];
                    multiply_plain_inplace(ctx, term, pt);
                    if (!has_h) { h_acc = std::move(term); has_h = true; }
                    else add_inplace(ctx, h_acc, term);
                }
                if (!has_h) continue;
                if (!cf_set[b][p]) { cf[b][p] = std::move(h_acc); cf_set[b][p] = 1; }
                else add_inplace(ctx, cf[b][p], h_acc);
            }
        }
    cudaFreeAsync(d_vec, cudaStreamPerThread);
    cudaFreeAsync(d_tab, cudaStreamPerThread);

    std::vector<PhantomCiphertext> out(blocks);
    for (int b = 0; b < blocks; b++) {
        bool inited = false;
        PhantomCiphertext acc;
        for (int p = 0; p < n2; p++) {
            if (!cf_set[b][p]) continue;
            PhantomCiphertext term = cf[b][p];
            if (p != 0) {
                rotate_inplace(ctx, term, norm_step(p * n1 * m, nslots), gk);
                ks.rots += 1;
            }
            if (!inited) { acc = std::move(term); inited = true; }
            else add_inplace(ctx, acc, term);
        }
        if (inited) out[b] = std::move(acc);
    }
    return out;
}

}
