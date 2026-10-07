#pragma once

#include <array>
#include <complex>
#include <map>
#include <tuple>

#include "pipe_ckks_eval.h"
#include "fused_ops.cuh"

namespace pipe_ckks {

struct Env {
    const PhantomContext &ctx;
    PhantomCKKSEncoder &enc;
    const PhantomGaloisKey &gk;
    const PhantomRelinKey &rk;
    int nslots;
    int m;
    uint32_t conj_elt;
    KSCounters ks{};
    long long pt_mults = 0;
    long long rescales = 0;
    bool fused = true;
    FusedMulPlainAcc fma{};
    bool device_encode = true;

    size_t slots() const { return enc.slot_count(); }
    int segs() const { return nslots / m; }
};

inline PhantomPlaintext encode_at(Env &e, const std::vector<pc64> &v, size_t ci, double scale) {
    PhantomPlaintext pt;
    e.enc.encode(e.ctx, v, scale, pt, ci);
    return pt;
}

inline void rescale(Env &e, PhantomCiphertext &ct) {
    rescale_to_next_inplace(e.ctx, ct);
    e.rescales++;
}

inline PhantomCiphertext rotated(Env &e, const PhantomCiphertext &ct, int step) {
    PhantomCiphertext r = ct;
    step = norm_step(step, e.nslots);
    if (step != 0) {
        rotate_inplace(e.ctx, r, step, e.gk);
        e.ks.rots++;
    }
    return r;
}

inline void mul_plain(Env &e, PhantomCiphertext &ct, const PhantomPlaintext &pt) {
    multiply_plain_inplace(e.ctx, ct, pt);
    e.pt_mults++;
}

inline void acc_add(Env &e, PhantomCiphertext &acc, bool &inited, PhantomCiphertext &&term) {
    if (!inited) { acc = std::move(term); inited = true; }
    else add_inplace(e.ctx, acc, term);
}

inline PhantomCiphertext conj(Env &e, const PhantomCiphertext &ct) {
    PhantomCiphertext c = ct;
    apply_galois_inplace(e.ctx, c, e.conj_elt, e.gk);
    e.ks.conj++;
    return c;
}

class MaskCache {
public:
    using Key = std::tuple<int, int, size_t, int>;
    template <class F>
    const PhantomPlaintext &get(const Key &k, F make) {
        auto it = cache_.find(k);
        if (it == cache_.end()) it = cache_.emplace(k, make()).first;
        return it->second;
    }
    size_t size() const { return cache_.size(); }
private:
    std::map<Key, PhantomPlaintext> cache_;
};

enum MaskKind { MK_HEAD = 1, MK_TAIL, MK_SEGS, MK_COL, MK_WIN };

inline pc64 tag_value(int tag) { return tag == 0 ? pcplx(1.0) : (tag == 1 ? pcplx(0.0, 1.0) : pcplx(0.0, -1.0)); }

inline PhantomCiphertext rot_within_rs(Env &e, MaskCache &mc, const PhantomCiphertext &ct, int t, int nseg,
                                       int value_tag, std::map<int, PhantomCiphertext> *rot_cache = nullptr) {
    const int m = e.m;
    t = ((t % m) + m) % m;
    const size_t ci = ct.chain_index();
    const double q = dropped_prime_at(e.ctx, ci);
    const size_t slots = e.slots();
    if (t == 0) {
        if (value_tag == 0 && nseg == e.segs()) {
            PhantomCiphertext r = ct;
            mod_switch_to_next_inplace(e.ctx, r);
            return r;
        }
        const auto &pt = mc.get({MK_SEGS, nseg, ci, value_tag}, [&] {
            std::vector<pc64> v(slots, pcplx(0.0));
            for (size_t i = 0; i < (size_t)nseg * m; i++) v[i] = tag_value(value_tag);
            return encode_at(e, v, ci, q);
        });
        PhantomCiphertext r = ct;
        mul_plain(e, r, pt);
        rescale(e, r);
        return r;
    }
    auto get_rot = [&](int step) -> PhantomCiphertext {
        if (!rot_cache) return rotated(e, ct, step);
        auto it = rot_cache->find(step);
        if (it == rot_cache->end()) it = rot_cache->emplace(step, rotated(e, ct, step)).first;
        return it->second;
    };
    PhantomCiphertext r1 = get_rot(t), r2 = get_rot(t - m);
    const auto &hp = mc.get({MK_HEAD, t + 1000 * nseg, ci, value_tag}, [&] {
        std::vector<pc64> v(slots, pcplx(0.0));
        for (int s = 0; s < nseg; s++)
            for (int i = 0; i < m - t; i++) v[(size_t)s * m + i] = tag_value(value_tag);
        return encode_at(e, v, ci, q);
    });
    const auto &tp = mc.get({MK_TAIL, t + 1000 * nseg, ci, value_tag}, [&] {
        std::vector<pc64> v(slots, pcplx(0.0));
        for (int s = 0; s < nseg; s++)
            for (int i = m - t; i < m; i++) v[(size_t)s * m + i] = tag_value(value_tag);
        return encode_at(e, v, ci, q);
    });
    mul_plain(e, r1, hp);
    mul_plain(e, r2, tp);
    add_inplace(e.ctx, r1, r2);
    rescale(e, r1);
    return r1;
}

using Babies = std::vector<std::vector<PhantomCiphertext>>;

inline Babies babies_of(Env &e, const std::vector<PhantomCiphertext> &x, int n1) {
    return build_babies(e.ctx, e.gk, x, n1, e.m, e.nslots, e.ks);
}

inline std::vector<PhantomCiphertext> linear_layer(Env &e, const Babies &babies,
                                                   const std::vector<pc64> &wtab, int d_out, int n1, int n2,
                                                   int c_used, int blocks, const double *bias, int bias_c_used,
                                                   double in_scale_override = 0.0) {
    const auto &x0 = babies.at(0).at(0);
    const size_t ci = x0.chain_index();
    const double q = dropped_prime_at(e.ctx, ci);

    const double s_in = x0.scale();
    const double target = (in_scale_override > 0.0) ? in_scale_override : s_in;
    const double w_scale = q * target / s_in;
    auto y = e.device_encode
                 ? linear_complex_paired_dev(e.ctx, e.enc, e.gk, babies, wtab, d_out, e.m, n1, n2, e.segs(), e.nslots,
                                             w_scale, e.ks, blocks, ci)
                 : linear_complex_paired(e.ctx, e.enc, e.gk, babies, wtab, d_out, e.m, n1, n2, e.segs(), e.nslots,
                                         w_scale, e.ks, blocks, ci);
    e.pt_mults += (long long)blocks * n1 * n2 * (long long)babies.size();
    ct_real_blocks(e.ctx, e.gk, y, static_cast<size_t>(e.conj_elt), e.ks);
    if (bias)
        add_bias_blocks(e.ctx, e.enc, y, bias, d_out, e.m, e.nslots, bias_c_used, y[0].scale(), ci);
    for (auto &ct : y) rescale(e, ct);
    return y;
}

inline std::vector<PhantomCiphertext> linear_layer(Env &e, const std::vector<PhantomCiphertext> &x,
                                                   const std::vector<pc64> &wtab, int d_out, int n1, int n2,
                                                   int c_used, int blocks, const double *bias, int bias_c_used,
                                                   double in_scale_override = 0.0) {
    return linear_layer(e, babies_of(e, x, n1), wtab, d_out, n1, n2, c_used, blocks, bias, bias_c_used,
                        in_scale_override);
}

inline PhantomCiphertext times_i(Env &e, const PhantomCiphertext &ct) {
    std::vector<pc64> v(e.slots(), pcplx(0.0, 1.0));
    PhantomCiphertext r = ct;
    mul_plain(e, r, encode_at(e, v, ct.chain_index(), 1.0));
    return r;
}

inline std::vector<PhantomCiphertext> linear_layer_paired(Env &e, const Babies &babies, const std::vector<pc64> &wtab,
                                                          int d_out, int n1, int n2, int blocks, double target) {
    const auto &x0 = babies.at(0).at(0);
    const size_t ci = x0.chain_index();
    const double q = dropped_prime_at(e.ctx, ci);
    const double w_scale = q * target / x0.scale();
    auto y = e.device_encode
                 ? linear_complex_paired_dev(e.ctx, e.enc, e.gk, babies, wtab, d_out, e.m, n1, n2, e.segs(), e.nslots,
                                             w_scale, e.ks, blocks, ci)
                 : linear_complex_paired(e.ctx, e.enc, e.gk, babies, wtab, d_out, e.m, n1, n2, e.segs(), e.nslots,
                                         w_scale, e.ks, blocks, ci);
    e.pt_mults += (long long)blocks * n1 * n2 * (long long)babies.size();
    std::vector<PhantomCiphertext> z;
    for (size_t u = 0; 2 * u < y.size(); u++) {
        if (2 * u + 1 >= y.size()) {
            PhantomCiphertext c = y[2 * u];
            add_inplace(e.ctx, c, conj(e, y[2 * u]));
            rescale(e, c);
            z.push_back(std::move(c));
            continue;
        }
        PhantomCiphertext io = times_i(e, y[2 * u + 1]);
        PhantomCiphertext a = y[2 * u], b = y[2 * u];
        add_inplace(e.ctx, a, io);
        sub_inplace(e.ctx, b, io);
        add_inplace(e.ctx, a, conj(e, b));
        rescale(e, a);
        z.push_back(std::move(a));
    }
    return z;
}

inline std::vector<PhantomCiphertext> linear_layer_paired(Env &e, const std::vector<PhantomCiphertext> &x,
                                                          const std::vector<pc64> &wtab, int d_out, int n1, int n2,
                                                          int blocks, double target) {
    return linear_layer_paired(e, babies_of(e, x, n1), wtab, d_out, n1, n2, blocks, target);
}

struct ScoreGeom {
    int H, m, b_fold, g_fold, c_used_qk;
    int half() const { return m / 2; }
    int blen() const { return H * m; }
};

struct MapEntry2 { int j; int s; };

inline std::vector<MapEntry2> score_map(const ScoreGeom &g) {
    std::vector<MapEntry2> mp(g.half(), {-1, -1});
    std::vector<char> seen(g.half(), 0);
    int count = 0;
    for (int j = 0; j < g.g_fold && count < g.half(); j++)
        for (int s = 0; s < g.b_fold && count < g.half(); s++) {
            int t = ((j * g.b_fold - s) % g.m + g.m) % g.m;
            if (t < g.half() && !seen[t]) { seen[t] = 1; mp[t] = {j, s}; count++; }
        }
    return mp;
}

struct ScoreSlot { int ct, slot, h, row, col_re, col_im; };

inline std::vector<ScoreSlot> score_unpack_index(const ScoreGeom &g, int nslots) {
    auto mp = score_map(g);
    std::vector<ScoreSlot> out;
    out.reserve((size_t)g.half() * g.H * g.m);
    for (int t = 0; t < g.half(); t++)
        for (int h = 0; h < g.H; h++)
            for (int i = 0; i < g.m; i++) {
                long p = (long)t * g.blen() + (long)h * g.m + i;
                int dst = (i + mp[t].s) % g.m;
                out.push_back({(int)(p / nslots), (int)(p % nslots), h, dst, (dst + t) % g.m,
                               (dst + g.half() + t) % g.m});
            }
    return out;
}

inline int score_packed_cts(const ScoreGeom &g, int nslots) {
    return (int)(((long)g.half() * g.blen() + nslots - 1) / nslots);
}

inline std::vector<PhantomCiphertext> score_kernel(Env &e, MaskCache &mc, const ScoreGeom &g,
                                                   const std::vector<PhantomCiphertext> &Q,
                                                   const std::vector<PhantomCiphertext> &K, double out_scale = 0.0,
                                                   const double *row_scale = nullptr, double add_c = 0.0) {

    const int blocks = (int)Q.size();
    const double s_in = Q[0].scale();
    const double s_out = out_scale > 0.0 ? out_scale : s_in;
    auto mp = score_map(g);

    std::vector<std::vector<PhantomCiphertext>> qb(blocks), kb(blocks), kbh(blocks);
    for (int b = 0; b < blocks; b++) {
        std::map<int, PhantomCiphertext> qc, kc;
        qb[b].resize(g.b_fold);
        for (int s = 0; s < g.b_fold; s++) qb[b][s] = rot_within_rs(e, mc, Q[b], s, g.c_used_qk, 0, &qc);
        kb[b].resize(g.g_fold);
        kbh[b].resize(g.g_fold);
        for (int j = 0; j < g.g_fold; j++) {
            kb[b][j] = rot_within_rs(e, mc, K[b], j * g.b_fold, g.c_used_qk, 0, &kc);
            kbh[b][j] = rot_within_rs(e, mc, K[b], ((j + g.g_fold / 2) % g.g_fold) * g.b_fold, g.c_used_qk, 1, &kc);
        }
    }

    std::vector<PhantomCiphertext> dfold(g.half());
    for (int t = 0; t < g.half(); t++) {
        PhantomCiphertext acc;
        bool inited = false;
        for (int b = 0; b < blocks; b++) {
            PhantomCiphertext combo = kb[b][mp[t].j];
            add_inplace(e.ctx, combo, kbh[b][mp[t].j]);
            PhantomCiphertext term = qb[b][mp[t].s];
            multiply_and_relin_inplace(e.ctx, term, combo, e.rk);
            e.ks.muls_ctct++;
            acc_add(e, acc, inited, std::move(term));
        }
        rescale(e, acc);
        dfold[t] = std::move(acc);
    }
    qb.clear(); kb.clear(); kbh.clear();

    const int U = (g.c_used_qk + g.H - 1) / g.H;
    const int npk = score_packed_cts(g, e.nslots);
    std::vector<PhantomCiphertext> packed(npk);
    std::vector<char> pinit(npk, 0);
    const size_t slots = e.slots();
    for (int t = 0; t < g.half(); t++) {
        PhantomCiphertext z = dfold[t];
        for (int d = 1; d < U; d++) {
            PhantomCiphertext r = rotated(e, dfold[t], d * g.H * g.m);
            add_inplace(e.ctx, z, r);
        }
        const long gofs = (long)t * g.blen();
        const int k0 = (int)(gofs / e.nslots);
        const int off = (int)(gofs % e.nslots);
        if (off) z = rotated(e, z, -off);
        const size_t ci = z.chain_index();
        const double mscale = dropped_prime_at(e.ctx, ci) * s_out / z.scale();
        for (int part = 0; part < 2; part++) {
            int lo, hi, k;
            if (part == 0) { lo = off; hi = std::min(e.nslots, off + g.blen()); k = k0; }
            else { lo = 0; hi = off + g.blen() - e.nslots; k = k0 + 1; }
            if (hi <= lo) continue;
            std::vector<pc64> v(slots, pcplx(0.0));
            for (int i = lo; i < hi; i++) {
                double val = 1.0;
                if (row_scale) {
                    const int el = (part == 0) ? (i - off) : (i + e.nslots - off);
                    const int h = el / g.m, row = ((el % g.m) + mp[t].s) % g.m;
                    val = row_scale[(size_t)h * g.m + row];
                }
                v[i] = pcplx(val);
            }
            PhantomPlaintext pt = encode_at(e, v, ci, mscale);
            PhantomCiphertext term = z;
            mul_plain(e, term, pt);
            rescale(e, term);
            if (!pinit[k]) { packed[k] = std::move(term); pinit[k] = 1; }
            else add_inplace(e.ctx, packed[k], term);
        }
    }
    if (row_scale && add_c != 0.0) {
        std::vector<std::vector<pc64>> cv(npk, std::vector<pc64>(slots, pcplx(0.0)));
        for (const auto &sl : score_unpack_index(g, e.nslots)) {
            const double v = add_c * row_scale[(size_t)sl.h * g.m + sl.row];
            cv[sl.ct][sl.slot] = pcplx(v, v);
        }
        for (int k = 0; k < npk; k++)
            add_plain_inplace(e.ctx, packed[k], encode_at(e, cv[k], packed[k].chain_index(), packed[k].scale()));
    }
    return packed;
}

struct ValueGeom {
    int m, d_h;
    int half() const { return m / 2; }
};

inline PhantomCiphertext value_block(Env &e, MaskCache &mc, const ValueGeom &g, const PhantomCiphertext &V,
                                     const PhantomCiphertext &P, bool odd) {
    const int m = g.m, half = g.half();
    const int nseg = e.segs();
    if (P.chain_index() != V.chain_index() + 1)
        throw std::runtime_error("value_block: P_fd must sit one level below V");

    PhantomCiphertext u = V;
    mod_switch_to_next_inplace(e.ctx, u);
    {
        PhantomCiphertext vh = rot_within_rs(e, mc, V, half, nseg, 2);
        add_inplace(e.ctx, u, vh);
    }

    std::vector<PhantomCiphertext> ut(half);
    {
        std::map<int, PhantomCiphertext> rc;
        for (int t = 0; t < half; t++) ut[t] = rot_within_rs(e, mc, u, m - t, nseg, 0, &rc);
    }

    std::map<int, PhantomCiphertext> bank;
    for (int d = -(g.d_h - 1); d <= g.d_h - 1; d++) bank.emplace(d, d == 0 ? P : rotated(e, P, d * m));

    const size_t cp = P.chain_index();
    const double q = dropped_prime_at(e.ctx, cp);
    const size_t slots = e.slots();
    std::vector<const PhantomPlaintext *> ns(g.d_h);
    for (int s = 0; s < g.d_h; s++)
        ns[s] = &mc.get({MK_COL, s, cp, odd ? 1 : 0}, [&] {
            std::vector<pc64> v(slots, pcplx(0.0));
            for (int seg = s; seg < nseg; seg += g.d_h)
                for (int i = 0; i < m; i++) v[(size_t)seg * m + i] = odd ? pcplx(0.0, 1.0) : pcplx(1.0);
            return encode_at(e, v, cp, q * V.scale() / P.scale());
        });

    PhantomCiphertext o;
    bool oinit = false;
    std::vector<const PhantomCiphertext *> bc(g.d_h);
    for (int t = 0; t < half; t++) {
        PhantomCiphertext bt;
        if (e.fused) {
            for (int s = 0; s < g.d_h; s++) bc[s] = &bank.at(t - s);
            e.fma.run(e.ctx, bc, ns, bt);
            e.pt_mults += g.d_h;
        } else {
            bool binit = false;
            for (int s = 0; s < g.d_h; s++) {
                PhantomCiphertext term = bank.at(t - s);
                mul_plain(e, term, *ns[s]);
                acc_add(e, bt, binit, std::move(term));
            }
        }
        rescale(e, bt);
        PhantomCiphertext term = ut[t];
        multiply_and_relin_inplace(e.ctx, term, bt, e.rk);
        e.ks.muls_ctct++;
        acc_add(e, o, oinit, std::move(term));
    }
    rescale(e, o);
    return o;
}

inline std::vector<PhantomCiphertext> pair_value_blocks(Env &e, const std::vector<PhantomCiphertext> &o) {
    std::vector<PhantomCiphertext> z;
    for (size_t u = 0; 2 * u < o.size(); u++) {
        if (2 * u + 1 >= o.size()) { PhantomCiphertext c = o[2 * u]; add_inplace(e.ctx, c, conj(e, o[2 * u])); z.push_back(c); continue; }
        PhantomCiphertext a = o[2 * u], b = o[2 * u];
        add_inplace(e.ctx, a, o[2 * u + 1]);
        sub_inplace(e.ctx, b, o[2 * u + 1]);
        add_inplace(e.ctx, a, conj(e, b));
        z.push_back(std::move(a));
    }
    return z;
}

namespace bolt {
constexpr double A = 0.020848611754127593, B = -0.18352506127082727, C = 0.5410550166368381;
constexpr double D = -0.03798164612714154, E = 0.001620808531841547;
}

struct GeluCandidates {
    std::vector<PhantomCiphertext> x_paired;
    std::vector<PhantomCiphertext> f01;
};

inline PhantomCiphertext times_const(Env &e, const PhantomCiphertext &ct, pc64 k, double target) {
    std::vector<pc64> v(e.slots(), k);
    PhantomPlaintext pt = encode_at(e, v, ct.chain_index(), target / ct.scale());
    PhantomCiphertext r = ct;
    mul_plain(e, r, pt);

    if (std::fabs(r.scale() / target - 1.0) > 1e-12) throw std::runtime_error("times_const: scale drift");
    r.set_scale(target);
    return r;
}

inline GeluCandidates gelu_candidates(Env &e, const std::vector<PhantomCiphertext> &x, double s_x, double s_f) {
    GeluCandidates out;
    const size_t c1 = x.at(0).chain_index();
    const double q1 = dropped_prime_at(e.ctx, c1), q2 = dropped_prime_at(e.ctx, c1 + 1),
                 q3 = dropped_prime_at(e.ctx, c1 + 2);

    for (size_t b = 0; b < x.size(); b += 2) {
        PhantomCiphertext z = times_const(e, x[b], pcplx(1.0), q1 * s_x);
        if (b + 1 < x.size()) {
            PhantomCiphertext o = times_const(e, x[b + 1], pcplx(0.0, 1.0), q1 * s_x);
            add_inplace(e.ctx, z, o);
        }
        rescale(e, z);
        out.x_paired.push_back(std::move(z));
    }
    const double T = s_f * q3;
    for (const auto &xb : x) {
        PhantomCiphertext u = xb;
        multiply_and_relin_inplace(e.ctx, u, xb, e.rk);
        e.ks.muls_ctct++;
        rescale(e, u);

        PhantomCiphertext xm = xb;
        {
            std::vector<pc64> one(e.slots(), pcplx(1.0));
            mul_plain(e, xm, encode_at(e, one, c1, xb.scale()));
            rescale(e, xm);
        }
        PhantomCiphertext uu = u, ux = u;
        multiply_and_relin_inplace(e.ctx, uu, u, e.rk);
        multiply_and_relin_inplace(e.ctx, ux, xm, e.rk);
        e.ks.muls_ctct += 2;
        rescale(e, uu);
        rescale(e, ux);
        PhantomCiphertext um = u, xmm = xm;
        mod_switch_to_next_inplace(e.ctx, um);
        mod_switch_to_next_inplace(e.ctx, xmm);

        PhantomCiphertext f = times_const(e, uu, pcplx(bolt::A, bolt::A), T);
        PhantomCiphertext t1 = times_const(e, ux, pcplx(-bolt::B, bolt::B), T);
        PhantomCiphertext t2 = times_const(e, um, pcplx(bolt::C, bolt::C), T);
        PhantomCiphertext t3 = times_const(e, xmm, pcplx(0.5 - bolt::D, 0.5 + bolt::D), T);
        add_inplace(e.ctx, f, t1);
        add_inplace(e.ctx, f, t2);
        add_inplace(e.ctx, f, t3);
        {
            std::vector<pc64> v(e.slots(), pcplx(bolt::E, bolt::E));
            add_plain_inplace(e.ctx, f, encode_at(e, v, f.chain_index(), f.scale()));
        }
        rescale(e, f);
        out.f01.push_back(std::move(f));
    }
    (void)q2;
    return out;
}
}
