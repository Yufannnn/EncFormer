#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pipe_ckks_eval.h"

namespace pipe_ckks {

struct CkksParams {
    size_t poly_degree = 0;
    size_t special_primes = 0;
    std::vector<uint64_t> primes;
    std::vector<uint32_t> galois_elts;

    void to_parms(EncryptionParameters &parms) const {
        std::vector<Modulus> mods;
        mods.reserve(primes.size());
        for (uint64_t p : primes) mods.emplace_back(p);
        parms.set_poly_modulus_degree(poly_degree);
        parms.set_special_modulus_size(special_primes);
        parms.set_coeff_modulus(mods);
        parms.set_galois_elts(galois_elts);
    }

    uint64_t fingerprint() const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) {
            for (int i = 0; i < 8; i++) { h ^= (v >> (8 * i)) & 0xff; h *= 1099511628211ULL; }
        };
        mix(poly_degree); mix(special_primes);
        mix(primes.size()); for (auto p : primes) mix(p);
        mix(galois_elts.size()); for (auto g : galois_elts) mix(g);
        return h;
    }

    void save(const std::string &path) const {
        std::ofstream f(path);
        f << "poly_degree " << poly_degree << "\nspecial_primes " << special_primes << "\nprimes";
        for (auto p : primes) f << " " << p;
        f << "\ngalois_elts";
        for (auto g : galois_elts) f << " " << g;
        f << "\nfingerprint " << fingerprint() << "\n";
        if (!f) throw std::runtime_error("cannot write " + path);
    }

    static CkksParams load(const std::string &path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read " + path);
        CkksParams p;
        uint64_t fp = 0;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ls(line);
            std::string key;
            ls >> key;
            if (key == "poly_degree") ls >> p.poly_degree;
            else if (key == "special_primes") ls >> p.special_primes;
            else if (key == "primes") { uint64_t v; while (ls >> v) p.primes.push_back(v); }
            else if (key == "galois_elts") { uint32_t v; while (ls >> v) p.galois_elts.push_back(v); }
            else if (key == "fingerprint") ls >> fp;
        }
        if (fp != p.fingerprint()) throw std::runtime_error("parameter fingerprint mismatch in " + path);
        return p;
    }
};

inline CkksParams params_from(const EncryptionParameters &parms) {
    CkksParams p;
    p.poly_degree = parms.poly_modulus_degree();
    p.special_primes = parms.special_modulus_size();
    for (const auto &q : parms.coeff_modulus()) p.primes.push_back(q.value());
    p.galois_elts = parms.galois_elts();
    return p;
}

template <class T>
inline size_t save_obj(const T &obj, const std::string &path) {
    std::ofstream f(path, std::ios::binary);
    obj.save(f);
    f.flush();
    if (!f) throw std::runtime_error("cannot write " + path);
    return static_cast<size_t>(f.tellp());
}

template <class T>
inline void load_obj(T &obj, const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    obj.load(f);
}

inline size_t save_ct(const PhantomCiphertext &ct, const std::string &path) { return save_obj(ct, path); }

inline PhantomCiphertext load_ct(const std::string &path) {
    PhantomCiphertext ct;
    load_obj(ct, path);
    return ct;
}

inline size_t save_cts(const std::vector<PhantomCiphertext> &cts, const std::string &path) {
    std::ofstream f(path, std::ios::binary);
    uint64_t n = cts.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof n);
    for (const auto &ct : cts) ct.save(f);
    f.flush();
    if (!f) throw std::runtime_error("cannot write " + path);
    return static_cast<size_t>(f.tellp());
}

inline size_t save_cts_seeded(const std::vector<PhantomCiphertext> &cts, const std::string &path) {
    std::ofstream f(path, std::ios::binary);
    uint64_t n = cts.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof n);
    for (const auto &ct : cts) ct.save_symmetric(f);
    f.flush();
    if (!f) throw std::runtime_error("cannot write " + path);
    return static_cast<size_t>(f.tellp());
}

inline std::vector<PhantomCiphertext> load_cts_seeded(const PhantomContext &ctx, const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    uint64_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof n);
    std::vector<PhantomCiphertext> cts(n);
    for (auto &ct : cts) ct.load_symmetric(ctx, f);
    return cts;
}

inline std::vector<PhantomCiphertext> load_cts(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    uint64_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof n);
    std::vector<PhantomCiphertext> cts(n);
    for (auto &ct : cts) ct.load(f);
    return cts;
}

constexpr size_t kPhantomCtHeader = 4 * sizeof(size_t) + sizeof(double) + sizeof(uint64_t) + sizeof(size_t) + 2;

inline std::vector<int> wire_limb_bits(const PhantomContext &ctx, size_t chain_index) {
    std::vector<int> b;
    for (const auto &q : ctx.get_context_data(chain_index).parms().coeff_modulus())
        b.push_back(64 - __builtin_clzll(q.value()));
    return b;
}

inline std::string pack_ct_bytes(const PhantomContext &ctx, const std::string &raw, bool seeded) {
    size_t hdr[4];
    std::memcpy(hdr, raw.data(), sizeof hdr);
    const size_t ci = hdr[0], polys = seeded ? 1 : hdr[1], N = hdr[2], L = hdr[3];
    const auto bits = wire_limb_bits(ctx, ci);
    if (bits.size() != L) throw std::runtime_error("pack_ct_bytes: limb count mismatch");
    const uint64_t *w = reinterpret_cast<const uint64_t *>(raw.data() + kPhantomCtHeader);
    std::string out(raw.data(), kPhantomCtHeader);
    size_t total_bits = 0;
    for (int b : bits) total_bits += (size_t)b * N;
    std::vector<uint64_t> packed((total_bits * polys + 63) / 64 + 1, 0);
    size_t pos = 0;
    for (size_t p = 0; p < polys; p++)
        for (size_t l = 0; l < L; l++)
            for (size_t i = 0; i < N; i++) {
                const uint64_t v = w[(p * L + l) * N + i];
                const int b = bits[l];
                packed[pos >> 6] |= v << (pos & 63);
                if ((pos & 63) + b > 64) packed[(pos >> 6) + 1] |= v >> (64 - (pos & 63));
                pos += b;
            }
    out.append(reinterpret_cast<const char *>(packed.data()), (pos + 7) / 8);
    const size_t data_bytes = polys * L * N * 8;
    out.append(raw.data() + kPhantomCtHeader + data_bytes, raw.size() - kPhantomCtHeader - data_bytes);
    return out;
}

inline std::string unpack_ct_bytes(const PhantomContext &ctx, const std::string &pk, bool seeded) {
    size_t hdr[4];
    std::memcpy(hdr, pk.data(), sizeof hdr);
    const size_t ci = hdr[0], polys = seeded ? 1 : hdr[1], N = hdr[2], L = hdr[3];
    const auto bits = wire_limb_bits(ctx, ci);
    std::vector<uint64_t> w(polys * L * N);
    const unsigned char *src = reinterpret_cast<const unsigned char *>(pk.data() + kPhantomCtHeader);
    size_t pos = 0;
    for (size_t p = 0; p < polys; p++)
        for (size_t l = 0; l < L; l++) {
            const int b = bits[l];
            const uint64_t mask = (b == 64) ? ~0ULL : ((1ULL << b) - 1);
            for (size_t i = 0; i < N; i++) {
                uint64_t v = 0;
                const size_t byte = pos >> 3, sh = pos & 7;
                for (int k = 0; k < 9 && (k * 8 - (int)sh) < b; k++) {
                    const uint64_t bt = src[byte + k];
                    if (k == 0) v |= bt >> sh;
                    else v |= (k * 8 - (int)sh < 64) ? (bt << (k * 8 - sh)) : 0;
                }
                w[(p * L + l) * N + i] = v & mask;
                pos += b;
            }
        }
    std::string out(pk.data(), kPhantomCtHeader);
    out.append(reinterpret_cast<const char *>(w.data()), w.size() * 8);
    const size_t packed_bytes = (pos + 7) / 8;
    out.append(pk.data() + kPhantomCtHeader + packed_bytes, pk.size() - kPhantomCtHeader - packed_bytes);
    return out;
}

inline size_t save_cts_wire(const PhantomContext &ctx, const std::vector<PhantomCiphertext> &cts, const std::string &path,
                            bool seeded) {
    std::ofstream f(path, std::ios::binary);
    uint64_t n = cts.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof n);
    for (const auto &ct : cts) {
        std::ostringstream os;
        if (seeded) ct.save_symmetric(os); else ct.save(os);
        std::string pk = pack_ct_bytes(ctx, os.str(), seeded);
        uint64_t len = pk.size();
        f.write(reinterpret_cast<const char *>(&len), sizeof len);
        f.write(pk.data(), (std::streamsize)len);
    }
    f.flush();
    if (!f) throw std::runtime_error("cannot write " + path);
    return static_cast<size_t>(f.tellp());
}

inline std::vector<PhantomCiphertext> load_cts_wire(const PhantomContext &ctx, const std::string &path, bool seeded) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    uint64_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof n);
    std::vector<PhantomCiphertext> cts(n);
    for (auto &ct : cts) {
        uint64_t len = 0;
        f.read(reinterpret_cast<char *>(&len), sizeof len);
        std::string pk(len, '\0');
        f.read(pk.data(), (std::streamsize)len);
        std::istringstream is(unpack_ct_bytes(ctx, pk, seeded));
        if (seeded) ct.load_symmetric(ctx, is); else ct.load(is);
    }
    return cts;
}

}
