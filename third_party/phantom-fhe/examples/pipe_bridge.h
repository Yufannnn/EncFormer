// pipe_bridge.h — Server-side CKKS <-> MPC bridge protocol for native pipeline.
//
// Implements the integer masking protocol from ckks_mpc_bridge.py / secure_bridge.py
// so that transitions between FHE and MPC use secret-shared values instead of plaintext.
//
// Server (C++ binary, holds sk):
//   c2m: decrypt ct → x, generate mask r, write (x+r) mod q, keep -r
//   m2c: read client share, add server share, encrypt → new ct
//
// Client (Python MPC process):
//   c2m: read masked share, that's client_share
//   m2c: write client share of MPC result

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "pipe_io.h"

namespace pipe_bridge {

// ---------------------------------------------------------------------------
// Bridge parameters (matching ckks_mpc_bridge.py _conv_params)
// ---------------------------------------------------------------------------

struct BridgeParams {
    int64_t q_conv;    // 1 << Q_CONV_BITS (default 52)
    int64_t ring_mod;  // 1 << RING_BITS   (default 43)
    int64_t scale;     // 1 << PREC_BITS   (default 13)
};

inline int _env_int(const char *name, int dflt) {
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}

inline BridgeParams get_params() {
    int q_bits = _env_int("CKKS_Q_CONV_BITS", 52);
    int r_bits = _env_int("CKKS_RING_BITS", 43);
    int p_bits = _env_int("CKKS_MPC_PREC_BITS", 13);
    return {1LL << q_bits, 1LL << r_bits, 1LL << p_bits};
}

// ---------------------------------------------------------------------------
// Modular arithmetic (matching _mod / _center_lift in ckks_mpc_bridge.py)
// ---------------------------------------------------------------------------

inline int64_t mod_pos(int64_t x, int64_t q) {
    int64_t r = x % q;
    return r < 0 ? r + q : r;
}

inline int64_t center_lift(int64_t x_mod, int64_t q) {
    return (x_mod >= q / 2) ? x_mod - q : x_mod;
}

// Full ring reduction: value → int → mod ring → center_lift → mod q_conv → center_lift → float
inline double reduce_share(double v, const BridgeParams &bp) {
    int64_t x_int = std::llround(v * static_cast<double>(bp.scale));
    int64_t x_ring = mod_pos(x_int, bp.ring_mod);
    int64_t t_q = mod_pos(center_lift(x_ring, bp.ring_mod), bp.q_conv);
    return static_cast<double>(center_lift(t_q, bp.q_conv)) / static_cast<double>(bp.scale);
}

// ---------------------------------------------------------------------------
// Atomic file write (write to .tmp, rename)
// ---------------------------------------------------------------------------

inline void write_f64_atomic(const char *path, const double *data, size_t n) {
    std::string tmp = std::string(path) + ".tmp";
    pipe_io::write_f64(tmp.c_str(), data, n);
    std::rename(tmp.c_str(), path);
}

// ---------------------------------------------------------------------------
// Server-side CKKS → MPC  (operates on already-decrypted plaintext doubles)
//
// Produces:
//   masked_path:       (x + r) mod q  / scale   →  client's share
//   server_share_path: center_lift(-r)           →  server's share
// ---------------------------------------------------------------------------

inline void ckks_to_mpc(
    const double *x, size_t n,
    std::mt19937_64 &rng,
    const std::string &masked_path,
    const std::string &server_share_path)
{
    BridgeParams bp = get_params();
    std::uniform_int_distribution<int64_t> dist(0, bp.q_conv - 1);

    std::vector<double> masked(n), server_share(n);
    for (size_t i = 0; i < n; i++) {
        int64_t x_int = std::llround(x[i] * static_cast<double>(bp.scale));
        int64_t r = dist(rng);

        // Client share: (x + r) mod q_conv, reduced through ring
        int64_t masked_q = mod_pos(x_int + r, bp.q_conv);
        int64_t masked_ring = mod_pos(center_lift(masked_q, bp.q_conv), bp.ring_mod);
        masked[i] = static_cast<double>(center_lift(masked_ring, bp.ring_mod))
                     / static_cast<double>(bp.scale);

        // Server share: -r, reduced through ring
        int64_t neg_r_q = mod_pos(-r, bp.q_conv);
        int64_t neg_r_ring = mod_pos(center_lift(neg_r_q, bp.q_conv), bp.ring_mod);
        server_share[i] = static_cast<double>(center_lift(neg_r_ring, bp.ring_mod))
                           / static_cast<double>(bp.scale);
    }

    write_f64_atomic(masked_path.c_str(), masked.data(), n);
    write_f64_atomic(server_share_path.c_str(), server_share.data(), n);
}

// ---------------------------------------------------------------------------
// Server-side MPC → plaintext  (reconstruct from two share files)
//
// Used when the FHE stage needs plaintext (e.g. A_heads for Value SV multiply).
// Reads both shares, reduces each through ring, sums.
// ---------------------------------------------------------------------------

inline std::vector<double> mpc_to_plain(
    const std::string &client_share_path,
    const std::string &server_share_path,
    size_t n)
{
    BridgeParams bp = get_params();

    std::vector<double> client_buf(n), server_buf(n);
    pipe_io::read_f64(client_share_path.c_str(), client_buf.data(), n);
    pipe_io::read_f64(server_share_path.c_str(), server_buf.data(), n);

    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = reduce_share(client_buf[i], bp)
                + reduce_share(server_buf[i], bp);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Server-side MPC → plaintext (single vector, for when we have both shares
// already in memory as a single reconstructed array)
// ---------------------------------------------------------------------------

inline std::vector<double> reduce_plain(const double *x, size_t n) {
    BridgeParams bp = get_params();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = reduce_share(x[i], bp);
    return out;
}

} // namespace pipe_bridge
