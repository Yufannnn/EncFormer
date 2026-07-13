// config.h — Compile-time and runtime configuration for EzPC/SCI protocols.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace ezpc_sci {

struct ProtocolConfig {
    int ring_bits;      // Ring modulus = 2^ring_bits
    int scale_bits;     // Fixed-point scale = 2^scale_bits
    double threshold;   // GELU clipping threshold

    int64_t ring_mod()  const { return 1LL << ring_bits; }
    int64_t ring_half() const { return 1LL << (ring_bits - 1); }
    int64_t scale()     const { return 1LL << scale_bits; }
};

inline int env_int(const char *name, int dflt) {
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}

inline double env_double(const char *name, double dflt) {
    const char *v = std::getenv(name);
    return v ? std::atof(v) : dflt;
}

inline ProtocolConfig default_config() {
    return {
        env_int("MPC_GELU_RING_BITS", 43),
        env_int("MPC_GELU_SCALE_BITS", 13),
        env_double("MPC_GELU_THRESH", 2.7),
    };
}

} // namespace ezpc_sci
