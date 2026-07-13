// context.h — SCIContext: manages 2PC protocol state and SCI party objects.
//
// When compiled with EZPC_HAS_SCI, holds real SCI IOPack/OTPack/FixOp objects
// for OT-based secure computation.  Without SCI, serves as a lightweight
// configuration carrier for the emulated path.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "ezpc_sci/config.h"

#ifdef EZPC_HAS_SCI
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include "FloatingPoint/fixed-point.h"
#endif

namespace ezpc_sci {

// SCI party constants (matching defines.h)
constexpr int SCI_SERVER = 1;
constexpr int SCI_CLIENT = 2;

class SCIContext {
public:
    // role: 0 = server (we map to SCI_SERVER=1), 1 = client (SCI_CLIENT=2)
    explicit SCIContext(int role,
                       const std::string &address = "127.0.0.1",
                       int port = 32000,
                       ProtocolConfig cfg = default_config())
        : role_(role), address_(address), port_(port), cfg_(cfg),
          sci_party_(role == 0 ? SCI_SERVER : SCI_CLIENT)
    {
#ifdef EZPC_HAS_SCI
        // IOPack opens 3 TCP channels: io (port), io_rev (port+50), io_GC (port+100)
        iopack_ = new sci::IOPack(sci_party_, port, address);
        otpack_ = new sci::OTPack(iopack_, sci_party_);
        fixop_  = new FixOp(sci_party_, iopack_, otpack_);
#endif
    }

    ~SCIContext() {
#ifdef EZPC_HAS_SCI
        delete fixop_;
        delete otpack_;
        delete iopack_;
#endif
    }

    // Non-copyable
    SCIContext(const SCIContext &) = delete;
    SCIContext &operator=(const SCIContext &) = delete;

    int role() const { return role_; }
    int sci_party() const { return sci_party_; }
    const ProtocolConfig &config() const { return cfg_; }

#ifdef EZPC_HAS_SCI
    sci::IOPack *iopack() { return iopack_; }
    sci::OTPack *otpack() { return otpack_; }
    FixOp *fixop() { return fixop_; }

    uint64_t get_comm() const { return iopack_->get_comm(); }
    uint64_t get_rounds() const { return iopack_->get_rounds(); }
#endif

private:
    int role_;
    std::string address_;
    int port_;
    ProtocolConfig cfg_;
    int sci_party_;

#ifdef EZPC_HAS_SCI
    sci::IOPack *iopack_ = nullptr;
    sci::OTPack *otpack_ = nullptr;
    FixOp *fixop_ = nullptr;
#endif
};

} // namespace ezpc_sci
