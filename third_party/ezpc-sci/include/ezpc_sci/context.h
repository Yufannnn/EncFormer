#pragma once

#include <algorithm>
#include <chrono>
#include <deque>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <exception>
#include <vector>

#include "ezpc_sci/config.h"

#ifdef EZPC_HAS_SCI
#include "utils/io_pack.h"
#include "OT/ot_pack.h"
#include "FloatingPoint/fixed-point.h"
#endif

namespace ezpc_sci {

constexpr int SCI_SERVER = 1;
constexpr int SCI_CLIENT = 2;

struct CommSnapshot {
    uint64_t bytes_sent = 0;
    uint64_t bytes_recv = 0;
    uint64_t rounds = 0;
    uint64_t ext_bytes_sent = 0;
    uint64_t ext_choice_bits = 0;
    uint64_t ext_ots = 0;
    double ext_seconds = 0.0;
    uint64_t derand_bytes = 0;
};

struct PhaseStats {
    double seconds = 0.0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_recv = 0;
    uint64_t rounds = 0;
    uint64_t ext_bytes_sent = 0, ext_choice_bits = 0, ext_ots = 0;
    double ext_seconds = 0.0;
    uint64_t derand_bytes = 0;
    std::vector<uint64_t> ot_counts;
};

struct Material {
    std::string sig;
    std::vector<uint64_t> A, A2, TA, TB, TC;
};

struct CallStats {
    PhaseStats offline, online;
    bool planned = false;
    int threads = 1;
    size_t n = 0;
};

class SCIContext {
public:
    CallStats last_stats;

    std::deque<std::shared_ptr<Material>> plan_queue;
    bool planned = false;
    static constexpr int OT_INSTANCES = 2 + 8;

    explicit SCIContext(int role,
                        const std::string &address = "127.0.0.1",
                        int port = 32000,
                        ProtocolConfig cfg = default_config(),
                        int threads = 1)
        : role_(role), address_(address), port_(port), cfg_(cfg),
          sci_party_(role == 0 ? SCI_SERVER : SCI_CLIENT), threads_(threads)
    {
        if (role != 0 && role != 1) throw std::invalid_argument("SCIContext: role must be 0 (server) or 1 (client)");
        if (threads < 1 || threads > 32) throw std::invalid_argument("SCIContext: threads must be in [1, 32]");
#ifdef EZPC_HAS_SCI
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < threads; t++) {

            auto *io = new sci::IOPack(sci_party_, port + t, address);
            iopacks_.push_back(io);
            if (t == 0) {

                uint32_t mine = static_cast<uint32_t>(threads), theirs = 0;
                if (sci_party_ == SCI_SERVER) {
                    io->io->send_data(&mine, sizeof(mine));
                    io->io->recv_data(&theirs, sizeof(theirs));
                } else {
                    io->io->recv_data(&theirs, sizeof(theirs));
                    io->io->send_data(&mine, sizeof(mine));
                }
                io->io->flush();
                if (theirs != mine)
                    throw std::runtime_error("SCIContext: peer uses a different thread count");
            }
        }
        for (int t = 0; t < threads; t++) {
            otpacks_.push_back(new sci::OTPack(iopacks_[t], sci_party_));
            fixops_.push_back(new FixOp(sci_party_, iopacks_[t], otpacks_[t]));
        }
        setup_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        setup_comm_ = comm();
#endif
    }

    ~SCIContext() {
#ifdef EZPC_HAS_SCI
        for (auto *f : fixops_) delete f;
        for (auto *o : otpacks_) delete o;
        for (auto *i : iopacks_) delete i;
#endif
    }

    SCIContext(const SCIContext &) = delete;
    SCIContext &operator=(const SCIContext &) = delete;

    int role() const { return role_; }
    int sci_party() const { return sci_party_; }
    int threads() const { return threads_; }
    int port() const { return port_; }
    const ProtocolConfig &config() const { return cfg_; }
    double setup_seconds() const { return setup_seconds_; }

#ifdef EZPC_HAS_SCI
    sci::IOPack *iopack(int t = 0) { return iopacks_.at(t); }
    sci::OTPack *otpack(int t = 0) { return otpacks_.at(t); }
    FixOp *fixop(int t = 0) { return fixops_.at(t); }

    uint64_t get_comm() const { return comm().bytes_sent; }

    uint64_t get_rounds() const { return comm().rounds; }

    CommSnapshot comm_thread(int t) const {
        CommSnapshot s;
        auto *io = iopacks_.at(t);
        s.bytes_sent = io->get_comm();
#ifdef EZPC_SCI_HAS_RECV_TAP
        s.bytes_recv = io->io->recv_counter + io->io_rev->recv_counter + io->io_GC->recv_counter;
#endif
#ifdef EZPC_SCI_HAS_EXT_COUNTER
        for (auto *ch : {io->io, io->io_rev, io->io_GC}) {
            s.ext_bytes_sent += ch->ext_bytes;
            s.ext_choice_bits += ch->ext_choice_bits;
            s.ext_ots += ch->ext_ots;
            s.ext_seconds += ch->ext_seconds;
#ifdef EZPC_SCI_HAS_OT_POOL
            s.derand_bytes += ch->ef_derand_bytes;
#endif
        }
#endif
        s.rounds = io->get_rounds();
        return s;
    }

    CommSnapshot comm() const {
        CommSnapshot s;
        for (size_t t = 0; t < iopacks_.size(); t++) {
            CommSnapshot c = comm_thread(static_cast<int>(t));
            s.bytes_sent += c.bytes_sent;
            s.bytes_recv += c.bytes_recv;
            s.rounds = std::max<uint64_t>(s.rounds, c.rounds);
        }
        return s;
    }
    CommSnapshot setup_comm() const { return setup_comm_; }

#ifdef EZPC_SCI_HAS_OT_POOL
    template <class F> void for_each_ot(int t, F f) {
        sci::OTPack *o = otpacks_.at(t);
        f(0, o->iknp_straight);
        f(1, o->iknp_reversed);
        for (int k = 0; k < 8; k++) f(2 + k, o->kkot[k]);
    }

    std::vector<uint64_t> ot_counts() {
        std::vector<uint64_t> v(threads_ * OT_INSTANCES, 0);
        for (int t = 0; t < threads_; t++)
            for_each_ot(t, [&](int i, auto *ot) { v[t * OT_INSTANCES + i] = ot->ef_ot_count; });
        return v;
    }
    std::vector<uint64_t> pool_left() {
        std::vector<uint64_t> v(threads_ * OT_INSTANCES, 0);
        for (int t = 0; t < threads_; t++)
            for_each_ot(t, [&](int i, auto *ot) { v[t * OT_INSTANCES + i] = ot->ef_pool_left(); });
        return v;
    }

    void pool_fill(const std::vector<uint64_t> &counts) {
        if (counts.size() != size_t(threads_) * OT_INSTANCES) throw std::invalid_argument("pool_fill: bad counts size");
        std::vector<std::thread> ths;
        std::vector<std::exception_ptr> errs(threads_);
        for (int t = 0; t < threads_; t++)
            ths.emplace_back([&, t] {
                try {
                    for_each_ot(t, [&](int i, auto *ot) {
                        if (counts[t * OT_INSTANCES + i]) ot->ef_pool_fill((int64_t)counts[t * OT_INSTANCES + i]);
                    });
                    iopacks_[t]->io->flush();
                    iopacks_[t]->io_rev->flush();
                } catch (...) { errs[t] = std::current_exception(); }
            });
        for (auto &th : ths) th.join();
        for (auto &e : errs) if (e) std::rethrow_exception(e);
    }
    void pool_set_on(bool on) {
        for (int t = 0; t < threads_; t++) for_each_ot(t, [&](int, auto *ot) { ot->ef_pool_on = on; });
    }
    void pool_clear() {
        for (int t = 0; t < threads_; t++) for_each_ot(t, [&](int, auto *ot) { ot->ef_pool_clear(); });
    }
#endif

#endif

private:
    int role_;
    std::string address_;
    int port_;
    ProtocolConfig cfg_;
    int sci_party_;
    int threads_;
    double setup_seconds_ = 0.0;
    CommSnapshot setup_comm_;

#ifdef EZPC_HAS_SCI
    std::vector<sci::IOPack *> iopacks_;
    std::vector<sci::OTPack *> otpacks_;
    std::vector<FixOp *> fixops_;
#endif
};

}
