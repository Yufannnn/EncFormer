from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CommStats:
    ckks_to_mpc_cts: int = 0
    mpc_to_ckks_cts: int = 0
    ct_bytes: int = 0

    mpc_rounds: Dict[str, int] = field(default_factory=dict)

    mpc_msg_bytes: int = 0

    def total_bridge_bytes(self) -> int:
        return (self.ckks_to_mpc_cts + self.mpc_to_ckks_cts) * self.ct_bytes

    def total_mpc_rounds(self) -> int:
        return sum(self.mpc_rounds.values())

    def total_mpc_bytes(self) -> int:
        return self.total_mpc_rounds() * self.mpc_msg_bytes

    def total_bytes(self) -> int:
        return self.total_bridge_bytes() + self.total_mpc_bytes()

    def add_bridge_c2m(self, n: int = 1) -> None:
        self.ckks_to_mpc_cts += n

    def add_bridge_m2c(self, n: int = 1) -> None:
        self.mpc_to_ckks_cts += n

    def add_mpc_rounds(self, op_name: str, rounds: int) -> None:
        self.mpc_rounds[op_name] = self.mpc_rounds.get(op_name, 0) + rounds

    def __iadd__(self, other: CommStats) -> CommStats:
        self.ckks_to_mpc_cts += other.ckks_to_mpc_cts
        self.mpc_to_ckks_cts += other.mpc_to_ckks_cts
        for k, v in other.mpc_rounds.items():
            self.mpc_rounds[k] = self.mpc_rounds.get(k, 0) + v
        return self


MPC_ROUNDS_BPMAX = 3

MPC_ROUNDS_BATCHLN = 0

MPC_ROUNDS_GELU = 4


@dataclass
class NetworkProfile:
    name: str
    bandwidth_bps: float
    rtt_s: float


LAN = NetworkProfile("LAN", 1e9, 0.0003)
WAN1 = NetworkProfile("WAN1", 4e8, 0.004)
WAN2 = NetworkProfile("WAN2", 1e8, 0.004)
WAN3 = NetworkProfile("WAN3", 1e8, 0.080)

ALL_PROFILES: List[NetworkProfile] = [LAN, WAN1, WAN2, WAN3]


def compute_latency(stats: CommStats, profile: NetworkProfile) -> Dict[str, float]:

    t_conv = (stats.total_bridge_bytes() * 8) / profile.bandwidth_bps
    t_mpc = stats.total_mpc_rounds() * profile.rtt_s
    return {
        "T_conv_s": t_conv,
        "T_mpc_s": t_mpc,
        "T_total_s": t_conv + t_mpc,
    }


def estimate_ct_bytes(nslots: int) -> int:

    return nslots * 16


def estimate_bridge_bytes(nslots: int) -> int:

    return 2 * nslots * 8


def print_comm_summary(stats: CommStats, fhe_time_s: float = 0.0) -> None:

    bridge_mb = stats.total_bridge_bytes() / 1e6
    mpc_mb = stats.total_mpc_bytes() / 1e6
    total_gb = stats.total_bytes() / 1e9

    print("\n=== Communication Cost Summary ===")
    print(f"  Bridge CKKS->MPC: {stats.ckks_to_mpc_cts} cts")
    print(f"  Bridge MPC->CKKS: {stats.mpc_to_ckks_cts} cts")
    print(f"  CT size: {stats.ct_bytes // 1024} KB")
    print(f"  Bridge payload: {bridge_mb:.1f} MB")
    for op, r in sorted(stats.mpc_rounds.items()):
        print(f"  MPC {op}: {r} rounds")
    print(f"  MPC payload: {mpc_mb:.1f} MB")
    print(f"  Total comm: {total_gb:.3f} GB")

    print("\n  Network profile latency estimates:")
    print(f"  {'Profile':<8} {'T_conv':>8} {'T_mpc':>8} {'T_comm':>8} {'+ FHE':>10}")
    for p in ALL_PROFILES:
        lat = compute_latency(stats, p)
        t_with_fhe = lat["T_total_s"] + fhe_time_s
        print(
            f"  {p.name:<8} {lat['T_conv_s']:.2f}s  {lat['T_mpc_s']:.3f}s  {lat['T_total_s']:.2f}s  {t_with_fhe:.1f}s"
        )
