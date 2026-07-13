from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.models.model_config import ModelConfig, get_config
from src.utils_comm import (
    ALL_PROFILES,
    LAN,
    MPC_ROUNDS_BATCHLN,
    MPC_ROUNDS_BPMAX,
    MPC_ROUNDS_GELU,
    WAN1,
    WAN2,
    WAN3,
    NetworkProfile,
    estimate_bridge_bytes,
    estimate_ct_bytes,
)


@dataclass
class DesignCost:
    name: str
    conversions: int
    repack_ops: int
    mpc_rounds: int
    ct_bytes: int
    t_conv_s: float
    t_repack_s: float
    t_mpc_s: float
    t_total_s: float


@dataclass
class BackendProfile:
    name: str
    repack_time_s: float
    use_bridge_bytes: bool = False
    fhe_stage_times_s: Optional[Dict[str, float]] = None

    def ct_bytes_for(self, nslots: int) -> int:

        if self.use_bridge_bytes:
            return estimate_bridge_bytes(nslots)
        return estimate_ct_bytes(nslots)

    def total_fhe_time_s(self) -> float:

        if self.fhe_stage_times_s is None:
            return 0.0
        return sum(self.fhe_stage_times_s.values())


PLAIN_PROFILE = BackendProfile(
    name="plain",
    repack_time_s=0.0,
    use_bridge_bytes=True,
)


PHANTOM_GPU_PROFILE = BackendProfile(
    name="phantom",
    repack_time_s=1.1e-3,
    use_bridge_bytes=False,
    fhe_stage_times_s={
        "QKV": 2.50,
        "Score": 0.42,
        "Value": 0.38,
        "OUT": 0.25,
        "FF1": 0.35,
        "FF2": 0.30,
    },
)


DESILO_PROFILE = BackendProfile(
    name="desilo",
    repack_time_s=2.5e-3,
    use_bridge_bytes=False,
)


LEGACY_PROFILE = BackendProfile(
    name="legacy",
    repack_time_s=1.5e-3,
    use_bridge_bytes=False,
)

ALL_BACKEND_PROFILES: Dict[str, BackendProfile] = {
    "plain": PLAIN_PROFILE,
    "phantom": PHANTOM_GPU_PROFILE,
    "desilo": DESILO_PROFILE,
    "legacy": LEGACY_PROFILE,
}


def get_backend_profile(name: str) -> BackendProfile:

    p = ALL_BACKEND_PROFILES.get(name)
    if p is None:
        raise ValueError(f"Unknown backend profile '{name}'. Available: {list(ALL_BACKEND_PROFILES.keys())}")
    return p


MPC_ROUNDS_PER_LAYER = MPC_ROUNDS_BPMAX + MPC_ROUNDS_BATCHLN + MPC_ROUNDS_GELU + MPC_ROUNDS_BATCHLN


REPACK_TIME_S = LEGACY_PROFILE.repack_time_s


BASELINE_CONVERSIONS_PER_MPC_BLOCK = 2


def _cc_factor(use_cc: bool) -> float:

    return 0.5 if use_cc else 1


def _variant_name(*, use_cc: bool, use_scp: bool) -> str:
    if use_cc and use_scp:
        return "EncFormer"
    parts = ["EncFormer"]
    if not use_cc:
        parts.append("wo_CC")
    if not use_scp:
        parts.append("wo_SCP")
    return " ".join(parts)


def minimal_baseline_conversions(cfg: ModelConfig, *, use_cc: bool = True) -> int:

    C = cfg.nslots // cfg.m
    blocks = (cfg.d_model + C - 1) // C
    ff_blocks = (cfg.d_ff + C - 1) // C
    H = cfg.H

    softmax = 2 * H

    ln1 = 2 * blocks

    gelu = 2 * ff_blocks

    ln2 = 2 * blocks

    total = softmax + ln1 + gelu + ln2
    return total * _cc_factor(use_cc)


def encformer_conversions(cfg: ModelConfig, *, use_cc: bool = True) -> int:

    C = cfg.nslots // cfg.m
    blocks = (cfg.d_model + C - 1) // C
    ff_blocks = (cfg.d_ff + C - 1) // C
    H = cfg.H

    softmax = 2 * H

    ln1 = 2 * blocks

    gelu_in = 3 * ff_blocks
    gelu_out = ff_blocks

    ln2 = 2 * blocks

    total = softmax + ln1 + gelu_in + gelu_out + ln2
    return total * _cc_factor(use_cc)


def baseline_repack_ops(cfg: ModelConfig) -> int:

    C = cfg.nslots // cfg.m
    blocks = (cfg.d_model + C - 1) // C
    score_blocks = (cfg.H * cfg.m * cfg.m + 2 * cfg.nslots - 1) // (2 * cfg.nslots)
    ff_blocks = (cfg.d_ff + C - 1) // C

    return 5 * blocks + score_blocks + ff_blocks


def encformer_repack_ops(cfg: ModelConfig, *, use_scp: bool = True) -> int:

    if use_scp:
        return 0
    return baseline_repack_ops(cfg)


def compute_design_cost(
    name: str,
    conversions: int,
    repack_ops: int,
    cfg: ModelConfig,
    profile: NetworkProfile,
    *,
    backend: BackendProfile | None = None,
) -> DesignCost:

    bp = backend or LEGACY_PROFILE
    ct_bytes = bp.ct_bytes_for(cfg.nslots)
    total_conv_bytes = conversions * ct_bytes
    t_conv = (total_conv_bytes * 8) / profile.bandwidth_bps
    t_repack = repack_ops * bp.repack_time_s
    t_mpc = MPC_ROUNDS_PER_LAYER * profile.rtt_s
    return DesignCost(
        name=name,
        conversions=conversions,
        repack_ops=repack_ops,
        mpc_rounds=MPC_ROUNDS_PER_LAYER,
        ct_bytes=ct_bytes,
        t_conv_s=t_conv,
        t_repack_s=t_repack,
        t_mpc_s=t_mpc,
        t_total_s=t_conv + t_repack + t_mpc,
    )


def should_prefer_encformer(
    cfg: ModelConfig,
    profile: NetworkProfile,
    *,
    use_cc: bool = True,
    use_scp: bool = True,
    backend: BackendProfile | None = None,
) -> bool:

    d_cost = compute_design_cost(
        _variant_name(use_cc=use_cc, use_scp=use_scp),
        encformer_conversions(cfg, use_cc=use_cc),
        encformer_repack_ops(cfg, use_scp=use_scp),
        cfg,
        profile,
        backend=backend,
    )
    b_cost = compute_design_cost(
        "Baseline",
        minimal_baseline_conversions(cfg, use_cc=use_cc),
        baseline_repack_ops(cfg),
        cfg,
        profile,
        backend=backend,
    )
    return d_cost.t_total_s < b_cost.t_total_s


def print_cost_comparison(
    cfg: ModelConfig,
    profiles: list[NetworkProfile] | None = None,
    *,
    use_cc: bool = True,
    use_scp: bool = True,
    backend: BackendProfile | None = None,
) -> None:

    if profiles is None:
        profiles = ALL_PROFILES

    bp = backend or LEGACY_PROFILE
    variant_name = _variant_name(use_cc=use_cc, use_scp=use_scp)
    print(f"\n{'=' * 78}")
    print(f"Cost Analysis: {cfg.name} (m={cfg.m}, d={cfg.d_model}, H={cfg.H})  backend={bp.name}")
    print(f"  NOTE: boundary overhead only (conversion + repack + MPC rounds);")
    print(f"        excludes FHE kernel compute. See paper Section 5.")
    print(f"{'=' * 78}")

    enc_convs = encformer_conversions(cfg, use_cc=use_cc)
    base_convs = minimal_baseline_conversions(cfg, use_cc=use_cc)
    enc_repacks = encformer_repack_ops(cfg, use_scp=use_scp)
    base_repacks = baseline_repack_ops(cfg)

    ct_bytes = bp.ct_bytes_for(cfg.nslots)
    if not use_cc:
        print("  Note: CC disabled, so each logical conversion uses twice the ciphertext payload.")
    if not use_scp:
        print("  Note: SCP disabled, so the design pays repacking cost at stage boundaries.")
    print(f"  {variant_name}:  {enc_convs} conversions, {enc_repacks} repacks")
    print(f"  Baseline:   {base_convs} conversions, {base_repacks} repacks")
    print(f"  MPC rounds: {MPC_ROUNDS_PER_LAYER} per layer")
    print(f"  CT payload: {ct_bytes // 1024} KB ({'bridge wire' if bp.use_bridge_bytes else 'analytical'})")
    print(f"  Repack time: {bp.repack_time_s * 1000:.1f}ms per op")
    if bp.fhe_stage_times_s:
        print(f"  FHE compute: {bp.total_fhe_time_s():.2f}s/layer (calibrated)")
    print()

    header = (
        f"  {'Profile':<8s} {'T_conv(D)':<12s} {'T_conv(B)':<12s} "
        f"{'T_repack(B)':<12s} {'T_mpc':<10s} {'T(D)':<10s} "
        f"{'T(B)':<10s} {'Î”T':<10s} {'Better'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for p in profiles:
        d = compute_design_cost(variant_name, enc_convs, enc_repacks, cfg, p, backend=backend)
        b = compute_design_cost("B", base_convs, base_repacks, cfg, p, backend=backend)
        delta = d.t_total_s - b.t_total_s
        better = variant_name if delta < 0 else "Baseline"
        print(
            f"  {p.name:<8s} {d.t_conv_s:<12.4f} {b.t_conv_s:<12.4f} "
            f"{b.t_repack_s:<12.4f} {d.t_mpc_s:<10.4f} "
            f"{d.t_total_s:<10.4f} {b.t_total_s:<10.4f} "
            f"{delta:<+10.4f} {better}"
        )
    print()


def calibrate_repack_time(
    total_fhe_ks_time_s: float,
    total_ks_count: int,
) -> float:

    if total_ks_count <= 0:
        return 0.0
    return total_fhe_ks_time_s / total_ks_count


def calibrate_ct_bytes_from_bridge(
    measured_bytes: int,
    total_conversions: int,
) -> int:

    if total_conversions <= 0:
        return 0
    return measured_bytes // total_conversions


def calibrate_backend_from_timings(
    name: str,
    stage_timings: Dict[str, float],
    ks_counts: Dict[str, int] | None = None,
    measured_bridge_bytes: int | None = None,
    total_conversions: int | None = None,
    nslots: int | None = None,
) -> BackendProfile:

    if ks_counts:
        total_ks = sum(ks_counts.values())
        total_ks_time = sum(stage_timings.values())
        repack_s = calibrate_repack_time(total_ks_time, total_ks)
    else:
        repack_s = LEGACY_PROFILE.repack_time_s

    use_bridge = False
    if measured_bridge_bytes is not None and total_conversions and total_conversions > 0:
        effective_ct_bytes = calibrate_ct_bytes_from_bridge(measured_bridge_bytes, total_conversions)

        if nslots:
            bridge_est = estimate_bridge_bytes(nslots)
            analytical_est = estimate_ct_bytes(nslots)
            bridge_err = abs(effective_ct_bytes - bridge_est)
            analytical_err = abs(effective_ct_bytes - analytical_est)
            use_bridge = bridge_err < analytical_err

    return BackendProfile(
        name=name,
        repack_time_s=repack_s,
        use_bridge_bytes=use_bridge,
        fhe_stage_times_s=dict(stage_timings),
    )


def calibrate_from_native_output(results: Dict[str, dict]) -> BackendProfile:

    stage_timings: Dict[str, float] = {}
    ks_counts: Dict[str, int] = {}

    for stage_name, data in results.items():
        elapsed_s = data.get("elapsed_sec", data.get("elapsed_ms", 0) / 1000.0)
        stage_timings[stage_name] = elapsed_s

        ks = data.get("ks", {})
        total_ks = ks.get("ks_rots", 0) + ks.get("ks_muls_ctct", 0) + ks.get("ks_conj", 0)
        if total_ks > 0:
            ks_counts[stage_name] = total_ks

    return calibrate_backend_from_timings(
        name="phantom_calibrated",
        stage_timings=stage_timings,
        ks_counts=ks_counts if ks_counts else None,
    )


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description="EncFormer cost analysis (Section 5)")
    parser.add_argument(
        "--backend", default=None, choices=list(ALL_BACKEND_PROFILES.keys()), help="Backend profile (default: show all)"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["bert-base", "bert-large", "gpt2-base"],
        help="Model config (default: show all)",
    )
    args = parser.parse_args()

    from src.models.model_config import BERT_BASE, BERT_LARGE, GPT2_BASE

    configs = [BERT_BASE, BERT_LARGE, GPT2_BASE]
    if args.model:
        configs = [get_config(args.model)]

    backends = [LEGACY_PROFILE, PHANTOM_GPU_PROFILE, PLAIN_PROFILE]
    if args.backend:
        backends = [get_backend_profile(args.backend)]

    for cfg in configs:
        for bp in backends:
            print_cost_comparison(cfg, backend=bp)


if __name__ == "__main__":
    main()
