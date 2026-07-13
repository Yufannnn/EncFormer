from __future__ import annotations

from typing import Any

import numpy as np


def prepare_layer_running_denoms(
    running_denoms: dict[str, np.ndarray] | None,
    layer_idx: int,
) -> dict[str, np.ndarray]:

    prepared: dict[str, np.ndarray] = {}
    if running_denoms is None:
        return prepared

    pfx = f"layer_{layer_idx}_"
    mapping = (
        (f"{pfx}bpmax_running_denominator", "bpmax_running_denominator"),
        (f"{pfx}ln1_running_denominator", "ln1_running_denominator"),
        (f"{pfx}ln2_running_denominator", "ln2_running_denominator"),
    )
    for src_key, dst_key in mapping:
        if src_key not in running_denoms:
            continue
        rd = np.asarray(running_denoms[src_key], dtype=np.float64)
        if dst_key == "bpmax_running_denominator":
            if rd.ndim == 4:
                rd = rd.squeeze(0).squeeze(-1)
            elif rd.ndim == 3:
                rd = rd.squeeze(-1)
        else:
            if rd.ndim == 3:
                rd = rd.squeeze(0)
        prepared[dst_key] = rd
    return prepared


def inject_layer_running_denoms(
    mpc_backend: Any,
    prepared_denoms: dict[str, np.ndarray] | None,
) -> None:

    keys = (
        "bpmax_running_denominator",
        "ln1_running_denominator",
        "ln2_running_denominator",
    )
    prepared = prepared_denoms or {}

    to_set = {k: prepared.get(k) for k in keys}
    mpc_backend.set_running_denominators(to_set)
