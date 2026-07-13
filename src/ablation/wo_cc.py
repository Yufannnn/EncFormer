from __future__ import annotations

import os

from src.cost_model import print_cost_comparison
from src.encformer import run
from src.models.model_config import get_config


def main() -> None:
    run(use_cc=False, use_scp=True)

    cfg = get_config(os.environ.get("ENCFORMER_MODEL", "bert-base"))
    print("\n--- wo_CC Cost Analysis ---")
    print("Without complex-conjugate packing, each conversion handles only")
    print("real slots (halving throughput), effectively doubling conversion payload.")
    print_cost_comparison(cfg, use_cc=False, use_scp=True)


if __name__ == "__main__":
    main()
