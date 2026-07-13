from __future__ import annotations

import os

from src.cost_model import print_cost_comparison
from src.encformer import run
from src.models.model_config import get_config


def main() -> None:
    run(use_cc=True, use_scp=False)

    cfg = get_config(os.environ.get("ENCFORMER_MODEL", "bert-base"))
    print("\n--- wo_SCP Cost Analysis ---")
    print("Without stage-compatible patterns, repacking is needed at every")
    print("stage boundary, increasing compute overhead at each stage handoff.")
    print_cost_comparison(cfg, use_cc=True, use_scp=False)


if __name__ == "__main__":
    main()
