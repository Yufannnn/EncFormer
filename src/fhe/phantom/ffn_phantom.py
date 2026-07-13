from __future__ import annotations

import src.fhe.desilo.ffn_desilo as _impl
from src.fhe.phantom.phantom_runtime import (
    LevelCKKSContext,
    attempts_summary,
    probe_max_level,
    search_lowest_working_level,
    set_visible_gpus,
)

_impl.LevelCKKSContext = LevelCKKSContext
_impl.attempts_summary = attempts_summary
_impl.probe_max_level = probe_max_level
_impl.search_lowest_working_level = search_lowest_working_level
_impl.set_visible_gpus = set_visible_gpus

run_once = _impl.run_once
main = _impl.main


if __name__ == "__main__":
    main()
