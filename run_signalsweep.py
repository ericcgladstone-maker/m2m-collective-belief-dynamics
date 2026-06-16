"""Signal-structure robustness sweep (reviewer concern #3). Across three variants of
the misleading-majority structure (narrow gap, wide gap, heavy wrong-majority), confirm
the crowd starts wise (baseline) and that evidence integrates while conclusions herd —
so the content effect is not an artifact of one signal parameterization. n=5, gpt-4.1-mini."""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "gpt-4.1-mini"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)

import run_full
run_full.N_AGENTS = 5
run_full.OUT_TAG = "gpt-4.1-mini_sigsweep"
run_full.REGIMES = ["mm_gap_narrow", "mm_gap_wide", "mm_maj_heavy"]
run_full.TOPO_CONTENT = [
    ("baseline", "none"),
    ("all_channel", "evidence_only"),
    ("all_channel", "conclusion_only"),
]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_full.main(reps=reps, workers=workers)
