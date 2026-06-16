"""Power up the cross-family CONTENT claim on claude-sonnet-4-6: bring the decisive
cells (baseline + all_channel x {evidence, conclusion, both}, both regimes) from N=8
to N=50. Resumable — merges into data/claude-sonnet-4-6/."""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "claude-sonnet-4-6"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)

import run_full
run_full.OUT_TAG = "claude-sonnet-4-6"
run_full.REGIMES = ["misleading_majority", "concordant"]
run_full.TOPO_CONTENT = [
    ("baseline", "none"),
    ("all_channel", "evidence_only"),
    ("all_channel", "conclusion_only"),
    ("all_channel", "both"),
]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4   # gentle on Anthropic rate limits
    run_full.main(reps=reps, workers=workers)
