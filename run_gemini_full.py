"""Third-family replication on Gemini (reviewer concern #7). gemini-2.5-flash passed the
multi-evidence adequacy gate (misleading_majority MAE 0.02, truth 1.00) once the thinking
budget was raised to 8192. Replicate the decisive CONTENT cells (baseline + all_channel x
{evidence, conclusion, both}) so 'substrate-general' rests on three families across three
vendors, not two closed APIs."""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "gemini-2.5-flash"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)
config.GEMINI_THINKING = True   # required for multi-evidence integration

import run_full
run_full.N_AGENTS = 5
run_full.OUT_TAG = "gemini-2.5-flash"
run_full.REGIMES = ["misleading_majority", "concordant"]
run_full.TOPO_CONTENT = [
    ("baseline", "none"),
    ("all_channel", "evidence_only"),
    ("all_channel", "conclusion_only"),
    ("all_channel", "both"),
]
# correct cost labels for the gemini substrate (config computed defaults for gpt-4.1-mini at import)
if "gemini-2.5-flash" in config._PRICES:
    run_full.COST_IN, run_full.COST_OUT = config._PRICES["gemini-2.5-flash"]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run_full.main(reps=reps, workers=workers)
