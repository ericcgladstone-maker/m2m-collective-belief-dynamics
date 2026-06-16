"""Within-family reasoning-effort test (disambiguates the Gemini result). SAME model
(claude-sonnet-4-6), SAME task/topology/content (all_channel, conclusion_only,
misleading_majority + concordant), toggling ONE knob: extended private reasoning ON
(vs the existing thinking-OFF cell at 0.62). If thinking-ON resists herding, per-agent
reasoning effort — not the Gemini family — is the lever. Not a benchmark: a controlled
single-variable manipulation, read mechanistically (ē/D dynamics, round-0 calibration).
Sized to ~$18 of the ~$20 Anthropic budget (~$0.30/comms-trial metered)."""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "claude-sonnet-4-6"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)
config.CLAUDE_THINKING = True
config.CLAUDE_THINKING_BUDGET = 2048   # ceiling; model self-limits (~490 observed)

import run_full
run_full.N_AGENTS = 5
run_full.OUT_TAG = "claude-sonnet-4-6_think"   # separate dir; never mixes with thinking-off
run_full.REGIMES = ["misleading_majority", "concordant"]
run_full.TOPO_CONTENT = [("all_channel", "conclusion_only")]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run_full.main(reps=reps, workers=workers)
