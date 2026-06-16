"""Small-world condition (n=11) — the path-length-vs-density dissociation.

A real small-world needs ~10+ nodes (Mutation used 11), so this is a SCALED
condition, kept separate from the n=5 core (own data dir). It pits three n=11
structures under the same evidence/conclusion manipulation:
  - circle           : sparse, long paths, high clustering
  - all_channel      : dense, short paths
  - ring_shortcuts   : ring backbone + 3 long-range chords = SHORT paths (like
                       dense) but LOW density / high clustering (like sparse)

ring_shortcuts therefore dissociates the two properties all-channel confounds:
if conclusion-herding tracks short path length, small-world herds like all-channel;
if it tracks density, small-world resists like circle. Reuses run_full's machinery.
"""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "gpt-4.1-mini"          # gate-validated, affordable substrate for the scaled run
config.PROVIDER = config.provider_for(config.AGENT_MODEL)

import run_full

run_full.N_AGENTS = 11
run_full.OUT_TAG = "gpt-4.1-mini_n11"        # separate dir — never mixes with the n=5 core
run_full.REGIMES = ["misleading_majority", "concordant"]
run_full.TOPO_CONTENT = [
    ("baseline", "none"),
    ("circle", "evidence_only"),
    ("circle", "conclusion_only"),
    ("circle", "both"),
    ("all_channel", "evidence_only"),
    ("all_channel", "conclusion_only"),
    ("all_channel", "both"),
    ("ring_shortcuts", "evidence_only"),
    ("ring_shortcuts", "conclusion_only"),
    ("ring_shortcuts", "both"),
]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_full.main(reps=reps, workers=workers)
