"""Degree-vs-path-length identification (reviewer concern #2). At n=11, conclusion_only,
fill in the degree gradient with ring-lattice topologies of degree 4, 6, 8. Combined with
existing circle (deg 2, diam 5), ring_shortcuts (deg ~2.5, diam 3), and all_channel
(deg 10, diam 1), this yields contrasts that hold PATH LENGTH constant while varying DEGREE:
  - diameter 3: ring_shortcuts (deg 2.5) vs lattice_k4 (deg 4)
  - diameter 2: lattice_k6 (deg 6) vs lattice_k8 (deg 8)
If herding tracks degree (fan-in), accuracy falls with degree at fixed diameter."""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "gpt-4.1-mini"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)

import run_full
run_full.N_AGENTS = 11
run_full.OUT_TAG = "gpt-4.1-mini_degree"
run_full.REGIMES = ["misleading_majority", "concordant"]
run_full.TOPO_CONTENT = [
    ("lattice_k4", "conclusion_only"),
    ("lattice_k6", "conclusion_only"),
    ("lattice_k8", "conclusion_only"),
]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_full.main(reps=reps, workers=workers)
