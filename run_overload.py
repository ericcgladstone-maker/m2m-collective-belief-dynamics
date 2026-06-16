"""Overload isolation (review): does EVIDENCE-sharing degrade with per-node degree even in
the CONCORDANT regime (no misleading majority)? Fills the evidence-sharing degree gradient at
n=11 — lattice_k4/k6/k8 — to combine with existing circle (deg 2) and all-channel (deg 10)
evidence-concordant cells. Monotonic decline ⇒ integration overload is fan-in-driven and
regime-general, isolating it from group size n. gpt-4.1-mini, concordant only (cheap)."""
from __future__ import annotations
import sys
import config
config.AGENT_MODEL = "gpt-4.1-mini"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)

import run_full
run_full.N_AGENTS = 11
run_full.OUT_TAG = "gpt-4.1-mini_overload"
run_full.REGIMES = ["concordant"]
run_full.TOPO_CONTENT = [
    ("lattice_k4", "evidence_only"),
    ("lattice_k6", "evidence_only"),
    ("lattice_k8", "evidence_only"),
]

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_full.main(reps=reps, workers=workers)
