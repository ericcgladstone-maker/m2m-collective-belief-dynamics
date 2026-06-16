"""Dependence diagnostic (review point: separate demonstrated majority-movement from the
INFERRED dependence-amplification mechanism). GPT's suggestion: measure correlation in signed
belief UPDATES, not raw diversity — diversity can rise while updates become socially aligned.

For gpt-4.1-mini, misleading-majority, all-channel, per condition we compute, from beliefs.csv:
 (1) mean pairwise Pearson r of per-agent update vectors (Δbelief across the 3 comm rounds),
 (2) update-direction concentration |mean sign(Δ)| per transition (1 = lockstep, 0 = balanced),
 (3) PC1 share of cross-agent update variance (effective-dimensionality / common-component).
If conclusion >> evidence on these, the dependence mechanism has a measured correlate; if not,
it stays explicitly inferential. Writes stats_report5.md. System python3 (numpy)."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)

TAG = "gpt-4.1-mini"
RG, TP = "misleading_majority", "all_channel"
CONDS = [("evidence_only", "evidence"), ("both", "both"), ("conclusion_only", "conclusion")]

# beliefs[(content,seed)][round] -> {agent: belief}
B = defaultdict(lambda: defaultdict(dict))
with open(HERE / "data" / TAG / "beliefs.csv") as f:
    for r in csv.DictReader(f):
        if (r["regime"], r["topology"]) != (RG, TP):
            continue
        B[(r["content_mode"], r["seed"])][int(r["round"])][r["agent"]] = float(r["belief_pA"])

w("# Dependence diagnostic — correlation in belief UPDATES (analyze_stats5.py)\n")
w(f"{TAG}, {RG}, {TP}. Updates Δ = belief(round t) − belief(round t−1) over the 3 communication rounds.\n")
w("| condition | mean pairwise r(update vectors) | update-direction concentration |mean sign| | PC1 share of update variance | trials |")
w("|---|---|---|---|---|")

for cm, lab in CONDS:
    trials = [(c, s) for (c, s) in B if c == cm]
    pair_rs, concs, pc1s = [], [], []
    for key in trials:
        rounds = B[key]
        rs = sorted(rounds)
        if len(rs) < 3:
            continue
        agents = sorted(set.intersection(*[set(rounds[t]) for t in rs]))
        if len(agents) < 3:
            continue
        # update matrix U: agents x transitions
        U = np.array([[rounds[rs[i+1]][a] - rounds[rs[i]][a] for i in range(len(rs)-1)] for a in agents])
        # (1) mean pairwise correlation across agents (skip zero-variance agents)
        var = U.var(axis=1)
        good = np.where(var > 1e-9)[0]
        if len(good) >= 2:
            C = np.corrcoef(U[good])
            iu = np.triu_indices(len(good), 1)
            vals = C[iu]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                pair_rs.append(vals.mean())
        # (2) update-direction concentration per transition, averaged
        sg = np.sign(U)
        conc = np.abs(sg.mean(axis=0)).mean()
        concs.append(conc)
        # (3) PC1 share of cross-agent update variance (transitions as observations)
        M = U - U.mean(axis=1, keepdims=True)
        cov = np.cov(M)  # agents x agents
        ev = np.linalg.eigvalsh(cov)
        ev = ev[ev > 0]
        if ev.sum() > 0:
            pc1s.append(ev.max() / ev.sum())
    def m(x): return f"{np.mean(x):.3f}" if x else "—"
    w(f"| {lab} | {m(pair_rs)} | {m(concs)} | {m(pc1s)} | {len(trials)} |")

w("\nReading: higher pairwise update-correlation, higher direction-concentration, and higher PC1 share")
w("under conclusion-sharing would mean agents' belief CHANGES co-move (socially aligned updates) even")
w("though belief LEVELS stay dispersed — a measured correlate of dependence amplification. If conclusion")
w("≈ evidence, the dependence account remains inferential and should be labeled as such.")
(HERE / "stats_report5.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
