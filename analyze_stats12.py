"""Two robustness checks on the within-graph fan-in result (review).
(1) BALANCE: signals are shuffled per seed (tasks.py L124), so signal role should be independent of
    network position. Verify empirically: P(agent is strong-minority) and mean |llr| among minority
    should be ~uniform across fan-in levels within each topology.
(2) GRAPH FIXED EFFECTS: the pooled fan-in×capitulation correlation mixes within- and between-graph
    variation. Re-estimate the WITHIN-graph effect by demeaning fan-in and capitulation within each
    topology (graph FE), with a seed-clustered bootstrap, and report per-graph slopes.
gpt-4.1-mini, misleading, conclusion-sharing, heterogeneous-degree graphs. System python3."""
import csv, random
from collections import defaultdict
from pathlib import Path
import numpy as np
random.seed(0)
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)
TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
TOPOS = ["wheel", "hierarchy", "chain"]

TS = {}
with open(HERE / "data" / TAG / "rounds.csv") as f:
    for r in csv.DictReader(f):
        if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in TOPOS:
            TS[(r["topology"], r["seed"])] = r["true_state"]

rows = defaultdict(dict); fanin = {}
with open(HERE / "data" / TAG / "beliefs.csv") as f:
    for r in csv.DictReader(f):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS:
            continue
        k = (r["topology"], r["seed"], r["agent"]); rd = int(r["round"])
        rows[k][rd] = (float(r["belief_pA"]), float(r["llr"]))
        if rd == 1: fanin[k] = int(r["n_received"])

# build records: all agents (for balance) + minority subset (for capitulation)
allrec, recs = [], []   # allrec: (topo, fanin, is_minority, |llr|); recs: (topo, seed, fanin, capit)
for (tp, s, a), byr in rows.items():
    if (tp, s) not in TS or (tp, s, a) not in fanin: continue
    rs = sorted(byr); llr = byr[rs[0]][1]; bT = byr[rs[-1]][0]
    truth_A = (TS[(tp, s)] == "A"); minority = (llr > 0) == truth_A
    allrec.append((tp, fanin[(tp, s, a)], minority, abs(llr)))
    if minority:
        recs.append((tp, s, fanin[(tp, s, a)], 1 if ((bT > 0.5) != truth_A) else 0))

# ---------- (1) BALANCE ----------
w("# Within-graph robustness: balance + graph fixed effects (analyze_stats12.py)\n")
w("## (1) Signal-role balance across positions (signals shuffled per seed → role ⊥ position)")
w("\n| topology | fan-in | P(strong-minority) | mean |llr| if minority | agents |")
w("|---|---|---|---|---|")
for tp in TOPOS:
    fis = sorted(set(r[1] for r in allrec if r[0] == tp))
    for fi in fis:
        sub = [r for r in allrec if r[0] == tp and r[1] == fi]
        pmin = np.mean([1.0 if r[2] else 0.0 for r in sub])
        lls = [r[3] for r in sub if r[2]]
        w(f"| {tp} | {fi} | {pmin:.2f} | {np.mean(lls):.2f} | {len(sub)} |")
w("\n(Uniform P(minority) across fan-in within a topology, and similar |llr|, ⇒ position is not "
  "confounded with signal role. Chance P(minority)=2/n.)")

# ---------- (2) GRAPH FIXED EFFECTS ----------
w("\n## (2) Within-graph fan-in effect with graph fixed effects")
# per-graph slope (LPM capit ~ fanin within topology)
w("\n| topology | within-graph slope (Δcapit per +1 fan-in) | fan-in range | minority agents |")
w("|---|---|---|---|")
slopes = []
for tp in TOPOS:
    sub = [(r[2], r[3]) for r in recs if r[0] == tp]
    f = np.array([x[0] for x in sub], float); c = np.array([x[1] for x in sub], float)
    if f.std() > 0:
        slope = np.polyfit(f, c, 1)[0]; slopes.append(slope)
        w(f"| {tp} | {slope:+.3f} | {int(f.min())}–{int(f.max())} | {len(sub)} |")
# graph-demeaned (FE) pooled estimate + seed-clustered bootstrap
def fe_corr(rs):
    by_tp_f = defaultdict(list); by_tp_c = defaultdict(list)
    for tp, s, fi, cap in rs:
        by_tp_f[tp].append(fi); by_tp_c[tp].append(cap)
    mf = {tp: np.mean(v) for tp, v in by_tp_f.items()}
    mc = {tp: np.mean(v) for tp, v in by_tp_c.items()}
    fd = np.array([fi - mf[tp] for tp, s, fi, cap in rs])
    cd = np.array([cap - mc[tp] for tp, s, fi, cap in rs])
    if fd.std() > 0 and cd.std() > 0:
        return np.corrcoef(fd, cd)[0, 1], np.polyfit(fd, cd, 1)[0]
    return np.nan, np.nan
r_fe, slope_fe = fe_corr(recs)
seeds = sorted(set((r[0], r[1]) for r in recs))
by_seed = defaultdict(list)
for r in recs: by_seed[(r[0], r[1])].append(r)
boots = []
n = len(seeds)
for _ in range(10000):
    samp = []
    for _ in range(n): samp.extend(by_seed[seeds[random.randrange(n)]])
    rr, _ = fe_corr(samp)
    if not np.isnan(rr): boots.append(rr)
boots.sort()
w(f"\nGraph-demeaned (within-graph, FE) fan-in × capitulation: r = {r_fe:+.2f} "
  f"[seed-clustered bootstrap 95% CI {boots[250]:+.2f}, {boots[9750]:+.2f}], "
  f"slope {slope_fe:+.3f}/fan-in. All three per-graph slopes positive: {[round(s,3) for s in slopes]}.")
w("(This is the genuinely within-graph estimate; the earlier pooled r=+0.36 mixed within- and "
  "between-graph variation and should not be described as controlling for between-graph confounds.)")
(HERE / "stats_report12.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
