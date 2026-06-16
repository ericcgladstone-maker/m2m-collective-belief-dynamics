"""WITHIN-GRAPH fan-in test (Bavelas-proper). The degree ladder varies fan-in BETWEEN graphs;
here we ask whether an agent's OWN fan-in predicts its capitulation WITHIN a single heterogeneous-
degree graph (wheel: hub deg n-1 vs spokes deg 1; hierarchy: by depth). `n_received` at round 1 is
the agent's realized fan-in. Among strong-minority agents (signal points to truth) under
conclusion-sharing, misleading-majority, does higher own fan-in → more capitulation? This controls
for between-graph confounds the degree ladder cannot. n=5 (small per-position samples — read as
suggestive). System python3. Writes stats_report11.md."""
import csv, random
from collections import defaultdict
from pathlib import Path
import numpy as np
random.seed(0)
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)

TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
TOPOS = ["wheel", "hierarchy", "chain"]   # heterogeneous-degree graphs

def true_state():
    T = {}
    with open(HERE / "data" / TAG / "rounds.csv") as f:
        for r in csv.DictReader(f):
            if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in TOPOS:
                T[(r["topology"], r["seed"])] = r["true_state"]
    return T

# per (topo,seed,agent): llr, terminal belief, fan-in (n_received at round 1)
rows = defaultdict(dict)
fanin = {}
with open(HERE / "data" / TAG / "beliefs.csv") as f:
    for r in csv.DictReader(f):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS:
            continue
        k = (r["topology"], r["seed"], r["agent"]); rd = int(r["round"])
        rows[k][rd] = (float(r["belief_pA"]), float(r["llr"]))
        if rd == 1:
            fanin[k] = int(r["n_received"])

T = true_state()
# collect minority (strong-truth) agents: (topo, seed, fanin, capitulated)
recs = []
for (tp, s, a), byr in rows.items():
    if (tp, s) not in T or (tp, s, a) not in fanin:
        continue
    rs = sorted(byr); b0, llr = byr[rs[0]]; bT = byr[rs[-1]][0]
    truth_A = (T[(tp, s)] == "A")
    if (llr > 0) != truth_A:        # keep only strong-minority (signal toward truth)
        continue
    recs.append((tp, s, fanin[(tp, s, a)], 1 if ((bT > 0.5) != truth_A) else 0))

w("# Within-graph fan-in vs capitulation (analyze_stats11.py)\n")
w(f"{TAG}, {RG}, {CM}. Strong-minority agents only; fan-in = messages received (round 1). "
  "Capitulation = minority agent ends on the false side.\n")

# (1) within WHEEL: hub (high fan-in) vs spokes (low fan-in)
w("## Within-graph contrast by own fan-in\n")
w("| topology | fan-in group | capitulation | minority agents |")
w("|---|---|---|---|")
for tp in TOPOS:
    sub = [r for r in recs if r[0] == tp]
    if not sub: continue
    fis = sorted(set(r[2] for r in sub))
    lo_thr = fis[0]; hi_thr = fis[-1]
    if hi_thr == lo_thr:
        continue
    lo = [r[3] for r in sub if r[2] == lo_thr]
    hi = [r[3] for r in sub if r[2] == hi_thr]
    w(f"| {tp} | low (fan-in {lo_thr}) | {np.mean(lo):.2f} | {len(lo)} |")
    w(f"| {tp} | high (fan-in {hi_thr}) | {np.mean(hi):.2f} | {len(hi)} |")

# (2) pooled: capitulation by own fan-in across heterogeneous graphs (seed-clustered bootstrap on slope)
w("\n## Pooled: capitulation by own fan-in (all heterogeneous-degree graphs)\n")
byf = defaultdict(list)
for tp, s, fi, cap in recs:
    byf[fi].append(cap)
w("| own fan-in | capitulation | minority agents |")
w("|---|---|---|")
for fi in sorted(byf):
    w(f"| {fi} | {np.mean(byf[fi]):.2f} | {len(byf[fi])} |")

# point-biserial-ish correlation of fan-in with capitulation + seed-clustered bootstrap
fis = np.array([r[2] for r in recs], float); caps = np.array([r[3] for r in recs], float)
r_obs = np.corrcoef(fis, caps)[0, 1]
seeds = sorted(set((r[0], r[1]) for r in recs))
by_seed = defaultdict(list)
for r in recs: by_seed[(r[0], r[1])].append(r)
boots = []
n = len(seeds)
for _ in range(10000):
    samp = []
    for _ in range(n): samp.extend(by_seed[seeds[random.randrange(n)]])
    f = np.array([x[2] for x in samp], float); c = np.array([x[3] for x in samp], float)
    if f.std() > 0 and c.std() > 0: boots.append(np.corrcoef(f, c)[0, 1])
boots.sort()
w(f"\nFan-in × capitulation correlation r = {r_obs:+.2f} "
  f"[seed-clustered bootstrap 95% CI {boots[250]:+.2f}, {boots[9750]:+.2f}], "
  f"{len(recs)} minority agents across {len(seeds)} trials.")
w("\nReading: positive → within heterogeneous graphs, agents with higher OWN fan-in capitulate more,")
w("a within-graph replication of the between-graph degree gradient. Small n=5 per-position samples —")
w("report as a robustness check, not a primary result.")
(HERE / "stats_report11.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
