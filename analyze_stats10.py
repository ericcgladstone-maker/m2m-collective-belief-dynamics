"""Does the BEHAVIORAL mechanism scale with fan-in? Under conclusion-sharing (misleading, n=11),
compute weak-majority correction and strong-minority capitulation across the degree ladder
(circle 2 → all-channel 10). If capitulation rises and correction falls with degree, fan-in
amplifies majority-deference behaviorally (not only the accuracy loss). Assignment-based roles,
truth-anchored, seeds as the unit. System python3. Writes stats_report10.md."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)

# (tag, topology, degree) under conclusion_only, misleading, n=11
LADDER = [
    ("gpt-4.1-mini_n11", "circle", 2.0),
    ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55),
    ("gpt-4.1-mini_degree", "lattice_k4", 4.0),
    ("gpt-4.1-mini_degree", "lattice_k6", 6.0),
    ("gpt-4.1-mini_degree", "lattice_k8", 8.0),
    ("gpt-4.1-mini_n11", "all_channel", 10.0),
]

def true_state(tag, tp):
    T = {}
    p = HERE / "data" / tag / "rounds.csv"
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) == ("misleading_majority", tp):
                T[r["seed"]] = r["true_state"]
    return T

def load(tag, tp, cm):
    B = defaultdict(lambda: defaultdict(dict))
    with open(HERE / "data" / tag / "beliefs.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"], r["content_mode"]) == ("misleading_majority", tp, cm):
                B[r["seed"]][int(r["round"])][r["agent"]] = (float(r["belief_pA"]), float(r["llr"]))
    return B

def rates(tag, tp, cm):
    T = true_state(tag, tp); B = load(tag, tp, cm)
    corr, capit = [], []
    for s, rounds in B.items():
        if s not in T: continue
        rs = sorted(rounds)
        if len(rs) < 2: continue
        t0, tT = rs[0], rs[-1]
        truth_A = (T[s] == "A")
        agents = sorted(set(rounds[t0]) & set(rounds[tT]))
        if len(agents) < 3: continue
        c, k = [], []
        for a in agents:
            _, llr = rounds[t0][a]; bT = rounds[tT][a][0]
            sig_truth = (llr > 0) == truth_A
            ends_true = (bT > 0.5) == truth_A
            if sig_truth: k.append(0.0 if ends_true else 1.0)
            else: c.append(1.0 if ends_true else 0.0)
        if c: corr.append(np.mean(c))
        if k: capit.append(np.mean(k))
    return (np.mean(corr) if corr else np.nan, np.mean(capit) if capit else np.nan, len(B))

w("# Behavioral mechanism vs. fan-in (analyze_stats10.py)\n")
w("Conclusion-sharing, misleading-majority, n=11, across the degree ladder. Assignment-based roles.\n")
w("| degree | topology | weak-majority correction | strong-minority capitulation | seeds |")
w("|---|---|---|---|---|")
deg_list, cap_list, cor_list = [], [], []
for tag, tp, dg in LADDER:
    cor, cap, n = rates(tag, tp, "conclusion_only")
    w(f"| {dg} | {tp} | {cor:.2f} | {cap:.2f} | {n} |")
    if not np.isnan(cap):
        deg_list.append(dg); cap_list.append(cap); cor_list.append(cor)
# trend: correlation of capitulation/correction with degree
if len(deg_list) > 2:
    rc = np.corrcoef(deg_list, cap_list)[0, 1]
    rr = np.corrcoef(deg_list, cor_list)[0, 1]
    w(f"\nTrend across degree (conclusion-sharing): capitulation vs degree r={rc:+.2f}; correction vs degree r={rr:+.2f}.")
w("\nReading: if capitulation rises and correction falls with degree, fan-in amplifies the majority-")
w("deference BEHAVIOR (not only the accuracy loss). NB the highest-degree cell (all-channel) is also")
w("the overload regime, so a clean monotone rise up to ~degree 8 is the cleaner read.")
(HERE / "stats_report10.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
