"""Centerpiece mechanism figure: assignment-based, truth-anchored role transitions with
seed-clustered paired-bootstrap CIs. Panel A: standard misleading majority — weak-majority
correction and strong-minority capitulation under evidence vs verdict sharing. Panel B: the same
at extreme fan-in (n=11), where the content contrast disappears (overload). Mirrors
analyze_stats9.py exactly. System python3. Writes ../figures/fig3_mechanism.png."""
import csv, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 8.5, "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})
EVID = "#009E73"   # evidence (teal, colorblind-safe)
CONC = "#D55E00"   # verdict (vermillion)
random.seed(0)
HERE = Path(__file__).parent
FIG = HERE.parent / "figures"

def true_state(tag, rg, tp):
    T = {}
    with open(HERE / "data" / tag / "rounds.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) == (rg, tp):
                T[(r["content_mode"], r["seed"])] = r["true_state"]
    return T

def load(tag, rg, tp):
    B = defaultdict(lambda: defaultdict(dict))
    with open(HERE / "data" / tag / "beliefs.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) == (rg, tp):
                B[(r["content_mode"], r["seed"])][int(r["round"])][r["agent"]] = (
                    float(r["belief_pA"]), float(r["llr"]))
    return B

def per_seed(B, T, cm):
    out = {}
    for (c, s), rounds in B.items():
        if c != cm or (c, s) not in T:
            continue
        rs = sorted(rounds)
        if len(rs) < 2:
            continue
        t0, tT = rs[0], rs[-1]
        truth_A = (T[(c, s)] == "A")
        agents = sorted(set(rounds[t0]) & set(rounds[tT]))
        if len(agents) < 3:
            continue
        corr, capit = [], []
        for a in agents:
            _, llr = rounds[t0][a]
            bT = rounds[tT][a][0]
            sig_truth = (llr > 0) == truth_A
            ends_true = (bT > 0.5) == truth_A
            (capit if sig_truth else corr).append(
                (0.0 if ends_true else 1.0) if sig_truth else (1.0 if ends_true else 0.0))
        out[s] = (np.mean(corr) if corr else np.nan, np.mean(capit) if capit else np.nan)
    return out

def est_ci(d, idx, B=10000):
    seeds = sorted(d)
    vals = [d[s][idx] for s in seeds]
    base = np.nanmean(vals)
    n = len(seeds)
    bs = [np.nanmean([d[seeds[random.randrange(n)]][idx] for _ in range(n)]) for _ in range(B)]
    bs.sort()
    return base, bs[int(.025*B)], bs[int(.975*B)]

def panel(ax, tag, rg, tp, title):
    B = load(tag, rg, tp); T = true_state(tag, rg, tp)
    ev, co = per_seed(B, T, "evidence_only"), per_seed(B, T, "conclusion_only")
    # groups: weak-majority correction (idx0), strong-minority capitulation (idx1)
    groups = ["weak-majority\ncorrection", "strong-minority\ncapitulation"]
    x = np.arange(2); wbar = 0.36
    for j, (d, lab, col) in enumerate([(ev, "evidence", EVID), (co, "verdict", CONC)]):
        ys, los, his = [], [], []
        for idx in (0, 1):
            b, lo, hi = est_ci(d, idx)
            ys.append(b); los.append(max(0, b-lo)); his.append(max(0, hi-b))
        ax.bar(x + (j-0.5)*wbar, ys, wbar, label=lab, color=col,
               yerr=[los, his], capsize=4, error_kw=dict(lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05); ax.set_ylabel("P(terminal transition)")
    ax.axhline(0.5, ls=":", c="gray", lw=0.8); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
panel(axA, "gpt-4.1-mini", "misleading_majority", "all_channel",
      "A  Standard misleading majority (n=5)")
panel(axB, "gpt-4.1-mini_n11", "misleading_majority", "all_channel",
      "B  Extreme fan-in (n=11): content contrast disappears")
fig.suptitle("Evidence enables majority correction and limits minority capitulation; "
             "verdicts do the reverse", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG / "fig3_mechanism.png", dpi=200); plt.close()
print("wrote fig3_mechanism.png")
