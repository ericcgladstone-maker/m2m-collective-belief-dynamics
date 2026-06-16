"""Generate manuscript data figures from the saved raw data (run with system python3).
Numbering matches the manuscript: Fig 2 content effect+diversity, Fig 3 mechanism (make_fig_mechanism.py),
Fig 4 causal, Fig 5 structure, Fig 6 two failure modes. Fig 1 (design) = make_fig_design.py.
Colorblind-safe Wong palette; evidence=teal, conclusion/verdict=vermillion, structure/accuracy=blue."""
import csv, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "tight",
})
# Wong (2011) colorblind-safe palette
EVID = "#009E73"    # bluish-green — evidence / accurate
CONC = "#D55E00"    # vermillion  — conclusion / verdict / herd
STRUCT = "#0072B2"  # blue        — structure / collective accuracy
FAM2 = "#E69F00"    # amber       — second model family

HERE = Path(__file__).parent
FIG = HERE.parent / "figures"; FIG.mkdir(exist_ok=True)


def terminal_rows(tag, regime, topo, content):
    best = {}
    with open(HERE / "data" / tag / "rounds.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"], r["content_mode"]) != (regime, topo, content):
                continue
            s = r["seed"]
            if s not in best or int(r["round"]) > int(best[s]["round"]):
                best[s] = r
    return list(best.values())


def acc_ci(rows):
    n = len(rows); k = sum(int(r["acc_logodds"]) for r in rows)
    p = k / n; z = 1.96; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0, c-h), min(1, c+h)


def round_traj(tag, regime, topo, content, field):
    by = defaultdict(list)
    with open(HERE / "data" / tag / "rounds.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"], r["content_mode"]) == (regime, topo, content):
                by[int(r["round"])].append(float(r[field]))
    return [sum(by[k])/len(by[k]) for k in sorted(by)]


def capit_by_cell(tag, regime, topo, content):
    """strong-minority capitulation rate (assignment-based, truth-anchored) for one cell."""
    TS = {}
    with open(HERE / "data" / tag / "rounds.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"], r["content_mode"]) == (regime, topo, content):
                TS[r["seed"]] = r["true_state"]
    rws = defaultdict(dict)
    with open(HERE / "data" / tag / "beliefs.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"], r["content_mode"]) == (regime, topo, content):
                rws[(r["seed"], r["agent"])][int(r["round"])] = (float(r["belief_pA"]), float(r["llr"]))
    caps = []
    for (s, a), byr in rws.items():
        if s not in TS:
            continue
        rs = sorted(byr); bT = byr[rs[-1]][0]; llr = byr[rs[0]][1]
        truth_A = (TS[s] == "A")
        if (llr > 0) == truth_A:   # strong-minority (signal toward truth)
            caps.append(0.0 if ((bT > 0.5) == truth_A) else 1.0)
    return sum(caps) / len(caps) if caps else float("nan")


# ---------- FIG 2: content effect + mechanism ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))
conts = [("baseline", "none", "baseline"), ("all_channel", "evidence_only", "evidence"),
         ("all_channel", "both", "both"), ("all_channel", "conclusion_only", "conclusion")]
for j, (tag, lab, n, col) in enumerate([("gpt-4.1-mini", "gpt-4.1-mini", 100, STRUCT), ("claude-sonnet-4-6", "claude-sonnet-4-6", 50, FAM2)]):
    xs, ys, los, his = [], [], [], []
    for i, (tp, ct, name) in enumerate(conts):
        rows = terminal_rows(tag, "misleading_majority", tp, ct)
        if not rows: continue
        p, lo, hi = acc_ci(rows)
        xs.append(i + (j-0.5)*0.18); ys.append(p); los.append(max(0.0, p-lo)); his.append(max(0.0, hi-p))
    axA.bar(xs, ys, width=0.18, label=lab, yerr=[los, his], capsize=3, color=col)
axA.set_xticks(range(len(conts))); axA.set_xticklabels([c[2] for c in conts])
axA.set_ylabel("terminal collective accuracy"); axA.set_ylim(0, 1.05)
axA.axhline(0.5, ls=":", c="gray", lw=0.8); axA.legend(fontsize=8); axA.set_title("A  Content effect (misleading-majority, all-channel)")
rounds = [0, 1, 2, 3]
for ct, name, col in [("evidence_only", "evidence", EVID), ("conclusion_only", "conclusion", CONC)]:
    eb = round_traj("gpt-4.1-mini", "misleading_majority", "all_channel", ct, "e_bar")
    D = round_traj("gpt-4.1-mini", "misleading_majority", "all_channel", ct, "D")
    axB.plot(rounds[:len(eb)], eb, "-o", c=col, label=f"{name}: ē (error)")
    axB.plot(rounds[:len(D)], D, "--s", c=col, alpha=0.6, label=f"{name}: D (diversity)")
axB.set_xlabel("communication round"); axB.set_ylabel("ē  /  D"); axB.set_xticks(rounds)
axB.legend(fontsize=7); axB.set_title("B  Mechanism: individual error rises, diversity persists")
plt.tight_layout(); plt.savefig(FIG / "fig2_content_mechanism.png", dpi=200); plt.close()
print("wrote fig2")

# ---------- FIG 4: causal do-operator (with bootstrap CIs) ----------
import random as _rng
_rng.seed(0)
def _ci(vals, B=5000):
    n = len(vals); m = sum(vals)/n
    bs = sorted(sum(vals[_rng.randrange(n)] for _ in range(n))/n for _ in range(B))
    return m, max(0.0, m-bs[int(.025*B)]), max(0.0, bs[int(.975*B)]-m)
fig, ax = plt.subplots(figsize=(6.4, 4))
modes = [("correct", "plant\ncorrect"), ("none", "no plant\n(control)"), ("wrong", "plant\nwrong")]
for j, (tag, lab) in enumerate([("replay-gpt-4.1-mini", "gpt-4.1-mini"), ("replay-claude-sonnet-4-6", "claude-sonnet-4-6")]):
    p = HERE / "data" / tag / "replay.csv"
    if not p.exists(): continue
    by = defaultdict(list)
    for r in csv.DictReader(open(p)): by[r["mode"]].append(float(r["p_truth"]))
    ys, los, his = [], [], []
    for m, _ in modes:
        mean, lo, hi = _ci(by[m]); ys.append(mean); los.append(lo); his.append(hi)
    xs = [i + (j-0.5)*0.2 for i in range(len(modes))]
    ace = (sum(by["correct"])/len(by["correct"])) - (sum(by["wrong"])/len(by["wrong"]))
    n_seeds = len(by["correct"])
    ax.bar(xs, ys, width=0.2, yerr=[los, his], capsize=4, error_kw=dict(lw=1.1),
           color=[STRUCT, FAM2][j], label=f"{lab}  (ACE +{ace:.2f}, N={n_seeds})")
ax.set_xticks(range(len(modes))); ax.set_xticklabels([m[1] for m in modes])
ax.set_ylabel("terminal P(truth)"); ax.set_ylim(0, 1.05); ax.axhline(0.5, ls=":", c="gray", lw=0.8)
ax.legend(fontsize=9); ax.set_title("A planted conclusion moves the collective\n(private evidence held fixed; bootstrap 95% CIs)")
plt.tight_layout(); plt.savefig(FIG / "fig3_causal.png", dpi=200); plt.close()
print("wrote fig causal (with CIs)")

# ---------- FIG 4: structure (heatmap + degree curve) ----------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
topos = ["baseline", "circle", "chain", "wheel", "hierarchy", "all_channel"]
cols = [("evidence_only", "evidence"), ("both", "both"), ("conclusion_only", "conclusion")]
M = []
for tp in topos:
    row = []
    for ct, _ in cols:
        c = ct if tp != "baseline" else "none"
        rows = terminal_rows("gpt-4.1-mini", "misleading_majority", tp, c)
        row.append(acc_ci(rows)[0] if rows else float("nan"))
    M.append(row)
im = axA.imshow(M, cmap="cividis", vmin=0.3, vmax=1.0, aspect="auto")
axA.set_xticks(range(len(cols))); axA.set_xticklabels([c[1] for c in cols])
axA.set_yticks(range(len(topos))); axA.set_yticklabels(topos)
for i in range(len(topos)):
    for k in range(len(cols)):
        val = M[i][k]
        txtc = "white" if (val != val or val < 0.72) else "black"  # contrast on cividis
        axA.text(k, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txtc)
axA.set_title("A  Topology × content (n=5, misleading)"); fig.colorbar(im, ax=axA, fraction=0.046)
# degree curve (n=11, conclusion)
deg = [("gpt-4.1-mini_n11", "circle", 2.0), ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55),
       ("gpt-4.1-mini_degree", "lattice_k4", 4.0), ("gpt-4.1-mini_degree", "lattice_k6", 6.0),
       ("gpt-4.1-mini_degree", "lattice_k8", 8.0), ("gpt-4.1-mini_n11", "all_channel", 10.0)]
xs, ys, los, his = [], [], [], []
for tag, tp, dval in deg:
    rows = terminal_rows(tag, "misleading_majority", tp, "conclusion_only")
    p, lo, hi = acc_ci(rows)
    xs.append(dval); ys.append(p); los.append(max(0.0, p-lo)); his.append(max(0.0, hi-p))
l1 = axB.errorbar(xs, ys, yerr=[los, his], fmt="-o", capsize=3, color=STRUCT, label="collective accuracy")
axB.set_xlabel("per-node degree (fan-in)"); axB.set_ylabel("terminal collective accuracy", color=STRUCT)
axB.set_ylim(0, 1.05); axB.axhline(0.5, ls=":", c="gray", lw=0.8)
axB.tick_params(axis="y", labelcolor=STRUCT)
# overlay: strong-minority capitulation vs degree (the behavioral mechanism scaling with fan-in)
axB2 = axB.twinx()
cap_ys = [capit_by_cell(tag, "misleading_majority", tp, "conclusion_only") for tag, tp, _ in deg]
l2, = axB2.plot(xs, cap_ys, "-s", color=CONC, alpha=0.9, label="minority capitulation")
axB2.set_ylabel("strong-minority capitulation", color=CONC); axB2.set_ylim(0, 1.05)
axB2.tick_params(axis="y", labelcolor=CONC)
axB.set_title("B  Fan-in scales accuracy loss and capitulation (n=11, conclusion)")
axB.legend(handles=[l1, l2], fontsize=8, loc="center right")
plt.tight_layout(); plt.savefig(FIG / "fig4_structure.png", dpi=200); plt.close()
print("wrote fig4 (with capitulation overlay)")

# ---------- FIG 5: two failure modes (herding vs overload), n=11 ----------
import numpy as np
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
# Panel A: regime x content, all-channel
conds = [("conclusion_only", "conclusion-sharing\n(majority-deference)"),
         ("evidence_only", "evidence-sharing\n(integration overload)")]
regs = [("misleading_majority", "misleading majority", CONC),
        ("concordant", "concordant (control)", STRUCT)]
xx = np.arange(len(conds)); wbar = 0.35
for j, (rg, lab, col) in enumerate(regs):
    ys, los, his = [], [], []
    for ct, _ in conds:
        rows = terminal_rows("gpt-4.1-mini_n11", rg, "all_channel", ct)
        if rows:
            p, lo, hi = acc_ci(rows); ys.append(p); los.append(max(0.0, p-lo)); his.append(max(0.0, hi-p))
        else:
            ys.append(float("nan")); los.append(0); his.append(0)
    axA.bar(xx + (j-0.5)*wbar, ys, wbar, label=lab, color=col, yerr=[los, his], capsize=4, error_kw=dict(lw=1.1))
axA.set_xticks(xx); axA.set_xticklabels([c[1] for c in conds])
axA.set_ylabel("terminal collective accuracy (n=11, all-channel)"); axA.set_ylim(0, 1.05)
axA.axhline(0.5, ls=":", c="gray", lw=0.8); axA.legend(fontsize=8)
axA.set_title("A  Herding is regime-specific; overload is regime-general")
# Panel B: concordant evidence-sharing accuracy vs per-node degree (overload isolation)
odeg = [("gpt-4.1-mini_n11", "circle", 2.0), ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55),
        ("gpt-4.1-mini_overload", "lattice_k4", 4.0), ("gpt-4.1-mini_overload", "lattice_k6", 6.0),
        ("gpt-4.1-mini_overload", "lattice_k8", 8.0), ("gpt-4.1-mini_n11", "all_channel", 10.0)]
xs, ys, los, his = [], [], [], []
for tag, tp, dval in odeg:
    try:
        rows = terminal_rows(tag, "concordant", tp, "evidence_only")
    except FileNotFoundError:
        rows = []
    if not rows:
        continue
    p, lo, hi = acc_ci(rows)
    xs.append(dval); ys.append(p); los.append(max(0.0, p-lo)); his.append(max(0.0, hi-p))
axB.errorbar(xs, ys, yerr=[los, his], fmt="-o", capsize=3, color=STRUCT)
axB.set_xlabel("per-node degree (fan-in)"); axB.set_ylabel("terminal collective accuracy")
axB.set_ylim(0, 1.05); axB.axhline(0.5, ls=":", c="gray", lw=0.8)
axB.set_title("B  Concordant evidence-sharing degrades with fan-in\n(overload is regime-general, no misleading majority)")
plt.tight_layout(); plt.savefig(FIG / "fig5_two_regimes.png", dpi=200); plt.close()
print("wrote fig5")
print("figures ->", FIG)
