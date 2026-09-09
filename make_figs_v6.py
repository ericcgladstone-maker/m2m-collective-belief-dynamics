"""Regenerate the two manuscript figures that change in v6.

Fig. 2 gains panel C: an empirical cumulative distribution of terminal pooled probability
assigned to the true state, by communication content. An ECDF is used deliberately rather
than a fitted density so that no modality is imposed on the data (consensus §8/§11).

Fig. 3 loses panel B (n=11 role transitions) to the Supplementary Materials and becomes a
single-panel figure; the relocated panel is written separately as figS2 for the SI.

Reuses the generators' Wong palette and rcParams. System python3, no new runs.
"""
import csv, math, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
FIG = HERE.parent / "figures"

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "tight",
})
EVID, CONC, STRUCT, FAM2 = "#009E73", "#D55E00", "#0072B2", "#E69F00"


def rows(tag, regime, topo, content, rnd=None):
    """rnd=None -> the terminal round available for each seed (matches make_figures.terminal_rows,
    which matters for the no-communication baseline, where only round 0 exists)."""
    best = {}
    f = HERE / "data" / tag / "rounds.csv"
    for r in csv.DictReader(open(f)):
        if (r["regime"], r["topology"], r["content_mode"]) != (regime, topo, content):
            continue
        if rnd is not None:
            if int(r["round"]) == rnd: best[r["seed"]] = r
        else:
            s = r["seed"]
            if s not in best or int(r["round"]) > int(best[s]["round"]): best[s] = r
    return list(best.values())


def acc_ci(rs, z=1.96):
    n = len(rs); k = sum(int(r["acc_logodds"]) for r in rs)
    if not n: return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, c-h, c+h


def traj(tag, regime, topo, content, field):
    out = []
    for rd in range(4):
        rs = rows(tag, regime, topo, content, rd)
        if rs: out.append(sum(float(r[field]) for r in rs) / len(rs))
    return out


# ================================ FIG 2 (three panels) ================================
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4))
conts = [("baseline", "none", "baseline"), ("all_channel", "evidence_only", "evidence"),
         ("all_channel", "both", "both"), ("all_channel", "conclusion_only", "conclusion")]
for j, (tag, lab, col) in enumerate([("gpt-4.1-mini", "gpt-4.1-mini", STRUCT),
                                     ("claude-sonnet-4-6", "claude-sonnet-4-6", FAM2)]):
    xs, ys, los, his = [], [], [], []
    for i, (tp, ct, _) in enumerate(conts):
        rs = rows(tag, "misleading_majority", tp, ct)
        if not rs: continue
        p, lo, hi = acc_ci(rs)
        xs.append(i + (j-0.5)*0.18); ys.append(p)
        los.append(max(0.0, p-lo)); his.append(max(0.0, hi-p))
    axA.bar(xs, ys, width=0.18, label=lab, yerr=[los, his], capsize=3, color=col)
axA.set_xticks(range(len(conts))); axA.set_xticklabels([c[2] for c in conts])
axA.set_ylabel("terminal collective accuracy"); axA.set_ylim(0, 1.05)
axA.axhline(0.5, ls=":", c="gray", lw=0.8); axA.legend(fontsize=8)
axA.set_title("A  Content effect (misleading-majority, all-channel)")

rds = [0, 1, 2, 3]
for ct, name, col in [("evidence_only", "evidence sharing", EVID),
                      ("conclusion_only", "conclusion sharing", CONC)]:
    eb = traj("gpt-4.1-mini", "misleading_majority", "all_channel", ct, "e_bar")
    D = traj("gpt-4.1-mini", "misleading_majority", "all_channel", ct, "D")
    axB.plot(rds[:len(eb)], eb, "-o", c=col, label=f"{name}: ē (error)")
    axB.plot(rds[:len(D)], D, "--s", c=col, alpha=0.6, label=f"{name}: D (diversity)")
axB.set_xlabel("communication round"); axB.set_ylabel("ē  /  D"); axB.set_xticks(rds)
axB.legend(fontsize=7); axB.set_title("B  Individual error rises as diversity widens")

for ct, name, col in [("evidence_only", "evidence sharing", EVID),
                      ("conclusion_only", "conclusion sharing", CONC)]:
    v = sorted(float(r["p_truth_logodds"]) for r in
               rows("gpt-4.1-mini", "misleading_majority", "all_channel", ct))
    n = len(v)
    axC.step([0] + v + [1], [0] + [(i+1)/n for i in range(n)] + [1], where="post",
             c=col, lw=1.8, label=f"{name} (N = {n})")
    axC.plot(v, [(i+1)/n for i in range(n)], ".", c=col, ms=2.5, alpha=0.45)
axC.axvline(0.5, ls="--", c="gray", lw=0.9)
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
axC.set_xlabel("terminal pooled P(true state)"); axC.set_ylabel("cumulative fraction of trials")
axC.legend(fontsize=7.5, loc="upper left")
axC.set_title("C  Terminal outcome distribution")
plt.tight_layout(); plt.savefig(FIG / "fig2_content_diversity.png", dpi=600); plt.close()
print("wrote fig2_content_diversity.png (3 panels)")

# ============================ FIG 3 (single panel) + SI figS2 ============================
# reuse make_fig_mechanism's validated panel()/estimators WITHOUT running its module-level
# plotting block (importing it normally would regenerate the old two-panel Fig. 3).
_src = (HERE / "make_fig_mechanism.py").read_text()
_src = _src[:_src.index("fig, (axA, axB) = plt.subplots")]
M = type(sys)("mech"); M.__dict__["__file__"] = str(HERE / "make_fig_mechanism.py")
exec(compile(_src, "make_fig_mechanism.py", "exec"), M.__dict__)
for name, tag, out, title in [
    ("fig3", "gpt-4.1-mini", FIG / "fig3_role_transitions.png",
     "Role transitions by communication content (n = 5)"),
    ("figS2", "gpt-4.1-mini_n11", FIG / "figS2_role_transitions_n11.png",
     "Role transitions at extreme fan-in (n = 11)")]:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    M.panel(ax, tag, "misleading_majority", "all_channel", title)
    plt.tight_layout(); plt.savefig(out, dpi=600); plt.close()
    print("wrote", out.name)
