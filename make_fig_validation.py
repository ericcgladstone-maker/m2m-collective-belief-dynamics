"""SI Fig. S1 — belief read-out validation: round-0 (pre-communication) elicited belief vs the
single-signal Bayesian posterior, per model family. Near-identity (r≈0.999) shows agents compute
normative beliefs, so downstream failures reflect communication, not an inability to reason.
System python3. Writes ../figures/figS1_validation.png."""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
HERE = Path(__file__).parent; FIG = HERE.parent / "figures"
FAMS = [("gpt-4.1-mini", "#0072B2"), ("claude-sonnet-4-6", "#E69F00"), ("gemini-2.5-flash", "#009E73")]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (tag, col) in zip(axes, FAMS):
    p = HERE / "data" / tag / "beliefs.csv"
    xs, ys = [], []
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                if int(r["round"]) == 0:
                    try:
                        xs.append(float(r["private_post_a"])); ys.append(float(r["belief_pA"]))
                    except (KeyError, ValueError):
                        pass
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    if xs:
        ax.scatter(xs, ys, s=6, alpha=0.15, color=col, edgecolors="none")
        x = np.array(xs); y = np.array(ys)
        r = np.corrcoef(x, y)[0, 1]; mae = np.mean(np.abs(x - y))
        ax.text(0.05, 0.93, f"r = {r:.3f}\nMAE = {mae:.3f}\nN = {len(xs):,}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(tag, fontsize=10)
    ax.set_xlabel("single-signal Bayesian posterior P(A)")
axes[0].set_ylabel("elicited round-0 belief P(A)")
fig.suptitle("Belief read-out recovers the Bayesian posterior before communication", fontsize=11.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG / "figS1_validation.png", dpi=200, bbox_inches="tight")
plt.close()
print("wrote figS1_validation.png")
