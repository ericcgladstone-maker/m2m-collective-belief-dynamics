"""Fig 1 — experimental design schematic (programmatic, publication-draft quality). Four panels:
(A) the misleading-majority task and the vote-vs-weigh crux; (B) the four content conditions;
(C) the network topologies / degree ladder; (D) the planting intervention. Colorblind-safe Wong
palette. System python3. Writes ../figures/fig1_design.png. (Intended as a clean, accurate draft
that an illustrator can polish; it already encodes the correct framing.)"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np
matplotlib.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
EVID = "#009E73"; CONC = "#D55E00"; STRUCT = "#0072B2"; GREY = "#999999"
HERE = Path(__file__).parent; FIG = HERE.parent / "figures"

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax in axes.flat:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
aA, aB, aC, aD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

def panel_title(ax, s):
    ax.text(0.0, 1.0, s, fontsize=12.5, fontweight="bold", va="top", ha="left", transform=ax.transAxes)

def rbox(ax, x, y, w, h, text, fc, ec, tc="black", fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight)

# ---------- A: task + misleading-majority crux ----------
panel_title(aA, "A   The task: a misleading majority")
aA.text(0.5, 0.86, "Hidden state: A or B  (50/50 prior)", ha="center", fontsize=10, style="italic")
xs = np.linspace(0.10, 0.90, 5); y = 0.62
roles = [("weak→B", GREY, "↓"), ("weak→B", GREY, "↓"), ("weak→B", GREY, "↓"),
         ("strong→A", EVID, "↑"), ("strong→A", EVID, "↑")]
for x, (lab, col, arr) in zip(xs, roles):
    aA.add_patch(Circle((x, y), 0.052, fc=col, ec="black", lw=1.1, alpha=0.9))
    aA.text(x, y, arr, ha="center", va="center", fontsize=13, color="white", weight="bold")
    aA.text(x, y - 0.10, lab, ha="center", fontsize=7.5, color=col if col != GREY else "black")
aA.text(0.5, 0.74, "3 weak readings → wrong state B        2 strong readings → true state A",
        ha="center", fontsize=8.2)
rbox(aA, 0.06, 0.16, 0.40, 0.16, "Count votes\n→ B  (3 vs 2)\nWRONG", "#fbe4da", CONC, CONC, 9.5, "bold")
rbox(aA, 0.54, 0.16, 0.40, 0.16, "Weigh evidence\n→ A\nRIGHT", "#d8efe8", EVID, EVID, 9.5, "bold")
aA.annotate("", xy=(0.26, 0.33), xytext=(0.30, 0.50), arrowprops=dict(arrowstyle="->", color=CONC, lw=1.6))
aA.annotate("", xy=(0.74, 0.33), xytext=(0.70, 0.50), arrowprops=dict(arrowstyle="->", color=EVID, lw=1.6))

# ---------- B: content conditions ----------
panel_title(aB, "B   What each agent transmits (content)")
rows = [
    ("none (baseline)", "— (no message)", GREY, ""),
    ("evidence", "\"My reading occurs with prob. 0.59\nunder A, 0.41 under B.\"", EVID, "carries diagnosticity → re-weightable"),
    ("conclusion", "\"My best guess is A, ~80% confident.\"", CONC, "verdict only → provenance hidden"),
    ("both", "reading + diagnosticity  +  verdict", STRUCT, ""),
]
yb = 0.78
for name, msg, col, tag in rows:
    aB.text(0.04, yb + 0.052, name, ha="left", fontsize=8.8, color=col, weight="bold")
    rbox(aB, 0.04, yb - 0.085, 0.92, 0.115, msg, "white", col, "black", 8.3)
    if tag:
        aB.text(0.94, yb + 0.052, tag, ha="right", fontsize=7.2, color=col, style="italic")
    yb -= 0.205

# ---------- C: topologies / degree ladder ----------
panel_title(aC, "C   Network topologies (per-node fan-in)")
def draw_graph(ax, cx, cy, r, edges, npts, hub=False, label="", deg=""):
    if hub:
        pos = [(cx, cy)] + [(cx + r*np.cos(t), cy + r*np.sin(t))
                            for t in np.linspace(0, 2*np.pi, npts-1, endpoint=False)]
    else:
        pos = [(cx + r*np.cos(t), cy + r*np.sin(t))
               for t in np.linspace(np.pi/2, np.pi/2 + 2*np.pi, npts, endpoint=False)]
    for i, j in edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], "-", color=GREY, lw=0.8, zorder=1)
    for (px, py) in pos:
        ax.add_patch(Circle((px, py), 0.018, fc=STRUCT, ec="black", lw=0.6, zorder=2))
    ax.text(cx, cy - r - 0.06, label, ha="center", fontsize=8, weight="bold")
    ax.text(cx, cy - r - 0.105, deg, ha="center", fontsize=7, color=GREY)

n = 6
ring = [(i, (i+1) % n) for i in range(n)]
draw_graph(aC, 0.16, 0.62, 0.11, ring, n, label="circle", deg="deg 2")
wheel = [(0, k) for k in range(1, n)]
draw_graph(aC, 0.5, 0.62, 0.11, wheel, n, hub=True, label="wheel", deg="hub deg n−1")
allc = [(i, j) for i in range(n) for j in range(i+1, n)]
draw_graph(aC, 0.84, 0.62, 0.11, allc, n, label="all-channel", deg="deg n−1")
# degree ladder annotation
aC.text(0.5, 0.30, "Degree ladder (n = 11):  circle (2) → ring+shortcuts (≈2.5)\n→ lattice 4 / 6 / 8 → all-channel (10)",
        ha="center", fontsize=8.5)
aC.annotate("", xy=(0.86, 0.16), xytext=(0.14, 0.16), arrowprops=dict(arrowstyle="->", color=STRUCT, lw=2))
aC.text(0.5, 0.10, "increasing per-node fan-in →", ha="center", fontsize=8.5, color=STRUCT, weight="bold")

# ---------- D: planting intervention ----------
panel_title(aD, "D   Planting intervention (causal test)")
cx, cy, r = 0.30, 0.60, 0.13
pos = [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 5, endpoint=False)]
for i in range(5):
    for j in range(i+1, 5):
        aD.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], "-", color=GREY, lw=0.7)
for k, (px, py) in enumerate(pos):
    col = CONC if k == 0 else STRUCT
    aD.add_patch(Circle((px, py), 0.026, fc=col, ec="black", lw=0.9, zorder=3))
aD.text(pos[0][0], pos[0][1] + 0.10, "planted first-mover\n\"state is X (confident)\"",
        ha="center", fontsize=7.6, color=CONC, weight="bold")
aD.text(0.30, 0.40, "private evidence held fixed", ha="center", fontsize=8, style="italic")
rbox(aD, 0.60, 0.62, 0.36, 0.12, "do(plant correct)\n→ collective → truth", "#d8efe8", EVID, "black", 8.3)
rbox(aD, 0.60, 0.40, 0.36, 0.12, "do(plant wrong)\n→ collective → error", "#fbe4da", CONC, "black", 8.3)
aD.annotate("", xy=(0.60, 0.68), xytext=(0.44, 0.63), arrowprops=dict(arrowstyle="->", color=EVID, lw=1.5))
aD.annotate("", xy=(0.60, 0.46), xytext=(0.44, 0.57), arrowprops=dict(arrowstyle="->", color=CONC, lw=1.5))
aD.text(0.5, 0.22, "ACE = E[P(truth) | do(correct)] − E[P(truth) | do(wrong)]",
        ha="center", fontsize=8.2, weight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(FIG / "fig1_design.png", dpi=200)
plt.close()
print("wrote fig1_design.png")
