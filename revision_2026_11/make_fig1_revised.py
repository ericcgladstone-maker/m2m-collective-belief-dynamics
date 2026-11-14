"""Revised Figure 1 (two-panel schematic) for the minor-revision build. Labels are taken
verbatim from the edit specification. Writes vector masters (PDF, SVG), a 600 dpi TIFF,
and a 300 dpi PNG.

Run:  python3 make_fig1_revised.py [output directory, default revision_2026_11/figures]"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from pathlib import Path
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
GRAY, GRAYE = "#a6a6a6", "#4d4d4d"
TEAL, TEALE = "#1b9e77", "#0f5c45"
INK = "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
W, H = 7.0, 3.1
fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
gut = 0.30; pw = (W - 3 * gut) / 2          # two equal panels with explicit gutter
xa, xb = gut, 2 * gut + pw
def box(x, y, w, h, ec, fc="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", ec=ec, fc=fc, lw=lw))
# ---- Panel A
ax.text(xa, H - 0.22, "A  The misleading-majority task", fontsize=9.5, fontweight="bold", va="top", color=INK)
ax.text(xa + pw / 2, H - 0.62, "True state in this example: A", fontsize=8, ha="center", va="top", color=INK)
r = 0.17; ym = H - 1.25
xs_left = [xa + 0.35 + i * 0.52 for i in range(3)]
xs_right = [xa + pw - 0.35 - 0.52, xa + pw - 0.35]
for x in xs_left: ax.add_patch(Circle((x, ym), r, fc=GRAY, ec=GRAYE, lw=0.9))
for x in xs_right: ax.add_patch(Circle((x, ym), r, fc=TEAL, ec=TEALE, lw=0.9))
cl = sum(xs_left) / 3; cr = sum(xs_right) / 2
ax.text(cl, ym - 0.30, "Three weak readings\nfavor B", fontsize=7.5, ha="center", va="top", color=INK)
ax.text(cr, ym - 0.30, "Two stronger readings\nfavor A", fontsize=7.5, ha="center", va="top", color=INK)
bw, bh, by = 1.30, 0.50, 0.30
bxl = cl - bw / 2; bxr = cr - bw / 2
box(bxl, by, bw, bh, GRAYE, "#f2f2f2"); box(bxr, by, bw, bh, TEALE, "#e3f2ed")
ax.text(cl, by + bh / 2, "Count the readings\n→ B (incorrect)", fontsize=7.5, ha="center", va="center", color=INK, linespacing=1.3)
ax.text(cr, by + bh / 2, "Weigh the evidence\n→ A (correct)", fontsize=7.5, linespacing=1.3, ha="center", va="center", color=INK)
for x0, x1, c in ((cl, cl, GRAYE), (cr, cr, TEALE)):
    ax.add_patch(FancyArrowPatch((x0, ym - 0.68), (x1, by + bh + 0.04), arrowstyle="-|>", mutation_scale=9, color=c, lw=1.0))
# ---- Panel B
ax.text(xb, H - 0.22, "B  What agents are asked to share", fontsize=9.5, fontweight="bold", va="top", color=INK)
mw = pw - 0.1; mh = 0.78
for i, (title, body, ec, fc) in enumerate((
        ("Evidence sharing", "Reading r3 occurs with probability 0.88\nunder A and 0.12 under B.", TEALE, "#e3f2ed"),
        ("Conclusion sharing", "My best guess is A, with about 94% confidence.", GRAYE, "#f2f2f2"))):
    y = H - 1.45 - i * 1.00
    box(xb + 0.05, y, mw, mh, ec, fc)
    ax.text(xb + 0.18, y + mh - 0.10, title, fontsize=8, fontweight="bold", va="top", color=INK)
    ax.text(xb + 0.18, y + mh - 0.34, body, fontsize=7.5, va="top", color=INK)
ax.text(xb + pw / 2, 0.30, "Same private readings and network in both conditions", fontsize=7.5, ha="center", va="center", style="italic", color=INK)
for ext in ("pdf", "svg"):
    fig.savefig(OUT / f"Figure_1.{ext}", facecolor="white")
fig.savefig(OUT / "Figure_1.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "Figure_1.png", dpi=300, facecolor="white")
print("written to", OUT)
