"""Pivotal-agent test of majority-deference. The averaged move-to-majority (analyze_stats6) is
swamped because weak-majority agents already sit at the majority position. The decisive actors are
the HOLDOUT agents whose private belief OPPOSES the numerical majority (the strong-right minority in
misleading-majority). Majority-deference predicts: holdouts capitulate toward the wrong majority MORE
under conclusion-sharing than evidence-sharing, where their diagnostic readings let them stand firm.

Per trial we set the numerical-majority side from round-0 beliefs, label each agent holdout (round-0
belief on the minority side) or majority, and track each holdout's terminal movement toward the
majority side = (terminal − round0) signed so + = capitulation toward the (wrong) majority. System
python3. Writes stats_report7.md."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)

CONDS = [("evidence_only", "evidence"), ("both", "both"), ("conclusion_only", "conclusion")]
CELLS = [
    ("gpt-4.1-mini", "misleading_majority", "all_channel", "n=5 standard"),
    ("gpt-4.1-mini_n11", "misleading_majority", "all_channel", "n=11"),
    ("gpt-4.1-mini_sigsweep", "mm_maj_heavy", "all_channel", "heavy majority"),
]

def load(tag, rg, tp):
    R = defaultdict(lambda: defaultdict(dict)); p = HERE / "data" / tag / "beliefs.csv"
    if not p.exists(): return R
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp): continue
            R[(r["content_mode"], r["seed"])][int(r["round"])][r["agent"]] = float(r["belief_pA"])
    return R

w("# Pivotal holdout-agent capitulation test (analyze_stats7.py)\n")
w("Holdout = agent whose round-0 belief opposes the numerical majority. capitulation = terminal")
w("movement of holdout beliefs TOWARD the (wrong) majority side (+ = caved, − = held/strengthened).\n")

for tag, rg, tp, celllab in CELLS:
    R = load(tag, rg, tp)
    if not R:
        w(f"## {celllab} [{tag}] — NO DATA\n"); continue
    w(f"## {celllab}  [{tag} / {rg}]\n")
    w("| condition | holdout capitulation (term − r0) | holdout share that flipped sides | n holdouts |")
    w("|---|---|---|---|")
    for cm, lab in CONDS:
        caps, flips = [], []
        for (c, s), rounds in R.items():
            if c != cm: continue
            rs = sorted(rounds)
            if len(rs) < 2: continue
            t0, tT = rs[0], rs[-1]
            agents = sorted(set(rounds[t0]) & set(rounds[tT]))
            if len(agents) < 3: continue
            share_A0 = np.mean([1.0 if rounds[t0][a] > 0.5 else 0.0 for a in agents])
            maj_is_A = share_A0 > 0.5            # numerical-majority side at round 0
            for a in agents:
                b0, bT = rounds[t0][a], rounds[tT][a]
                a_side_A = b0 > 0.5
                if a_side_A == maj_is_A:          # majority agent — skip
                    continue
                # holdout: opposes majority. capitulation = movement toward majority side
                cap = (bT - b0) if maj_is_A else (b0 - bT)   # + = moved toward majority(A) / toward majority(B)
                caps.append(cap)
                flips.append(1.0 if (bT > 0.5) == maj_is_A else 0.0)  # crossed to majority side
        if caps:
            w(f"| {lab} | {np.mean(caps):+.3f} | {np.mean(flips):.2f} | {len(caps)} |")
        else:
            w(f"| {lab} | — | — | 0 |")
    w("")

w("Reading: larger positive capitulation and higher flip share under conclusion than evidence is the")
w("direct signature of majority-deference among the pivotal holdouts. If evidence holdouts hold firm")
w("(≈0 or negative) while conclusion holdouts cave (positive), the mechanism is measured, not inferred.")
(HERE / "stats_report7.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
