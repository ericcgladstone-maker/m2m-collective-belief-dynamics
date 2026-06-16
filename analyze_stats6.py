"""Direct test of the majority-deference mechanism (review: elevate the MEASURED behavioral
mechanism over inferred 'dependence'). In misleading-majority the numerical majority points
WRONG by construction, so 'movement toward the numerical majority' is a clean, signed,
discriminating signature: it should be POSITIVE under conclusion-sharing (agents defer to the
weak majority → error) and ~0 or NEGATIVE under evidence-sharing (the strong minority's
diagnostic readings pull agents toward truth, against the majority).

For gpt-4.1-mini, misleading-majority, all-channel (neighbors = all other agents), per round
transition t→t+1 and per agent i we compute, from beliefs.csv:
  peer_share_A(t)   = fraction of OTHER agents with belief_pA>0.5 at round t
  Δ_i               = belief_pA(i, t+1) − belief_pA(i, t)
  move_to_majority  = Δ_i · sign(peer_share_A − 0.5)   (>0 = moved toward the numerical majority)
And an OLS per condition: Δ_pA ~ (peer_share_A−0.5) [social/vote pull] + (private_post_a−0.5)
[own-evidence pull]. Discriminating contrast = social-pull coefficient, conclusion vs evidence.
Writes stats_report6.md. System python3 (numpy)."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)

CONDS = [("evidence_only", "evidence"), ("both", "both"), ("conclusion_only", "conclusion")]
# (tag, regime, topology, label, approx herding rate) — test across cells of increasing herding
CELLS = [
    ("gpt-4.1-mini", "misleading_majority", "all_channel", "n=5 standard (herds ~36%)"),
    ("gpt-4.1-mini_n11", "misleading_majority", "all_channel", "n=11 (herds ~42%)"),
    ("gpt-4.1-mini_sigsweep", "mm_maj_heavy", "all_channel", "heavy majority (herds ~74%)"),
]

def load(tag, rg, tp):
    R = defaultdict(lambda: defaultdict(dict))
    p = HERE / "data" / tag / "beliefs.csv"
    if not p.exists():
        return R
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp):
                continue
            R[(r["content_mode"], r["seed"])][int(r["round"])][r["agent"]] = (
                float(r["belief_pA"]), float(r["private_post_a"]))
    return R

w("# Majority-deference: signed movement toward the numerical majority (analyze_stats6.py)\n")
w("The numerical majority points WRONG by construction, so move_to_majority>0 = deference to the weak majority.")
w("Tested across cells of increasing herding rate (signal sharpens where herding is more frequent).\n")

def run_cell(tag, rg, tp, celllab):
    R = load(tag, rg, tp)
    if not R:
        w(f"\n## {celllab} [{tag}/{rg}] — NO DATA\n"); return
    w(f"\n## {celllab}  [{tag} / {rg} / {tp}]\n")
    w("| condition | mean move→majority | round 0→1 | social-pull β | own-evidence β | obs |")
    w("|---|---|---|---|---|---|")
    for cm, lab in CONDS:
        moves, moves_r1 = [], []
        Y, Xsoc, Xown = [], [], []
        n_obs = 0
        for (c, s), rounds in R.items():
            if c != cm:
                continue
            rs = sorted(rounds)
            for ti in range(len(rs) - 1):
                t, tn = rs[ti], rs[ti + 1]
                agents = sorted(set(rounds[t]) & set(rounds[tn]))
                if len(agents) < 3:
                    continue
                for a in agents:
                    others = [rounds[t][o][0] for o in agents if o != a]
                    share_A = np.mean([1.0 if b > 0.5 else 0.0 for b in others])
                    d = rounds[tn][a][0] - rounds[t][a][0]
                    m = 1.0 if share_A > 0.5 else (-1.0 if share_A < 0.5 else 0.0)
                    moves.append(d * m)
                    if ti == 0:
                        moves_r1.append(d * m)
                    Y.append(d); Xsoc.append(share_A - 0.5); Xown.append(rounds[t][a][1] - 0.5)
                    n_obs += 1
        if not Y:
            w(f"| {lab} | — | — | — | — | 0 |"); continue
        X = np.column_stack([np.ones(len(Y)), Xsoc, Xown])
        beta, *_ = np.linalg.lstsq(X, np.array(Y), rcond=None)
        w(f"| {lab} | {np.mean(moves):+.4f} | {np.mean(moves_r1):+.4f} | {beta[1]:+.3f} | {beta[2]:+.3f} | {n_obs} |")

for tag, rg, tp, celllab in CELLS:
    run_cell(tag, rg, tp, celllab)

w("\nReading: move→majority POSITIVE under conclusion and ≤0 under evidence would directly demonstrate")
w("majority-deference (agents follow the weak numerical majority) vs evidence-weighting (the strong")
w("minority prevails). A larger social-pull β under conclusion than evidence shows verdict-sharing")
w("shifts weight from private diagnosticity onto the peer vote count. NB own-evidence β is contaminated")
w("by mean-reversion (Δ on a belief-level proxy); the signed move→majority and round 0→1 are cleaner.")
(HERE / "stats_report6.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
