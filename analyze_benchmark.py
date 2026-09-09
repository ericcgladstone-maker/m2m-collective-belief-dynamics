"""Confidence-weighted benchmark: were the conclusion messages informative enough
to support correct aggregation, so that the failure lies in HOW agents used them
(head-counting) rather than in the information available?

For each misleading-majority, all-channel, conclusion-only trial we take the round-0
verdicts (each agent's stated belief_pA = its verdict and confidence at first broadcast)
and read the SAME verdicts three ways:
  (1) unweighted head-count   : majority of sign(belief_pA - 0.5)        [the naive vote]
  (2) confidence-weighted vote : sign(mean(belief_pA) - 0.5)             [arithmetic pool]
  (3) logarithmic opinion pool : sign(sum logit(belief_pA))             [the paper's primary]
We compare each reading's accuracy against the truth, and against the agents' ACTUAL
terminal collective accuracy (acc_logodds at the final round).

System python3. Reads code/data/<family>/{beliefs,rounds}.csv. No new runs.
"""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
FAMILIES = [("gpt-4.1-mini", "gpt-4.1-mini"),
            ("claude-sonnet-4-6", "claude-sonnet-4-6"),
            ("gemini-2.5-flash", "gemini-2.5-flash")]
CONTENTS = ["evidence_only", "conclusion_only", "both"]
REGIME, TOPO = "misleading_majority", "all_channel"
EPS = 1e-6

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def summarize(family, tag, content):
    bf, rf = HERE / "data" / tag / "beliefs.csv", HERE / "data" / tag / "rounds.csv"
    if not bf.exists():
        return None
    b = pd.read_csv(bf); r = pd.read_csv(rf)
    b = b[(b.regime == REGIME) & (b.content_mode == content) & (b.topology == TOPO)]
    r = r[(r.regime == REGIME) & (r.content_mode == content) & (r.topology == TOPO)]
    if b.empty:
        return None

    rows = []
    for seed, g0 in b[b["round"] == 0].groupby("seed"):
        rr = r[r.seed == seed]
        if rr.empty:
            continue
        truth_is_A = (rr["true_state"].iloc[0] == "A")
        bp = g0["belief_pA"].to_numpy()
        vote_A = np.mean(bp > 0.5) > 0.5            # (1) unweighted head-count
        meanpool_A = bp.mean() > 0.5                # (2) confidence-weighted vote (mean pool)
        logpool_A = logit(bp).sum() > 0             # (3) log-opinion pool
        actual_correct = int(rr.sort_values("round").iloc[-1]["acc_logodds"])
        rows.append(dict(headcount=int(vote_A == truth_is_A),
                         cw_vote=int(meanpool_A == truth_is_A),
                         logpool=int(logpool_A == truth_is_A),
                         actual=actual_correct))
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    return dict(family=family, content=content, n=len(d),
                headcount=d.headcount.mean(), cw_vote=d.cw_vote.mean(),
                logpool=d.logpool.mean(), actual=d.actual.mean())

if __name__ == "__main__":
    print("Confidence-weighted benchmark (misleading-majority, all-channel)")
    print("Same round-0 verdicts read three ways, vs the agents' actual terminal accuracy.\n")
    hdr = f"{'family':<20}{'content':<16}{'n':>4}{'headcount':>11}{'conf-wtd':>10}{'logpool':>9}{'actual':>9}{'gap':>7}"
    print(hdr); print("-" * len(hdr))
    for fam, tag in FAMILIES:
        for content in CONTENTS:
            o = summarize(fam, tag, content)
            if o is None:
                continue
            gap = o["cw_vote"] - o["actual"]   # achievable (confidence-weighted) minus actual
            print(f"{o['family']:<20}{o['content']:<16}{o['n']:>4}"
                  f"{o['headcount']:>11.2f}{o['cw_vote']:>10.2f}{o['logpool']:>9.2f}"
                  f"{o['actual']:>9.2f}{gap:>7.2f}")
        print()
