"""Normative-recipient counterfactual: do strong-minority agents capitulate MORE than a
confidence-weighted reading of the verdicts they actually received would justify, and does
that excess grow with fan-in?

For each strong-minority agent (its private signal favors the true state) under conclusion
sharing in the misleading-majority regime, we reconstruct the verdicts it received from its
network neighbors (received_from) and compute the NORMATIVE posterior it would hold if it
combined its own private posterior with those neighbors' first-round verdicts by a
logarithmic opinion pool (confidence-weighted, treating the round-0 verdicts as independent).
We then compare:
  normative capitulation : the confidence-weighted posterior favors the FALSE state
  actual capitulation    : the agent's terminal belief favors the FALSE state
The excess (actual - normative) is capitulation beyond what confidence-weighting justifies,
i.e. the behavioral signature of head-counting. We report it by own fan-in.

Fan-in ladder (n=11, conclusion_only, misleading_majority), pooled across two datasets:
  circle (2), ring_shortcuts (~2.5), lattice_k4 (4), lattice_k6 (6), lattice_k8 (8), all_channel (10).
System python3. No new runs.
"""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EPS = 1e-6
SOURCES = [("gpt-4.1-mini_n11", ["circle", "ring_shortcuts", "all_channel"]),
           ("gpt-4.1-mini_degree", ["lattice_k4", "lattice_k6", "lattice_k8"])]
REGIME, CONTENT = "misleading_majority", "conclusion_only"

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def collect(tag, topos):
    b = pd.read_csv(HERE / "data" / tag / "beliefs.csv")
    r = pd.read_csv(HERE / "data" / tag / "rounds.csv")
    m = pd.read_csv(HERE / "data" / tag / "messages.csv")
    b = b[(b.regime == REGIME) & (b.content_mode == CONTENT) & (b.topology.isin(topos))]
    r = r[(r.regime == REGIME) & (r.content_mode == CONTENT) & (r.topology.isin(topos))]
    m = m[(m.regime == REGIME) & (m.content_mode == CONTENT) & (m.topology.isin(topos))]
    truth = r.groupby(["topology", "seed"])["true_state"].first()

    out = []
    for (topo, seed), g in b.groupby(["topology", "seed"]):
        try:
            truth_is_A = (truth.loc[(topo, seed)] == "A")
        except KeyError:
            continue
        r0 = g[g["round"] == 0].set_index("agent")
        if r0.empty:
            continue
        post_truth0 = (r0["belief_pA"] if truth_is_A else 1 - r0["belief_pA"])   # neighbors' round-0 verdicts
        priv_truth = (r0["private_post_a"] if truth_is_A else 1 - r0["private_post_a"])
        term_round = g["round"].max()
        term = g[g["round"] == term_round].set_index("agent")
        post_truth_T = (term["belief_pA"] if truth_is_A else 1 - term["belief_pA"])
        # neighbor sets from round-1 received_from
        nbr = m[(m.seed == seed) & (m.topology == topo) & (m["round"] == 1)].set_index("agent")["received_from"]

        for a in r0.index:
            # strong-minority = this agent's own private signal favors the TRUE state
            if priv_truth.get(a, 0.5) <= 0.5:
                continue
            rf = nbr.get(a, None)
            if not isinstance(rf, str) or not rf.strip():
                continue
            peers = [int(x) for x in rf.split(";") if x != ""]
            peers = [p for p in peers if p in post_truth0.index]
            if not peers:
                continue
            # normative confidence-weighted posterior (log-opinion pool of own + neighbors' verdicts)
            norm_logodds = logit(priv_truth[a]) + float(np.sum(logit(post_truth0.loc[peers].to_numpy())))
            out.append(dict(
                topo=topo, seed=seed, agent=a, fan_in=len(peers),
                normative_capitulate=int(norm_logodds < 0),
                actual_capitulate=int(post_truth_T.get(a, 1.0) < 0.5),
            ))
    return out

if __name__ == "__main__":
    rows = []
    for tag, topos in SOURCES:
        rows += collect(tag, topos)
    d = pd.DataFrame(rows)
    print("Normative-recipient counterfactual (strong-minority agents, conclusion-only, misleading-majority)\n")
    print(f"Strong-minority agent-trials analyzed: {len(d)}  (across {d.seed.nunique()} seeds x topologies)\n")

    norm = d.normative_capitulate.mean()
    act = d.actual_capitulate.mean()
    print(f"  Normative (confidence-weighted) capitulation : {norm:.3f}")
    print(f"  Actual capitulation                          : {act:.3f}")
    print(f"  Excess (actual - normative)                  : {act - norm:.3f}\n")

    print("  By own fan-in:")
    print(f"    {'fan_in':>7}{'n':>6}{'normative':>11}{'actual':>9}{'excess':>9}")
    for fi, g in d.groupby("fan_in"):
        print(f"    {fi:>7}{len(g):>6}{g.normative_capitulate.mean():>11.3f}"
              f"{g.actual_capitulate.mean():>9.3f}{(g.actual_capitulate.mean()-g.normative_capitulate.mean()):>9.3f}")

    # seed-clustered correlation of excess with fan-in (per seed-topology mean), reported simply
    cell = d.groupby(["topo", "fan_in"]).agg(excess=("actual_capitulate", "mean"),
                                             norm=("normative_capitulate", "mean")).reset_index()
    cell["excess"] = cell["excess"] - cell["norm"]
    print(f"\n  Correlation of excess capitulation with fan-in (cell-level): r = {cell['fan_in'].corr(cell['excess']):.2f}")
