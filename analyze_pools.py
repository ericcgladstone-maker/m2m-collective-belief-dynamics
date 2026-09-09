"""Aggregation-rule appendix (PNAS Nexus R&R, Reviewer 1 concern 1).

Extends the aggregation-rule comparison from the single gpt-4.1-mini all-channel cell
to all model families and the full degree ladder, and adds extreme-belief diagnostics.

Rules compared (all applied to the SAME terminal beliefs):
  log        log opinion pool, sigmoid(mean logit)                  [paper's primary]
  arith      arithmetic mean of probabilities
  major      unweighted majority head-count of sign(p - 0.5)
  median     median belief                                          [robust reference]
  clip05     log pool with beliefs clipped to [0.05, 0.95]          [extreme-belief check]
  dropmax    log pool with the largest-|logit| agent removed        [influence check]

Note on "confidence-weighted" pooling: a linear confidence-weighted vote, with weight
|2p-1| on vote sign(p-0.5), sums to 2n(mean(p) - 0.5). Its sign is therefore identical
to the arithmetic pool's, so it is not a distinct classification rule and is reported as
such rather than duplicated. The median pool is used as the distinct robust reference.

System python3. Reads code/data/<tag>/{beliefs,rounds}.csv. No new runs.
"""
import csv, math, collections, statistics as st
from pathlib import Path

HERE = Path(__file__).parent
TERMINAL = 3
RULES = ["log", "arith", "major", "median", "clip05", "dropmax"]


def logit(p, eps=1e-6):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def classify(bel, rule):
    """Return True if the rule puts the collective on state A."""
    if rule == "log":
        return sum(logit(b) for b in bel) / len(bel) >= 0
    if rule == "arith":
        return sum(bel) / len(bel) >= 0.5
    if rule == "major":
        return sum(1 for b in bel if b >= 0.5) / len(bel) >= 0.5
    if rule == "median":
        return st.median(bel) >= 0.5
    if rule == "clip05":
        c = [min(0.95, max(0.05, b)) for b in bel]
        return sum(logit(b) for b in c) / len(c) >= 0
    if rule == "dropmax":
        L = [logit(b) for b in bel]
        j = max(range(len(L)), key=lambda i: abs(L[i]))
        Lo = [x for i, x in enumerate(L) if i != j]
        return sum(Lo) / len(Lo) >= 0
    raise ValueError(rule)


def trials(tag, regime, topology, content, rnd=TERMINAL):
    """seed -> (beliefs list, true_state)"""
    bel, truth = collections.defaultdict(list), {}
    bf = HERE / "data" / tag / "beliefs.csv"
    rf = HERE / "data" / tag / "rounds.csv"
    if not bf.exists():
        return {}
    for r in csv.DictReader(open(bf)):
        if r["regime"] != regime or r["topology"] != topology: continue
        if r["content_mode"] != content or int(r["round"]) != rnd: continue
        bel[r["seed"]].append(float(r["belief_pA"]))
    for r in csv.DictReader(open(rf)):
        if r["regime"] != regime or r["topology"] != topology: continue
        if r["content_mode"] != content: continue
        truth[r["seed"]] = r["true_state"]
    return {s: (b, truth[s]) for s, b in bel.items() if s in truth and len(b) >= 3}


def row(tag, regime, topology, content, label=None):
    t = trials(tag, regime, topology, content)
    if not t:
        return None
    acc = {}
    for rule in RULES:
        acc[rule] = st.mean(int(classify(b, rule) == (ts == "A")) for b, ts in t.values())
    return dict(label=label or f"{topology}/{content}", n=len(t), **acc)


def table(title, rows):
    print(f"\n{'='*112}\n{title}\n{'='*112}")
    print(f"  {'cell':<46}{'n':>5}" + "".join(f"{r:>10}" for r in RULES))
    for r in rows:
        if r is None: continue
        print(f"  {r['label']:<46}{r['n']:>5}" + "".join(f"{r[k]:>10.2f}" for k in RULES))


if __name__ == "__main__":
    # --- 1. Content contrast across all three families, focal all-channel cell ----------
    rows = []
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6", "gemini-2.5-flash"):
        for content in ("evidence_only", "conclusion_only", "both"):
            rows.append(row(tag, "misleading_majority", "all_channel", content,
                            f"{tag} / {content}"))
    table("A. Content contrast by model family (misleading-majority, all-channel, terminal)", rows)

    print("\n  Content effect (evidence - conclusion) by family and rule:")
    print(f"  {'family':<24}" + "".join(f"{r:>10}" for r in RULES))
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6", "gemini-2.5-flash"):
        e = row(tag, "misleading_majority", "all_channel", "evidence_only")
        c = row(tag, "misleading_majority", "all_channel", "conclusion_only")
        if e and c:
            print(f"  {tag:<24}" + "".join(f"{e[k]-c[k]:>+10.2f}" for k in RULES))

    # --- 2. Topology sweep, main family --------------------------------------------------
    rows = []
    for top in ("all_channel", "wheel", "circle", "chain", "hierarchy"):
        for content in ("evidence_only", "conclusion_only"):
            rows.append(row("gpt-4.1-mini", "misleading_majority", top, content,
                            f"{top} / {content}"))
    table("B. Topology sweep (gpt-4.1-mini, misleading-majority, n=5, terminal)", rows)

    # --- 3. Degree ladder, n=11 ----------------------------------------------------------
    LADDER = [("gpt-4.1-mini_n11", "circle", 2.0), ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55),
              ("gpt-4.1-mini_overload", "lattice_k4", 4.0), ("gpt-4.1-mini_overload", "lattice_k6", 6.0),
              ("gpt-4.1-mini_overload", "lattice_k8", 8.0), ("gpt-4.1-mini_n11", "all_channel", 10.0)]
    rows = []
    for tag, top, deg in LADDER:
        rows.append(row(tag, "misleading_majority", top, "conclusion_only",
                        f"degree {deg:<5} {top} / conclusion"))
    table("C. Degree ladder, failure 1 (misleading-majority, conclusion sharing, n=11)", rows)

    rows = []
    for tag, top, deg in LADDER:
        rows.append(row(tag, "concordant", top, "evidence_only",
                        f"degree {deg:<5} {top} / evidence"))
    table("D. Degree ladder, failure 2 (concordant, evidence sharing, n=11)", rows)

    rows = []
    for tag, top, deg in LADDER:
        rows.append(row(tag, "misleading_majority", top, "evidence_only",
                        f"degree {deg:<5} {top} / evidence"))
    table("E. Degree ladder, misleading-majority evidence sharing (n=11)", rows)
