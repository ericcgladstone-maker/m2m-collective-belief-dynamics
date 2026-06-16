"""Role-transition mechanism, ASSIGNMENT-BASED & TRUTH-ANCHORED (review round 4 — the version that
becomes the manuscript centerpiece). Roles are defined from the ASSIGNED signal, not realized
round-0 beliefs (avoids post-measurement selection):
  strong-minority = agent whose private signal points to the TRUE state (sign(llr) == truth)
  weak-majority   = agent whose private signal points to the FALSE state
Transitions are anchored to TRUTH:
  majority correction  = a weak (wrong-signal) agent ENDS on the true side
  minority capitulation = a strong (truth-signal) agent ENDS on the false side
true_state is joined from rounds.csv. Inference = seed-clustered PAIRED bootstrap (unit = seed).
Also reports agreement with the round-0-belief classification (robustness) and a bounded mediation
check. System python3. Writes stats_report9.md."""
import csv, random
from collections import defaultdict
from pathlib import Path
import numpy as np
random.seed(0)
HERE = Path(__file__).parent
OUT = []
def w(s=""): OUT.append(s)

CELLS = [
    ("gpt-4.1-mini", "misleading_majority", "all_channel", "n=5 standard"),
    ("gpt-4.1-mini_sigsweep", "mm_maj_heavy", "all_channel", "heavy majority"),
    ("gpt-4.1-mini_n11", "misleading_majority", "all_channel", "n=11 (overload ref)"),
]

def true_state(tag, rg, tp):
    T = {}; p = HERE / "data" / tag / "rounds.csv"
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp): continue
            T[(r["content_mode"], r["seed"])] = r["true_state"]
    return T

def acc_map(tag, rg, tp):
    best = {}; p = HERE / "data" / tag / "rounds.csv"
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp): continue
            k = (r["content_mode"], r["seed"])
            if k not in best or int(r["round"]) > best[k][0]:
                best[k] = (int(r["round"]), int(r["acc_logodds"]))
    return {k: v[1] for k, v in best.items()}

def load(tag, rg, tp):
    # B[(content,seed)][round][agent] = (belief_pA, llr)
    B = defaultdict(lambda: defaultdict(dict))
    with open(HERE / "data" / tag / "beliefs.csv") as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp): continue
            B[(r["content_mode"], r["seed"])][int(r["round"])][r["agent"]] = (
                float(r["belief_pA"]), float(r["llr"]))
    return B

def per_seed(B, T, cm):
    """seed -> (maj_correct_prop, min_capit_prop, role_agreement_with_belief0)."""
    out = {}
    for (c, s), rounds in B.items():
        if c != cm or (c, s) not in T: continue
        rs = sorted(rounds)
        if len(rs) < 2: continue
        t0, tT = rs[0], rs[-1]
        truth_A = (T[(c, s)] == "A")
        agents = sorted(set(rounds[t0]) & set(rounds[tT]))
        if len(agents) < 3: continue
        corr, capit, agree, n_agree = [], [], 0, 0
        for a in agents:
            b0, llr = rounds[t0][a]
            bT = rounds[tT][a][0]
            sig_truth = (llr > 0) == truth_A          # assigned signal points to truth?
            endsT_true = (bT > 0.5) == truth_A
            if sig_truth:                              # strong-minority (truth signal)
                capit.append(0.0 if endsT_true else 1.0)
            else:                                      # weak-majority (wrong signal)
                corr.append(1.0 if endsT_true else 0.0)
            # robustness: does assigned role match round-0-belief role?
            belief0_truth = (b0 > 0.5) == truth_A
            n_agree += 1; agree += 1 if (sig_truth == belief0_truth) else 0
        out[s] = (np.mean(corr) if corr else np.nan,
                  np.mean(capit) if capit else np.nan,
                  agree / n_agree if n_agree else np.nan)
    return out

def boot(ev, co, idx, B=10000):
    seeds = sorted(set(ev) & set(co))
    if not seeds: return None
    rate = lambda d, s: d[s][idx]
    base = np.nanmean([rate(co, s) for s in seeds]) - np.nanmean([rate(ev, s) for s in seeds])
    diffs = []
    n = len(seeds)
    for _ in range(B):
        samp = [seeds[random.randrange(n)] for _ in range(n)]
        diffs.append(np.nanmean([rate(co, s) for s in samp]) - np.nanmean([rate(ev, s) for s in samp]))
    diffs.sort()
    return base, diffs[int(.025*B)], diffs[int(.975*B)], len(seeds)

w("# Assignment-based, truth-anchored role transitions (analyze_stats9.py)\n")
w("Roles from ASSIGNED signal (sign(llr) vs true_state); transitions anchored to truth. "
  "Inference = seed-clustered paired bootstrap (unit=seed), 10k resamples.\n")

for tag, rg, tp, lab in CELLS:
    B = load(tag, rg, tp); T = true_state(tag, rg, tp); ACC = acc_map(tag, rg, tp)
    w(f"## {lab}  [{tag} / {rg}]\n")
    w("| condition | weak-majority correction | strong-minority capitulation | role↔belief0 agreement | seeds |")
    w("|---|---|---|---|---|")
    role = {}
    for cm, clab in [("evidence_only","evidence"),("both","both"),("conclusion_only","conclusion")]:
        d = per_seed(B, T, cm); role[cm] = d
        if not d: w(f"| {clab} | — | — | — | 0 |"); continue
        cor = np.nanmean([v[0] for v in d.values()])
        cap = np.nanmean([v[1] for v in d.values()])
        agr = np.nanmean([v[2] for v in d.values()])
        w(f"| {clab} | {cor:.2f} | {cap:.2f} | {agr:.2f} | {len(d)} |")
    ev, co = role.get("evidence_only", {}), role.get("conclusion_only", {})
    rc = boot(ev, co, 0); rk = boot(ev, co, 1)
    if rc: w(f"\n- **weak-majority correction** RD (conclusion−evidence) = {rc[0]:+.2f} [95% CI {rc[1]:+.2f}, {rc[2]:+.2f}], paired seeds={rc[3]}")
    if rk: w(f"- **strong-minority capitulation** RD (conclusion−evidence) = {rk[0]:+.2f} [95% CI {rk[1]:+.2f}, {rk[2]:+.2f}], paired seeds={rk[3]}")
    # bounded mediation
    if ACC:
        Y, C, CR, CP = [], [], [], []
        for cm, code in [("evidence_only",0),("conclusion_only",1)]:
            for s, v in role.get(cm, {}).items():
                if (cm, s) not in ACC: continue
                Y.append(ACC[(cm, s)]); C.append(code)
                CR.append(0.0 if np.isnan(v[0]) else v[0]); CP.append(0.0 if np.isnan(v[1]) else v[1])
        if len(set(C)) == 2 and len(Y) > 20:
            Y = np.array(Y, float)
            ols = lambda cols: np.linalg.lstsq(np.column_stack([np.ones(len(Y))]+cols), Y, rcond=None)[0]
            b1 = ols([np.array(C,float)])[1]
            b2 = ols([np.array(C,float), np.array(CR), np.array(CP)])[1]
            w(f"- mediation (descriptive): content→accuracy = {b1:+.2f} alone → {b2:+.2f} with role transitions added")
    w("")

w("Reading: evidence enables weak-majority correction AND limits minority capitulation; verdicts")
w("suppress correction and increase capitulation. role↔belief0 agreement near 1.0 means assigned-signal")
w("and round-0-belief classifications coincide (robustness). n=11 = overload reference (content washes out).")
(HERE / "stats_report9.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
