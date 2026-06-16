"""Role-based mechanism analysis with trial-clustered inference (review round 3).

Fixes pseudo-replication (holdouts are nested within seed-paired trials, not independent) and
broadens the mechanism beyond 'holdouts hold firm' to the full four-quantity role decomposition
GPT recommended. In misleading-majority the numerical majority is WRONG by construction, so the
truth side = the side OPPOSING the round-0 numerical majority (no true_state join needed).

Per trial, agents are split by their round-0 belief side:
  HOLDOUT  = on the truth (minority) side          MAJORITY = on the wrong-majority side
Four role quantities per content condition:
  (1) holdout capitulation rate  = P(holdout ends on the majority side)        [herding]
  (2) majority correction rate   = P(majority agent ends on the truth side)    [persuasion]
  (3) mean |Δbelief| holdouts    (4) mean |Δbelief| majority                    [movement]
Inference: seed-clustered PAIRED bootstrap (resample seeds present in BOTH cells; unit = seed) for
the conclusion−evidence risk difference in (1) and (2). Plus a trial-level mediation check: does
adding holdout-flip + majority-correction attenuate the content effect on collective accuracy?
System python3. Writes stats_report8.md."""
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

def load_beliefs(tag, rg, tp):
    R = defaultdict(lambda: defaultdict(dict)); p = HERE / "data" / tag / "beliefs.csv"
    if not p.exists(): return R
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp): continue
            R[(r["content_mode"], r["seed"])][int(r["round"])][r["agent"]] = float(r["belief_pA"])
    return R

def load_acc(tag, rg, tp):
    """terminal collective accuracy per (content,seed)."""
    best = {}; p = HERE / "data" / tag / "rounds.csv"
    if not p.exists(): return {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if (r["regime"], r["topology"]) != (rg, tp): continue
            k = (r["content_mode"], r["seed"])
            if k not in best or int(r["round"]) > best[k][0]:
                best[k] = (int(r["round"]), int(r["acc_logodds"]))
    return {k: v[1] for k, v in best.items()}

def roles(rounds_by_seed, cm):
    """per seed -> (holdout_flips list, majority_corrections list, |Δ|holdout, |Δ|maj)."""
    out = {}
    for (c, s), rounds in rounds_by_seed.items():
        if c != cm: continue
        rs = sorted(rounds)
        if len(rs) < 2: continue
        t0, tT = rs[0], rs[-1]
        agents = sorted(set(rounds[t0]) & set(rounds[tT]))
        if len(agents) < 3: continue
        maj_A = np.mean([1.0 if rounds[t0][a] > 0.5 else 0.0 for a in agents]) > 0.5  # wrong majority side
        hf, mc, hmv, mmv = [], [], [], []
        for a in agents:
            b0, bT = rounds[t0][a], rounds[tT][a]
            on_maj0 = (b0 > 0.5) == maj_A
            on_majT = (bT > 0.5) == maj_A
            if on_maj0:           # majority agent: correction = ends on truth side
                mc.append(0.0 if on_majT else 1.0); mmv.append(abs(bT - b0))
            else:                  # holdout: capitulation = ends on majority side
                hf.append(1.0 if on_majT else 0.0); hmv.append(abs(bT - b0))
        out[s] = (hf, mc, hmv, mmv)
    return out

def paired_boot(ev, co, idx, B=10000):
    """seed-clustered paired bootstrap RD (conclusion-evidence) of a per-seed rate at tuple index idx."""
    seeds = sorted(set(ev) & set(co))
    if not seeds: return None
    def rate(d, s):
        vals = d[s][idx]
        return np.mean(vals) if vals else np.nan
    base = np.nanmean([rate(co, s) for s in seeds]) - np.nanmean([rate(ev, s) for s in seeds])
    diffs = []
    n = len(seeds)
    for _ in range(B):
        samp = [seeds[random.randrange(n)] for _ in range(n)]
        ce = np.nanmean([rate(co, s) for s in samp]); ee = np.nanmean([rate(ev, s) for s in samp])
        diffs.append(ce - ee)
    diffs.sort()
    return base, diffs[int(.025*B)], diffs[int(.975*B)], len(seeds)

w("# Role-based mechanism with trial-clustered inference (analyze_stats8.py)\n")
w("HOLDOUT = round-0 belief on truth (minority) side; MAJORITY = on wrong-majority side. Truth side =")
w("opposite the round-0 numerical majority (misleading-majority construction). Capitulation/correction")
w("are terminal side-crossings. Inference = seed-clustered PAIRED bootstrap (unit = seed), 10k resamples.\n")

for tag, rg, tp, lab in CELLS:
    R = load_beliefs(tag, rg, tp); ACC = load_acc(tag, rg, tp)
    if not R:
        w(f"## {lab} [{tag}] — NO DATA\n"); continue
    w(f"## {lab}  [{tag} / {rg}]\n")
    w("| condition | holdout capitulation | majority correction | |Δ| holdout | |Δ| majority | seeds |")
    w("|---|---|---|---|---|---|")
    role = {}
    for cm, clab in [("evidence_only","evidence"),("both","both"),("conclusion_only","conclusion")]:
        role[cm] = roles(R, cm)
        d = role[cm]
        if not d:
            w(f"| {clab} | — | — | — | — | 0 |"); continue
        cap = np.nanmean([np.mean(v[0]) if v[0] else np.nan for v in d.values()])
        cor = np.nanmean([np.mean(v[1]) if v[1] else np.nan for v in d.values()])
        hmv = np.nanmean([np.mean(v[2]) if v[2] else np.nan for v in d.values()])
        mmv = np.nanmean([np.mean(v[3]) if v[3] else np.nan for v in d.values()])
        w(f"| {clab} | {cap:.2f} | {cor:.2f} | {hmv:.2f} | {mmv:.2f} | {len(d)} |")
    # paired bootstrap conclusion vs evidence
    ev, co = role.get("evidence_only", {}), role.get("conclusion_only", {})
    for idx, name in [(0, "holdout capitulation"), (1, "majority correction")]:
        r = paired_boot(ev, co, idx)
        if r:
            base, lo, hi, ns = r
            w(f"\n- **{name}** RD (conclusion−evidence) = {base:+.2f} [seed-clustered paired bootstrap 95% CI {lo:+.2f}, {hi:+.2f}], paired seeds={ns}")
    # trial-level mediation: collective_correct ~ content (+ holdout-flip, majority-corr)
    if ACC:
        rowsY, C, HF, MC = [], [], [], []
        for cm, code in [("evidence_only",0),("conclusion_only",1)]:
            for s, v in role.get(cm, {}).items():
                if (cm, s) not in ACC: continue
                rowsY.append(ACC[(cm, s)]); C.append(code)
                HF.append(np.mean(v[0]) if v[0] else 0.0)
                MC.append(np.mean(v[1]) if v[1] else 0.0)
        if len(set(C)) == 2 and len(rowsY) > 20:
            Y = np.array(rowsY, float)
            def ols(cols):
                X = np.column_stack([np.ones(len(Y))] + cols)
                b, *_ = np.linalg.lstsq(X, Y, rcond=None); return b
            b1 = ols([np.array(C, float)])[1]
            b2 = ols([np.array(C, float), np.array(HF), np.array(MC)])[1]
            w(f"\n- mediation: content→accuracy coefficient = {b1:+.2f} alone; {b2:+.2f} after adding holdout-flip + majority-correction (attenuation ⇒ those roles mediate). Suggestive, not a formal mediation test.")
    w("")

w("Reading: evidence enables BOTH resistance (low holdout capitulation) AND persuasion (high majority")
w("correction); conclusion produces herding (holdouts capitulate, majority fails to correct). The")
w("n=11 overload cell should show both roles updating unreliably regardless of content.")
(HERE / "stats_report8.md").write_text("\n".join(OUT) + "\n")
print("\n".join(OUT))
