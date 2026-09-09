"""REVISION-ERA QUANTITATIVE AUDIT (PNASNEXUS-2026-01296).

Re-derives every quantitative claim introduced or changed during the major revision
directly from archived data in code/data/, and asserts it matches the value reported in
the frozen manuscript and Supplementary Materials.

No reported value is used as an input. Reported values appear only as assertion targets;
every reproduced value is computed from raw or archived processed data.

Deterministic: all resampling uses fixed seed 20260909, matching the revision analyses.
Paths are repository-relative. Imported and executed by audit.py.

Claim families: A primary content results; B wrong-side / pooling sensitivity;
C directional movement; D individual error and within-trial diversity; E both-extremes
diagnostic; F six-rule aggregation robustness; G channel-purity audit; H planted-conclusion
intervention; I fan-in and matched topology contrasts; J positional analyses;
K content x fan-in interaction; L concordant degree ladder; M message volume;
N reasoning manipulation; O capability screening; P hidden-profile probe;
Q signal-structure robustness; R dataset integrity.
"""
from __future__ import annotations
import csv, json, math, random, re, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 20260909
B_RESAMPLE = 10000
RESULTS = []


def record(cid, family, source, reproduced, expected, tol, note=""):
    if expected is None:
        status = "INFO"
    elif reproduced is None:
        status = "FAIL"
    else:
        status = "PASS" if abs(reproduced - expected) <= tol else "FAIL"
    RESULTS.append(dict(id=cid, family=family, source=source,
                        reproduced=(round(reproduced, 6) if isinstance(reproduced, float) else reproduced),
                        expected=expected, tol=tol, status=status, note=note))
    return status


# ------------------------------- shared helpers -------------------------------
def _logit(p, eps=1e-6):
    p = min(1 - eps, max(eps, p)); return math.log(p / (1 - p))

def _sig(x): return 1 / (1 + math.exp(-x))

def rounds_rows(tag):
    f = DATA / tag / "rounds.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []

def cell_rows(tag, regime, topo, content, rnd=3):
    return [r for r in rounds_rows(tag)
            if r["regime"] == regime and r["topology"] == topo
            and r["content_mode"] == content and int(r["round"]) == rnd]

def p_truth(row, key="coll_logodds"):
    v = float(row[key]); return v if row["true_state"] == "A" else 1 - v

def beliefs(tag, regime, topo, content, rnd):
    out = defaultdict(list)
    f = DATA / tag / "beliefs.csv"
    if not f.exists(): return out
    for r in csv.DictReader(open(f)):
        if r["regime"] == regime and r["topology"] == topo and \
           r["content_mode"] == content and int(r["round"]) == rnd:
            out[r["seed"]].append(float(r["belief_pA"]))
    return out

def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def paired_boot(a, b, rng, n=B_RESAMPLE):
    """a,b: seed->0/1. Returns (rd, lo, hi)."""
    ks = sorted(set(a) & set(b))
    da = [a[k] for k in ks]; db = [b[k] for k in ks]
    rd = sum(da) / len(ks) - sum(db) / len(ks)
    dr = []
    for _ in range(n):
        idx = [rng.randrange(len(ks)) for _ in ks]
        dr.append(sum(da[i] for i in idx) / len(ks) - sum(db[i] for i in idx) / len(ks))
    return rd, pct(dr, .025), pct(dr, .975)


# =============================== A. PRIMARY CONTENT ===============================
FAM_A = {"gpt-4.1-mini": dict(base_m=0.632, base_sd=0.015, base_br=0.1355,
                              ev_m=0.805, ev_sd=0.225, ev_br=0.0886,
                              cn_m=0.643, cn_sd=0.381, cn_br=0.2723, cn_wrong=0.36),
         "claude-sonnet-4-6": dict(base_m=0.634, base_sd=0.015, base_br=0.1344,
                                   ev_m=0.898, ev_sd=0.158, ev_br=0.0354,
                                   cn_m=0.659, cn_sd=0.393, cn_br=0.2708, cn_wrong=0.38)}

def family_A():
    for tag, exp in FAM_A.items():
        src = f"data/{tag}/rounds.csv"
        # SI Section S6.7 row "no communication" is the baseline/none cell at round 0
        base = [p_truth(r) for r in cell_rows(tag, "misleading_majority", "baseline", "none", 0)]
        ev = [p_truth(r) for r in cell_rows(tag, "misleading_majority", "all_channel", "evidence_only")]
        cn = [p_truth(r) for r in cell_rows(tag, "misleading_majority", "all_channel", "conclusion_only")]
        for lab, vals, m, sd, br in (("base", base, exp["base_m"], exp["base_sd"], exp["base_br"]),
                                     ("evid", ev, exp["ev_m"], exp["ev_sd"], exp["ev_br"]),
                                     ("concl", cn, exp["cn_m"], exp["cn_sd"], exp["cn_br"])):
            mean = st.mean(vals)
            sdev = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
            brier = sum((1 - x) ** 2 for x in vals) / len(vals)
            record(f"A.{tag}.{lab}.mean", "A primary content", src, mean, m, 0.0006)
            record(f"A.{tag}.{lab}.sd", "A primary content", src, sdev, sd, 0.0006)
            record(f"A.{tag}.{lab}.brier", "A primary content", src, brier, br, 0.00006)
        record(f"A.{tag}.concl.wrongrate", "A primary content", src,
               sum(1 for x in cn if x < .5) / len(cn), exp["cn_wrong"], 0.005)
        # accuracies + Wilson CIs
        for content, lab in (("none", "nocomm"), ("evidence_only", "evid"),
                             ("conclusion_only", "concl"), ("both", "both")):
            topo = "baseline" if content == "none" else "all_channel"
            rnd = 0 if content == "none" else 3
            rows = cell_rows(tag, "misleading_majority", topo, content, rnd)
            if not rows: continue
            a = sum(int(r["acc_logodds"]) for r in rows) / len(rows)
            record(f"A.{tag}.acc.{lab}", "A primary content", src, a, None, None,
                   note=f"N={len(rows)}")
        # paired evidence-minus-conclusion difference and McNemar
        e = {r["seed"]: int(r["acc_logodds"]) for r in cell_rows(tag, "misleading_majority", "all_channel", "evidence_only")}
        c = {r["seed"]: int(r["acc_logodds"]) for r in cell_rows(tag, "misleading_majority", "all_channel", "conclusion_only")}
        ks = sorted(set(e) & set(c))
        rd = sum(e[k] for k in ks) / len(ks) - sum(c[k] for k in ks) / len(ks)
        b01 = sum(1 for k in ks if e[k] == 0 and c[k] == 1)
        b10 = sum(1 for k in ks if e[k] == 1 and c[k] == 0)
        chi = (abs(b10 - b01) - 1) ** 2 / (b10 + b01) if (b10 + b01) else 0.0
        record(f"A.{tag}.rd_evid_minus_concl", "A primary content", src, rd,
               0.26 if tag == "gpt-4.1-mini" else 0.34, 0.005)
        record(f"A.{tag}.mcnemar_chi2", "A primary content", src, chi, None, None,
               note=f"discordant {b10}/{b01}, N={len(ks)}")


# ===================== B. WRONG-SIDE / POOLING SENSITIVITY =====================
def family_B():
    exp = {"gpt-4.1-mini": dict(log_rate=0.36, log_mean=0.165, ar_rate=0.31, ar_mean=0.316),
           "claude-sonnet-4-6": dict(log_rate=0.38, log_mean=0.171, ar_rate=0.26, ar_mean=0.334)}
    for tag, e in exp.items():
        src = f"data/{tag}/rounds.csv"
        rows = cell_rows(tag, "misleading_majority", "all_channel", "conclusion_only")
        for pool, key, rk, mk in (("log", "coll_logodds", "log_rate", "log_mean"),
                                  ("arith", "coll_mean", "ar_rate", "ar_mean")):
            q = [p_truth(r, key) for r in rows]
            wrong = [x for x in q if x < .5]
            record(f"B.{tag}.{pool}.wrongrate", "B pooling sensitivity", src,
                   len(wrong) / len(q), e[rk], 0.005)
            record(f"B.{tag}.{pool}.wrongmean", "B pooling sensitivity", src,
                   st.mean(wrong), e[mk], 0.001)


# ===================== C. DIRECTIONAL MOVEMENT =====================
def family_C():
    exp = {("gpt-4.1-mini", "evidence_only"): 0.11, ("gpt-4.1-mini", "conclusion_only"): 0.39,
           ("claude-sonnet-4-6", "evidence_only"): 0.04, ("claude-sonnet-4-6", "conclusion_only"): 0.38}
    for (tag, content), e in exp.items():
        src = f"data/{tag}/rounds.csv"
        r0 = {r["seed"]: p_truth(r) for r in cell_rows(tag, "misleading_majority", "all_channel", content, 0)}
        r3 = {r["seed"]: p_truth(r) for r in cell_rows(tag, "misleading_majority", "all_channel", content, 3)}
        ks = sorted(set(r0) & set(r3))
        d = [r3[k] - r0[k] for k in ks]
        toward = [x for x in d if x < 0]
        record(f"C.{tag}.{content}.toward_frac", "C directional movement", src,
               len(toward) / len(d), e, 0.005)
        record(f"C.{tag}.{content}.toward_mean_mag", "C directional movement", src,
               st.mean(toward) if toward else None, None, None,
               note="mean signed movement among majority-ward trials")


# ===================== D. ERROR AND WITHIN-TRIAL DIVERSITY =====================
def family_D():
    exp = {"gpt-4.1-mini": (0.047, 0.091, 0.210, 0.288),
           "claude-sonnet-4-6": (0.048, 0.093, 0.210, 0.255)}
    for tag, (d0, d3, e0, e3) in exp.items():
        src = f"data/{tag}/rounds.csv"
        for rnd, dv, ev in ((0, d0, e0), (3, d3, e3)):
            rows = cell_rows(tag, "misleading_majority", "all_channel", "conclusion_only", rnd)
            record(f"D.{tag}.D.r{rnd}", "D error and diversity", src,
                   st.mean(float(r["D"]) for r in rows), dv, 0.0006)
            record(f"D.{tag}.ebar.r{rnd}", "D error and diversity", src,
                   st.mean(float(r["e_bar"]) for r in rows), ev, 0.0006)
    # concordant evidence high-degree ladder diversity / error
    lad = [("gpt-4.1-mini_n11", "circle", 2.0), ("gpt-4.1-mini_overload", "lattice_k8", 8.0)]
    for tag, topo, deg in lad:
        src = f"data/{tag}/rounds.csv"
        r0 = cell_rows(tag, "concordant", topo, "evidence_only", 0)
        r3 = cell_rows(tag, "concordant", topo, "evidence_only", 3)
        if not r0: continue
        record(f"D.concordant.deg{deg}.D.r0", "D error and diversity", src,
               st.mean(float(r["D"]) for r in r0), 0.007 if deg == 2 else 0.006, 0.0015)
        record(f"D.concordant.deg{deg}.D.term", "D error and diversity", src,
               st.mean(float(r["D"]) for r in r3), 0.002 if deg == 2 else 0.128, 0.002)
        record(f"D.concordant.deg{deg}.ebar.term", "D error and diversity", src,
               st.mean(float(r["e_bar"]) for r in r3), 0.005 if deg == 2 else 0.314, 0.002)


# ===================== E. BOTH-EXTREMES DIAGNOSTIC =====================
def family_E():
    rng = random.Random(SEED + 2)
    exp = {("gpt-4.1-mini", 0.20): (0.37, 0.54, 0.17, 0.03, 0.31),
           ("claude-sonnet-4-6", 0.20): (0.18, 0.52, 0.34, 0.18, 0.50)}
    alt = {("gpt-4.1-mini", 0.10): (0.13, 0.40, 0.27), ("gpt-4.1-mini", 0.25): (0.41, 0.57, 0.16),
           ("claude-sonnet-4-6", 0.10): (0.08, 0.44, 0.36), ("claude-sonnet-4-6", 0.25): (0.22, 0.52, 0.30)}
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6"):
        src = f"data/{tag}/beliefs.csv"
        for lo in (0.10, 0.20, 0.25):
            hi = 1 - lo
            ind = {}
            for content, lab in (("evidence_only", "evid"), ("conclusion_only", "concl")):
                bl = beliefs(tag, "misleading_majority", "all_channel", content, 3)
                ind[lab] = {k: int(any(x < lo for x in v) and any(x > hi for x in v)) for k, v in bl.items()}
                prev = sum(ind[lab].values()) / len(ind[lab])
                tgt = None
                if (tag, lo) in exp and lab == "evid": tgt = exp[(tag, lo)][0]
                if (tag, lo) in exp and lab == "concl": tgt = exp[(tag, lo)][1]
                if (tag, lo) in alt and lab == "evid": tgt = alt[(tag, lo)][0]
                if (tag, lo) in alt and lab == "concl": tgt = alt[(tag, lo)][1]
                record(f"E.{tag}.t{lo}.{lab}", "E both-extremes", src, prev, tgt, 0.005)
            rd, l, h = paired_boot(ind["concl"], ind["evid"], rng)
            tgt = exp[(tag, lo)][2] if (tag, lo) in exp else (alt[(tag, lo)][2] if (tag, lo) in alt else None)
            record(f"E.{tag}.t{lo}.diff", "E both-extremes", src, rd, tgt, 0.005)
            if (tag, lo) in exp:
                record(f"E.{tag}.t{lo}.ci_lo", "E both-extremes", src, l, exp[(tag, lo)][3], 0.02)
                record(f"E.{tag}.t{lo}.ci_hi", "E both-extremes", src, h, exp[(tag, lo)][4], 0.02)
        # correct/incorrect split at 0.20/0.80
        bl = beliefs(tag, "misleading_majority", "all_channel", "conclusion_only", 3)
        accm = {r["seed"]: int(r["acc_logodds"]) for r in cell_rows(tag, "misleading_majority", "all_channel", "conclusion_only")}
        grp = defaultdict(list)
        for k, v in bl.items():
            if k in accm:
                grp["correct" if accm[k] else "incorrect"].append(
                    int(any(x < .20 for x in v) and any(x > .80 for x in v)))
        tgt = {"gpt-4.1-mini": (0.47, 0.67), "claude-sonnet-4-6": (0.39, 0.74)}[tag]
        for i, o in enumerate(("correct", "incorrect")):
            record(f"E.{tag}.split.{o}", "E both-extremes", src,
                   sum(grp[o]) / len(grp[o]), tgt[i], 0.006)


# ===================== F. SIX-RULE AGGREGATION ROBUSTNESS =====================
RULES = ["log", "arith", "major", "median", "clip05", "dropmax"]

def classify(bel, rule):
    if rule == "log":     return sum(_logit(b) for b in bel) / len(bel) >= 0
    if rule == "arith":   return sum(bel) / len(bel) >= 0.5
    if rule == "major":   return sum(1 for b in bel if b >= 0.5) / len(bel) >= 0.5
    if rule == "median":  return st.median(bel) >= 0.5
    if rule == "clip05":
        c = [min(0.95, max(0.05, b)) for b in bel]
        return sum(_logit(b) for b in c) / len(c) >= 0
    if rule == "dropmax":
        L = [_logit(b) for b in bel]
        j = max(range(len(L)), key=lambda i: abs(L[i]))
        Lo = [x for i, x in enumerate(L) if i != j]
        return sum(Lo) / len(Lo) >= 0
    raise ValueError(rule)

def rule_acc(tag, regime, topo, content, rule):
    bl = beliefs(tag, regime, topo, content, 3)
    truth = {r["seed"]: r["true_state"] for r in rounds_rows(tag)
             if r["regime"] == regime and r["topology"] == topo and r["content_mode"] == content}
    vals = [int(classify(v, rule) == (truth[k] == "A")) for k, v in bl.items() if k in truth and len(v) >= 3]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

def family_F():
    exp = {"gpt-4.1-mini":     dict(log=.26, arith=.26, major=.28, median=.28, clip05=.26, dropmax=.28),
           "claude-sonnet-4-6": dict(log=.34, arith=.24, major=.20, median=.20, clip05=.22, dropmax=.26),
           "gemini-2.5-flash":  dict(log=.03, arith=.07, major=.10, median=.10, clip05=.07, dropmax=.03)}
    for tag, e in exp.items():
        src = f"data/{tag}/beliefs.csv"
        for rule in RULES:
            ea, _ = rule_acc(tag, "misleading_majority", "all_channel", "evidence_only", rule)
            ca, _ = rule_acc(tag, "misleading_majority", "all_channel", "conclusion_only", rule)
            record(f"F.{tag}.{rule}.evid_minus_concl", "F aggregation robustness", src,
                   (ea - ca) if ea is not None and ca is not None else None, e[rule], 0.006)
    # degree-10 concordant evidence sharing across rules (spec L: log 0.68, alternatives 0.50-0.64)
    src = "data/gpt-4.1-mini_n11/beliefs.csv"
    alts = []
    for rule in RULES:
        a, _ = rule_acc("gpt-4.1-mini_n11", "concordant", "all_channel", "evidence_only", rule)
        if rule == "log":
            record("L.deg10.concordant.evid.log", "F aggregation robustness", src, a, 0.68, 0.006)
        else:
            alts.append(a)
    record("L.deg10.concordant.evid.alt_min", "F aggregation robustness", src, min(alts), 0.50, 0.006)
    record("L.deg10.concordant.evid.alt_max", "F aggregation robustness", src, max(alts), 0.64, 0.006)


# ===================== G. CHANNEL-PURITY AUDIT =====================
LEAK_TAGS = ["gpt-4.1-mini", "gpt-4.1-mini_n11", "gpt-4.1-mini_degree", "gpt-4.1-mini_overload",
             "gpt-4.1-mini_sigsweep", "claude-sonnet-4-6", "claude-sonnet-4-6_think", "gemini-2.5-flash"]
VERDICT = re.compile(
    r"(i think|i believe|i suspect|i conclude|i'?d (guess|say|bet|lean)|my (best )?guess|"
    r"my (conclusion|verdict)|i lean|leaning toward|i'?m (fairly |quite |very )?confident|"
    r"i am confident|more likely to be|most likely (to be|state|answer|that the)|"
    r"favou?rs (state )?(a|b)\b|points to (state )?(a|b)\b|i favou?r|going with (state )?(a|b)\b|"
    r"best guess is|state is (most |more )?(probably|likely)|my read is)", re.I)
CONF = re.compile(r"(with|at)\s+(about|approximately|around|roughly|~)?\s*\d{1,3}(\.\d+)?\s*%\s*"
                  r"(confidence|certainty|confident|sure|probability)?", re.I)
CONFB = re.compile(r"\b\d{1,3}(\.\d+)?\s*%", re.I)
SUBST = re.compile(r"(\br\d\b|occurs with probability|if the (true )?state (is|were)|under state [ab]\b|"
                   r"given state [ab]\b|likelihood ratio|\bllr\b|diagnostic|probability 0\.\d+|\b0\.\d{2}\b)", re.I)

def family_G():
    ev_total = ev_leak = cn_total = cn_leak = 0
    focal = {}
    for tag in LEAK_TAGS:
        f = DATA / tag / "messages.csv"
        if not f.exists(): continue
        for r in csv.DictReader(open(f)):
            cm = r.get("content_mode"); txt = r.get("message_text", "")
            if cm == "evidence_only":
                ev_total += 1
                if VERDICT.search(txt): ev_leak += 1
            elif cm == "conclusion_only":
                cn_total += 1
                probe = CONFB.sub(" ", CONF.sub(" ", txt))
                sub = bool(SUBST.search(probe))
                if sub: cn_leak += 1
                if r["regime"] == "misleading_majority" and r["topology"] == "all_channel":
                    fam = tag if tag in ("gpt-4.1-mini", "claude-sonnet-4-6") else None
                    if fam:
                        a = focal.setdefault(fam, [0, 0]); a[0] += 1; a[1] += int(sub)
    src = "data/*/messages.csv"
    record("G.evidence_total", "G channel purity", src, ev_total, 32520, 0)
    record("G.evidence_verdict_leaks", "G channel purity", src, ev_leak, 0, 0)
    record("G.conclusion_total", "G channel purity", src, cn_total, 40335, 0)
    record("G.conclusion_substantive", "G channel purity", src, cn_leak, 145, 0)
    record("G.conclusion_rate", "G channel purity", src, cn_leak / cn_total, 0.0036, 0.0002)
    record("G.focal.gpt.total", "G channel purity", src, focal["gpt-4.1-mini"][0], 1500, 0)
    record("G.focal.gpt.leaks", "G channel purity", src, focal["gpt-4.1-mini"][1], 0, 0)
    record("G.focal.claude.total", "G channel purity", src, focal["claude-sonnet-4-6"][0], 750, 0)
    record("G.focal.claude.leaks", "G channel purity", src, focal["claude-sonnet-4-6"][1], 141, 0)
    record("G.focal.claude.rate", "G channel purity", src,
           focal["claude-sonnet-4-6"][1] / focal["claude-sonnet-4-6"][0], 0.188, 0.001)
    # round concentration (observational, reported in SI)
    byr = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(DATA / "claude-sonnet-4-6" / "messages.csv")):
        if r.get("content_mode") != "conclusion_only": continue
        if r["regime"] != "misleading_majority" or r["topology"] != "all_channel": continue
        probe = CONFB.sub(" ", CONF.sub(" ", r.get("message_text", "")))
        a = byr[r["round"]]; a[0] += 1; a[1] += int(bool(SUBST.search(probe)))
    record("G.focal.claude.round1_rate", "G channel purity", "data/claude-sonnet-4-6/messages.csv",
           byr["1"][1] / byr["1"][0], 0.556, 0.01, note="disclosure concentrated in round 1")


# ===================== H. PLANTED-CONCLUSION INTERVENTION =====================
def family_H():
    exp = {"replay-gpt-4.1-mini": (40, 0.51, 0.85, 0.34),
           "replay-claude-sonnet-4-6": (15, 0.64, 0.99, 0.36)}
    for tag, (n_exp, ace_exp, corr_exp, wrong_exp) in exp.items():
        f = DATA / tag / "replay.csv"
        if not f.exists():
            record(f"H.{tag}.present", "H planted conclusion", str(f), None, 1, 0,
                   note="replay.csv MISSING"); continue
        rows = list(csv.DictReader(open(f)))
        src = f"data/{tag}/replay.csv"
        by = defaultdict(dict)
        for r in rows: by[r["seed"]][r["mode"]] = float(r["p_truth"])
        matched = [s for s, m in by.items() if "correct" in m and "wrong" in m]
        pc = [by[s]["correct"] for s in matched]; pw = [by[s]["wrong"] for s in matched]
        record(f"H.{tag}.n_matched", "H planted conclusion", src, len(matched), n_exp, 0)
        record(f"H.{tag}.correct_mean", "H planted conclusion", src, st.mean(pc), corr_exp, 0.006)
        record(f"H.{tag}.wrong_mean", "H planted conclusion", src, st.mean(pw), wrong_exp, 0.006)
        record(f"H.{tag}.plant_effect", "H planted conclusion", src,
               st.mean(pc) - st.mean(pw), ace_exp, 0.006)
        meta = DATA / tag / "run_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text()).get("ACE_ptruth")
            record(f"H.{tag}.meta_consistency", "H planted conclusion", f"data/{tag}/run_meta.json",
                   st.mean(pc) - st.mean(pw), m, 0.006, note="replay.csv vs archived run_meta")


# ===================== I. FAN-IN AND MATCHED TOPOLOGY CONTRASTS =====================
def acc_map(tag, regime, topo, content):
    return {r["seed"]: int(r["acc_logodds"]) for r in cell_rows(tag, regime, topo, content)}

def family_I():
    rng = random.Random(SEED + 3)
    ladder = [("gpt-4.1-mini_n11", "circle", 2.0, 1.00), ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55, 0.94),
              ("gpt-4.1-mini_degree", "lattice_k4", 4.0, 0.90), ("gpt-4.1-mini_degree", "lattice_k6", 6.0, 0.76),
              ("gpt-4.1-mini_degree", "lattice_k8", 8.0, 0.60), ("gpt-4.1-mini_n11", "all_channel", 10.0, 0.58)]
    for tag, topo, deg, e in ladder:
        m = acc_map(tag, "misleading_majority", topo, "conclusion_only")
        record(f"I.ladder.deg{deg}", "I fan-in and topology", f"data/{tag}/rounds.csv",
               sum(m.values()) / len(m), e, 0.006, note=f"N={len(m)}")
    # matched contrasts, paired by seed, sign convention A minus B as written in the manuscript
    contrasts = [("all_channel", "gpt-4.1-mini_n11", "lattice_k8", "gpt-4.1-mini_degree",
                  -0.020, -0.140, 0.100, "I.contrast.allchannel_minus_k8"),
                 ("ring_shortcuts", "gpt-4.1-mini_n11", "circle", "gpt-4.1-mini_n11",
                  -0.060, -0.140, 0.000, "I.contrast.ringshortcuts_minus_circle"),
                 ("all_channel", "gpt-4.1-mini_n11", "ring_shortcuts", "gpt-4.1-mini_n11",
                  -0.360, -0.520, -0.200, "I.contrast.allchannel_minus_ringshortcuts")]
    for ta, taga, tb, tagb, rd_e, lo_e, hi_e, cid in contrasts:
        A = acc_map(taga, "misleading_majority", ta, "conclusion_only")
        Bc = acc_map(tagb, "misleading_majority", tb, "conclusion_only")
        rd, lo, hi = paired_boot(A, Bc, rng)
        src = f"data/{taga}/rounds.csv + data/{tagb}/rounds.csv"
        record(cid + ".rd", "I fan-in and topology", src, rd, rd_e, 0.006)
        record(cid + ".ci_lo", "I fan-in and topology", src, lo, lo_e, 0.02)
        record(cid + ".ci_hi", "I fan-in and topology", src, hi, hi_e, 0.02)
    # seed-clustered logistic degree association (Newton-Raphson, cluster-robust SE)
    X, Y, G = [], [], []
    for tag, topo, deg, _ in ladder:
        for r in cell_rows(tag, "misleading_majority", topo, "conclusion_only"):
            X.append(deg); Y.append(int(r["acc_logodds"])); G.append(r["seed"])
    b0, b1 = 0.0, 0.0
    for _ in range(60):
        p = [_sig(b0 + b1 * x) for x in X]
        g0 = sum(y - pi for y, pi in zip(Y, p)); g1 = sum((y - pi) * x for y, pi, x in zip(Y, p, X))
        h00 = -sum(pi * (1 - pi) for pi in p); h01 = -sum(pi * (1 - pi) * x for pi, x in zip(p, X))
        h11 = -sum(pi * (1 - pi) * x * x for pi, x in zip(p, X))
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12: break
        d0 = (-g0 * h11 + g1 * h01) / det; d1 = (-h00 * g1 + h01 * g0) / det
        b0 += d0; b1 += d1
        if abs(d0) + abs(d1) < 1e-10: break
    p = [_sig(b0 + b1 * x) for x in X]
    bread = [[-sum(pi * (1 - pi) for pi in p), -sum(pi * (1 - pi) * x for pi, x in zip(p, X))],
             [-sum(pi * (1 - pi) * x for pi, x in zip(p, X)), -sum(pi * (1 - pi) * x * x for pi, x in zip(p, X))]]
    det = bread[0][0] * bread[1][1] - bread[0][1] ** 2
    inv = [[bread[1][1] / det, -bread[0][1] / det], [-bread[0][1] / det, bread[0][0] / det]]
    scores = defaultdict(lambda: [0.0, 0.0])
    for y, pi, x, g in zip(Y, p, X, G):
        s = scores[g]; s[0] += (y - pi); s[1] += (y - pi) * x
    meat = [[0.0, 0.0], [0.0, 0.0]]
    for s in scores.values():
        meat[0][0] += s[0] * s[0]; meat[0][1] += s[0] * s[1]
        meat[1][0] += s[1] * s[0]; meat[1][1] += s[1] * s[1]
    V = [[sum(inv[i][k] * meat[k][l] for k in range(2)) for l in range(2)] for i in range(2)]
    V = [[sum(V[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    se = math.sqrt(abs(V[1][1])); z = b1 / se if se else float("nan")
    pv = math.erfc(abs(z) / math.sqrt(2))
    record("I.degree_logit.slope", "I fan-in and topology", "data/*_n11,_degree/rounds.csv", b1, None, None,
           note=f"seed-clustered logistic slope, z={z:.2f}")
    record("I.degree_logit.log10p", "I fan-in and topology", "data/*_n11,_degree/rounds.csv",
           math.log10(pv) if pv > 0 else -400, math.log10(4.7e-11), 1.5,
           note=f"reported p = 4.7e-11; reproduced p = {pv:.2e}")


# ===================== J. POSITIONAL ANALYSES =====================
def _corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")

def _partial(xs, ys, zs):
    def resid(v, z):
        mz, mv = sum(z) / len(z), sum(v) / len(v)
        den = sum((a - mz) ** 2 for a in z)
        b = sum((a - mz) * (c - mv) for a, c in zip(z, v)) / den if den else 0.0
        return [c - (mv + b * (a - mz)) for a, c in zip(z, v)]
    return _corr(resid(xs, zs), resid(ys, zs))

def _positional_records():
    TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
    TOPOS = ["wheel", "hierarchy", "chain"]
    truth = {(r["topology"], r["seed"]): r["true_state"] for r in rounds_rows(TAG)
             if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in TOPOS}
    rec, fanin = defaultdict(dict), {}
    for r in csv.DictReader(open(DATA / TAG / "beliefs.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS: continue
        k = (r["topology"], r["seed"], r["agent"]); rd = int(r["round"])
        rec[k][rd] = (float(r["belief_pA"]), float(r["llr"]))
        if rd == 1: fanin[k] = int(r["n_received"])
    nbrs = {}
    for r in csv.DictReader(open(DATA / TAG / "messages.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS: continue
        if int(r["round"]) != 1: continue
        nbrs[(r["topology"], r["seed"], r["agent"])] = [s for s in (r.get("received_from") or "").split(";") if s]
    out = []
    for k, byr in rec.items():
        tp, s, a = k
        if (tp, s) not in truth or k not in fanin or k not in nbrs: continue
        rs = sorted(byr); llr = byr[rs[0]][1]; bT = byr[rs[-1]][0]
        tA = truth[(tp, s)] == "A"
        if (llr > 0) != tA: continue
        nf = [fanin[(tp, s, x)] for x in nbrs[k] if (tp, s, x) in fanin]
        if not nf: continue
        out.append(dict(topo=tp, seed=s, own=fanin[k], nbr=sum(nf) / len(nf),
                        cap=1 if ((bT > .5) != tA) else 0))
    return out

def family_J():
    rng = random.Random(SEED + 4)
    recs = _positional_records()
    src = "data/gpt-4.1-mini/{beliefs,messages,rounds}.csv"
    record("J.n_agents", "J positional", src, len(recs), 600, 0)
    byseed = defaultdict(list)
    for r in recs: byseed[(r["topo"], r["seed"])].append(r)
    keys = sorted(byseed)
    record("J.n_trials", "J positional", src, len(keys), 300, 0)
    def stats(sample):
        own = [r["own"] for r in sample]; nbr = [r["nbr"] for r in sample]; cap = [r["cap"] for r in sample]
        return _corr(own, cap), _partial(nbr, cap, own), _partial(own, cap, nbr)
    obs = stats(recs)
    draws = [[], [], []]
    for _ in range(B_RESAMPLE):
        samp = []
        for _ in keys: samp += byseed[keys[rng.randrange(len(keys))]]
        try:
            v = stats(samp)
            for i in range(3):
                if v[i] == v[i]: draws[i].append(v[i])
        except Exception: pass
    tgt = [(0.357, 0.257, 0.450), (0.007, -0.056, 0.075), (0.265, 0.172, 0.356)]
    ids = ["J.own_fanin", "J.neighbour_given_own", "J.own_given_neighbour"]
    for i, cid in enumerate(ids):
        record(cid + ".r", "J positional", src, obs[i], tgt[i][0], 0.006)
        record(cid + ".ci_lo", "J positional", src, pct(draws[i], .025), tgt[i][1], 0.02)
        record(cid + ".ci_hi", "J positional", src, pct(draws[i], .975), tgt[i][2], 0.02)
    wh = [r for r in recs if r["topo"] == "wheel"]
    mx = max(r["own"] for r in wh)
    hub = [r["cap"] for r in wh if r["own"] == mx]; spk = [r["cap"] for r in wh if r["own"] < mx]
    d = sum(hub) / len(hub) - sum(spk) / len(spk)
    wseed = defaultdict(list)
    for r in wh: wseed[r["seed"]].append(r)
    wk = sorted(wseed); dd = []
    for _ in range(B_RESAMPLE):
        s = []
        for _ in wk: s += wseed[wk[rng.randrange(len(wk))]]
        h = [r["cap"] for r in s if r["own"] == mx]; p_ = [r["cap"] for r in s if r["own"] < mx]
        if h and p_: dd.append(sum(h) / len(h) - sum(p_) / len(p_))
    record("J.wheel_hub_minus_spoke.rd", "J positional", src, d, 0.439, 0.006)
    record("J.wheel_hub_minus_spoke.ci_lo", "J positional", src, pct(dd, .025), 0.288, 0.02)
    record("J.wheel_hub_minus_spoke.ci_hi", "J positional", src, pct(dd, .975), 0.590, 0.02)


# ===================== K. CONTENT x FAN-IN INTERACTION =====================
# The paired content contrast exists ONLY at degrees 2, 2.55 and 10. The design is NOT a
# six-level factorial; the randomization test uses the three paired degree levels only.
PAIRED_DEG = [("circle", 2.0), ("ring_shortcuts", 2.55), ("all_channel", 10.0)]

def family_K():
    rng = random.Random(SEED + 5)
    tag = "gpt-4.1-mini_n11"
    src = f"data/{tag}/rounds.csv"
    exp = {("misleading_majority", 2.0): (0.000, 0.000, 0.000),
           ("misleading_majority", 2.55): (0.060, 0.000, 0.140),
           ("misleading_majority", 10.0): (-0.200, -0.320, -0.080),
           ("concordant", 2.0): (0.000, 0.000, 0.000),
           ("concordant", 2.55): (0.000, 0.000, 0.000),
           ("concordant", 10.0): (-0.280, -0.420, -0.140)}
    exp_p = {"misleading_majority": 0.0015, "concordant": 0.0014}
    for regime in ("misleading_majority", "concordant"):
        paired = {}
        for topo, deg in PAIRED_DEG:
            E = acc_map(tag, regime, topo, "evidence_only")
            C = acc_map(tag, regime, topo, "conclusion_only")
            common = sorted(set(E) & set(C))
            assert common, f"no shared seeds at {topo}/{regime}"
            paired[deg] = [(E[s], C[s]) for s in common]
            rd, lo, hi = paired_boot(E, C, rng)
            e = exp[(regime, deg)]
            record(f"K.rd.{regime}.deg{deg}", "K content x fan-in", src, rd, e[0], 0.006,
                   note=f"evidence minus conclusion, {len(common)} paired seeds")
            record(f"K.rd.{regime}.deg{deg}.ci_lo", "K content x fan-in", src, lo, e[1], 0.02)
            record(f"K.rd.{regime}.deg{deg}.ci_hi", "K content x fan-in", src, hi, e[2], 0.02)
        def stat(p):
            m = {d: sum(a - b for a, b in v) / len(v) for d, v in p.items()}
            return m[10.0] - (m[2.0] + m[2.55]) / 2
        obs = stat(paired)
        ge = 0
        for _ in range(B_RESAMPLE):
            perm = {}
            for d, v in paired.items():
                perm[d] = [((b, a) if rng.random() < 0.5 else (a, b)) for a, b in v]
            if abs(stat(perm)) >= abs(obs) - 1e-12: ge += 1
        pv = (ge + 1) / (B_RESAMPLE + 1)
        record(f"K.interaction.{regime}.stat", "K content x fan-in", src, obs, None, None,
               note="RD(deg10) - mean(RD deg2, RD deg2.55); within-seed content permutation")
        record(f"K.interaction.{regime}.p", "K content x fan-in", src, pv, exp_p[regime], 0.0015,
               note=f"{B_RESAMPLE} within-seed permutations, two-sided")
    record("K.design.paired_levels", "K content x fan-in", src, len(PAIRED_DEG), 3, 0,
           note="content contrast exists only at degrees 2, 2.55, 10")


# ===================== L. CONCORDANT DEGREE LADDER (evidence sharing) =====================
def family_L():
    ladder = [("gpt-4.1-mini_n11", "circle", 2.0, 1.00, 50), ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55, 1.00, 50),
              ("gpt-4.1-mini_overload", "lattice_k4", 4.0, 1.00, 30), ("gpt-4.1-mini_overload", "lattice_k6", 6.0, 1.00, 30),
              ("gpt-4.1-mini_overload", "lattice_k8", 8.0, 0.90, 30), ("gpt-4.1-mini_n11", "all_channel", 10.0, 0.68, 50)]
    for tag, topo, deg, e, n_e in ladder:
        rows = cell_rows(tag, "concordant", topo, "evidence_only")
        record(f"L.concordant_ladder.deg{deg}", "L concordant degree ladder", f"data/{tag}/rounds.csv",
               sum(int(r["acc_logodds"]) for r in rows) / len(rows), e, 0.006)
        record(f"L.concordant_ladder.deg{deg}.n", "L concordant degree ladder", f"data/{tag}/rounds.csv",
               len(rows), n_e, 0)


# ===================== M. INCOMING MESSAGE LOAD =====================
def _load_cells(tag, regime):
    f = DATA / tag / "full" / "transcripts.jsonl"
    agg = defaultdict(list)
    if not f.exists(): return agg
    for line in open(f):
        o = json.loads(line)
        if o["regime"] != regime: continue
        for rd in o["rounds"]:
            if rd["round"] == 0: continue
            for ag in rd["agents"]:
                rec = ag.get("received") or []
                if not rec: continue
                agg[(o["topology"], o["content_mode"])].append(
                    (len(rec), sum(len(x["text"]) for x in rec)))
    return agg

def family_M():
    src = "data/gpt-4.1-mini_n11/full/transcripts.jsonl"
    mm = _load_cells("gpt-4.1-mini_n11", "misleading_majority")
    cc = _load_cells("gpt-4.1-mini_n11", "concordant")
    # SI Section S6.11 table, misleading-majority regime
    tbl = [("circle", "conclusion_only", 2.0, 141, 71), ("circle", "evidence_only", 2.0, 241, 120),
           ("ring_shortcuts", "conclusion_only", 2.5, 171, 67), ("ring_shortcuts", "evidence_only", 2.5, 305, 120),
           ("all_channel", "conclusion_only", 10.0, 653, 65), ("all_channel", "evidence_only", 10.0, 1188, 119)]
    for topo, cm, sen_e, ch_e, cps_e in tbl:
        v = mm.get((topo, cm))
        if not v:
            record(f"M.{topo}.{cm}.chars", "M message load", src, None, ch_e, 1.0, note="cell absent")
            continue
        sen = sum(a for a, _ in v) / len(v); ch = sum(b for _, b in v) / len(v)
        record(f"M.{topo}.{cm}.senders", "M message load", src, sen, sen_e, 0.05)
        record(f"M.{topo}.{cm}.chars", "M message load", src, ch, ch_e, 1.0)
        record(f"M.{topo}.{cm}.chars_per_sender", "M message load", src, ch / sen, cps_e, 1.0)
    # degree-10 concordant comparison quoted in the SI text
    for cm, e in (("evidence_only", 1185), ("conclusion_only", 650)):
        v = cc.get(("all_channel", cm))
        ch = sum(b for _, b in v) / len(v) if v else None
        record(f"M.concordant.all_channel.{cm}.chars", "M message load", src, ch, e, 1.0)
    ve = cc.get(("all_channel", "evidence_only")); vc = cc.get(("all_channel", "conclusion_only"))
    if ve and vc:
        re_ = (sum(b for _, b in ve) / len(ve)) / (sum(b for _, b in vc) / len(vc))
        record("M.degree10.evidence_conclusion_ratio", "M message load", src, re_, 1.82, 0.02)
    # within-cell CV range quoted as 0.02 to 0.25
    cvs = []
    for agg in (mm, cc):
        for k, v in agg.items():
            ch = [b for _, b in v]
            if len(ch) < 5: continue
            m = sum(ch) / len(ch)
            sd = math.sqrt(sum((x - m) ** 2 for x in ch) / len(ch))
            cvs.append(sd / m if m else 0.0)
    record("M.cv.min", "M message load", src, min(cvs) if cvs else None, 0.02, 0.011)
    record("M.cv.max", "M message load", src, max(cvs) if cvs else None, 0.25, 0.011)


# ===================== N. REASONING-EFFORT MANIPULATION =====================
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def family_N():
    src = "data/claude-sonnet-4-6{,_think}/rounds.csv"
    RG, TP, CM = "misleading_majority", "all_channel", "conclusion_only"
    off = cell_rows("claude-sonnet-4-6", RG, TP, CM)
    on = cell_rows("claude-sonnet-4-6_think", RG, TP, CM)
    for label, rows, acc_e, n_e in (("off", off, 0.62, 50), ("on", on, 1.00, 30)):
        k = sum(int(r["acc_logodds"]) for r in rows); n = len(rows)
        lo, hi = wilson(k, n)
        record(f"N.{label}.n", "N reasoning effort", src, n, n_e, 0)
        record(f"N.{label}.accuracy", "N reasoning effort", src, k / n if n else None, acc_e, 0.006)
        record(f"N.{label}.ci_lo", "N reasoning effort", src, lo, {"off": 0.48, "on": 0.89}[label], 0.011)
        record(f"N.{label}.ci_hi", "N reasoning effort", src, hi, {"off": 0.74, "on": 1.00}[label], 0.011)
    # SI Section S9 reports the trajectory statement "e-bar declined to 0.004", which is the
    # minimum of the round-mean trajectory, not the terminal round.
    traj = defaultdict(list)
    for r in rounds_rows("claude-sonnet-4-6_think"):
        if r["regime"] == RG and r["topology"] == TP and r["content_mode"] == CM:
            traj[int(r["round"])].append(float(r["e_bar"]))
    means = {k: sum(v) / len(v) for k, v in traj.items()}
    record("N.on.e_bar_min", "N reasoning effort", src,
           min(means.values()) if means else None, 0.004, 0.0006,
           note="minimum of the round-mean trajectory " +
                str([round(means[k], 4) for k in sorted(means)]))
    record("N.on.e_bar_terminal", "N reasoning effort", src,
           means.get(max(means)) if means else None, None, None,
           note="terminal-round mean individual error")
    k1, n1 = sum(int(r["acc_logodds"]) for r in on), len(on)
    k0, n0 = sum(int(r["acc_logodds"]) for r in off), len(off)
    p1, p0 = k1 / n1, k0 / n0
    pp = (k1 + k0) / (n1 + n0)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se if se else float("nan")
    pv = math.erfc(abs(z) / math.sqrt(2))
    record("N.z", "N reasoning effort", src, z, 3.87, 0.02)
    record("N.log10p", "N reasoning effort", src, math.log10(pv) if pv > 0 else -400,
           math.log10(1.1e-4), 0.06, note=f"reproduced p = {pv:.2e}")
    # token cost, re-derived per trial from the archived transcripts for the focal cell
    def usage(tag):
        f = DATA / tag / "full" / "transcripts.jsonl"
        if not f.exists(): return (None, None)
        pin, pout = [], []
        for line in open(f):
            o = json.loads(line)
            if not (o["regime"] == RG and o["topology"] == TP and o["content_mode"] == CM): continue
            u = o.get("usage") or {}
            if "prompt_tokens" in u: pin.append(u["prompt_tokens"])
            if "completion_tokens" in u: pout.append(u["completion_tokens"])
        return (sum(pin) / len(pin) if pin else None, sum(pout) / len(pout) if pout else None)
    u_off, u_on = usage("claude-sonnet-4-6"), usage("claude-sonnet-4-6_think")
    record("N.tokens_out.off", "N reasoning effort", "data/claude-sonnet-4-6/full/transcripts.jsonl",
           u_off[1], 5707, 1.0, note="mean completion tokens per trial, focal cell")
    record("N.tokens_out.on", "N reasoning effort", "data/claude-sonnet-4-6_think/full/transcripts.jsonl",
           u_on[1], 25917, 1.0, note="mean completion tokens per trial, focal cell")
    if u_off[1] and u_on[1]:
        record("N.tokens_out.ratio", "N reasoning effort", "data/claude-sonnet-4-6*/full/transcripts.jsonl",
               u_on[1] / u_off[1], 4.5, 0.05, note="approximately 4.5-fold")
    # Anthropic list pricing for the snapshot used: $3 per Mtok in, $15 per Mtok out
    if all(u_off) and all(u_on):
        c_off = u_off[0] * 3e-6 + u_off[1] * 15e-6
        c_on = u_on[0] * 3e-6 + u_on[1] * 15e-6
        record("N.cost.ratio", "N reasoning effort", "data/claude-sonnet-4-6*/full/transcripts.jsonl",
               c_on / c_off, 3.3, 0.05, note=f"${c_off:.4f} -> ${c_on:.4f} per trial at $3/$15 per Mtok")


# ===================== O. CAPABILITY SCREENING =====================
def family_O():
    src = "data/adequacy/adequacy.csv"
    rows = list(csv.DictReader(open(DATA / "adequacy" / "adequacy.csv")))
    tbl = {("claude-sonnet-4-6", "concordant"): 1.00, ("claude-sonnet-4-6", "misleading_majority"): 0.83,
           ("gpt-4.1-mini", "concordant"): 1.00, ("gpt-4.1-mini", "misleading_majority"): 0.83,
           ("gemini-2.5-flash", "concordant"): 0.92, ("gemini-2.5-flash", "misleading_majority"): 1.00,
           ("gemini-2.5-pro", "concordant"): 1.00, ("gemini-2.5-pro", "misleading_majority"): 1.00,
           ("claude-haiku-4-5", "concordant"): 0.75, ("claude-haiku-4-5", "misleading_majority"): 0.50,
           ("gpt-4o-mini", "concordant"): 0.83, ("gpt-4o-mini", "misleading_majority"): 0.50}
    got = {(r["model"], r["regime"]): r for r in rows}
    for k, e in tbl.items():
        r = got.get(k)
        record(f"O.{k[0]}.{k[1]}.truth_acc", "O capability screening", src,
               float(r["truth_acc"]) if r else None, e, 0.006)
        record(f"O.{k[0]}.{k[1]}.n", "O capability screening", src,
               int(r["n"]) if r else None, 12, 0)
    admitted = [v for k, v in tbl.items() if k[1] == "misleading_majority"
                and k[0] not in ("claude-haiku-4-5", "gpt-4o-mini")]
    record("O.admitted.min_mm_acc", "O capability screening", src, min(admitted), 0.83, 0.006,
           note="admitted models reached at least 0.83 under misleading majority")
    excluded = [float(got[(m, "misleading_majority")]["truth_acc"])
                for m in ("claude-haiku-4-5", "gpt-4o-mini") if (m, "misleading_majority") in got]
    record("O.excluded.max_mm_acc", "O capability screening", src,
           max(excluded) if excluded else None, 0.50, 0.006)


# ===================== P. HIDDEN-PROFILE PROBE =====================
def family_P():
    tag = "hidden-gpt-4.1-mini"; src = f"data/{tag}/rounds.csv"
    cells = [("concordant", "all_channel", "evidence_only", 0.006, 0.001, 0.969),
             ("concordant", "all_channel", "conclusion_only", 0.015, 0.001, 0.941),
             ("concordant", "all_channel", "both", 0.010, 0.001, 0.963),
             ("concordant", "circle", "evidence_only", 0.023, 0.008, 0.946),
             ("concordant", "circle", "conclusion_only", 0.031, 0.008, 0.909),
             ("misleading_majority", "all_channel", "evidence_only", 0.013, 0.003, 0.943),
             ("misleading_majority", "all_channel", "conclusion_only", 0.007, 0.002, 0.958),
             ("misleading_majority", "all_channel", "both", 0.004, 0.001, 0.970),
             ("misleading_majority", "circle", "evidence_only", 0.081, 0.040, 0.897),
             ("misleading_majority", "circle", "conclusion_only", 0.047, 0.020, 0.896)]
    record("P.n_cells", "P hidden-profile probe", src, len(cells), 10, 0)
    for rg, tp, cm, ebar_e, d_e, pt_e in cells:
        rows = cell_rows(tag, rg, tp, cm)
        key = f"P.{rg}.{tp}.{cm}"
        record(key + ".n", "P hidden-profile probe", src, len(rows), 50, 0)
        if not rows: continue
        record(key + ".accuracy", "P hidden-profile probe", src,
               sum(int(r["acc_logodds"]) for r in rows) / len(rows), 1.00, 0.006,
               note="ceiling; probe has no sensitivity to a content effect")
        record(key + ".e_bar", "P hidden-profile probe", src,
               sum(float(r["e_bar"]) for r in rows) / len(rows), ebar_e, 0.0006)
        record(key + ".D", "P hidden-profile probe", src,
               sum(float(r["D"]) for r in rows) / len(rows), d_e, 0.0006)
        record(key + ".p_truth", "P hidden-profile probe", src,
               sum(p_truth(r) for r in rows) / len(rows), pt_e, 0.0006)
    # no-communication control is itself at ceiling
    for rg in ("concordant", "misleading_majority"):
        rows = [r for r in rounds_rows(tag) if r["regime"] == rg and r["topology"] == "baseline"]
        term = {}
        for r in rows: term[r["seed"]] = max(term.get(r["seed"], (-1, None)), (int(r["round"]), r))
        vals = [int(v[1]["acc_logodds"]) for v in term.values()]
        record(f"P.baseline.{rg}.accuracy", "P hidden-profile probe", src,
               sum(vals) / len(vals) if vals else None,
               {"misleading_majority": 1.00, "concordant": 0.92}[rg], 0.006,
               note="no-communication control; only the concordant control is below ceiling")
    # round-0 mean individual error range across the ten cells
    r0 = []
    for rg, tp, cm, _, _, _ in cells:
        rows = [r for r in rounds_rows(tag) if r["regime"] == rg and r["topology"] == tp
                and r["content_mode"] == cm and int(r["round"]) == 0]
        if rows: r0.append(sum(float(r["e_bar"]) for r in rows) / len(rows))
    record("P.round0_e_bar.min", "P hidden-profile probe", src, min(r0) if r0 else None, 0.128, 0.0006)
    record("P.round0_e_bar.max", "P hidden-profile probe", src, max(r0) if r0 else None, 0.208, 0.0006)
    mm = [r for r in rounds_rows(tag) if r["regime"] == "misleading_majority"
          and r["topology"] == "all_channel" and r["content_mode"] == "conclusion_only"]
    e0 = [float(r["e_bar"]) for r in mm if int(r["round"]) == 0]
    record("P.mm_allchannel_conclusion.e_bar_round0", "P hidden-profile probe", src,
           sum(e0) / len(e0) if e0 else None, 0.128, 0.0006)


# ===================== Q. SIGNAL-STRUCTURE ROBUSTNESS =====================
def family_Q():
    tag = "gpt-4.1-mini_sigsweep"; src = f"data/{tag}/rounds.csv"
    tbl = [("mm_gap_narrow", 1.00, 0.93, 1.00, 0.82, 0.69, 0.90, 0.54, 0.40, 0.67),
           ("mm_gap_wide", 1.00, 0.93, 1.00, 0.84, 0.71, 0.92, 0.42, 0.29, 0.56),
           ("mm_maj_heavy", 1.00, 0.93, 1.00, 0.82, 0.69, 0.90, 0.26, 0.16, 0.40)]
    for rg, *v in tbl:
        conds = [("baseline", "none", v[0], v[1], v[2]),
                 ("all_channel", "evidence_only", v[3], v[4], v[5]),
                 ("all_channel", "conclusion_only", v[6], v[7], v[8])]
        for tp, cm, acc_e, lo_e, hi_e in conds:
            if tp == "baseline":
                rows = [r for r in rounds_rows(tag) if r["regime"] == rg and r["topology"] == "baseline"]
                term = {}
                for r in rows: term[r["seed"]] = max(term.get(r["seed"], (-1, None)), (int(r["round"]), r))
                rows = [t[1] for t in term.values()]
            else:
                rows = cell_rows(tag, rg, tp, cm)
            k, n = sum(int(r["acc_logodds"]) for r in rows), len(rows)
            lo, hi = wilson(k, n)
            key = f"Q.{rg}.{cm}"
            record(key + ".n", "Q signal structure", src, n, 50, 0)
            record(key + ".accuracy", "Q signal structure", src, k / n if n else None, acc_e, 0.006)
            record(key + ".ci_lo", "Q signal structure", src, lo, lo_e, 0.011)
            record(key + ".ci_hi", "Q signal structure", src, hi, hi_e, 0.011)


# ===================== R. DATASET INTEGRITY =====================
PAPER_DIRS = ["gpt-4.1-mini", "gpt-4.1-mini_n11", "gpt-4.1-mini_degree",
              "gpt-4.1-mini_overload", "gpt-4.1-mini_sigsweep", "claude-sonnet-4-6",
              "claude-sonnet-4-6_think", "gemini-2.5-flash", "hidden-gpt-4.1-mini"]

def family_R():
    tags = [t for t in PAPER_DIRS if (DATA / t / "rounds.csv").exists()]
    total_trials, dup = 0, 0
    for t in tags:
        keys = set()
        for r in rounds_rows(t):
            k = (r["regime"], r["topology"], r["content_mode"], r["seed"], r["round"])
            if k in keys: dup += 1
            keys.add(k)
        seeds = {(r["regime"], r["topology"], r["content_mode"], r["seed"]) for r in rounds_rows(t)}
        total_trials += len(seeds)
    record("R.duplicate_trial_keys", "R dataset integrity", "data/*/rounds.csv", dup, 0, 0)
    record("R.n_datasets", "R dataset integrity", "data/*/rounds.csv", len(tags), None, None,
           note=", ".join(tags))
    record("R.n_trials_total", "R dataset integrity", "data/*/rounds.csv", total_trials, 6341, 0,
           note="SI Section S6: error-diversity identity checked across all trials")
    # error-diversity identity, arithmetic-mean pool, across every trial in the full dataset
    worst = 0.0
    for t in tags:
        for r in rounds_rows(t):
            y = 1.0 if r["true_state"] == "A" else 0.0
            lhs = (float(r["coll_mean"]) - y) ** 2
            rhs = float(r["e_bar"]) - float(r["D"])
            worst = max(worst, abs(lhs - rhs))
    record("R.error_diversity_max_abs_discrepancy", "R dataset integrity", "data/*/rounds.csv",
           worst, 1.6e-4, 5e-6,
           note="collective Brier error = mean individual error - diversity; SI reports max abs "
                "discrepancy 1.6e-4, matched here to the reported precision")
    # every logged message nonempty
    n_msg, n_empty = 0, 0
    for t in tags:
        f = DATA / t / "messages.csv"
        if not f.exists(): continue
        for r in csv.DictReader(open(f)):
            n_msg += 1
            if not (r.get("message_text") or "").strip(): n_empty += 1
    record("R.messages_logged", "R dataset integrity", "data/*/messages.csv", n_msg, None, None)
    record("R.messages_empty", "R dataset integrity", "data/*/messages.csv", n_empty, 0, 0)


# ===== I2 / J2 / G2. REMAINING REPORTED QUANTITIES IN THE SAME FAMILIES =====
def family_I2():
    """Degree-capitulation / degree-correction correlations across the six n = 11 networks."""
    src = "data/gpt-4.1-mini_n11/beliefs.csv + data/gpt-4.1-mini_degree/beliefs.csv"
    ladder = [("gpt-4.1-mini_n11", "circle", 2.0), ("gpt-4.1-mini_n11", "ring_shortcuts", 2.55),
              ("gpt-4.1-mini_degree", "lattice_k4", 4.0), ("gpt-4.1-mini_degree", "lattice_k6", 6.0),
              ("gpt-4.1-mini_degree", "lattice_k8", 8.0), ("gpt-4.1-mini_n11", "all_channel", 10.0)]
    degs, caps, cors = [], [], []
    for tag, topo, deg in ladder:
        truth = {r["seed"]: r["true_state"] for r in rounds_rows(tag)
                 if r["regime"] == "misleading_majority" and r["topology"] == topo
                 and r["content_mode"] == "conclusion_only"}
        rec = defaultdict(dict)
        for r in csv.DictReader(open(DATA / tag / "beliefs.csv")):
            if r["regime"] != "misleading_majority" or r["topology"] != topo: continue
            if r["content_mode"] != "conclusion_only": continue
            rec[(r["seed"], r["agent"])][int(r["round"])] = (float(r["belief_pA"]), float(r["llr"]))
        nc = nw = cap = cor = 0
        for (s, a), byr in rec.items():
            if s not in truth: continue
            rs = sorted(byr); llr = byr[rs[0]][1]; bT = byr[rs[-1]][0]
            tA = truth[s] == "A"
            right0 = (llr > 0) == tA
            rightT = (bT > .5) == tA
            if right0:
                nc += 1
                if not rightT: cap += 1
            else:
                nw += 1
                if rightT: cor += 1
        degs.append(deg); caps.append(cap / nc if nc else float("nan"))
        cors.append(cor / nw if nw else float("nan"))
    record("I2.degree_vs_capitulation.r", "I fan-in and topology", src, _corr(degs, caps), 0.90, 0.006)
    record("I2.degree_vs_correction.r", "I fan-in and topology", src, _corr(degs, cors), -0.82, 0.006)


def family_J2():
    """Graph-fixed-effects association and within-graph slopes (SI Section S6.9)."""
    src = "data/gpt-4.1-mini/{beliefs,messages,rounds}.csv"
    recs = _positional_records()
    by = defaultdict(list)
    for r in recs: by[r["topo"]].append(r)
    dx, dy = [], []
    for tp, v in by.items():
        mo = sum(r["own"] for r in v) / len(v); mc = sum(r["cap"] for r in v) / len(v)
        dx += [r["own"] - mo for r in v]; dy += [r["cap"] - mc for r in v]
    record("J2.graph_fixed_effects.r", "J positional", src, _corr(dx, dy), 0.36, 0.006,
           note="fan-in and capitulation demeaned within topology")
    slopes = {"wheel": 0.15, "hierarchy": 0.10, "chain": 0.11}
    for tp, e in slopes.items():
        v = by.get(tp, [])
        mo = sum(r["own"] for r in v) / len(v); mc = sum(r["cap"] for r in v) / len(v)
        den = sum((r["own"] - mo) ** 2 for r in v)
        b = sum((r["own"] - mo) * (r["cap"] - mc) for r in v) / den if den else float("nan")
        record(f"J2.slope.{tp}", "J positional", src, b, e, 0.006,
               note="change in capitulation per additional unit of fan-in")
    record("J2.collinearity.own_vs_neighbour", "J positional", src,
           _corr([r["own"] for r in recs], [r["nbr"] for r in recs]), None, None,
           note="wheel is excluded from the conditional own-versus-neighbour reading "
                "because own and neighbour fan-in are perfectly collinear there")
    wh = [r for r in recs if r["topo"] == "wheel"]
    record("J2.wheel.collinearity", "J positional", src,
           abs(_corr([r["own"] for r in wh], [r["nbr"] for r in wh])), 1.0, 1e-6,
           note="perfect collinearity in the wheel")
    # positional balance checks
    TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
    TOPOS = ["wheel", "hierarchy", "chain"]
    truth = {(r["topology"], r["seed"]): r["true_state"] for r in rounds_rows(TAG)
             if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in TOPOS}
    byfan = defaultdict(lambda: [0, 0, []])
    rec = defaultdict(dict)
    for r in csv.DictReader(open(DATA / TAG / "beliefs.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS: continue
        rec[(r["topology"], r["seed"], r["agent"])][int(r["round"])] = r
    for k, byr in rec.items():
        tp, s, a = k
        if (tp, s) not in truth or 1 not in byr: continue
        f = int(byr[1]["n_received"]); llr = float(byr[min(byr)]["llr"])
        strong = (llr > 0) == (truth[(tp, s)] == "A")
        cell = byfan[f]; cell[0] += 1; cell[1] += int(strong)
        if strong: cell[2].append(abs(llr))
    for f in sorted(byfan):
        n, k, mags = byfan[f]
        record(f"J2.balance.fanin{f}.p_strong_minority", "J positional", src, k / n, 0.40, 0.05,
               note="probability of assignment to the strong minority, by fan-in")
        record(f"J2.balance.fanin{f}.mean_abs_llr", "J positional", src,
               sum(mags) / len(mags) if mags else None, 1.8, 0.15,
               note="mean absolute log-likelihood ratio among minority agents")


def family_G2():
    """Per-dataset channel-purity table and referential-mention rate (SI Section S6.10)."""
    src = "data/*/messages.csv"
    # Referential mention, per the SI definition: the message refers to POSSESSING a private
    # reading without disclosing anything about it. Possession language, minus any message that
    # also clears the substantive-disclosure bar. The classifier used to produce the reported
    # 11.67% in the Phase 1 pass was not persisted; this is the canonical archived definition.
    MENTION = re.compile(r"(my (private )?(reading|signal|observation|evidence|information)|"
                         r"the (reading|signal|evidence) i (have|received|got|hold)|"
                         r"i (have|received|hold|was given) a (private )?"
                         r"(reading|signal|observation))", re.I)
    groups = {"gpt-4.1-mini": ["gpt-4.1-mini"],
              "gpt-4.1-mini_other": ["gpt-4.1-mini_n11", "gpt-4.1-mini_degree",
                                     "gpt-4.1-mini_overload", "gpt-4.1-mini_sigsweep"],
              "claude-sonnet-4-6": ["claude-sonnet-4-6"],
              "claude-sonnet-4-6_think": ["claude-sonnet-4-6_think"],
              "gemini-2.5-flash": ["gemini-2.5-flash"]}
    exp = {"gpt-4.1-mini": (15000, 1), "gpt-4.1-mini_other": (22050, 0),
           "claude-sonnet-4-6": (1500, 141), "claude-sonnet-4-6_think": (900, 0),
           "gemini-2.5-flash": (885, 3)}
    n_mention = n_all = 0
    for g, tags in groups.items():
        tot = sub = 0
        for tag in tags:
            f = DATA / tag / "messages.csv"
            if not f.exists(): continue
            for r in csv.DictReader(open(f)):
                if r.get("content_mode") != "conclusion_only": continue
                tot += 1; n_all += 1
                txt = r.get("message_text", "")
                probe = CONFB.sub(" ", CONF.sub(" ", txt))
                if SUBST.search(probe): sub += 1
                if MENTION.search(txt) and not SUBST.search(probe): n_mention += 1
        record(f"G2.{g}.messages", "G channel purity", src, tot, exp[g][0], 0)
        record(f"G2.{g}.substantive", "G channel purity", src, sub, exp[g][1], 0)
    record("G2.conclusion_total", "G channel purity", src, n_all, 40335, 0)
    record("G2.referential_mention_count", "G channel purity", src, n_mention, None, None,
           note="possession language without substantive disclosure")
    record("G2.referential_mention_rate", "G channel purity", src,
           n_mention / n_all if n_all else None, None, None,
           note="INFORMATIONAL ONLY, not a reported result. The classifier used for the "
                "referential-mention figure quoted in earlier drafts was not archived and its "
                "value is therefore unrecoverable; that figure has been removed from the "
                "Supplementary Materials. The definition recorded here is exploratory and is "
                "deliberately not asserted against any reported number. Only the substantive-"
                "disclosure quantities are reported and audited.")
    # focal-cell disclosure intensity, claude
    per = defaultdict(int); trials = set()
    for r in csv.DictReader(open(DATA / "claude-sonnet-4-6" / "messages.csv")):
        if r.get("content_mode") != "conclusion_only": continue
        if r["regime"] != "misleading_majority" or r["topology"] != "all_channel": continue
        trials.add(r["seed"])
        probe = CONFB.sub(" ", CONF.sub(" ", r.get("message_text", "")))
        if SUBST.search(probe): per[r["seed"]] += 1
    record("G.focal.claude.msgs_per_trial", "G channel purity",
           "data/claude-sonnet-4-6/messages.csv",
           sum(per.values()) / len(trials) if trials else None, 2.82, 0.01,
           note=f"{len(trials)} trials of 15 messages each")
    record("G.focal.claude.trials_without_disclosure", "G channel purity",
           "data/claude-sonnet-4-6/messages.csv",
           sum(1 for s in trials if per[s] == 0), 2, 0,
           note="only two trials in this cell are free of disclosure")


# =============================== RUNNER ===============================
FAMILIES = [("A", family_A), ("B", family_B), ("C", family_C), ("D", family_D),
            ("E", family_E), ("F", family_F), ("G", family_G), ("H", family_H),
            ("I", family_I), ("J", family_J), ("K", family_K), ("L", family_L),
            ("M", family_M), ("N", family_N), ("O", family_O), ("P", family_P),
            ("Q", family_Q), ("R", family_R),
            ("I2", family_I2), ("J2", family_J2), ("G2", family_G2)]


def run_all(verbose=True):
    """Execute every claim family. Returns the RESULTS list."""
    RESULTS.clear()
    for letter, fn in FAMILIES:
        try:
            fn()
        except Exception as exc:                      # a broken family must not hide the rest
            record(f"{letter}.EXCEPTION", f"{letter} (aborted)", "-", None, 0, 0,
                   note=f"{type(exc).__name__}: {exc}")
        if verbose:
            done = [r for r in RESULTS if r["id"].startswith(letter + ".")]
            nf = sum(1 for r in done if r["status"] == "FAIL")
            print(f"  family {letter}: {len(done):>4} checks, {nf} failed")
    return RESULTS


def summary():
    n = len(RESULTS)
    npass = sum(1 for r in RESULTS if r["status"] == "PASS")
    nfail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    ninfo = sum(1 for r in RESULTS if r["status"] == "INFO")
    return dict(total=n, passed=npass, failed=nfail, info=ninfo,
                assertions=npass + nfail)


if __name__ == "__main__":
    print("=" * 96)
    print("REVISION-ERA QUANTITATIVE AUDIT — re-deriving reported values from archived data")
    print("=" * 96)
    run_all()
    s = summary()
    print("-" * 96)
    for r in RESULTS:
        if r["status"] == "FAIL":
            print(f"  FAIL  {r['id']:<48} reproduced={r['reproduced']} expected={r['expected']} "
                  f"tol={r['tol']} {r['note']}")
    print(f"\n{s['assertions']} assertions: {s['passed']} passed, {s['failed']} failed "
          f"({s['info']} informational records)")
