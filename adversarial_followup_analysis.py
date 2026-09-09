"""Existing-data follow-up analysis adjudicating the adversarial-review vulnerabilities.

No new experiments, no API calls. Everything is recomputed from code/data/<tag>/{rounds,beliefs,messages}.csv.

DESIGN STRUCTURE (established by inspection, documented here because it constrains estimation):

  gpt-4.1-mini_n11      circle(deg 2), ring_shortcuts(deg 2.55), all_channel(deg 10)
                        x {misleading_majority, concordant} x {evidence_only, conclusion_only, both}
                        N=50 per cell. Seeds are SHARED across content conditions AND across
                        topologies, so these cells are fully PAIRED by task seed.
  gpt-4.1-mini_degree   lattice_k4/k6/k8 x both regimes x conclusion_only ONLY, N=50.
  gpt-4.1-mini_overload lattice_k4/k6/k8 x concordant x evidence_only ONLY, N=30.

  CONSEQUENCE: the content contrast (evidence vs conclusion) exists ONLY at degrees 2, 2.55
  and 10. At degrees 4, 6 and 8 the two content conditions were never run in the same regime
  with matched seeds, so no content contrast is estimable there. Any "content x fan-in"
  statement is therefore based on three degree levels, not six.

ESTIMANDS
  Collective accuracy   = terminal (round 3) acc_logodds, i.e. whether the log-pool belief
                          falls on the correct side of 0.5.
  Risk difference (RD)  = accuracy(evidence) - accuracy(conclusion), signed consistently.
  B20                   = 1 if, within a trial, >=1 agent reports p<0.20 AND >=1 reports p>0.80.

UNCERTAINTY
  Resampling unit is the task seed throughout. Paired cells use a paired seed bootstrap
  (resample seeds with replacement, recompute the contrast). Unpaired comparisons use an
  independent seed bootstrap in each arm. Percentile intervals, 10,000 draws, seed 20260909.
  Agent-level analyses cluster the bootstrap on the task seed.

Run: /usr/bin/python3 v6/adversarial_followup_analysis.py
"""
import csv, math, json, collections, random
from pathlib import Path

# Portable: works whether this file sits in the repository root next to data/, or one
# directory above it (the working-copy layout in which it was first written).
HERE = Path(__file__).parent
DATA = HERE / "data" if (HERE / "data").is_dir() else HERE.parent / "code" / "data"
OUTDIR = DATA.parent
B = 10000
SEED = 20260909
DEG = {"circle": 2.0, "ring_shortcuts": 2.55, "lattice_k4": 4.0,
       "lattice_k6": 6.0, "lattice_k8": 8.0, "all_channel": 10.0}
OUT = []
def w(s=""): OUT.append(s); print(s)


def rounds(tag):
    f = DATA / tag / "rounds.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []


def cell(tag, regime, topo, content, rnd=3):
    """seed -> accuracy (log pool), terminal round."""
    return {r["seed"]: int(r["acc_logodds"]) for r in rounds(tag)
            if r["regime"] == regime and r["topology"] == topo
            and r["content_mode"] == content and int(r["round"]) == rnd}


def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def paired_rd(a, b, rng):
    """a, b: seed->0/1 dicts sharing seeds. Returns RD=mean(a)-mean(b) and bootstrap CI."""
    ks = sorted(set(a) & set(b))
    da = [a[k] for k in ks]; db = [b[k] for k in ks]
    rd = sum(da) / len(ks) - sum(db) / len(ks)
    draws = []
    n = len(ks)
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(da[i] for i in idx) / n - sum(db[i] for i in idx) / n)
    return rd, pct(draws, .025), pct(draws, .975), n


def unpaired_rd(a, b, rng):
    da = list(a.values()); db = list(b.values())
    rd = sum(da) / len(da) - sum(db) / len(db)
    draws = []
    for _ in range(B):
        x = [da[rng.randrange(len(da))] for _ in da]
        y = [db[rng.randrange(len(db))] for _ in db]
        draws.append(sum(x) / len(x) - sum(y) / len(y))
    return rd, pct(draws, .025), pct(draws, .975), (len(da), len(db))


# ============================== A. CONTENT x FAN-IN ==============================
def section_A():
    rng = random.Random(SEED)
    w("\n" + "=" * 100)
    w("A. CONTENT x FAN-IN")
    w("=" * 100)
    w("\nContent contrast is estimable only at degrees 2, 2.55 and 10 (see module docstring).")
    w("Degrees 4, 6 and 8 have only one content condition per regime and are shown for context.")
    rows = []
    for regime in ("misleading_majority", "concordant"):
        w(f"\n--- {regime} , n = 11 , terminal accuracy (log pool) ---")
        w(f"  {'deg':>5} {'topology':<16}{'N':>5}{'evid':>7}{'concl':>7}{'RD(e-c)':>9}{'95% CI':>18}  pairing")
        for tag, topo in (("gpt-4.1-mini_n11", "circle"), ("gpt-4.1-mini_n11", "ring_shortcuts"),
                          ("gpt-4.1-mini_overload", "lattice_k4"), ("gpt-4.1-mini_overload", "lattice_k6"),
                          ("gpt-4.1-mini_overload", "lattice_k8"), ("gpt-4.1-mini_n11", "all_channel")):
            ev = cell(tag, regime, topo, "evidence_only")
            if not ev and topo.startswith("lattice"):
                ev = cell("gpt-4.1-mini_overload", regime, topo, "evidence_only")
            cn = cell("gpt-4.1-mini_n11", regime, topo, "conclusion_only") or \
                 cell("gpt-4.1-mini_degree", regime, topo, "conclusion_only")
            if not ev and not cn: continue
            ea = sum(ev.values()) / len(ev) if ev else float("nan")
            ca = sum(cn.values()) / len(cn) if cn else float("nan")
            if ev and cn:
                shared = set(ev) & set(cn)
                if len(shared) == len(ev) == len(cn):
                    rd, lo, hi, n = paired_rd(ev, cn, rng); tagp = "paired by seed"
                else:
                    rd, lo, hi, n = unpaired_rd(ev, cn, rng); tagp = f"unpaired {n}"
                rows.append(dict(regime=regime, degree=DEG[topo], topology=topo,
                                 n_evid=len(ev), n_concl=len(cn), acc_evid=ea, acc_concl=ca,
                                 rd=rd, lo=lo, hi=hi, pairing=tagp))
                w(f"  {DEG[topo]:>5.2f} {topo:<16}{len(ev):>5}{ea:>7.2f}{ca:>7.2f}{rd:>+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}  {tagp}")
            else:
                which = "evidence only" if ev else "conclusion only"
                got = ea if ev else ca
                w(f"  {DEG[topo]:>5.2f} {topo:<16}{len(ev) or len(cn):>5}{got:>7.2f}{'--':>7}{'--':>9}{'--':>18}  {which}, no contrast")

        # permutation test of content x fan-in across the three paired degree levels
        paired = [(t, tp) for t, tp in (("gpt-4.1-mini_n11", "circle"),
                                        ("gpt-4.1-mini_n11", "ring_shortcuts"),
                                        ("gpt-4.1-mini_n11", "all_channel"))]
        cells = []
        for t, tp in paired:
            ev = cell(t, regime, tp, "evidence_only"); cn = cell(t, regime, tp, "conclusion_only")
            ks = sorted(set(ev) & set(cn))
            cells.append((tp, [(ev[k], cn[k]) for k in ks]))
        def rd_of(pairs, flips=None):
            if flips is None: return sum(a - b for a, b in pairs) / len(pairs)
            return sum((b - a) if f else (a - b) for (a, b), f in zip(pairs, flips)) / len(pairs)
        obs = rd_of(cells[-1][1]) - (rd_of(cells[0][1]) + rd_of(cells[1][1])) / 2
        rng2 = random.Random(SEED + 1)
        null = []
        for _ in range(B):
            vals = []
            for _, pairs in cells:
                flips = [rng2.random() < .5 for _ in pairs]
                vals.append(rd_of(pairs, flips))
            null.append(vals[-1] - (vals[0] + vals[1]) / 2)
        p = (sum(1 for x in null if abs(x) >= abs(obs)) + 1) / (B + 1)
        w(f"\n  Permutation test, content x fan-in (degree 10 vs mean of degrees 2 and 2.55):")
        w(f"    statistic RD(deg10) - mean RD(low) = {obs:+.3f}")
        w(f"    content labels permuted within seed, {B:,} draws, two-sided p = {p:.4f}")
    return rows


# ============================== B. BOTH-EXTREMES ==============================
def beliefs(tag, regime, topo, content, rnd):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(DATA / tag / "beliefs.csv")):
        if r["regime"] != regime or r["topology"] != topo: continue
        if r["content_mode"] != content or int(r["round"]) != rnd: continue
        out[r["seed"]].append(float(r["belief_pA"]))
    return out


def section_B():
    rng = random.Random(SEED + 2)
    w("\n" + "=" * 100)
    w("B. BOTH-EXTREMES (BIPOLAR) BELIEF CONFIGURATION")
    w("=" * 100)
    w("\nB(lo,hi) = 1 if within a trial at least one agent reports p<lo and at least one reports p>hi.")
    w("Descriptive only. Not an identified polarization mechanism.")
    res = []
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6"):
        w(f"\n--- {tag} , misleading-majority , all-channel , n = 5 ---")
        for lo_t, hi_t in ((.10, .90), (.20, .80), (.25, .75)):
            w(f"\n  threshold {lo_t:.2f} / {hi_t:.2f}")
            w(f"    {'condition':<22}{'N':>5}{'B':>8}{'95% CI':>18}")
            arms = {}
            for lab, content, rnd in (("round 0 (pre-comm)", "conclusion_only", 0),
                                      ("evidence terminal", "evidence_only", 3),
                                      ("conclusion terminal", "conclusion_only", 3)):
                bl = beliefs(tag, "misleading_majority", "all_channel", content, rnd)
                ks = sorted(bl)
                ind = {k: int(any(x < lo_t for x in bl[k]) and any(x > hi_t for x in bl[k])) for k in ks}
                arms[lab] = ind
                v = list(ind.values()); m = sum(v) / len(v)
                draws = [sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(B)]
                w(f"    {lab:<22}{len(v):>5}{m:>8.2f}{f'[{pct(draws,.025):.2f}, {pct(draws,.975):.2f}]':>18}")
                res.append(dict(family=tag, thresh=f"{lo_t}/{hi_t}", condition=lab,
                                n=len(v), prevalence=m, lo=pct(draws, .025), hi=pct(draws, .975)))
            for a, b in (("conclusion terminal", "evidence terminal"),
                         ("conclusion terminal", "round 0 (pre-comm)")):
                rd, lo, hi, n = paired_rd(arms[a], arms[b], rng)
                w(f"    {a} minus {b}: {rd:+.2f}  [{lo:+.2f}, {hi:+.2f}]  (paired, n={n})")
                res.append(dict(family=tag, thresh=f"{lo_t}/{hi_t}", condition=f"{a} - {b}",
                                n=n, prevalence=rd, lo=lo, hi=hi))
    # association with collective correctness, conclusion condition, 0.20/0.80
    w("\n--- B20 by terminal collective outcome (conclusion sharing; descriptive, post-treatment) ---")
    w(f"  {'family':<20}{'outcome':<12}{'N':>5}{'B20':>8}")
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6"):
        bl = beliefs(tag, "misleading_majority", "all_channel", "conclusion_only", 3)
        acc = cell(tag, "misleading_majority", "all_channel", "conclusion_only")
        grp = collections.defaultdict(list)
        for k in bl:
            if k not in acc: continue
            b20 = int(any(x < .20 for x in bl[k]) and any(x > .80 for x in bl[k]))
            grp["correct" if acc[k] else "incorrect"].append(b20)
        for o in ("correct", "incorrect"):
            v = grp[o]
            if v: w(f"  {tag:<20}{o:<12}{len(v):>5}{sum(v)/len(v):>8.2f}")
    # wrong-side depth, log vs arithmetic
    w("\n--- Wrong-side depth of the collective belief, log vs arithmetic pooling ---")
    w("  (conclusion sharing, terminal; 'depth' = mean P(true state) among trials ending incorrect)")
    w(f"  {'family':<20}{'pool':<12}{'wrong-side rate':>17}{'depth':>8}")
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6"):
        rs = [r for r in rounds(tag) if r["regime"] == "misleading_majority"
              and r["topology"] == "all_channel" and r["content_mode"] == "conclusion_only"
              and int(r["round"]) == 3]
        for pool, key in (("logarithmic", "coll_logodds"), ("arithmetic", "coll_mean")):
            q = [(float(r[key]) if r["true_state"] == "A" else 1 - float(r[key])) for r in rs]
            wrong = [x for x in q if x < .5]
            w(f"  {tag:<20}{pool:<12}{len(wrong)/len(q):>17.2f}{sum(wrong)/len(wrong):>8.3f}")
    return res


# ==================== C. FAN-IN vs PATH-LENGTH CONTRAST PRECISION ====================
def section_C():
    rng = random.Random(SEED + 3)
    w("\n" + "=" * 100)
    w("C. PRECISION OF THE MATCHED FAN-IN vs PATH-LENGTH CONTRASTS")
    w("=" * 100)
    w("\nAll cells: misleading-majority, conclusion sharing, n = 11, terminal accuracy (log pool).")
    w("No equivalence test is reported: no equivalence margin was defined before seeing these results.")
    rows = []
    contrasts = [
        ("all_channel (deg 10, diam 1)", "gpt-4.1-mini_n11", "all_channel",
         "lattice_k8 (deg 8, diam 2)", "gpt-4.1-mini_degree", "lattice_k8",
         "degree differs by 2, diameter differs by 1"),
        ("ring_shortcuts (deg 2.55, diam 3)", "gpt-4.1-mini_n11", "ring_shortcuts",
         "circle (deg 2, diam 5)", "gpt-4.1-mini_n11", "circle",
         "degree nearly equal, diameter differs by 2"),
        ("all_channel (deg 10, diam 1)", "gpt-4.1-mini_n11", "all_channel",
         "ring_shortcuts (deg 2.55, diam 3)", "gpt-4.1-mini_n11", "ring_shortcuts",
         "degree differs by 7.45, diameter differs by 2"),
    ]
    for la, ta, pa, lb, tb, pb, note in contrasts:
        A = cell(ta, "misleading_majority", pa, "conclusion_only")
        Bc = cell(tb, "misleading_majority", pb, "conclusion_only")
        if not A or not Bc: continue
        shared = set(A) & set(Bc)
        if len(shared) == len(A) == len(Bc):
            d, lo, hi, n = paired_rd(A, Bc, rng); pairing = f"paired by seed, n={n}"
        else:
            d, lo, hi, n = unpaired_rd(A, Bc, rng); pairing = f"unpaired, n={n}"
        w(f"\n  {la}  vs  {lb}")
        w(f"    {note}")
        w(f"    accuracy {sum(A.values())/len(A):.2f} vs {sum(Bc.values())/len(Bc):.2f}"
          f"   difference {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {pairing}")
        width = hi - lo
        w(f"    CI width {width:.3f}. Effects up to {max(abs(lo),abs(hi)):.2f} in absolute accuracy remain compatible with these data.")
        rows.append(dict(contrast=f"{la} vs {lb}", diff=d, lo=lo, hi=hi, pairing=pairing))
    return rows


# ==================== D. NEIGHBOUR FAN-IN, CLUSTER-AWARE ====================
def section_D():
    rng = random.Random(SEED + 4)
    w("\n" + "=" * 100)
    w("D. NEIGHBOUR FAN-IN, WITH SEED-CLUSTERED UNCERTAINTY")
    w("=" * 100)
    TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
    TOPOS = ["wheel", "hierarchy", "chain"]
    truth = {(r["topology"], r["seed"]): r["true_state"] for r in rounds(TAG)
             if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in TOPOS}
    rec = collections.defaultdict(dict); fanin = {}
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
    recs = []
    for k, byr in rec.items():
        tp, s, a = k
        if (tp, s) not in truth or k not in fanin or k not in nbrs: continue
        rs = sorted(byr); llr = byr[rs[0]][1]; bT = byr[rs[-1]][0]
        truth_A = truth[(tp, s)] == "A"
        if (llr > 0) != truth_A: continue          # strong-minority agents only
        nf = [fanin[(tp, s, x)] for x in nbrs[k] if (tp, s, x) in fanin]
        if not nf: continue
        recs.append(dict(topo=tp, seed=s, own=fanin[k], nbr=sum(nf)/len(nf),
                         cap=1 if ((bT > .5) != truth_A) else 0))
    def corr(xs, ys):
        n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
        return num/den if den else float("nan")
    def partial(xs, ys, zs):
        def resid(v, z):
            mz, mv = sum(z)/len(z), sum(v)/len(v)
            den = sum((a-mz)**2 for a in z)
            b = sum((a-mz)*(c-mv) for a, c in zip(z, v))/den if den else 0.0
            return [c-(mv+b*(a-mz)) for a, c in zip(z, v)]
        return corr(resid(xs, zs), resid(ys, zs))
    byseed = collections.defaultdict(list)
    for r in recs: byseed[(r["topo"], r["seed"])].append(r)
    keys = sorted(byseed)
    def stats(sample):
        own=[r["own"] for r in sample]; nbr=[r["nbr"] for r in sample]; cap=[r["cap"] for r in sample]
        return (corr(own,cap), corr(nbr,cap), partial(nbr,cap,own), partial(own,cap,nbr))
    obs = stats(recs)
    draws = [[],[],[],[]]
    for _ in range(B):
        samp = []
        for _ in keys: samp += byseed[keys[rng.randrange(len(keys))]]
        try:
            v = stats(samp)
            for i in range(4):
                if v[i] == v[i]: draws[i].append(v[i])
        except Exception: pass
    labs = ["own fan-in ~ capitulation", "neighbour fan-in ~ capitulation",
            "neighbour ~ capitulation | own", "own ~ capitulation | neighbour"]
    w(f"\n  strong-minority agent-trials n = {len(recs)} in {len(keys)} trial-graphs; bootstrap clustered on trial")
    w(f"  {'quantity':<38}{'r':>8}{'95% CI':>20}")
    rows=[]
    for i, l in enumerate(labs):
        lo, hi = pct(draws[i], .025), pct(draws[i], .975)
        w(f"  {l:<38}{obs[i]:>+8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}")
        rows.append(dict(quantity=l, r=obs[i], lo=lo, hi=hi, n_agents=len(recs), n_trials=len(keys)))
    # wheel hub vs spoke
    wh = [r for r in recs if r["topo"] == "wheel"]
    if wh:
        mx = max(r["own"] for r in wh)
        hub = [r["cap"] for r in wh if r["own"] == mx]; spk = [r["cap"] for r in wh if r["own"] < mx]
        d = sum(hub)/len(hub) - sum(spk)/len(spk)
        wseed = collections.defaultdict(list)
        for r in wh: wseed[r["seed"]].append(r)
        wk = sorted(wseed); dd = []
        for _ in range(B):
            s = []
            for _ in wk: s += wseed[wk[rng.randrange(len(wk))]]
            h = [r["cap"] for r in s if r["own"] == mx]; p = [r["cap"] for r in s if r["own"] < mx]
            if h and p: dd.append(sum(h)/len(h) - sum(p)/len(p))
        w(f"\n  wheel: hub capitulation {sum(hub)/len(hub):.2f} (n={len(hub)}) vs spoke {sum(spk)/len(spk):.2f} (n={len(spk)})")
        w(f"    difference {d:+.3f}  95% CI [{pct(dd,.025):+.3f}, {pct(dd,.975):+.3f}]  (bootstrap clustered on trial)")
        rows.append(dict(quantity="wheel hub - spoke capitulation", r=d, lo=pct(dd,.025), hi=pct(dd,.975),
                         n_agents=len(hub)+len(spk), n_trials=len(wk)))
    return rows


# ==================== E. MESSAGE VOLUME AT HIGH FAN-IN ====================
def section_E():
    import json as _json
    w("\n" + "=" * 100)
    w("E. INCOMING MESSAGE VOLUME AT THE HIGHEST FAN-IN")
    w("=" * 100)
    f = DATA / "gpt-4.1-mini_n11" / "full" / "transcripts.jsonl"
    agg = collections.defaultdict(lambda: [0, 0, 0])   # senders, chars, obs
    for line in open(f):
        o = _json.loads(line)
        if o["topology"] != "all_channel": continue
        for rd in o["rounds"]:
            if rd["round"] == 0: continue
            for ag in rd["agents"]:
                rec = ag.get("received") or []
                if not rec: continue
                a = agg[(o["regime"], o["content_mode"])]
                a[0] += len(rec); a[1] += sum(len(x["text"]) for x in rec); a[2] += 1
    w(f"\n  degree 10 (all-channel, n = 11), per receiving agent-round")
    w(f"  {'regime':<22}{'content':<18}{'obs':>7}{'senders':>9}{'chars in':>10}{'chars/sender':>14}")
    rows=[]
    for k in sorted(agg):
        s, c, n = agg[k]
        w(f"  {k[0]:<22}{k[1]:<18}{n:>7}{s/n:>9.1f}{c/n:>10.0f}{c/s:>14.0f}")
        rows.append(dict(regime=k[0], content=k[1], obs=n, senders=s/n, chars=c/n, per_sender=c/s))
    ev = [r for r in rows if r["content"] == "evidence_only"]
    cn = [r for r in rows if r["content"] == "conclusion_only"]
    if ev and cn:
        ratio = (sum(r["per_sender"] for r in ev)/len(ev)) / (sum(r["per_sender"] for r in cn)/len(cn))
        w(f"\n  evidence : conclusion characters per sender at degree 10 = {ratio:.2f}x")
    return rows


# ==================== F. CROSS-FAMILY CALIBRATION ====================
def section_F():
    w("\n" + "=" * 100)
    w("F. CROSS-FAMILY CALIBRATION (factual summary, no new comparison)")
    w("=" * 100)
    import re
    CONF = re.compile(r"(with|at)\s+(about|approximately|around|roughly|~)?\s*\d{1,3}(\.\d+)?\s*%\s*(confidence|certainty|confident|sure|probability)?", re.I)
    CONFB = re.compile(r"\b\d{1,3}(\.\d+)?\s*%", re.I)
    SUBST = re.compile(r"(\br\d\b|occurs with probability|if the (true )?state (is|were)|under state [ab]\b|"
                       r"given state [ab]\b|likelihood ratio|\bllr\b|diagnostic|probability 0\.\d+|\b0\.\d{2}\b)", re.I)
    w(f"\n  {'family':<22}{'focal-cell concl msgs':>23}{'substantive disclosure':>24}{'content effect (log pool)':>27}")
    rows=[]
    for tag, N in (("gpt-4.1-mini", 100), ("claude-sonnet-4-6", 50), ("gemini-2.5-flash", 30)):
        tot = leak = 0
        p = DATA / tag / "messages.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                if r.get("content_mode") != "conclusion_only": continue
                if r["regime"] != "misleading_majority" or r["topology"] != "all_channel": continue
                tot += 1
                probe = CONFB.sub(" ", CONF.sub(" ", r.get("message_text", "")))
                if SUBST.search(probe): leak += 1
        ev = cell(tag, "misleading_majority", "all_channel", "evidence_only")
        cn = cell(tag, "misleading_majority", "all_channel", "conclusion_only")
        eff = (sum(ev.values())/len(ev) - sum(cn.values())/len(cn)) if ev and cn else float("nan")
        w(f"  {tag:<22}{tot:>23}{f'{leak} ({100*leak/max(tot,1):.1f}%)':>24}{eff:>+27.2f}")
        rows.append(dict(family=tag, focal_msgs=tot, disclosures=leak,
                         rate=leak/max(tot,1), content_effect=eff))
    return rows


if __name__ == "__main__":
    A = section_A(); Bres = section_B(); C = section_C(); D = section_D(); E = section_E(); F = section_F()
    (OUTDIR / "adversarial_followup_results.md").write_text(
        "# Adversarial follow-up analysis — raw output\n\n"
        "Generated by `v6/adversarial_followup_analysis.py`. Existing data only; no new runs.\n\n```\n"
        + "\n".join(OUT) + "\n```\n")
    import csv as _csv
    with open(OUTDIR / "adversarial_followup_estimates.csv", "w", newline="") as fh:
        wr = _csv.writer(fh); wr.writerow(["section", "key", "estimate", "ci_lo", "ci_hi", "detail"])
        for r in A: wr.writerow(["A", f"{r['regime']} deg{r['degree']}", f"{r['rd']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["pairing"]])
        for r in Bres: wr.writerow(["B", f"{r['family']} {r['thresh']} {r['condition']}", f"{r['prevalence']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", f"n={r['n']}"])
        for r in C: wr.writerow(["C", r["contrast"], f"{r['diff']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["pairing"]])
        for r in D: wr.writerow(["D", r["quantity"], f"{r['r']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", f"agents={r['n_agents']} trials={r['n_trials']}"])
        for r in E: wr.writerow(["E", f"{r['regime']} {r['content']}", f"{r['per_sender']:.1f}", "", "", f"chars={r['chars']:.0f} senders={r['senders']:.1f}"])
        for r in F: wr.writerow(["F", r["family"], f"{r['content_effect']:.4f}", "", "", f"disclosure={r['rate']:.4f}"])
    print("\n\nwrote v6/Adversarial Follow-up Analysis Results — Nexus R&R (2026-09-09).md and .csv")
