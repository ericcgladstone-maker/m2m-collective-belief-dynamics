"""Closing audit against R1's letter. Read-only checks on existing records:
September transcripts (M2M_DATA), the 2,048-token readout replay (cap2048_readouts.jsonl),
and the full reruns (rerun_trials.jsonl), both in M2M_REVISION_DATA. No API calls.

Run:  python3 close_audit.py > close_audit_output.txt
"""
import collections, csv, json, math, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_readout as RR  # noqa: E402
sys.path.insert(0, str(RR.GITHUB))
import audit_revision as AR, check_leakage as CL  # noqa: E402
AR.DATA = RR.DATA

R1CELLS = ["gpt5_mm_ev", "gpt5_mm_co", "cl5_mm_ev", "cl5_mm_co",
           "gpt11_cc_ev", "gpt11_cc_co", "gpt11_mm_ev", "gpt11_mm_co"]
TOPO = ["gpt5_mm_co_hier", "gpt5_mm_co_chain", "gpt5_mm_co_wheel"]
SEPT = {c: RR.load_trials(c) for c in R1CELLS + TOPO}
REP = [json.loads(l) for l in open(RR.REV / "cap2048_readouts.jsonl")]
RER = [json.loads(l) for l in open(RR.REV / "rerun_trials.jsonl")]
capstop = lambda r: r["stop_reason"] in ("length", "max_tokens")

def logpool(bs):
    e = 1e-6
    s = sum(math.log(min(1 - e, max(e, b)) / (1 - min(1 - e, max(e, b)))) for b in bs) / len(bs)
    return 1 / (1 + math.exp(-s))

def H(t): print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)

# ------------------------------------------------------------------ A. degree-10 concordant evidence messages
H("A. Degree-10 concordant evidence messages (September): does any message misstate which state its reading favors?")
PA = re.compile(r"(\d?\.\d+)\s*(?:if|under|when|given)\s+(?:the\s+)?(?:true\s+)?(?:hidden\s+)?state\s+(?:is\s+)?A\b", re.I)
PB = re.compile(r"(\d?\.\d+)\s*(?:if|under|when|given)\s+(?:the\s+)?(?:true\s+)?(?:hidden\s+)?state\s+(?:is\s+)?B\b", re.I)
QA = re.compile(r"P\(\s*r\d+\s*\|\s*A\s*\)\s*=\s*(\d?\.\d+)", re.I)   # P(r3|A) = 0.17 notation
QB = re.compile(r"P\(\s*r\d+\s*\|\s*B\s*\)\s*=\s*(\d?\.\d+)", re.I)
c = collections.Counter(); deliv = collections.Counter()
for t in SEPT["gpt11_cc_ev"]:
    meta = {m["id"]: m for m in t["agents_meta"]}
    rounds = t["rounds"]
    for rd in rounds:
        for a in rd["agents"]:
            if rd["round"] >= 1:
                prev = {x["id"]: x["message"] for x in rounds[rd["round"] - 1]["agents"]}
                for d in a["received"]:
                    deliv["deliveries"] += 1; deliv["mismatch"] += prev.get(d["src"]) != d["text"]
            m = a.get("message")
            if not m: continue
            c["messages"] += 1
            pa, pb = PA.search(m) or QA.search(m), PB.search(m) or QB.search(m)
            if not (pa and pb): c["pair not parsed"] += 1; continue
            va, vb = float(pa.group(1)), float(pb.group(1)); s = meta[a["id"]]
            c["pair parsed"] += 1
            c["direction matches own reading"] += (va > vb) == (s["p_a"] > s["p_b"])
            c["values match own reading to 2 dp"] += abs(va - s["p_a"]) < 0.0051 and abs(vb - s["p_b"]) < 0.0051
            c["direction favors truth"] += (va > vb) == (t["true_state"] == "A")
print(dict(c)); print("deliveries equal to the sender's previous-round message:", dict(deliv))
print("Readout origin of wrong-side beliefs: see the pilot and replay memos (generated text truncated at 320 tokens,")
print("then misread by the last-number fallback); replay legacy and strict parses agree on complete text.")

# ------------------------------------------------------------------ B. exact 0, 0.5, 1 by cell and round
H("B. Beliefs exactly 0, 0.5, 1 by cell and round (counts of agent-readouts; denominators in n)")
def ex_rows(label, rows):
    print(f"\n{label}")
    print(f"  {'cell':18}{'round':>6}{'n':>6}{'=0':>6}{'=0.5':>6}{'=1':>6}")
    for k in sorted(rows):
        v = rows[k]; print(f"  {k[0]:18}{k[1]:6d}{len(v):6d}{sum(x == 0.0 for x in v):6d}{sum(x == 0.5 for x in v):6d}{sum(x == 1.0 for x in v):6d}")
rows = collections.defaultdict(list)
for cell, ts in SEPT.items():
    for t in ts:
        for rd in t["rounds"]:
            for a in rd["agents"]: rows[(cell, rd["round"])].append(float(a["belief"]))
ex_rows("September recorded beliefs, unrounded values from transcripts.jsonl", rows)
rows = collections.defaultdict(list)
for r in REP:
    if r["strict_status"] == "ok": rows[(r["cell"], r["round"])].append(r["strict_parse"])
ex_rows("Replay (2,048-token cap, logged conversations), strict-valid readouts only", rows)
rows = collections.defaultdict(list)
for t in RER:
    for rd in t["rounds"]:
        for a in rd["agents"]:
            if a["readout"]["strict_status"] == "ok": rows[(t["cell"], rd["round"])].append(a["readout"]["strict_parse"])
ex_rows("Full rerun, strict-valid readouts only", rows)

# ------------------------------------------------------------------ C. accuracy by true state
H("C. Trial-level log-pool accuracy at the terminal round, by true state (correct / trials)")
def acc_sept(cell, T):
    ts = [t for t in SEPT[cell] if T == "all" or t["true_state"] == T]
    k = sum((logpool([float(a["belief"]) for a in t["rounds"][-1]["agents"]]) > .5) == (t["true_state"] == "A") for t in ts)
    return f"{k}/{len(ts)}"
def outcomes(groups):  # dict (cell,seed)->(truth, [rec...]) ; rec has strict_status/strict_parse
    out = {}
    for k, (tr, recs) in groups.items():
        out[k] = (tr, None if any(r["strict_status"] != "ok" for r in recs) else
                  (logpool([r["strict_parse"] for r in recs]) > .5) == (tr == "A"))
    return out
term = {}
for r in REP: term[(r["cell"], r["seed"])] = max(term.get((r["cell"], r["seed"]), 0), r["round"])
g = collections.defaultdict(lambda: [None, []])
for r in REP:
    if r["round"] == term[(r["cell"], r["seed"])]: g[(r["cell"], r["seed"])][0] = r["truth"]; g[(r["cell"], r["seed"])][1].append(r)
rep_o = outcomes(g)
rer_o = outcomes({(t["cell"], t["seed"]): (t["true_state"], [a["readout"] for a in t["rounds"][-1]["agents"]]) for t in RER})
def fmt(o, cell, T):
    v = [x for (c_, s), (tr, x) in o.items() if c_ == cell and (T == "all" or tr == T)]
    if not v: return "n/a"
    return f"{sum(x is True for x in v)}/{sum(x is not None for x in v)} (+{sum(x is None for x in v)} incomplete)"
print(f"  {'cell':18}{'T':4}{'September':>12}{'replay strict':>28}{'rerun strict':>28}")
for cell in R1CELLS + TOPO:
    for T in ("A", "B", "all"):
        print(f"  {cell:18}{T:4}{acc_sept(cell, T):>12}{fmt(rep_o, cell, T):>28}{fmt(rer_o, cell, T):>28}")

# ------------------------------------------------------------------ D. invalid and truncated readouts by cell, model, round, degree
H("D. Readout instrument by cell, model, degree, round (denominator n = readouts)")
def inst(label, items):
    print(f"\n{label}")
    print(f"  {'cell':18}{'model':19}{'deg':>4}{'rnd':>4}{'n':>6}{'>320 tok':>9}{'cap stop':>9}{'invalid':>8}  invalid by status")
    for k in sorted(items):
        v = items[k]; st = collections.Counter(r["strict_status"] for r in v if r["strict_status"] != "ok")
        print(f"  {k[0]:18}{k[1]:19}{k[2]:4d}{k[3]:4d}{len(v):6d}"
              f"{sum(r['usage'].get('completion_tokens', 0) > 320 for r in v):9d}{sum(map(capstop, v)):9d}"
              f"{sum(st.values()):8d}  {dict(st)}")
deg = lambda cell: RR.CELLS[cell][2] - 1 if RR.CELLS[cell][3] == "all_channel" else -1
it = collections.defaultdict(list)
for r in REP: it[(r["cell"], r["model"], deg(r["cell"]), r["round"])].append(r)
inst("Replay, 2,048-token cap (deg -1 = mixed-degree topology cell). '>320 tok' = would have exceeded the original cap; not an observed truncation.", it)
it = collections.defaultdict(list)
for t in RER:
    for rd in t["rounds"]:
        for a in rd["agents"]: it[(t["cell"], t["model"], deg(t["cell"]), rd["round"])].append(a["readout"])
inst("Full rerun, 2,048-token cap", it)
tot = sum(len(v) for v in it.values()); print(f"\n  rerun totals: readouts {tot}, cap stops {sum(sum(map(capstop, v)) for v in it.values())}, "
      f"invalid {sum(sum(r['strict_status'] != 'ok' for r in v) for v in it.values())}, no response {sum(sum(r['text'] is None for r in v) for v in it.values())}")
print("  Retry policy: transport failures only, up to 6 attempts; a received response is never retried;"
      " invalid readouts are kept as invalid and bounded, never set to 0.5 or dropped.")
print(f"  transport retries used: replay {sum(r['attempts'] > 1 for r in REP)}, rerun "
      f"{sum(a['readout']['attempts'] > 1 for t in RER for rd in t['rounds'] for a in rd['agents'])}")

# ------------------------------------------------------------------ E. messages: cap stops and disclosure, two classifiers
H("E. Conclusion messages: cap stops (rerun only; September stop reasons were not logged) and evidence disclosure")
def subst(txt):    # classifier asserted in audit_revision.family_G, which produced the published 141
    return bool(AR.SUBST.search(AR.CONFB.sub(" ", AR.CONF.sub(" ", txt))))
def broad(txt):    # broader pattern in check_leakage.EVIDENCE (also flags possession language)
    return bool(CL.EVIDENCE.search(CL.CONF_BARE.sub(" ", CL.CONFIDENCE.sub(" ", txt))))
def disc(label, trials, get):
    print(f"\n{label}")
    print(f"  {'cell':14}{'rnd':>4}{'msgs':>6}{'cap stops':>10}{'SUBST':>7}{'EVIDENCE':>9}{'both clf':>9}{'EVID only':>10}")
    by = collections.defaultdict(list); trial_hit = collections.defaultdict(lambda: [0, 0, 0])
    for cell, t in trials:
        if RR.CELLS[cell][4] != "conclusion_only": continue
        hit_s = hit_b = False
        for rd in t["rounds"]:
            for a in rd["agents"]:
                m = get(a)
                if m is None: continue
                txt, cs = m
                s, b = subst(txt), broad(txt); hit_s |= s; hit_b |= b
                by[(cell, rd["round"])].append((cs, s, b))
        h = trial_hit[cell]; h[0] += 1; h[1] += hit_s; h[2] += hit_b
    for cell in sorted({k[0] for k in by}):
        ks = sorted(k for k in by if k[0] == cell) + [(cell, "all")]
        for k in ks:
            v = by[k] if k[1] != "all" else [x for kk in by if kk[0] == cell for x in by[kk]]
            cs = ["n/a" if x[0] is None else x[0] for x in v]
            csn = "n/a" if any(x == "n/a" for x in cs) else str(sum(cs))
            print(f"  {cell:14}{str(k[1]):>4}{len(v):6d}{csn:>10}{sum(x[1] for x in v):7d}{sum(x[2] for x in v):9d}"
                  f"{sum(x[1] and x[2] for x in v):9d}{sum(x[2] and not x[1] for x in v):10d}")
        h = trial_hit[cell]
        print(f"  {cell:14} trials with >=1 disclosing message: SUBST {h[1]} of {h[0]}, EVIDENCE {h[2]} of {h[0]}")
disc("September (transcripts.jsonl)", [(c_, t) for c_ in R1CELLS for t in SEPT[c_]],
     lambda a: (a["message"], None) if a.get("message") else None)
disc("Full rerun", [(t["cell"], t) for t in RER],
     lambda a: (a["message"]["text"], capstop(a["message"])) if "message" in a else None)
# examples of the September messages flagged only by the broader pattern
ex = [a["message"] for t in SEPT["cl5_mm_co"] for rd in t["rounds"] for a in rd["agents"]
      if a.get("message") and broad(a["message"]) and not subst(a["message"])]
print(f"\n  September cl5_mm_co messages flagged by EVIDENCE but not SUBST: {len(ex)}; first three:")
for m in ex[:3]: print("   -", m[:200].replace("\n", " "))
# cross-check: published 141 from messages.csv with the asserted classifier
n = k = 0
for r in csv.DictReader(open(RR.DATA / "claude-sonnet-4-6" / "messages.csv")):
    if r["content_mode"] == "conclusion_only" and r["regime"] == "misleading_majority" and r["topology"] == "all_channel":
        n += 1; k += subst(r["message_text"])
print(f"  cross-check, messages.csv + SUBST (the published procedure): {k} of {n}")

# ------------------------------------------------------------------ F. neighbor fan-in, recorded then remeasured
H("F. Neighbor fan-in, hierarchy and chain only, graph fixed effects, 2,000 trial-level resamples (seed 7)")
def demean(rs):
    by = collections.defaultdict(list)
    for r in rs: by[r["topo"]].append(r)
    out = []
    for v in by.values():
        m = {k: sum(x[k] for x in v) / len(v) for k in ("own", "nbr", "cap")}
        out += [{k: x[k] - m[k] for k in ("own", "nbr", "cap")} for x in v]
    return out
def stats(rs):
    d = demean(rs); o = [x["own"] for x in d]; nb = [x["nbr"] for x in d]; cp = [x["cap"] for x in d]
    return AR._partial(nb, cp, o), AR._partial(o, cp, nb)
def boot(recs):
    obs = stats(recs); by = collections.defaultdict(list)
    for r in recs: by[(r["topo"], r["seed"])].append(r)
    keys = sorted(by); rng = random.Random(7); d0, d1 = [], []
    for _ in range(2000):
        s = []
        for _ in keys: s += by[keys[rng.randrange(len(keys))]]
        v = stats(s); d0.append(v[0]); d1.append(v[1])
    q = lambda x, p: sorted(x)[int(p * len(x))]
    return obs, (q(d0, .025), q(d0, .975)), (q(d1, .025), q(d1, .975)), len(recs), len(keys)
# recorded: the audit's own records (beliefs.csv), with agent ids kept for the remeasured join
def positional(with_ids):
    TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
    truth = {(r["topology"], r["seed"]): r["true_state"] for r in AR.rounds_rows(TAG)
             if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in ("hierarchy", "chain")}
    rec, fan = collections.defaultdict(dict), {}
    for r in csv.DictReader(open(AR.DATA / TAG / "beliefs.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in ("hierarchy", "chain"): continue
        k = (r["topology"], r["seed"], r["agent"]); rec[k][int(r["round"])] = (float(r["belief_pA"]), float(r["llr"]))
        if r["round"] == "1": fan[k] = int(r["n_received"])
    nbrs = {}
    for r in csv.DictReader(open(AR.DATA / TAG / "messages.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in ("hierarchy", "chain") or r["round"] != "1": continue
        nbrs[(r["topology"], r["seed"], r["agent"])] = [s for s in (r.get("received_from") or "").split(";") if s]
    out = []
    for k, byr in rec.items():
        tp, s, a = k
        if (tp, s) not in truth or k not in fan or k not in nbrs: continue
        rs = sorted(byr); llr = byr[rs[0]][1]; tA = truth[(tp, s)] == "A"
        if (llr > 0) != tA: continue
        nf = [fan[(tp, s, x)] for x in nbrs[k] if (tp, s, x) in fan]
        if not nf: continue
        out.append(dict(topo=tp, seed=s, agent=a, tA=tA, own=fan[k], nbr=sum(nf) / len(nf),
                        cap=1 if ((byr[rs[-1]][0] > .5) != tA) else 0))
    return out
recd = positional(True)
aud = [r for r in AR._positional_records() if r["topo"] in ("hierarchy", "chain")]
print(f"  agent set check: this reconstruction {len(recd)} agents, audit _positional_records {len(aud)} agents, "
      f"capitulations {sum(r['cap'] for r in recd)} vs {sum(r['cap'] for r in aud)}")
o, i0, i1, na, nt = boot(recd)
print(f"  RECORDED September beliefs:  neighbor|own r={o[0]:.3f} [{i0[0]:.3f}, {i0[1]:.3f}]   "
      f"own|neighbor r={o[1]:.3f} [{i1[0]:.3f}, {i1[1]:.3f}]   agents {na}, trials {nt}")
cellof = {"hierarchy": "gpt5_mm_co_hier", "chain": "gpt5_mm_co_chain"}
fin = {(r["cell"], r["seed"], r["agent"]): r for r in REP if r["round"] == term[(r["cell"], r["seed"])]}
rem, inv = [], 0
for r in recd:
    x = fin[(cellof[r["topo"]], int(r["seed"]), int(r["agent"]))]
    if x["strict_status"] != "ok": inv += 1; continue
    rem.append(dict(r, cap=1 if ((x["strict_parse"] > .5) != r["tA"]) else 0))
o, i0, i1, na, nt = boot(rem)
print(f"  REMEASURED (replay, strict):  neighbor|own r={o[0]:.3f} [{i0[0]:.3f}, {i0[1]:.3f}]   "
      f"own|neighbor r={o[1]:.3f} [{i1[0]:.3f}, {i1[1]:.3f}]   agents {na}, trials {nt}, excluded invalid {inv}")
print(f"  capitulations: recorded {sum(r['cap'] for r in recd)} of {len(recd)}, remeasured {sum(r['cap'] for r in rem)} of {len(rem)}")

# ------------------------------------------------------------------ G. two statements in the revised text
H("G. Reading identifiers in the degree-10 concordant evidence messages, and exact pooled ties in the regenerated trials")
# G1. SI: all 1,650 messages preserved the sender's reading identifier and both likelihoods (section A checks the values).
LBL = re.compile(r"\br\d+\b")
c = collections.Counter()
for t in SEPT["gpt11_cc_ev"]:
    meta = {m["id"]: m for m in t["agents_meta"]}
    for rd in t["rounds"]:
        for a in rd["agents"]:
            m = a.get("message")
            if not m: continue
            own, labels = meta[a["id"]]["signal_label"], set(LBL.findall(m))
            pa, pb = PA.search(m) or QA.search(m), PB.search(m) or QB.search(m)
            vals = bool(pa and pb) and abs(float(pa.group(1)) - meta[a["id"]]["p_a"]) < 0.0051 \
                and abs(float(pb.group(1)) - meta[a["id"]]["p_b"]) < 0.0051
            c["messages"] += 1
            c["sender's identifier present"] += own in labels
            c["no other reading identifier"] += labels <= {own}
            c["identifier present and both likelihoods match to 2 dp"] += (own in labels) and vals
print("  September gpt11_cc_ev messages:", dict(c))
# G2. Methods: no complete regenerated-message trial had an exact pooled tie (log pool, 1e-6 clipping, as scored).
def meanlogodds(bs):
    e = 1e-6
    return sum(math.log(min(1 - e, max(e, b)) / (1 - min(1 - e, max(e, b)))) for b in bs) / len(bs)
c = collections.Counter()
for t in RER:
    term = t["rounds"][-1]["agents"]
    c["trials"] += 1
    if any(a["readout"]["strict_status"] != "ok" for a in term): continue
    bs = [a["readout"]["strict_parse"] for a in term]
    c["complete trials"] += 1
    c["mean log-odds exactly 0"] += meanlogodds(bs) == 0
    c["pooled probability exactly 0.5"] += logpool(bs) == 0.5
print("  Regenerated-message trials, terminal round, strict readouts:", dict(c))
