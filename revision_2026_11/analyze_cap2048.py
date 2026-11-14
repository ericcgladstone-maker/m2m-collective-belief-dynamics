"""Analysis of the 2,048-token readout replay on the logged September conversations.

Reports, separately: responses longer than 320 tokens (NOT an observed truncation at 320),
responses that stopped at the 2,048 cap, and responses with no valid FINAL= value.
Invalid readouts are never converted to 0.5 or silently dropped: trial-level results
are given for complete trials, with bounds that score incomplete trials as wrong or right.
"""
import json, math, random, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_readout as RR  # noqa: E402

R = [json.loads(l) for l in open(Path(sys.argv[1]) if len(sys.argv) > 1 else RR.REV / "cap2048_readouts.jsonl")]
side = lambda v, t: (v > 0.5) if t == "A" else (v < 0.5)
capstop = lambda r: r["stop_reason"] in ("length", "max_tokens")
ok = lambda r: r["strict_status"] == "ok"

def logpool(bs):
    e = 1e-6
    s = sum(math.log(min(1 - e, max(e, b)) / (1 - min(1 - e, max(e, b)))) for b in bs) / len(bs)
    return 1 / (1 + math.exp(-s))

print(f"records {len(R)}  no_response {sum(r['text'] is None for r in R)}  "
      f"transport retries (attempts>1) {sum(r['attempts'] > 1 for r in R)}")
print("returned models", collections.Counter(r.get("model_returned") for r in R))

# ---------------- 1. instrument
print("\n1. Instrument, by cell / truth / round group")
print(f"{'cell':18}{'T':2}{'rnd':>5}{'n':>6}{'>320tok':>9}{'cap-stop':>9}{'noValid':>8}{'status breakdown'}")
g = collections.defaultdict(list)
for r in R: g[(r["cell"], r["truth"], "r0" if r["round"] == 0 else "post")].append(r)
for k, v in sorted(g.items()):
    n = len(v)
    ex = sum(r["usage"].get("completion_tokens", 0) > 320 for r in v)
    cs = sum(map(capstop, v)); nv = sum(not ok(r) for r in v)
    st = collections.Counter(r["strict_status"] for r in v if not ok(r))
    print(f"{k[0]:18}{k[1]:2}{k[2]:>5}{n:6d}{ex:9d}{cs:9d}{nv:8d}  {dict(st)}")

# ---------------- 2. trial-level accuracy (terminal round)
trials = collections.defaultdict(dict)   # (cell, seed) -> {agent: rec} at terminal round
term = {}
for r in R:
    k = (r["cell"], r["seed"])
    term[k] = max(term.get(k, 0), r["round"])
for r in R:
    k = (r["cell"], r["seed"])
    if r["round"] == term[k]: trials[k][r["agent"]] = r

def trial_outcome(recs, which):
    t = next(iter(recs.values()))["truth"]
    if which == "strict":
        if not all(ok(x) for x in recs.values()): return None
        bs = [x["strict_parse"] for x in recs.values()]
    elif which == "legacy":
        bs = [x["legacy_parse"] for x in recs.values()]
    else:
        bs = [x["logged_belief"] for x in recs.values()]
    return (logpool(bs) > 0.5) == (t == "A")

print("\n2. Trial-level log-pool accuracy, terminal round")
print(f"{'cell':18}{'T':3}{'trials':>7}{'Sept':>7}{'legacy':>8}{'strict(complete)':>18}{'bounds':>16}")
acc = {}
for cell in RR.CELLS:
    for T in ("A", "B", "all"):
        ks = [k for k in trials if k[0] == cell and (T == "all" or next(iter(trials[k].values()))["truth"] == T)]
        if not ks: continue
        n = len(ks)
        sept = sum(trial_outcome(trials[k], "logged") for k in ks) / n
        leg = sum(trial_outcome(trials[k], "legacy") for k in ks) / n
        so = [trial_outcome(trials[k], "strict") for k in ks]
        comp = [x for x in so if x is not None]
        sc = sum(comp) / len(comp) if comp else float("nan")
        lo = sum(1 for x in so if x) / n; hi = sum(1 for x in so if x is not False) / n
        acc[(cell, T)] = dict(n=n, complete=len(comp), strict=sc, lo=lo, hi=hi, sept=sept)
        print(f"{cell:18}{T:3}{n:7d}{sept:7.2f}{leg:8.2f}{sc:10.2f} ({len(comp):3d}){lo:8.2f}–{hi:.2f}")

# ---------------- 3. content contrast, paired by seed
print("\n3. Evidence minus conclusion, paired by seed (complete pairs only)")
pairs = [("gpt5_mm_ev", "gpt5_mm_co"), ("cl5_mm_ev", "cl5_mm_co"),
         ("gpt11_cc_ev", "gpt11_cc_co"), ("gpt11_mm_ev", "gpt11_mm_co")]
rng = random.Random(11)
for ev, co in pairs:
    for T in ("A", "B", "all"):
        seeds = sorted({k[1] for k in trials if k[0] == ev} & {k[1] for k in trials if k[0] == co})
        rows, allp = [], []
        for s in seeds:
            if T != "all" and next(iter(trials[(ev, s)].values()))["truth"] != T: continue
            a, b = trial_outcome(trials[(ev, s)], "strict"), trial_outcome(trials[(co, s)], "strict")
            allp.append((a, b))
            if a is None or b is None: continue
            rows.append((a, b))
        if not allp: continue
        # bounds on the paired contrast itself: incomplete outcomes scored adversarially
        N = len(allp)
        dlo = (sum(a is True for a, _ in allp) - sum(b is not False for _, b in allp)) / N
        dhi = (sum(a is not False for a, _ in allp) - sum(b is True for _, b in allp)) / N
        inc = sum(a is None or b is None for a, b in allp)
        print(f"  {ev[:-3]:12}{T:4} all pairs={N:3d}  incomplete={inc:3d}  contrast bounds [{dlo:+.2f}, {dhi:+.2f}]")
        if not rows: continue
        d = sum(a for a, _ in rows) / len(rows) - sum(b for _, b in rows) / len(rows)
        e_only = sum(a and not b for a, b in rows); c_only = sum(b and not a for a, b in rows)
        bs = []
        for _ in range(2000):
            smp = [rows[rng.randrange(len(rows))] for _ in rows]
            bs.append(sum(a for a, _ in smp) / len(smp) - sum(b for _, b in smp) / len(smp))
        bs.sort()
        print(f"  {ev[:-3]:12}{T:4} pairs={len(rows):3d}  diff={d:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]"
              f"  evidence-only correct={e_only}  conclusion-only correct={c_only}")

# ---------------- 4. topology on remeasured beliefs
print("\n4. Topology (misleading majority, conclusion sharing), remeasured terminal beliefs")
recs, excluded = [], 0
cnt = collections.defaultdict(lambda: collections.Counter())   # (topo, position) -> counts
for cell, topo in (("gpt5_mm_co_wheel", "wheel"), ("gpt5_mm_co_hier", "hierarchy"), ("gpt5_mm_co_chain", "chain")):
    for t in RR.load_trials(cell):
        s = int(t["seed"]); meta = {m["id"]: m for m in t["agents_meta"]}
        r1 = {a["id"]: [d["src"] for d in a["received"] if d["src"] >= 0] for a in t["rounds"][1]["agents"]}
        fan = {i: len(v) for i, v in r1.items()}
        for i in r1:
            tA = t["true_state"] == "A"
            if (meta[i]["llr"] > 0) != tA: continue          # own reading favors the truth
            pos = ("hub" if fan[i] == max(fan.values()) else "spoke") if topo == "wheel" else f"fan-in {fan[i]}"
            c = cnt[(topo, pos)]; c["eligible"] += 1
            rec = trials.get((cell, s), {}).get(i)
            if rec is None: c["missing"] += 1; continue
            if not ok(rec): excluded += 1; c["excluded_invalid"] += 1; continue
            nf = [fan[x] for x in r1[i]]
            if not nf: c["excluded_no_senders"] += 1; continue
            c["used"] += 1
            cnt[(topo, pos, "seeds")][s] += 1
            recs.append(dict(topo=topo, seed=s, own=fan[i], nbr=sum(nf) / len(nf),
                             cap=1 if ((rec["strict_parse"] > .5) != tA) else 0))
print(f"  agents used {len(recs)}, excluded for invalid terminal readout {excluded}")
for k in sorted(k for k in cnt if len(k) == 2):
    c = cnt[k]; ntr = len(cnt[(k[0], k[1], "seeds")])
    print(f"    {k[0]:10}{k[1]:12} trials={ntr:3d}  agents eligible={c['eligible']:4d} used={c['used']:4d} "
          f"excluded(invalid)={c['excluded_invalid']:3d} excluded(no senders)={c['excluded_no_senders']:3d} missing={c['missing']:3d}")
wh = [r for r in recs if r["topo"] == "wheel"]
if wh:
    mx = max(r["own"] for r in wh)
    hub = [r["cap"] for r in wh if r["own"] == mx]; spk = [r["cap"] for r in wh if r["own"] < mx]
    print(f"  wheel capitulation: hub {sum(hub)}/{len(hub)} = {sum(hub)/max(1,len(hub)):.2f}, "
          f"spokes {sum(spk)}/{len(spk)} = {sum(spk)/max(1,len(spk)):.2f}")
sys.path.insert(0, str(RR.GITHUB)); import audit_revision as AR  # noqa: E402
hc = [r for r in recs if r["topo"] in ("hierarchy", "chain")]
def fe(rs):
    by = collections.defaultdict(list)
    for r in rs: by[r["topo"]].append(r)
    out = []
    for v in by.values():
        m = {k: sum(x[k] for x in v) / len(v) for k in ("own", "nbr", "cap")}
        out += [{k: x[k] - m[k] for k in ("own", "nbr", "cap")} for x in v]
    o = [x["own"] for x in out]; n = [x["nbr"] for x in out]; c = [x["cap"] for x in out]
    return AR._partial(n, c, o), AR._partial(o, c, n)
if hc:
    obs = fe(hc); by = collections.defaultdict(list)
    for r in hc: by[(r["topo"], r["seed"])].append(r)
    keys = sorted(by); d0, d1 = [], []
    for _ in range(2000):
        smp = []
        for _ in keys: smp += by[keys[rng.randrange(len(keys))]]
        v = fe(smp); d0.append(v[0]); d1.append(v[1])
    d0.sort(); d1.sort()
    print(f"  hierarchy+chain, graph FE: neighbour|own r={obs[0]:.3f} [{d0[50]:.3f}, {d0[1949]:.3f}]; "
          f"own|neighbour r={obs[1]:.3f} [{d1[50]:.3f}, {d1[1949]:.3f}]  (agents {len(hc)}, trials {len(keys)})")
