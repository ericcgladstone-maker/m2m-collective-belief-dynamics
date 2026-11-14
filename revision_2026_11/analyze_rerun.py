"""Analysis of the corrected-protocol full reruns (same seeds, regenerated conversations).

1. Instrument: readouts over 320 tokens, cap stops, invalid FINAL; message cap stops.
2. Trial-level log-pool accuracy at the terminal round (strict, complete trials) with
   bounds scoring incomplete trials wrong / right; September value for reference.
3. Evidence minus conclusion paired by seed, with contrast bounds over all pairs and a
   complete-pair estimate with a 2,000-resample bootstrap interval.
"""
import json, math, os, random, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_readout as RR  # noqa: E402

T = [json.loads(l) for l in open(Path(sys.argv[1]) if len(sys.argv) > 1 else RR.REV / "rerun_trials.jsonl")]
# ---------------- 0. completeness and uniqueness gate (before any analysis)
import rerun_full as RF  # noqa: E402
planned = set(RF.plan()[0])
okc = collections.Counter((t["cell"], t["seed"]) for t in T if t["status"] == "ok")
bad = collections.Counter(t["status"] for t in T if t["status"] != "ok")
missing = planned - set(okc); extra = set(okc) - planned; dup = {k: v for k, v in okc.items() if v > 1}
print(f"0. records {len(T)}  planned {len(planned)}  ok unique {len(okc)}  missing {len(missing)}  "
      f"duplicates {len(dup)}  unplanned {len(extra)}  non-ok records {dict(bad)}")
if missing or dup or extra:
    print("   GATE FAIL", sorted(missing)[:10], list(dup.items())[:10], sorted(extra)[:10])
    if os.environ.get("PARTIAL_TEST") != "1": sys.exit(1)
    print("   PARTIAL_TEST=1: continuing on incomplete data for code testing only; not a result")
print("   GATE PASS: all 500 planned cell-trials present exactly once")
T = {(t["cell"], t["seed"]): t for t in T if t["status"] == "ok"}

def logpool(bs):
    e = 1e-6
    s = sum(math.log(min(1 - e, max(e, b)) / (1 - min(1 - e, max(e, b)))) for b in bs) / len(bs)
    return 1 / (1 + math.exp(-s))

def ntok(r): return r["usage"].get("completion_tokens", 0)
def capstop(r): return r["stop_reason"] in ("length", "max_tokens")

print("\n1. Instrument")
print(f"{'cell':13}{'T':2}{'post reads':>11}{'>320':>6}{'capstop':>8}{'invalid':>8}{'msgs':>6}{'msg capstop':>12}  invalid status")
g = collections.defaultdict(lambda: collections.Counter()); st = collections.defaultdict(collections.Counter)
for t in T.values():
    k = (t["cell"], t["true_state"]); c = g[k]
    for rd in t["rounds"]:
        for a in rd["agents"]:
            r = a["readout"]
            if rd["round"] > 0:
                c["n"] += 1; c["over"] += ntok(r) > 320; c["cap"] += capstop(r)
            if r["strict_status"] != "ok":
                c["inv"] += 1; st[k][f'r{rd["round"]}:{r["strict_status"]}'] += 1
            if "message" in a:
                c["m"] += 1; c["mcap"] += capstop(a["message"])
for k in sorted(g):
    c = g[k]
    print(f"{k[0]:13}{k[1]:2}{c['n']:11d}{c['over']:6d}{c['cap']:8d}{c['inv']:8d}{c['m']:6d}{c['mcap']:12d}  {dict(st[k])}")

def outcome(t):
    term = t["rounds"][-1]["agents"]
    if any(a["readout"]["strict_status"] != "ok" for a in term): return None
    return (logpool([a["readout"]["strict_parse"] for a in term]) > 0.5) == (t["true_state"] == "A")

sept = {}
for cell in {k[0] for k in T}:
    for t in RR.load_trials(cell):
        b = [float(a["belief"]) for a in t["rounds"][-1]["agents"]]
        sept[(cell, int(t["seed"]))] = (logpool(b) > 0.5) == (t["true_state"] == "A")

print("\n2. Trial-level log-pool accuracy, terminal round")
for cell in [c for c in ["gpt5_mm_ev", "gpt5_mm_co", "cl5_mm_ev", "cl5_mm_co", "gpt11_cc_ev",
                          "gpt11_cc_co", "gpt11_mm_ev", "gpt11_mm_co"] if any(k[0] == c for k in T)]:
    for TT in ("A", "B", "all"):
        ks = [k for k in T if k[0] == cell and (TT == "all" or T[k]["true_state"] == TT)]
        o = [outcome(T[k]) for k in ks]; comp = [x for x in o if x is not None]
        n = len(o); sp = sum(sept[k] for k in ks) / n
        print(f"  {cell:12}{TT:4} trials={n:3d} Sept={sp:.2f}  strict={sum(comp)/max(1,len(comp)):.2f} ({len(comp)})"
              f"  bounds {sum(x is True for x in o)/n:.2f}-{sum(x is not False for x in o)/n:.2f}")

print("\n3. Evidence minus conclusion, paired by seed")
rng = random.Random(11)
for ev, co in [("gpt5_mm_ev", "gpt5_mm_co"), ("cl5_mm_ev", "cl5_mm_co"),
               ("gpt11_cc_ev", "gpt11_cc_co"), ("gpt11_mm_ev", "gpt11_mm_co")]:
    seeds = sorted({k[1] for k in T if k[0] == ev} & {k[1] for k in T if k[0] == co})
    for TT in ("A", "B", "all"):
        allp = [(outcome(T[(ev, s)]), outcome(T[(co, s)])) for s in seeds
                if TT == "all" or T[(ev, s)]["true_state"] == TT]
        if not allp: continue
        N = len(allp); inc = sum(a is None or b is None for a, b in allp)
        dlo = (sum(a is True for a, _ in allp) - sum(b is not False for _, b in allp)) / N
        dhi = (sum(a is not False for a, _ in allp) - sum(b is True for _, b in allp)) / N
        rows = [(a, b) for a, b in allp if a is not None and b is not None]
        d = (sum(a for a, _ in rows) - sum(b for _, b in rows)) / len(rows)
        bs = sorted((lambda s: (sum(a for a, _ in s) - sum(b for _, b in s)) / len(s))(
            [rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(2000))
        print(f"  {ev[:-3]:9}{TT:4} pairs={N:3d} incomplete={inc:2d} bounds [{dlo:+.2f}, {dhi:+.2f}]"
              f"  complete diff={d:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]"
              f"  ev-only={sum(a and not b for a, b in rows)} co-only={sum(b and not a for a, b in rows)}")

# ---------------- 4. messages: cap stops and evidence disclosure, reported separately
sys.path.insert(0, str(RR.GITHUB)); import check_leakage as CL  # noqa: E402
def disclosed(txt):
    probe = CL.CONF_BARE.sub(" ", CL.CONFIDENCE.sub(" ", txt))
    return bool(CL.EVIDENCE.search(probe))
print("\n4. Messages. Cap stop = stop reason at the 120-token cap. Disclosure = deposited classifier")
print("   (check_leakage.EVIDENCE after stripping the confidence value), applied to conclusion messages only.")
print(f"{'cell':13}{'T':2}{'msgs':>6}{'cap stops':>10}{'disclosed':>10}{'both':>6}   September disclosed (same classifier)")
for cell in sorted({k[0] for k in T}):
    for TT in ("A", "B", "all"):
        ms = [a["message"] for k, t in T.items() if k[0] == cell and (TT == "all" or t["true_state"] == TT)
              for rd in t["rounds"] for a in rd["agents"] if "message" in a]
        cap = [capstop(m) for m in ms]
        if T[next(k for k in T if k[0] == cell)]["content_mode"] == "conclusion_only":
            dis = [disclosed(m["text"]) for m in ms]
            sm = [a["message"] for t in RR.load_trials(cell) if TT == "all" or t["true_state"] == TT
                  for rd in t["rounds"] for a in rd["agents"] if a.get("message")]
            sd = sum(map(disclosed, sm))
            print(f"{cell:13}{TT:4}{len(ms):6d}{sum(cap):10d}{sum(dis):10d}{sum(c and d for c, d in zip(cap, dis)):6d}"
                  f"   {sd} of {len(sm)}")
        else:
            print(f"{cell:13}{TT:4}{len(ms):6d}{sum(cap):10d}{'n/a':>10}{'':6}")

cost = collections.Counter()
for t in T.values():
    for rd in t["rounds"]:
        for a in rd["agents"]:
            for r in [a["readout"]] + ([a["message"]] if "message" in a else []):
                cost[(t["model"], "in")] += r["usage"].get("prompt_tokens", 0)
                cost[(t["model"], "out")] += r["usage"].get("completion_tokens", 0)
pr = {"gpt-4.1-mini": (0.4e-6, 1.6e-6), "claude-sonnet-4-6": (3e-6, 15e-6)}
print("\ncost", {m: round(cost[(m, "in")] * p[0] + cost[(m, "out")] * p[1], 2) for m, p in pr.items()})
