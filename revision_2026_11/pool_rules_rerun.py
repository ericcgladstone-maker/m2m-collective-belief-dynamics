"""Focal gpt-4.1-mini n=5 evidence-minus-conclusion contrast under the six published
aggregation rules (analyze_pools.classify), on the full-rerun strict readouts.
Read-only: existing records only, no API calls. Also reproduces the September row of
SI Table S15 with the same code as a check on the rule implementation."""
import json, sys, random, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_readout as RR  # puts the deposited code on sys.path
import analyze_pools as AP
RULES = AP.RULES
T = {}
for l in open(RR.REV / "rerun_trials.jsonl"):
    t = json.loads(l)
    if t["status"] == "ok" and t["cell"] in ("gpt5_mm_ev", "gpt5_mm_co"): T[(t["cell"], t["seed"])] = t
def outcome(t, rule):
    term = t["rounds"][-1]["agents"]
    if any(a["readout"]["strict_status"] != "ok" for a in term): return None
    b = [a["readout"]["strict_parse"] for a in term]
    return AP.classify(b, rule) == (t["true_state"] == "A")
def ties(t):
    term = t["rounds"][-1]["agents"]
    if any(a["readout"]["strict_status"] != "ok" for a in term): return 0
    b = [a["readout"]["strict_parse"] for a in term]
    return int(sum(AP.logit(x) for x in b) == 0) + int(sum(b)/len(b) == 0.5) + int(sorted(b)[2] == 0.5) + sum(x == 0.5 for x in b)
seeds = sorted({s for c, s in T})
assert len(seeds) == 100 and all(("gpt5_mm_ev", s) in T and ("gpt5_mm_co", s) in T for s in seeds)
print("Exact ties or 0.5 values touching a rule boundary (terminal, complete trials):",
      sum(ties(T[(c, s)]) for c in ("gpt5_mm_ev", "gpt5_mm_co") for s in seeds))
# September check: same rule code on the logged transcript beliefs
print("\nSeptember check (transcript beliefs, all 100 trials, unpaired difference as in Table S15)")
sept = {}
for cell in ("gpt5_mm_ev", "gpt5_mm_co"):
    for t in RR.load_trials(cell):
        sept[(cell, int(t["seed"]))] = ([float(a["belief"]) for a in t["rounds"][-1]["agents"]], t["true_state"])
for rule in RULES:
    acc = {c: sum(AP.classify(b, rule) == (ts == "A") for (cc, s), (b, ts) in sept.items() if cc == c) / 100
           for c in ("gpt5_mm_ev", "gpt5_mm_co")}
    print(f"  {rule:8} ev {acc['gpt5_mm_ev']:.2f}  co {acc['gpt5_mm_co']:.2f}  diff {acc['gpt5_mm_ev']-acc['gpt5_mm_co']:+.2f}")
print("\nRerun (strict readouts). Accuracy over complete trials. Paired difference over complete pairs,")
print("2,000 pair-bootstrap resamples (seed 11). Bounds over all 100 pairs.")
print(f"  {'rule':8}{'ev acc (n)':>13}{'co acc (n)':>13}{'pairs':>6}{'diff':>7}{'95% CI':>18}{'ev-only':>8}{'co-only':>8}{'bounds':>16}")
for rule in RULES:
    ev = {s: outcome(T[("gpt5_mm_ev", s)], rule) for s in seeds}
    co = {s: outcome(T[("gpt5_mm_co", s)], rule) for s in seeds}
    ce = [v for v in ev.values() if v is not None]; cc = [v for v in co.values() if v is not None]
    pairs = [(ev[s], co[s]) for s in seeds if ev[s] is not None and co[s] is not None]
    d = sum(e - c for e, c in pairs) / len(pairs)
    rng = random.Random(11)
    bs = sorted(sum(e - c for e, c in (pairs[rng.randrange(len(pairs))] for _ in pairs)) / len(pairs) for _ in range(2000))
    lo_ci, hi_ci = bs[49], bs[1949]
    eo = sum(e and not c for e, c in pairs); co_ = sum(c and not e for e, c in pairs)
    N = len(seeds)
    dlo = (sum(ev[s] is True for s in seeds) - sum(co[s] is not False for s in seeds)) / N
    dhi = (sum(ev[s] is not False for s in seeds) - sum(co[s] is True for s in seeds)) / N
    print(f"  {rule:8}{sum(ce)/len(ce):>8.2f} ({len(ce)}){sum(cc)/len(cc):>8.2f} ({len(cc)}){len(pairs):>6}{d:>+7.2f}"
          f"   [{lo_ci:+.2f}, {hi_ci:+.2f}]{eo:>8}{co_:>8}   {dlo:+.2f} to {dhi:+.2f}")
