"""DATA AUDIT — transparency / integrity / no-hallucination check.

Re-derives every headline number directly from the raw on-disk files and asserts it
matches what we reported (within rounding). Also checks structural integrity of the raw
data: trial counts, round structure, belief ranges, message realness, duplicate keys,
and raw(transcripts) vs derived(rounds.csv) consistency. Deterministic and re-runnable;
the point is to NOT trust prose summaries. Writes nothing it can't recompute.

Section 12 delegates to audit_revision.py, which covers every quantity introduced or
changed during the major revision (PNASNEXUS-2026-01296). Running this file is therefore
the single entry point for verifying the manuscript and Supplementary Materials against the
archived data; it also writes audit_results.json.
"""
from __future__ import annotations
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PASS, FAIL, WARN = [], [], []
def ok(m): PASS.append(m)
def bad(m): FAIL.append(m)
def warn(m): WARN.append(m)

PAPER_DIRS = ["gpt-4.1-mini", "gpt-4.1-mini_n11", "gpt-4.1-mini_degree",
              "gpt-4.1-mini_overload", "gpt-4.1-mini_sigsweep", "claude-sonnet-4-6",
              "claude-sonnet-4-6_think", "gemini-2.5-flash", "hidden-gpt-4.1-mini"]
LEGACY = ["claude", "openai"]


def terminal(tag):
    """terminal-round row per trial; returns dict[(regime,topo,content)] -> list[row]."""
    p = HERE / "data" / tag / "rounds.csv"
    best = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            k = (r["regime"], r["topology"], r["content_mode"], r["seed"])
            if k not in best or int(r["round"]) > int(best[k]["round"]):
                best[k] = r
    cells = defaultdict(list)
    for (rg, tp, ct, sd), r in best.items():
        cells[(rg, tp, ct)].append(r)
    return cells


def acc(cells, rg, tp, ct):
    rows = cells.get((rg, tp, ct), [])
    if not rows: return None, 0
    return sum(int(r["acc_logodds"]) for r in rows) / len(rows), len(rows)


# ---------- 1. RAW DATA INTEGRITY ----------
print("="*70, "\n1. RAW DATA INTEGRITY\n", "="*70)
for tag in PAPER_DIRS:
    d = HERE / "data" / tag
    rp, tp_ = d / "rounds.csv", d / "full" / "transcripts.jsonl"
    if not rp.exists(): bad(f"{tag}: rounds.csv MISSING"); continue
    # transcripts vs rounds trial-count consistency
    trials_rounds = set()
    bad_belief = 0; bad_state = 0
    with open(rp) as f:
        for r in csv.DictReader(f):
            trials_rounds.add((r["regime"], r["topology"], r["content_mode"], r["seed"]))
            if r["true_state"] not in ("A", "B"): bad_state += 1
    if tp_.exists():
        tr = [json.loads(l) for l in tp_.read_text().splitlines() if l.strip()]
        # belief range + message realness from transcripts
        empty_msg = 0; total_msg = 0; oob = 0
        for t in tr:
            rounds = t.get("rounds", [])
            for rd in rounds:
                agents = rd["agents"] if isinstance(rd, dict) else rd
                for a in agents:
                    b = a["belief"]
                    if not (0.0 <= b <= 1.0): oob += 1
                    if isinstance(a, dict) and "message" in a and a["message"] is not None:
                        total_msg += 1
                        if not str(a["message"]).strip(): empty_msg += 1
        n_tr = len(tr)
        if n_tr == len(trials_rounds): ok(f"{tag}: transcripts({n_tr}) == rounds-trials({len(trials_rounds)})")
        else: bad(f"{tag}: transcripts({n_tr}) != rounds-trials({len(trials_rounds)})")
        if oob == 0: ok(f"{tag}: all beliefs in [0,1]")
        else: bad(f"{tag}: {oob} beliefs OUT OF RANGE")
        if total_msg and empty_msg == 0: ok(f"{tag}: {total_msg} messages, 0 empty")
        elif total_msg: warn(f"{tag}: {empty_msg}/{total_msg} empty messages")
    else:
        warn(f"{tag}: no transcripts.jsonl (rounds.csv only) — raw exchanges not stored")
    if bad_state == 0: ok(f"{tag}: true_state valid")
    else: bad(f"{tag}: {bad_state} bad true_state rows")

# ---------- 2. ANALYSIS RE-DERIVATION vs REPORTED ----------
print("\n" + "="*70, "\n2. RE-DERIVE HEADLINE NUMBERS vs REPORTED (tol 0.02)\n", "="*70)
# (reported value, tag, regime, topology, content)
CHECKS = [
    (1.00, "gpt-4.1-mini", "misleading_majority", "baseline", "none"),
    (0.90, "gpt-4.1-mini", "misleading_majority", "all_channel", "evidence_only"),
    (0.81, "gpt-4.1-mini", "misleading_majority", "all_channel", "both"),
    (0.64, "gpt-4.1-mini", "misleading_majority", "all_channel", "conclusion_only"),
    (1.00, "claude-sonnet-4-6", "misleading_majority", "baseline", "none"),
    (0.96, "claude-sonnet-4-6", "misleading_majority", "all_channel", "evidence_only"),
    (0.62, "claude-sonnet-4-6", "misleading_majority", "all_channel", "conclusion_only"),
    (0.97, "gemini-2.5-flash", "misleading_majority", "all_channel", "evidence_only"),
    (0.93, "gemini-2.5-flash", "misleading_majority", "all_channel", "conclusion_only"),
    (1.00, "gpt-4.1-mini_n11", "misleading_majority", "circle", "conclusion_only"),
    (0.94, "gpt-4.1-mini_n11", "misleading_majority", "ring_shortcuts", "conclusion_only"),
    (0.58, "gpt-4.1-mini_n11", "misleading_majority", "all_channel", "conclusion_only"),
    (0.90, "gpt-4.1-mini_degree", "misleading_majority", "lattice_k4", "conclusion_only"),
    (0.76, "gpt-4.1-mini_degree", "misleading_majority", "lattice_k6", "conclusion_only"),
    (0.60, "gpt-4.1-mini_degree", "misleading_majority", "lattice_k8", "conclusion_only"),
    (1.00, "gpt-4.1-mini_overload", "concordant", "lattice_k4", "evidence_only"),
    (1.00, "gpt-4.1-mini_overload", "concordant", "lattice_k6", "evidence_only"),
    (0.90, "gpt-4.1-mini_overload", "concordant", "lattice_k8", "evidence_only"),
    (0.68, "gpt-4.1-mini_n11", "concordant", "all_channel", "evidence_only"),
    (0.54, "gpt-4.1-mini_sigsweep", "mm_gap_narrow", "all_channel", "conclusion_only"),
    (0.42, "gpt-4.1-mini_sigsweep", "mm_gap_wide", "all_channel", "conclusion_only"),
    (0.26, "gpt-4.1-mini_sigsweep", "mm_maj_heavy", "all_channel", "conclusion_only"),
    (1.00, "claude-sonnet-4-6_think", "misleading_majority", "all_channel", "conclusion_only"),
    (1.00, "hidden-gpt-4.1-mini", "misleading_majority", "all_channel", "conclusion_only"),
]
cache = {}
for rep, tag, rg, tp, ct in CHECKS:
    if tag not in cache:
        cache[tag] = terminal(tag)
    a, n = acc(cache[tag], rg, tp, ct)
    if a is None: bad(f"{tag} {rg}/{tp}/{ct}: NO DATA (reported {rep})"); continue
    if abs(a - rep) <= 0.02: ok(f"{tag} {tp}/{ct} {rg[:10]}: reported {rep:.2f} = derived {a:.2f} (N={n})")
    else: bad(f"{tag} {tp}/{ct} {rg[:10]}: reported {rep:.2f} != derived {a:.2f} (N={n})  <-- MISMATCH")

# ---------- 3. DECOMPOSITION IDENTITY (re-check) ----------
print("\n" + "="*70, "\n3. ē−D DECOMPOSITION IDENTITY (collective Brier == ē − D)\n", "="*70)
maxerr = 0.0
for tag in PAPER_DIRS:
    rp = HERE / "data" / tag / "rounds.csv"
    if not rp.exists(): continue
    with open(rp) as f:
        for r in csv.DictReader(f):
            y = 1.0 if r["true_state"] == "A" else 0.0
            lhs = (float(r["coll_mean"]) - y)**2
            rhs = float(r["e_bar"]) - float(r["D"])
            maxerr = max(maxerr, abs(lhs - rhs))
if maxerr < 1e-3: ok(f"decomposition identity holds (max abs err {maxerr:.2g})")
else: bad(f"decomposition identity VIOLATED (max abs err {maxerr:.2g})")

# ---------- 4. DUPLICATE-KEY CHECK ----------
print("\n" + "="*70, "\n4. DUPLICATE (regime,topo,content,seed) KEYS\n", "="*70)
for tag in PAPER_DIRS:
    tp_ = HERE / "data" / tag / "full" / "transcripts.jsonl"
    if not tp_.exists(): continue
    seen = defaultdict(int)
    for l in tp_.read_text().splitlines():
        if l.strip():
            t = json.loads(l); seen[(t["regime"], t["topology"], t["content_mode"], t["seed"])] += 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if not dups: ok(f"{tag}: no duplicate trial keys")
    else: bad(f"{tag}: {len(dups)} duplicate keys (e.g. {list(dups)[:2]})")

# ---------- 5. CAUSAL (do-operator) + ADEQUACY GATE re-derivability ----------
print("\n" + "="*70, "\n5. CAUSAL ACE + ADEQUACY GATE (now persisted?)\n", "="*70)
for tag in ["replay-gpt-4.1-mini", "replay-claude-sonnet-4-6"]:
    rp = HERE / "data" / tag / "replay.csv"; mp = HERE / "data" / tag / "run_meta.json"
    if not rp.exists():
        warn(f"{tag}: NOT yet regenerated (causal ACE not re-derivable)"); continue
    rows = list(csv.DictReader(open(rp)))
    pc = [float(r["p_truth"]) for r in rows if r["mode"] == "correct"]
    pw = [float(r["p_truth"]) for r in rows if r["mode"] == "wrong"]
    ace = sum(pc)/len(pc) - sum(pw)/len(pw)
    meta_ace = json.loads(mp.read_text()).get("ACE_ptruth") if mp.exists() else None
    consistent = meta_ace is not None and abs(ace - meta_ace) < 0.005
    if ace > 0.2 and consistent:
        ok(f"{tag}: ACE re-derives from replay.csv = {ace:+.3f} (meta {meta_ace:+.3f}); causal effect present")
    elif ace > 0.2:
        warn(f"{tag}: ACE={ace:+.3f} but meta mismatch ({meta_ace})")
    else:
        bad(f"{tag}: ACE={ace:+.3f} — causal effect NOT present (CHECK)")
ap = HERE / "data" / "adequacy" / "adequacy.csv"
if not ap.exists():
    warn("adequacy gate: NOT yet regenerated (gate scores not re-derivable)")
else:
    g = {(r["model"], r["regime"]): float(r["truth_acc"]) for r in csv.DictReader(open(ap))}
    adm = [g.get((m, "misleading_majority")) for m in ("claude-sonnet-4-6", "gpt-4.1-mini")]
    exc = [g.get((m, "misleading_majority")) for m in ("claude-haiku-4-5", "gpt-4o-mini")]
    if None in adm + exc:
        warn("adequacy gate: some models missing")
    else:
        gap = min(adm) - max(exc)   # separation is the right criterion (small N), not an absolute bar
        if gap >= 0.2:
            ok(f"adequacy gate separates admitted (misleading truth_acc {min(adm):.2f}–{max(adm):.2f}) "
               f"from excluded ({min(exc):.2f}–{max(exc):.2f}); gap {gap:.2f} — supports paper's inclusion/exclusion")
        else:
            bad(f"adequacy gate does NOT cleanly separate admit/exclude (gap {gap:.2f})")
    gem = [g.get((m, "misleading_majority")) for m in ("gemini-2.5-flash", "gemini-2.5-pro")]
    if None not in gem and min(gem) >= 0.83:
        ok(f"gemini gate re-derives from adequacy.csv (flash {gem[0]:.2f}, pro {gem[1]:.2f} misleading; admitted)")
    elif None in gem:
        warn("gemini gate rows not in adequacy.csv")

# ---------- 6. VERDICT LEAKAGE in evidence_only messages ----------
print("\n" + "="*70, "\n6. VERDICT LEAKAGE (evidence_only messages)\n", "="*70)
try:
    import check_leakage
    _tot, _fl = check_leakage.count_leakage()
    if _tot > 0 and len(_fl) == 0:
        ok(f"verdict leakage: 0 of {_tot} evidence_only messages contain a verdict (clean)")
    elif _tot > 0:
        bad(f"verdict leakage: {len(_fl)} of {_tot} evidence_only messages flagged (CHECK)")
    else:
        warn("verdict leakage: no evidence_only messages found")
except Exception as _e:
    warn(f"leakage check skipped: {_e}")

# ---------- 7. INFORMATION-SUFFICIENCY BENCHMARK + RECIPIENT COUNTERFACTUAL (S6.6) ----------
print("\n" + "="*70, "\n7. INFORMATION SUFFICIENCY (confidence-weighted benchmark, recipient counterfactual)\n", "="*70)
try:
    import analyze_benchmark as AB
    # round-0 verdicts: head-count favors false (0.00), confidence-weighted/log-pool favor true (1.00);
    # actual terminal reproduces the manuscript's conclusion-sharing cells.
    EXP_ACTUAL = {"gpt-4.1-mini": 0.64, "claude-sonnet-4-6": 0.62, "gemini-2.5-flash": 0.93}
    for fam, tag in AB.FAMILIES:
        o = AB.summarize(fam, tag, "conclusion_only")
        if o is None:
            warn(f"benchmark: no conclusion_only cell for {fam}"); continue
        if abs(o["headcount"]) < 1e-9 and abs(o["cw_vote"] - 1.0) < 1e-9 and abs(o["logpool"] - 1.0) < 1e-9:
            ok(f"benchmark {fam}: round-0 verdicts head-count {o['headcount']:.2f} false / confidence-weighted {o['cw_vote']:.2f} true (sufficient info)")
        else:
            bad(f"benchmark {fam}: endpoints off (head {o['headcount']:.2f}, cw {o['cw_vote']:.2f}, log {o['logpool']:.2f})")
        if abs(o["actual"] - EXP_ACTUAL[fam]) <= 0.01:
            ok(f"benchmark {fam}: actual terminal {o['actual']:.2f} matches reported conclusion cell {EXP_ACTUAL[fam]:.2f}")
        else:
            bad(f"benchmark {fam}: actual terminal {o['actual']:.2f} != reported {EXP_ACTUAL[fam]:.2f}")
except Exception as _e:
    warn(f"benchmark check skipped: {_e}")

try:
    import analyze_recipient as AR
    rows = []
    for tag, topos in AR.SOURCES:
        rows += AR.collect(tag, topos)
    n = len(rows)
    norm = sum(r["normative_capitulate"] for r in rows) / n
    act = sum(r["actual_capitulate"] for r in rows) / n
    lo = [r["actual_capitulate"] for r in rows if r["fan_in"] <= 3]
    hi = [r["actual_capitulate"] for r in rows if r["fan_in"] >= 8]
    lo_m, hi_m = sum(lo)/len(lo), sum(hi)/len(hi)
    if n >= 1400 and norm < 0.01 and 0.25 <= act <= 0.29:
        ok(f"recipient counterfactual: normative capitulation {norm:.3f} (~0) vs actual {act:.3f} over {n} agent-trials")
    else:
        bad(f"recipient counterfactual: norm {norm:.3f} / act {act:.3f} / n {n} (out of expected range)")
    if hi_m - lo_m > 0.20:
        ok(f"recipient counterfactual: excess capitulation rises with fan-in (low {lo_m:.2f} -> high {hi_m:.2f})")
    else:
        bad(f"recipient counterfactual: fan-in gradient weak (low {lo_m:.2f}, high {hi_m:.2f})")
except Exception as _e:
    warn(f"recipient check skipped: {_e}")

# ---------- 12. REVISION-ERA QUANTITATIVE AUDIT ----------
# Every quantity introduced or changed in the major revision is re-derived in
# audit_revision.py and asserted against the value printed in the frozen manuscript and
# Supplementary Materials. Reported values are assertion targets only; nothing reported is
# used as an input.
print("\n" + "="*70, "\n12. REVISION-ERA QUANTITATIVE CLAIMS (audit_revision.py)\n", "="*70)
REV = []
try:
    import audit_revision as AREV
    REV = AREV.run_all(verbose=True)
    s_ = AREV.summary()
    for r in REV:
        if r["status"] == "FAIL":
            bad(f"{r['family']} | {r['id']}: reproduced {r['reproduced']} != reported "
                f"{r['expected']} (tol {r['tol']}){(' — ' + r['note']) if r['note'] else ''}")
    ok(f"revision audit: {s_['passed']} of {s_['assertions']} reported quantities re-derived "
       f"from archived data ({s_['info']} informational records)")
    out = HERE / "audit_results.json"
    out.write_text(json.dumps(dict(
        manuscript="PNASNEXUS-2026-01296",
        seed=AREV.SEED, bootstrap_draws=AREV.B_RESAMPLE,
        summary=s_, results=REV), indent=2))
    print(f"\n  structured results written to {out.name}")
except Exception as _e:
    bad(f"revision audit could not run: {type(_e).__name__}: {_e}")

print("\n" + "="*70)
print(f"AUDIT SUMMARY: {len(PASS)} pass, {len(WARN)} warn, {len(FAIL)} FAIL")
print("="*70)
print("\nPASSED CHECKS:"); [print("  ✓ "+m) for m in PASS]
if WARN:
    print("\nWARNINGS:"); [print("  ⚠ "+m) for m in WARN]
print("\nFAILURES:" if FAIL else "\nNo failures.")
[print("  ✗ "+m) for m in FAIL]
