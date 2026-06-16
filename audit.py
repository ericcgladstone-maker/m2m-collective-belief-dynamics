"""DATA AUDIT — transparency / integrity / no-hallucination check.

Re-derives every headline number directly from the raw on-disk files and asserts it
matches what we reported (within rounding). Also checks structural integrity of the raw
data: trial counts, round structure, belief ranges, message realness, duplicate keys,
and raw(transcripts) vs derived(rounds.csv) consistency. Deterministic and re-runnable;
the point is to NOT trust prose summaries. Writes nothing it can't recompute.
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

print("\n" + "="*70)
print(f"AUDIT SUMMARY: {len(PASS)} pass, {len(WARN)} warn, {len(FAIL)} FAIL")
print("="*70)
print("\nPASSED CHECKS:"); [print("  ✓ "+m) for m in PASS]
if WARN:
    print("\nWARNINGS:"); [print("  ⚠ "+m) for m in WARN]
print("\nFAILURES:" if FAIL else "\nNo failures.")
[print("  ✗ "+m) for m in FAIL]
