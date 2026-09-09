"""Extract one complete worked trial for main-text/SI exposition (R1 presentation point 1,
and the scored "Repeatable Procedures: No").

Prints, for a single matched seed: the drawn state, every agent's likelihood pair and LLR,
round-0 beliefs, the verbatim round-1 message under BOTH content conditions, and the
resulting pooled belief. System python3.
"""
import json, math, statistics as st
from pathlib import Path
HERE = Path(__file__).parent
TAG, RG, TOP = "gpt-4.1-mini", "misleading_majority", "all_channel"

def logit(p, eps=1e-6):
    p = min(1-eps, max(eps, p)); return math.log(p/(1-p))
def pool(bs): return 1/(1+math.exp(-sum(logit(b) for b in bs)/len(bs)))

recs = {}
for line in open(HERE/"data"/TAG/"full"/"transcripts.jsonl"):
    o = json.loads(line)
    if o["regime"] == RG and o["topology"] == TOP and o["content_mode"] in ("evidence_only","conclusion_only"):
        recs[(o["content_mode"], o["seed"])] = o

seeds = sorted({s for _, s in recs} )
pick = None
for s in seeds:
    e, c = recs.get(("evidence_only", s)), recs.get(("conclusion_only", s))
    if not (e and c): continue
    truth_A = e["true_state"] == "A"
    pe = pool([a["belief"] for a in e["rounds"][-1]["agents"]])
    pc = pool([a["belief"] for a in c["rounds"][-1]["agents"]])
    # a representative trial: evidence ends correct, conclusion ends wrong
    if ((pe >= .5) == truth_A) and ((pc >= .5) != truth_A):
        pick = s; break

e, c = recs[("evidence_only", pick)], recs[("conclusion_only", pick)]
print(f"WORKED TRIAL — seed {pick}, {RG}, {TOP}, n={e['n_agents']}")
print(f"True state: {e['true_state']}   |   full-information Bayesian P(A) = {e['full_info_post_a']:.3f}"
      f"   |   naive vote share for A = {e['naive_vote_a']:.2f}")
print("\nPrivate signals (identical across both content conditions):")
print(f"  {'agent':>6}{'reading':>9}{'P(r|A)':>9}{'P(r|B)':>9}{'LLR':>9}{'normative P(A)':>16}{'round-0 belief':>16}")
b0 = {a["id"]: a["belief"] for a in e["rounds"][0]["agents"]}
for m in e["agents_meta"]:
    print(f"  {m['id']:>6}{m['signal_label']:>9}{m['p_a']:>9.2f}{m['p_b']:>9.2f}{m['llr']:>+9.2f}"
          f"{m['private_post_a']:>16.3f}{b0[m['id']]:>16.3f}")
print(f"\n  Round-0 pooled P(A) = {pool([b0[i] for i in sorted(b0)]):.3f}  "
      f"(truth is {e['true_state']}; a head-count of round-0 verdicts favours "
      f"{'A' if e['naive_vote_a']>0.5 else 'B'})")

for tagname, o in (("EVIDENCE SHARING", e), ("CONCLUSION SHARING", c)):
    print(f"\n{'-'*96}\n{tagname} — verbatim round-1 messages\n{'-'*96}")
    for a in o["rounds"][1]["agents"]:
        msg = (a.get("message") or "").replace("\n", " ")
        print(f"  agent {a['id']}: {msg[:200]}")
    term = [a["belief"] for a in o["rounds"][-1]["agents"]]
    print(f"\n  terminal beliefs: {[round(x,3) for x in term]}")
    p = pool(term)
    print(f"  terminal pooled P(A) = {p:.3f}  ->  collective says "
          f"{'A' if p>=.5 else 'B'}, truth is {o['true_state']}  "
          f"[{'CORRECT' if (p>=.5)==(o['true_state']=='A') else 'WRONG'}]")
