"""Hidden-profile generalization run (reviewer option 2). Self-contained pipeline that
reuses engine.receives_from (topology adjacency) and metrics (scoring), with naturalistic
prose prompts for the materials-comparison task in tasks_hidden.py. Same evidence-vs-
conclusion x topology manipulation as the formal task. Writes rounds.csv in the SAME schema
as run_full so analyze_stats.py scores it directly. Belief = P(Material Zeta is better),
mapped to P('A') so metrics reuse unchanged (A=Zeta, B=Theta)."""
from __future__ import annotations
import csv, json, sys, threading, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import engine, metrics, provider, config
from tasks_hidden import generate_hidden_trial

SYSTEM = (
    "You are one of several analysts evaluating which of two materials, Material Zeta or "
    "Material Theta, is the better choice for an engineering application. Each analyst has "
    "privately reviewed a DIFFERENT engineering test, and no single test is decisive: the "
    "correct choice is the one favored by the weight of ALL the evidence combined. Weigh a "
    "decisive, large-margin finding far more heavily than a marginal, small-margin one. "
    "Combine independent findings; do not double-count a finding that reaches you again after "
    "being relayed by someone else."
)


def _parse_final(txt: str) -> float:
    m = re.search(r"FINAL\s*=\s*([0-9]*\.?[0-9]+)", txt, re.IGNORECASE)
    nums = re.findall(r"[0-9]*\.[0-9]+|[0-9]+", txt)
    raw = m.group(1) if m else (nums[-1] if nums else None)
    if raw is None:
        return 0.5
    v = float(raw)
    if v > 1.0:
        v /= 100.0
    return min(1.0, max(0.0, v))


def read_belief(finding_desc, received_block, model):
    user = (
        f"Your private evidence: you reviewed {finding_desc}.\n{received_block}"
        "Combine your own finding with each DISTINCT finding reported by others, weighting "
        "decisive findings far more than marginal ones. Work through it in 1-3 short sentences, "
        "then end with exactly this line:\nFINAL=<your probability that Material Zeta is the "
        "better choice, between 0 and 1>"
    )
    out = provider.chat([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}], model=model, max_tokens=320, temperature=0.0)
    return _parse_final(out["text"]), out.get("usage", {})


def make_message(finding_desc, received_block, content_mode, model):
    if content_mode == "evidence_only":
        instr = ("Write a short message sharing ONLY your finding: which test you reviewed and "
                 "what it showed (which material it favored, and whether the margin was decisive "
                 "or marginal). Do NOT say which material you overall think is better. One or two sentences.")
    elif content_mode == "conclusion_only":
        instr = ("Write a short message sharing ONLY your current best guess of which material is "
                 "the better choice (Zeta or Theta) and how confident you are. Do NOT describe your "
                 "test or finding. One sentence.")
    elif content_mode == "both":
        instr = ("Write a short message sharing your finding (test + which material it favored, "
                 "decisive or marginal) AND your current best guess of the better material with your "
                 "confidence. Two or three sentences.")
    else:
        raise ValueError(content_mode)
    user = f"Your private evidence: you reviewed {finding_desc}.\n{received_block}{instr}"
    out = provider.chat([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}], model=model, max_tokens=120, temperature=0.7)
    return out["text"].strip(), out.get("usage", {})


def fmt_received(items):
    if not items:
        return ""
    return "Messages you received this round:\n" + "\n".join(
        f"  - [from analyst {s}] {t}" for s, t in items) + "\n"


def run_trial(trial, topology, content_mode, model, r_comm=3):
    n = len(trial.findings)
    adj = engine.receives_from(topology, n)
    desc = [f.describe() for f in trial.findings]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    def acc(u):
        usage["prompt_tokens"] += u.get("prompt_tokens", 0); usage["completion_tokens"] += u.get("completion_tokens", 0)
    rounds = []
    r0 = []
    for i in range(n):
        b, u = read_belief(desc[i], "", model); acc(u)
        r0.append({"id": i, "belief": b})
    rounds.append(r0)
    has = topology != "baseline" and r_comm > 0
    if has:
        msgs = {}
        for i in range(n):
            m, u = make_message(desc[i], "", content_mode, model); acc(u); msgs[i] = m
        for r in range(1, r_comm + 1):
            inb = {i: [(s, msgs[s]) for s in adj[i]] for i in range(n)}
            ra = []
            for i in range(n):
                b, u = read_belief(desc[i], fmt_received(inb[i]), model); acc(u)
                ra.append({"id": i, "belief": b})
            if r < r_comm:
                nm = {}
                for i in range(n):
                    m, u = make_message(desc[i], fmt_received(inb[i]), content_mode, model); acc(u); nm[i] = m
                msgs = nm
            rounds.append(ra)
    return {"seed": trial.seed, "regime": trial.regime, "topology": topology,
            "content_mode": content_mode, "true_zeta": trial.true_zeta,
            "snippet_majority_zeta": trial.snippet_majority_zeta,
            "rounds": rounds, "usage": usage}


REGIMES = ["misleading_majority", "concordant"]
TOPO_CONTENT = [("baseline", "none"), ("all_channel", "evidence_only"),
                ("all_channel", "conclusion_only"), ("all_channel", "both"),
                ("circle", "evidence_only"), ("circle", "conclusion_only")]
_lock = threading.Lock(); _done = {"n": 0}


def run_one(regime, topo, content, seed, model, n_agents):
    t = generate_hidden_trial(seed, regime, n_agents)
    res = run_trial(t, topo, content, model)
    with _lock:
        _done["n"] += 1
        if _done["n"] % 50 == 0:
            print(f"  ... {_done['n']} trials done", flush=True)
    return res


def beliefs_at(res, ri):
    return [a["belief"] for a in res["rounds"][ri]]


def main(reps=50, workers=8, model=None, n_agents=5):
    model = model or config.AGENT_MODEL
    out = Path(__file__).parent / "data" / f"hidden-{model}"
    (out / "full").mkdir(parents=True, exist_ok=True)
    p = out / "full" / "transcripts.jsonl"
    existing = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    done = {(r["regime"], r["topology"], r["content_mode"], r["seed"]) for r in existing}
    jobs = [(rg, tp, ct, s) for rg in REGIMES for (tp, ct) in TOPO_CONTENT
            for s in range(reps) if (rg, tp, ct, s) not in done]
    print(f"=== HIDDEN RUN model={model} reps={reps} existing={len(existing)} todo={len(jobs)} ===", flush=True)
    new, fails, quota = [], 0, False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, rg, tp, ct, s, model, n_agents) for (rg, tp, ct, s) in jobs]
        for f in as_completed(futs):
            try:
                new.append(f.result())
            except provider.DailyLimitError:
                quota = True; break
            except Exception as e:
                fails += 1; print(f"  [warn] dropped: {e}", flush=True)
    results = existing + new
    if not results:
        print("no results"); return
    p.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    # rounds.csv in run_full schema (A=Zeta). ideal posterior = truth indicator.
    with open(out / "rounds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "topology", "content_mode", "seed", "true_state", "full_info_post_a",
                    "naive_vote_a", "round", "e_bar", "D", "coll_mean", "coll_logodds",
                    "acc_mean", "acc_logodds", "regret_mean", "regret_logodds", "p_truth_logodds"])
        for r in results:
            ts = "A" if r["true_zeta"] else "B"
            fia = 1.0 if r["true_zeta"] else 0.0
            nv = 1.0 if r["snippet_majority_zeta"] else 0.0
            for ri in range(len(r["rounds"])):
                b = beliefs_at(r, ri)
                cm, cl = metrics.collective_belief(b), metrics.collective_logodds(b)
                w.writerow([r["regime"], r["topology"], r["content_mode"], r["seed"], ts, fia, nv, ri,
                            round(metrics.mean_individual_error(b, ts), 4), round(metrics.diversity(b), 4),
                            round(cm, 4), round(cl, 4), metrics.accuracy_of(cm, ts), metrics.accuracy_of(cl, ts),
                            round(metrics.regret_of(cm, fia), 4), round(metrics.regret_of(cl, fia), 4),
                            round(metrics.p_truth_of(cl, ts), 4)])
    tin = sum(r["usage"]["prompt_tokens"] for r in results); tout = sum(r["usage"]["completion_tokens"] for r in results)
    pin, pout = config._PRICES.get(model, (config.COST_IN, config.COST_OUT))
    cost = tin * pin + tout * pout
    (out / "run_meta.json").write_text(json.dumps({"model": model, "reps": reps, "n_agents": n_agents,
        "n_trials_total": len(results), "new_this_run": len(new), "trials_dropped": fails,
        "quota_hit": quota, "tokens_in": tin, "tokens_out": tout, "cost_usd_total": round(cost, 4)}, indent=2))
    print(f"trials={len(results)} (+{len(new)}) dropped={fails} cost=${cost:.4f} quota_hit={quota}")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(reps=reps, workers=workers)
