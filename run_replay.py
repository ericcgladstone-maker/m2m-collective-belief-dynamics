"""First-conclusion replay — the causal (do-operator) test.

Correlationally we see: conclusion-sharing -> collective herds onto the wrong
majority. Does the conclusion CONTENT cause the terminal belief, holding the real
evidence fixed? We plant a confident first-mover conclusion visible to all agents
from round 0 (their private signals unchanged), and compare planting the CORRECT
vs the WRONG state on identical seeds:

  ACE = E[P(truth) | do(first_conclusion = correct)] - E[P(truth) | do(... = wrong)]

A large positive ACE means the injected conclusion *causes* the collective's
terminal belief — i.e., herding is driven by conclusion content, not evidence.
Run in the herding condition (misleading_majority / all_channel / conclusion_only).
"""
from __future__ import annotations
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed

import engine, metrics, config
from tasks import generate_trial

REGIME, TOPO, CONTENT, N_AGENTS, R_COMM = "misleading_majority", "all_channel", "conclusion_only", 5, 3


def _other(s):
    return "B" if s == "A" else "A"


def run_fork(seed, mode):
    t = generate_trial(seed, REGIME, N_AGENTS)
    sc = t.true_state if mode == "correct" else _other(t.true_state) if mode == "wrong" else None
    res = engine.run_trial(t, TOPO, CONTENT, r_comm=R_COMM, seed_conclusion=sc)
    last = engine.beliefs_at(res, len(res["rounds"]) - 1)
    cl = metrics.collective_logodds(last)
    return {"seed": seed, "mode": mode,
            "p_truth": metrics.p_truth_of(cl, t.true_state),
            "acc": metrics.accuracy_of(cl, t.true_state),
            "usage": res["usage"]}


def main(n=40, workers=10):
    jobs = [(s, m) for s in range(n) for m in ("correct", "wrong", "none")]
    rows, tin, tout = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_fork, s, m) for s, m in jobs]
        for f in as_completed(futs):
            try:
                r = f.result(); rows.append(r)
                tin += r["usage"]["prompt_tokens"]; tout += r["usage"]["completion_tokens"]
            except Exception as e:
                print(f"  [warn] fork dropped: {e}", flush=True)

    def agg(mode):
        v = [r["p_truth"] for r in rows if r["mode"] == mode]
        a = [r["acc"] for r in rows if r["mode"] == mode]
        if not v:
            return (float("nan"), float("nan"), 0)
        return (sum(v) / len(v), sum(a) / len(a), len(v))

    pc, ac, nc = agg("correct"); pw, aw, nw = agg("wrong"); pn, an, nn = agg("none")
    print(f"\n=== FIRST-CONCLUSION REPLAY (do-operator) — {REGIME}/{TOPO}/{CONTENT} | model {config.AGENT_MODEL} ===")
    print(f"  planted CORRECT first conclusion (n={nc}):  P(truth)={pc:.2f}  acc={ac:.2f}")
    print(f"  planted WRONG   first conclusion (n={nw}):  P(truth)={pw:.2f}  acc={aw:.2f}")
    print(f"  no plant (control)              (n={nn}):  P(truth)={pn:.2f}  acc={an:.2f}")
    print(f"  ACE (correct - wrong) on P(truth) = {pc - pw:+.3f}")
    print(f"  -> large positive ACE = the injected conclusion CAUSES terminal collective belief (herding is conclusion-driven).")
    pin, pout = config._PRICES.get(config.AGENT_MODEL, (config.COST_IN, config.COST_OUT))
    cost = tin * pin + tout * pout
    print(f"  spend ${cost:.4f}  (in={tin} out={tout} tok)")
    # persist raw per-fork data so the causal result is auditable / re-derivable
    from pathlib import Path
    import csv as _csv, json as _json
    out = Path(__file__).parent / "data" / f"replay-{config.AGENT_MODEL}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "replay.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["seed", "mode", "p_truth", "acc"])
        for r in sorted(rows, key=lambda r: (r["seed"], r["mode"])):
            w.writerow([r["seed"], r["mode"], round(r["p_truth"], 4), r["acc"]])
    (out / "run_meta.json").write_text(_json.dumps({"model": config.AGENT_MODEL,
        "regime": REGIME, "topology": TOPO, "content": CONTENT, "n_seeds": n,
        "ACE_ptruth": round(pc - pw, 4), "p_correct": round(pc, 4), "p_wrong": round(pw, 4),
        "p_control": round(pn, 4), "n_forks": len(rows), "tokens_in": tin, "tokens_out": tout,
        "cost_usd": round(cost, 4)}, indent=2))
    print(f"  saved -> {out}/replay.csv + run_meta.json")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(n=n)
