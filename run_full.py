"""Full powered run: concurrent, resumable, crux-prioritized, daily-cap aware,
with complete transcript logging.

Cells (per regime): baseline/none; all_channel x {evidence_only, conclusion_only,
both}; wheel/both; circle/both. Regimes: misleading_majority (crux, run FIRST so a
daily-cap interruption starves the control, not the crux) + concordant (control).
Paired seeds across cells.

RESUMABLE: existing data/full/transcripts.jsonl is loaded; already-completed
(regime,topology,content_mode,seed) cells are skipped; new results are merged and
all outputs rewritten. Re-run after the daily request cap (RPD) resets to finish.

Outputs in data/ (see data/README.md): full/transcripts.jsonl, messages.csv,
beliefs.csv, rounds.csv, run_meta.json.
"""
from __future__ import annotations
import csv, json, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import engine, metrics, provider, config
from tasks import generate_trial

REGIMES = ["misleading_majority", "concordant"]   # crux first (priority on a quota)
TOPO_CONTENT = [
    ("baseline", "none"),
    ("all_channel", "evidence_only"),
    ("all_channel", "conclusion_only"),
    ("all_channel", "both"),
    ("wheel", "both"),
    ("circle", "both"),
    # full topology x content factorial fill-in (2026-06-11)
    ("wheel", "evidence_only"),
    ("wheel", "conclusion_only"),
    ("circle", "evidence_only"),
    ("circle", "conclusion_only"),
    # expanded topology set: chain + hierarchy x all contents (2026-06-11)
    ("chain", "evidence_only"),
    ("chain", "conclusion_only"),
    ("chain", "both"),
    ("hierarchy", "evidence_only"),
    ("hierarchy", "conclusion_only"),
    ("hierarchy", "both"),
]
N_AGENTS = 5
R_COMM = 3
OUT_TAG = config.DATA_TAG   # overridable (e.g., for the n=11 small-world condition -> separate data dir)
COST_IN, COST_OUT = config.COST_IN, config.COST_OUT

_lock = threading.Lock()
_done = {"n": 0}


def run_one(regime, topo, content, seed):
    trial = generate_trial(seed, regime, N_AGENTS)
    res = engine.run_trial(trial, topo, content, r_comm=R_COMM)
    with _lock:
        _done["n"] += 1
        if _done["n"] % 50 == 0:
            print(f"  ... {_done['n']} new trials done", flush=True)
    return res


def load_existing(out: Path):
    p = out / "full" / "transcripts.jsonl"
    res = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                res.append(json.loads(line))
    return res


def main(reps=50, workers=8):
    out = Path(__file__).parent / "data" / OUT_TAG
    (out / "full").mkdir(parents=True, exist_ok=True)

    existing = load_existing(out)
    done = {(r["regime"], r["topology"], r["content_mode"], r["seed"]) for r in existing}
    jobs = [(rg, tp, ct, s) for rg in REGIMES for (tp, ct) in TOPO_CONTENT
            for s in range(reps) if (rg, tp, ct, s) not in done]
    print(f"=== FULL RUN  reps={reps} | existing={len(existing)} | todo={len(jobs)} | "
          f"workers={workers} r_comm={R_COMM} ===", flush=True)

    new_results, failures, quota_hit = [], 0, False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        try:
            for f in as_completed(futs):
                try:
                    new_results.append(f.result())
                except provider.DailyLimitError:
                    quota_hit = True
                    break
                except Exception as e:
                    failures += 1
                    with _lock:
                        print(f"  [warn] trial dropped: {e}", flush=True)
        finally:
            if quota_hit:
                for fu in futs:
                    fu.cancel()
                print("  [STOP] daily request cap (RPD) reached — saving progress; "
                      "re-run after reset to finish.", flush=True)

    results = existing + new_results
    if not results:
        print("No results to write."); return

    order = {(rg, tp, ct): i for i, (rg, (tp, ct)) in
             enumerate((rg, tc) for rg in REGIMES for tc in TOPO_CONTENT)}
    # order.get fallback: results loaded from disk may include cells not in the current
    # TOPO_CONTENT (e.g. when powering a subset of cells); sort those to the end rather
    # than KeyError-ing AFTER the expensive trials have run but BEFORE they are written.
    results.sort(key=lambda r: (order.get((r["regime"], r["topology"], r["content_mode"]), 1_000_000), r["seed"]))

    with open(out / "full" / "transcripts.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open(out / "messages.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "topology", "content_mode", "seed", "round", "agent",
                    "n_received", "received_from", "message_text"])
        for r in results:
            for rd in r["rounds"]:
                for a in rd["agents"]:
                    if a["message"] is not None:
                        srcs = ";".join(str(d["src"]) for d in a["received"])
                        w.writerow([r["regime"], r["topology"], r["content_mode"], r["seed"],
                                    rd["round"], a["id"], len(a["received"]), srcs,
                                    a["message"].replace("\n", " ").strip()])

    with open(out / "beliefs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "topology", "content_mode", "seed", "round", "agent",
                    "belief_pA", "signal_p_a", "signal_p_b", "llr", "private_post_a",
                    "n_received"])
        for r in results:
            meta = {m["id"]: m for m in r["agents_meta"]}
            for rd in r["rounds"]:
                for a in rd["agents"]:
                    m = meta[a["id"]]
                    w.writerow([r["regime"], r["topology"], r["content_mode"], r["seed"],
                                rd["round"], a["id"], round(a["belief"], 4),
                                m["p_a"], m["p_b"], m["llr"], m["private_post_a"],
                                len(a["received"])])

    summary = {}
    with open(out / "rounds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "topology", "content_mode", "seed", "true_state",
                    "full_info_post_a", "naive_vote_a", "round", "e_bar", "D",
                    "coll_mean", "coll_logodds", "acc_mean", "acc_logodds",
                    "regret_mean", "regret_logodds", "p_truth_logodds"])
        for r in results:
            ts, fia = r["true_state"], r["full_info_post_a"]
            for ri in range(len(r["rounds"])):
                b = engine.beliefs_at(r, ri)
                cm, cl = metrics.collective_belief(b), metrics.collective_logodds(b)
                w.writerow([r["regime"], r["topology"], r["content_mode"], r["seed"], ts,
                            fia, r["naive_vote_a"], ri,
                            round(metrics.mean_individual_error(b, ts), 4),
                            round(metrics.diversity(b), 4), round(cm, 4), round(cl, 4),
                            metrics.accuracy_of(cm, ts), metrics.accuracy_of(cl, ts),
                            round(metrics.regret_of(cm, fia), 4),
                            round(metrics.regret_of(cl, fia), 4),
                            round(metrics.p_truth_of(cl, ts), 4)])
            last = engine.beliefs_at(r, len(r["rounds"]) - 1)
            key = (r["regime"], r["topology"], r["content_mode"])
            s = summary.setdefault(key, {"acc_m": [], "acc_l": [], "reg_l": [],
                                         "e_traj": {}, "D_traj": {}})
            s["acc_m"].append(metrics.accuracy_of(metrics.collective_belief(last), ts))
            s["acc_l"].append(metrics.accuracy_of(metrics.collective_logodds(last), ts))
            s["reg_l"].append(metrics.regret_of(metrics.collective_logodds(last), fia))
            for ri in range(len(r["rounds"])):
                bb = engine.beliefs_at(r, ri)
                s["e_traj"].setdefault(ri, []).append(metrics.mean_individual_error(bb, ts))
                s["D_traj"].setdefault(ri, []).append(metrics.diversity(bb))

    tin = sum(r["usage"]["prompt_tokens"] for r in results)
    tout = sum(r["usage"]["completion_tokens"] for r in results)
    # price by the ACTUAL model (module COST_IN/OUT were fixed at import for the default
    # model, which under-reported non-default substrates like Sonnet ~9x).
    pin, pout = config._PRICES.get(config.AGENT_MODEL, (COST_IN, COST_OUT))
    cost = tin * pin + tout * pout
    (out / "run_meta.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"), "model": config.AGENT_MODEL,
        "reps_target": reps, "regimes": REGIMES, "topo_content": TOPO_CONTENT,
        "n_agents": N_AGENTS, "r_comm": R_COMM, "n_trials_total": len(results),
        "new_this_run": len(new_results), "trials_dropped": failures,
        "quota_hit": quota_hit, "tokens_in": tin, "tokens_out": tout,
        "cost_usd_total": round(cost, 4)}, indent=2))

    print(f"\n=== SUMMARY (terminal round) ===\n")
    print(f"{'regime':20s} {'topology':12s} {'content':16s} {'n':>3s} {'acc_m':>6s} "
          f"{'acc_lo':>6s} {'regret':>6s}  ē_traj / D_traj")
    for rg in REGIMES:
        for tp, ct in TOPO_CONTENT:
            s = summary.get((rg, tp, ct))
            if not s:
                print(f"{rg:20s} {tp:12s} {ct:16s}   0   (none yet)"); continue
            n = len(s["acc_m"]); nr = len(s["e_traj"])
            et = [round(sum(s["e_traj"][i]) / len(s["e_traj"][i]), 3) for i in range(nr)]
            dt = [round(sum(s["D_traj"][i]) / len(s["D_traj"][i]), 3) for i in range(nr)]
            print(f"{rg:20s} {tp:12s} {ct:16s} {n:3d} "
                  f"{sum(s['acc_m'])/n:6.2f} {sum(s['acc_l'])/n:6.2f} "
                  f"{sum(s['reg_l'])/n:6.3f}  {et} / {dt}")
    print(f"\nTotal trials on disk: {len(results)} (+{len(new_results)} this run)")
    print(f"Cumulative spend: ${cost:.4f}  | quota_hit={quota_hit}")
    if quota_hit:
        print("Re-run `python3 run_full.py 50` after the daily cap resets to finish the missing cells.")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(reps=reps, workers=workers)
