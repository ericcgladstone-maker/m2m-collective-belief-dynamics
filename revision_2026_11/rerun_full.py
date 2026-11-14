"""Full reruns requested by R1 (PNAS Nexus minor revision), corrected protocol.

Regenerates the conversations of the agreed cells with the same seeds:
  n = 5 all-channel misleading-majority evidence and conclusion, gpt-4.1-mini and claude-sonnet-4-6;
  n = 11 all-channel evidence and conclusion, concordant and misleading majority, gpt-4.1-mini.

Protocol, as in engine.run_trial except where marked CHANGED:
  - task instances from tasks.generate_trial(seed, regime, n) with the September seeds;
    the dry run checks that every regenerated signal and true state matches the logged trial;
  - prompts are the deposited agents.read_belief / agents.make_message prompts, captured
    by stubbing provider.chat, so wording is byte-identical to the deposit;
  - messages: max_tokens 120, temperature 0.7 (unchanged); stop reason now logged;
  - readouts: temperature 0, CHANGED max_tokens 320 -> 2048; CHANGED strict parser
    (whole FINAL= value validated; invalid readouts are recorded as invalid, never set to 0.5);
    the legacy parse is also stored for comparison. Readouts are side calls and never enter
    any later prompt, so the parser cannot affect the conversation;
  - every call stores the first raw response, stop reason, usage, returned model, attempts;
  - retry policy: only transport failures (no response) are retried, up to 6 times. A response
    that arrives is never retried. If a message call fails after 6 attempts the trial is
    recorded as failed and not used.

Usage:
  python3 rerun_full.py --dry-run
  python3 rerun_full.py --workers 12
"""
from __future__ import annotations
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_readout as RR  # noqa: E402  (optional keys, puts the deposited code on sys.path)
import agents, provider, engine  # noqa: E402
from tasks import generate_trial  # noqa: E402

# Write output to a local, non-synced directory (M2M_OUT): on 2026-11-13 a file-sync client
# replaced an open output file and the first ~55 trials were lost to an unlinked inode.
OUT = RR.OUT
READ = dict(max_tokens=2048, temperature=0.0)
MSG = dict(max_tokens=120, temperature=0.7)
CELLS = ["gpt5_mm_ev", "gpt5_mm_co", "cl5_mm_ev", "cl5_mm_co",
         "gpt11_cc_ev", "gpt11_cc_co", "gpt11_mm_ev", "gpt11_mm_co"]
R_COMM = 3


def capture(fn, *args):
    cap = {}
    def fake(messages, **kw):
        cap["messages"] = messages; cap["kw"] = kw
        return {"text": "FINAL=0.5", "usage": {}}
    real = provider.chat; provider.chat = fake
    try:
        fn(*args)
    finally:
        provider.chat = real
    return cap["messages"], cap["kw"]


def call(model, messages, s):
    f = RR.call_anthropic if model.startswith("claude") else RR.call_openai
    return f(model, messages, s["max_tokens"], s["temperature"])


def readout(model, desc, block):
    msgs, kw = capture(agents.read_belief, desc, block, "stub")
    assert kw["max_tokens"] == 320 and kw["temperature"] == 0.0   # deposit unchanged
    r = call(model, msgs, READ)
    if r["text"] is not None:
        r["legacy_parse"] = RR.legacy_parse(r["text"])
        r["strict_parse"], r["strict_status"] = RR.strict_parse(r["text"])
    else:
        r["legacy_parse"] = r["strict_parse"] = None; r["strict_status"] = "no_response"
    return r


def message(model, desc, block, mode):
    msgs, kw = capture(agents.make_message, desc, block, mode, "stub")
    assert kw["max_tokens"] == MSG["max_tokens"] and kw["temperature"] == MSG["temperature"]
    r = call(model, msgs, MSG)
    if r["text"] is not None:
        r["text"] = r["text"].strip()            # as agents.make_message does
    return r


def run_trial(cell, seed):
    tag, model, n, topo, mode, regime = RR.CELLS[cell]
    trial = generate_trial(seed, regime, n)
    adj = engine.receives_from(topo, n)
    desc = [s.describe() for s in trial.signals]
    rec = dict(cell=cell, seed=seed, model=model, n_agents=n, topology=topo, content_mode=mode,
               regime=regime, true_state=trial.true_state, read_settings=READ, msg_settings=MSG,
               agents_meta=[{"id": i, "p_a": s.p_a, "p_b": s.p_b, "llr": round(s.llr, 4)}
                            for i, s in enumerate(trial.signals)],
               rounds=[], status="ok", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["rounds"].append({"round": 0, "agents": [dict(id=i, received=[], readout=readout(model, desc[i], ""))
                                                 for i in range(n)]})
    msgs = {}
    for i in range(n):
        m = message(model, desc[i], "", mode)
        if m["text"] is None:
            rec["status"] = "failed_message"; return rec
        rec["rounds"][0]["agents"][i]["message"] = m; msgs[i] = m["text"]
    for r in range(1, R_COMM + 1):
        inbox = {i: [{"src": s, "text": msgs[s]} for s in adj[i]] for i in range(n)}
        blocks = {i: agents.format_received([(d["src"], d["text"]) for d in inbox[i]]) for i in range(n)}
        ra = [dict(id=i, received=inbox[i], readout=readout(model, desc[i], blocks[i])) for i in range(n)]
        if r < R_COMM:
            new = {}
            for i in range(n):
                m = message(model, desc[i], blocks[i], mode)
                if m["text"] is None:
                    rec["status"] = "failed_message"; rec["rounds"].append({"round": r, "agents": ra}); return rec
                ra[i]["message"] = m; new[i] = m["text"]
            msgs = new
        rec["rounds"].append({"round": r, "agents": ra})
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def plan():
    jobs, bad = [], []
    for cell in CELLS:
        tag, model, n, topo, mode, regime = RR.CELLS[cell]
        for t in RR.load_trials(cell):
            s = int(t["seed"]); g = generate_trial(s, regime, n)
            meta = {m["id"]: m for m in t["agents_meta"]}
            same = g.true_state == t["true_state"] and all(
                abs(sg.p_a - meta[i]["p_a"]) < 1e-9 and abs(sg.p_b - meta[i]["p_b"]) < 1e-9
                and sg.describe() == RR.sig_desc(meta[i]) for i, sg in enumerate(g.signals))
            if not same: bad.append((cell, s))
            jobs.append((cell, s))
    return jobs, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    print("parser self-test:"); pt = RR.parser_selftest()
    jobs, bad = plan()
    count = {}
    for c, _ in jobs: count[c] = count.get(c, 0) + 1
    print("trials per cell:", count, " total", len(jobs))
    print("seed regeneration mismatches:", len(bad), bad[:5])
    calls = {}
    for c, _ in jobs:
        n = RR.CELLS[c][2]; m = RR.CELLS[c][1]
        calls[m] = calls.get(m, 0) + n * (R_COMM + 1) + n * R_COMM
    print("API calls by model:", calls)
    ok = pt and not bad
    print("OFFLINE CHECKS", "PASS" if ok else "FAIL")
    if a.dry_run or not ok:
        return
    OUT.mkdir(exist_ok=True)
    path = OUT / "rerun_trials.jsonl"
    done = set()
    if path.exists():
        done = {(r["cell"], r["seed"]) for r in map(json.loads, open(path)) if r["status"] == "ok"}
    todo = [j for j in jobs if j not in done]
    # Claude cells first so a GPT balance stop cannot starve them
    todo.sort(key=lambda j: (not j[0].startswith("cl"), j))
    print(f"running {len(todo)} trials -> {path}")
    lock = threading.Lock()
    with open(path, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_trial, c, s) for c, s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 10 == 0: print(f"  {i}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
