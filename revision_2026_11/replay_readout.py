"""Readout-only remeasurement of logged September conversations (PNAS Nexus minor revision).

Holds each logged conversation fixed and re-asks only the belief readout, using the
deposited code's own prompt builder (agents.read_belief) at the original settings.
For every readout it saves the FIRST raw response, the stop reason, usage, the legacy
parse (agents._parse_final) and a strict parse, all on the same text.

This is a remeasurement, not a recovery of the September readouts: the legacy parser is
many-to-one, so agreement with the stored belief does not show the text was the same.

Retry policy: only transport failures (no response received) are retried, up to 6 times,
and the attempt count is logged. A response that arrives is never retried, whatever it
contains. Invalid or truncated responses are recorded as such and are never converted to
0.5 or dropped.

Usage:
  python3 replay_readout.py --plan pilot --dry-run     # offline checks, no API calls
  python3 replay_readout.py --plan pilot               # pilot run (paid)
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, re, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402  (locations; see paths.py)
GITHUB = paths.REPO                  # public deposit: prompt builder + legacy parser (read-only)
DATA = paths.DATA                    # September data (the OSF archive m2m_raw_data.zip, unzipped)
REV = paths.REV                      # revision records (OSF folder revision-2026-11)
OUT = paths.OUT                      # new model calls are written here

# optional key file, read only if present; keys are needed only for new model calls
if paths.ENV.exists():
    for line in paths.ENV.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(GITHUB))
import agents, provider  # noqa: E402

# ---------------------------------------------------------------- original settings
# Standard (non-thinking) readout call in agents.read_belief: max_tokens=320, temperature=0.0.
SETTINGS = {"gpt-4.1-mini": dict(max_tokens=320, temperature=0.0),
            "claude-sonnet-4-6": dict(max_tokens=320, temperature=0.0)}

# ---------------------------------------------------------------- cells
# (data dir, model, n_agents, topology, content_mode, regime)
CELLS = {
    "gpt5_mm_ev":   ("gpt-4.1-mini", "gpt-4.1-mini", 5, "all_channel", "evidence_only", "misleading_majority"),
    "gpt5_mm_co":   ("gpt-4.1-mini", "gpt-4.1-mini", 5, "all_channel", "conclusion_only", "misleading_majority"),
    "cl5_mm_ev":    ("claude-sonnet-4-6", "claude-sonnet-4-6", 5, "all_channel", "evidence_only", "misleading_majority"),
    "cl5_mm_co":    ("claude-sonnet-4-6", "claude-sonnet-4-6", 5, "all_channel", "conclusion_only", "misleading_majority"),
    "gpt11_cc_ev":  ("gpt-4.1-mini_n11", "gpt-4.1-mini", 11, "all_channel", "evidence_only", "concordant"),
    "gpt11_cc_co":  ("gpt-4.1-mini_n11", "gpt-4.1-mini", 11, "all_channel", "conclusion_only", "concordant"),
    "gpt11_mm_ev":  ("gpt-4.1-mini_n11", "gpt-4.1-mini", 11, "all_channel", "evidence_only", "misleading_majority"),
    "gpt11_mm_co":  ("gpt-4.1-mini_n11", "gpt-4.1-mini", 11, "all_channel", "conclusion_only", "misleading_majority"),
    "gpt5_mm_co_hier":  ("gpt-4.1-mini", "gpt-4.1-mini", 5, "hierarchy", "conclusion_only", "misleading_majority"),
    "gpt5_mm_co_chain": ("gpt-4.1-mini", "gpt-4.1-mini", 5, "chain", "conclusion_only", "misleading_majority"),
    "gpt5_mm_co_wheel": ("gpt-4.1-mini", "gpt-4.1-mini", 5, "wheel", "conclusion_only", "misleading_majority"),
}
PILOT_CELLS = ["gpt5_mm_ev", "gpt5_mm_co", "cl5_mm_ev", "cl5_mm_co", "gpt11_cc_ev", "gpt11_cc_co"]
PLANS = {"pilot": dict(cells=PILOT_CELLS, trials_per_truth=2, max_tokens=320),
         # larger-cap readout replay on ALL logged trials of the agreed cells (fixed conversations)
         "cap2048": dict(cells=list(CELLS), trials_per_truth=None, max_tokens=2048)}


# ---------------------------------------------------------------- parsers
legacy_parse = agents._parse_final

_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_FINAL_LINE = re.compile(r"^\s*\**\s*FINAL\s*=\s*(.*?)\s*\**\s*$", re.IGNORECASE)

def strict_parse(txt: str):
    """Validate the WHOLE FINAL= value. Returns (value or None, status).
    status: ok | no_final | malformed | out_of_range | multiple_final."""
    lines = [m.group(1) for m in (_FINAL_LINE.match(l) for l in txt.splitlines()) if m]
    if not lines:
        return None, "no_final"
    if len(set(lines)) > 1:
        return None, "multiple_final"
    raw = lines[-1].rstrip(".")
    if not re.fullmatch(_NUM, raw):
        return None, "malformed"
    v = float(raw)
    if not (0.0 <= v <= 1.0) or math.isnan(v):
        return None, "out_of_range"
    return v, "ok"


def parser_selftest():
    # R1's examples (legacy) and the intended strict reading
    cases = [("FINAL=1e-6", 1.0, 1e-6), ("FINAL=9.5e-1", 0.095, 0.95), ("FINAL=3e-6", 0.03, 3e-6),
             ("FINAL=1/5", 1.0, None), ("x 0.79/0.21=3.76", 0.0376, None), ("no number", 0.5, None),
             ("FINAL=5.25e-6", 0.0525, 5.25e-6), ("FINAL=0.9999", 0.9999, 0.9999),
             ("work\nFINAL=0.02", 0.02, 0.02), ("FINAL=1.2", 0.012, None)]
    ok = True
    for txt, leg, st in cases:
        l = legacy_parse(txt); s, _ = strict_parse(txt)
        good = abs(l - leg) < 1e-12 and (s == st if st is None else (s is not None and abs(s - st) < 1e-15))
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {txt!r:24} legacy={l!r:<8} strict={s!r}")
    return ok


# ---------------------------------------------------------------- prompt reconstruction
def P(v):
    return ast.literal_eval(v) if isinstance(v, str) and v[:1] in "[{" else v

def sig_desc(meta):
    from tasks import Signal
    return Signal(p_a=meta["p_a"], p_b=meta["p_b"], label=meta["signal_label"]).describe()

def build_messages(desc, received):
    """Capture the exact messages agents.read_belief sends, by stubbing provider.chat."""
    cap = {}
    def fake_chat(messages, **kw):
        cap["messages"] = messages; cap["kw"] = kw
        return {"text": "FINAL=0.5", "usage": {}}
    real = provider.chat
    provider.chat = fake_chat
    try:
        agents.read_belief(desc, agents.format_received([(d["src"], d["text"]) for d in received]),
                           model="stub")
    finally:
        provider.chat = real
    return cap["messages"], cap["kw"]

def load_trials(cell):
    tag, model, n, topo, mode, regime = CELLS[cell]
    out = []
    for line in open(DATA / tag / "full" / "transcripts.jsonl"):
        t = json.loads(line)
        if t["topology"] == topo and t["content_mode"] == mode and t["regime"] == regime \
                and int(t.get("n_agents", n)) == n:
            t["rounds"] = P(t["rounds"]); t["agents_meta"] = P(t["agents_meta"])
            out.append(t)
    return sorted(out, key=lambda t: int(t["seed"]))

def select(cell, k):
    ts = load_trials(cell)
    if k is None:
        return ts
    return [t for t in ts if t["true_state"] == "A"][:k] + [t for t in ts if t["true_state"] == "B"][:k]

def readout_jobs(plan):
    jobs, checks = [], {"delivery_mismatch": 0, "deliveries": 0, "settings_mismatch": 0}
    for cell in plan["cells"]:
        tag, model, n, topo, mode, regime = CELLS[cell]
        for t in select(cell, plan["trials_per_truth"]):
            meta = {m["id"]: m for m in t["agents_meta"]}
            rounds = t["rounds"]
            for r in rounds:
                for a in r["agents"]:
                    # deliveries in round r must equal round r-1 outgoing messages of the sources
                    if r["round"] >= 1:
                        prev = {x["id"]: x["message"] for x in rounds[r["round"] - 1]["agents"]}
                        for d in a["received"]:
                            checks["deliveries"] += 1
                            if d["src"] >= 0 and prev.get(d["src"]) != d["text"]:
                                checks["delivery_mismatch"] += 1
                    msgs, kw = build_messages(sig_desc(meta[a["id"]]), a["received"])
                    if kw.get("max_tokens") != SETTINGS[model]["max_tokens"] or \
                            kw.get("temperature") != SETTINGS[model]["temperature"]:
                        checks["settings_mismatch"] += 1
                    key = f'{cell}|{t["seed"]}|{r["round"]}|{a["id"]}'
                    jobs.append(dict(key=key, cell=cell, model=model, max_tokens=plan["max_tokens"], seed=int(t["seed"]),
                                     truth=t["true_state"], round=r["round"], agent=a["id"],
                                     logged_belief=float(a["belief"]), messages=msgs,
                                     prompt_sha1=hashlib.sha1(json.dumps(msgs).encode()).hexdigest()))
    return jobs, checks


# ---------------------------------------------------------------- callers (first response kept)
def call_openai(model, messages, max_tokens, temperature):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    attempts, last = 0, None
    for attempts in range(1, 7):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                d = json.loads(resp.read().decode())
            ch = d["choices"][0]
            return dict(text=ch["message"].get("content") or "", stop_reason=ch.get("finish_reason"),
                        usage=d.get("usage", {}), model_returned=d.get("model"), attempts=attempts)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (408, 409, 425, 429, 431) or e.code >= 500:
                time.sleep(min(2 ** attempts, 30)); continue
            break
        except Exception as e:  # network
            last = f"network: {e}"; time.sleep(min(2 ** attempts, 30))
    return dict(text=None, stop_reason=None, usage={}, model_returned=None, attempts=attempts, error=last)

_ac = None
def call_anthropic(model, messages, max_tokens, temperature):
    global _ac
    import anthropic
    if _ac is None:
        _ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0,
                                  default_headers={"anthropic-version": "2023-06-01"})
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    attempts, last = 0, None
    for attempts in range(1, 7):
        try:
            r = _ac.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                    system=system, messages=msgs)
            text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", None) == "text")
            return dict(text=text, stop_reason=r.stop_reason, model_returned=r.model, attempts=attempts,
                        usage={"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens})
        except anthropic.APIStatusError as e:
            last = f"HTTP {e.status_code}: {str(e)[:300]}"
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                time.sleep(min(2 ** attempts, 30)); continue
            break
        except anthropic.APIConnectionError as e:
            last = f"network: {e}"; time.sleep(min(2 ** attempts, 30))
    return dict(text=None, stop_reason=None, usage={}, model_returned=None, attempts=attempts, error=last)


def run_job(j):
    s = dict(SETTINGS[j["model"]], max_tokens=j["max_tokens"])
    f = call_anthropic if j["model"].startswith("claude") else call_openai
    res = f(j["model"], j["messages"], s["max_tokens"], s["temperature"])
    rec = {k: v for k, v in j.items() if k != "messages"}
    rec.update(res, settings=s, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if res["text"] is not None:
        rec["legacy_parse"] = legacy_parse(res["text"])
        rec["strict_parse"], rec["strict_status"] = strict_parse(res["text"])
    else:
        rec["legacy_parse"] = rec["strict_parse"] = None; rec["strict_status"] = "no_response"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="pilot", choices=list(PLANS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    plan = PLANS[a.plan]

    print("parser self-test:"); pt = parser_selftest()
    jobs, checks = readout_jobs(plan)
    print(f"jobs: {len(jobs)}  checks: {checks}")
    by = {}
    for j in jobs: by.setdefault((j["cell"], j["truth"]), set()).add(j["seed"])
    for k, v in sorted(by.items()): print(f"  {k}: seeds {sorted(v)}")
    chars = {}
    for j in jobs: chars[j["model"]] = chars.get(j["model"], 0) + sum(len(m["content"]) for m in j["messages"])
    print("approx input tokens (chars/4):", {k: round(v / 4) for k, v in chars.items()})
    print("example prompt:\n" + json.dumps(jobs[len(jobs) // 2]["messages"], indent=1)[:1800])
    ok = pt and checks["delivery_mismatch"] == 0 and checks["settings_mismatch"] == 0
    print("OFFLINE CHECKS", "PASS" if ok else "FAIL")
    if a.dry_run or not ok:
        return

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{a.plan}_readouts.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["key"] for l in open(path)}
    todo = [j for j in jobs if j["key"] not in done]
    print(f"running {len(todo)} readouts -> {path}")
    lock = threading.Lock()
    with open(path, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run_job, j) for j in todo]), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 50 == 0: print(f"  {i}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    main()
