"""Rebuild the terminal-round result tables from saved data/<model>/rounds.csv,
and write a permanent RESULTS.md. Re-run any time: ./.venv/bin/python analyze.py
(Run summaries otherwise live only in ephemeral task logs; this persists them.)"""
from __future__ import annotations
import csv, collections
from pathlib import Path

DATA = Path(__file__).parent / "data"
MODELS = ["gpt-4.1-mini", "claude-sonnet-4-6", "claude-haiku-4-5", "openai"]
ORDER = {("misleading_majority", "baseline", "none"): 0,
         ("misleading_majority", "all_channel", "evidence_only"): 1,
         ("misleading_majority", "all_channel", "conclusion_only"): 2,
         ("misleading_majority", "all_channel", "both"): 3,
         ("misleading_majority", "wheel", "both"): 4,
         ("misleading_majority", "circle", "both"): 5}


def summarize(model):
    p = DATA / model / "rounds.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    cells = collections.OrderedDict()
    for r in rows:
        cells.setdefault((r["regime"], r["topology"], r["content_mode"]), {}).setdefault(
            r["seed"], {})[int(r["round"])] = r
    out = []
    for key, seeds in cells.items():
        accs, regrets = [], []
        te, td = collections.defaultdict(list), collections.defaultdict(list)
        for s, rnds in seeds.items():
            term = rnds[max(rnds)]
            accs.append(int(term["acc_logodds"])); regrets.append(float(term["regret_logodds"]))
            for rd, row in rnds.items():
                te[rd].append(float(row["e_bar"])); td[rd].append(float(row["D"]))
        n = len(seeds); nr = max(te) + 1
        et = [round(sum(te[i]) / len(te[i]), 3) for i in range(nr)]
        dt = [round(sum(td[i]) / len(td[i]), 3) for i in range(nr)]
        out.append((key, n, sum(accs) / n, sum(regrets) / n, et, dt))
    return out


def main():
    lines = ["# RESULTS — M2M Collective Belief Dynamics", "",
             "Auto-generated from `data/<model>/rounds.csv` by `analyze.py`. "
             "Accuracy/regret use the log-odds collective pool; ē = mean individual error (Brier), "
             "D = cross-agent belief diversity. Stock public LLMs via API, no tuning.", ""]
    for model in MODELS:
        s = summarize(model)
        if not s:
            continue
        s.sort(key=lambda x: ORDER.get(x[0], 99))
        lines += [f"## {model}", "",
                  "| regime | topology | content | n | acc | regret | ē trajectory | D trajectory |",
                  "|---|---|---|---|---|---|---|---|"]
        for (rg, tp, ct), n, acc, reg, et, dt in s:
            lines.append(f"| {rg} | {tp} | {ct} | {n} | {acc:.2f} | {reg:.3f} | {et} | {dt} |")
        lines.append("")
    lines += ["## Causal do-operator (first-conclusion replay; ACE on P(truth))",
              "Plant a confident first conclusion (correct vs wrong), evidence held fixed, "
              "misleading_majority / all_channel / conclusion_only. Re-run: `run_replay.py <n>`.", "",
              "| model | n | plant correct → P(truth) | plant wrong → P(truth) | control | ACE |",
              "|---|---|---|---|---|---|",
              "| claude-sonnet-4-6 | 15 | 0.99 | 0.38 | 0.70 | **+0.61** |",
              "| gpt-4.1-mini | 40 | 0.92 | 0.34 | 0.76 | **+0.57** |", "",
              "_(Replay per-trial outputs are regenerable from the seeds; ACE values logged here "
              "and in DECISIONS.md.)_"]
    out = DATA.parent / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
