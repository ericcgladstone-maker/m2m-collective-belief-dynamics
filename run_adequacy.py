"""Multi-evidence adequacy gate (the one missing from Stage 1).

Single-signal calibration is not enough: the task requires agents to INTEGRATE
several readings, weighting by likelihood ratio (strong few should out-weigh weak
many). This gate gives one agent all 5 readings (its own + 4 synthesized as
received) and checks whether its integrated belief tracks the normative
full-information posterior. A model that can't do this confounds 'communication
hurts' with 'the agent can't add up evidence'. Pick the cheapest model that PASSES
(low MAE vs full-info, high sign-match, high truth-accuracy) for the full run.
"""
from __future__ import annotations
import statistics as st
import config, provider, agents
from tasks import generate_trial

CANDIDATES = ["claude-haiku-4-5", "claude-sonnet-4-6", "gpt-4o-mini", "gpt-4.1-mini"]
N = 12
GATE = []  # (model, regime, mae, sign, truth, n) — collected so the gate is auditable


def synth_received(trial, me):
    items = [(j, f"I observed {trial.signals[j].describe()}.")
             for j in range(len(trial.signals)) if j != me]
    return agents.format_received(items)


def test_model(model):
    config.AGENT_MODEL = model
    config.PROVIDER = config.provider_for(model)
    rows = []
    for regime in ["concordant", "misleading_majority"]:
        for seed in range(N):
            t = generate_trial(seed, regime, 5)
            b, _ = agents.read_belief(t.signals[0].describe(), synth_received(t, 0), model)
            rows.append((regime, b, t.full_info_post_a, t.true_state))
    print(f"\n{model}:")
    for regime in ["concordant", "misleading_majority"]:
        rs = [r for r in rows if r[0] == regime]
        mae = st.mean(abs(b - tgt) for _, b, tgt, _ in rs)
        sign = st.mean(int((b >= 0.5) == (tgt >= 0.5)) for _, b, tgt, _ in rs)
        truth = st.mean(int((b >= 0.5) == (ts == "A")) for _, b, tgt, ts in rs)
        print(f"  {regime:20s} MAE_vs_fullinfo={mae:.2f}  sign_match={sign:.2f}  truth_acc={truth:.2f}")
        GATE.append((model, regime, round(mae, 3), round(sign, 3), round(truth, 3), len(rs)))


if __name__ == "__main__":
    print(f"=== Multi-evidence integration adequacy (n={N}/regime, agent sees all 5 readings) ===")
    print("PASS = low MAE, sign_match~1.0, truth_acc~1.0 (esp. misleading_majority)")
    for m in CANDIDATES:
        try:
            test_model(m)
        except Exception as e:
            print(f"\n{m}: ERROR {e}")
    import csv as _csv, json as _json
    from pathlib import Path as _P
    out = _P(__file__).parent / "data" / "adequacy"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "adequacy.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["model", "regime", "mae_vs_fullinfo", "sign_match", "truth_acc", "n"])
        for row in GATE:
            w.writerow(row)
    (out / "run_meta.json").write_text(_json.dumps({"n_per_regime": N, "rows": len(GATE),
        "note": "multi-evidence integration gate; PASS = low MAE + sign~1 + truth~1, esp. misleading_majority"}, indent=2))
    print(f"\nsaved -> {out}/adequacy.csv")
