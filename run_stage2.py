"""Stage 2 — pass 1: the decisive contrast, low N.

Regime: misleading_majority (evidence-pooling correct, vote-pooling wrong).
Cells: no-comms baseline; all-channel under evidence-only / conclusion-only / both;
plus wheel & circle (both) for a topology spread. Paired seeds across cells.

Hypothesis the run adjudicates:
  baseline               -> collective correct (pools the strong minority signal)
  all_channel/conclusion -> herds to the wrong majority; D collapses, e_bar high
  all_channel/evidence   -> stays correct (raw strong signal is visible to integrate)
If that pattern appears, premature consensus is conclusion-driven, not connectivity
per se -> proceed to the full design.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import engine, metrics
from tasks import generate_trial

CELLS = [
    ("baseline", "none"),
    ("all_channel", "evidence_only"),
    ("all_channel", "conclusion_only"),
    ("all_channel", "both"),
    ("wheel", "both"),
    ("circle", "both"),
]
REGIME = "misleading_majority"
COST_IN, COST_OUT = 0.15e-6, 0.60e-6


def p_truth(coll_a, true_state):
    return coll_a if true_state == "A" else 1 - coll_a


def main(reps=12, r_comm=2):
    rows = []
    summary = []
    tot_in = tot_out = 0
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    for topology, content in CELLS:
        term_acc, term_regret, term_ptruth = [], [], []
        traj_e = {}; traj_D = {}
        for seed in range(reps):
            trial = generate_trial(seed, REGIME, 5)
            res = engine.run_trial(trial, topology, content, r_comm=r_comm)
            tot_in += res["usage"]["prompt_tokens"]; tot_out += res["usage"]["completion_tokens"]
            bbr = res["beliefs_by_round"]
            for rnd, beliefs in enumerate(bbr):
                e = metrics.mean_individual_error(beliefs, trial.true_state)
                D = metrics.diversity(beliefs)
                coll = metrics.collective_belief(beliefs)
                rows.append({
                    "topology": topology, "content_mode": content, "seed": seed,
                    "regime": REGIME, "true_state": trial.true_state,
                    "full_info_post_a": round(trial.full_info_post_a, 4),
                    "round": rnd, "e_bar": round(e, 4), "D": round(D, 4),
                    "collective_a": round(coll, 4),
                    "p_truth": round(p_truth(coll, trial.true_state), 4),
                    "collective_correct": metrics.collective_accuracy(beliefs, trial.true_state),
                    "regret": round(metrics.regret(beliefs, trial.full_info_post_a), 4),
                })
                traj_e.setdefault(rnd, []).append(e)
                traj_D.setdefault(rnd, []).append(D)
            last = bbr[-1]
            term_acc.append(metrics.collective_accuracy(last, trial.true_state))
            term_regret.append(metrics.regret(last, trial.full_info_post_a))
            term_ptruth.append(p_truth(metrics.collective_belief(last), trial.true_state))
        nr = len(traj_e)
        e_traj = [round(sum(traj_e[r]) / len(traj_e[r]), 3) for r in range(nr)]
        D_traj = [round(sum(traj_D[r]) / len(traj_D[r]), 3) for r in range(nr)]
        summary.append((topology, content, sum(term_acc) / len(term_acc),
                        sum(term_ptruth) / len(term_ptruth),
                        sum(term_regret) / len(term_regret), e_traj, D_traj))

    csv_path = out_dir / "pass1_rounds.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"=== STAGE 2 / PASS 1  (regime={REGIME}, reps={reps}, r_comm={r_comm}) ===\n")
    print(f"{'topology':14s} {'content':16s} {'term_acc':>8s} {'P(truth)':>9s} "
          f"{'regret':>7s}  e_bar_traj            D_traj")
    for topo, cont, acc, pt, reg, et, dt in summary:
        print(f"{topo:14s} {cont:16s} {acc:8.2f} {pt:9.2f} {reg:7.3f}  "
              f"{str(et):20s}  {dt}")
    cost = tot_in * COST_IN + tot_out * COST_OUT
    print(f"\nWrote {csv_path}")
    print(f"Pass-1 spend: ${cost:.4f}  (in={tot_in} out={tot_out} tok)")
    print("\nRead: baseline & all_channel/evidence_only should hold P(truth)>0.5 and "
          "term_acc high; all_channel/conclusion_only should fall (herding) with D "
          "collapsing across rounds while e_bar stays high.")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(reps=reps)
