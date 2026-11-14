# Evidence sharing improves collective accuracy in a language model experiment

Code for the distributed-inference experiments in Gladstone (in minor revision at *PNAS Nexus*), the third paper in the machine-to-machine (M2M) program. Stateless LLM agents each hold a private, noisy signal about a hidden binary state and exchange messages over a network. We manipulate **communication content** (sharing evidence vs. sharing conclusions) and **network topology**, and measure collective accuracy, mean individual error, and belief diversity.

## Which entry point to use

- **Revised manuscript and Supplementary Materials (minor revision, November 2026):** start from `revision_2026_11/README.md`. That folder holds the strict readout parser, the fixed-message readout replay, the regenerated-message harness, the analyses behind the revised tables, and the Figure 1 script. Its README gives the commands that recompute the reported results from the archived OSF records without API keys and compare them with the reference outputs.
- **Earlier analysis (September 2026 version of the manuscript):** `audit.py` and the experiment runners, analysis scripts, and figure scripts listed below are preserved unchanged for that version. `audit.py` checks the values reported in that version, so it is not the reproduction entry point for the revised manuscript, and several of the analyses it covers are not reported there.

The revision scripts import the shared modules in the repository root: the simulation rig, the prompt builder and legacy parser in `agents.py`, the task generator, the aggregation rules in `analyze_pools.py`, and the disclosure classifiers in `audit_revision.py` and `check_leakage.py`.

## Repository layout

- **Simulation rig:** `engine.py`, `tasks.py`, `agents.py`, `provider.py`, `metrics.py`, `config.py`
- **Revision code:** `revision_2026_11/` (see its README)
- **Experiment runners (earlier analysis):** `run_full.py` (content × topology grid), `run_smallworld.py` / `run_degree.py` (n = 11 degree ladder), `run_overload.py` (concordant high-fan-in), `run_signalsweep.py` (signal-structure robustness), `run_sonnet_think.py` (reasoning moderator), `run_replay.py` (planted-conclusion causal test), `run_adequacy.py` (capability gate), `run_hidden.py` (naturalistic hidden-profile probe)
- **Analysis and figures (earlier analysis):** `analyze*.py`, `make_figures.py`, `make_figs_v6.py`, `make_fig_design.py`, `make_fig_mechanism.py`, `make_fig_validation.py`
- **Major-revision analyses (earlier analysis):** `analyze_dispersion.py` (collective Brier decomposition), `analyze_pools.py` (six aggregation rules), `analyze_load.py` (neighbour fan-in and incoming message load), `analyze_benchmark.py` and `analyze_recipient.py` (confidence-weighted benchmark and normative-recipient counterfactual), `check_leakage.py` (two-directional channel-purity audit), `extract_worked_trial.py` (worked example trial), `adversarial_followup_analysis.py` (pooling sensitivity, threshold sensitivity, and interval estimates)
- **Audit of the earlier analysis:** `audit.py` is the single entry point for verifying the September 2026 version. It re-derives every reported quantity directly from the raw data files, then delegates to `audit_revision.py` for every quantity introduced or changed in the major revision, and writes `audit_results.json`. Reported values are used only as assertion targets, never as inputs. All resampling is deterministic under a fixed seed of 20260909.

## Setup

- Python 3.9+ and `pip install -r requirements.txt`.
- Provide API keys through a local `.env` file (never committed): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.
- Example: `python run_full.py <reps>`. Runs are resumable and write to `data/`.
- To recompute the results of the revised manuscript: see `revision_2026_11/README.md` (requires the OSF data under `data/` and the OSF folder `revision-2026-11`, and no API keys).
- To verify the September 2026 version against the archived data: `python audit.py` (requires the OSF data under `data/`).

## Data

Raw per-round transcripts and per-trial result tables are deposited separately on the Open Science Framework (https://doi.org/10.17605/OSF.IO/AU2GS) and are not included here due to size. The records of the minor revision are in the OSF folder `revision-2026-11`. The runners regenerate new samples given API access.

## Citation

Gladstone, E. Evidence sharing improves collective accuracy in a language model experiment. Manuscript in minor revision at *PNAS Nexus*.
