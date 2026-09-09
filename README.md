# Communication content and local connectivity shape collective accuracy in networks of machine agents

Code for the distributed-inference experiments in Gladstone (manuscript in preparation), the third paper in the machine-to-machine (M2M) program. Stateless LLM agents each hold a private, noisy signal about a hidden binary state and exchange messages over a network. We manipulate **communication content** (sharing evidence vs. sharing conclusions) and **network topology**, and measure collective accuracy, mean individual error, and belief diversity.

## Repository layout

- **Simulation rig:** `engine.py`, `tasks.py`, `agents.py`, `provider.py`, `metrics.py`, `config.py`
- **Experiment runners:** `run_full.py` (content × topology grid), `run_smallworld.py` / `run_degree.py` (n = 11 degree ladder), `run_overload.py` (concordant high-fan-in), `run_signalsweep.py` (signal-structure robustness), `run_sonnet_think.py` (reasoning moderator), `run_replay.py` (planted-conclusion causal test), `run_adequacy.py` (capability gate), `run_hidden.py` (naturalistic hidden-profile probe)
- **Analysis and figures:** `analyze*.py`, `make_figures.py`, `make_figs_v6.py`, `make_fig_design.py`, `make_fig_mechanism.py`, `make_fig_validation.py`
- **Revision-era analyses:** `analyze_dispersion.py` (collective Brier decomposition), `analyze_pools.py` (six aggregation rules), `analyze_load.py` (neighbour fan-in and incoming message load), `analyze_benchmark.py` and `analyze_recipient.py` (confidence-weighted benchmark and normative-recipient counterfactual), `check_leakage.py` (two-directional channel-purity audit), `extract_worked_trial.py` (worked example trial), `adversarial_followup_analysis.py` (pooling sensitivity, threshold sensitivity, and interval estimates)
- **Audit:** `audit.py` is the single entry point for verification. It re-derives every reported quantity directly from the raw data files, then delegates to `audit_revision.py` for every quantity introduced or changed in the revision, and writes `audit_results.json`. Reported values are used only as assertion targets, never as inputs. All resampling is deterministic under a fixed seed of 20260909.

## Setup

- Python 3.9+ and `pip install -r requirements.txt`.
- Provide API keys through a local `.env` file (never committed): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.
- Example: `python run_full.py <reps>`. Runs are resumable and write to `data/`.
- To verify the reported numbers against the archived data: `python audit.py` (requires the OSF data under `data/`).

## Data

Raw per-round transcripts and per-trial result tables are deposited separately on the Open Science Framework (https://doi.org/10.17605/OSF.IO/AU2GS) and are not included here due to size. The runners regenerate them from scratch given API access.

## Citation

Gladstone, E. Communication content and local connectivity shape collective accuracy in networks of machine agents. Manuscript in preparation. (Full citation forthcoming.)
