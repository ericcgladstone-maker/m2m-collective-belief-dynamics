# Communication content and topology shape collective belief in networks of machine agents

Code for the distributed-inference experiments in Gladstone & Brashears (manuscript in preparation), the third paper in the machine-to-machine (M2M) program. Stateless LLM agents each hold a private, noisy signal about a hidden binary state and exchange messages over a network. We manipulate **communication content** (sharing evidence vs. sharing conclusions) and **network topology**, and measure collective accuracy, mean individual error, and belief diversity.

## Repository layout

- **Simulation rig:** `engine.py`, `tasks.py`, `agents.py`, `provider.py`, `metrics.py`, `config.py`
- **Experiment runners:** `run_full.py` (content × topology grid), `run_smallworld.py` / `run_degree.py` (n = 11 degree ladder), `run_overload.py` (concordant high-fan-in), `run_signalsweep.py` (signal-structure robustness), `run_sonnet_think.py` (reasoning moderator), `run_replay.py` (planted-conclusion causal test), `run_adequacy.py` (capability gate), `run_hidden.py` (naturalistic hidden-profile probe)
- **Analysis and figures:** `analyze*.py`, `make_figures.py`, `make_fig_design.py`, `make_fig_mechanism.py`, `make_fig_validation.py`
- **Audit:** `audit.py` re-derives every reported quantity directly from the raw data files.

## Setup

- Python 3.9+ and `pip install -r requirements.txt`.
- Provide API keys through a local `.env` file (never committed): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.
- Example: `python run_full.py <reps>`. Runs are resumable and write to `data/`.

## Data

Raw per-round transcripts and per-trial result tables are deposited separately (OSF; link forthcoming) and are not included here due to size. The runners regenerate them from scratch given API access.

## Citation

Gladstone, E., & Brashears, M. E. Communication content and topology shape collective belief in networks of machine agents. Manuscript in preparation. (Full citation forthcoming.)
