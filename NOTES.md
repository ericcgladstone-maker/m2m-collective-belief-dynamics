# Export notes

This folder is the public code export for the paper *Communication content and topology shape collective belief in networks of machine agents* (Gladstone & Brashears, in preparation), the third paper in the M2M program. It is curated from the project's working `code/` directory.

## What is included
- The simulation rig (`engine.py`, `tasks.py`, `agents.py`, `provider.py`, `metrics.py`, `config.py`).
- All experiment runners (`run_*.py`): content × topology grid, n = 11 degree ladder, high-fan-in overload, signal-structure sweep, reasoning-effort moderator, planted-conclusion causal replay, capability gate, and the naturalistic hidden-profile probe.
- Analysis and figure generation (`analyze*.py`, `make_fig*.py`, `make_figures.py`).
- `audit.py`, which re-derives every quantity reported in the manuscript from the raw data files.
- `README.md`, `requirements.txt`, `RUN_GUIDE.md`.

## What is excluded, and why
- **Raw data** (`data/`, transcripts, per-trial CSVs): deposited separately on OSF (link forthcoming); excluded here for size.
- **API keys** (`.env`): never committed.
- **Results outputs** (`stats_report*.md`, `RESULTS.md`, `AUDIT.md`) and **figures**: regenerable from the scripts; omitted pending publication.
- **Internal development log** (`DECISIONS.md`) and the **manuscript / supplementary materials**: not part of the code release.

## Reproducibility
Provide API keys in a local `.env`, install `requirements.txt`, and run the `run_*.py` scripts (resumable). The analysis and figure scripts regenerate all reported numbers and figures; `audit.py` checks them against the raw data.
