# Export notes

This folder is the public code export for the paper *Communication content and local connectivity shape collective accuracy in networks of machine agents* (Gladstone, in preparation), the third paper in the M2M program. It is curated from the project's working `code/` directory.

## What is included
- The simulation rig (`engine.py`, `tasks.py`, `agents.py`, `provider.py`, `metrics.py`, `config.py`).
- All experiment runners (`run_*.py`): content × topology grid, n = 11 degree ladder, high-fan-in overload, signal-structure sweep, reasoning-effort moderator, planted-conclusion causal replay, capability gate, and the naturalistic hidden-profile probe.
- Analysis and figure generation (`analyze*.py`, `make_fig*.py`, `make_figures.py`).
- `audit.py`, the single audit entry point, which re-derives every quantity reported in the manuscript and Supplementary Materials from the raw data files, and `audit_revision.py`, which covers the quantities introduced or changed in the major revision.
- `audit_results.json`, the structured record of the most recent audit run: one row per assertion with its identifier, claim family, source data file, reproduced value, reported value, tolerance, and status.
- `README.md`, `requirements.txt`, `RUN_GUIDE.md`.

## What is excluded, and why
- **Raw data** (`data/`, transcripts, per-trial CSVs): deposited separately on OSF (https://doi.org/10.17605/OSF.IO/AU2GS); excluded here for size.
- **API keys** (`.env`): never committed.
- **Results outputs** (`stats_report*.md`, `RESULTS.md`, `AUDIT.md`) and **figures**: regenerable from the scripts; omitted pending publication.
- **Internal development log** (`DECISIONS.md`) and the **manuscript / supplementary materials**: not part of the code release.

## Reproducibility
Provide API keys in a local `.env`, install `requirements.txt`, and run the `run_*.py` scripts (resumable). The analysis and figure scripts regenerate all reported numbers and figures. `audit.py` is the single entry point that checks them against the raw data: it runs the structural and original-submission checks, then the revision-era assertions in `audit_revision.py`, and writes `audit_results.json`. The audit is deterministic under a fixed seed of 20260909 with 10,000 resampling draws, so repeated runs produce identical output.
