# Export notes

This folder is the public code export for the paper *Evidence sharing improves collective accuracy in a language model experiment* (Gladstone, in minor revision at *PNAS Nexus*), the third paper in the M2M program. It is curated from the project's working `code/` directory.

## What is included
- `revision_2026_11/`, the code for the minor revision and the reproduction entry point for the revised manuscript and Supplementary Materials: the strict readout parser, the fixed-message readout replay, the regenerated-message harness, the analyses behind the revised tables, and the Figure 1 script. Its README gives the reproduction commands. Its records are in the OSF folder `revision-2026-11`.
- The simulation rig (`engine.py`, `tasks.py`, `agents.py`, `provider.py`, `metrics.py`, `config.py`).
- All experiment runners (`run_*.py`), preserved for the earlier analysis: content × topology grid, n = 11 degree ladder, high-fan-in overload, signal-structure sweep, reasoning-effort moderator, planted-conclusion causal replay, capability gate, and the naturalistic hidden-profile probe.
- Analysis and figure generation (`analyze*.py`, `make_fig*.py`, `make_figures.py`).
- `audit.py`, the audit entry point for the September 2026 version, which re-derives every quantity reported in that version's manuscript and Supplementary Materials from the raw data files, and `audit_revision.py`, which covers the quantities introduced or changed in the major revision.
- `audit_results.json`, the structured record of the most recent audit run: one row per assertion with its identifier, claim family, source data file, reproduced value, reported value, tolerance, and status.
- `README.md`, `requirements.txt`, `RUN_GUIDE.md`.

## What is excluded, and why
- **Raw data** (`data/`, transcripts, per-trial CSVs): deposited separately on OSF (https://doi.org/10.17605/OSF.IO/AU2GS); excluded here for size.
- **API keys** (`.env`): never committed.
- **Run logs of the minor revision**: omitted. They record the offline checks, which the dry runs reproduce, and run progress, and they contain local paths. Every reported result is computed from the deposited records.
- **Results outputs** (`stats_report*.md`, `RESULTS.md`, `AUDIT.md`) and **figures**: regenerable from the scripts; omitted pending publication.
- **Internal development log** (`DECISIONS.md`) and the **manuscript / supplementary materials**: not part of the code release.

## Reproducibility
For the revised manuscript and Supplementary Materials, follow `revision_2026_11/README.md`. It recomputes every reported result from the OSF records without API keys.

For the September 2026 version, provide API keys in a local `.env`, install `requirements.txt`, and run the `run_*.py` scripts (resumable). The analysis and figure scripts regenerate all reported numbers and figures. `audit.py` checks that version's values against the raw data: it runs the structural and original-submission checks, then the major-revision assertions in `audit_revision.py`, and writes `audit_results.json`. The audit is deterministic under a fixed seed of 20260909 with 10,000 resampling draws, so repeated runs produce identical output.
