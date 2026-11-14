# Revision analyses (PNAS Nexus minor revision, November 2026)

This folder holds the code for the readout audit and the regenerated-message trials reported in the revised manuscript and Supplementary Materials. The September experiments recorded each agent's belief through a readout call capped at 320 tokens and parsed by the last-number fallback in `agents._parse_final`. The scripts here measure that readout again on the logged conversations at larger caps, regenerate the conversations of the focal cells with a 2,048-token readout and a strict parser, and recompute every revision result from the saved records.

Every reported number can be recomputed from the archived records without model API keys. Rerunning the two harnesses with keys makes new model calls and produces a new sample. Those calls are stochastic and the provider-side models can change, so a rerun is not expected to reproduce the archived records.

## Contents

| Script | What it does | Reported results |
|---|---|---|
| `paths.py` | Locations of code and data, each overridable by an environment variable | |
| `replay_readout.py` | Strict parser, record loader, and the fixed-message readout replay harness (holds each logged conversation fixed and asks only the readout again) | Strict parser (Methods, SI S3). Generated `pilot_readouts.jsonl` and `cap2048_readouts.jsonl` |
| `analyze_pilot.py`, `analyze_pilot2.py` | Analysis of the 320-token replay | 132 truncated readouts, 13 of 66 and 61 of 66 correct-side readouts, no round-0 truncation (main text, SI S3) |
| `analyze_cap2048.py` | Analysis of the 2,048-token fixed-message replay | Replay rows of Table S5 |
| `rerun_full.py` | Regenerated-message harness: same seeds, prompts, and message settings, readout cap 2,048, strict parser | Generated `rerun_trials.jsonl` |
| `analyze_rerun.py` | Completeness gate and analysis of the 500 regenerated trials | Table 1, regenerated rows of Table S5 |
| `close_audit.py` | Diagnostics on the September records, the replay, and the regenerated trials | Tables S2, S3, S4A, S4B, S7, S8, the 1,650-message audit, and the exact-tie statement in Methods |
| `pool_rules_rerun.py` | Focal gpt-4.1-mini contrast under the six aggregation rules of `analyze_pools.py` | Table S6 |
| `make_fig1_revised.py` | Figure 1 schematic (a drawing, not an analysis) | Fig. 1 |

The scripts use the deposited prompt builder, legacy parser, task generator, aggregation rules, and disclosure classifiers in the repository root, so they must stay inside this repository. They need Python 3.9 or later. The analysis scripts use only the standard library. `make_fig1_revised.py` needs matplotlib.

## Data

Two OSF deposits are needed, both at https://doi.org/10.17605/OSF.IO/AU2GS.

1. `m2m_raw_data.zip`, the September transcripts and trial tables. It unzips to `data/`.
2. The folder `revision-2026-11`, containing the revision records and the reference outputs:
   - `pilot_readouts.jsonl`: the 320-token replay, 672 readouts
   - `cap2048_readouts.jsonl`: the 2,048-token fixed-message replay, 20,800 readouts
   - `rerun_trials.jsonl`: the 500 regenerated-message trials, with every message and readout, raw text, stop reasons, usage, settings, and signals
   - `cap2048_analysis.txt`, `analysis.txt`, `close_audit_output.txt`, `pool_rules_rerun_output.txt`: the outputs of the analysis scripts on these records

The readout records store the first raw response of each call, the stop reason, usage, the returned model identifier, the legacy parse, and the strict parse. The September readout calls stored only the parsed belief, so their raw text and stop reasons cannot be recovered.

By default the scripts look for the September data in `data/` at the repository root and for the revision records in `data/revision-2026-11/`. Set `M2M_DATA` or `M2M_REVISION_DATA` to use other locations.

## Reproducing the reported results

From the repository root, with both deposits downloaded:

```
unzip m2m_raw_data.zip                  # creates data/
mkdir -p data/revision-2026-11          # place the OSF folder revision-2026-11 here
cd revision_2026_11
python3 analyze_pilot.py
python3 analyze_pilot2.py
python3 analyze_cap2048.py   > cap2048_analysis.txt
python3 analyze_rerun.py     > analysis.txt
python3 close_audit.py       > close_audit_output.txt
python3 pool_rules_rerun.py  > pool_rules_rerun_output.txt
for f in cap2048_analysis.txt analysis.txt close_audit_output.txt pool_rules_rerun_output.txt; do
  cmp "$f" ../data/revision-2026-11/"$f" && echo "$f identical"
done
python3 make_fig1_revised.py           # writes figures/Figure_1.{pdf,svg,tif,png}
```

The four printed outputs should be byte-identical to the reference outputs. `analyze_rerun.py` stops with a gate failure if any planned trial is missing or duplicated. The resampling seeds are fixed in the scripts.

The two harnesses also have offline checks that make no model calls and need no keys. `python3 replay_readout.py --plan cap2048 --dry-run` rebuilds every readout prompt from the logged conversations, checks that each delivered message equals its sender's previous-round message, and runs the parser self-test. `python3 rerun_full.py --dry-run` regenerates every task instance from its seed and checks the signals and true states against the logged trials.

## New model calls

Running `replay_readout.py --plan pilot`, `replay_readout.py --plan cap2048`, or `rerun_full.py` without `--dry-run` calls the OpenAI and Anthropic APIs. Keys are read from the environment or from an optional `.env` file at the repository root (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `M2M_ENV` for another location). New records are written to `revision_2026_11/out/`, or to `M2M_OUT`, which should be a local directory outside any file-sync client. Runs resume from an existing output file.
