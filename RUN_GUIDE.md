# Run guide — gathering data (any model, scalable)

All runs use the venv interpreter: `./.venv/bin/python` (the Anthropic SDK lives there).
Keys are in `.env` (OpenAI, Gemini, Anthropic). Substrate is chosen by `config.AGENT_MODEL`.

## 1. Vet a model BEFORE trusting it (multi-evidence adequacy gate)
The experiment needs agents that correctly *weight* evidence by likelihood ratio
(strong few must out-weigh weak many), not just judge a single signal. Single-signal
calibration is NOT sufficient. Vet with:

```
./.venv/bin/python run_adequacy.py     # tests claude-haiku/sonnet, gpt-4o-mini, gpt-4.1-mini
```
PASS = low MAE vs full-info posterior, sign_match ~1.0, truth_acc ~1.0, ESPECIALLY in
misleading_majority.

**Results (2026-06-10), truth_acc in misleading_majority:**
| model | concordant | misleading | verdict |
|---|---|---|---|
| claude-sonnet-4-6 | 1.00 | **1.00** | ✅ gold standard |
| gpt-4.1-mini | 1.00 | **0.92** | ✅ adequate, ~8× cheaper than Sonnet |
| gemini-2.5-flash (thinking off) | 0.92 | 0.75 | ⚠️ borderline — below bar; excluded from headline (try Gemini Pro for a clean 3rd family) |
| claude-haiku-4-5 | 0.92 | 0.58 | ❌ counts instead of weighting |
| gpt-4o-mini | 0.75 | 0.58 | ❌ |

(gemini-2.5-flash *with* thinking on scored ~0.50 — a token-starvation artifact: thinking ate the output budget, so the `FINAL=` line was truncated. gemini-2.5-pro *requires* thinking (400s on thinkingBudget=0) and starved at 2048 tokens → 0.67. **Gemini parked:** to get a clean Google family, retry gemini-2.5-pro (or gemini-3-pro) with maxOutputTokens ≈ 8192 so thinking + the FINAL line both fit. Not pursued now — two clean families (Sonnet, gpt-4.1-mini) suffice.)

Only run full studies on whitelisted models.

## 2. Run / scale a family (resumable)
Set `config.AGENT_MODEL` to a whitelisted model, then:
```
./.venv/bin/python run_full.py <reps> <workers>
```
- Writes to `data/<AGENT_MODEL>/` (isolated per model; no cross-contamination).
- **Resumable + additive:** re-run with a larger `<reps>` to add seeds — it skips
  already-completed `(regime,topology,content,seed)` cells and merges. So scale
  8 → 30 → 50 incrementally, paying only for new trials.
- Crux-first ordering + daily-cap (RPD) fail-fast are built in (matters for OpenAI).
- Outputs per family: `full/transcripts.jsonl` (complete record), `messages.csv`,
  `beliefs.csv`, `rounds.csv`, `run_meta.json`. Schema in `data/README.md`.

## 3. Cost anchors (per ~600-trial / 50-rep × 12-cell full run, scaffolded readout)
- gpt-4.1-mini ≈ $9   | claude-sonnet-4-6 ≈ $70+ (output pricing × reasoning tokens)
- Scale Sonnet by reducing reps; gpt-4.1-mini affords full power cheaply.

## 4. Switching substrate / adding a family
- Edit `config.AGENT_MODEL`. Provider routing + pricing + data dir derive from it.
- OpenAI and Anthropic are wired in `provider.py`. Gemini is keyed in `.env` but needs
  a provider branch added before use (and an adequacy pass).

## 5. Belief elicitation note
Readout = brief Bayesian working then `FINAL=<p>` (parsed). A bare-number readout
works for gpt-4o-mini/gpt-4.1-mini but Claude models hedge to 0.5 on multi-evidence
prompts unless allowed to show working. Keep the scaffolded readout for cross-model
comparability.
