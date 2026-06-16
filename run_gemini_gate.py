"""Re-attempt Gemini as a third model family (reviewer concern #7). The earlier park
was token starvation; with the thinking budget raised to 8192, re-run the multi-evidence
adequacy gate on Gemini 2.5 flash and pro. PASS (low MAE, sign~1.0, truth_acc~1.0,
esp. misleading_majority) -> eligible as a third family for the decisive cells."""
from __future__ import annotations
import config, run_adequacy

config.GEMINI_THINKING = True   # Gemini 2.5 needs thinking to integrate; budget now 8192 (provider.py)
run_adequacy.N = 12
for m in ["gemini-2.5-flash", "gemini-2.5-pro"]:
    try:
        run_adequacy.test_model(m)
    except Exception as e:
        print(f"\n{m}: ERROR {e}")
