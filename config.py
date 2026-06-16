"""Config + .env loader (stdlib only)."""
from __future__ import annotations
import os
from pathlib import Path

def load_env(path: str = ".env") -> None:
    p = Path(__file__).parent / path
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_THINKING = False   # if True, let gemini-2.5 think (helps integration, costs tokens)

# Substrate switch. Validated integrators (multi-evidence adequacy gate):
#   claude-sonnet-4-6 (gold), gpt-4.1-mini (adequate, cheap).
#   claude-haiku-4-5 / gpt-4o-mini FAIL (count instead of weight) -> do not use.
# Everything downstream (provider routing, pricing, data dir) derives from this.
AGENT_MODEL = "gpt-4.1-mini"

def provider_for(model: str) -> str:
    if "claude" in model:
        return "anthropic"
    if "gemini" in model:
        return "gemini"
    return "openai"

PROVIDER = provider_for(AGENT_MODEL)
DATA_TAG = AGENT_MODEL   # per-model data dir: substrates never mix; resume scoped per model

# (input $/token, output $/token)
_PRICES = {
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4.1-mini": (0.40e-6, 1.60e-6),
    "claude-haiku-4-5": (1.00e-6, 5.00e-6),
    "claude-sonnet-4-6": (3.00e-6, 15.00e-6),
    "gemini-2.0-flash-001": (0.10e-6, 0.40e-6),
    "gemini-2.5-flash": (0.30e-6, 2.50e-6),
}
COST_IN, COST_OUT = _PRICES.get(AGENT_MODEL, (1.0e-6, 5.0e-6))

# Answer labels for the (currently binary) latent state.
STATE_LABELS = ("A", "B")
