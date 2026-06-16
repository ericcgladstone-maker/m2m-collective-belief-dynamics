"""Agent node behavior: belief readout (via answer-token logprobs) and message
generation under the evidence-only / conclusion-only / both content regimes."""
from __future__ import annotations
from typing import List, Tuple

import provider
import config

SYSTEM = (
    "You are one analyst in a group working to identify a hidden state, which is "
    "either A or B (equally likely before any evidence). Each analyst has seen a "
    "DIFFERENT, independent private reading. Reason like a careful Bayesian: each "
    "reading shifts your belief in proportion to its diagnosticity (how much more "
    "likely that reading is under one state than the other). Independent readings "
    "COMBINE — several distinct readings pointing the same way should make you MORE "
    "confident, not less. The only thing to avoid is counting the SAME original "
    "reading more than once if it reaches you again after being relayed by someone."
)


def format_received(received: List[Tuple[int, str]]) -> str:
    if not received:
        return ""
    lines = [f"  - [from analyst {src}] {msg}" for src, msg in received]
    return "Messages you received this round:\n" + "\n".join(lines) + "\n"


import re

def _parse_final(txt: str) -> float:
    """Prefer an explicit 'FINAL=p' line; fall back to the last number in the text."""
    m = re.search(r"FINAL\s*=\s*([0-9]*\.?[0-9]+)", txt, re.IGNORECASE)
    nums = re.findall(r"[0-9]*\.[0-9]+|[0-9]+", txt)
    raw = m.group(1) if m else (nums[-1] if nums else None)
    if raw is None:
        return 0.5
    v = float(raw)
    if v > 1.0:
        v = v / 100.0
    return min(1.0, max(0.0, v))


def read_belief(signal_desc: str, received_block: str, model: str = None):
    """Return (P(state=A), usage). Brief-reasoning readout: the agent works through
    the Bayesian combination and emits a parsed FINAL probability. A bare-number
    readout works for gpt-4o-mini but Claude models silently refuse the arithmetic
    and hedge to 0.5 on multi-evidence prompts — letting them show working fixes
    that and is robust across model families."""
    user = (
        f"Your private evidence: you observed {signal_desc}.\n"
        f"{received_block}"
        "Treat your own reading and each DISTINCT reading reported by others as "
        "independent Bayesian evidence about the hidden state (flat prior; combine by "
        "multiplying likelihood ratios — several independent readings pointing the same "
        "way make you more confident; count a given original reading only once even if "
        "relayed). Work through it in 1-3 short sentences, then end with a line in "
        "exactly this format:\nFINAL=<your probability the hidden state is A, between 0 and 1>"
    )
    out = provider.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        model=model, max_tokens=320, temperature=0.0,
    )
    return _parse_final(out["text"]), out.get("usage", {})


def make_message(signal_desc: str, received_block: str, content_mode: str,
                 model: str = None) -> Tuple[str, dict]:
    if content_mode == "evidence_only":
        instr = (
            "Write a short message to the other analysts sharing ONLY the raw "
            "evidence you have: state your reading and its diagnosticity (the "
            "probabilities of such a reading under state A and under state B). "
            "Do NOT say which state you think is more likely. Do NOT give a guess, "
            "conclusion, or recommendation. One or two sentences."
        )
    elif content_mode == "conclusion_only":
        instr = (
            "Write a short message to the other analysts sharing ONLY your current "
            "best guess of the hidden state (A or B) and how confident you are. "
            "Do NOT describe your underlying reading or any probabilities. "
            "One sentence."
        )
    elif content_mode == "both":
        instr = (
            "Write a short message to the other analysts sharing your reading and "
            "its diagnosticity, AND your current best guess of the hidden state with "
            "your confidence. Two or three sentences."
        )
    else:
        raise ValueError(f"unknown content_mode: {content_mode}")

    user = (
        f"Your private evidence: you observed {signal_desc}.\n"
        f"{received_block}"
        f"{instr}"
    )
    out = provider.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        model=model, max_tokens=120, temperature=0.7,
    )
    return out["text"].strip(), out.get("usage", {})
