"""Confirm Sonnet extended-reasoning works and METER its real token cost, so we can
size the within-family reasoning-effort run to the ~$20 Anthropic budget."""
from __future__ import annotations
import statistics as st
import config
config.AGENT_MODEL = "claude-sonnet-4-6"
config.PROVIDER = config.provider_for(config.AGENT_MODEL)
config.CLAUDE_THINKING = True
config.CLAUDE_THINKING_BUDGET = 2048

import agents
from tasks import generate_trial

t = generate_trial(0, "misleading_majority", 5)

def synth(trial, me):  # simulate an inbox of peers' CONCLUSIONS (the herding condition)
    items = [(j, f"My best guess is {'A' if trial.signals[j].llr > 0 else 'B'}, and I am 80% confident.")
             for j in range(len(trial.signals)) if j != me]
    return agents.format_received(items)

ins, outs = [], []
for i in range(4):
    b, u = agents.read_belief(t.signals[0].describe(), synth(t, 0), "claude-sonnet-4-6")
    print(f"call {i}: belief={b:.3f}  usage={u}")
    ins.append(u.get("prompt_tokens", 0)); outs.append(u.get("completion_tokens", 0))

mi, mo = st.mean(ins), st.mean(outs)
pin, pout = config._PRICES["claude-sonnet-4-6"]
per_call = mi * pin + mo * pout
per_comms_trial = 35 * per_call          # ~35 model calls per all_channel n=5 comms trial
print(f"\nmean in={mi:.0f} out={mo:.0f} tok/call | ${per_call:.4f}/call | ${per_comms_trial:.3f}/comms-trial")
for budget in (12, 16, 18):
    print(f"  ${budget} budget -> ~{int(budget/per_comms_trial)} comms trials affordable")
