"""Two-directional channel-purity audit.

Direction 1 (original): do `evidence_only` messages leak a VERDICT?
  Evidence messages should state the reading and its diagnosticity (the probability of
  the reading under each state) and NOT state which state the agent thinks is more likely.
  Likelihood phrasing ("...if the true state is A and ...if the true state is B") is
  permitted by design; only stated preferences for a state are flagged.

Direction 2 (added for the PNAS Nexus R&R, Reviewer 1 concern 2): do `conclusion_only`
  messages leak EVIDENCE?
  Conclusion messages are designed to carry a verdict plus a confidence value and NOT the
  underlying evidence. The confidence percentage is therefore permitted by design; what is
  flagged is evidence-like content: the private reading label, likelihood/diagnosticity of
  a reading under each state, decimal likelihood values, or explicit strength-of-signal
  descriptions of the observation itself.

Scans `message_text` across the binary-inference datasets. System python3.
Returns per-direction (total, flagged) plus flagged examples.
"""
import csv, re
from pathlib import Path

HERE = Path(__file__).parent
DATASETS = [
    "gpt-4.1-mini", "gpt-4.1-mini_n11", "gpt-4.1-mini_degree", "gpt-4.1-mini_overload",
    "gpt-4.1-mini_sigsweep", "claude-sonnet-4-6", "claude-sonnet-4-6_think", "gemini-2.5-flash",
]

# --- Direction 1: verdict markers inside evidence messages -------------------------------
# A stated preference for a STATE (the posterior), not likelihood language.
VERDICT = re.compile(
    r"(i think|i believe|i suspect|i conclude|i'?d (guess|say|bet|lean)|my (best )?guess|"
    r"my (conclusion|verdict)|i lean|leaning toward|i'?m (fairly |quite |very )?confident|"
    r"i am confident|more likely to be|most likely (to be|state|answer|that the)|"
    r"favou?rs (state )?(a|b)\b|points to (state )?(a|b)\b|i favou?r|going with (state )?(a|b)\b|"
    r"best guess is|state is (most |more )?(probably|likely)|my read is)", re.I)

# --- Direction 2: evidence markers inside conclusion messages ----------------------------
# The verdict itself and its confidence value are permitted by design, so strip the
# confidence expression before scanning, then look for evidence-like content.
CONFIDENCE = re.compile(
    r"(with|at)\s+(about|approximately|around|roughly|~)?\s*\d{1,3}(\.\d+)?\s*%\s*"
    r"(confidence|certainty|confident|sure|probability)?", re.I)
CONF_BARE = re.compile(r"\b\d{1,3}(\.\d+)?\s*%", re.I)

EVIDENCE = re.compile(
    r"("
    r"\br\d\b|"                                   # reading label r0..r9
    r"my (private )?(reading|signal|observation)|"  # names the observation
    r"private reading|"
    r"occurs with probability|"                   # diagnosticity phrasing
    r"if the (true )?state (is|were)|"            # likelihood-under-state
    r"under state [ab]\b|given state [ab]\b|"
    r"likelihood|diagnostic|diagnosticity|"
    r"probability 0\.\d+|"                        # decimal likelihood values
    r"0\.\d+\s*(if|under|for) |"
    r"likelihood ratio|llr"
    r")", re.I)


def _scan(mode, pattern, strip_confidence=False):
    total, flagged = 0, []
    for tag in DATASETS:
        f = HERE / "data" / tag / "messages.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r.get("content_mode") != mode:
                continue
            total += 1
            text = r.get("message_text", "")
            probe = text
            if strip_confidence:
                probe = CONF_BARE.sub(" ", CONFIDENCE.sub(" ", probe))
            if pattern.search(probe):
                flagged.append((tag, text[:220]))
    return total, flagged


def count_leakage():
    """Direction 1 — verdict leakage in evidence_only messages (original contract)."""
    return _scan("evidence_only", VERDICT)


def count_evidence_leakage():
    """Direction 2 — evidence leakage in conclusion_only messages."""
    return _scan("conclusion_only", EVIDENCE, strip_confidence=True)


if __name__ == "__main__":
    print("=" * 78)
    print("DIRECTION 1 — verdict leakage in evidence_only messages")
    print("=" * 78)
    t, fl = count_leakage()
    print(f"  scanned: {t}")
    print(f"  flagged: {len(fl)} ({100*len(fl)/max(t,1):.2f}%)")
    for tag, s in fl[:10]:
        print(f"    [{tag}] {s}")

    print()
    print("=" * 78)
    print("DIRECTION 2 — evidence leakage in conclusion_only messages")
    print("=" * 78)
    t2, fl2 = count_evidence_leakage()
    print(f"  scanned: {t2}")
    print(f"  flagged: {len(fl2)} ({100*len(fl2)/max(t2,1):.2f}%)")
    for tag, s in fl2[:15]:
        print(f"    [{tag}] {s}")
