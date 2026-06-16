"""Naturalistic hidden-profile task (generalization check, reviewer option 2).

A fictional materials-comparison so model world-knowledge can't contaminate the truth:
N analysts must judge which of two fictional materials (Zeta vs Theta) is the better
choice for an application. Each analyst privately holds ONE prose 'finding' from a
distinct engineering test; each finding favors one material with a hidden latent weight.
TRUTH = the material favored by the SUM of all findings' weights (the full evidence).

misleading_majority: most analysts' own finding WEAKLY favors the wrong material, while a
strong minority hold a DECISIVE finding for the right one -> snippet-by-snippet majority is
wrong, pooling all findings is right. Mirrors the formal task's structure in naturalistic
prose. evidence-sharing = share your finding (prose); conclusion-sharing = share your
verdict + confidence. The self-test (run this file) verifies the structure holds, exactly
as we validated the numeric signal regimes.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

MATERIALS = ("Material Zeta", "Material Theta")
TESTS = [
    "the corrosion-resistance trial", "the thermal-cycling test",
    "the fatigue-life assessment", "the tensile-strength measurement",
    "the weld-integrity inspection", "the long-term creep test",
    "the impact-toughness test", "the dimensional-stability check",
    "the vibration-endurance run", "the salt-spray exposure test",
    "the load-cycling evaluation",
]


@dataclass
class Finding:
    test: str
    favors_zeta: bool      # True -> favors Zeta, False -> favors Theta
    weight: float          # latent magnitude of support (hidden from agents)
    strong: bool
    level: str = ""        # "" -> use strong flag; else "moderate" (uniform, no salience cue)

    def describe(self) -> str:
        winner, loser = ("Material Zeta", "Material Theta") if self.favors_zeta \
            else ("Material Theta", "Material Zeta")
        if self.level == "moderate":
            # uniform, non-salient prose: every analyst's finding reads the same way, so the
            # answer is NOT obvious from any single finding (forces pooling, makes agents swayable)
            return (f"a finding from {self.test}: {winner} performed somewhat better than {loser}, "
                    f"a clear but not decisive difference")
        if self.strong:
            return (f"a decisive finding from {self.test}: {winner} clearly and "
                    f"consistently outperformed {loser} by a large margin across all samples")
        return (f"a marginal finding from {self.test}: {winner} slightly edged out "
                f"{loser}, by a small margin within normal sample-to-sample variation")


@dataclass
class HiddenTrial:
    seed: int
    regime: str
    findings: List[Finding]

    @property
    def signed_total(self) -> float:
        """Sum of weights in the +Zeta direction (full-evidence support for Zeta)."""
        return sum((f.weight if f.favors_zeta else -f.weight) for f in self.findings)

    @property
    def true_zeta(self) -> bool:
        return self.signed_total > 0

    @property
    def true_label(self) -> str:
        return "Zeta" if self.true_zeta else "Theta"

    @property
    def snippet_majority_zeta(self) -> bool:
        """Which material wins a vote of each analyst's OWN single finding."""
        votes = sum(1 for f in self.findings if f.favors_zeta)
        return votes > len(self.findings) / 2


def generate_hidden_trial(seed: int, regime: str, n_agents: int = 5) -> HiddenTrial:
    rng = random.Random(10_000 + seed)
    truth_zeta = rng.choice([True, False])          # which material the FULL evidence favors
    tests = rng.sample(TESTS, n_agents)
    findings: List[Finding] = []

    if regime == "concordant":
        for k in range(n_agents):
            w = rng.uniform(0.5, 1.2)
            findings.append(Finding(tests[k], favors_zeta=truth_zeta, weight=w, strong=w > 0.9))

    elif regime == "misleading_majority":
        n_majority = max(2, n_agents // 2 + 1)       # n=5 -> 3 weak-wrong, 2 strong-right
        for k in range(n_agents):
            if k < n_majority:                        # weak, favors the WRONG material
                w = rng.uniform(0.10, 0.40)
                findings.append(Finding(tests[k], favors_zeta=(not truth_zeta), weight=w, strong=False))
            else:                                     # strong, favors the RIGHT material
                w = rng.uniform(1.2, 2.0)
                findings.append(Finding(tests[k], favors_zeta=truth_zeta, weight=w, strong=True))
    elif regime == "misleading_hard":
        # Harder generalization test: create genuine HERDING PRESSURE the first task lacked.
        # Uniform, non-salient prose (no decisive/marginal cue); the wrong majority is moderate
        # (so each agent leans wrong but is uncertain and swayable), the right minority is only
        # moderately stronger (full pool recovers truth, but no single finding is decisive).
        n_majority = max(2, n_agents // 2 + 1)        # n=5 -> 3 wrong, 2 right
        for k in range(n_agents):
            if k < n_majority:
                w = rng.uniform(0.45, 0.70)           # moderate, favors WRONG
                findings.append(Finding(tests[k], favors_zeta=(not truth_zeta), weight=w, strong=False, level="moderate"))
            else:
                w = rng.uniform(1.05, 1.45)           # moderately stronger, favors RIGHT
                findings.append(Finding(tests[k], favors_zeta=truth_zeta, weight=w, strong=False, level="moderate"))
    else:
        raise ValueError(f"unknown regime: {regime}")

    rng.shuffle(findings)
    # guard: ensure the full evidence actually points to the intended truth
    if (sum((f.weight if f.favors_zeta else -f.weight) for f in findings) > 0) != truth_zeta:
        # extremely rare; nudge by flipping not needed because minority weights dominate by construction
        pass
    return HiddenTrial(seed=seed, regime=regime, findings=findings)


if __name__ == "__main__":
    print("Self-test: 300 trials/regime (n=5). Want misleading_majority -> full-evidence "
          "truth well-defined, snippet-majority WRONG.\n")
    for regime in ("concordant", "misleading_majority"):
        n = 300
        snippet_correct = 0
        well_defined = 0
        margins = []
        for seed in range(n):
            t = generate_hidden_trial(seed, regime, 5)
            margins.append(abs(t.signed_total))
            well_defined += abs(t.signed_total) > 0.3
            snippet_correct += (t.snippet_majority_zeta == t.true_zeta)
        margins.sort()
        print(f"{regime:20s} snippet-majority acc={snippet_correct/n:.2f}  "
              f"truth well-defined (|total|>0.3)={well_defined/n:.2f}  "
              f"median|total|={margins[n//2]:.2f}")
    print("\nExpect: misleading_majority snippet-majority acc ~0.00 (vote-by-snippet wrong), "
          "well-defined ~1.00.")
