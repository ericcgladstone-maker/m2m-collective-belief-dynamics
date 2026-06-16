"""Distributed probabilistic-inference task generator + Bayesian benchmark.

Binary latent state theta in {A, B}, flat prior. Each agent receives one private
signal carrying a known likelihood ratio (LR = P(obs|A)/P(obs|B)). The
full-information posterior is the prior updated by the SUM of per-agent
log-likelihood-ratios (LLRs) -- this is the normative target the collective is
scored against.

Regimes:
  concordant         - most signals weakly favor the true state; easy.
  misleading_majority- a numerical majority of weak signals favor the WRONG
                       state, but one/two strong signals favor the true state,
                       so EVIDENCE-pooling (sum LLR) is correct while
                       CONCLUSION/vote-pooling (majority of modal guesses) is
                       wrong. This is the crux regime for the evidence-vs-
                       conclusion contrast.
  ambiguous          - signals roughly cancel; correct answer is calibrated
                       uncertainty, not consensus.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Signal:
    p_a: float          # P(observation | state A)
    p_b: float          # P(observation | state B)
    label: str          # short tag, e.g. "reading-7"

    @property
    def llr(self) -> float:        # log-likelihood-ratio in favor of A
        return math.log(self.p_a / self.p_b)

    def describe(self) -> str:
        return (f'a private reading "{self.label}". Readings like this occur with '
                f'probability {self.p_a:.2f} when the true state is A, and probability '
                f'{self.p_b:.2f} when the true state is B')


@dataclass
class Trial:
    seed: int
    regime: str
    true_state: str
    signals: List[Signal]
    prior_a: float = 0.5

    @property
    def llrs(self) -> List[float]:
        return [s.llr for s in self.signals]

    @property
    def full_info_post_a(self) -> float:
        """Bayesian P(A) given ALL signals."""
        lo = math.log(self.prior_a / (1 - self.prior_a)) + sum(self.llrs)
        return 1.0 / (1.0 + math.exp(-lo))

    @property
    def naive_vote_a(self) -> float:
        """Fraction of agents whose OWN signal alone favors A (the 'conclusion
        majority'). Differs from full_info in misleading_majority."""
        votes = sum(1 for s in self.signals if s.llr > 0)
        return votes / len(self.signals)

    def private_post_a(self, i: int) -> float:
        lo = math.log(self.prior_a / (1 - self.prior_a)) + self.signals[i].llr
        return 1.0 / (1.0 + math.exp(-lo))


def _sig(rng, p_a, p_b, tag):
    return Signal(p_a=p_a, p_b=p_b, label=tag)


def generate_trial(seed: int, regime: str, n_agents: int = 5) -> Trial:
    rng = random.Random(seed)
    true_state = rng.choice(["A", "B"])
    # Build signals in "favor-A" orientation, then flip if true state is B so the
    # regime structure is symmetric w.r.t. the true state.
    sigs: List[Signal] = []

    if regime == "concordant":
        # Most signals weakly favor truth; net strongly correct.
        for k in range(n_agents):
            strong = rng.random() < 0.4
            pa = rng.uniform(0.62, 0.72) if not strong else rng.uniform(0.78, 0.88)
            sigs.append(_sig(rng, pa, 1 - pa, f"r{k}"))

    elif regime == "misleading_majority":
        # A numeric MAJORITY weakly favor the WRONG state; a strong MINORITY favor
        # truth. Vote-pooling -> wrong; evidence-pooling -> strongly right. Tuned so
        # the full-info posterior sits clearly off 0.5 (was near-boundary in pass 1).
        n_majority = max(2, n_agents // 2 + 1)        # n=5 -> 3 favor wrong
        for k in range(n_agents):
            if k < n_majority:
                pa = rng.uniform(0.38, 0.46)          # weak, favor B (wrong)
            else:
                pa = rng.uniform(0.82, 0.90)          # strong, favor A (truth)
            sigs.append(_sig(rng, pa, 1 - pa, f"r{k}"))

    elif regime in ("mm_gap_narrow", "mm_gap_wide", "mm_maj_heavy"):
        # Robustness variants of misleading_majority (reviewer concern #3): vary the
        # diagnosticity GAP and the wrong-majority SIZE while keeping the full-info
        # posterior on the truth (evidence-pooling correct, vote-pooling wrong).
        if regime == "mm_gap_narrow":          # weak gap: barely-wrong majority, moderate minority
            n_majority = max(2, n_agents // 2 + 1); maj = (0.42, 0.48); mino = (0.66, 0.74)
        elif regime == "mm_gap_wide":          # strong gap
            n_majority = max(2, n_agents // 2 + 1); maj = (0.30, 0.40); mino = (0.86, 0.94)
        else:                                  # mm_maj_heavy: only ONE strong-right minority node
            n_majority = max(2, n_agents - 1);     maj = (0.40, 0.46); mino = (0.88, 0.94)
        for k in range(n_agents):
            pa = rng.uniform(*maj) if k < n_majority else rng.uniform(*mino)
            sigs.append(_sig(rng, pa, 1 - pa, f"r{k}"))

    elif regime == "ambiguous":
        for k in range(n_agents):
            pa = rng.uniform(0.45, 0.55)
            sigs.append(_sig(rng, pa, 1 - pa, f"r{k}"))
    else:
        raise ValueError(f"unknown regime: {regime}")

    rng.shuffle(sigs)

    # Orient to true state: signals were built favor-A; if truth is B, mirror them.
    if true_state == "B":
        sigs = [Signal(p_a=s.p_b, p_b=s.p_a, label=s.label) for s in sigs]

    return Trial(seed=seed, regime=regime, true_state=true_state, signals=sigs)


if __name__ == "__main__":
    # Offline self-test: verify the regimes have the intended structure.
    print("Self-test: 200 trials per regime\n")
    for regime in ("concordant", "misleading_majority", "ambiguous"):
        n = 200
        full_correct = 0
        vote_correct = 0
        post_to_truth = 0.0
        for seed in range(n):
            t = generate_trial(seed, regime, n_agents=5)
            truth_is_a = (t.true_state == "A")
            p_truth = t.full_info_post_a if truth_is_a else 1 - t.full_info_post_a
            post_to_truth += p_truth
            full_argmax_a = t.full_info_post_a >= 0.5
            if full_argmax_a == truth_is_a:
                full_correct += 1
            vote_argmax_a = t.naive_vote_a >= 0.5
            if vote_argmax_a == truth_is_a:
                vote_correct += 1
        print(f"{regime:20s}  full-info acc={full_correct/n:.2f}  "
              f"naive-vote acc={vote_correct/n:.2f}  "
              f"mean P(truth)={post_to_truth/n:.2f}")
    print("\nExpect: misleading_majority -> full-info acc HIGH but naive-vote acc LOW.")
    print("This gap is the wisdom-of-crowds bonus the comms manipulation should destroy.")
