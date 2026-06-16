"""Belief-dynamics metrics: the diversity-prediction decomposition and friends.

For a set of agent beliefs b_i = P_i(state A) and the true state:
  mean_individual_error  e_bar(t) = mean_i (b_i - 1[theta=A])^2     (Brier)
  diversity              D(t)     = var_i(b_i)
  collective_belief      mean_i b_i  (or a designated readout node)
  collective_error                 = (collective - 1[theta=A])^2
  regret                           = |collective - bayes_full_info|

The premature-consensus signature: D(t) collapses toward 0 while e_bar(t) stays
flat or rises (correlated movement to a wrong attractor) -> agreement without
justification.
"""
from __future__ import annotations
from typing import List


def _truth_indicator(true_state: str) -> float:
    return 1.0 if true_state == "A" else 0.0


def mean_individual_error(beliefs_a: List[float], true_state: str) -> float:
    t = _truth_indicator(true_state)
    return sum((b - t) ** 2 for b in beliefs_a) / len(beliefs_a)


def diversity(beliefs_a: List[float]) -> float:
    m = sum(beliefs_a) / len(beliefs_a)
    return sum((b - m) ** 2 for b in beliefs_a) / len(beliefs_a)


def collective_belief(beliefs_a: List[float]) -> float:
    return sum(beliefs_a) / len(beliefs_a)


def collective_error(beliefs_a: List[float], true_state: str) -> float:
    t = _truth_indicator(true_state)
    return (collective_belief(beliefs_a) - t) ** 2


def collective_accuracy(beliefs_a: List[float], true_state: str) -> int:
    c = collective_belief(beliefs_a)
    pred_a = c >= 0.5
    return int(pred_a == (true_state == "A"))


def regret(beliefs_a: List[float], bayes_full_info_a: float) -> float:
    return abs(collective_belief(beliefs_a) - bayes_full_info_a)


import math

def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def collective_logodds(beliefs_a: List[float]) -> float:
    """Log opinion pool: sigmoid(mean of logits). Sharper than the linear-prob
    mean and the more defensible aggregator when beliefs encode evidence strength."""
    return _sigmoid(sum(_logit(b) for b in beliefs_a) / len(beliefs_a))

def accuracy_of(coll: float, true_state: str) -> int:
    return int((coll >= 0.5) == (true_state == "A"))

def regret_of(coll: float, bayes_full_info_a: float) -> float:
    return abs(coll - bayes_full_info_a)

def p_truth_of(coll: float, true_state: str) -> float:
    return coll if true_state == "A" else 1 - coll


def decomposition_check(beliefs_a: List[float], true_state: str) -> bool:
    """Diversity prediction theorem (squared error):
       collective_error == mean_individual_error - diversity (within fp tol)."""
    lhs = collective_error(beliefs_a, true_state)
    rhs = mean_individual_error(beliefs_a, true_state) - diversity(beliefs_a)
    return abs(lhs - rhs) < 1e-9
