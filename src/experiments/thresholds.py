"""
Layer 6 - Decision Threshold Selection
=======================================

A threshold is a *decision*, separate from the score and from the calibrated
probability.  This module holds every way the framework is allowed to pick one,
and enforces the only rule that matters scientifically: the threshold is chosen
on validation and applied unchanged to test.

``select_threshold`` takes validation labels and validation scores.  It has no
parameter through which test data could reach it.

Strategies
----------
``max_f1``
    Maximise F1 over candidate thresholds drawn from the observed scores.
    Requires validation positives.  This is what the legacy trainer did, and it
    is a poor operational choice for a rare-positive problem - F1 weights a
    false positive and a false negative equally, which no AML team does - so it
    is retained for comparability, not recommended.
``max_fbeta``
    As above with an explicit ``beta``; ``beta > 1`` favours recall.
``alert_budget``
    The score quantile that alerts on a fixed fraction of entities.  Label-free,
    so it works even when validation has no positives, and it is the strategy
    that matches how an investigation team is actually resourced.
``min_precision``
    Lowest threshold whose validation precision is at least ``target``; i.e.
    maximise recall subject to a precision floor.  Returns "not achievable"
    rather than a silent fallback when no threshold reaches the target.
``fixed``
    An explicitly supplied constant.  There is no default of 0.5 anywhere in
    the framework; if you want 0.5 you must say so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.experiments.spec import ThresholdSpec

logger = logging.getLogger(__name__)

__all__ = [
    "ThresholdDecision",
    "select_threshold",
    "available_threshold_strategies",
]


@dataclass(frozen=True)
class ThresholdDecision:
    """The chosen threshold and the evidence behind it.

    ``value is None`` means no threshold could be selected.  Downstream, every
    threshold-dependent metric is then reported as ``None``; the run is not
    aborted, because the ranking metrics are still valid and informative.
    """

    value: Optional[float]
    strategy: str
    params: dict[str, Any]
    selected_on: str
    objective: Optional[str] = None
    objective_value: Optional[float] = None
    reason: Optional[str] = None
    n_val_positives: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "strategy": self.strategy,
            "params": dict(self.params),
            "selected_on": self.selected_on,
            "objective": self.objective,
            "objective_value": self.objective_value,
            "reason": self.reason,
            "n_val_positives": self.n_val_positives,
        }


def _candidate_thresholds(scores: np.ndarray, max_candidates: int = 512) -> np.ndarray:
    """Distinct observed scores (subsampled if numerous), descending.

    Searching the observed scores rather than a fixed grid on [0, 1] matters:
    a model whose scores all sit in [0.49, 0.51] gets a usable threshold, and a
    model producing unbounded margins is handled without rescaling.
    """
    uniq = np.unique(scores)
    if len(uniq) > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        uniq = np.unique(np.quantile(uniq, quantiles))
    return uniq[::-1]


def _fbeta_at(y: np.ndarray, s: np.ndarray, thr: float, beta: float) -> Optional[float]:
    preds = s >= thr
    tp = float(np.sum(preds & (y == 1)))
    fp = float(np.sum(preds & (y == 0)))
    fn = float(np.sum(~preds & (y == 1)))
    if tp == 0.0:
        return 0.0 if (fp or fn) else None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return float((1 + b2) * precision * recall / denom)


def _select_max_fbeta(
    y: np.ndarray, s: np.ndarray, *, beta: float = 1.0
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if int(y.sum()) == 0:
        return None, None, "validation partition has no positives; F-beta is undefined"
    best_thr: Optional[float] = None
    best_val = -1.0
    for thr in _candidate_thresholds(s):
        val = _fbeta_at(y, s, float(thr), beta)
        if val is None:
            continue
        # Strict improvement keeps the highest threshold among ties, which is
        # the lower-alert-volume choice.
        if val > best_val:
            best_val = val
            best_thr = float(thr)
    if best_thr is None:
        return None, None, "no threshold produced a defined F-beta"
    return best_thr, best_val, None


def _select_alert_budget(
    y: np.ndarray, s: np.ndarray, *, rate: float = 0.01
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if not 0.0 < rate <= 1.0:
        return None, None, f"alert rate {rate} outside (0, 1]"
    if len(s) == 0:
        return None, None, "validation partition is empty"
    k = max(1, int(np.ceil(rate * len(s))))
    thr = float(np.sort(s)[::-1][k - 1])
    achieved = float(np.mean(s >= thr))
    return thr, achieved, None


def _select_min_precision(
    y: np.ndarray, s: np.ndarray, *, target: float = 0.5
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if int(y.sum()) == 0:
        return None, None, "validation partition has no positives; precision is undefined"
    best_thr: Optional[float] = None
    best_recall = -1.0
    n_pos = float(y.sum())
    for thr in _candidate_thresholds(s):
        preds = s >= thr
        alerts = float(preds.sum())
        if alerts == 0:
            continue
        precision = float(np.sum(preds & (y == 1)) / alerts)
        if precision < target:
            continue
        recall = float(np.sum(preds & (y == 1)) / n_pos)
        if recall > best_recall:
            best_recall = recall
            best_thr = float(thr)
    if best_thr is None:
        return (
            None,
            None,
            f"no threshold reached precision >= {target} on validation",
        )
    return best_thr, best_recall, None


def _select_fixed(
    y: np.ndarray, s: np.ndarray, *, value: Optional[float] = None
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if value is None:
        return None, None, "strategy 'fixed' requires an explicit 'value' parameter"
    return float(value), None, None


_STRATEGIES: dict[str, tuple[Callable[..., tuple], str]] = {
    "max_f1": (lambda y, s, **kw: _select_max_fbeta(y, s, beta=1.0, **kw), "val_f1"),
    "max_fbeta": (_select_max_fbeta, "val_fbeta"),
    "alert_budget": (_select_alert_budget, "val_alert_rate"),
    "min_precision": (_select_min_precision, "val_recall_at_precision_floor"),
    "fixed": (_select_fixed, None),
}


def available_threshold_strategies() -> list[str]:
    return sorted(_STRATEGIES)


def select_threshold(
    spec: ThresholdSpec,
    y_val: Sequence[int],
    scores_val: Sequence[float],
) -> ThresholdDecision:
    """Choose a decision threshold using validation data only.

    Parameters
    ----------
    spec:
        Strategy name and its parameters.
    y_val, scores_val:
        Validation labels and validation scores.  Test data must never be
        passed here; there is deliberately no parameter for it.
    """
    if spec.strategy not in _STRATEGIES:
        raise KeyError(
            f"Unknown threshold strategy {spec.strategy!r}. "
            f"Available: {available_threshold_strategies()}"
        )
    y = np.asarray(y_val).reshape(-1).astype(np.int64)
    s = np.asarray(scores_val, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())

    if len(s) == 0:
        return ThresholdDecision(
            value=None,
            strategy=spec.strategy,
            params=dict(spec.params),
            selected_on="validation",
            reason="validation partition is empty",
            n_val_positives=0,
        )

    fn, objective = _STRATEGIES[spec.strategy]
    value, objective_value, reason = fn(y, s, **spec.params)
    if reason:
        logger.warning("Threshold selection (%s) failed: %s", spec.strategy, reason)
    return ThresholdDecision(
        value=value,
        strategy=spec.strategy,
        params=dict(spec.params),
        selected_on="validation",
        objective=objective,
        objective_value=objective_value,
        reason=reason,
        n_val_positives=n_pos,
    )
