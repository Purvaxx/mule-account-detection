"""
Layer 6 - Evaluation Metrics
=============================

Metrics for a rare-positive detection problem, with one rule enforced
throughout: **an undefined metric is reported as ``None``, never as 0.0.**

The legacy trainer returns ``0.0`` for PR-AUC and ROC-AUC when an evaluation
partition contains a single class.  Averaged across folds that silently drags
the mean toward zero and is indistinguishable from a model that ranked every
positive last.  Here the value is ``None`` and the reason is recorded in
``undefined``.

Metric selection
----------------
Primary (ranking quality under class imbalance):
    ``average_precision``  - PR-AUC; the area a precision/recall curve encloses.
    ``ap_lift``            - AP divided by prevalence.  1.0 means the model is
                             no better than random at any operating point; this
                             is the number that makes a "0.02 PR-AUC" result
                             interpretable.
    ``precision_at_k`` / ``recall_at_k`` and the alert-rate equivalents - what
                             an investigator actually experiences: of the top K
                             alerts, how many are real, and what share of the
                             real cases did we surface.
    ``alerts`` / ``alert_rate`` / ``false_positive_burden`` - the operational
                             cost side.  ``false_positive_burden`` is false
                             positives per true positive at the chosen
                             threshold.

Secondary (reported, not optimised):
    ROC-AUC, and threshold-dependent precision / recall / F1 / confusion matrix.

Calibration (only meaningful when the scores claim to be probabilities):
    Brier score, expected calibration error, and the reliability bins the ECE
    was computed from.

Ties are handled explicitly in the ``@k`` family: entities are ranked by score
descending with the entity's positional index as a deterministic tiebreak, so
repeated runs give the same answer instead of depending on sort stability.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

__all__ = [
    "binary_metrics",
    "ranking_metrics",
    "threshold_metrics",
    "calibration_metrics",
    "aggregate_fold_metrics",
]


def _as_arrays(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).reshape(-1).astype(np.int64)
    ys = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if yt.shape != ys.shape:
        raise ValueError(f"y_true {yt.shape} and y_score {ys.shape} must have equal length.")
    return yt, ys


def _rank_order(y_score: np.ndarray) -> np.ndarray:
    """Indices sorted by score descending, ties broken by position ascending."""
    return np.lexsort((np.arange(len(y_score)), -y_score))


def ranking_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    k_values: Sequence[int] = (),
    alert_rates: Sequence[float] = (),
) -> dict[str, Any]:
    """Threshold-free ranking quality plus operating points.

    Returns a dict whose numeric entries are ``float`` or ``None``.  Every
    ``None`` has a matching entry in the returned ``undefined`` mapping giving
    the reason.
    """
    yt, ys = _as_arrays(y_true, y_score)
    n = int(len(yt))
    n_pos = int(yt.sum())
    n_neg = n - n_pos
    undefined: dict[str, str] = {}

    out: dict[str, Any] = {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(n_pos / n) if n else None,
    }
    if not n:
        undefined["prevalence"] = "empty partition"

    if n == 0:
        out.update({"average_precision": None, "ap_lift": None, "roc_auc": None})
        undefined["average_precision"] = "empty partition"
        undefined["roc_auc"] = "empty partition"
        out["precision_at_k"] = {}
        out["recall_at_k"] = {}
        out["precision_at_rate"] = {}
        out["recall_at_rate"] = {}
        out["undefined"] = undefined
        return out

    if n_pos == 0 or n_neg == 0:
        reason = (
            "no positives in partition" if n_pos == 0 else "no negatives in partition"
        )
        out["average_precision"] = None
        out["ap_lift"] = None
        out["roc_auc"] = None
        undefined["average_precision"] = reason
        undefined["roc_auc"] = reason
        undefined["ap_lift"] = reason
    else:
        ap = float(average_precision_score(yt, ys))
        prevalence = n_pos / n
        out["average_precision"] = ap
        out["ap_lift"] = float(ap / prevalence)
        out["roc_auc"] = float(roc_auc_score(yt, ys))

    order = _rank_order(ys)
    sorted_labels = yt[order]
    cum_tp = np.cumsum(sorted_labels)

    def _at(k: int) -> tuple[Optional[float], Optional[float]]:
        if k <= 0 or k > n:
            return None, None
        tp = int(cum_tp[k - 1])
        precision = float(tp / k)
        recall = float(tp / n_pos) if n_pos else None
        return precision, recall

    prec_k: dict[str, Optional[float]] = {}
    rec_k: dict[str, Optional[float]] = {}
    for k in k_values:
        p, r = _at(int(k))
        prec_k[str(k)] = p
        rec_k[str(k)] = r
        if p is None:
            undefined[f"precision_at_{k}"] = f"k={k} outside 1..{n}"
        if r is None and p is not None:
            undefined[f"recall_at_{k}"] = "no positives in partition"

    prec_rate: dict[str, Optional[float]] = {}
    rec_rate: dict[str, Optional[float]] = {}
    for rate in alert_rates:
        k = int(np.ceil(float(rate) * n))
        p, r = _at(k)
        key = f"{float(rate):g}"
        prec_rate[key] = p
        rec_rate[key] = r
        if p is None:
            undefined[f"precision_at_rate_{key}"] = (
                f"alert rate {rate} yields k={k}, outside 1..{n}"
            )

    out["precision_at_k"] = prec_k
    out["recall_at_k"] = rec_k
    out["precision_at_rate"] = prec_rate
    out["recall_at_rate"] = rec_rate
    out["undefined"] = undefined
    return out


def threshold_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: Optional[float],
) -> dict[str, Any]:
    """Confusion-matrix metrics at a fixed decision threshold.

    ``threshold is None`` means no threshold could be selected (see
    :mod:`src.experiments.thresholds`); every threshold-dependent value is then
    ``None`` rather than silently defaulting to 0.5.
    """
    yt, ys = _as_arrays(y_true, y_score)
    undefined: dict[str, str] = {}
    if threshold is None:
        return {
            "threshold": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion": None,
            "alerts": None,
            "alert_rate": None,
            "false_positive_burden": None,
            "undefined": {"all": "no threshold was selectable on validation"},
        }
    if len(yt) == 0:
        return {
            "threshold": float(threshold),
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion": None,
            "alerts": 0,
            "alert_rate": None,
            "false_positive_burden": None,
            "undefined": {"all": "empty partition"},
        }

    preds = (ys >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(yt, preds, labels=[0, 1]).ravel()
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    alerts = tp + fp

    precision: Optional[float]
    if alerts == 0:
        precision = None
        undefined["precision"] = "no entity scored above the threshold"
    else:
        precision = float(tp / alerts)

    n_pos = tp + fn
    if n_pos == 0:
        recall: Optional[float] = None
        undefined["recall"] = "no positives in partition"
    else:
        recall = float(tp / n_pos)

    if precision is None or recall is None or (precision + recall) == 0.0:
        f1: Optional[float] = None
        undefined.setdefault(
            "f1", "precision or recall undefined (or both zero) at this threshold"
        )
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    burden: Optional[float]
    if tp == 0:
        burden = None
        undefined["false_positive_burden"] = (
            "no true positives at this threshold; false positives per true "
            "positive is unbounded"
        )
    else:
        burden = float(fp / tp)

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": [[tn, fp], [fn, tp]],
        "alerts": alerts,
        "alert_rate": float(alerts / len(yt)),
        "false_positive_burden": burden,
        "undefined": undefined,
    }


def calibration_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Brier score, expected calibration error, and the reliability bins.

    Only call this with values that are meant to be probabilities.  If any
    value falls outside [0, 1] the result is ``None`` with a reason: a raw
    margin or logit has no calibration to measure.
    """
    yt, yp = _as_arrays(y_true, y_prob)
    if len(yt) == 0:
        return {
            "brier": None,
            "expected_calibration_error": None,
            "bins": [],
            "undefined": {"all": "empty partition"},
        }
    if yp.min() < 0.0 or yp.max() > 1.0:
        return {
            "brier": None,
            "expected_calibration_error": None,
            "bins": [],
            "undefined": {
                "all": "scores are not in [0, 1]; they are not probabilities"
            },
        }

    brier = float(brier_score_loss(yt, yp)) if len(np.unique(yt)) >= 1 else None

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    total = len(yt)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                {
                    "lower": float(lo),
                    "upper": float(hi),
                    "count": 0,
                    "mean_predicted": None,
                    "observed_rate": None,
                }
            )
            continue
        mean_pred = float(yp[in_bin].mean())
        observed = float(yt[in_bin].mean())
        ece += (count / total) * abs(mean_pred - observed)
        bins.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
            }
        )

    return {
        "brier": brier,
        "expected_calibration_error": float(ece),
        "bins": bins,
        "undefined": {},
    }


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: Optional[float] = None,
    k_values: Sequence[int] = (),
    alert_rates: Sequence[float] = (),
    probability_scale: bool = False,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Full metric block for one partition.

    Parameters
    ----------
    probability_scale:
        Set only when ``y_score`` is claimed to be a probability.  Controls
        whether calibration metrics are computed at all.
    """
    result: dict[str, Any] = {}
    result.update(
        ranking_metrics(y_true, y_score, k_values=k_values, alert_rates=alert_rates)
    )
    thr = threshold_metrics(y_true, y_score, threshold)
    undefined = dict(result.pop("undefined", {}))
    undefined.update({f"threshold.{k}": v for k, v in thr.pop("undefined", {}).items()})
    result["at_threshold"] = thr

    if probability_scale:
        cal = calibration_metrics(y_true, y_score, n_bins=calibration_bins)
        undefined.update({f"calibration.{k}": v for k, v in cal.pop("undefined", {}).items()})
        result["calibration"] = cal
    else:
        result["calibration"] = None
        undefined["calibration"] = "scores are not on a probability scale"

    result["undefined"] = undefined
    return result


def _leaf_numeric_paths(block: Any, prefix: str = "") -> dict[str, Optional[float]]:
    """Flatten a metric block to ``path -> value`` for numeric leaves."""
    out: dict[str, Optional[float]] = {}
    if isinstance(block, dict):
        for key, value in block.items():
            if key in ("undefined", "bins", "confusion"):
                continue
            out.update(_leaf_numeric_paths(value, f"{prefix}{key}."))
    elif block is None or isinstance(block, (int, float)) and not isinstance(block, bool):
        out[prefix.rstrip(".")] = None if block is None else float(block)
    return out


def aggregate_fold_metrics(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std across folds, counting how many folds actually defined each metric.

    A metric that was undefined in two of four folds is reported as the mean of
    the two that were defined, alongside ``n_defined = 2``.  Nothing is imputed,
    and ``n_defined = 0`` yields ``None`` rather than ``nan``.
    """
    flattened = [_leaf_numeric_paths(fm) for fm in fold_metrics]
    keys: list[str] = []
    for f in flattened:
        for k in f:
            if k not in keys:
                keys.append(k)

    aggregate: dict[str, Any] = {}
    for key in keys:
        values = [f[key] for f in flattened if f.get(key) is not None]
        if not values:
            aggregate[key] = {
                "mean": None,
                "std": None,
                "n_defined": 0,
                "n_folds": len(fold_metrics),
            }
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregate[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n_defined": int(len(values)),
            "n_folds": len(fold_metrics),
        }
    return aggregate
