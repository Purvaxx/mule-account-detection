"""
Layer 6 - Probability Calibration
===================================

Separates three things the legacy pipeline conflated:

    raw score              whatever the model emits (a sigmoid output, a margin,
                           a heuristic sum) - comparable within a model, not
                           across models, and not a probability
    calibrated probability the raw score mapped onto [0, 1] so that "0.3" means
                           "about 30 % of entities scored this way are positive"
    decision               the thresholded alert (see
                           :mod:`src.experiments.thresholds`)

Calibrators are fitted on the **validation** partition only.  Fitting on train
would inherit the model's overconfidence on data it has memorised; fitting on
test would leak.

Small-data guard
----------------
A reliability curve estimated from four positives is noise wearing a lab coat.
``build_calibrator`` refuses to fit below ``CalibrationSpec.min_val_positives``
and falls back to the identity map, recording why.  The run continues - ranking
metrics do not need calibration - but no calibrated probability is claimed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np

from src.experiments.spec import CalibrationSpec

logger = logging.getLogger(__name__)

__all__ = ["Calibrator", "IdentityCalibrator", "PlattCalibrator", "IsotonicCalibrator", "build_calibrator"]


class Calibrator(ABC):
    """Maps raw scores onto a probability scale."""

    name: str = "abstract"

    def __init__(self) -> None:
        self.fitted: bool = False
        self.diagnostics: dict[str, Any] = {}

    @abstractmethod
    def fit(self, y: np.ndarray, scores: np.ndarray) -> "Calibrator":
        ...

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def produces_probabilities(self) -> bool:
        """True when ``transform`` output may be read as a probability."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "fitted": self.fitted,
            "produces_probabilities": self.produces_probabilities,
            "diagnostics": dict(self.diagnostics),
        }


class IdentityCalibrator(Calibrator):
    """Pass-through.

    ``produces_probabilities`` is set by the caller: a model that already emits
    a sigmoid output is on a probability scale even without calibration, while a
    heuristic sum is not.  The distinction decides whether Brier score and ECE
    are computed at all.
    """

    name = "none"

    def __init__(self, *, probability_scale: bool = False, reason: Optional[str] = None) -> None:
        super().__init__()
        self._probability_scale = probability_scale
        self.fitted = True
        if reason:
            self.diagnostics["reason"] = reason

    def fit(self, y: np.ndarray, scores: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64)

    @property
    def produces_probabilities(self) -> bool:
        return self._probability_scale


class PlattCalibrator(Calibrator):
    """Logistic (sigmoid) calibration: a 1-D logistic regression on the score."""

    name = "platt"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def fit(self, y: np.ndarray, scores: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        yy = np.asarray(y).reshape(-1).astype(np.int64)
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(x, yy)
        self._model = model
        self.fitted = True
        self.diagnostics = {
            "n_fit": int(len(yy)),
            "n_fit_positive": int(yy.sum()),
            "coef": float(model.coef_.ravel()[0]),
            "intercept": float(model.intercept_.ravel()[0]),
        }
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PlattCalibrator.transform called before fit.")
        x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self._model.predict_proba(x)[:, 1].astype(np.float64)

    @property
    def produces_probabilities(self) -> bool:
        return self.fitted


class IsotonicCalibrator(Calibrator):
    """Non-parametric monotone calibration.

    More flexible than Platt and correspondingly hungrier for data; it will
    happily interpolate a step function through a handful of positives, which
    is why the ``min_val_positives`` guard exists.
    """

    name = "isotonic"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def fit(self, y: np.ndarray, scores: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        yy = np.asarray(y).reshape(-1).astype(np.float64)
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(s, yy)
        self._model = model
        self.fitted = True
        self.diagnostics = {
            "n_fit": int(len(yy)),
            "n_fit_positive": int(yy.sum()),
        }
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator.transform called before fit.")
        return np.asarray(
            self._model.predict(np.asarray(scores, dtype=np.float64).reshape(-1)),
            dtype=np.float64,
        )

    @property
    def produces_probabilities(self) -> bool:
        return self.fitted


_METHODS = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def build_calibrator(
    spec: CalibrationSpec,
    y_val: Sequence[int],
    scores_val: Sequence[float],
    *,
    model_emits_probabilities: bool,
) -> tuple[Calibrator, list[str]]:
    """Fit the requested calibrator on validation data, or explain why not.

    Returns the calibrator and a list of human-readable notes to carry into the
    experiment artifact.
    """
    notes: list[str] = []
    if spec.method not in _METHODS:
        raise KeyError(
            f"Unknown calibration method {spec.method!r}. Available: {sorted(_METHODS)}"
        )

    if spec.method == "none":
        return (
            IdentityCalibrator(probability_scale=model_emits_probabilities),
            notes,
        )

    y = np.asarray(y_val).reshape(-1).astype(np.int64)
    s = np.asarray(scores_val, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())

    if len(y) == 0:
        reason = "validation partition is empty; calibration not fitted"
        notes.append(reason)
        return IdentityCalibrator(probability_scale=model_emits_probabilities, reason=reason), notes

    if n_pos < spec.min_val_positives:
        reason = (
            f"validation has {n_pos} positives, below the configured minimum of "
            f"{spec.min_val_positives}; calibration not fitted and no calibrated "
            f"probability is claimed"
        )
        notes.append(reason)
        return IdentityCalibrator(probability_scale=model_emits_probabilities, reason=reason), notes

    if len(np.unique(y)) < 2:
        reason = "validation partition is single-class; calibration not fitted"
        notes.append(reason)
        return IdentityCalibrator(probability_scale=model_emits_probabilities, reason=reason), notes

    calibrator = _METHODS[spec.method]()
    calibrator.fit(y, s)
    notes.append(
        f"{spec.method} calibration fitted on {len(y)} validation entities "
        f"({n_pos} positive)"
    )
    return calibrator, notes
