"""
Layer 6 - Baseline Candidate Families
=======================================

One shared implementation for the tabular baselines, so a comparison between
them is a comparison of *models* rather than of four separately written
scripts with four different preprocessing bugs.

Families
--------
``structural_heuristic``
    Unsupervised AML rule score.  Reads named ``aml_core_v1`` columns, converts
    each to a within-population rank, and sums the ranks of the signals a
    typical typology cares about.  It never touches a label, which makes it the
    honest floor: any supervised model that cannot beat it has learned nothing
    the structure did not already say.
``logistic_regression``
    Median imputation + standardisation + L2 logistic regression, all fitted on
    the training partition only.
``gradient_boosting``
    Gradient-boosted trees.  Backend is chosen at fit time:
    ``xgboost`` -> ``lightgbm`` -> scikit-learn's ``HistGradientBoostingClassifier``,
    whichever is installed, and the choice is recorded in the fit report so two
    runs are never silently compared across backends.

Preprocessing and leakage
-------------------------
Imputers and scalers are fitted inside ``fit`` on ``ctx.train_idx`` rows only.
Fitting them on the full matrix - the usual quiet mistake - would let test
rows influence the median and the variance used to transform them.

Class imbalance
---------------
``ModelSpec.imbalance`` selects the strategy and is recorded in the artifact:
``"class_weight"`` (balanced weights), ``"pos_weight"`` (n_neg/n_pos, the
gradient-boosting equivalent) or ``"none"``.  Resampling is deliberately not
offered yet: on four positives it manufactures duplicates that inflate every
validation metric.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.experiments.candidate import Candidate, FitContext, register_candidate

logger = logging.getLogger(__name__)

__all__ = [
    "StructuralHeuristicCandidate",
    "LogisticRegressionCandidate",
    "GradientBoostingCandidate",
]


def _rows_as_float(features: pd.DataFrame, index: np.ndarray) -> np.ndarray:
    """The requested rows of ``features`` as a finite-or-NaN float array.

    Selects the rows *before* converting.  Converting the whole frame and then
    slicing costs a float64 copy of the entire matrix on every call, which at
    benchmark scale is hundreds of megabytes per partition per fold and was
    enough to exhaust memory on a 16 GB machine.
    """
    return _frame_as_float(features.take(index, axis=0))


def _frame_as_float(frame: pd.DataFrame) -> np.ndarray:
    """``frame`` as a finite-or-NaN float array."""
    x = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.where(np.isfinite(x), x, np.nan)


class _TabularCandidate(Candidate):
    """Shared train-fitted preprocessing for feature-matrix models."""

    emits_probabilities = True
    calendar_compatible = True

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._imputer = None
        self._scaler = None
        self._columns: list[str] = []

    def _fit_preprocessing(self, ctx: FitContext, *, scale: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if ctx.features is None:
            raise ValueError(f"{self.family} requires a feature matrix.")
        self._columns = [str(c) for c in ctx.features.columns]
        # Kept for scoring outside a fit context: a model that could not be
        # fitted emits this constant, exactly as predict_scores does.
        self._train_prior = float(ctx.y_train.mean()) if len(ctx.y_train) else 0.0
        train = _rows_as_float(ctx.features, ctx.train_idx)

        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x = self._imputer.fit_transform(train)
        if scale:
            self._scaler = StandardScaler()
            x = self._scaler.fit_transform(x)
        return x

    def _transform(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        if ctx.features is None:
            raise ValueError(f"{self.family} requires a feature matrix.")
        return self._transform_frame(ctx.features.take(index, axis=0))

    def _transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [str(c) for c in frame.columns]
        if cols != self._columns:
            raise ValueError(
                f"{self.family}: feature columns changed between fit and predict "
                f"({len(self._columns)} -> {len(cols)}). The experiment must use "
                f"one feature configuration per fold."
            )
        x = _frame_as_float(frame)
        x = self._imputer.transform(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
        return x

    def score_features(self, features: pd.DataFrame) -> np.ndarray:
        """Score a feature matrix with the train-fitted preprocessing and model."""
        if not self._fitted:
            raise ValueError(f"{self.family} must be fitted before scoring.")
        if len(features) == 0:
            return np.array([], dtype=np.float64)
        if getattr(self, "_model", None) is None:
            return np.full(len(features), self._train_prior, dtype=np.float64)
        x = self._transform_frame(features)
        return self._model.predict_proba(x)[:, 1].astype(np.float64)

    def _sample_weight(self, ctx: FitContext) -> Optional[np.ndarray]:
        """Per-sample weights implementing the configured imbalance strategy."""
        y = ctx.y_train
        if ctx.imbalance == "none":
            return None
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            self.fit_report["imbalance_note"] = (
                "training partition is single-class; no reweighting applied"
            )
            return None
        if ctx.imbalance == "pos_weight":
            weight_pos = n_neg / n_pos
            return np.where(y == 1, weight_pos, 1.0).astype(np.float64)
        # "class_weight": balanced
        weight_pos = len(y) / (2.0 * n_pos)
        weight_neg = len(y) / (2.0 * n_neg)
        return np.where(y == 1, weight_pos, weight_neg).astype(np.float64)


@register_candidate("logistic_regression")
class LogisticRegressionCandidate(_TabularCandidate):
    """L2-regularised logistic regression on the standardised feature matrix.

    The interpretable reference point.  Its coefficients are recorded in the
    fit report, so an implausible result can be traced to the column that
    produced it.
    """

    def __init__(self, *, C: float = 1.0, max_iter: int = 2000, **params: Any) -> None:
        super().__init__(C=C, max_iter=max_iter, **params)
        self._model = None

    def fit(self, ctx: FitContext) -> "LogisticRegressionCandidate":
        from sklearn.linear_model import LogisticRegression

        x = self._fit_preprocessing(ctx, scale=True)
        y = ctx.y_train
        if len(np.unique(y)) < 2:
            self._model = None
            self.fit_report = {
                "status": "not_fitted",
                "reason": "training partition is single-class",
                "n_train": int(len(y)),
                "n_train_positive": int(y.sum()),
            }
            self._fitted = True
            return self

        model = LogisticRegression(
            C=float(self.params["C"]),
            max_iter=int(self.params["max_iter"]),
            solver="lbfgs",
        )
        model.fit(x, y, sample_weight=self._sample_weight(ctx))
        self._model = model
        coefs = model.coef_.ravel()
        top = np.argsort(-np.abs(coefs))[:15]
        self.fit_report = {
            "status": "fitted",
            "n_train": int(len(y)),
            "n_train_positive": int(y.sum()),
            "n_features": int(x.shape[1]),
            "n_iter": int(np.max(model.n_iter_)),
            "top_coefficients": {
                self._columns[i]: float(coefs[i]) for i in top
            },
        }
        self._fitted = True
        return self

    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        if len(index) == 0:
            return np.array([], dtype=np.float64)
        if self._model is None:
            # Single-class training: emit the constant prior rather than a
            # fabricated ranking.  Every ranking metric is then correctly
            # reported as chance-level instead of accidentally good.
            prior = float(ctx.y_train.mean()) if len(ctx.y_train) else 0.0
            return np.full(len(index), prior, dtype=np.float64)
        x = self._transform(ctx, index)
        return self._model.predict_proba(x)[:, 1].astype(np.float64)


@register_candidate("gradient_boosting")
class GradientBoostingCandidate(_TabularCandidate):
    """Gradient-boosted decision trees over the feature matrix.

    ``backend="auto"`` prefers xgboost, then lightgbm, then scikit-learn's
    histogram implementation, which is always available.  The resolved backend
    and its version go into the fit report; a comparison across backends is a
    comparison of different models and the artifact must show that.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        early_stopping: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            backend=backend,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            early_stopping=early_stopping,
            **params,
        )
        self._model = None
        self._backend: Optional[str] = None

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        order = ["xgboost", "lightgbm", "sklearn_hist"] if requested == "auto" else [requested]
        for name in order:
            if name == "sklearn_hist":
                return name
            try:
                __import__(name)
                return name
            except ImportError:
                continue
        raise ImportError(f"No gradient-boosting backend available for {requested!r}.")

    def fit(self, ctx: FitContext) -> "GradientBoostingCandidate":
        x = self._fit_preprocessing(ctx, scale=False)
        y = ctx.y_train
        self._backend = self._resolve_backend(str(self.params["backend"]))

        if len(np.unique(y)) < 2:
            self._model = None
            self.fit_report = {
                "status": "not_fitted",
                "reason": "training partition is single-class",
                "backend": self._backend,
                "n_train": int(len(y)),
                "n_train_positive": int(y.sum()),
            }
            self._fitted = True
            return self

        weights = self._sample_weight(ctx)
        if self._backend == "xgboost":
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                max_depth=int(self.params["max_depth"]),
                tree_method="hist",
                eval_metric="aucpr",
                random_state=ctx.seed,
            )
            model.fit(x, y, sample_weight=weights)
            version = xgb.__version__
        elif self._backend == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                max_depth=int(self.params["max_depth"]),
                random_state=ctx.seed,
                verbose=-1,
            )
            model.fit(x, y, sample_weight=weights)
            version = lgb.__version__
        else:
            import sklearn
            from sklearn.ensemble import HistGradientBoostingClassifier

            # Early stopping on an internal split needs enough positives to put
            # some on both sides; below that it silently produces a validation
            # slice with none and stops at iteration one.
            n_pos = int(y.sum())
            use_early = bool(self.params["early_stopping"]) and n_pos >= 20
            model = HistGradientBoostingClassifier(
                max_iter=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                max_depth=int(self.params["max_depth"]),
                early_stopping=use_early,
                random_state=ctx.seed,
            )
            model.fit(x, y, sample_weight=weights)
            version = sklearn.__version__
            if not use_early and bool(self.params["early_stopping"]):
                self.fit_report["early_stopping_note"] = (
                    f"early stopping disabled: only {n_pos} training positives, "
                    f"an internal validation split would likely contain none"
                )

        self._model = model
        self.fit_report.update(
            {
                "status": "fitted",
                "backend": self._backend,
                "backend_version": version,
                "n_train": int(len(y)),
                "n_train_positive": int(y.sum()),
                "n_features": int(x.shape[1]),
            }
        )
        self._fitted = True
        return self

    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        if len(index) == 0:
            return np.array([], dtype=np.float64)
        if self._model is None:
            prior = float(ctx.y_train.mean()) if len(ctx.y_train) else 0.0
            return np.full(len(index), prior, dtype=np.float64)
        x = self._transform(ctx, index)
        return self._model.predict_proba(x)[:, 1].astype(np.float64)

    def save(self, path: Path) -> Optional[Path]:
        if self._model is None:
            return None
        import pickle

        target = Path(f"{path}.pkl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            pickle.dump({"backend": self._backend, "model": self._model}, fh)
        return target


#: Signals the heuristic sums, and the direction that counts as suspicious.
#: Column names are the ``aml_core_v1`` short names; the family prefix is
#: stripped before matching, so the same list works whichever family supplies a
#: column.
#:
#: The ``aml_*`` families in :mod:`src.experiments.aml_features` use different
#: short names for two of these (``min_in_to_out_hours`` rather than
#: ``min_in_to_out_gap_days``, and no ``txn_per_active_day`` outside
#: ``aml_temporal_v1``), so a spec built on those families passes its own
#: ``signals`` list - see ``configs/experiments/samld_heuristic_floor.json``.
#: The default is left alone because artifacts recorded it.
#: Whichever list is used, ``fit_report`` records both the signals that resolved
#: and the ones that did not, so a floor computed from half its signals is
#: visible rather than quietly weak.
_HEURISTIC_SIGNALS: tuple[tuple[str, int], ...] = (
    ("pass_through_ratio", +1),      # money in equals money out: layering
    ("fan_in", +1),                  # many payers into one account: gathering
    ("fan_out", +1),                 # one account into many payees: scattering
    ("reciprocal_ratio", +1),        # back-and-forth with the same parties
    ("txn_per_active_day", +1),      # velocity
    ("burstiness", +1),              # activity concentrated in short spells
    ("min_in_to_out_gap_days", -1),  # short turnaround is suspicious
    ("counterparty_diversity", +1),  # many distinct partners per transaction
)


@register_candidate("structural_heuristic")
class StructuralHeuristicCandidate(Candidate):
    """Unsupervised AML typology score - the floor every model must clear.

    Each configured signal is converted to a percentile rank within the scored
    population and summed with its sign.  The output is normalised to [0, 1] but
    it is a *rank aggregate*, not a probability, so ``emits_probabilities`` is
    ``False`` and calibration metrics are not claimed for it.

    ``fit`` does nothing except record which signals were found.  That is not an
    oversight: a rule score has no parameters to estimate, and giving it access
    to labels would turn the honest floor into another supervised model.
    """

    emits_probabilities = False
    calendar_compatible = True

    def __init__(self, *, signals: Optional[list[list[Any]]] = None, **params: Any) -> None:
        super().__init__(signals=signals, **params)
        self._signals: tuple[tuple[str, int], ...] = (
            tuple((str(name), int(sign)) for name, sign in signals)
            if signals
            else _HEURISTIC_SIGNALS
        )
        self._resolved: dict[str, int] = {}

    def _resolve_columns(self, features: pd.DataFrame) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for column in features.columns:
            short = str(column).split("::")[-1]
            for name, sign in self._signals:
                if short == name:
                    resolved[str(column)] = sign
        return resolved

    def fit(self, ctx: FitContext) -> "StructuralHeuristicCandidate":
        if ctx.features is None:
            raise ValueError("structural_heuristic requires a feature matrix.")
        self._resolved = self._resolve_columns(ctx.features)
        missing = sorted(
            {name for name, _ in self._signals}
            - {str(c).split("::")[-1] for c in self._resolved}
        )
        self.fit_report = {
            "status": "no_parameters_to_fit",
            "signals_used": sorted(self._resolved),
            "signals_missing": missing,
            "labels_used": False,
        }
        if not self._resolved:
            self.fit_report["warning"] = (
                "none of the configured signals are present in the feature "
                "matrix; the score will be constant"
            )
        self._fitted = True
        return self

    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        if len(index) == 0:
            return np.array([], dtype=np.float64)
        if ctx.features is None:
            raise ValueError("structural_heuristic requires a feature matrix.")
        if not self._resolved:
            return np.zeros(len(index), dtype=np.float64)
        # Ranks are computed over the whole population so a fold's score does
        # not depend on how many entities happen to be in that partition.
        return self._rank_scores(ctx.features)[index].astype(np.float64)

    def score_features(self, features: pd.DataFrame) -> np.ndarray:
        """Rank-aggregate score of every row, ranked within ``features``."""
        if not self._fitted:
            raise ValueError("structural_heuristic must be fitted before scoring.")
        if len(features) == 0:
            return np.array([], dtype=np.float64)
        if not self._resolved:
            return np.zeros(len(features), dtype=np.float64)
        return self._rank_scores(features).astype(np.float64)

    def _rank_scores(self, features: pd.DataFrame) -> np.ndarray:
        total = np.zeros(len(features), dtype=np.float64)
        for column, sign in self._resolved.items():
            series = pd.to_numeric(features[column], errors="coerce")
            ranked = series.rank(pct=True, na_option="keep").to_numpy(dtype=np.float64)
            # An entity for which a signal is undefined contributes the neutral
            # 0.5 for that signal rather than being pushed to either extreme.
            ranked = np.where(np.isnan(ranked), 0.5, ranked)
            total += sign * ranked
        n_signals = len(self._resolved)
        # Map the signed rank sum into [0, 1]; a signal with sign -1 has range
        # [-1, 0], so the achievable total lies in [-n_neg, n_pos].
        n_pos_signals = sum(1 for s in self._resolved.values() if s > 0)
        n_neg_signals = n_signals - n_pos_signals
        lo, hi = -float(n_neg_signals), float(n_pos_signals)
        span = hi - lo
        return (total - lo) / span if span > 0 else np.zeros_like(total)
