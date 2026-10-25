"""
Layer 6 - Leakage Audit
========================

Checks that run automatically inside every experiment and record what they
found in the artifact.  They are not decorative: on a dataset with four
positives, a single leaked label produces a perfect score, and a perfect score
is exactly the result nobody questions.

What is checked
---------------
``partition_overlap``
    Train / validation / test share no entity.  A hard error - a split that
    fails this is broken, not merely optimistic.
``label_columns``
    No column of the feature matrix is the label or a post-investigation
    artefact of it (``core``, ``investigation``, ``source_file``, ...).  Hard
    error.
``constant_or_perfect_features``
    Any single feature column that separates the training classes perfectly.
    Reported as a warning, because on a rare-positive dataset a perfect
    separator is far more often a leak than a discovery.
``temporal_ordering``
    For a time-aware split, every evaluation entity's reference time is at or
    after every training entity's.  Warning if violated, with the magnitude.
``feature_scope``
    Whether features were computed on the whole population or only the training
    window.  Not a failure - a statement of regime that belongs in the report.
``graph_adjacency``
    For a split claiming graph-leakage control, that no test entity is adjacent
    to a training entity.

Severity
--------
``error``   the run must stop; the result would be meaningless.
``warning`` the run continues and the finding is recorded prominently.
``info``    a regime declaration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.dataset import ExperimentDataset, FORBIDDEN_FEATURE_COLUMNS
from src.experiments.splits import SplitFold, SplitPlan

logger = logging.getLogger(__name__)

__all__ = [
    "LeakageFinding",
    "audit_split_plan",
    "audit_features",
    "audit_fold",
    "audit_label_columns",
    "audit_perfect_separators",
    "LeakageError",
]


class LeakageError(RuntimeError):
    """Raised when an audit finds a condition that invalidates the experiment."""


@dataclass(frozen=True)
class LeakageFinding:
    check: str
    severity: str  # error | warning | info
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
        }
        if self.detail:
            d["detail"] = self.detail
        return d


def raise_on_errors(findings: Sequence[LeakageFinding]) -> None:
    """Abort the run if any finding is an error."""
    errors = [f for f in findings if f.severity == "error"]
    if errors:
        raise LeakageError(
            "Leakage audit failed:\n"
            + "\n".join(f"  - [{f.check}] {f.message}" for f in errors)
        )


def audit_split_plan(plan: SplitPlan, dataset: ExperimentDataset) -> list[LeakageFinding]:
    """Structural checks on the whole plan."""
    findings: list[LeakageFinding] = []

    for fold in plan.folds:
        parts = {"train": fold.train_idx, "val": fold.val_idx, "test": fold.test_idx}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = np.intersect1d(parts[a], parts[b])
            if overlap.size:
                findings.append(
                    LeakageFinding(
                        check="partition_overlap",
                        severity="error",
                        message=(
                            f"fold {fold.fold_idx}: {a} and {b} share "
                            f"{overlap.size} entities"
                        ),
                        detail={"examples": dataset.ids_for(overlap[:5])},
                    )
                )

    findings.append(
        LeakageFinding(
            check="evaluation_regime",
            severity="info",
            message=(
                f"split '{plan.strategy}' is {plan.regime()}: "
                f"entity_disjoint={plan.entity_disjoint}, "
                f"time_aware={plan.time_aware}, "
                f"graph_leakage_controlled={plan.graph_leakage_controlled}"
            ),
        )
    )
    if not plan.time_aware and dataset.has_time():
        findings.append(
            LeakageFinding(
                check="temporal_ordering",
                severity="warning",
                message=(
                    "the dataset carries timestamps but the split ignores them; "
                    "the model may be fitted on behaviour that post-dates the "
                    "behaviour it is scored on"
                ),
            )
        )

    if plan.time_aware:
        findings.extend(_audit_temporal_ordering(plan, dataset))
    if plan.graph_leakage_controlled:
        findings.extend(_audit_graph_adjacency(plan, dataset))

    return findings


def _audit_temporal_ordering(plan: SplitPlan, dataset: ExperimentDataset) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    reference = plan.params.get("time_reference", "first")
    times = dataset.entity_times(how=reference)
    values = times.to_numpy()

    for fold in plan.folds:
        for name, idx in (("val", fold.val_idx), ("test", fold.test_idx)):
            if len(fold.train_idx) == 0 or len(idx) == 0:
                continue
            train_max = pd.Series(values[fold.train_idx]).max()
            eval_min = pd.Series(values[idx]).min()
            if pd.isna(train_max) or pd.isna(eval_min):
                continue
            if eval_min < train_max:
                findings.append(
                    LeakageFinding(
                        check="temporal_ordering",
                        severity="warning",
                        message=(
                            f"fold {fold.fold_idx}: earliest {name} entity "
                            f"({eval_min}) precedes the latest training entity "
                            f"({train_max})"
                        ),
                        detail={"overlap_days": int((train_max - eval_min).days)},
                    )
                )
    return findings


def _audit_graph_adjacency(plan: SplitPlan, dataset: ExperimentDataset) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    if dataset.edge_table is None or len(dataset.edge_table) == 0:
        return findings
    src = dataset.edge_table["source_entity"].to_numpy()
    dst = dataset.edge_table["target_entity"].to_numpy()

    for fold in plan.folds:
        train_ids = set(dataset.ids_for(fold.train_idx))
        test_ids = set(dataset.ids_for(fold.test_idx))
        crossing = sum(
            1
            for u, v in zip(src, dst)
            if (u in train_ids and v in test_ids) or (u in test_ids and v in train_ids)
        )
        if crossing:
            findings.append(
                LeakageFinding(
                    check="graph_adjacency",
                    severity="warning",
                    message=(
                        f"fold {fold.fold_idx}: split claims graph-leakage control "
                        f"but {crossing} transactions connect train and test entities"
                    ),
                )
            )
    return findings


def audit_features(
    features: Optional[pd.DataFrame],
    dataset: ExperimentDataset,
    fold: SplitFold,
    *,
    scope: str,
) -> list[LeakageFinding]:
    """Checks on the feature matrix for one fold."""
    findings: list[LeakageFinding] = []
    if features is None:
        return findings

    findings.extend(audit_label_columns(features))

    if scope == "global":
        findings.append(
            LeakageFinding(
                check="feature_scope",
                severity="info",
                message=(
                    "features were computed on the whole population, so "
                    "structural and aggregate columns encode information about "
                    "held-out entities (transductive feature regime)"
                ),
            )
        )
    else:
        findings.append(
            LeakageFinding(
                check="feature_scope",
                severity="info",
                message=(
                    f"features were computed only from transactions before "
                    f"{fold.train_window_end}"
                ),
            )
        )

    findings.extend(_audit_perfect_separators(features, dataset, fold))
    return findings


def audit_label_columns(features: Optional[pd.DataFrame]) -> list[LeakageFinding]:
    """Hard error when a column of ``features`` is the label or an artefact of it."""
    if features is None:
        return []
    lowered = {c.split("::")[-1].lower(): c for c in features.columns}
    hits = sorted(set(lowered) & {f.lower() for f in FORBIDDEN_FEATURE_COLUMNS})
    if not hits:
        return []
    return [
        LeakageFinding(
            check="label_columns",
            severity="error",
            message=f"feature matrix contains label or investigation columns: {hits}",
            detail={"columns": [lowered[h] for h in hits]},
        )
    ]


def _audit_perfect_separators(
    features: pd.DataFrame, dataset: ExperimentDataset, fold: SplitFold
) -> list[LeakageFinding]:
    """Flag single columns that perfectly separate the training classes."""
    return audit_perfect_separators(
        features, dataset.labels[fold.train_idx], rows=fold.train_idx
    )


def audit_perfect_separators(
    features: pd.DataFrame,
    y: np.ndarray,
    *,
    rows: Optional[np.ndarray] = None,
) -> list[LeakageFinding]:
    """Flag single columns that perfectly separate the classes of ``y``.

    ``rows`` are the positional rows of ``features`` that ``y`` is aligned to;
    by default every row, in order.  Taking the labels explicitly lets a split
    whose labels are not ``dataset.labels`` (the calendar-forward split) run the
    same check.
    """
    y = np.asarray(y)
    if rows is None:
        rows = np.arange(len(features), dtype=np.int64)
    if len(np.unique(y)) < 2:
        return []
    # One column at a time. Materialising the whole matrix as float64 and then
    # slicing it costs two full copies of the feature matrix, which on a
    # benchmark-scale population is gigabytes for an audit that only needs four
    # numbers per column.
    pos_rows = rows[y == 1]
    neg_rows = rows[y == 0]
    suspects: list[str] = []
    with np.errstate(invalid="ignore"):
        for column in features.columns:
            values = features[column].to_numpy(dtype=np.float64, na_value=np.nan)
            pos = values[pos_rows]
            neg = values[neg_rows]
            if not np.isfinite(pos).any() or not np.isfinite(neg).any():
                continue
            pos_min, pos_max = np.nanmin(pos), np.nanmax(pos)
            neg_min, neg_max = np.nanmin(neg), np.nanmax(neg)
            if pos_min > neg_max or neg_min > pos_max:
                suspects.append(str(column))
    if not suspects:
        return []
    return [
        LeakageFinding(
            check="constant_or_perfect_features",
            severity="warning",
            message=(
                f"{len(suspects)} feature column(s) separate the training classes "
                f"perfectly; on a rare-positive dataset that is more often a leak "
                f"than a discovery"
            ),
            detail={"columns": suspects[:20], "n_train_positive": int(y.sum())},
        )
    ]


def audit_fold(
    dataset: ExperimentDataset,
    fold: SplitFold,
    features: Optional[pd.DataFrame],
    *,
    scope: str,
) -> list[LeakageFinding]:
    """All per-fold checks, in one call."""
    findings = audit_features(features, dataset, fold, scope=scope)
    if len(fold.train_idx) == 0:
        findings.append(
            LeakageFinding(
                check="train_partition",
                severity="warning",
                message=(
                    f"fold {fold.fold_idx}: the training partition is empty; only "
                    f"label-free candidates can produce a meaningful score here"
                ),
            )
        )
    n_train_pos = int(dataset.labels[fold.train_idx].sum())
    if n_train_pos == 0 and len(fold.train_idx) > 0:
        findings.append(
            LeakageFinding(
                check="train_positives",
                severity="warning",
                message=(
                    f"fold {fold.fold_idx}: no positives in the training partition; "
                    f"a supervised candidate has nothing to learn from"
                ),
            )
        )
    return findings
