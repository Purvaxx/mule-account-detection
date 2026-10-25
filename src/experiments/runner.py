"""
Layer 6 - Experiment Runner
=============================

Executes one :class:`~src.experiments.spec.ExperimentSpec` against one
:class:`~src.experiments.dataset.ExperimentDataset` and returns a complete
artifact.

The pipeline, in the order the science requires:

    dataset
      -> split plan               (declared regime; audited)
      -> per fold:
           feature computation    (scoped: whole population, or training window)
           leakage audit
           fit on train only
           score validation
           fit calibration on validation
           choose threshold on validation
           score test, apply the frozen threshold
           evaluate
      -> aggregate across folds
      -> artifact

Two invariants are enforced here rather than trusted to callers:

1. **Test data reaches exactly one place.**  ``_score`` is called on the test
   index once per fold, after the threshold and the calibrator are already
   fixed.  Nothing downstream of that point can alter them.
2. **Undefined stays undefined.**  A fold whose validation partition has no
   positives yields ``threshold = None`` and null threshold metrics; it does
   not fall back to 0.5.

Hyperparameter search, when it is added, belongs *inside* the fold on the
validation partition - never across folds using test results.  The runner
deliberately provides no hook that could be used the other way.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.experiments.artifacts import ExperimentResult, FoldOutcome, git_commit
from src.experiments.calendar import CALENDAR_SCOPE, CALENDAR_STRATEGY, run_calendar_experiment
from src.experiments.detection_transfer import (
    DETECTION_SCOPE,
    DETECTION_TRANSFER_STRATEGY,
    run_detection_transfer_experiment,
)
from src.experiments.calibration import build_calibrator
from src.experiments.candidate import (
    Candidate,
    FitContext,
    build_candidate,
    checkpoint_identity,
)
from src.experiments.dataset import ExperimentDataset
from src.experiments.features import FeatureRequest, FeatureMatrix, build_feature_matrix
from src.experiments.leakage import audit_fold, audit_split_plan, raise_on_errors
from src.experiments.metrics import aggregate_fold_metrics, binary_metrics
from src.experiments.seeding import (
    derive_seed,
    determinism_report,
    environment_report,
    set_global_seeds,
)
from src.experiments.spec import ExperimentSpec
from src.experiments.splits import SplitFold, SplitPlan, build_split_plan
from src.experiments.thresholds import select_threshold

logger = logging.getLogger(__name__)

__all__ = ["run_experiment"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _needs_graph(spec: ExperimentSpec) -> bool:
    """Whether any requested feature family asks for a ``networkx`` graph.

    Building a ``MultiDiGraph`` over a nine-million-row edge table costs tens of
    gigabytes, and none of the ``aml_*`` families need one - they work on the
    edge table and a deduplicated sparse adjacency.  The legacy families do, so
    the object is built when, and only when, one of them is requested.
    """
    from src.experiments.features import get_feature_set

    return any(get_feature_set(name).requires_graph for name in spec.features.sets)


def _compute_features(
    dataset: ExperimentDataset,
    spec: ExperimentSpec,
    fold: SplitFold,
) -> Optional[FeatureMatrix]:
    """Build the feature matrix visible to this fold.

    Under ``scope == "train_window"`` the edge table is truncated at the fold's
    training-window boundary and the graph is rebuilt from exactly those rows,
    so no feature can summarise a transaction the model would not yet have
    seen.  Under ``scope == "global"`` the full history is used and the audit
    records that regime.
    """
    if not spec.features.sets:
        return None

    if spec.features.scope == CALENDAR_SCOPE:
        raise ValueError(
            f"FeatureSpec.scope={CALENDAR_SCOPE!r} is defined only by the "
            f"{CALENDAR_STRATEGY!r} split, which gives each snapshot its own "
            f"origin; split {spec.split.strategy!r} defines none."
        )
    if spec.features.scope == DETECTION_SCOPE:
        raise ValueError(
            f"FeatureSpec.scope={DETECTION_SCOPE!r} is defined only by the "
            f"{DETECTION_TRANSFER_STRATEGY!r} split, which gives each snapshot its "
            f"own detection window; split {spec.split.strategy!r} defines none."
        )

    if spec.features.scope == "train_window":
        if fold.train_window_end is None:
            raise ValueError(
                "FeatureSpec.scope='train_window' requires a time-aware split "
                "that defines a training-window boundary; "
                f"fold {fold.fold_idx} has none."
            )
        edges = dataset.edges_before(fold.train_window_end)
        graph = dataset.subgraph_from_edges(edges) if _needs_graph(spec) else None
        as_of = fold.train_window_end
    else:
        edges = dataset.edge_table
        graph = dataset.graph
        as_of = None

    request = FeatureRequest(
        entity_ids=list(dataset.entity_ids),
        edge_table=edges,
        graph=graph,
        as_of=as_of,
        attributes=dataset.attributes,
    )
    matrix = build_feature_matrix(request, spec.features.sets)
    matrix.scope = spec.features.scope
    return matrix


def run_experiment(
    spec: ExperimentSpec,
    dataset: ExperimentDataset,
    *,
    candidate_factory: Optional[Any] = None,
    output_dir: Optional[Path | str] = None,
    checkpoint_dir: Optional[Path | str] = None,
) -> ExperimentResult:
    """Run every fold of ``spec`` on ``dataset`` and return the artifact.

    Parameters
    ----------
    candidate_factory:
        Optional zero-argument callable returning a fresh, unfitted
        :class:`~src.experiments.candidate.Candidate`.  Defaults to building
        ``spec.model.family`` from the registry.  A fresh instance is created
        per fold; reusing one would carry the previous fold's fit across the
        partition boundary.
    output_dir:
        When given, the artifact JSON is written there.
    checkpoint_dir:
        When given, each fold's model is asked to persist itself there and the
        resulting file's digest is recorded as the checkpoint identity.
    """
    if spec.split.strategy == DETECTION_TRANSFER_STRATEGY:
        # Retrospective detection transfer: features and labels come from the
        # same completed window, and a fold applies a frozen model to a later
        # one.  Its own loop, with the same validation-only threshold rules.
        return run_detection_transfer_experiment(
            spec,
            dataset,
            candidate_factory=candidate_factory,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

    if spec.split.strategy == CALENDAR_STRATEGY:
        # A calendar split scores (entity, origin) samples with a label drawn
        # from a time window, which is not an entity partition; it has its own
        # loop and the same validation-only threshold and calibration rules.
        return run_calendar_experiment(
            spec,
            dataset,
            candidate_factory=candidate_factory,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

    started = _now()
    t_start = time.perf_counter()

    if candidate_factory is None:
        def candidate_factory() -> Candidate:  # type: ignore[misc]
            return build_candidate(spec.model.family, spec.model.params)

    seed_report = set_global_seeds(spec.seed)
    plan: SplitPlan = build_split_plan(dataset, spec.split)

    plan_findings = audit_split_plan(plan, dataset)
    raise_on_errors(plan_findings)
    all_findings = [f.to_dict() for f in plan_findings]

    notes: list[str] = list(plan.notes)
    if spec.dataset.synthetic:
        notes.append(
            "SYNTHETIC DATASET: these numbers describe generated data and are "
            "not evidence about real-world detection performance."
        )
    if spec.features.scope == "global":
        notes.append(
            "features were computed on the whole population (transductive "
            "feature regime); see the leakage block."
        )
        notes.append(
            "the feature matrix is identical for every fold under this regime "
            "and was computed once; per-fold feature timings after fold 0 "
            "measure the reuse, not the computation."
        )

    feature_summary: dict[str, Any] = {}
    outcomes: list[FoldOutcome] = []

    # Under the global feature regime every fold sees the same transactions, so
    # the matrix is identical across folds and is computed once. Under
    # train_window it must not be: each fold has its own boundary and reusing
    # another fold's matrix would import transactions from beyond it.
    global_matrix: Optional[FeatureMatrix] = None
    reuse_global = spec.features.scope == "global" and bool(spec.features.sets)

    for fold in plan.folds:
        fold_seed = derive_seed(spec.seed, "fold", fold.fold_idx)
        set_global_seeds(fold_seed)

        t_feat = time.perf_counter()
        if reuse_global:
            if global_matrix is None:
                global_matrix = _compute_features(dataset, spec, fold)
            matrix = global_matrix
        else:
            matrix = _compute_features(dataset, spec, fold)
        features = matrix.values if matrix is not None else None
        feat_seconds = time.perf_counter() - t_feat
        if matrix is not None and not feature_summary:
            feature_summary = matrix.to_dict()

        fold_findings = audit_fold(dataset, fold, features, scope=spec.features.scope)
        raise_on_errors(fold_findings)
        all_findings.extend(
            f.to_dict() | {"fold_idx": fold.fold_idx} for f in fold_findings
        )

        ctx = FitContext(
            dataset=dataset,
            features=features,
            fold=fold,
            train_idx=fold.train_idx,
            val_idx=fold.val_idx,
            y_train=dataset.labels[fold.train_idx],
            y_val=dataset.labels[fold.val_idx],
            train_edges=(
                dataset.edges_before(fold.train_window_end)
                if spec.features.scope == "train_window"
                else dataset.edge_table
            ),
            seed=fold_seed,
            imbalance=spec.model.imbalance,
        )

        candidate = candidate_factory()

        t_fit = time.perf_counter()
        candidate.fit(ctx)
        fit_seconds = time.perf_counter() - t_fit

        # --- validation: the only partition allowed to shape decisions ------
        t_score = time.perf_counter()
        val_raw = (
            candidate.predict_scores(ctx, fold.val_idx)
            if len(fold.val_idx)
            else np.array([], dtype=np.float64)
        )
        y_val = dataset.labels[fold.val_idx]

        calibrator, cal_notes = build_calibrator(
            spec.calibration,
            y_val,
            val_raw,
            model_emits_probabilities=candidate.emits_probabilities,
        )
        val_scores = calibrator.transform(val_raw) if len(val_raw) else val_raw

        decision = select_threshold(spec.threshold, y_val, val_scores)

        # --- test: scored once, with everything already frozen --------------
        test_raw = (
            candidate.predict_scores(ctx, fold.test_idx)
            if len(fold.test_idx)
            else np.array([], dtype=np.float64)
        )
        test_scores = calibrator.transform(test_raw) if len(test_raw) else test_raw
        score_seconds = time.perf_counter() - t_score

        probability_scale = calibrator.produces_probabilities
        val_metrics = binary_metrics(
            y_val,
            val_scores,
            threshold=decision.value,
            k_values=spec.evaluation.k_values,
            alert_rates=spec.evaluation.alert_rates,
            probability_scale=probability_scale,
            calibration_bins=spec.evaluation.calibration_bins,
        )
        test_metrics = binary_metrics(
            dataset.labels[fold.test_idx],
            test_scores,
            threshold=decision.value,
            k_values=spec.evaluation.k_values,
            alert_rates=spec.evaluation.alert_rates,
            probability_scale=probability_scale,
            calibration_bins=spec.evaluation.calibration_bins,
        )

        checkpoint: Optional[dict[str, Any]] = None
        if checkpoint_dir is not None:
            directory = Path(checkpoint_dir)
            directory.mkdir(parents=True, exist_ok=True)
            saved = candidate.save(directory / f"{spec.name}_fold{fold.fold_idx}")
            checkpoint = checkpoint_identity(saved)

        outcomes.append(
            FoldOutcome(
                fold_idx=fold.fold_idx,
                counts=fold.counts(dataset.labels),
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                threshold=decision.to_dict(),
                calibration=calibrator.to_dict(),
                leakage=[f.to_dict() for f in fold_findings],
                fit_report=dict(candidate.fit_report),
                runtime_seconds={
                    "features": feat_seconds,
                    "fit": fit_seconds,
                    "score": score_seconds,
                },
                checkpoint=checkpoint,
                notes=cal_notes + list(ctx.notes),
            )
        )
        logger.info(
            "fold %d: test AP=%s  threshold=%s",
            fold.fold_idx,
            test_metrics.get("average_precision"),
            decision.value,
        )

    aggregate = aggregate_fold_metrics([o.test_metrics for o in outcomes])
    aggregate_val = aggregate_fold_metrics([o.validation_metrics for o in outcomes])

    finished = _now()
    result = ExperimentResult(
        spec=spec,
        dataset_summary=dataset.summary(),
        split_summary=plan.to_dict(labels=dataset.labels),
        feature_summary=feature_summary,
        folds=outcomes,
        aggregate=aggregate,
        validation_aggregate=aggregate_val,
        environment=environment_report() | {"seeding": seed_report},
        determinism=determinism_report(),
        leakage=all_findings,
        notes=notes,
        started_at=started,
        finished_at=finished,
        total_runtime_seconds=time.perf_counter() - t_start,
        git_commit=git_commit(Path(__file__).resolve().parents[2]),
    )

    if output_dir is not None:
        from src.experiments.artifacts import save_result

        save_result(result, output_dir)

    return result
