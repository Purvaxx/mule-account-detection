"""
Layer 6 - Model-Agnostic Experiment Framework
===============================================

Everything needed to state an experiment, run it, and produce an artifact that
another session can compare against.  No part of it assumes a model family, a
feature count, a positive count, or a split.

Sub-modules
-----------
spec         Serializable description of one experiment.
dataset      ``ExperimentDataset``: entities, labels, edges, graph.
reference    Reference Dataset loader, via Layers 1-3.
sources      Loaders for external benchmarks (SAML-D, IBM AML, Kaggle fraud).
audit        Dataset scientific audit and usability verdict.
features     Versioned feature registry; legacy adapters plus ``aml_core_v1``.
aml_features Scalable AML families: transaction, temporal, flow, network,
             cycle, context. Separately registered so an ablation can turn one
             off.
splits       Named split strategies, each declaring its evaluation regime.
calendar     Calendar-forward evaluation: (entity, origin) samples, labels
             from a post-origin window, features from before the origin.
detection_transfer
             Cross-period retrospective detection: features and labels from the
             same completed window; a frozen model is applied to a later one.
snapshots    Window feature snapshots and their cache, shared by both regimes.
drift        Population Stability Index - one definition for the project.
candidate    The ``Candidate`` contract and its registry.
baselines    Structural heuristic, logistic regression, gradient boosting.
legacy_graphsage
             The Experiment 3 architecture behind the same contract.
metrics      Rare-positive metrics; undefined values stay ``None``.
thresholds   Validation-only decision-threshold selection.
calibration  Validation-only score-to-probability mapping.
leakage      Automatic audits recorded in every artifact.
seeding      Central seed control and an honest determinism report.
runner       Orchestration.
artifacts    Result containers and JSON persistence.

Before comparing models
-----------------------
A comparison is only meaningful on a dataset that can separate models, so the
dataset is classified first (``audit``) and the verdict is an artifact. As of
2026-09-06 no dataset available to this project is ``MODEL_SELECTION_READY``;
see ``docs/datasets.md``. Results produced on a dataset classified
``STRUCTURAL_VALIDATION_ONLY`` establish that the pipeline is correct and
affordable, not that one model is better than another.

Relationship to the legacy pipeline
-----------------------------------
Layers 1-5 (``src.data``, ``src.graph``, ``src.features``, ``src.models``) and
the frozen Experiment 3 checkpoint are untouched by this package.  The legacy
adapters read them; nothing here writes to them.  Experiment 3 is a benchmark
and a regression reference, not an architectural invariant.
"""

from src.experiments.spec import (
    CalibrationSpec,
    DatasetSpec,
    EvaluationSpec,
    ExperimentSpec,
    FeatureSpec,
    ModelSpec,
    SplitSpec,
    ThresholdSpec,
)
from src.experiments.dataset import ExperimentDataset, dataset_from_graph_result
from src.experiments.candidate import (
    Candidate,
    FitContext,
    available_candidates,
    build_candidate,
    register_candidate,
)
from src.experiments.features import (
    FeatureRequest,
    FeatureSet,
    available_feature_sets,
    build_feature_matrix,
    register_feature_set,
)
from src.experiments.audit import DatasetAudit, audit_dataset, save_audit
from src.experiments.sources import (
    SourceSpec,
    available_sources,
    build_transaction_dataset,
    load_source,
)
from src.experiments.splits import (
    SplitFold,
    SplitPlan,
    available_split_strategies,
    build_split_plan,
)
from src.experiments.calendar import (
    CALENDAR_STRATEGY,
    CalendarPlan,
    build_calendar_plan,
    run_calendar_experiment,
)
from src.experiments.detection_transfer import (
    DETECTION_TRANSFER_STRATEGY,
    DetectionTransferPlan,
    build_detection_transfer_plan,
    run_detection_transfer_experiment,
)
from src.experiments.metrics import aggregate_fold_metrics, binary_metrics
from src.experiments.thresholds import ThresholdDecision, select_threshold
from src.experiments.runner import run_experiment
from src.experiments.artifacts import ExperimentResult, FoldOutcome, load_result, save_result

__all__ = [
    "CalibrationSpec",
    "DatasetSpec",
    "EvaluationSpec",
    "ExperimentSpec",
    "FeatureSpec",
    "ModelSpec",
    "SplitSpec",
    "ThresholdSpec",
    "ExperimentDataset",
    "dataset_from_graph_result",
    "Candidate",
    "FitContext",
    "available_candidates",
    "build_candidate",
    "register_candidate",
    "FeatureRequest",
    "FeatureSet",
    "available_feature_sets",
    "build_feature_matrix",
    "register_feature_set",
    "DatasetAudit",
    "audit_dataset",
    "save_audit",
    "SourceSpec",
    "available_sources",
    "build_transaction_dataset",
    "load_source",
    "SplitFold",
    "SplitPlan",
    "available_split_strategies",
    "build_split_plan",
    "CALENDAR_STRATEGY",
    "CalendarPlan",
    "build_calendar_plan",
    "run_calendar_experiment",
    "DETECTION_TRANSFER_STRATEGY",
    "DetectionTransferPlan",
    "build_detection_transfer_plan",
    "run_detection_transfer_experiment",
    "aggregate_fold_metrics",
    "binary_metrics",
    "ThresholdDecision",
    "select_threshold",
    "run_experiment",
    "ExperimentResult",
    "FoldOutcome",
    "load_result",
    "save_result",
]
