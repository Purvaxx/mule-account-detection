"""
Layer 6 - Cross-Period Detection Transfer (R1)
==============================================

Does a detection model fitted on one **completed** analysis period still rank
entities usefully in a **later** completed period, applied unchanged?

This is retrospective detection, not forecasting
------------------------------------------------
For a detection window ``[A, B)``::

    features   transactions dated in [A, B)          the completed period
    labels     flagged involvement dated in [A, B)   the same period
    score at   B    "given this finished period, which entities do we investigate?"

Features and labels deliberately describe the *same* window: nothing is being
predicted before it happens. The transfer question is about *time*::

    fit on [A1, B1)  ->  freeze  ->  apply unchanged to [A2, B2),  B1 <= A2

``src/experiments/calendar.py`` answers a different question - features before an
origin, labels after it - and its numbers are not comparable to these. The two
regimes are separate strategies, separate feature scopes and separate artifacts
so that neither can be reported as the other.

What the plan guarantees
------------------------
* Windows are strictly increasing and non-overlapping, and
  ``train.end <= test.start``: no calendar instant is in both.
* Partitioning is by **calendar time**, never by an entity's own activity. The
  survivorship split that misled an earlier phase
  (``docs/temporal_collapse.md``) assigned entities by their last activity; this
  cannot, because an entity's membership follows only from the window it
  transacted in.
* Every window positive is scoreable: a flagged transaction inside the window
  makes both endpoints active in it, so ``n_positive_not_eligible`` is 0 by
  construction - and the plan refuses the run if it ever is not.
* Threshold selection, calibration and any tuning see only the training
  window's own validation partition. Test labels are read once, after the frozen
  model has scored, to compute metrics.
* The frozen model reaches the later window through
  ``CandidateRiskEngine`` - the production contract
  (``src/detection/contract.py``), so an R1 result is produced by the same path
  the product would use.

Split parameters (``SplitSpec.params``)
---------------------------------------
``windows``          list of ``[name, start, end]`` (or dicts with those keys).
``partition``        fractions of the training window, e.g.
                     ``{"fit": 0.6, "val": 0.2, "in_period": 0.2}``.
``max_window_end``   optional instant no window may pass (excludes a tail).
``min_positives``    refuse a window with fewer positive entities than this.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.detection.contract import (
    SCORE_CALIBRATED,
    SCORE_MODEL_PROBABILITY,
    SCORE_RANK_AGGREGATE,
    TASK_RETROSPECTIVE,
    AnalysisWindow,
    ModelIdentity,
)
from src.detection.engines import CandidateRiskEngine, TransactionWindow, feature_version_for
from src.experiments.artifacts import ExperimentResult, FoldOutcome, git_commit
from src.experiments.calibration import build_calibrator
from src.experiments.calendar import LedgerArrays, involved_entities, ledger_arrays
from src.experiments.candidate import Candidate, FitContext, build_candidate, checkpoint_identity
from src.experiments.dataset import ExperimentDataset
from src.experiments.drift import feature_psi, summarise_psi
from src.experiments.features import get_feature_set
from src.experiments.leakage import (
    LeakageFinding,
    audit_label_columns,
    audit_perfect_separators,
    raise_on_errors,
)
from src.experiments.metrics import aggregate_fold_metrics, binary_metrics
from src.experiments.seeding import (
    derive_seed,
    determinism_report,
    environment_report,
    set_global_seeds,
)
from src.experiments.snapshots import edge_dates, window_mask, window_snapshot
from src.experiments.spec import ExperimentSpec, SplitSpec
from src.experiments.splits import SplitFold
from src.experiments.thresholds import select_threshold

logger = logging.getLogger(__name__)

__all__ = [
    "DETECTION_TRANSFER_STRATEGY",
    "DETECTION_SCOPE",
    "DetectionWindow",
    "DetectionTransferPlan",
    "build_detection_transfer_plan",
    "resolve_detection_window",
    "run_detection_transfer_experiment",
]

#: Split-strategy name that routes a spec to this module.
DETECTION_TRANSFER_STRATEGY = "detection_transfer"
#: The only feature scope this strategy accepts.
DETECTION_SCOPE = "detection_window"

_PARAMS = frozenset({"windows", "partition", "max_window_end", "min_positives"})
_PARTITIONS: tuple[str, ...] = ("fit", "val", "in_period")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionWindow:
    """One completed detection period: who is in it, and who was flagged in it."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    eligible_idx: np.ndarray
    labels: np.ndarray
    n_transactions: int
    n_flagged_transactions: int
    n_positive_not_eligible: int

    @property
    def n_eligible(self) -> int:
        return int(len(self.eligible_idx))

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def days(self) -> float:
        return float((self.end - self.start) / pd.Timedelta(days=1))

    def positive_mask(self, n_entities: int) -> np.ndarray:
        """Boolean over the whole population: positive in *this* window."""
        mask = np.zeros(n_entities, dtype=bool)
        mask[self.eligible_idx[self.labels.astype(bool)]] = True
        return mask

    def eligible_mask(self, n_entities: int) -> np.ndarray:
        mask = np.zeros(n_entities, dtype=bool)
        mask[self.eligible_idx] = True
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": str(self.start),
            "end": str(self.end),
            "days": self.days,
            "n_transactions": self.n_transactions,
            "n_flagged_transactions": self.n_flagged_transactions,
            "n_eligible": self.n_eligible,
            "n_positive": self.n_positive,
            "prevalence": float(self.labels.mean()) if self.n_eligible else None,
            "n_positive_not_eligible": self.n_positive_not_eligible,
        }


def resolve_detection_window(
    ledger: LedgerArrays,
    name: str,
    start: Any,
    end: Any,
) -> DetectionWindow:
    """Eligible population and window labels for one detection window.

    The single definition of both rules: *eligible* = at least one transaction
    dated in the window; *positive* = payer or payee of at least one flagged
    transaction dated in the window.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not start < end:
        raise ValueError(f"detection window {name!r}: start {start} is not before end {end}")
    rows = window_mask(ledger.dates, start, end)
    flagged_rows = rows & ledger.flags
    eligible = involved_entities(ledger.n_entities, ledger.src, ledger.dst, rows)
    positive = involved_entities(ledger.n_entities, ledger.src, ledger.dst, flagged_rows)
    idx = np.flatnonzero(eligible).astype(np.int64)
    return DetectionWindow(
        name=name,
        start=start,
        end=end,
        eligible_idx=idx,
        labels=positive[idx].astype(np.int8),
        n_transactions=int(rows.sum()),
        n_flagged_transactions=int(flagged_rows.sum()),
        n_positive_not_eligible=int((positive & ~eligible).sum()),
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class DetectionTransferPlan:
    """Detection windows plus the chronological folds over them."""

    params: dict[str, Any]
    seed: int
    windows: list[DetectionWindow]
    folds: list[tuple[int, int]]
    partition: dict[str, float]
    notes: list[str] = field(default_factory=list)

    strategy: str = DETECTION_TRANSFER_STRATEGY
    #: The same entity may appear in both windows - that is the transfer
    #: question, and the overlap is measured and reported per fold.
    entity_disjoint: bool = False
    time_aware: bool = True
    graph_leakage_controlled: bool = False

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def regime(self) -> str:
        return "detection-transfer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "params": dict(self.params),
            "seed": self.seed,
            "regime": self.regime(),
            "entity_disjoint": self.entity_disjoint,
            "time_aware": self.time_aware,
            "graph_leakage_controlled": self.graph_leakage_controlled,
            "unit": "(entity, detection window)",
            "task": TASK_RETROSPECTIVE,
            "n_folds": self.n_folds,
            "partition": dict(self.partition),
            "notes": list(self.notes),
            "windows": [w.to_dict() for w in self.windows],
            "folds": [
                {
                    "fold_idx": i,
                    "train_window": self.windows[a].name,
                    "test_window": self.windows[b].name,
                    "train_span": [str(self.windows[a].start), str(self.windows[a].end)],
                    "test_span": [str(self.windows[b].start), str(self.windows[b].end)],
                    "n_train": self.windows[a].n_eligible,
                    "n_train_pos": self.windows[a].n_positive,
                    "n_test": self.windows[b].n_eligible,
                    "n_test_pos": self.windows[b].n_positive,
                }
                for i, (a, b) in enumerate(self.folds)
            ],
        }


def build_detection_transfer_plan(
    dataset: ExperimentDataset, split: SplitSpec
) -> DetectionTransferPlan:
    """Resolve the windows and the chronological folds, refusing unsound designs."""
    if split.strategy != DETECTION_TRANSFER_STRATEGY:
        raise ValueError(f"not a {DETECTION_TRANSFER_STRATEGY} split: {split.strategy!r}")
    params = dict(split.params)
    unknown = set(params) - _PARAMS
    if unknown:
        raise ValueError(
            f"unknown {DETECTION_TRANSFER_STRATEGY} parameter(s): {sorted(unknown)}"
        )
    raw = params.get("windows")
    if not raw:
        raise ValueError(f"{DETECTION_TRANSFER_STRATEGY} requires the parameter 'windows'")
    if len(raw) < 2:
        raise ValueError(
            f"{DETECTION_TRANSFER_STRATEGY} needs at least two windows to transfer between; "
            f"got {len(raw)}"
        )

    spans: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            name, start, end = item["name"], item["start"], item["end"]
        else:
            name, start, end = item
        spans.append((str(name), pd.Timestamp(start), pd.Timestamp(end)))

    for (name_a, _, end_a), (name_b, start_b, _) in zip(spans, spans[1:]):
        if end_a > start_b:
            raise ValueError(
                f"windows {name_a!r} and {name_b!r} overlap in calendar time "
                f"({end_a} > {start_b}): a cross-period result would train and test on "
                f"the same instants"
            )
    max_end = params.get("max_window_end")
    if max_end is not None:
        limit = pd.Timestamp(max_end)
        past = [n for n, _, e in spans if e > limit]
        if past:
            raise ValueError(f"windows {past} end after max_window_end {limit}")

    fractions = dict(params.get("partition") or {"fit": 0.6, "val": 0.2, "in_period": 0.2})
    if set(fractions) != set(_PARTITIONS):
        raise ValueError(f"partition must name exactly {list(_PARTITIONS)}; got {sorted(fractions)}")
    if any(v <= 0 for v in fractions.values()):
        raise ValueError("every partition fraction must be positive")
    total = sum(fractions.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"partition fractions must sum to 1; they sum to {total}")

    min_positives = int(params.get("min_positives", 1))
    ledger = ledger_arrays(dataset)
    last_observed = ledger.last_observed

    windows: list[DetectionWindow] = []
    notes = [
        "task is retrospective detection: for each window, features and labels "
        "both come from the transactions inside that window, and the model is "
        "asked which entities of a completed period to investigate.",
        "a fold fits on one window, freezes, and scores the next window - "
        "train.end <= test.start, so no calendar instant is in both.",
        "membership follows only from the window an entity transacted in, never "
        "from its own last activity: no survivorship selection.",
    ]
    durations = set()
    for name, start, end in spans:
        if start > last_observed:
            raise ValueError(
                f"window {name!r} starts at {start}, after the last observed transaction "
                f"({last_observed}); it is empty by construction"
            )
        window = resolve_detection_window(ledger, name, start, end)
        if window.n_eligible == 0:
            raise ValueError(f"window {name!r} holds no transactions")
        if window.n_positive_not_eligible:
            raise ValueError(
                f"window {name!r}: {window.n_positive_not_eligible} positive entities are not "
                f"eligible, which contradicts the eligibility rule - a flagged transaction "
                f"inside the window makes both endpoints active in it"
            )
        if window.n_positive < min_positives:
            raise ValueError(
                f"window {name!r} has {window.n_positive} positive entities, fewer than the "
                f"required {min_positives}: too few to estimate detection performance"
            )
        if window.end > last_observed:
            notes.append(
                f"window {name!r} ends at {window.end}, after the last observed transaction "
                f"({last_observed})."
            )
        durations.add(round(window.days, 6))
        windows.append(window)

    if len(durations) > 1:
        notes.append(
            f"windows are not equal length ({sorted(durations)} days); volume and prevalence "
            f"differences between folds are then partly a length effect."
        )
    folds = [(i, i + 1) for i in range(len(windows) - 1)]
    return DetectionTransferPlan(
        params=params,
        seed=split.seed,
        windows=windows,
        folds=folds,
        partition=fractions,
        notes=notes,
    )


def _stratified_partition(
    labels: np.ndarray, fractions: dict[str, float], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Deal positions into partitions, keeping each class's proportions.

    Positives and negatives are dealt separately so a partition cannot end up
    without positives while another holds them all.
    """
    names = list(_PARTITIONS)
    out: dict[str, list[int]] = {name: [] for name in names}
    for cls in (1, 0):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        if not len(idx):
            continue
        counts = [int(np.floor(fractions[name] * len(idx))) for name in names]
        # Give the remainder to the largest partition first, so rounding never
        # empties a small one.
        remainder = len(idx) - sum(counts)
        order = np.argsort([-fractions[name] for name in names])
        for j in range(remainder):
            counts[order[j % len(names)]] += 1
        start = 0
        for name, count in zip(names, counts):
            out[name].extend(idx[start : start + count].tolist())
            start += count
    return {name: np.sort(np.array(values, dtype=np.int64)) for name, values in out.items()}


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _audit_plan(plan: DetectionTransferPlan, n_entities: int) -> list[LeakageFinding]:
    findings = [
        LeakageFinding(
            check="evaluation_regime",
            severity="info",
            message=(
                "split 'detection_transfer' is detection-transfer: each window's "
                "features and labels come from that window, and a fold applies a "
                "frozen model to a strictly later window"
            ),
        )
    ]
    for i, (a, b) in enumerate(plan.folds):
        train, test = plan.windows[a], plan.windows[b]
        ok = train.end <= test.start
        findings.append(
            LeakageFinding(
                check="calendar_separation",
                severity="info" if ok else "error",
                message=(
                    f"fold {i}: train window {train.name} ends {train.end}, "
                    + ("at or before" if ok else "AFTER")
                    + f" the start of test window {test.name} ({test.start})"
                ),
            )
        )
        seen = train.eligible_mask(n_entities) & test.eligible_mask(n_entities)
        findings.append(
            LeakageFinding(
                check="entity_overlap",
                severity="info",
                message=(
                    f"fold {i}: {int(seen.sum())} of {test.n_eligible} test entities also "
                    f"transacted in the training window; reported as strata, not removed"
                ),
            )
        )
    return findings


def _audit_window_boundary(window: DetectionWindow, edges: pd.DataFrame) -> list[LeakageFinding]:
    """Check the rows actually handed to the feature families."""
    if edges.empty:
        return [
            LeakageFinding(
                check="detection_window_boundary",
                severity="error",
                message=f"{window.name}: the feature window holds no transactions",
            )
        ]
    dates = pd.to_datetime(edges["date"], errors="coerce")
    problems = []
    if dates.isna().any():
        problems.append("undated rows reached the feature window")
    if dates.min() < window.start:
        problems.append(f"earliest feature transaction {dates.min()} precedes the window start")
    if dates.max() >= window.end:
        problems.append(f"latest feature transaction {dates.max()} is not before the window end")
    if problems:
        return [
            LeakageFinding(
                check="detection_window_boundary",
                severity="error",
                message=f"{window.name}: " + "; ".join(problems),
            )
        ]
    return [
        LeakageFinding(
            check="detection_window_boundary",
            severity="info",
            message=(
                f"{window.name}: {len(edges)} feature transactions dated {dates.min()} .. "
                f"{dates.max()}, all inside [{window.start}, {window.end})"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_spec(spec: ExperimentSpec) -> None:
    if spec.features.scope != DETECTION_SCOPE:
        raise ValueError(
            f"{DETECTION_TRANSFER_STRATEGY} computes each window's features from the "
            f"transactions inside that window, so FeatureSpec.scope must be "
            f"{DETECTION_SCOPE!r}; got {spec.features.scope!r}."
        )
    if not spec.features.sets:
        raise ValueError(f"{DETECTION_TRANSFER_STRATEGY} requires at least one feature set")
    for name in spec.features.sets:
        fs = get_feature_set(name)
        if fs.requires_graph:
            raise ValueError(
                f"feature set {name!r} requires a networkx graph; "
                f"{DETECTION_TRANSFER_STRATEGY} builds no per-window graph snapshot"
            )
        if not fs.time_scopable:
            raise ValueError(
                f"feature set {name!r} does not honour a time window and could summarise "
                f"behaviour outside the detection window"
            )


def _score_kind(candidate: Candidate, calibrated: bool) -> str:
    if calibrated:
        return SCORE_CALIBRATED
    return SCORE_MODEL_PROBABILITY if candidate.emits_probabilities else SCORE_RANK_AGGREGATE


def _metrics(y: np.ndarray, scores: np.ndarray, spec: ExperimentSpec, threshold, probability: bool):
    return binary_metrics(
        y,
        scores,
        threshold=threshold,
        k_values=spec.evaluation.k_values,
        alert_rates=spec.evaluation.alert_rates,
        probability_scale=probability,
        calibration_bins=spec.evaluation.calibration_bins,
    )


def _retention(cross: dict[str, Any], in_period: dict[str, Any]) -> dict[str, Any]:
    """Cross-period value / in-period value, where both are defined."""
    def ratio(a, b):
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)

    rate = "0.01"
    return {
        "average_precision": ratio(cross.get("average_precision"), in_period.get("average_precision")),
        "ap_lift": ratio(cross.get("ap_lift"), in_period.get("ap_lift")),
        "roc_auc": ratio(cross.get("roc_auc"), in_period.get("roc_auc")),
        "precision_at_1pct": ratio(
            (cross.get("precision_at_rate") or {}).get(rate),
            (in_period.get("precision_at_rate") or {}).get(rate),
        ),
        "recall_at_1pct": ratio(
            (cross.get("recall_at_rate") or {}).get(rate),
            (in_period.get("recall_at_rate") or {}).get(rate),
        ),
    }


def run_detection_transfer_experiment(
    spec: ExperimentSpec,
    dataset: ExperimentDataset,
    *,
    candidate_factory: Optional[Callable[[], Candidate]] = None,
    output_dir: Optional[Path | str] = None,
    checkpoint_dir: Optional[Path | str] = None,
) -> ExperimentResult:
    """Fit on each window, freeze, and score the next one through the contract."""
    started = _now()
    t_start = time.perf_counter()
    _check_spec(spec)

    if candidate_factory is None:
        def candidate_factory() -> Candidate:  # type: ignore[misc]
            return build_candidate(spec.model.family, spec.model.params)

    probe = candidate_factory()
    if not getattr(probe, "calendar_compatible", False):
        raise ValueError(
            f"candidate family {probe.family!r} cannot be scored outside its fit context "
            f"(it is not declared calendar_compatible), so it cannot be frozen and applied "
            f"to a later window"
        )
    del probe

    seed_report = set_global_seeds(spec.seed)
    plan = build_detection_transfer_plan(dataset, spec.split)
    n = dataset.n_entities
    findings: list[LeakageFinding] = _audit_plan(plan, n)
    raise_on_errors(findings)

    notes = list(plan.notes)
    if spec.dataset.synthetic:
        notes.append(
            "SYNTHETIC DATASET: this measures detection transfer inside a simulator and is "
            "not evidence about real-world temporal generalisation."
        )

    dates = edge_dates(dataset)
    entity_ids = list(dataset.entity_ids)
    id_index = pd.Index(entity_ids)
    outcomes: list[FoldOutcome] = []
    feature_summary: dict[str, Any] = {}

    for fold_idx, (a, b) in enumerate(plan.folds):
        train, test = plan.windows[a], plan.windows[b]
        fold_seed = derive_seed(spec.seed, "fold", fold_idx)
        set_global_seeds(fold_seed)

        # --- the training window -------------------------------------------
        t_feat = time.perf_counter()
        frame, provenance, reused, edges = window_snapshot(
            dataset, spec.features.sets, start=train.start, end=train.end, as_of=train.end, dates=dates
        )
        fold_findings = _audit_window_boundary(train, edges)
        fold_findings.extend(audit_label_columns(frame))
        rows = frame.take(train.eligible_idx, axis=0)
        rows.index = pd.Index([entity_ids[i] for i in train.eligible_idx], name="entity_id")
        feat_seconds = time.perf_counter() - t_feat

        parts = _stratified_partition(
            train.labels,
            plan.partition,
            np.random.default_rng(derive_seed(plan.seed, "partition", fold_idx)),
        )
        y = train.labels.astype(np.int8)
        fold_findings.extend(audit_perfect_separators(rows, y[parts["fit"]], rows=parts["fit"]))
        raise_on_errors(fold_findings)

        ctx = FitContext(
            dataset=dataset,
            features=rows,
            fold=SplitFold(
                fold_idx=fold_idx,
                train_idx=train.eligible_idx[parts["fit"]],
                val_idx=train.eligible_idx[parts["val"]],
                test_idx=test.eligible_idx,
                train_window_end=train.end,
                val_window_end=train.end,
            ),
            train_idx=parts["fit"],
            val_idx=parts["val"],
            y_train=y[parts["fit"]],
            y_val=y[parts["val"]],
            train_edges=edges,
            seed=fold_seed,
            imbalance=spec.model.imbalance,
        )
        candidate = candidate_factory()
        t_fit = time.perf_counter()
        candidate.fit(ctx)
        fit_seconds = time.perf_counter() - t_fit

        # --- validation: the only partition allowed to shape decisions ------
        t_score = time.perf_counter()
        val_raw = candidate.predict_scores(ctx, parts["val"])
        calibrator, cal_notes = build_calibrator(
            spec.calibration, y[parts["val"]], val_raw,
            model_emits_probabilities=candidate.emits_probabilities,
        )
        val_scores = calibrator.transform(val_raw) if len(val_raw) else val_raw
        decision = select_threshold(spec.threshold, y[parts["val"]], val_scores)
        if calibrator.produces_probabilities and spec.calibration.method != "none":
            raise ValueError(
                "detection_transfer applies the frozen model through CandidateRiskEngine, "
                "which scores the model directly; a fitted calibrator would not travel with "
                "it. Configure calibration.method='none' - the score kind is then recorded "
                "as a model probability rather than a calibrated one."
            )

        # --- in-period reference: same model, same window, held-out rows ----
        in_raw = candidate.predict_scores(ctx, parts["in_period"])
        probability = candidate.emits_probabilities
        in_metrics = _metrics(y[parts["in_period"]], in_raw, spec, decision.value, probability)
        val_metrics = _metrics(y[parts["val"]], val_scores, spec, decision.value, probability)

        # --- FREEZE, then the later window through the production contract --
        identity = ModelIdentity(
            model_id=f"{spec.name}",
            model_version=(
                f"{spec.model.family}@{train.name}"
                f"[{train.start.date()}..{train.end.date()}]:seed{fold_seed}"
            ),
            family=spec.model.family,
            feature_version=feature_version_for(spec.features.sets),
            task=TASK_RETROSPECTIVE,
            score_kind=_score_kind(candidate, calibrated=False),
            evaluation_reference="docs/r1_detection_transfer.md",
            evaluation_regime=(
                f"IBM HI-Small (synthetic): fitted on completed window {train.name} "
                f"[{train.start}, {train.end}) and applied unchanged to a later window"
            ),
        )
        engine = CandidateRiskEngine(
            candidate,
            spec.features.sets,
            identity,
            threshold=decision.value,
            threshold_basis=(
                f"{spec.threshold.strategy} {spec.threshold.params} on the {train.name} "
                f"validation partition"
            ),
        )
        assessment = engine.score(
            TransactionWindow(
                entity_ids=entity_ids,
                edge_table=dataset.edge_table,
                window=AnalysisWindow(start=test.start, end=test.end),
            )
        )
        score_seconds = time.perf_counter() - t_score

        scored_ids = assessment.scores["entity_id"].tolist()
        expected = {entity_ids[i] for i in test.eligible_idx}
        if set(scored_ids) != expected:
            raise ValueError(
                f"fold {fold_idx}: the engine scored {len(scored_ids)} entities but the test "
                f"window has {len(expected)} eligible; the window rule and the engine "
                f"disagree"
            )

        # Labels are read only now, after scoring, to evaluate.
        test_label = pd.Series(
            test.labels.astype(np.int8), index=pd.Index([entity_ids[i] for i in test.eligible_idx])
        )
        y_test = test_label.reindex(scored_ids).to_numpy(dtype=np.int8)
        scores = assessment.scores["risk_score"].to_numpy(dtype=np.float64)
        test_metrics = _metrics(y_test, scores, spec, decision.value, probability)

        # --- strata ---------------------------------------------------------
        positions = id_index.get_indexer(scored_ids)
        train_elig = train.eligible_mask(n)[positions]
        earlier_pos = np.zeros(n, dtype=bool)
        for window in plan.windows[: b]:
            earlier_pos |= window.positive_mask(n)
        first_time = (~earlier_pos[positions]) & (y_test == 1)
        strata = {
            "seen_entities": train_elig,
            "new_entities": ~train_elig,
            # positives here are those never flagged in any earlier window; the
            # negatives are the rest of the population, so this reads as "can it
            # find first-time involvement".
            "first_time_positives": first_time | (y_test == 0),
        }
        subgroup: dict[str, Any] = {
            "in_period_reference": {
                "definition": (
                    "held-out entities of the training window, scored by the same frozen "
                    "model on the window it was fitted on - the same-period yardstick"
                ),
                "n": int(len(parts["in_period"])),
                "n_positive": int(y[parts["in_period"]].sum()),
                "metrics": in_metrics,
            }
        }
        for name, mask in strata.items():
            subgroup[name] = {
                "definition": {
                    "seen_entities": "test entities that also transacted in the training window",
                    "new_entities": "test entities absent from the training window",
                    "first_time_positives": (
                        "test population whose positives were never flagged in an earlier window"
                    ),
                }[name],
                "n": int(mask.sum()),
                "n_positive": int(y_test[mask].sum()),
                "metrics": _metrics(y_test[mask], scores[mask], spec, decision.value, probability),
            }

        # --- drift ----------------------------------------------------------
        drift_table = feature_psi(rows.take(parts["fit"], axis=0), assessment.evidence)
        diagnostics = {
            "windows": {"train": train.to_dict(), "test": test.to_dict()},
            "drift_psi": summarise_psi(drift_table),
            "retention_vs_in_period": _retention(test_metrics, in_metrics),
            "risk_assessment": assessment.metadata(),
            "snapshot_reuse": reused,
        }

        checkpoint: Optional[dict[str, Any]] = None
        if checkpoint_dir is not None:
            directory = Path(checkpoint_dir)
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_identity(
                candidate.save(directory / f"{spec.name}_fold{fold_idx}")
            )

        counts = {
            "n_train_window": train.n_eligible,
            "n_train_window_pos": train.n_positive,
            "n_fit": int(len(parts["fit"])),
            "n_fit_pos": int(y[parts["fit"]].sum()),
            "n_val": int(len(parts["val"])),
            "n_val_pos": int(y[parts["val"]].sum()),
            "n_in_period": int(len(parts["in_period"])),
            "n_in_period_pos": int(y[parts["in_period"]].sum()),
            "n_test": int(len(y_test)),
            "n_test_pos": int(y_test.sum()),
            "n_test_seen": int(train_elig.sum()),
            "n_test_new": int((~train_elig).sum()),
            "n_test_first_time_pos": int(first_time.sum()),
        }
        if not feature_summary:
            feature_summary = {
                "scope": DETECTION_SCOPE,
                "sets": provenance,
                "n_columns": int(rows.shape[1]),
                "as_of": "each window's own end",
                "notes": [
                    "features are computed from the transactions inside the detection "
                    "window they are scored for; no history before it is used."
                ],
            }

        outcomes.append(
            FoldOutcome(
                fold_idx=fold_idx,
                counts=counts,
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
                subgroup_test_metrics=subgroup,
                diagnostics=diagnostics,
            )
        )
        findings.extend(fold_findings)
        logger.info(
            "fold %d (%s -> %s): cross-period AP=%s, in-period AP=%s",
            fold_idx,
            train.name,
            test.name,
            test_metrics.get("average_precision"),
            in_metrics.get("average_precision"),
        )

    result = ExperimentResult(
        spec=spec,
        dataset_summary=dataset.summary(),
        split_summary=plan.to_dict(),
        feature_summary=feature_summary,
        folds=outcomes,
        aggregate=aggregate_fold_metrics([o.test_metrics for o in outcomes]),
        validation_aggregate=aggregate_fold_metrics([o.validation_metrics for o in outcomes]),
        environment=environment_report() | {"seeding": seed_report},
        determinism=determinism_report(),
        leakage=[f.to_dict() for f in findings],
        notes=notes,
        started_at=started,
        finished_at=_now(),
        total_runtime_seconds=time.perf_counter() - t_start,
        git_commit=git_commit(Path(__file__).resolve().parents[2]),
    )
    if output_dir is not None:
        from src.experiments.artifacts import save_result

        save_result(result, output_dir)
    return result
