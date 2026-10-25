"""
Layer 6 - Calendar-Forward Evaluation
======================================

A time-forward benchmark in which train and test differ in *time*, not in *who
is in them*.

Why a separate evaluation path
------------------------------
Every strategy in :mod:`src.experiments.splits` partitions **entities**: an
entity is in train, validation or test, and it carries one label computed over
the whole file.  ``time_forward_holdout`` places entities by their own last
activity, which on IBM HI-Small turned it into a survivorship split - test
entities are by construction the ones still transacting at the end (train
median 3 transactions, test median 37).  ``docs/temporal_collapse.md`` measured
that.

A calendar cutoff cannot be expressed as an entity partition, because the unit
of prediction changes.  Here a row is an ``(entity, origin)`` pair:

``origin``
    A fixed calendar instant ``T``, the same for every entity.
features
    Computed only from transactions dated strictly before ``T`` (and at or
    after ``T - lookback`` when a lookback is configured).  Recency and tenure
    are measured from ``T``.
label
    1 when the entity is the payer or the payee of at least one transaction
    that the publisher flagged, dated in ``[T, T + horizon)``.  This is the
    project's existing entity label - involvement in a flagged transaction -
    restricted to a prediction window instead of the whole file.
eligibility
    The entity has at least one dated transaction in its feature window.  An
    entity with no history before ``T`` cannot be scored from history; it is
    excluded and counted (``n_cold_start_positive``), never given a score.

Three origins, one per role: the model is fitted at the train origin,
calibrated and thresholded at the validation origin, and scored once at the
test origin.  The same account may appear at all three - with a different
feature snapshot and a different label each time.  What must not overlap is
*time*, and the plan enforces it:

* ``origin_train + horizon <= origin_val`` and
  ``origin_val + horizon <= origin_test``: every label used to fit or calibrate
  the model has been fully observed before the snapshot it is applied to.
* every feature row is dated before its own origin; the audit checks the rows
  actually handed to the feature families, not the intention.
* an optional ``max_label_end`` bounds the last label window, so a period in
  which the label process is known to differ can be excluded by declaration
  rather than by accident.

Whole-file entity labels (``ExperimentDataset.labels``) are never used here:
they include the future.  The window label is derived from
``ExperimentDataset.transaction_labels``, which sits beside the edge table and
is not visible to feature families.

Split parameters (``SplitSpec.params``)
---------------------------------------
``origins``         three ISO timestamps: train, validation, test.
``horizon_days``    length of each label window, in days.
``lookback_days``   optional; features see only ``[T - lookback, T)``.
                    ``null`` means the whole history before ``T``.
``max_label_end``   optional ISO timestamp the test label window may not pass.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.experiments.artifacts import ExperimentResult, FoldOutcome, git_commit
from src.experiments.calibration import build_calibrator
from src.experiments.candidate import Candidate, FitContext, build_candidate, checkpoint_identity
from src.experiments.dataset import ExperimentDataset
from src.experiments.features import FeatureRequest, build_feature_matrix, get_feature_set
from src.experiments.leakage import (
    LeakageFinding,
    audit_label_columns,
    audit_perfect_separators,
    raise_on_errors,
)
from src.experiments.metrics import aggregate_fold_metrics, binary_metrics
from src.experiments.seeding import derive_seed, determinism_report, environment_report, set_global_seeds
from src.experiments.snapshots import clear_snapshot_cache, edge_dates, window_mask, window_snapshot
from src.experiments.sources import entity_codes
from src.experiments.spec import ExperimentSpec, SplitSpec
from src.experiments.splits import SplitFold
from src.experiments.thresholds import select_threshold

logger = logging.getLogger(__name__)

__all__ = [
    "CALENDAR_STRATEGY",
    "CALENDAR_SCOPE",
    "CalendarOrigin",
    "CalendarPlan",
    "LedgerArrays",
    "build_calendar_plan",
    "involved_entities",
    "ledger_arrays",
    "resolve_origin",
    "run_calendar_experiment",
    "snapshot_features",
    "clear_snapshot_cache",
]

#: Split-strategy name that routes a spec to this module.
CALENDAR_STRATEGY = "calendar_forward"
#: The only feature scope this strategy accepts.
CALENDAR_SCOPE = "origin_window"
#: Role order.  Origins must be strictly increasing in this order.
ROLES: tuple[str, ...] = ("train", "val", "test")

_PARAMS = frozenset({"origins", "horizon_days", "lookback_days", "max_label_end"})


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarOrigin:
    """One scored snapshot: who is scored at ``origin``, from what, against what.

    Attributes
    ----------
    role:
        ``"train"``, ``"val"`` or ``"test"``.
    origin:
        Feature cutoff (exclusive) and label-window start (inclusive).
    feature_start:
        Inclusive start of the feature window, or ``None`` for all history.
    label_end:
        Exclusive end of the label window.
    eligible_idx:
        Positional entity indices scored at this origin.
    labels:
        ``int8`` window labels aligned to ``eligible_idx``.
    prior_involvement:
        ``bool`` aligned to ``eligible_idx``: the entity was already on a
        flagged transaction before ``origin``.  Used only to report the
        new-involvement subgroup.  Never a feature.
    n_window_positive:
        Entities involved in the window, eligible or not.
    n_cold_start_positive:
        Window positives excluded because they have no feature-window history.
    n_feature_transactions:
        Transactions inside the feature window.
    """

    role: str
    origin: pd.Timestamp
    feature_start: Optional[pd.Timestamp]
    label_end: pd.Timestamp
    eligible_idx: np.ndarray
    labels: np.ndarray
    prior_involvement: np.ndarray
    n_window_positive: int
    n_cold_start_positive: int
    n_feature_transactions: int

    @property
    def n_eligible(self) -> int:
        return int(len(self.eligible_idx))

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())

    def to_dict(self) -> dict[str, Any]:
        new = ~self.prior_involvement
        return {
            "role": self.role,
            "origin": str(self.origin),
            "feature_window": [
                None if self.feature_start is None else str(self.feature_start),
                str(self.origin),
            ],
            "label_window": [str(self.origin), str(self.label_end)],
            "n_eligible": self.n_eligible,
            "n_positive": self.n_positive,
            "prevalence": float(self.labels.mean()) if self.n_eligible else None,
            "n_prior_involvement": int(self.prior_involvement.sum()),
            "n_positive_with_prior_involvement": int(self.labels[self.prior_involvement].sum()),
            "n_positive_new_involvement": int(self.labels[new].sum()),
            "n_window_positive": self.n_window_positive,
            "n_cold_start_positive": self.n_cold_start_positive,
            "n_feature_transactions": self.n_feature_transactions,
        }


@dataclass
class CalendarPlan:
    """Three origins plus the methodological declaration of the regime."""

    params: dict[str, Any]
    seed: int
    origins: dict[str, CalendarOrigin]
    horizon: pd.Timedelta
    lookback: Optional[pd.Timedelta]
    notes: list[str] = field(default_factory=list)

    strategy: str = CALENDAR_STRATEGY
    #: Rows are (entity, origin) samples.  The same entity recurs across roles.
    entity_disjoint: bool = False
    time_aware: bool = True
    graph_leakage_controlled: bool = False

    @property
    def n_folds(self) -> int:
        return 1

    def regime(self) -> str:
        return "calendar-forward"

    def to_dict(self) -> dict[str, Any]:
        tr, va, te = (self.origins[r] for r in ROLES)
        return {
            "strategy": self.strategy,
            "params": dict(self.params),
            "seed": self.seed,
            "regime": self.regime(),
            "entity_disjoint": self.entity_disjoint,
            "time_aware": self.time_aware,
            "graph_leakage_controlled": self.graph_leakage_controlled,
            "unit": "(entity, origin)",
            "n_folds": self.n_folds,
            "notes": list(self.notes),
            "origins": {r: self.origins[r].to_dict() for r in ROLES},
            "folds": [
                {
                    "fold_idx": 0,
                    "n_train": tr.n_eligible,
                    "n_train_pos": tr.n_positive,
                    "n_val": va.n_eligible,
                    "n_val_pos": va.n_positive,
                    "n_test": te.n_eligible,
                    "n_test_pos": te.n_positive,
                    "train_window_end": str(tr.origin),
                    "val_window_end": str(va.origin),
                    "test_window_end": str(te.origin),
                }
            ],
        }


def _involved(n: int, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Entities that are payer or payee on at least one row of ``mask``."""
    out = np.zeros(n, dtype=bool)
    out[src[mask]] = True
    out[dst[mask]] = True
    return out


#: Public name for the rule above.  ``src.experiments.detection_transfer``
#: resolves its detection windows with exactly this definition of "took part",
#: and the two regimes must not drift apart on it.
involved_entities = _involved


def _edge_dates(dataset: ExperimentDataset) -> np.ndarray:
    return edge_dates(dataset)


def _feature_mask(dates: np.ndarray, origin: CalendarOrigin) -> np.ndarray:
    """Rows a snapshot may summarise.  Undated rows are never included: an
    undated transaction cannot be proven to precede the origin."""
    return window_mask(dates, origin.feature_start, origin.origin)


@dataclass(frozen=True)
class LedgerArrays:
    """The per-transaction arrays every origin is resolved from.

    ``dates`` (``datetime64``, NaT where undated), ``src`` / ``dst`` (positional
    entity indices) and ``flags`` (the publisher's transaction flag), all
    row-aligned to the edge table.  Built once per dataset and reused across
    origins.
    """

    dates: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    flags: np.ndarray
    n_entities: int

    @property
    def last_observed(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[~np.isnat(self.dates)].max())


def ledger_arrays(dataset: ExperimentDataset) -> LedgerArrays:
    """Arrays for :func:`resolve_origin`; refuses a dataset no window label fits."""
    if dataset.transaction_labels is None:
        raise ValueError(
            f"{CALENDAR_STRATEGY} derives each label from the transactions flagged "
            f"inside a time window, and this dataset carries no transaction-level "
            f"flags (ExperimentDataset.transaction_labels). A whole-file entity "
            f"label cannot be windowed after the fact."
        )
    if not dataset.has_time():
        raise ValueError(f"{CALENDAR_STRATEGY} requires a dated edge table")
    src, dst = entity_codes(dataset, dataset.entity_ids)
    return LedgerArrays(
        dates=_edge_dates(dataset),
        src=np.asarray(src),
        dst=np.asarray(dst),
        flags=dataset.transaction_labels,
        n_entities=dataset.n_entities,
    )


def resolve_origin(
    ledger: LedgerArrays,
    role: str,
    origin: pd.Timestamp,
    horizon: pd.Timedelta,
    lookback: Optional[pd.Timedelta] = None,
) -> CalendarOrigin:
    """Eligible population, window labels and prior involvement at one origin.

    The single definition of the label rule and the eligibility rule; the plan
    builder and ``calendar_design_audit.py`` both call it.
    """
    origin = pd.Timestamp(origin)
    start = None if lookback is None else origin - lookback
    end = origin + horizon
    empty = CalendarOrigin(
        role, origin, start, end,
        np.array([], dtype=np.int64), np.array([], dtype=np.int8),
        np.array([], dtype=bool), 0, 0, 0,
    )
    dates, n = ledger.dates, ledger.n_entities
    dated = ~np.isnat(dates)
    feature_rows = _feature_mask(dates, empty)
    eligible = _involved(n, ledger.src, ledger.dst, feature_rows)
    window_rows = dated & (dates >= np.datetime64(origin)) & (dates < np.datetime64(end))
    positive = _involved(n, ledger.src, ledger.dst, window_rows & ledger.flags)
    prior = _involved(n, ledger.src, ledger.dst, dated & (dates < np.datetime64(origin)) & ledger.flags)
    idx = np.flatnonzero(eligible).astype(np.int64)
    return dataclasses.replace(
        empty,
        eligible_idx=idx,
        labels=positive[idx].astype(np.int8),
        prior_involvement=prior[idx],
        n_window_positive=int(positive.sum()),
        n_cold_start_positive=int((positive & ~eligible).sum()),
        n_feature_transactions=int(feature_rows.sum()),
    )


def build_calendar_plan(dataset: ExperimentDataset, split: SplitSpec) -> CalendarPlan:
    """Resolve the three origins, their eligible populations and window labels.

    Raises ``ValueError`` for any configuration under which a label used to fit
    or calibrate could post-date the snapshot it is applied to, or under which
    no label can be derived at all.
    """
    if split.strategy != CALENDAR_STRATEGY:
        raise ValueError(f"not a {CALENDAR_STRATEGY} split: {split.strategy!r}")
    params = dict(split.params)
    unknown = set(params) - _PARAMS
    if unknown:
        raise ValueError(f"unknown {CALENDAR_STRATEGY} parameter(s): {sorted(unknown)}")
    for required in ("origins", "horizon_days"):
        if required not in params:
            raise ValueError(f"{CALENDAR_STRATEGY} requires the parameter {required!r}")

    origins = [pd.Timestamp(o) for o in params["origins"]]
    if len(origins) != len(ROLES):
        raise ValueError(
            f"{CALENDAR_STRATEGY} takes exactly {len(ROLES)} origins "
            f"(train, validation, test); got {len(origins)}"
        )
    if any(later <= earlier for earlier, later in zip(origins, origins[1:])):
        raise ValueError(f"origins must be strictly increasing; got {[str(o) for o in origins]}")

    horizon = pd.Timedelta(days=float(params["horizon_days"]))
    if horizon <= pd.Timedelta(0):
        raise ValueError("horizon_days must be positive")
    lookback: Optional[pd.Timedelta] = None
    if params.get("lookback_days") is not None:
        lookback = pd.Timedelta(days=float(params["lookback_days"]))
        if lookback <= pd.Timedelta(0):
            raise ValueError("lookback_days must be positive, or null for all history")

    for role, earlier, later in zip(ROLES, origins, origins[1:]):
        if earlier + horizon > later:
            raise ValueError(
                f"the {role} label window [{earlier}, {earlier + horizon}) is not over "
                f"before the next origin {later}: a label used to fit or calibrate "
                f"the model would not yet be observable when the later snapshot is "
                f"scored"
            )

    test_end = origins[-1] + horizon
    max_label_end = params.get("max_label_end")
    if max_label_end is not None and test_end > pd.Timestamp(max_label_end):
        raise ValueError(
            f"the test label window ends at {test_end}, after max_label_end "
            f"{pd.Timestamp(max_label_end)}"
        )

    ledger = ledger_arrays(dataset)
    dated = ~np.isnat(ledger.dates)
    last_observed = ledger.last_observed

    notes = [
        "unit of prediction is the (entity, origin) pair: every entity with "
        "feature-window history is scored at each origin, so the same account "
        "recurs across train, validation and test with a different snapshot and "
        "label; the partitions are disjoint in time, not in entity.",
        f"label = the entity is payer or payee of at least one transaction flagged "
        f"by the publisher and dated in [origin, origin + {horizon}).",
        "eligibility = at least one dated transaction in the feature window; "
        "window positives without one are excluded and counted, never scored.",
        "features = transactions dated strictly before the origin"
        + (f" and at or after origin - {lookback}" if lookback is not None else "")
        + "; recency and tenure are measured from the origin.",
    ]
    n_undated = int((~dated).sum())
    if n_undated:
        notes.append(
            f"{n_undated} undated transactions contribute to no feature window and "
            f"no label window."
        )

    resolved: dict[str, CalendarOrigin] = {}
    for role, origin in zip(ROLES, origins):
        end = origin + horizon
        if origin > last_observed:
            raise ValueError(
                f"the {role} origin {origin} is after the last observed transaction "
                f"({last_observed}); its label window is empty by construction"
            )
        if end > last_observed:
            notes.append(
                f"the {role} label window ends at {end}, after the last observed "
                f"transaction ({last_observed}); involvement after that is unobserved."
            )
        resolved[role] = resolve_origin(ledger, role, origin, horizon, lookback)
        if resolved[role].n_eligible == 0:
            raise ValueError(f"no entity has a transaction in the {role} feature window")
        cold = resolved[role].n_cold_start_positive
        if cold:
            notes.append(
                f"{role}: {cold} of {resolved[role].n_window_positive} window positives have no "
                f"feature-window history and are excluded."
            )
        if resolved[role].n_positive == 0:
            notes.append(
                f"{role}: no eligible positives; label-dependent metrics are "
                f"undefined there."
            )

    return CalendarPlan(
        params=params,
        seed=split.seed,
        origins=resolved,
        horizon=horizon,
        lookback=lookback,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Snapshot features
# ---------------------------------------------------------------------------
#
# The snapshot itself, and the cache that makes a batch of configs over the same
# windows affordable, live in :mod:`src.experiments.snapshots` - shared with the
# detection-transfer regime, which needs exactly the same thing with a different
# label rule.


def _snapshot(
    dataset: ExperimentDataset,
    sets: tuple[str, ...],
    origin: CalendarOrigin,
    dates: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], pd.DataFrame]:
    """Feature matrix over every entity at ``origin``, plus the rows it used.

    Returns ``(frame, provenance, reused_sets, feature_edges)``.
    """
    return window_snapshot(
        dataset,
        sets,
        start=origin.feature_start,
        end=origin.origin,
        as_of=origin.origin,
        dates=dates,
    )


def snapshot_features(
    dataset: ExperimentDataset, sets: tuple[str, ...] | list[str], origin: CalendarOrigin
) -> pd.DataFrame:
    """The feature matrix a calendar run would compute at ``origin``.

    One row per entity of ``dataset`` (eligible or not), computed only from the
    transactions in that origin's feature window.
    """
    frame, _, _, _ = _snapshot(dataset, tuple(sets), origin, _edge_dates(dataset))
    return frame


def _rows_for(frame: pd.DataFrame, origin: CalendarOrigin, entity_ids: list[str]) -> pd.DataFrame:
    """The eligible rows of a snapshot, indexed ``"<role>|<entity>"``.

    The role prefix keeps the index unique when two snapshots are stacked for
    fitting - the same entity legitimately appears in both.
    """
    rows = frame.take(origin.eligible_idx, axis=0)
    rows.index = pd.Index(
        [f"{origin.role}|{entity_ids[i]}" for i in origin.eligible_idx], name="sample_id"
    )
    return rows


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _audit_plan(plan: CalendarPlan) -> list[LeakageFinding]:
    findings = [
        LeakageFinding(
            check="evaluation_regime",
            severity="info",
            message=(
                "split 'calendar_forward' is calendar-forward: rows are (entity, "
                "origin) samples scored at fixed calendar origins; partitions are "
                "disjoint in time, not in entity"
            ),
        )
    ]
    for role, later_role in zip(ROLES, ROLES[1:]):
        a, b = plan.origins[role], plan.origins[later_role]
        ok = a.label_end <= b.origin
        findings.append(
            LeakageFinding(
                check="calendar_label_ordering",
                severity="info" if ok else "error",
                message=(
                    f"{role} label window ends {a.label_end}, "
                    + ("at or before" if ok else "AFTER")
                    + f" the {later_role} origin {b.origin}"
                ),
            )
        )
    train_ids = set(plan.origins["train"].eligible_idx.tolist())
    test_idx = plan.origins["test"].eligible_idx
    shared = int(sum(1 for i in test_idx.tolist() if i in train_ids))
    findings.append(
        LeakageFinding(
            check="entity_overlap",
            severity="info",
            message=(
                f"{shared} of {len(test_idx)} test entities were also scored at the "
                f"train origin; expected under this regime, where the label and the "
                f"feature snapshot, not the entity, are what is held out"
            ),
        )
    )
    return findings


def _audit_boundary(origin: CalendarOrigin, edges: pd.DataFrame) -> list[LeakageFinding]:
    """Check the rows actually handed to the feature families."""
    if edges.empty:
        return [
            LeakageFinding(
                check="calendar_information_boundary",
                severity="warning",
                message=f"{origin.role}: the feature window holds no transactions",
            )
        ]
    dates = pd.to_datetime(edges["date"], errors="coerce")
    latest, earliest = dates.max(), dates.min()
    problems = []
    if dates.isna().any():
        problems.append("undated rows reached the feature window")
    if latest >= origin.origin:
        problems.append(f"latest feature transaction {latest} is not before the origin")
    if origin.feature_start is not None and earliest < origin.feature_start:
        problems.append(f"earliest feature transaction {earliest} precedes the lookback start")
    if problems:
        return [
            LeakageFinding(
                check="calendar_information_boundary",
                severity="error",
                message=f"{origin.role}: " + "; ".join(problems),
            )
        ]
    return [
        LeakageFinding(
            check="calendar_information_boundary",
            severity="info",
            message=(
                f"{origin.role}: {len(edges)} feature transactions dated "
                f"{earliest} .. {latest}, all before the origin {origin.origin}"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_spec(spec: ExperimentSpec) -> None:
    if spec.features.scope != CALENDAR_SCOPE:
        raise ValueError(
            f"{CALENDAR_STRATEGY} scores each snapshot from the transactions before "
            f"its own origin, so FeatureSpec.scope must be {CALENDAR_SCOPE!r}; got "
            f"{spec.features.scope!r}. 'global' would summarise the prediction "
            f"window itself; 'train_window' would score the test origin with "
            f"features frozen at the training origin."
        )
    if not spec.features.sets:
        raise ValueError(f"{CALENDAR_STRATEGY} requires at least one feature set")
    for name in spec.features.sets:
        fs = get_feature_set(name)
        if fs.requires_graph:
            raise ValueError(
                f"feature set {name!r} requires a networkx graph; {CALENDAR_STRATEGY} "
                f"builds no per-origin graph snapshot"
            )
        if not fs.time_scopable:
            raise ValueError(
                f"feature set {name!r} does not honour a time window and could "
                f"summarise behaviour after the origin"
            )


def run_calendar_experiment(
    spec: ExperimentSpec,
    dataset: ExperimentDataset,
    *,
    candidate_factory: Optional[Callable[[], Candidate]] = None,
    output_dir: Optional[Path | str] = None,
    checkpoint_dir: Optional[Path | str] = None,
) -> ExperimentResult:
    """Fit at the train origin, calibrate and threshold at validation, score test once."""
    started = _now()
    t_start = time.perf_counter()
    _check_spec(spec)

    if candidate_factory is None:
        def candidate_factory() -> Candidate:  # type: ignore[misc]
            return build_candidate(spec.model.family, spec.model.params)

    candidate = candidate_factory()
    if not getattr(candidate, "calendar_compatible", False):
        raise ValueError(
            f"candidate family {candidate.family!r} is not declared "
            f"calendar_compatible. Under {CALENDAR_STRATEGY} a model is fitted on "
            f"one snapshot and scored on another; a candidate that reads "
            f"ctx.dataset.labels (whole-file labels, which include the future) or "
            f"a graph cannot be run here."
        )

    seed_report = set_global_seeds(spec.seed)
    plan = build_calendar_plan(dataset, spec.split)
    findings: list[LeakageFinding] = _audit_plan(plan)
    raise_on_errors(findings)

    notes: list[str] = list(plan.notes)
    if spec.dataset.synthetic:
        notes.append(
            "SYNTHETIC DATASET: these numbers describe generated data and are "
            "not evidence about real-world detection performance."
        )

    # --- one snapshot per origin ----------------------------------------------
    dates = _edge_dates(dataset)
    t_feat = time.perf_counter()
    rows: dict[str, pd.DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    reuse: dict[str, list[str]] = {}
    train_edges: Optional[pd.DataFrame] = None
    for role in ROLES:
        origin = plan.origins[role]
        frame, provenance, reused, edges = _snapshot(dataset, spec.features.sets, origin, dates)
        findings.extend(_audit_boundary(origin, edges))
        findings.extend(audit_label_columns(frame))
        rows[role] = _rows_for(frame, origin, dataset.entity_ids)
        reuse[role] = reused
        if role == "train":
            train_edges = edges
        del frame, edges
    feat_seconds = time.perf_counter() - t_feat

    tr, va, te = (plan.origins[r] for r in ROLES)
    y_train = tr.labels.astype(np.int8)
    y_val = va.labels.astype(np.int8)
    y_test = te.labels.astype(np.int8)
    findings.extend(audit_perfect_separators(rows["train"], y_train))
    findings.append(
        LeakageFinding(
            check="feature_scope",
            severity="info",
            message=(
                "each origin's features were computed only from transactions dated "
                "before that origin"
                + (f" and within a {plan.lookback} lookback" if plan.lookback is not None else "")
            ),
        )
    )
    raise_on_errors(findings)

    fold = SplitFold(
        fold_idx=0,
        train_idx=tr.eligible_idx,
        val_idx=va.eligible_idx,
        test_idx=te.eligible_idx,
        train_window_end=tr.origin,
        val_window_end=va.origin,
    )
    fold_seed = derive_seed(spec.seed, "fold", 0)
    set_global_seeds(fold_seed)

    # Fitting context: train and validation snapshots stacked, so train_idx and
    # val_idx address rows of ctx.features exactly as they do elsewhere.
    n_tr, n_va = len(rows["train"]), len(rows["val"])
    ctx = FitContext(
        dataset=dataset,
        features=pd.concat([rows["train"], rows["val"]], axis=0),
        fold=fold,
        train_idx=np.arange(n_tr, dtype=np.int64),
        val_idx=np.arange(n_tr, n_tr + n_va, dtype=np.int64),
        y_train=y_train,
        y_val=y_val,
        train_edges=train_edges,
        seed=fold_seed,
        imbalance=spec.model.imbalance,
    )
    t_fit = time.perf_counter()
    candidate.fit(ctx)
    fit_seconds = time.perf_counter() - t_fit

    # Scoring: each snapshot on its own, so a rank-based floor ranks the
    # population as it stood at that origin and nothing else.
    def score(role: str) -> np.ndarray:
        scoring = dataclasses.replace(ctx, features=rows[role])
        return candidate.predict_scores(scoring, np.arange(len(rows[role]), dtype=np.int64))

    ctx.features = None  # the stacked copy is no longer needed
    t_score = time.perf_counter()
    val_raw = score("val")
    calibrator, cal_notes = build_calibrator(
        spec.calibration, y_val, val_raw, model_emits_probabilities=candidate.emits_probabilities
    )
    val_scores = calibrator.transform(val_raw) if len(val_raw) else val_raw
    decision = select_threshold(spec.threshold, y_val, val_scores)

    # --- test: scored once, with everything already frozen ------------------
    test_raw = score("test")
    test_scores = calibrator.transform(test_raw) if len(test_raw) else test_raw
    score_seconds = time.perf_counter() - t_score

    metric_kwargs = dict(
        threshold=decision.value,
        k_values=spec.evaluation.k_values,
        alert_rates=spec.evaluation.alert_rates,
        probability_scale=calibrator.produces_probabilities,
        calibration_bins=spec.evaluation.calibration_bins,
    )
    val_metrics = binary_metrics(y_val, val_scores, **metric_kwargs)
    test_metrics = binary_metrics(y_test, test_scores, **metric_kwargs)
    new = ~te.prior_involvement
    subgroup = {
        "no_prior_involvement": {
            "definition": (
                "test entities with no flagged transaction before the test origin; "
                "the population for which the task is forecasting first involvement"
            ),
            "n": int(new.sum()),
            "n_positive": int(y_test[new].sum()),
            "metrics": binary_metrics(y_test[new], test_scores[new], **metric_kwargs),
        }
    }

    checkpoint: Optional[dict[str, Any]] = None
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_identity(candidate.save(directory / f"{spec.name}_fold0"))

    counts = {
        "n_train": n_tr,
        "n_train_pos": int(y_train.sum()),
        "n_val": n_va,
        "n_val_pos": int(y_val.sum()),
        "n_test": int(len(y_test)),
        "n_test_pos": int(y_test.sum()),
        "n_test_no_prior_involvement": int(new.sum()),
        "n_test_pos_no_prior_involvement": int(y_test[new].sum()),
        "n_test_cold_start_pos_excluded": te.n_cold_start_positive,
    }
    outcome = FoldOutcome(
        fold_idx=0,
        counts=counts,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        threshold=decision.to_dict(),
        calibration=calibrator.to_dict(),
        leakage=[f.to_dict() for f in findings],
        fit_report=dict(candidate.fit_report),
        runtime_seconds={"features": feat_seconds, "fit": fit_seconds, "score": score_seconds},
        checkpoint=checkpoint,
        notes=cal_notes + list(ctx.notes),
        subgroup_test_metrics=subgroup,
    )
    logger.info(
        "calendar test AP=%s  threshold=%s",
        test_metrics.get("average_precision"),
        decision.value,
    )

    feature_summary = {
        "scope": CALENDAR_SCOPE,
        "as_of": {r: str(plan.origins[r].origin) for r in ROLES},
        "windows": {
            r: [
                None if plan.origins[r].feature_start is None else str(plan.origins[r].feature_start),
                str(plan.origins[r].origin),
            ]
            for r in ROLES
        },
        "n_columns": int(rows["test"].shape[1]),
        "sets": provenance,
        "snapshot_reuse": reuse,
        "notes": [],
    }
    n_nan = int(rows["test"].isna().to_numpy().sum())
    if n_nan:
        feature_summary["notes"].append(
            f"{n_nan} NaN cells in the test snapshot were left in place; candidates "
            f"impute inside their train-fitted preprocessing."
        )

    result = ExperimentResult(
        spec=spec,
        dataset_summary=dataset.summary(),
        split_summary=plan.to_dict(),
        feature_summary=feature_summary,
        folds=[outcome],
        aggregate=aggregate_fold_metrics([test_metrics]),
        validation_aggregate=aggregate_fold_metrics([val_metrics]),
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
