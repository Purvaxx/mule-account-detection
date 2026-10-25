"""
Layer 6 - Split Strategies
===========================

Choosing the partition is the single most consequential decision in this
project's evaluation, so it is a named, registered, recorded strategy rather
than an implicit property of a helper function.

Every strategy returns a :class:`SplitPlan` that declares what kind of question
its folds actually answer:

``entity_disjoint``
    An entity appears in exactly one of train / validation / test.  False means
    the same entity is scored in more than one partition, which inflates every
    metric.

``time_aware``
    Training data precedes evaluation data in wall-clock time.  False means the
    model may have been fitted on behaviour that occurred *after* the behaviour
    it is scored on - impossible in deployment.

``graph_leakage_controlled``
    Held-out entities are not direct graph neighbours of training entities.
    Only the component-grouped strategy achieves this; on a single connected
    transaction network it is often unachievable without destroying the graph,
    and the plan says so rather than pretending.

No strategy is "the correct one".  A transductive plan is a legitimate research
setup; it just has to be labelled as one so its numbers are not read as
operational performance.

Available strategies
--------------------
``stratified_kfold``
    Entity-disjoint, label-stratified K folds.  Not time-aware.
``time_forward_holdout``
    Single train/val/test partition cut at two time quantiles.  Time-aware and
    entity-disjoint (by the entity's reference activity time).
``time_forward_rolling``
    Expanding-window folds: fold i trains on everything before cut i, validates
    on the next window, tests on the window after that.
``component_grouped_kfold``
    K folds grouped by weakly-connected component, so no test entity shares a
    component with a training entity.
``legacy_exp3_transductive``
    Reproduces the historical Experiment 1-5 protocol generically (round-robin
    over positives, random negative quarters, whole graph visible).  Retained
    as the regression reference, and explicitly flagged transductive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.experiments.dataset import ExperimentDataset
from src.experiments.spec import SplitSpec

logger = logging.getLogger(__name__)

__all__ = [
    "SplitFold",
    "SplitPlan",
    "build_split_plan",
    "register_split_strategy",
    "available_split_strategies",
]


@dataclass(frozen=True)
class SplitFold:
    """One train / validation / test partition.

    Attributes
    ----------
    train_idx, val_idx, test_idx:
        Positional indices into ``ExperimentDataset.entity_ids``.
    train_window_end:
        For time-aware folds, the exclusive upper time bound of the training
        window.  Feature computation under ``FeatureSpec.scope ==
        "train_window"`` uses only transactions strictly before this.  ``None``
        for non-temporal folds.
    val_window_end:
        Exclusive upper bound of the validation window, for the same purpose
        when scoring validation entities.
    """

    fold_idx: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_window_end: Optional[pd.Timestamp] = None
    val_window_end: Optional[pd.Timestamp] = None

    def counts(self, labels: np.ndarray) -> dict[str, int]:
        """Sizes and positive counts of each partition."""
        out: dict[str, int] = {}
        for name, idx in (
            ("train", self.train_idx),
            ("val", self.val_idx),
            ("test", self.test_idx),
        ):
            out[f"n_{name}"] = int(len(idx))
            out[f"n_{name}_pos"] = int(labels[idx].sum()) if len(idx) else 0
        return out


@dataclass
class SplitPlan:
    """A named, reproducible set of folds plus its methodological declaration."""

    strategy: str
    params: dict[str, Any]
    seed: int
    folds: list[SplitFold]
    entity_disjoint: bool
    time_aware: bool
    graph_leakage_controlled: bool
    notes: list[str] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def regime(self) -> str:
        """One-word description used in reports."""
        if self.time_aware:
            return "time-forward"
        if self.graph_leakage_controlled:
            return "inductive-grouped"
        if self.entity_disjoint:
            return "entity-disjoint"
        return "transductive"

    def to_dict(self, labels: Optional[np.ndarray] = None) -> dict[str, Any]:
        folds: list[dict[str, Any]] = []
        for f in self.folds:
            entry: dict[str, Any] = {"fold_idx": f.fold_idx}
            if labels is not None:
                entry.update(f.counts(labels))
            if f.train_window_end is not None:
                entry["train_window_end"] = str(f.train_window_end)
            if f.val_window_end is not None:
                entry["val_window_end"] = str(f.val_window_end)
            folds.append(entry)
        return {
            "strategy": self.strategy,
            "params": dict(self.params),
            "seed": self.seed,
            "regime": self.regime(),
            "entity_disjoint": self.entity_disjoint,
            "time_aware": self.time_aware,
            "graph_leakage_controlled": self.graph_leakage_controlled,
            "n_folds": self.n_folds,
            "notes": list(self.notes),
            "folds": folds,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, Callable[..., SplitPlan]] = {}


def register_split_strategy(name: str) -> Callable[[Callable[..., SplitPlan]], Callable[..., SplitPlan]]:
    """Decorator registering a split strategy under ``name``."""

    def decorator(fn: Callable[..., SplitPlan]) -> Callable[..., SplitPlan]:
        if name in _STRATEGIES:
            raise ValueError(f"Split strategy {name!r} is already registered.")
        _STRATEGIES[name] = fn
        return fn

    return decorator


def available_split_strategies() -> list[str]:
    return sorted(_STRATEGIES)


def build_split_plan(dataset: ExperimentDataset, spec: SplitSpec) -> SplitPlan:
    """Resolve ``spec.strategy`` and produce the folds.

    The resulting plan is validated for partition disjointness before it is
    returned; a strategy that produces overlapping partitions is a bug, not a
    configuration choice.
    """
    if spec.strategy not in _STRATEGIES:
        raise KeyError(
            f"Unknown split strategy {spec.strategy!r}. "
            f"Available: {available_split_strategies()}"
        )
    plan = _STRATEGIES[spec.strategy](dataset, seed=spec.seed, **spec.params)
    _validate_plan(plan, dataset)
    _annotate_positive_coverage(plan, dataset)
    return plan


def _validate_plan(plan: SplitPlan, dataset: ExperimentDataset) -> None:
    n = dataset.n_entities
    for fold in plan.folds:
        parts = {"train": fold.train_idx, "val": fold.val_idx, "test": fold.test_idx}
        for name, idx in parts.items():
            if len(idx) and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(
                    f"{plan.strategy} fold {fold.fold_idx}: {name} index out of range."
                )
            if len(set(idx.tolist())) != len(idx):
                raise ValueError(
                    f"{plan.strategy} fold {fold.fold_idx}: {name} contains duplicates."
                )
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = np.intersect1d(parts[a], parts[b])
            if overlap.size:
                raise ValueError(
                    f"{plan.strategy} fold {fold.fold_idx}: {a}/{b} partitions "
                    f"overlap on {overlap.size} entities."
                )


def _annotate_positive_coverage(plan: SplitPlan, dataset: ExperimentDataset) -> None:
    """Record, honestly, which folds cannot support label-dependent metrics."""
    labels = dataset.labels
    for fold in plan.folds:
        for name, idx in (("validation", fold.val_idx), ("test", fold.test_idx)):
            if len(idx) == 0:
                plan.notes.append(f"fold {fold.fold_idx}: {name} partition is empty.")
            elif int(labels[idx].sum()) == 0:
                plan.notes.append(
                    f"fold {fold.fold_idx}: {name} partition contains no positives; "
                    f"label-dependent metrics are undefined there."
                )
        if int(labels[fold.train_idx].sum()) == 0:
            plan.notes.append(
                f"fold {fold.fold_idx}: training partition contains no positives; "
                f"supervised candidates cannot learn the positive class."
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _round_robin_groups(items: np.ndarray, n_groups: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Deal ``items`` into ``n_groups`` after a reproducible shuffle.

    Round-robin rather than ``array_split`` so that a population smaller than
    ``n_groups`` still spreads one item per group instead of emptying the tail.
    """
    shuffled = rng.permutation(items)
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for position, item in enumerate(shuffled):
        groups[position % n_groups].append(int(item))
    return [np.array(sorted(g), dtype=np.int64) for g in groups]


def _concat(*arrays: np.ndarray) -> np.ndarray:
    parts = [a for a in arrays if len(a)]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(parts)).astype(np.int64)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@register_split_strategy("stratified_kfold")
def stratified_kfold(
    dataset: ExperimentDataset,
    *,
    seed: int = 42,
    n_folds: int = 5,
    val_folds: int = 1,
) -> SplitPlan:
    """Label-stratified, entity-disjoint K-fold cross-validation.

    Fold *i* tests on group *i*, validates on the next ``val_folds`` groups and
    trains on the rest.  Positives and negatives are dealt into groups
    separately so the positive class is spread as evenly as the count allows -
    with fewer positives than folds, some folds legitimately get none, and that
    is recorded rather than worked around.

    Not time-aware.  Use it for a first look at a dataset, not for an
    operational claim.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if not 1 <= val_folds <= n_folds - 2 + 1:
        # need at least one group each for train and test
        if val_folds >= n_folds - 1:
            raise ValueError("val_folds leaves no training data.")
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    pos_groups = _round_robin_groups(pos, n_folds, rng)
    neg_groups = _round_robin_groups(neg, n_folds, rng)
    groups = [_concat(pos_groups[i], neg_groups[i]) for i in range(n_folds)]

    folds: list[SplitFold] = []
    for i in range(n_folds):
        test = groups[i]
        val_ids = [groups[(i + 1 + j) % n_folds] for j in range(val_folds)]
        val = _concat(*val_ids)
        used = {i} | {(i + 1 + j) % n_folds for j in range(val_folds)}
        train = _concat(*[groups[j] for j in range(n_folds) if j not in used])
        folds.append(SplitFold(fold_idx=i, train_idx=train, val_idx=val, test_idx=test))

    notes = []
    if len(pos) < n_folds:
        notes.append(
            f"only {len(pos)} positives for {n_folds} folds: at least "
            f"{n_folds - len(pos)} fold(s) have a positive-free partition."
        )
    return SplitPlan(
        strategy="stratified_kfold",
        params={"n_folds": n_folds, "val_folds": val_folds},
        seed=seed,
        folds=folds,
        entity_disjoint=True,
        time_aware=False,
        graph_leakage_controlled=False,
        notes=notes,
    )


def _time_reference(dataset: ExperimentDataset, time_reference: str) -> pd.Series:
    if not dataset.has_time():
        raise ValueError(
            "A time-aware split was requested but the dataset has no usable "
            "'date' column. Use an explicitly non-temporal strategy instead of "
            "pretending an ordering exists."
        )
    return dataset.entity_times(how=time_reference)


@register_split_strategy("time_forward_holdout")
def time_forward_holdout(
    dataset: ExperimentDataset,
    *,
    seed: int = 42,
    train_quantile: float = 0.6,
    val_quantile: float = 0.8,
    time_reference: str = "first",
) -> SplitPlan:
    """One time-ordered train / validation / test partition.

    Entities are placed by their reference activity time (``"first"`` by
    default: the model is asked to score entities it has only just met).
    Entities without a dated transaction cannot be placed on the timeline and
    are excluded from all three partitions - assigning them arbitrarily would
    silently break the temporal ordering.
    """
    if not 0.0 < train_quantile < val_quantile < 1.0:
        raise ValueError("Require 0 < train_quantile < val_quantile < 1.")
    times = _time_reference(dataset, time_reference)
    positional = np.arange(dataset.n_entities)
    dated = times.notna().to_numpy()

    values = times[dated]
    train_cut = values.quantile(train_quantile)
    val_cut = values.quantile(val_quantile)

    t = times.to_numpy()
    train = positional[dated & (t < np.datetime64(train_cut))]
    val = positional[dated & (t >= np.datetime64(train_cut)) & (t < np.datetime64(val_cut))]
    test = positional[dated & (t >= np.datetime64(val_cut))]

    notes = [
        f"entities placed by {time_reference} activity date; "
        f"train < {train_cut}, val < {val_cut}, test >= {val_cut}."
    ]
    n_undated = int((~dated).sum())
    if n_undated:
        notes.append(
            f"{n_undated} entities have no dated transaction and are excluded "
            f"from every partition of this time-aware split."
        )

    return SplitPlan(
        strategy="time_forward_holdout",
        params={
            "train_quantile": train_quantile,
            "val_quantile": val_quantile,
            "time_reference": time_reference,
        },
        seed=seed,
        folds=[
            SplitFold(
                fold_idx=0,
                train_idx=train.astype(np.int64),
                val_idx=val.astype(np.int64),
                test_idx=test.astype(np.int64),
                train_window_end=pd.Timestamp(train_cut),
                val_window_end=pd.Timestamp(val_cut),
            )
        ],
        entity_disjoint=True,
        time_aware=True,
        graph_leakage_controlled=False,
        notes=notes,
    )


@register_split_strategy("time_forward_rolling")
def time_forward_rolling(
    dataset: ExperimentDataset,
    *,
    seed: int = 42,
    n_folds: int = 3,
    initial_quantile: float = 0.5,
    time_reference: str = "first",
) -> SplitPlan:
    """Expanding-window time-series cross-validation over entity arrival time.

    Fold *i* trains on everything before cut *i*, validates on the slice
    between cut *i* and cut *i+1*, and tests on the slice after cut *i+1*
    up to cut *i+2* (the remainder for the last fold).  Training windows grow;
    evaluation always looks forward.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")
    if not 0.0 < initial_quantile < 1.0:
        raise ValueError("initial_quantile must be in (0, 1).")

    times = _time_reference(dataset, time_reference)
    positional = np.arange(dataset.n_entities)
    dated = times.notna().to_numpy()
    values = times[dated]

    remaining = 1.0 - initial_quantile
    step = remaining / (n_folds + 1)
    quantiles = [initial_quantile + step * i for i in range(n_folds + 2)]
    cuts = [values.quantile(min(q, 1.0)) for q in quantiles]

    t = times.to_numpy()
    folds: list[SplitFold] = []
    for i in range(n_folds):
        train_cut = np.datetime64(cuts[i])
        val_cut = np.datetime64(cuts[i + 1])
        test_cut = np.datetime64(cuts[i + 2])
        train = positional[dated & (t < train_cut)]
        val = positional[dated & (t >= train_cut) & (t < val_cut)]
        if i == n_folds - 1:
            test = positional[dated & (t >= val_cut)]
        else:
            test = positional[dated & (t >= val_cut) & (t < test_cut)]
        folds.append(
            SplitFold(
                fold_idx=i,
                train_idx=train.astype(np.int64),
                val_idx=val.astype(np.int64),
                test_idx=test.astype(np.int64),
                train_window_end=pd.Timestamp(cuts[i]),
                val_window_end=pd.Timestamp(cuts[i + 1]),
            )
        )

    notes = [
        f"expanding window over {time_reference} activity date; "
        f"cuts at quantiles {[round(q, 4) for q in quantiles]}."
    ]
    n_undated = int((~dated).sum())
    if n_undated:
        notes.append(f"{n_undated} undated entities excluded from every partition.")

    return SplitPlan(
        strategy="time_forward_rolling",
        params={
            "n_folds": n_folds,
            "initial_quantile": initial_quantile,
            "time_reference": time_reference,
        },
        seed=seed,
        folds=folds,
        entity_disjoint=True,
        time_aware=True,
        graph_leakage_controlled=False,
        notes=notes,
    )


@register_split_strategy("component_grouped_kfold")
def component_grouped_kfold(
    dataset: ExperimentDataset,
    *,
    seed: int = 42,
    n_folds: int = 5,
    val_folds: int = 1,
) -> SplitPlan:
    """K-fold where whole weakly-connected components stay together.

    This is the strongest available control against graph-neighbourhood
    leakage: no test entity is reachable from a training entity, so a
    neighbourhood-aggregating model cannot see a training label through the
    graph.  On a network dominated by one giant component the folds become very
    unbalanced; the plan records the component-size distribution so that is
    visible rather than surprising.
    """
    import networkx as nx

    if dataset.graph is None:
        raise ValueError("component_grouped_kfold requires a graph.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    rng = np.random.default_rng(seed)
    undirected = nx.Graph()
    undirected.add_nodes_from(dataset.entity_ids)
    if dataset.edge_table is not None and len(dataset.edge_table):
        undirected.add_edges_from(
            zip(dataset.edge_table["source_entity"], dataset.edge_table["target_entity"])
        )

    components = [sorted(c) for c in nx.connected_components(undirected)]
    components.sort(key=lambda c: (-len(c), c[0]))
    comp_indices = [
        np.array([dataset.index_of(e) for e in comp], dtype=np.int64) for comp in components
    ]

    # Greedy largest-first bin packing keeps groups as even as the component
    # sizes permit; the shuffle only breaks ties among equal-sized components.
    order = list(range(len(comp_indices)))
    rng.shuffle(order)
    order.sort(key=lambda i: -len(comp_indices[i]))
    groups: list[list[np.ndarray]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for i in order:
        target = int(np.argmin(sizes))
        groups[target].append(comp_indices[i])
        sizes[target] += len(comp_indices[i])
    group_idx = [_concat(*g) if g else np.array([], dtype=np.int64) for g in groups]

    folds: list[SplitFold] = []
    for i in range(n_folds):
        test = group_idx[i]
        used = {i} | {(i + 1 + j) % n_folds for j in range(val_folds)}
        val = _concat(*[group_idx[(i + 1 + j) % n_folds] for j in range(val_folds)])
        train = _concat(*[group_idx[j] for j in range(n_folds) if j not in used])
        folds.append(SplitFold(fold_idx=i, train_idx=train, val_idx=val, test_idx=test))

    comp_sizes = sorted((len(c) for c in comp_indices), reverse=True)
    notes = [
        f"{len(comp_sizes)} weakly-connected components; "
        f"largest {comp_sizes[:5]}; fold sizes {sizes}."
    ]
    if comp_sizes and comp_sizes[0] > 0.5 * dataset.n_entities:
        notes.append(
            "one component holds over half the population, so fold sizes are "
            "necessarily unbalanced; per-fold metrics are not exchangeable."
        )

    return SplitPlan(
        strategy="component_grouped_kfold",
        params={"n_folds": n_folds, "val_folds": val_folds},
        seed=seed,
        folds=folds,
        entity_disjoint=True,
        time_aware=False,
        graph_leakage_controlled=True,
        notes=notes,
    )


@register_split_strategy("legacy_exp3_transductive")
def legacy_exp3_transductive(
    dataset: ExperimentDataset,
    *,
    seed: int = 42,
    n_folds: int = 4,
) -> SplitPlan:
    """The historical Experiment 1-5 protocol, generalised to any positive count.

    Reproduces the structure of ``src.models.dataset.build_cv_folds``: positives
    are dealt round-robin so each appears once per role, negatives are dealt
    into ``n_folds`` random groups, and every fold uses one group for test, one
    for validation and the rest for training.  Unlike the original it does not
    require exactly four positives.

    Declared transductive: the graph handed to the model contains the held-out
    entities and all their edges, so a neighbourhood-aggregating model sees test
    entities during training even though their labels are masked.  Numbers from
    this plan are a regression reference for the frozen checkpoint, not an
    estimate of deployed performance.
    """
    if n_folds < 3:
        raise ValueError("legacy protocol needs at least 3 folds (train/val/test).")
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    pos_groups = _round_robin_groups(pos, n_folds, rng)
    neg_groups = _round_robin_groups(neg, n_folds, rng)

    folds: list[SplitFold] = []
    for i in range(n_folds):
        test = _concat(pos_groups[i], neg_groups[i])
        j = (i + 1) % n_folds
        val = _concat(pos_groups[j], neg_groups[j])
        rest = [k for k in range(n_folds) if k not in (i, j)]
        train = _concat(*[_concat(pos_groups[k], neg_groups[k]) for k in rest])
        folds.append(SplitFold(fold_idx=i, train_idx=train, val_idx=val, test_idx=test))

    notes = [
        "transductive protocol: whole-graph topology and whole-graph features "
        "are visible at training time; only labels are masked.",
        "retained as the legacy regression reference for Experiment 3.",
    ]
    if len(pos) < n_folds:
        notes.append(
            f"{len(pos)} positives across {n_folds} folds: some partitions have none."
        )

    return SplitPlan(
        strategy="legacy_exp3_transductive",
        params={"n_folds": n_folds},
        seed=seed,
        folds=folds,
        entity_disjoint=True,
        time_aware=False,
        graph_leakage_controlled=False,
        notes=notes,
    )
