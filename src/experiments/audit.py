"""
Layer 6 - Dataset Scientific Audit
====================================

Answers, for one dataset, the question that has to be settled before any model
comparison is worth running: **can this data separate candidate models at all?**

A dataset earns that role by having enough positives to estimate a metric,
labels that are not recoverable from a trivial quantity, and a partition
strategy that does not leak.  A dataset that fails any of those can still be
useful - as a regression fixture, as a scaling test - but it cannot rank
models, and reporting a ranking from it would be reporting noise or reporting
the label generator.

Three blocks, in the order they matter
--------------------------------------
``data``
    Sizes, time coverage, missingness, duplicate structure, field inventory.
    Facts, no interpretation.

``labels``
    What is labelled and at what granularity; whether the positive count
    supports cross-validation; whether temporal and entity-disjoint evaluation
    are possible; which columns could carry the label into a feature.

``structural_triviality``
    The decisive block.  Each simple quantity an analyst could compute without
    a model - transaction count, degree, fan-in, fan-out, volume, counterparty
    count, centrality, component size, first and last activity, velocity - is
    scored as a *ranking* of the population, and its average precision is
    compared with the prevalence floor.  Then a small unregularised tree is
    fitted on those same quantities, because a label can be trivial as a
    *combination* while no single column looks suspicious: SAML-D's entity
    labels are exactly that case.

    The block also reports **class-free bands**: discrete values of a quantity
    that no negative entity takes at all.  A hole in the negative distribution
    is the fingerprint of a label written by the same process that wrote the
    structure, and it is the specific defect that disqualified SAML-D.

The tree is a probe, not a model.  It is fitted on a stratified half of the
population and scored on the other half, it uses no tuning, and its result is
reported as a property of the *dataset*.  A probe average precision at or near
1.0 means the labels are a deterministic function of structure, and no
experiment run on that dataset can distinguish a good model from a bad one.

Verdict vocabulary
------------------
``MODEL_SELECTION_READY``
    Enough positives, non-trivial labels, a usable split.  Results from it can
    rank candidates.
``STRUCTURAL_VALIDATION_ONLY``
    The pipeline can be exercised end to end and its scaling verified, but the
    labels are recoverable from structure, so ranking is meaningless.
``REGRESSION_ONLY``
    Too few positives to estimate anything.  Its value is that its numbers are
    pinned and a change in them signals a broken pipeline.
``BLOCKED``
    Cannot be used at all, with the blocking reason stated.

Nothing here fabricates a missing value.  A quantity the dataset cannot support
is ``None`` with a reason beside it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.experiments.dataset import ExperimentDataset

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetAudit",
    "audit_dataset",
    "save_audit",
    "MODEL_SELECTION_READY",
    "STRUCTURAL_VALIDATION_ONLY",
    "REGRESSION_ONLY",
    "BLOCKED",
    "DOMAIN_AML",
    "DOMAIN_CARD_FRAUD",
    "DOMAIN_UNKNOWN",
]

#: Declared label domain.  Whether a dataset's positive class is money
#: laundering is a judgement about the publisher's definition, not something
#: measurable from the file, so the loader declares it and the audit records it.
DOMAIN_AML = "money_laundering"
DOMAIN_CARD_FRAUD = "card_fraud"
DOMAIN_UNKNOWN = "unknown"

MODEL_SELECTION_READY = "MODEL_SELECTION_READY"
STRUCTURAL_VALIDATION_ONLY = "STRUCTURAL_VALIDATION_ONLY"
REGRESSION_ONLY = "REGRESSION_ONLY"
BLOCKED = "BLOCKED"

#: A dataset with fewer positives than this cannot support k-fold estimation:
#: at four folds it would put fewer than five positives in a test partition,
#: where average precision is dominated by the position of one entity.
MIN_POSITIVES_FOR_CV = 40

#: Probe average precision at or above this is treated as trivial recovery.
#: It is deliberately strict. A probe reaching 0.99 on a 1 % prevalence
#: population is not "a good model"; it is a label that was written by the same
#: process that wrote the structure.
TRIVIAL_AP = 0.99

#: Probe average precision above this, while below TRIVIAL_AP, is flagged as a
#: strong structural correlate worth stating but not disqualifying.
STRONG_AP = 0.80


@dataclass
class DatasetAudit:
    """The complete audit of one dataset."""

    name: str
    spec: dict[str, Any]
    domain: str
    data: dict[str, Any]
    labels: dict[str, Any]
    structural_triviality: dict[str, Any]
    split_feasibility: dict[str, Any]
    verdict: str
    reason: str
    recommended_use: str
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Block 1 - data
# ---------------------------------------------------------------------------


def _data_block(dataset: ExperimentDataset) -> dict[str, Any]:
    edges = dataset.edge_table
    block: dict[str, Any] = {
        "n_entities": dataset.n_entities,
        "n_transactions": 0 if edges is None else int(len(edges)),
        "n_positive": dataset.n_positive,
        "n_negative": dataset.n_negative,
        "positive_rate": dataset.prevalence,
        "has_graph_object": dataset.graph is not None,
        "transaction_fields": [] if edges is None else [str(c) for c in edges.columns],
        "entity_fields": (
            [] if dataset.attributes is None else [str(c) for c in dataset.attributes.columns]
        ),
    }
    if edges is None or len(edges) == 0:
        block["timestamp_range"] = None
        block["timestamp_note"] = "dataset carries no transactions"
        return block

    if "date" in edges.columns:
        dates = pd.to_datetime(edges["date"], errors="coerce")
        n_dated = int(dates.notna().sum())
        block["timestamp_range"] = (
            None
            if n_dated == 0
            else {
                "min": str(dates.min()),
                "max": str(dates.max()),
                "span_days": float((dates.max() - dates.min()).days),
                "n_dated": n_dated,
                "n_undated": int(len(dates) - n_dated),
            }
        )
    else:
        block["timestamp_range"] = None
        block["timestamp_note"] = "no 'date' column"

    block["missingness"] = {
        str(c): float(edges[c].isna().mean()) for c in edges.columns
    }

    src = edges["source_entity"]
    dst = edges["target_entity"]
    pair_keys = pd.Series(
        _codes(src).astype(np.int64) * (dataset.n_entities + 1) + _codes(dst).astype(np.int64)
    )
    block["duplicate_structure"] = {
        "n_distinct_ordered_pairs": int(pair_keys.nunique()),
        "mean_transactions_per_pair": float(len(edges) / max(pair_keys.nunique(), 1)),
        "n_self_loops": int((_codes(src) == _codes(dst)).sum()),
        "n_fully_duplicated_rows": int(
            edges.drop(columns=[c for c in ("edge_key",) if c in edges.columns]).duplicated().sum()
        ),
    }
    block["identity_representation"] = (
        "entity ids are the source dataset's account/name identifiers as loaded; "
        "no resolution or merging was applied by the audit"
    )
    return block


def _codes(series: pd.Series) -> np.ndarray:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    return pd.factorize(series, use_na_sentinel=False)[0]


# ---------------------------------------------------------------------------
# Block 2 - labels
# ---------------------------------------------------------------------------


def _label_block(dataset: ExperimentDataset, label_alias_columns: list[str]) -> dict[str, Any]:
    edges = dataset.edge_table
    n_pos = dataset.n_positive
    block: dict[str, Any] = {
        "definition": dataset.spec.label_definition,
        "granularity": "entity",
        "derived_from_transaction_labels": "at least one transaction" in dataset.spec.label_definition,
        "n_positive": n_pos,
        "positive_rate": dataset.prevalence,
        "supports_cross_validation": n_pos >= MIN_POSITIVES_FOR_CV,
        "min_positives_for_cross_validation": MIN_POSITIVES_FOR_CV,
        "label_alias_columns_present": label_alias_columns,
    }

    if n_pos == 0:
        block["cross_validation_note"] = "no positives at all"
    elif n_pos == dataset.n_entities:
        block["cross_validation_note"] = (
            "every entity is positive; there is no negative class to learn against"
        )
    elif n_pos < MIN_POSITIVES_FOR_CV:
        block["cross_validation_note"] = (
            f"{n_pos} positives cannot populate k folds; average precision on a "
            f"test partition holding one or two positives is determined by a "
            f"single entity's rank"
        )

    # Temporal feasibility: does a forward-looking cut leave positives on the
    # later side?  This is the check that disqualified the Reference Dataset.
    if dataset.has_time():
        times = dataset.entity_times("first")
        values = pd.to_datetime(times, errors="coerce")
        labels = pd.Series(dataset.labels, index=times.index)
        usable = values.notna()
        block["entity_time_coverage"] = float(usable.mean())
        quantiles: dict[str, Any] = {}
        for q in (0.5, 0.6, 0.7, 0.8):
            if usable.sum() == 0:
                break
            cut = values[usable].quantile(q)
            after = labels[usable & (values >= cut)]
            quantiles[f"cut_at_q{int(q * 100)}"] = {
                "cutoff": str(cut),
                "n_entities_after": int(len(after)),
                "n_positive_after": int(after.sum()),
            }
        block["forward_split_positives"] = quantiles
        block["supports_time_forward_evaluation"] = any(
            v["n_positive_after"] >= MIN_POSITIVES_FOR_CV for v in quantiles.values()
        )
    else:
        block["entity_time_coverage"] = 0.0
        block["forward_split_positives"] = None
        block["supports_time_forward_evaluation"] = False
        block["temporal_note"] = "no usable transaction timestamps"

    # Entity-disjoint evaluation is possible whenever the unit of prediction is
    # the entity and entities are distinct, which the dataset container already
    # guarantees. It is recorded rather than assumed.
    block["supports_entity_disjoint_evaluation"] = dataset.n_entities > 1
    block["prospective_or_retrospective"] = (
        "retrospective: the flag was attached after the behaviour it describes "
        "was complete, so a feature computed over an entity's whole history "
        "sees the period the label was assigned from"
    )
    if edges is not None and label_alias_columns:
        block["leakage_note"] = (
            f"columns {label_alias_columns} reproduce the label and must never "
            f"enter a feature matrix"
        )
    return block


# ---------------------------------------------------------------------------
# Block 3 - structural triviality
# ---------------------------------------------------------------------------

#: The simple quantities the audit tests.  Deliberately the ones an analyst
#: would reach for first, and deliberately computed here rather than imported
#: from the feature registry, so that the audit stays valid if the registry
#: changes.
_SIMPLE_QUANTITIES = (
    "transaction_count",
    "out_degree",
    "in_degree",
    "fan_in",
    "fan_out",
    "counterparty_count",
    "incoming_volume",
    "outgoing_volume",
    "total_volume",
    "degree_centrality",
    "component_size",
    "first_activity",
    "last_activity",
    "active_span_days",
    "velocity_per_active_day",
)


def _simple_quantities(dataset: ExperimentDataset) -> Optional[pd.DataFrame]:
    """Compute the trivial quantities the triviality probe is run on."""
    edges = dataset.edge_table
    if edges is None or len(edges) == 0:
        return None

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    from src.experiments.sources import amount_column, entity_codes

    src, dst = entity_codes(edges, dataset.entity_ids)
    n = dataset.n_entities
    out = pd.DataFrame(index=pd.Index(dataset.entity_ids, name="entity_id"))

    out_degree = np.bincount(src, minlength=n).astype(np.float64)
    in_degree = np.bincount(dst, minlength=n).astype(np.float64)
    out["transaction_count"] = out_degree + in_degree
    out["out_degree"] = out_degree
    out["in_degree"] = in_degree

    # Self-loops are excluded from every counterparty-shaped quantity: an
    # account paying itself is not a fan-in, a fan-out or a counterparty. The
    # Reference Dataset has 33 and the IBM AML extract 1,709, so the choice is
    # visible in the numbers and is made once, here, rather than per column.
    not_loop = src != dst
    adjacency = csr_matrix(
        (np.ones(int(not_loop.sum()), dtype=np.int8), (src[not_loop], dst[not_loop])),
        shape=(n, n),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    out["fan_out"] = np.diff(adjacency.indptr).astype(np.float64)
    out["fan_in"] = np.diff(adjacency.tocsc().indptr).astype(np.float64)
    undirected = (adjacency + adjacency.T).astype(bool).astype(np.int8)
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    counterparties = np.diff(undirected.indptr).astype(np.float64)
    out["counterparty_count"] = counterparties
    out["degree_centrality"] = counterparties / max(n - 1, 1)
    n_components, membership = connected_components(undirected, directed=False)
    out["component_size"] = np.bincount(membership, minlength=n_components)[
        membership
    ].astype(np.float64)

    name = amount_column(edges)
    if name:
        amounts = pd.to_numeric(edges[name], errors="coerce").abs().to_numpy(dtype=np.float64)
        finite = np.isfinite(amounts)
        out["incoming_volume"] = np.bincount(
            dst[finite], weights=amounts[finite], minlength=n
        )
        out["outgoing_volume"] = np.bincount(
            src[finite], weights=amounts[finite], minlength=n
        )
        out["total_volume"] = out["incoming_volume"] + out["outgoing_volume"]
    else:
        for column in ("incoming_volume", "outgoing_volume", "total_volume"):
            out[column] = np.nan

    if "date" in edges.columns:
        dates = pd.to_datetime(edges["date"], errors="coerce")
        seconds = dates.to_numpy(dtype="datetime64[s]").astype(np.float64)
        seconds[dates.isna().to_numpy()] = np.nan
        ent = np.concatenate([src, dst])
        both = np.concatenate([seconds, seconds])
        dated = np.isfinite(both)
        grouped = pd.Series(both[dated]).groupby(ent[dated], sort=False)
        first = np.full(n, np.nan)
        last = np.full(n, np.nan)
        agg_min, agg_max = grouped.min(), grouped.max()
        first[agg_min.index.to_numpy()] = agg_min.to_numpy()
        last[agg_max.index.to_numpy()] = agg_max.to_numpy()
        out["first_activity"] = first
        out["last_activity"] = last
        span = (last - first) / 86400.0
        out["active_span_days"] = span
        with np.errstate(invalid="ignore", divide="ignore"):
            out["velocity_per_active_day"] = np.where(
                span > 0, out["transaction_count"].to_numpy() / span, np.nan
            )
    else:
        for column in ("first_activity", "last_activity", "active_span_days",
                       "velocity_per_active_day"):
            out[column] = np.nan

    return out[list(_SIMPLE_QUANTITIES)]


def _triviality_block(
    dataset: ExperimentDataset, *, seed: int = 42
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = dataset.labels.astype(int)
    prevalence = float(labels.mean()) if len(labels) else 0.0
    block: dict[str, Any] = {
        "prevalence_floor_average_precision": prevalence,
        "single_quantity": {},
        "combined_probe": None,
        "verdict": None,
    }

    if labels.sum() == 0 or labels.sum() == len(labels):
        block["note"] = (
            "the population is single-class; no ranking metric is defined and "
            "no triviality statement can be made"
        )
        block["verdict"] = "undefined"
        return block

    quantities = _simple_quantities(dataset)
    if quantities is None:
        block["note"] = "dataset has no transactions to derive simple quantities from"
        block["verdict"] = "undefined"
        return block

    best_name, best_ap = None, -1.0
    for column in quantities.columns:
        values = quantities[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).any():
            block["single_quantity"][column] = {
                "average_precision": None,
                "reason": "quantity is undefined for every entity in this dataset",
            }
            continue
        filled = np.where(np.isfinite(values), values, np.nanmin(values) - 1.0)
        if np.ptp(filled) == 0:
            block["single_quantity"][column] = {
                "average_precision": None,
                "reason": "quantity is constant across the population",
            }
            continue
        ascending = float(average_precision_score(labels, -filled))
        descending = float(average_precision_score(labels, filled))
        direction = "descending" if descending >= ascending else "ascending"
        ap = max(ascending, descending)
        block["single_quantity"][column] = {
            "average_precision": ap,
            "direction": direction,
            "lift_over_prevalence": ap / prevalence if prevalence > 0 else None,
            "roc_auc": float(
                roc_auc_score(labels, filled if direction == "descending" else -filled)
            ),
        }
        if ap > best_ap:
            best_ap, best_name = ap, column

    block["best_single_quantity"] = {"name": best_name, "average_precision": best_ap}
    block["class_free_bands"] = _class_free_bands(quantities, labels)

    probe = _combined_probe(quantities, labels, seed=seed)
    block["combined_probe"] = probe

    probe_ap = probe.get("test_average_precision") if probe else None
    if probe_ap is None:
        block["verdict"] = "undefined"
    elif probe_ap >= TRIVIAL_AP:
        block["verdict"] = "trivially_recoverable"
    elif probe_ap >= STRONG_AP:
        block["verdict"] = "strong_structural_correlate"
    else:
        block["verdict"] = "non_trivial"
    return block


#: A quantity with more distinct values than this is not treated as discrete
#: and is skipped by the class-free-band check.
_MAX_DISCRETE_VALUES = 200
#: A band is only worth reporting when it holds at least this many entities of
#: the class that *is* present.
_MIN_BAND_SIZE = 25


def _class_free_bands(
    quantities: pd.DataFrame, labels: np.ndarray
) -> dict[str, Any]:
    """Discrete quantity values at which one class is entirely absent.

    This is the compact, reproducible form of the failure that disqualified
    SAML-D. A single column can look healthy on average precision while
    containing values that *no negative ever takes* - a hole in the negative
    distribution that a tree finds immediately and that no real population would
    have. Reporting it as data means the finding survives without anyone having
    to re-derive a contingency table by hand.

    Only quantities with at most :data:`_MAX_DISCRETE_VALUES` distinct values
    are examined; a continuous column has empty bands for uninteresting reasons.
    """
    out: dict[str, Any] = {}
    for column in quantities.columns:
        values = quantities[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        distinct = np.unique(values[finite])
        if len(distinct) > _MAX_DISCRETE_VALUES or len(distinct) < 2:
            continue
        pos_counts = pd.Series(values[finite & (labels == 1)]).value_counts()
        neg_counts = pd.Series(values[finite & (labels == 0)]).value_counts()
        # Only *negative*-free values are reported. A value that only negatives
        # take is the ordinary situation at low prevalence and says nothing
        # about how the label was constructed.
        bands: list[dict[str, Any]] = []
        for value in distinct:
            n_pos = int(pos_counts.get(value, 0))
            n_neg = int(neg_counts.get(value, 0))
            if n_neg == 0 and n_pos >= _MIN_BAND_SIZE:
                bands.append({"value": float(value), "n_positive": n_pos, "n_negative": 0})
        if bands:
            out[str(column)] = {
                "negative_free_values": bands[:20],
                "n_positive_in_negative_free_values": sum(b["n_positive"] for b in bands),
                "note": (
                    "these values of this quantity contain no negative entity at "
                    "all; a label that can be read off a hole in the negative "
                    "distribution was written by the same process as the structure"
                ),
            }
    return out


def _combined_probe(
    quantities: pd.DataFrame, labels: np.ndarray, *, seed: int
) -> Optional[dict[str, Any]]:
    """Fit a small tree on the simple quantities and score a held-out half.

    Untuned on purpose.  The question is not "how well can we do", it is
    "does the label fall out of structure without any modelling effort".
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    n_pos = int(labels.sum())
    if n_pos < 8:
        return {
            "test_average_precision": None,
            "reason": (
                f"{n_pos} positives cannot be split into a train and a test half "
                f"with enough of each to estimate average precision"
            ),
        }

    # A quantity the dataset cannot support is all-NaN, and a quantity with a
    # single distinct value carries nothing. Both are dropped here rather than
    # imputed: a histogram binner cannot bin a column with fewer than two
    # distinct values, and imputing an absent quantity would invent evidence.
    usable, dropped = [], []
    for column in quantities.columns:
        values = quantities[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        (usable if finite.size and np.unique(finite).size > 1 else dropped).append(
            str(column)
        )
    if not usable:
        return {
            "test_average_precision": None,
            "reason": (
                "none of the simple quantities is defined and varying on this "
                "dataset, so there is nothing for a probe to learn from"
            ),
            "quantities_dropped": dropped,
        }

    index = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        index, test_size=0.5, stratify=labels, random_state=seed
    )
    values = quantities[usable].to_numpy(dtype=np.float64)
    y_train = labels[train_idx]
    n_train_pos = float(y_train.sum())
    n_train_neg = float(len(y_train) - n_train_pos)
    weights = np.where(y_train == 1, n_train_neg / max(n_train_pos, 1.0), 1.0)

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed
    )
    model.fit(values[train_idx], y_train, sample_weight=weights)
    scores = model.predict_proba(values[test_idx])[:, 1]
    y_test = labels[test_idx]
    return {
        "test_average_precision": float(average_precision_score(y_test, scores)),
        "test_roc_auc": float(roc_auc_score(y_test, scores)),
        "n_train": int(len(train_idx)),
        "n_train_positive": int(n_train_pos),
        "n_test": int(len(test_idx)),
        "n_test_positive": int(y_test.sum()),
        "quantities": usable,
        "quantities_dropped": dropped,
        "probe": (
            "untuned HistGradientBoostingClassifier on the simple quantities "
            "only; a stand-in for 'what falls out without modelling', not a "
            "candidate model"
        ),
    }


# ---------------------------------------------------------------------------
# Split feasibility
# ---------------------------------------------------------------------------


def _split_feasibility(dataset: ExperimentDataset, label_block: dict[str, Any]) -> dict[str, Any]:
    strategies: dict[str, Any] = {}
    n_pos = dataset.n_positive

    strategies["stratified_kfold"] = {
        "usable": bool(label_block["supports_cross_validation"]),
        "note": (
            f"{n_pos} positives over 4 folds leaves about {n_pos // 4} per test "
            f"partition"
            if n_pos
            else "no positives"
        ),
    }
    strategies["time_forward_holdout"] = {
        "usable": bool(label_block["supports_time_forward_evaluation"]),
        "note": (
            "a forward cut leaves enough positives in the later partition"
            if label_block["supports_time_forward_evaluation"]
            else "a forward cut leaves too few positives in the later partition "
            "for a defined average precision"
        ),
    }
    strategies["component_grouped_kfold"] = {
        "usable": bool(label_block["supports_cross_validation"]),
        "note": (
            "components are the grouping unit; usable whenever k-fold is, but "
            "fold sizes follow the component-size distribution and can be very "
            "uneven"
        ),
    }
    return strategies


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _verdict(
    dataset: ExperimentDataset,
    labels: dict[str, Any],
    triviality: dict[str, Any],
    domain: str,
) -> tuple[str, str, str, list[str]]:
    limitations: list[str] = []
    if dataset.spec.synthetic:
        limitations.append(
            "SYNTHETIC: rows were generated, not observed. No result on this "
            "dataset is evidence about real-world detection performance."
        )
    if labels.get("derived_from_transaction_labels"):
        limitations.append(
            "The entity label is derived from transaction flags, which makes a "
            "flagged entity's counterparty positive as well."
        )
    if labels.get("label_alias_columns_present"):
        limitations.append(
            f"The source file carries label-alias columns "
            f"{labels['label_alias_columns_present']}; they are excluded at load "
            f"time and must stay excluded."
        )

    n_pos = dataset.n_positive
    if n_pos == 0:
        return (
            BLOCKED,
            "the dataset has no positive entities",
            "none",
            limitations,
        )
    if n_pos == dataset.n_entities:
        return (
            BLOCKED,
            "every entity is positive; there is no negative class, so no "
            "classifier can be trained or evaluated",
            "none until a negative population is obtained",
            limitations,
        )

    if domain != DOMAIN_AML:
        return (
            STRUCTURAL_VALIDATION_ONLY,
            f"the declared label domain is {domain!r}, not money laundering; a "
            f"model ranked on it would be ranked for a different task, however "
            f"well it scores",
            "use it to check that the pipeline produces a non-degenerate result "
            "on a non-trivial label; never quote it as AML evidence",
            limitations
            + [
                f"DOMAIN MISMATCH: this dataset's positive class is {domain!r}."
            ],
        )

    if not labels["supports_cross_validation"]:
        return (
            REGRESSION_ONLY,
            f"{n_pos} positives is below the {MIN_POSITIVES_FOR_CV} needed for a "
            f"fold-level estimate; every metric would be determined by one or "
            f"two entities",
            "pin its numbers as a regression fixture; never quote them as "
            "detection capability",
            limitations,
        )

    verdict = triviality.get("verdict")
    probe = (triviality.get("combined_probe") or {}).get("test_average_precision")
    if verdict == "trivially_recoverable":
        return (
            STRUCTURAL_VALIDATION_ONLY,
            f"an untuned probe on simple structural quantities alone reaches "
            f"average precision {probe:.4f}; the labels are a function of "
            f"structure, so no experiment on this dataset can separate a good "
            f"model from a bad one",
            "exercise and scale-test the pipeline; do not rank models on it",
            limitations,
        )
    if verdict == "strong_structural_correlate":
        return (
            MODEL_SELECTION_READY,
            f"enough positives, and the structural probe reaches "
            f"{probe:.4f} - high, but with headroom a model can compete for",
            "usable for model comparison, with the structural floor reported "
            "beside every result",
            limitations
            + [
                "Structure alone already explains most of the label; report the "
                "label-free floor next to every supervised number."
            ],
        )
    return (
        MODEL_SELECTION_READY,
        f"enough positives and no trivial structural recovery "
        f"(probe average precision {probe:.4f})"
        if probe is not None
        else "enough positives and no trivial structural recovery",
        "usable for model comparison",
        limitations,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def audit_dataset(
    dataset: ExperimentDataset,
    *,
    domain: str = DOMAIN_UNKNOWN,
    label_alias_columns: Optional[list[str]] = None,
    seed: int = 42,
) -> DatasetAudit:
    """Run the full audit on ``dataset``.

    Parameters
    ----------
    domain:
        What the publisher's positive class actually is - :data:`DOMAIN_AML`,
        :data:`DOMAIN_CARD_FRAUD` or :data:`DOMAIN_UNKNOWN`.  A dataset can be
        statistically excellent and still be the wrong task; the verdict has to
        know which, and no measurement on the file can tell it.
    label_alias_columns:
        Names of columns *in the source file* that reproduce the label and are
        therefore deliberately not loaded.  Recorded so the exclusion is visible
        rather than tacit.
    seed:
        Seed for the triviality probe's split and fit.
    """
    aliases = list(label_alias_columns or [])
    logger.info("Auditing dataset %s (%d entities)", dataset.spec.name, dataset.n_entities)

    data = _data_block(dataset)
    labels = _label_block(dataset, aliases)
    triviality = _triviality_block(dataset, seed=seed)
    feasibility = _split_feasibility(dataset, labels)
    verdict, reason, recommended, limitations = _verdict(
        dataset, labels, triviality, domain
    )

    return DatasetAudit(
        name=dataset.spec.name,
        spec=dataset.spec.to_dict(),
        domain=domain,
        data=data,
        labels=labels,
        structural_triviality=triviality,
        split_feasibility=feasibility,
        verdict=verdict,
        reason=reason,
        recommended_use=recommended,
        limitations=limitations,
    )


def save_audit(audit: DatasetAudit, output_dir: Path | str) -> Path:
    """Write the audit as JSON and return the path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"audit_{audit.name}.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump(audit.to_dict(), fh, indent=2, default=str)
    logger.info("Wrote %s", target)
    return target
