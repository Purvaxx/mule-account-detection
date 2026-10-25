"""
Layer 6 - Versioned Feature Registry
======================================

Feature families are named, versioned and composable.  An experiment names the
families it wants; nothing in the framework knows about "167 features".

    legacy_exp3_v1   the frozen Experiment 3 representation (160 baseline + 7
                     directed-multigraph, with that experiment's exact
                     preprocessing).  Retained verbatim as the comparison
                     feature set, never as an invariant.
    baseline_v1      the 160-column baseline block alone
    multigraph_v1    the 7-column directed-multigraph block alone
    temporal_v1      the temporal block
    tga_v1           time-respecting cycle counts (the structure family)
    aml_core_v1      transaction / flow / network AML evidence, computed from
                     the edge table with no frozen vocabulary

Versioning rule
---------------
A family's *name* pins its semantics.  Changing what ``aml_core_v1`` computes
invalidates every artifact that recorded it, so a change means a new
``aml_core_v2`` registered alongside, not an edit in place.  The registry
records each family's declared version and column list so an artifact can be
checked against the code that claims to reproduce it.

Time scoping
------------
Every compute function receives a :class:`FeatureRequest` carrying the edge
rows it may use.  Under ``FeatureSpec.scope == "train_window"`` the runner
passes only the fold's training-window edges, so a feature cannot summarise
behaviour that had not happened yet.  Families that cannot honour a window say
so via ``time_scopable=False`` and the runner records that the guarantee does
not hold for them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureRequest",
    "FeatureSet",
    "FeatureMatrix",
    "register_feature_set",
    "available_feature_sets",
    "get_feature_set",
    "build_feature_matrix",
]

_EPS = 1e-9


@dataclass
class FeatureRequest:
    """What a feature family is given.

    Attributes
    ----------
    entity_ids:
        The rows to produce, in order.  A family must return a frame covering
        exactly these, in any order (the caller reindexes).
    edge_table:
        The transaction rows the family may use.  Already restricted to the
        fold's window when the experiment asked for that.
    graph:
        A MultiDiGraph built from exactly ``edge_table`` over exactly
        ``entity_ids``.
    as_of:
        The exclusive upper time bound the edges were restricted to, or
        ``None`` for the full history.
    attributes:
        Per-entity static attributes (entity type, bank country, ...).
    cache:
        Scratch space shared by every family computing against *this* request.
        Families that need the same expensive intermediate - the flattened
        event table, the deduplicated sparse adjacency - build it once and
        leave it here.  It is keyed by the request, so a later fold with a
        different time window gets a fresh one and cannot inherit a summary of
        transactions it is not allowed to see.
    """

    entity_ids: list[str]
    edge_table: Optional[pd.DataFrame]
    graph: Optional[Any]
    as_of: Optional[pd.Timestamp] = None
    attributes: Optional[pd.DataFrame] = None
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureSet:
    """A named, versioned feature family."""

    name: str
    version: str
    family: str  # transaction | flow | network | structure | legacy | composite
    description: str
    compute: Callable[[FeatureRequest], pd.DataFrame]
    time_scopable: bool = True
    requires_graph: bool = False
    requires_edges: bool = True

    def to_dict(self, columns: Optional[Sequence[str]] = None) -> dict[str, Any]:
        d = {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "time_scopable": self.time_scopable,
        }
        if columns is not None:
            d["n_columns"] = len(columns)
            d["columns"] = list(columns)
        return d


@dataclass
class FeatureMatrix:
    """The assembled matrix plus the provenance of every block in it."""

    values: pd.DataFrame
    sets: list[dict[str, Any]] = field(default_factory=list)
    scope: str = "global"
    as_of: Optional[pd.Timestamp] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "as_of": None if self.as_of is None else str(self.as_of),
            "n_columns": int(self.values.shape[1]),
            "sets": list(self.sets),
            "notes": list(self.notes),
        }


_REGISTRY: dict[str, FeatureSet] = {}
_BUILTINS_LOADED = False


def register_feature_set(feature_set: FeatureSet) -> FeatureSet:
    if feature_set.name in _REGISTRY:
        raise ValueError(f"Feature set {feature_set.name!r} is already registered.")
    _REGISTRY[feature_set.name] = feature_set
    return feature_set


def _load_builtin_feature_sets() -> None:
    """Import the modules that register the shipped families.

    Mirrors the lazy import :mod:`src.experiments.candidate` uses.  The legacy
    adapters above are defined in this module and register on import; the
    scalable AML families live in :mod:`src.experiments.aml_features`, which
    imports *from* here and therefore cannot be imported at the top of this
    file.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        __import__("src.experiments.aml_features")
    except ImportError as exc:  # pragma: no cover - only on a broken install
        logger.warning("AML feature families unavailable: %s", exc)


def available_feature_sets() -> list[str]:
    _load_builtin_feature_sets()
    return sorted(_REGISTRY)


def get_feature_set(name: str) -> FeatureSet:
    if name not in _REGISTRY:
        _load_builtin_feature_sets()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown feature set {name!r}. Available: {available_feature_sets()}"
        )
    return _REGISTRY[name]


def build_feature_matrix(request: FeatureRequest, names: Sequence[str]) -> FeatureMatrix:
    """Compute and concatenate the named families, in the given order.

    Column names are prefixed with the family name so two families may use the
    same short name without colliding, and so a coefficient or importance can
    always be traced back to the family that produced it.
    """
    blocks: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    notes: list[str] = []
    index = pd.Index(request.entity_ids, name="entity_id")

    for name in names:
        fs = get_feature_set(name)
        if fs.requires_graph and request.graph is None:
            raise ValueError(f"Feature set {name!r} requires a graph.")
        if fs.requires_edges and request.edge_table is None:
            raise ValueError(f"Feature set {name!r} requires an edge table.")
        if request.as_of is not None and not fs.time_scopable:
            notes.append(
                f"feature set {name!r} does not honour a time window; its columns "
                f"may summarise behaviour after {request.as_of}."
            )
        block = fs.compute(request)
        block = block.reindex(index)
        block = block.rename(columns={c: f"{name}::{c}" for c in block.columns})
        blocks.append(block)
        provenance.append(fs.to_dict(columns=list(block.columns)))

    if not blocks:
        values = pd.DataFrame(index=index)
    else:
        values = pd.concat(blocks, axis=1)

    n_nan = int(values.isna().to_numpy().sum()) if values.size else 0
    if n_nan:
        notes.append(
            f"{n_nan} NaN cells remained after computation and were left in "
            f"place; candidates are responsible for imputing them inside their "
            f"train-fitted preprocessing."
        )
    return FeatureMatrix(
        values=values,
        sets=provenance,
        as_of=request.as_of,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------
#
# These wrap the existing Layer 4 functions unchanged.  They exist so the
# historical representation stays runnable and comparable under the new
# contract; they are not a statement that it is the right representation.

def _legacy_baseline(request: FeatureRequest) -> pd.DataFrame:
    from src.features.baseline import compute_baseline

    return compute_baseline(request.graph, request.edge_table).features


def _legacy_multigraph_raw(request: FeatureRequest) -> pd.DataFrame:
    from src.features.multigraph import compute_multigraph

    return compute_multigraph(request.graph, request.edge_table).features


def _legacy_multigraph(request: FeatureRequest) -> pd.DataFrame:
    """Directed-multigraph block with Experiment 3's exact preprocessing.

    ``src.models.dataset.build_pyg_data`` fills NaN with -1.0 and applies
    ``log1p`` to defined (>= 0) values.  That transform is part of the frozen
    representation, so it is reproduced here rather than left to the candidate.
    """
    df = _legacy_multigraph_raw(request).reindex(pd.Index(request.entity_ids)).copy()
    df = df.fillna(-1.0)
    for col in df.columns:
        mask = df[col] >= 0
        df.loc[mask, col] = np.log1p(df.loc[mask, col])
    return df


def _legacy_exp3(request: FeatureRequest) -> pd.DataFrame:
    """The 167-column Experiment 3 representation: baseline then multigraph."""
    baseline = _legacy_baseline(request).reindex(pd.Index(request.entity_ids)).fillna(0.0)
    multigraph = _legacy_multigraph(request)
    return pd.concat([baseline, multigraph], axis=1)


def _legacy_temporal(request: FeatureRequest) -> pd.DataFrame:
    from src.features.temporal import compute_temporal

    result = compute_temporal(request.edge_table, list(request.entity_ids), as_of=request.as_of)
    df = result.features.reindex(pd.Index(request.entity_ids)).copy().fillna(-1.0)
    for col in ("activity_7d", "activity_30d", "mean_iat"):
        if col in df.columns:
            mask = df[col] >= 0
            df.loc[mask, col] = np.log1p(df.loc[mask, col])
    return df


def _legacy_tga(request: FeatureRequest) -> pd.DataFrame:
    from src.features.tda import compute_tga

    df = compute_tga(request.graph, request.edge_table).features
    df = df.reindex(pd.Index(request.entity_ids)).copy().fillna(-1.0)
    for col in df.columns:
        df[col] = df[col].astype(np.float64)
        mask = df[col] >= 0
        df.loc[mask, col] = np.log1p(df.loc[mask, col])
    return df


register_feature_set(
    FeatureSet(
        name="baseline_v1",
        version="1.0",
        family="legacy",
        description=(
            "Experiment 1 baseline block: entity type one-hot, frozen 151-code "
            "bank-country indicator, log degrees, log USD volumes, flow imbalance."
        ),
        compute=_legacy_baseline,
        requires_graph=True,
    )
)

register_feature_set(
    FeatureSet(
        name="multigraph_v1",
        version="1.0",
        family="legacy",
        description=(
            "Experiment 3 directed-multigraph block with its frozen "
            "fillna(-1)/log1p preprocessing."
        ),
        compute=_legacy_multigraph,
        requires_graph=True,
    )
)

register_feature_set(
    FeatureSet(
        name="legacy_exp3_v1",
        version="1.0",
        family="legacy",
        description=(
            "The complete 167-column Experiment 3 representation. Comparison "
            "feature set for the frozen checkpoint; not a required input format."
        ),
        compute=_legacy_exp3,
        requires_graph=True,
    )
)

register_feature_set(
    FeatureSet(
        name="temporal_v1",
        version="1.0",
        family="transaction",
        description="Experiment 2 temporal block (rolling activity, IAT, burstiness).",
        compute=_legacy_temporal,
    )
)

register_feature_set(
    FeatureSet(
        name="tga_v1",
        version="1.0",
        family="structure",
        description=(
            "Time-respecting directed-cycle counts on the 1-hop ego network "
            "(reciprocal count, cycle count, earliest completion)."
        ),
        compute=_legacy_tga,
        requires_graph=True,
    )
)


# ---------------------------------------------------------------------------
# aml_core_v1
# ---------------------------------------------------------------------------

def _aml_core(request: FeatureRequest) -> pd.DataFrame:
    """Transaction, flow and network AML evidence computed from the edge table.

    Nothing here is derived from a frozen vocabulary or from the label.  Every
    column is a summary of transactions the entity actually took part in, so
    the block honours a time window exactly.

    Columns
    -------
    transaction
        ``txn_in``, ``txn_out``, ``txn_total``, ``amount_in_sum``,
        ``amount_out_sum``, ``amount_in_mean``, ``amount_out_mean``,
        ``amount_max``, ``amount_std``, ``active_days``, ``span_days``,
        ``txn_per_active_day``, ``mean_iat_days``, ``std_iat_days``,
        ``burstiness``
    flow
        ``fan_in``, ``fan_out``, ``inflow_outflow_ratio``, ``flow_imbalance``,
        ``pass_through_ratio``, ``min_in_to_out_gap_days``,
        ``same_day_in_out``
    network
        ``degree_total``, ``counterparty_count``, ``counterparty_diversity``,
        ``reciprocal_partners``, ``reciprocal_ratio``, ``clustering``,
        ``degree_centrality``

    Undefined values
    ----------------
    A statistic that has no meaning for an entity - inter-arrival time with a
    single transaction, a ratio with an empty denominator - is left as ``NaN``.
    It is not replaced with 0, which would be indistinguishable from a real
    zero, nor with a sentinel, which would be a magic number a tree model
    happily splits on.  Imputation is the candidate's train-fitted business.
    """
    entity_ids = list(request.entity_ids)
    index = pd.Index(entity_ids, name="entity_id")
    columns = [
        "txn_in", "txn_out", "txn_total",
        "amount_in_sum", "amount_out_sum", "amount_in_mean", "amount_out_mean",
        "amount_max", "amount_std",
        "active_days", "span_days", "txn_per_active_day",
        "mean_iat_days", "std_iat_days", "burstiness",
        "fan_in", "fan_out", "inflow_outflow_ratio", "flow_imbalance",
        "pass_through_ratio", "min_in_to_out_gap_days", "same_day_in_out",
        "degree_total", "counterparty_count", "counterparty_diversity",
        "reciprocal_partners", "reciprocal_ratio", "clustering", "degree_centrality",
    ]
    out = pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float64)

    et = request.edge_table
    if et is None or len(et) == 0:
        return out

    df = pd.DataFrame(
        {
            "src": et["source_entity"].astype(str),
            "dst": et["target_entity"].astype(str),
            "amount": pd.to_numeric(et.get("amount_usd"), errors="coerce").abs()
            if "amount_usd" in et.columns
            else np.nan,
            "date": pd.to_datetime(et["date"], errors="coerce")
            if "date" in et.columns
            else pd.NaT,
        }
    )

    # --- counts and amounts, per direction ---------------------------------
    in_grp = df.groupby("dst")
    out_grp = df.groupby("src")

    out["txn_in"] = in_grp.size().reindex(index).fillna(0.0)
    out["txn_out"] = out_grp.size().reindex(index).fillna(0.0)
    out["txn_total"] = out["txn_in"] + out["txn_out"]

    out["amount_in_sum"] = in_grp["amount"].sum(min_count=1).reindex(index)
    out["amount_out_sum"] = out_grp["amount"].sum(min_count=1).reindex(index)
    out["amount_in_mean"] = in_grp["amount"].mean().reindex(index)
    out["amount_out_mean"] = out_grp["amount"].mean().reindex(index)

    # --- per-entity event table (both directions) --------------------------
    events = pd.concat(
        [
            pd.DataFrame(
                {
                    "entity": df["src"],
                    "counterparty": df["dst"],
                    "amount": df["amount"],
                    "date": df["date"],
                    "direction": "out",
                }
            ),
            pd.DataFrame(
                {
                    "entity": df["dst"],
                    "counterparty": df["src"],
                    "amount": df["amount"],
                    "date": df["date"],
                    "direction": "in",
                }
            ),
        ],
        ignore_index=True,
    )
    ev_grp = events.groupby("entity")

    out["amount_max"] = ev_grp["amount"].max().reindex(index)
    out["amount_std"] = ev_grp["amount"].std(ddof=0).reindex(index)

    dated = events.dropna(subset=["date"])
    if not dated.empty:
        dated_grp = dated.groupby("entity")
        out["active_days"] = (
            dated_grp["date"].nunique().reindex(index).astype(np.float64)
        )
        span = (dated_grp["date"].max() - dated_grp["date"].min()).dt.days
        out["span_days"] = span.reindex(index).astype(np.float64)

        counts = dated_grp.size().reindex(index).astype(np.float64)
        active = out["active_days"]
        out["txn_per_active_day"] = counts.where(active > 0) / active.where(active > 0)

        # Inter-arrival statistics, day resolution.  Undefined with <2 events.
        ordered = dated.sort_values(["entity", "date"])
        gaps = ordered.groupby("entity")["date"].diff().dt.total_seconds() / 86400.0
        ordered = ordered.assign(gap=gaps)
        gap_grp = ordered.dropna(subset=["gap"]).groupby("entity")["gap"]
        mean_iat = gap_grp.mean().reindex(index)
        std_iat = gap_grp.std(ddof=0).reindex(index)
        out["mean_iat_days"] = mean_iat
        out["std_iat_days"] = std_iat
        # Burstiness coefficient (Goh & Barabasi): (sigma - mu) / (sigma + mu),
        # -1 for perfectly regular timing, +1 for maximally bursty.
        denom = std_iat + mean_iat
        out["burstiness"] = ((std_iat - mean_iat) / denom).where(denom.abs() > _EPS)

    # --- flow ---------------------------------------------------------------
    out["fan_in"] = in_grp["src"].nunique().reindex(index).fillna(0.0)
    out["fan_out"] = out_grp["dst"].nunique().reindex(index).fillna(0.0)

    in_sum = out["amount_in_sum"]
    out_sum = out["amount_out_sum"]
    both = in_sum.notna() | out_sum.notna()
    i = in_sum.fillna(0.0)
    o = out_sum.fillna(0.0)
    total = i + o
    out["inflow_outflow_ratio"] = (i / (o + _EPS)).where(both)
    out["flow_imbalance"] = ((i - o) / (total + _EPS)).where(both & (total > 0))
    out["pass_through_ratio"] = (
        np.minimum(i, o) / (np.maximum(i, o) + _EPS)
    ).where(both & (total > 0))

    # Rapid inflow -> outflow: the shortest delay between money arriving and
    # money leaving.  Zero means same-day turnaround.
    if not dated.empty:
        ins = dated[dated["direction"] == "in"][["entity", "date"]]
        outs = dated[dated["direction"] == "out"][["entity", "date"]]
        if not ins.empty and not outs.empty:
            merged = ins.merge(outs, on="entity", suffixes=("_in", "_out"))
            merged = merged[merged["date_out"] >= merged["date_in"]]
            if not merged.empty:
                gap = (merged["date_out"] - merged["date_in"]).dt.days
                merged = merged.assign(gap=gap)
                out["min_in_to_out_gap_days"] = (
                    merged.groupby("entity")["gap"].min().reindex(index).astype(np.float64)
                )
                out["same_day_in_out"] = (
                    merged.assign(same=(merged["gap"] == 0).astype(np.float64))
                    .groupby("entity")["same"]
                    .sum()
                    .reindex(index)
                    .fillna(0.0)
                )
        if "same_day_in_out" in out.columns:
            out["same_day_in_out"] = out["same_day_in_out"].fillna(0.0)

    # --- network ------------------------------------------------------------
    counterparties = ev_grp["counterparty"].nunique().reindex(index).fillna(0.0)
    out["counterparty_count"] = counterparties
    out["degree_total"] = out["txn_total"]
    out["counterparty_diversity"] = (
        counterparties / out["txn_total"]
    ).where(out["txn_total"] > 0)

    pairs_out = set(zip(df["src"], df["dst"]))
    reciprocal_by_entity: dict[str, set[str]] = {}
    for u, v in pairs_out:
        if (v, u) in pairs_out and u != v:
            reciprocal_by_entity.setdefault(u, set()).add(v)
    recip = pd.Series(
        {k: float(len(v)) for k, v in reciprocal_by_entity.items()}, dtype=np.float64
    )
    out["reciprocal_partners"] = recip.reindex(index).fillna(0.0)
    out["reciprocal_ratio"] = (
        out["reciprocal_partners"] / counterparties
    ).where(counterparties > 0)

    if request.graph is not None:
        import networkx as nx

        simple = nx.Graph()
        simple.add_nodes_from(entity_ids)
        simple.add_edges_from(pairs_out)
        simple.remove_edges_from(nx.selfloop_edges(simple))
        clustering = pd.Series(nx.clustering(simple), dtype=np.float64)
        out["clustering"] = clustering.reindex(index)
        n_nodes = simple.number_of_nodes()
        if n_nodes > 1:
            degree = pd.Series(dict(simple.degree()), dtype=np.float64)
            out["degree_centrality"] = (degree / (n_nodes - 1)).reindex(index)

    return out


register_feature_set(
    FeatureSet(
        name="aml_core_v1",
        version="1.0",
        family="composite",
        description=(
            "Transaction, flow and network AML evidence computed directly from "
            "the edge table: velocity, burstiness, fan-in/fan-out, flow "
            "imbalance, pass-through behaviour, inflow-to-outflow latency, "
            "counterparty diversity, reciprocity, local density."
        ),
        compute=_aml_core,
        requires_graph=False,
    )
)
