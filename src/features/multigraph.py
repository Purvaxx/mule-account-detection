"""
Layer 4 — Directed-Multigraph Upgrade Features
================================================

Computes features that are ONLY meaningful because the authoritative graph is
a directed multigraph (``networkx.MultiDiGraph``) that preserves every
parallel transaction edge.  These features would be undefined, incorrect, or
collapsed away on a simple ``DiGraph``.

These features MUST NOT be included in the baseline feature set.

Directed-multigraph feature set (docs/architecture.md §12.3)
-------------------------------------------------------------
1.  ``out_repeated_counterparty_ratio``
        D_out / U_out
        where D_out = number of outgoing transaction edges (including
        repeated edges to the same target), U_out = number of unique
        target entities.
        = 1.0 for entities with no outgoing edges (no information).
        High values indicate that the entity repeatedly transacts to the
        same small set of counterparties.

2.  ``in_repeated_counterparty_ratio``
        D_in / U_in  (same logic for incoming edges)

3.  ``max_parallel_edge_count``
        Maximum number of transactions on any single directed entity pair
        (A → B) involving this entity (as source or target).

4.  ``mean_parallel_edge_count``
        Mean parallel-edge count across all directed entity pairs involving
        this entity.

5.  ``parallel_temporal_dispersion``
        For entity pairs with > 1 transaction, the mean of the standard
        deviations of transaction dates (in days) computed pair-by-pair.
        0.0 if no pair has > 1 transaction.
        NaN if the entity has no qualifying pairs.
        Measures how spread-out repeated transactions between the same pair are.

6.  ``reciprocal_out_count``
        Number of distinct target entities B such that both A→B and B→A
        edges exist.  Captures entities embedded in two-way flows.

7.  ``reciprocal_in_count``
        Number of distinct source entities B such that both A→B and B→A
        edges exist.  (Same set of entity pairs, different direction view.)

Notes
-----
- The authoritative ``MultiDiGraph`` is NEVER modified.
- Self-loops (A → A) are included in degree counts and edge tables as in the
  canonical graph.  They are correctly reflected in the features.
- Reciprocity is measured structurally (edge existence), NOT causally
  (temporal ordering).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MultigraphFeatures:
    """Container returned by :func:`compute_multigraph`.

    Attributes
    ----------
    features:
        DataFrame indexed by ``entity_id``.  One row per entity.
    entity_count:
        Number of entities in the feature matrix.
    feature_names:
        Ordered list of feature column names.
    nan_count:
        Total NaN values in the feature matrix.
    inf_count:
        Total ±∞ values in the feature matrix.
    """

    features: pd.DataFrame
    entity_count: int
    feature_names: list[str]
    nan_count: int
    inf_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_multigraph(
    graph: nx.MultiDiGraph,
    edge_table: pd.DataFrame,
) -> MultigraphFeatures:
    """Compute directed-multigraph upgrade features.

    Parameters
    ----------
    graph:
        Authoritative ``networkx.MultiDiGraph`` produced by Layer 3.
        **Not modified.**
    edge_table:
        Authoritative transaction-edge table produced by Layer 3.
        Required columns: ``source_entity``, ``target_entity``, ``date``.
        **Not modified.**

    Returns
    -------
    MultigraphFeatures
    """
    nodes: list[str] = sorted(graph.nodes())
    n = len(nodes)
    logger.info("Computing multigraph features for %d entities.", n)

    # ------------------------------------------------------------------
    # Pre-build per-directed-pair edge count and date lists
    # ------------------------------------------------------------------
    et = edge_table.copy()
    et["date"] = pd.to_datetime(et["date"])

    # Map (source, target) → list of timestamps
    pair_dates: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for _, row in et.iterrows():
        key = (row["source_entity"], row["target_entity"])
        pair_dates.setdefault(key, []).append(row["date"])

    # ------------------------------------------------------------------
    # Pre-build reciprocal pair set
    # ------------------------------------------------------------------
    all_directed_pairs: set[tuple[str, str]] = set(pair_dates.keys())
    reciprocal_pairs: set[tuple[str, str]] = {
        (a, b)
        for (a, b) in all_directed_pairs
        if (b, a) in all_directed_pairs
    }

    # ------------------------------------------------------------------
    # Per-entity computations
    # ------------------------------------------------------------------
    records: list[dict] = []

    for entity in nodes:
        # --- Outgoing ---
        out_neighbors_with_edges = [
            (entity, tgt)
            for tgt in graph.successors(entity)
        ]  # includes self-loops
        d_out = graph.out_degree(entity)   # total outgoing edges (multi)
        u_out = len(set(graph.successors(entity)))  # unique target nodes

        out_repeated_ratio = (float(d_out) / float(u_out)) if u_out > 0 else 1.0

        # --- Incoming ---
        d_in  = graph.in_degree(entity)
        u_in  = len(set(graph.predecessors(entity)))
        in_repeated_ratio = (float(d_in) / float(u_in)) if u_in > 0 else 1.0

        # --- Parallel-edge counts for pairs involving this entity ---
        involved_pairs = (
            [(entity, tgt) for tgt in graph.successors(entity)]
            + [(src, entity) for src in graph.predecessors(entity)]
        )
        # Deduplicate (self-loops appear in both lists)
        involved_pairs = list(set(involved_pairs))

        parallel_counts = [len(pair_dates.get(p, [])) for p in involved_pairs]
        if parallel_counts:
            max_parallel = float(max(parallel_counts))
            mean_parallel = float(np.mean(parallel_counts))
        else:
            max_parallel  = 0.0
            mean_parallel = 0.0

        # --- Parallel temporal dispersion ---
        # Only pairs with > 1 transaction contribute.
        dispersions: list[float] = []
        for pair in involved_pairs:
            dates_list = pair_dates.get(pair, [])
            if len(dates_list) >= 2:
                day_nums = np.array(
                    [(d - dates_list[0]).days for d in dates_list], dtype=float
                )
                dispersions.append(float(np.std(day_nums)))

        if dispersions:
            parallel_temporal_dispersion = float(np.mean(dispersions))
        elif parallel_counts:
            # Entity has pairs, but none have > 1 transaction.
            parallel_temporal_dispersion = 0.0
        else:
            parallel_temporal_dispersion = np.nan

        # --- Reciprocal behaviour ---
        # Count target entities B where both (entity→B) and (B→entity) exist.
        out_targets = set(graph.successors(entity))
        in_sources  = set(graph.predecessors(entity))

        reciprocal_out_count = float(len(out_targets & in_sources))   # B in both
        reciprocal_in_count  = reciprocal_out_count  # same set by definition

        records.append({
            "entity_id":                       entity,
            "out_repeated_counterparty_ratio": out_repeated_ratio,
            "in_repeated_counterparty_ratio":  in_repeated_ratio,
            "max_parallel_edge_count":         max_parallel,
            "mean_parallel_edge_count":        mean_parallel,
            "parallel_temporal_dispersion":    parallel_temporal_dispersion,
            "reciprocal_out_count":            reciprocal_out_count,
            "reciprocal_in_count":             reciprocal_in_count,
        })

    features = pd.DataFrame(records).set_index("entity_id")

    nan_count = int(features.isna().sum().sum())
    inf_count = int(np.isinf(features.fillna(0).values.astype(float)).sum())

    if nan_count:
        logger.warning("Multigraph features contain %d NaN value(s).", nan_count)

    logger.info(
        "Multigraph feature matrix: %d entities × %d features.",
        n, features.shape[1],
    )

    return MultigraphFeatures(
        features=features,
        entity_count=n,
        feature_names=list(features.columns),
        nan_count=nan_count,
        inf_count=inf_count,
    )
