"""
Layer 4 — Temporal Upgrade Features
======================================

Computes the temporal behavioural feature set for every canonical entity.
These features MUST NOT be mixed into the baseline; they are a separate
experimental upgrade layer.

Temporal feature set (docs/architecture.md §12.2)
--------------------------------------------------
1.  ``activity_7d``         Number of transactions involving this entity in
                             the trailing 7-day window ending on `as_of`.
2.  ``activity_30d``        Same for the trailing 30-day window.
3.  ``mean_iat``            Mean inter-transaction time (days) across all
                             transactions involving this entity, ordered by
                             date.
4.  ``burstiness``          Coefficient of variation of inter-transaction
                             times: σ(Δt) / μ(Δt).  Undefined (NaN) when
                             there are fewer than two transactions.
5.  ``active_day_ratio``    Distinct active dates / total calendar days in
                             the entity's activity span [first_date, last_date].
                             Returns 1.0 when span == 0 (single-day entity).
6.  ``same_day_cooccurrence`` Binary (0/1).  1 when the entity has at least
                             one calendar date on which it appears in BOTH a
                             payer role and a beneficiary role.

Day-level precision constraint (docs/architecture.md §13)
----------------------------------------------------------
All timestamps in the dataset carry day-level precision only (time component
is always midnight).  Therefore:

- ``same_day_cooccurrence`` measures whether the entity sent AND received
  funds on the same **calendar date**.  It does NOT imply that the inflow
  causally preceded the outflow on that date, and it does NOT imply
  pass-through behaviour.
- Intra-day ordering is NOT available and must NOT be assumed.

Two computation modes
---------------------
``as_of=None`` (static / retrospective)
    Features are computed over the full historical transaction set.  This is
    valid for transductive node classification over a frozen dataset (e.g.
    retrospective AML investigation).  Global graph statistics computed in
    this mode may include future information relative to early transactions.

``as_of=<date>`` (causal / temporal prediction)
    Features are computed using only transactions with ``date <= as_of``.
    No future transactions influence the result.  This mode is required when
    building training data for a causal temporal prediction task.

Leakage protection
------------------
- ``core`` / ``payer_core`` / ``beneficiary_core`` are never used.
- ``source_file``, ``investigation``, raw names, and account IDs are excluded.

Data integrity
--------------
The authoritative graph and edge table are never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TemporalFeatures:
    """Container returned by :func:`compute_temporal`.

    Attributes
    ----------
    features:
        DataFrame indexed by ``entity_id``.  One row per entity.
    entity_count:
        Number of entities in the feature matrix.
    feature_names:
        Ordered list of feature column names.
    as_of:
        The cut-off date used.  ``None`` means the full historical set.
    nan_count:
        Total NaN values in the feature matrix.
    inf_count:
        Total ±∞ values in the feature matrix.
    """

    features: pd.DataFrame
    entity_count: int
    feature_names: list[str]
    as_of: Optional[pd.Timestamp]
    nan_count: int
    inf_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_temporal(
    edge_table: pd.DataFrame,
    entity_ids: list[str],
    *,
    as_of: Optional[pd.Timestamp] = None,
) -> TemporalFeatures:
    """Compute temporal upgrade features for every canonical entity.

    Parameters
    ----------
    edge_table:
        Authoritative transaction-edge table produced by Layer 3.
        Required columns: ``source_entity``, ``target_entity``, ``date``.
        **Not modified.**
    entity_ids:
        Ordered list of all canonical entity IDs (the node list of the graph).
    as_of:
        Cut-off date (inclusive) for causal temporal prediction mode.
        ``None`` (default) selects static / retrospective mode using the
        entire historical transaction set.

    Returns
    -------
    TemporalFeatures
    """
    logger.info(
        "Computing temporal features for %d entities (as_of=%s).",
        len(entity_ids), as_of,
    )

    # ------------------------------------------------------------------
    # Apply temporal cut-off (causal mode)
    # ------------------------------------------------------------------
    et = edge_table.copy()
    et["date"] = pd.to_datetime(et["date"])

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        et = et[et["date"] <= as_of_ts]
        logger.debug(
            "Causal mode: %d transactions retained (cut-off %s).",
            len(et), as_of_ts.date(),
        )
    else:
        as_of_ts = et["date"].max() if len(et) > 0 else pd.Timestamp("1970-01-01")
        logger.debug("Static/retrospective mode: using full dataset (%d tx).", len(et))

    # ------------------------------------------------------------------
    # Build per-entity transaction date lists
    # ------------------------------------------------------------------
    # An entity is involved in a transaction either as payer (source_entity)
    # or as beneficiary (target_entity).  We combine both roles.

    payer_dates = et.groupby("source_entity")["date"].apply(list)
    benef_dates = et.groupby("target_entity")["date"].apply(list)

    def _all_dates(entity: str) -> list[pd.Timestamp]:
        p = payer_dates.get(entity, [])
        b = benef_dates.get(entity, [])
        return sorted(p + b)

    def _payer_dates(entity: str) -> set[pd.Timestamp]:
        return set(payer_dates.get(entity, []))

    def _benef_dates(entity: str) -> set[pd.Timestamp]:
        return set(benef_dates.get(entity, []))

    # ------------------------------------------------------------------
    # Compute rolling-window activity
    # ------------------------------------------------------------------
    cutoff_7d  = as_of_ts - pd.Timedelta(days=7)
    cutoff_30d = as_of_ts - pd.Timedelta(days=30)

    # Precompute: total transactions in last 7 / 30 days per entity
    # (as payer OR beneficiary)
    src_7d  = et[et["date"] > cutoff_7d ].groupby("source_entity").size()
    tgt_7d  = et[et["date"] > cutoff_7d ].groupby("target_entity").size()
    src_30d = et[et["date"] > cutoff_30d].groupby("source_entity").size()
    tgt_30d = et[et["date"] > cutoff_30d].groupby("target_entity").size()

    # ------------------------------------------------------------------
    # Main loop: compute all features per entity
    # ------------------------------------------------------------------
    records: list[dict] = []

    for entity in entity_ids:
        dates = _all_dates(entity)
        n_tx = len(dates)

        # 1 & 2 — Rolling activity windows
        activity_7d  = int(src_7d.get(entity, 0)  + tgt_7d.get(entity, 0))
        activity_30d = int(src_30d.get(entity, 0) + tgt_30d.get(entity, 0))

        # 3 & 4 — Inter-transaction time
        if n_tx >= 2:
            diffs_days = np.array([
                (dates[i] - dates[i - 1]).days for i in range(1, n_tx)
            ], dtype=float)
            mean_iat = float(np.mean(diffs_days))
            std_iat  = float(np.std(diffs_days))
            burstiness = std_iat / mean_iat if mean_iat > 0 else np.nan
        else:
            mean_iat   = np.nan
            burstiness = np.nan

        # 5 — Active-day ratio
        if n_tx == 0:
            active_day_ratio = np.nan
        else:
            unique_dates = {d.date() for d in dates}
            first_date = min(unique_dates)
            last_date  = max(unique_dates)
            span_days  = (last_date - first_date).days
            active_day_ratio = (
                len(unique_dates) / (span_days + 1)  # +1 to include both endpoints
                if span_days >= 0
                else 1.0
            )

        # 6 — Same-day inflow/outflow co-occurrence
        # Strictly: does this entity have at least one calendar date on which
        # it appears in BOTH a payer and a beneficiary role?
        # This does NOT imply causality or pass-through.
        payer_day_set = {d.date() for d in _payer_dates(entity)}
        benef_day_set = {d.date() for d in _benef_dates(entity)}
        same_day_cooc = 1.0 if (payer_day_set & benef_day_set) else 0.0

        records.append({
            "entity_id":            entity,
            "activity_7d":          float(activity_7d),
            "activity_30d":         float(activity_30d),
            "mean_iat":             mean_iat,
            "burstiness":           burstiness,
            "active_day_ratio":     active_day_ratio,
            "same_day_cooccurrence": same_day_cooc,
        })

    features = pd.DataFrame(records).set_index("entity_id")

    nan_count = int(features.isna().sum().sum())
    inf_count = int(np.isinf(features.fillna(0).values.astype(float)).sum())

    if nan_count:
        logger.info(
            "Temporal features contain %d NaN value(s) "
            "(expected: entities with < 2 transactions have NaN burstiness/IAT).",
            nan_count,
        )

    logger.info(
        "Temporal feature matrix: %d entities × %d features.",
        len(entity_ids), features.shape[1],
    )

    return TemporalFeatures(
        features=features,
        entity_count=len(entity_ids),
        feature_names=list(features.columns),
        as_of=None if as_of is None else pd.Timestamp(as_of),
        nan_count=nan_count,
        inf_count=inf_count,
    )
