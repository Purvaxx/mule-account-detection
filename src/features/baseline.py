"""
Layer 4 — Baseline Feature Engineering
========================================

Computes the static, graph-structural baseline feature set for every canonical
entity node.  These features use only information available in the authoritative
MultiDiGraph and the preprocessed edge table; they do NOT use temporal dynamics,
parallel-edge ratios, or any model label.

Baseline feature set (docs/architecture.md §12.1)
--------------------------------------------------
1.  ``entity_type_*``          One-hot encoding of the entity type
                               (Company / Person / Invalid).
2.  ``bank_country_*``         Multi-valued binary indicator over the frozen
                               151-code vocabulary in ``_KNOWN_COUNTRY_CODES``.
                               An entity with accounts in EE and LV has both
                               ``bank_country_EE`` and ``bank_country_LV`` set
                               to 1.  Codes are matched exactly as supplied; no
                               normalisation is applied here.
3.  ``log_in_degree``          log10(in_degree + 1)
4.  ``log_out_degree``         log10(out_degree + 1)
5.  ``log_total_incoming_usd`` log10(total_incoming_usd + 1)
6.  ``log_total_outgoing_usd`` log10(total_outgoing_usd + 1)
7.  ``flow_imbalance``         (incoming - outgoing) / (incoming + outgoing + ε)

Leakage protection
------------------
- ``core`` (payer_core / beneficiary_core) is NEVER used.  It is the target
  label for node classification.
- ``source_file``, ``investigation``, raw entity names, and account IDs are
  excluded.

Data integrity
--------------
The authoritative MultiDiGraph and the preprocessed DataFrame are never
modified.  All derived features are stored in a new DataFrame indexed by
``entity_id``.

Categorical vocabularies are frozen
-----------------------------------
``_KNOWN_ENTITY_TYPES`` and ``_KNOWN_COUNTRY_CODES`` are fixed constants that
reproduce the categories the Experiment 3 checkpoint was trained on.  They are
part of the model's contract, NOT a property of whatever dataset is being
processed, so the column set and its ordering are identical for every input.

The vocabulary holds both ISO codes (e.g. "EE", "LV") and the numeric or custom
codes present in the reference dataset (e.g. "33", ".D", "90").  Some of those
are encoding variants of the same country, but collapsing them now would change
the feature ordering and invalidate the checkpoint, so they are kept distinct.

Out-of-vocabulary policy is fixed:

- an unseen country code sets no column at all (no "other" bucket);
- an unseen entity type maps to ``entity_type_unknown``.

Resolving an external dataset's own notation *into* this vocabulary happens
upstream, in ``src/datasets/geography.py``, before features are computed.
Changing any of this requires a superseding decision record and a retrain
(see the frozen model contract in README.md).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Small constant to prevent division by zero in flow imbalance.
_EPSILON: float = 1e-9

#: Known entity type categories (in order).  Any unseen type is mapped to a
#: separate ``entity_type_unknown`` column rather than silently dropped.
_KNOWN_ENTITY_TYPES: tuple[str, ...] = ("Company", "Person", "Invalid")

_KNOWN_COUNTRY_CODES: tuple[str, ...] = (
    ".A",
    ".D",
    ".K",
    "/M",
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "60",
    "61",
    "62",
    "63",
    "64",
    "65",
    "66",
    "67",
    "68",
    "69",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "80",
    "81",
    "82",
    "83",
    "84",
    "85",
    "86",
    "87",
    "88",
    "89",
    "90",
    "92",
    "93",
    "94",
    "95",
    "96",
    "97",
    "99",
    "AE",
    "AT",
    "AZ",
    "BE",
    "BG",
    "CH",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "G8",
    "GB",
    "GE",
    "GR",
    "HU",
    "IL",
    "IT",
    "JH",
    "KZ",
    "LB",
    "LI",
    "LT",
    "LU",
    "LV",
    "MD",
    "ME",
    "MF",
    "MK",
    "MT",
    "NL",
    "NR",
    "OS",
    "PL",
    "PT",
    "QA",
    "R0",
    "RO",
    "RS",
    "SE",
    "SI",
    "SK",
    "T1",
    "TA",
    "TR",
    "TX",
    "TZ"
)




# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BaselineFeatures:
    """Container returned by :func:`compute_baseline`.

    Attributes
    ----------
    features:
        DataFrame indexed by ``entity_id``.  One row per entity.
        Columns are the feature names listed in the module docstring.
    entity_count:
        Number of entities in the feature matrix.
    feature_names:
        Ordered list of feature column names.
    country_codes:
        Sorted list of all distinct raw bank-country codes that appear in
        the dataset.  These correspond to the ``bank_country_*`` columns.
    entity_type_categories:
        Sorted list of entity-type categories that appear in the dataset.
    nan_count:
        Total number of NaN values in the feature matrix after construction.
    inf_count:
        Total number of ±∞ values in the feature matrix after construction.
    """

    features: pd.DataFrame
    entity_count: int
    feature_names: list[str]
    country_codes: list[str]
    entity_type_categories: list[str]
    nan_count: int
    inf_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_baseline(graph: nx.MultiDiGraph, edge_table: pd.DataFrame) -> BaselineFeatures:
    """Compute the static baseline feature matrix.

    Parameters
    ----------
    graph:
        Authoritative ``networkx.MultiDiGraph`` produced by Layer 3.
        **Not modified.**
    edge_table:
        Authoritative transaction-edge table produced by Layer 3
        (:attr:`~src.graph.builder.GraphResult.edge_table`).
        Must contain the columns ``source_entity``, ``target_entity``,
        ``amount_usd``, ``payer_bank_country``, ``beneficiary_bank_country``.
        **Not modified.**

    Returns
    -------
    BaselineFeatures
    """
    nodes: list[str] = sorted(graph.nodes())
    n = len(nodes)
    node_index = {e: i for i, e in enumerate(nodes)}
    logger.info("Computing baseline features for %d entities.", n)

    # ------------------------------------------------------------------
    # 1. Degree features (from the authoritative graph)
    # ------------------------------------------------------------------
    in_deg  = np.array([graph.in_degree(e)  for e in nodes], dtype=float)
    out_deg = np.array([graph.out_degree(e) for e in nodes], dtype=float)
    log_in_deg  = np.log10(in_deg  + 1.0)
    log_out_deg = np.log10(out_deg + 1.0)

    # ------------------------------------------------------------------
    # 2. Volume features (from the edge table)
    # ------------------------------------------------------------------
    # amount_usd is always the absolute USD equivalent (positive in current
    # dataset).  We do NOT touch amount_orig.
    inflow_series  = edge_table.groupby("target_entity")["amount_usd"].sum()
    outflow_series = edge_table.groupby("source_entity")["amount_usd"].sum()

    incoming_usd = np.array([inflow_series.get(e, 0.0)  for e in nodes], dtype=float)
    outgoing_usd = np.array([outflow_series.get(e, 0.0) for e in nodes], dtype=float)

    log_incoming = np.log10(incoming_usd + 1.0)
    log_outgoing = np.log10(outgoing_usd + 1.0)

    # ------------------------------------------------------------------
    # 3. Flow imbalance
    # ------------------------------------------------------------------
    flow_imbalance = (incoming_usd - outgoing_usd) / (incoming_usd + outgoing_usd + _EPSILON)

    # ------------------------------------------------------------------
    # 4. Entity type (one-hot)
    # ------------------------------------------------------------------
    # Read entity_type from node attributes.  Any node without a type gets
    # None and will be mapped to an "unknown" category.
    entity_types_raw = [graph.nodes[e].get("entity_type") for e in nodes]

    # Build a matrix: rows = entities, cols = entity type categories + unknown
    type_order = list(_KNOWN_ENTITY_TYPES)
    type_cols = [f"entity_type_{t}" for t in type_order] + ["entity_type_unknown"]
    type_matrix = np.zeros((n, len(type_cols)), dtype=float)
    type_to_col = {t: i for i, t in enumerate(type_order)}
    unknown_col = len(type_order)
    for i, t in enumerate(entity_types_raw):
        if t in type_to_col:
            type_matrix[i, type_to_col[t]] = 1.0
        else:
            type_matrix[i, unknown_col] = 1.0

    # ------------------------------------------------------------------
    # 5. Bank country (multi-valued / multi-hot)
    # ------------------------------------------------------------------
    # Collect all raw bank-country codes an entity is associated with via
    # its transactions.  We look at both payer_bank_country (when the entity
    # was the payer) and beneficiary_bank_country (when it was the beneficiary).
    # Raw values are preserved; no normalization is applied.

    entity_countries: dict[str, set[str]] = {e: set() for e in nodes}

    for _, row in edge_table.iterrows():
        src = row["source_entity"]
        tgt = row["target_entity"]
        pbc = row.get("payer_bank_country")
        bbc = row.get("beneficiary_bank_country")
        if src in entity_countries and pd.notna(pbc) and str(pbc).strip():
            entity_countries[src].add(str(pbc).strip())
        if tgt in entity_countries and pd.notna(bbc) and str(bbc).strip():
            entity_countries[tgt].add(str(bbc).strip())

    all_country_codes = list(_KNOWN_COUNTRY_CODES)
    country_to_col = {c: i for i, c in enumerate(all_country_codes)}
    country_cols = [f"bank_country_{c}" for c in all_country_codes]
    country_matrix = np.zeros((n, len(all_country_codes)), dtype=float)

    # Unseen countries must follow one explicit documented policy:
    # They are simply ignored, producing a vector of all zeros for that entity's country features.
    for i, e in enumerate(nodes):
        for c in entity_countries[e]:
            if c in country_to_col:
                country_matrix[i, country_to_col[c]] = 1.0

    # ------------------------------------------------------------------
    # Assemble the feature DataFrame
    # ------------------------------------------------------------------
    scalar_data = {
        "log_in_degree":          log_in_deg,
        "log_out_degree":         log_out_deg,
        "log_total_incoming_usd": log_incoming,
        "log_total_outgoing_usd": log_outgoing,
        "flow_imbalance":         flow_imbalance,
    }
    df_scalar  = pd.DataFrame(scalar_data, index=nodes)
    df_type    = pd.DataFrame(type_matrix,    columns=type_cols,    index=nodes)
    df_country = pd.DataFrame(country_matrix, columns=country_cols, index=nodes)

    # Order: structural scalars → entity type → bank country
    features = pd.concat([df_scalar, df_type, df_country], axis=1)
    features.index.name = "entity_id"

    # ------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------
    nan_count = int(features.isna().sum().sum())
    inf_count = int(np.isinf(features.values.astype(float)).sum())
    if nan_count:
        logger.warning("Baseline features contain %d NaN value(s).", nan_count)
    if inf_count:
        logger.warning("Baseline features contain %d ±inf value(s).", inf_count)

    logger.info(
        "Baseline feature matrix: %d entities × %d features.",
        n, features.shape[1],
    )

    return BaselineFeatures(
        features=features,
        entity_count=n,
        feature_names=list(features.columns),
        country_codes=all_country_codes,
        entity_type_categories=type_order,
        nan_count=nan_count,
        inf_count=inf_count,
    )
