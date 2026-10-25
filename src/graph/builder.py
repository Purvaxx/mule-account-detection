"""
Layer 3 — Graph Construction
=============================

Constructs the authoritative transaction graph from the validated output of
Layer 2 (:mod:`src.data.preprocessing`).

Canonical Graph Model
---------------------
- **Graph type:** ``networkx.MultiDiGraph`` (directed multigraph with
  self-loop support).
- **Nodes:** One node per canonical entity (``Entity_ID = name_norm``).
  Current expected count: 3,440.
- **Edges:** One directed edge per preprocessed transaction
  (``payer_entity_id → beneficiary_entity_id``).
  Current expected count: 16,940.
- **Edge key:** Unique transaction ``id``.  Because transaction IDs are
  guaranteed unique by the ingestion layer, using them as multigraph edge
  keys gives O(1) edge look-up by transaction ID without hash collisions.

WHY MultiDiGraph?
-----------------
The audit of the preprocessed dataset established that:
- 4,024 distinct directed entity pairs carry 16,940 transactions.
- 1,796 pairs carry more than one transaction (max 437).
- 69 transactions are self-loops (entity → itself).
- 148 undirected reciprocal pairs exist (A→B and B→A).
A standard ``DiGraph`` would collapse 12,916 transactions (76 %) into a
single edge per pair.  A ``MultiDiGraph`` preserves all 16,940 edges.

Node Attributes
---------------
Node attributes are determined **deterministically** from transaction data
using the following rules:

``core``
    Whether the entity is a "core" node in the AML network.  Derived from
    ``payer_core`` (when the entity appears as payer) and ``beneficiary_core``
    (when it appears as beneficiary).  The read-only audit confirmed that
    ``core`` is perfectly consistent per ``name_norm`` (0 conflicts).
    This layer re-validates this and reports any conflict found.

``entity_type``
    The ``payer_type`` / ``beneficiary_type`` value for this entity.
    Company / Person / Invalid.  Validated for consistency.

``jurisdiction``
    The ``payer_jurisdiction`` / ``beneficiary_jurisdiction`` value.
    Validated for consistency.

``bank_country``
    The ``payer_bank_country`` / ``beneficiary_bank_country`` value.
    Validated for consistency.  Missing values (NaN) are stored as ``None``.

NOT added at this layer (handled by later layers):
    - degree, in-degree, out-degree
    - PageRank, centrality
    - velocity, rolling-window features
    - embeddings, risk scores

Edge Attributes
---------------
Every edge carries the full per-transaction information:
    - ``transaction_id``
    - ``amount_orig`` (signed; preserved exactly)
    - ``amount_orig_currency``
    - ``amount_usd`` (preserved exactly)
    - ``amount_eur``
    - ``date`` (day-level only; no intra-day ordering)
    - ``purpose``
    - ``payer_account_used``
    - ``beneficiary_account_used``
    - ``payer_type``
    - ``beneficiary_type``
    - ``payer_jurisdiction``
    - ``beneficiary_jurisdiction``
    - ``payer_bank_country``
    - ``beneficiary_bank_country``
    - ``payer_core``
    - ``beneficiary_core``
    - ``source_file``
    - ``investigation``

Temporal semantics
------------------
``date`` is stored at day-level precision (a ``pandas.Timestamp`` with the
time component at midnight).  No intra-day ordering is invented.  Temporal
feature engineering belongs to a later layer.

Amount semantics
----------------
``amount_orig`` is signed (negative = debit from payer's ledger) and is
preserved exactly as produced by the preprocessing layer.  It is NEVER
converted to an unsigned value.

Transaction-edge table
----------------------
The :attr:`GraphResult.edge_table` is the authoritative row-for-row record
of which transaction maps to which graph edge.  The graph object is a
computational representation derived from it; the edge table is the source
of truth.

Not implemented here
--------------------
- GNN / PyTorch Geometric conversion      → src/models/
- Cycle / subgraph detection              → src/detection/
- Temporal feature engineering            → src/features/
- Temporal Graph Topology (TGA)           → src/features/tda.py
- Risk scoring / explainability           → src/detection/ / src/evaluation/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd

from src.data.preprocessing import PreprocessingResult, load_and_preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-attribute column groups
# ---------------------------------------------------------------------------

# For each canonical attribute we try to read it from either the payer side
# or the beneficiary side of the transaction DataFrame.
_NODE_ATTR_SOURCES: dict[str, tuple[str, str]] = {
    # graph attribute → (payer column,      beneficiary column)
    "core":         ("payer_core",         "beneficiary_core"),
    "entity_type":  ("payer_type",          "beneficiary_type"),
    "jurisdiction": ("payer_jurisdiction",  "beneficiary_jurisdiction"),
    "bank_country": ("payer_bank_country",  "beneficiary_bank_country"),
}

# ---------------------------------------------------------------------------
# Edge attribute mapping: preprocessed column → edge attribute name
# ---------------------------------------------------------------------------

_EDGE_ATTR_MAP: dict[str, str] = {
    # transaction identity
    "id":                        "transaction_id",
    # amounts (preserved exactly; amount_orig is signed)
    "amount_orig":               "amount_orig",
    "amount_orig_currency":      "amount_orig_currency",
    "amount_usd":                "amount_usd",
    "amount_eur":                "amount_eur",
    # temporal (day-level only; no intra-day ordering)
    "date":                      "date",
    # narrative
    "purpose":                   "purpose",
    # account metadata (not node identifiers; transaction-level only)
    "payer_account":             "payer_account_used",
    "beneficiary_account":       "beneficiary_account_used",
    # entity categorical attributes (denormalized per-edge for convenience)
    "payer_type":                "payer_type",
    "beneficiary_type":          "beneficiary_type",
    "payer_jurisdiction":        "payer_jurisdiction",
    "beneficiary_jurisdiction":  "beneficiary_jurisdiction",
    "payer_bank_country":        "payer_bank_country",
    "beneficiary_bank_country":  "beneficiary_bank_country",
    "payer_core":                "payer_core",
    "beneficiary_core":          "beneficiary_core",
    # provenance
    "source_file":               "source_file",
    "investigation":             "investigation",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GraphResult:
    """Container returned by :func:`build_graph`.

    Attributes
    ----------
    graph:
        The constructed ``networkx.MultiDiGraph``.
    edge_table:
        The authoritative transaction-edge table.  One row per transaction.
        Columns: ``source_entity``, ``target_entity``, ``edge_key``, plus
        all edge attributes.  This is the source of truth; the graph object
        is a computational representation derived from it.
    node_count:
        Number of nodes in the graph.
    edge_count:
        Number of edges in the graph.
    node_conflict_report:
        Mapping of ``entity_id → {attribute → (value_a, value_b)}`` for
        any entity whose attribute was not consistent across its transactions.
        An empty dict means no conflicts were found.
    validation_warnings:
        Non-fatal observations.
    validation_errors:
        Fatal issues.  When non-empty the graph must not be used for
        downstream processing.
    """

    graph: nx.MultiDiGraph
    edge_table: pd.DataFrame
    node_count: int
    edge_count: int
    node_conflict_report: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no fatal validation errors were detected."""
        return len(self.validation_errors) == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nan_to_none(value: Any) -> Any:
    """Convert pandas NA / NaN to Python ``None``; pass everything else."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _resolve_node_attributes(
    df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Determine canonical node attributes and detect conflicts.

    Parameters
    ----------
    df:
        Preprocessed transaction DataFrame containing ``payer_entity_id``,
        ``beneficiary_entity_id``, and all payer/beneficiary attribute columns.

    Returns
    -------
    node_attrs:
        ``{entity_id: {attr: value}}`` — canonical attribute for each entity.
    conflicts:
        ``{entity_id: {attr: (val_a, val_b)}}`` — attribute pairs that were
        not consistent.  Empty dict if no conflicts found.
    """
    # Collect all (entity_id, attr, value) triples from both sides
    records_payer = df[
        ["payer_entity_id"] + [col for col, _ in _NODE_ATTR_SOURCES.values()]
    ].rename(columns={"payer_entity_id": "entity_id"})

    records_ben = df[
        ["beneficiary_entity_id"] + [col for _, col in _NODE_ATTR_SOURCES.values()]
    ].rename(columns={"beneficiary_entity_id": "entity_id"})

    # Rename so both DataFrames use the attribute-name keys
    payer_rename = {v[0]: k for k, v in _NODE_ATTR_SOURCES.items()}
    ben_rename = {v[1]: k for k, v in _NODE_ATTR_SOURCES.items()}

    records_payer = records_payer.rename(columns=payer_rename)
    records_ben = records_ben.rename(columns=ben_rename)

    combined = pd.concat([records_payer, records_ben], ignore_index=True)

    node_attrs: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, dict[str, Any]] = {}

    for entity_id, group in combined.groupby("entity_id"):
        attrs: dict[str, Any] = {}
        entity_conflicts: dict[str, Any] = {}

        for attr in _NODE_ATTR_SOURCES:
            unique_vals = group[attr].dropna().unique()
            if len(unique_vals) == 0:
                # All values are NaN; store None
                attrs[attr] = None
            elif len(unique_vals) == 1:
                attrs[attr] = _nan_to_none(unique_vals[0])
            else:
                # Conflict — report both; use the first non-null value
                entity_conflicts[attr] = tuple(unique_vals.tolist())
                attrs[attr] = _nan_to_none(unique_vals[0])

        attrs["entity_id"] = entity_id
        node_attrs[entity_id] = attrs
        if entity_conflicts:
            conflicts[entity_id] = entity_conflicts

    return node_attrs, conflicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(preprocessing_result: PreprocessingResult) -> GraphResult:
    """Construct the authoritative ``networkx.MultiDiGraph`` from preprocessed
    transactions.

    Parameters
    ----------
    preprocessing_result:
        A valid :class:`~src.data.preprocessing.PreprocessingResult` returned
        by :func:`~src.data.preprocessing.preprocess`.

    Returns
    -------
    GraphResult
        Always returns a result.  Callers must inspect ``result.is_valid``
        before passing the graph to downstream layers.

    Raises
    ------
    ValueError
        If ``preprocessing_result`` is invalid.
    """
    if not preprocessing_result.is_valid:
        raise ValueError(
            "Cannot build graph from an invalid PreprocessingResult. "
            f"Errors: {preprocessing_result.validation_errors}"
        )

    df = preprocessing_result.transactions
    warnings: list[str] = []
    errors: list[str] = []

    logger.info(
        "Building MultiDiGraph from %d transactions and %d entities.",
        len(df),
        preprocessing_result.entity_count,
    )

    # ------------------------------------------------------------------
    # 1. Resolve node attributes (and detect any cross-role conflicts)
    # ------------------------------------------------------------------
    node_attrs, conflicts = _resolve_node_attributes(df)

    if conflicts:
        for entity_id, conflict_map in conflicts.items():
            warnings.append(
                f"Node attribute conflict for entity '{entity_id}': {conflict_map}. "
                "Using first non-null value."
            )
        logger.warning(
            "Node attribute conflicts detected for %d entity/entities.",
            len(conflicts),
        )

    # ------------------------------------------------------------------
    # 2. Initialise the MultiDiGraph
    # ------------------------------------------------------------------
    G = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # 3. Add nodes with canonical attributes
    # ------------------------------------------------------------------
    for entity_id, attrs in node_attrs.items():
        G.add_node(entity_id, **attrs)

    logger.info("Nodes added: %d", G.number_of_nodes())

    # ------------------------------------------------------------------
    # 4. Build the authoritative edge table
    # ------------------------------------------------------------------
    edge_table_rows: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        source = row.payer_entity_id
        target = row.beneficiary_entity_id
        edge_key = row.id  # unique transaction ID → multigraph edge key

        # Build edge attribute dict from the mapping
        edge_attrs: dict[str, Any] = {}
        for df_col, edge_attr in _EDGE_ATTR_MAP.items():
            raw = getattr(row, df_col, None)
            edge_attrs[edge_attr] = _nan_to_none(raw)

        G.add_edge(source, target, key=edge_key, **edge_attrs)

        edge_table_rows.append(
            {
                "source_entity": source,
                "target_entity": target,
                "edge_key": edge_key,
                **edge_attrs,
            }
        )

    edge_table = pd.DataFrame(edge_table_rows)

    logger.info("Edges added: %d", G.number_of_edges())

    # ------------------------------------------------------------------
    # 5. Post-construction validation
    # ------------------------------------------------------------------

    # 5a. Node count
    if G.number_of_nodes() != preprocessing_result.entity_count:
        errors.append(
            f"Node count mismatch: expected {preprocessing_result.entity_count}, "
            f"got {G.number_of_nodes()}."
        )

    # 5b. Edge count
    if G.number_of_edges() != preprocessing_result.transaction_count:
        errors.append(
            f"Edge count mismatch: expected {preprocessing_result.transaction_count}, "
            f"got {G.number_of_edges()}."
        )

    # 5c. Transaction IDs are unique edge keys (no collision)
    # The edge table has one row per transaction; verify no tx_id repeated
    if edge_table["edge_key"].duplicated().any():
        n_dup = int(edge_table["edge_key"].duplicated().sum())
        errors.append(
            f"Duplicate transaction IDs used as edge keys: {n_dup} duplicate(s) found."
        )

    # 5d. Every entity in preprocessing entity_ids has a node
    missing_nodes = [
        eid for eid in preprocessing_result.entity_ids if eid not in G
    ]
    if missing_nodes:
        errors.append(
            f"{len(missing_nodes)} entity/entities present in preprocessing "
            "entity_ids but absent from the graph: "
            f"{missing_nodes[:5]}{'...' if len(missing_nodes) > 5 else ''}"
        )

    # 5e. source and target nodes of every edge exist in the graph
    dangling = [
        (u, v, k) for u, v, k in G.edges(keys=True)
        if u not in G.nodes or v not in G.nodes
    ]
    if dangling:
        errors.append(
            f"{len(dangling)} edge(s) have a source or target that is not a "
            "recognised graph node. This is an internal consistency error."
        )

    logger.info(
        "Graph construction complete. Nodes: %d | Edges: %d | "
        "Conflicts: %d | Warnings: %d | Errors: %d",
        G.number_of_nodes(),
        G.number_of_edges(),
        len(conflicts),
        len(warnings),
        len(errors),
    )

    return GraphResult(
        graph=G,
        edge_table=edge_table,
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
        node_conflict_report=conflicts,
        validation_warnings=warnings,
        validation_errors=errors,
    )


def build_graph_from_dataset(dataset_path: str) -> GraphResult:
    """Convenience: ingest → preprocess → build graph in one call.

    Parameters
    ----------
    dataset_path:
        Path to ``dkdata.json``.

    Returns
    -------
    GraphResult
    """
    prep = load_and_preprocess(dataset_path)
    return build_graph(prep)


def describe_result(result: GraphResult) -> str:
    """Return a human-readable summary of a :class:`GraphResult`.

    Parameters
    ----------
    result:
        Result returned by :func:`build_graph`.

    Returns
    -------
    str
        Multi-line summary string.
    """
    G = result.graph

    # Directed pair count
    directed_pairs = len(set((u, v) for u, v, _ in G.edges(keys=True)))
    # Self-loop count
    self_loops = sum(1 for u, v, _ in G.edges(keys=True) if u == v)
    # Reciprocal pairs
    pair_set = {(u, v) for u, v, _ in G.edges(keys=True)}
    reciprocal = len({
        tuple(sorted([u, v]))
        for u, v in pair_set
        if u != v and (v, u) in pair_set
    })

    lines: list[str] = [
        "=" * 60,
        "GRAPH CONSTRUCTION SUMMARY",
        "=" * 60,
        f"Graph type        : {type(G).__name__}",
        f"Node count        : {result.node_count}",
        f"Edge count        : {result.edge_count}",
        f"Directed pairs    : {directed_pairs}",
        f"Self-loops        : {self_loops}",
        f"Reciprocal pairs  : {reciprocal}",
        f"Validation status : {'VALID' if result.is_valid else 'INVALID'}",
        "",
    ]

    if result.node_conflict_report:
        lines.append(f"Node attribute conflicts ({len(result.node_conflict_report)}):")
        for entity_id, conflict in list(result.node_conflict_report.items())[:5]:
            lines.append(f"  {entity_id}: {conflict}")
        if len(result.node_conflict_report) > 5:
            lines.append(f"  ... and {len(result.node_conflict_report) - 5} more")
        lines.append("")

    if result.validation_warnings:
        lines.append(f"Warnings ({len(result.validation_warnings)}):")
        for w in result.validation_warnings:
            lines.append(f"  [WARN] {w[:120]}")
        lines.append("")

    if result.validation_errors:
        lines.append(f"Errors ({len(result.validation_errors)}):")
        for e in result.validation_errors:
            lines.append(f"  [ERROR] {e}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
