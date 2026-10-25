"""
Layer 6 - Experiment Dataset
=============================

A model-agnostic view of one labelled entity population.

``ExperimentDataset`` is deliberately thin.  It holds the things every
candidate family needs and nothing that presumes a particular one:

    entity_ids   ordered, deterministic; positional index i means the same
                 entity everywhere in the framework
    labels       binary int array aligned to ``entity_ids``
    edge_table   the authoritative transaction rows (may be ``None``)
    graph        a ``networkx.MultiDiGraph`` view of those rows (may be ``None``)

A tabular model ignores ``graph``; a GNN uses it; a temporal model uses the
``date`` column of ``edge_table``.  None of them need a different container.

Labels are stored separately from the graph on purpose.  The legacy pipeline
reads the positive class off a ``core`` node attribute, which makes it easy for
a label to leak into a feature computed from node attributes.  Here the label
never travels with the structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.spec import DatasetSpec

logger = logging.getLogger(__name__)

__all__ = ["ExperimentDataset", "dataset_from_graph_result"]

#: Column names that must never appear in a feature matrix.  They are either
#: the label itself or post-event bookkeeping added by the investigation that
#: produced the label.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "core",
        "payer_core",
        "beneficiary_core",
        "label",
        "y",
        "target",
        "investigation",
        "source_file",
    }
)


@dataclass
class ExperimentDataset:
    """One labelled entity population plus the evidence available about it.

    Attributes
    ----------
    spec:
        Identity, version, provenance and label definition.
    entity_ids:
        Deterministically ordered entity identifiers.  Positional index is the
        framework's universal entity address.
    labels:
        ``int8`` array of 0/1 aligned to ``entity_ids``.
    edge_table:
        One row per transaction.  Expected columns when present:
        ``source_entity``, ``target_entity``, and optionally ``amount_usd`` and
        ``date``.  ``None`` when the dataset is purely tabular.
    graph:
        ``networkx.MultiDiGraph`` over ``entity_ids``.  ``None`` when the
        dataset has no graph representation.
    attributes:
        Optional per-entity attribute frame indexed by entity id (entity type,
        bank country, ...).  Never contains the label.
    transaction_labels:
        The publisher's transaction-level flag, ``bool`` and row-aligned to
        ``edge_table``, or ``None`` when the source labels entities rather than
        transactions.  It is label data: it is *not* a column of the edge table,
        so no feature family can read it, and a candidate must not either.  It
        exists so a split can derive a label from a time window
        (:mod:`src.experiments.calendar`) instead of from the whole file.
    """

    spec: DatasetSpec
    entity_ids: list[str]
    labels: np.ndarray
    edge_table: Optional[pd.DataFrame] = None
    graph: Optional[Any] = None
    attributes: Optional[pd.DataFrame] = None
    transaction_labels: Optional[np.ndarray] = None
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels).astype(np.int8).reshape(-1)
        if len(self.labels) != len(self.entity_ids):
            raise ValueError(
                f"labels length {len(self.labels)} does not match "
                f"entity_ids length {len(self.entity_ids)}."
            )
        bad = set(np.unique(self.labels)) - {0, 1}
        if bad:
            raise ValueError(f"labels must be binary 0/1; found {sorted(bad)}.")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids contains duplicates.")
        if self.transaction_labels is not None:
            self.transaction_labels = np.asarray(self.transaction_labels).astype(bool).reshape(-1)
            n_edges = 0 if self.edge_table is None else len(self.edge_table)
            if len(self.transaction_labels) != n_edges:
                raise ValueError(
                    f"transaction_labels length {len(self.transaction_labels)} does "
                    f"not match the edge table's {n_edges} rows."
                )
        self._index = {e: i for i, e in enumerate(self.entity_ids)}

    # -- basic properties -------------------------------------------------

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def n_negative(self) -> int:
        return self.n_entities - self.n_positive

    @property
    def prevalence(self) -> float:
        """Positive rate.  0.0 for an empty dataset."""
        return float(self.labels.mean()) if self.n_entities else 0.0

    def index_of(self, entity_id: str) -> int:
        return self._index[entity_id]

    def ids_for(self, idx: Sequence[int] | np.ndarray) -> list[str]:
        return [self.entity_ids[int(i)] for i in idx]

    # -- temporal helpers -------------------------------------------------

    def has_time(self) -> bool:
        """True when the edge table carries a usable ``date`` column."""
        if self.edge_table is None or "date" not in self.edge_table.columns:
            return False
        return bool(pd.to_datetime(self.edge_table["date"], errors="coerce").notna().any())

    def entity_times(self, how: str = "last") -> pd.Series:
        """Per-entity activity time, indexed by entity id.

        Parameters
        ----------
        how:
            ``"first"`` or ``"last"``.  A time-forward split on *first*
            activity asks "can the model score an entity we have only just met";
            a split on *last* activity asks "can it score entities whose
            behaviour has already concluded".  Both are legitimate; the choice
            is recorded in the split plan.

        Returns
        -------
        ``pd.Series`` of ``datetime64`` indexed by entity id.  Entities with no
        dated transaction get ``NaT`` - they are never silently given a time.
        """
        if how not in ("first", "last"):
            raise ValueError(f"how must be 'first' or 'last'; got {how!r}.")
        times = pd.Series(pd.NaT, index=pd.Index(self.entity_ids), dtype="datetime64[ns]")
        if not self.has_time():
            return times

        et = self.edge_table
        dates = pd.to_datetime(et["date"], errors="coerce")
        stacked = pd.concat(
            [
                pd.DataFrame({"entity": et["source_entity"], "date": dates}),
                pd.DataFrame({"entity": et["target_entity"], "date": dates}),
            ],
            ignore_index=True,
        ).dropna(subset=["date"])
        if stacked.empty:
            return times
        agg = stacked.groupby("entity")["date"].min() if how == "first" else (
            stacked.groupby("entity")["date"].max()
        )
        times.loc[times.index.intersection(agg.index)] = agg.reindex(
            times.index.intersection(agg.index)
        )
        return times

    def edges_before(self, cutoff: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
        """Edge rows strictly before ``cutoff``.

        ``None`` cutoff returns the full table.  Rows with an unparseable date
        are *dropped* when a cutoff is given: an undated transaction cannot be
        proven to predate the boundary, and assuming it does would leak future
        evidence into a training window.
        """
        if self.edge_table is None:
            return None
        if cutoff is None:
            return self.edge_table
        dates = pd.to_datetime(self.edge_table["date"], errors="coerce")
        return self.edge_table.loc[dates.notna() & (dates < cutoff)]

    def subgraph_from_edges(self, edges: Optional[pd.DataFrame]):
        """Build a MultiDiGraph over *all* entities using only ``edges``.

        Node set is always the full entity population, so feature matrices stay
        row-aligned; only the edges are restricted.  Node attributes are copied
        from ``self.graph`` when it exists, minus anything in
        :data:`FORBIDDEN_FEATURE_COLUMNS`.
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        for entity in self.entity_ids:
            attrs: dict[str, Any] = {}
            if self.graph is not None and entity in self.graph:
                attrs = {
                    k: v
                    for k, v in self.graph.nodes[entity].items()
                    if k not in FORBIDDEN_FEATURE_COLUMNS
                }
            g.add_node(entity, **attrs)

        if edges is None or edges.empty:
            return g

        attr_cols = [
            c
            for c in edges.columns
            if c not in ("source_entity", "target_entity", "edge_key")
            and c not in FORBIDDEN_FEATURE_COLUMNS
        ]
        for row in edges.itertuples(index=False):
            key = getattr(row, "edge_key", None)
            payload = {c: getattr(row, c, None) for c in attr_cols}
            g.add_edge(row.source_entity, row.target_entity, key=key, **payload)
        return g

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Machine-readable description for an experiment artifact."""
        return {
            "spec": self.spec.to_dict(),
            "n_entities": self.n_entities,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "prevalence": self.prevalence,
            "n_edges": 0 if self.edge_table is None else int(len(self.edge_table)),
            "has_graph": self.graph is not None,
            "has_time": self.has_time(),
            "has_transaction_labels": self.transaction_labels is not None,
        }


def dataset_from_graph_result(
    graph_result: Any,
    spec: DatasetSpec,
    *,
    label_attr: str = "core",
) -> ExperimentDataset:
    """Adapt a Layer 3 :class:`~src.graph.builder.GraphResult` to the framework.

    The label is read off the ``label_attr`` node attribute and then *removed
    from the copy of the node attributes the framework carries forward*, so the
    label cannot re-enter through an attribute-derived feature.  The caller's
    graph object is not modified: the label attribute is left on it because
    downstream legacy code (``src.models.dataset.build_pyg_data``) reads it.

    Parameters
    ----------
    graph_result:
        Anything exposing ``.graph`` (MultiDiGraph) and ``.edge_table``.
    spec:
        Dataset identity; ``spec.label_definition`` should state the rule that
        ``label_attr`` encodes.
    label_attr:
        Node attribute holding the ground-truth positive flag.
    """
    graph = graph_result.graph
    entity_ids = sorted(graph.nodes())
    labels = np.array(
        [1 if bool(graph.nodes[e].get(label_attr, False)) else 0 for e in entity_ids],
        dtype=np.int8,
    )

    attr_rows: list[dict[str, Any]] = []
    for entity in entity_ids:
        attr_rows.append(
            {
                k: v
                for k, v in graph.nodes[entity].items()
                if k not in FORBIDDEN_FEATURE_COLUMNS and k != label_attr
            }
        )
    attributes = pd.DataFrame(attr_rows, index=pd.Index(entity_ids, name="entity_id"))

    logger.info(
        "ExperimentDataset '%s' v%s: %d entities, %d positive (%.4f%%), %d edges.",
        spec.name,
        spec.version,
        len(entity_ids),
        int(labels.sum()),
        100.0 * float(labels.mean()) if len(entity_ids) else 0.0,
        len(graph_result.edge_table) if graph_result.edge_table is not None else 0,
    )

    return ExperimentDataset(
        spec=spec,
        entity_ids=entity_ids,
        labels=labels,
        edge_table=graph_result.edge_table,
        graph=graph,
        attributes=attributes,
    )
