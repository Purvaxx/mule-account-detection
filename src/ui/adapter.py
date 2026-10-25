import pandas as pd
import networkx as nx
import math
import numpy as np
from datetime import date, datetime
from src.ui.config import UIConfig
from src.ui.context import AnalysisContext
from src.detection.cycles import CycleAnalysis, CycleAnalysisUnavailable

# ---------------------------------------------------------------------------
# Cycle evidence — response states
# ---------------------------------------------------------------------------
#: Cycle analysis ran to completion.  ``cycles`` is a list, possibly empty; an
#: empty list is the positive finding that the graph (or the queried entity) has
#: no cycles within the configured bounds.
CYCLE_STATE_COMPLETE = "COMPLETE"

#: No cycle evidence exists for this context.  ``cycles`` and ``metadata`` are
#: ``null`` — never ``[]`` — so "we could not analyse" can never be read as
#: "there are none".
CYCLE_STATE_UNAVAILABLE = "UNAVAILABLE"

#: ``unavailable.reason`` when the context never attempted analysis at all.
CYCLE_REASON_NOT_ATTEMPTED = "NOT_ATTEMPTED"

#: Fixed operator-facing text per reason.  The underlying exception message is
#: deliberately **not** returned: it is logged by ``analyze_cycles_safely`` with
#: a traceback, and an API response is the wrong place for interpreter detail.
_UNAVAILABLE_MESSAGES = {
    CYCLE_REASON_NOT_ATTEMPTED: (
        "Cycle analysis was not run for this dataset. No conclusion about "
        "cycles can be drawn from this response."
    ),
    "CYCLE_ANALYSIS_FAILED": (
        "Cycle analysis was attempted for this dataset and did not complete. "
        "This is not a finding that the dataset has no cycles. Model "
        "predictions are unaffected; see the server log for the cause."
    ),
}

_UNAVAILABLE_FALLBACK_MESSAGE = (
    "Cycle evidence is not available for this dataset. This is not a finding "
    "that the dataset has no cycles."
)


def _json_scalar(value):
    """Coerce one leaf value to a JSON-native equivalent.

    Follows the convention already established by
    ``src.datasets.profiler._normalize_for_json``: ISO 8601 for instants,
    native ``int`` / ``float`` / ``bool`` for numpy scalars, ``None`` for
    anything absent or non-finite.

    A missing value stays ``None``.  It is never turned into ``0``.
    """
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        # pd.NaT is a Timestamp subclass instance that is also NA.
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _serialize_leg(leg) -> dict:
    """One directed step of a cycle, with its evidence, JSON-safe."""
    return {
        "source": _json_scalar(leg.source),
        "target": _json_scalar(leg.target),
        "transaction_count": int(leg.transaction_count),
        # Aggregates over the transactions that carried an amount.  When
        # amount_evidence is PARTIAL the total is a sum over part of the
        # bundle; transactions_with_amount is what says so.
        "amount_usd_total": _json_scalar(leg.amount_usd_total),
        "amount_usd_min": _json_scalar(leg.amount_usd_min),
        "amount_usd_max": _json_scalar(leg.amount_usd_max),
        "transactions_with_amount": int(leg.transactions_with_amount),
        "amount_evidence": leg.amount_evidence,
        "first_timestamp": _json_scalar(leg.first_timestamp),
        "last_timestamp": _json_scalar(leg.last_timestamp),
        "transactions_with_timestamp": int(leg.transactions_with_timestamp),
        # Identifiers are passed through in their own type (the Reference
        # Dataset's are integers, a workspace's derived ids are strings), only
        # coerced where they are not JSON-native.
        "transaction_id_sample": [
            _json_scalar(tx_id) for tx_id in leg.transaction_id_sample
        ],
        "transaction_id_sample_truncated": bool(leg.transaction_id_sample_truncated),
    }


def _serialize_cycle(cycle) -> dict:
    """One entity-level directed cycle.

    A faithful projection of ``CycleEvidence``: no value is recomputed, no
    score is derived, and the three-valued ``is_time_respecting`` keeps its
    ``true`` / ``false`` / ``null`` exactly as the domain produced it.
    """
    return {
        "cycle_id": cycle.cycle_id,
        "entities": [_json_scalar(entity) for entity in cycle.entities],
        "length": int(cycle.length),
        "participant_count": int(cycle.participant_count),
        "transaction_count": int(cycle.transaction_count),
        "is_self_loop": bool(cycle.is_self_loop),
        "is_reciprocal_pair": bool(cycle.is_reciprocal_pair),
        "amount_evidence": cycle.amount_evidence,
        "legs_with_complete_amounts": int(cycle.legs_with_complete_amounts),
        "leg_amount_totals": [
            _json_scalar(total) for total in cycle.leg_amount_totals
        ],
        "leg_amount_balance": _json_scalar(cycle.leg_amount_balance),
        "amount_retention": _json_scalar(cycle.amount_retention),
        "amount_retention_basis": cycle.amount_retention_basis,
        "first_transaction_at": _json_scalar(cycle.first_transaction_at),
        "last_transaction_at": _json_scalar(cycle.last_transaction_at),
        "time_span_days": _json_scalar(cycle.time_span_days),
        # Three-valued on purpose: null means "not establishable", never false.
        "is_time_respecting": cycle.is_time_respecting,
        "temporal_basis": cycle.temporal_basis,
        "temporal_explanation": cycle.temporal_explanation,
        "legs": [_serialize_leg(leg) for leg in cycle.legs],
    }


#: Route a client fetches one cycle's complete evidence from.
CYCLE_DETAIL_ROUTE = "/api/cycles/{cycle_id}"

#: Fields of the full evidence that a summary row deliberately omits.  Named in
#: every index response so the reduction is explicit rather than silent: a
#: client can see exactly what it does not have and where to get it.
CYCLE_DETAIL_ONLY_FIELDS = (
    "legs",
    "leg_amount_totals",
    "legs_with_complete_amounts",
    "amount_retention",
    "amount_retention_basis",
    "temporal_explanation",
    "matched_rules",
)

#: Tier-3 named selection rules are **not defined** at this layer.  Stage 1
#: implemented Tier 1 (structural facts) and Tier 2 (derived properties) only,
#: and no rule may be invented here: a threshold with no evidence behind it
#: would read as a finding.  The field is served as a declared, empty seam so
#: its absence is visible rather than silently missing.
CYCLE_MATCHED_RULES_NOTE = (
    "No named selection rules are defined for cycle evidence. A cycle is a "
    "structural observation; this layer applies no suspiciousness rule and "
    "invents no threshold."
)


def _serialize_cycle_summary(cycle) -> dict:
    """A compact index row: enough to list, group, order and choose from.

    Carries no per-leg evidence.  Everything omitted is named in
    :data:`CYCLE_DETAIL_ONLY_FIELDS` and reachable at
    :data:`CYCLE_DETAIL_ROUTE`.

    Values are the same ones the full evidence carries — nothing is rounded,
    recomputed or re-derived for the summary.
    """
    return {
        "cycle_id": cycle.cycle_id,
        "entities": [_json_scalar(entity) for entity in cycle.entities],
        "length": int(cycle.length),
        "participant_count": int(cycle.participant_count),
        "transaction_count": int(cycle.transaction_count),
        "is_self_loop": bool(cycle.is_self_loop),
        "is_reciprocal_pair": bool(cycle.is_reciprocal_pair),
        "amount_evidence": cycle.amount_evidence,
        "leg_amount_balance": _json_scalar(cycle.leg_amount_balance),
        "first_transaction_at": _json_scalar(cycle.first_transaction_at),
        "last_transaction_at": _json_scalar(cycle.last_transaction_at),
        "time_span_days": _json_scalar(cycle.time_span_days),
        "is_time_respecting": cycle.is_time_respecting,
        "temporal_basis": cycle.temporal_basis,
        "has_detail": True,
    }


def _summarize_selection(cycles) -> dict:
    """Aggregates over the selected cycles — the whole selection, not the page.

    Computed before paging, so a dashboard gets true totals from any page and
    never has to add up rows to learn how many there are.
    """
    by_length: dict[int, int] = {}
    for cycle in cycles:
        by_length[cycle.length] = by_length.get(cycle.length, 0) + 1

    participants = {entity for cycle in cycles for entity in cycle.entities}

    return {
        "total_cycles": len(cycles),
        # JSON object keys must be strings; the values are the counts.
        "by_length": {str(length): by_length[length] for length in sorted(by_length)},
        "shortest_cycle_length": min(by_length) if by_length else None,
        "longest_cycle_length": max(by_length) if by_length else None,
        "participating_entity_count": len(participants),
        "reciprocal_pair_count": sum(1 for c in cycles if c.is_reciprocal_pair),
    }


def _paginate(cycles, limit: int, offset: int) -> dict:
    """Slice a selection and state the slice, so no reduction is implicit."""
    total = len(cycles)
    start = min(offset, total)
    window = cycles[start:start + limit]
    return {
        "page": {
            "limit": int(limit),
            "offset": int(start),
            "returned": len(window),
            "total": total,
            "has_more": start + len(window) < total,
        },
        "window": window,
    }


def _serialize_metadata(metadata) -> dict:
    """The bounds and counts a reader needs to reproduce and qualify a result."""
    config = metadata.config
    return {
        "bounds": {
            "max_cycle_length": int(config.max_cycle_length),
            "min_cycle_length": int(config.min_cycle_length),
            "max_cycles": int(config.max_cycles),
            "max_transaction_samples_per_leg": int(
                config.max_transaction_samples_per_leg
            ),
        },
        "temporal_precision": metadata.temporal_precision,
        "temporal_precision_basis": metadata.temporal_precision_basis,
        "enumeration_complete": bool(metadata.enumeration_complete),
        "truncated": bool(metadata.truncated),
        "truncation_reason": metadata.truncation_reason,
        "elapsed_seconds": _json_scalar(metadata.elapsed_seconds),
        "counts": {
            "entities": int(metadata.node_count),
            "transactions": int(metadata.transaction_count),
            "unique_directed_pairs": int(metadata.unique_directed_pair_count),
            "self_loop_pairs": int(metadata.self_loop_pair_count),
            # Counts for the whole analysis, before any entity filter.
            "cycles_enumerated": int(metadata.cycles_enumerated),
            "cycles_retained": int(metadata.cycles_retained),
            "cycles_excluded_by_min_length": int(
                metadata.cycles_excluded_by_min_length
            ),
            "timestamped_transactions": int(metadata.timestamped_transaction_count),
            "unparseable_timestamps": int(metadata.unparseable_timestamp_count),
            "transactions_with_amount": int(
                metadata.amount_present_transaction_count
            ),
        },
    }


class DetectionAdapter:
    """
    Adapter layer that safely interfaces with the frozen ML pipeline.
    It builds the read-only structured view models for the UI.
    """
    def get_overview(self, context: AnalysisContext):
        df, graph = context.df, context.graph

        suspicious = df[df['predicted_class'] == 1]

        # Sort by risk score descending for the table
        sorted_df = df.sort_values(by='risk_score', ascending=False)
        all_entities = sorted_df.to_dict(orient="records")

        # JSON compliance: replace nan/inf with None
        for row in all_entities:
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = None

        # We explicitly return safe UI models
        return {
            "total_entities": len(df),
            "total_transactions": graph.number_of_edges(),
            "suspicious_count": len(suspicious),
            "top_entities": all_entities[:UIConfig.TOP_K],
            "all_entities": all_entities
        }

    def get_entity_details(self, entity_id, context: AnalysisContext):
        df, _ = context.df, context.graph
        row_df = df[df['entity_id'] == entity_id]
        if len(row_df) == 0:
            return None
        row = row_df.iloc[0].to_dict()

        # Replace float nan/inf with string or null for JSON compliance
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                row[k] = None

        flags = []
        if row['reciprocal_out'] > 0 or row['reciprocal_in'] > 0:
            val = max(row['reciprocal_in'], row['reciprocal_out'])
            flags.append({
                "title": "RECIPROCAL RELATIONSHIPS",
                "value": f"{val} counterparty(ies)",
                "observed": "Transaction flow is observed in both directions between the entity and the counterparty.",
                "why_it_matters": "This identifies a two-way relationship within the observed dataset rather than simple one-way transfers.",
                "rule": "Reciprocal count > 0"
            })
        if row['out_repeated_ratio'] > 2.0:
            flags.append({
                "title": "OUTGOING REPETITION",
                "value": f"{row['out_repeated_ratio']:.2f}x",
                "observed": "The entity's outgoing transactions are concentrated among repeatedly used counterparties.",
                "why_it_matters": "Concentrated transaction relationships can be useful for further review of the surrounding network.",
                "rule": f"{row['out_repeated_ratio']:.2f}x > configured threshold of 2.00x"
            })
        if row['in_repeated_ratio'] > 2.0:
            flags.append({
                "title": "INCOMING REPETITION",
                "value": f"{row['in_repeated_ratio']:.2f}x",
                "observed": "The entity's incoming transactions are concentrated among repeatedly used counterparties.",
                "why_it_matters": "Concentrated transaction relationships can be useful for further review of the surrounding network.",
                "rule": f"{row['in_repeated_ratio']:.2f}x > configured threshold of 2.00x"
            })
        if row['max_parallel_edge'] > 5:
            flags.append({
                "title": "MAXIMUM PARALLEL TRANSACTIONS",
                "value": str(row['max_parallel_edge']),
                "observed": "The highest number of individual transactions observed between this entity and a single counterparty.",
                "why_it_matters": "High multiplicity can help an analyst understand the intensity of a particular relationship.",
                "rule": f"{row['max_parallel_edge']} > configured threshold of 5"
            })
        if abs(row['flow_imbalance']) < 0.1 and (row['in_degree'] > 0 and row['out_degree'] > 0):
            flags.append({
                "title": "BALANCED FLOW",
                "value": f"abs({row['flow_imbalance']:.4f})",
                "observed": "Incoming and outgoing transaction volume are relatively balanced.",
                "why_it_matters": "This is a descriptive structural observation reflecting balanced flow volume. Balanced flow alone does not establish suspicious activity.",
                "rule": "< 0.1 variance"
            })

        row['flags'] = flags
        return row

    def get_network(self, entity_id, context: AnalysisContext, max_neighbors=UIConfig.MAX_VISIBLE_NEIGHBORS, min_tx=UIConfig.DEFAULT_MIN_TX, reciprocal_only=False):
        _, graph = context.df, context.graph
        if entity_id not in graph:
            return {"nodes": [], "edges": [], "hidden_count": 0}

        subgraph = nx.ego_graph(graph, entity_id, radius=UIConfig.GRAPH_RADIUS)

        # Aggregate parallel transactions
        out_counts = {}
        in_counts = {}
        for u, v in subgraph.edges():
            if u == entity_id:
                out_counts[v] = out_counts.get(v, 0) + 1
            elif v == entity_id:
                in_counts[u] = in_counts.get(u, 0) + 1

        # Rank counterparties
        counterparties = set(out_counts.keys()).union(set(in_counts.keys()))
        counterparties.discard(entity_id)

        scored_neighbors = []
        for neighbor in counterparties:
            in_tx = in_counts.get(neighbor, 0)
            out_tx = out_counts.get(neighbor, 0)
            total_tx = in_tx + out_tx
            is_recip = (in_tx > 0 and out_tx > 0)

            if total_tx < min_tx:
                continue
            if reciprocal_only and not is_recip:
                continue

            score = (1000000 if is_recip else 0) + total_tx
            scored_neighbors.append((score, neighbor, in_tx, out_tx))

        # Sort by reciprocal/total_tx, then deterministically by ID
        scored_neighbors.sort(key=lambda x: (x[0], x[1]), reverse=True)

        hidden_count = 0
        if max_neighbors > 0 and len(scored_neighbors) > max_neighbors:
            hidden_count = len(scored_neighbors) - max_neighbors
            scored_neighbors = scored_neighbors[:max_neighbors]

        relationships = []
        visible_neighbors = {x[1] for x in scored_neighbors}
        visible_neighbors.add(entity_id)

        for neighbor in counterparties:
            in_tx = in_counts.get(neighbor, 0)
            out_tx = out_counts.get(neighbor, 0)
            is_recip = (in_tx > 0 and out_tx > 0)

            direction = "RECIPROCAL" if is_recip else ("IN" if in_tx > 0 else "OUT")

            relationships.append({
                "counterparty": neighbor,
                "direction": direction,
                "transaction_count": in_tx + out_tx,
                "reciprocal": is_recip,
                "visible_in_network": neighbor in visible_neighbors
            })

        # Sort relationships by visible first, then transaction count
        relationships.sort(key=lambda x: (not x["visible_in_network"], -x["transaction_count"], x["counterparty"]))

        nodes = []
        for n in visible_neighbors:
            nodes.append({
                "id": n,
                "label": str(n),
                "is_target": n == entity_id
            })

        edges = []
        for score, neighbor, in_tx, out_tx in scored_neighbors:
            if out_tx > 0:
                edges.append({
                    "from": entity_id,
                    "to": neighbor,
                    "value": out_tx,
                    "title": f"{out_tx} transactions"
                })
            if in_tx > 0:
                edges.append({
                    "from": neighbor,
                    "to": entity_id,
                    "value": in_tx,
                    "title": f"{in_tx} transactions"
                })

        return {"nodes": nodes, "edges": edges, "hidden_count": hidden_count, "relationships": relationships}

    def _cycle_envelope(self, context: AnalysisContext, extra: dict) -> dict:
        """The shared head of every cycle response.

        Returns ``(envelope, analysis)``-style handling in one place so the
        index and the detail route cannot drift on how they report
        ``UNAVAILABLE``.
        """
        analysis = context.cycles
        if isinstance(analysis, CycleAnalysis):
            return None

        if isinstance(analysis, CycleAnalysisUnavailable):
            reason = analysis.reason
            error_type = analysis.error_type
        else:
            # context.cycles is None: never attempted.
            reason = CYCLE_REASON_NOT_ATTEMPTED
            error_type = None

        return {
            "workspace_id": context.workspace_id,
            **extra,
            "state": CYCLE_STATE_UNAVAILABLE,
            "metadata": None,
            "unavailable": {
                "reason": reason,
                "error_type": error_type,
                "message": _UNAVAILABLE_MESSAGES.get(
                    reason, _UNAVAILABLE_FALLBACK_MESSAGE
                ),
            },
        }

    def get_cycle_detail(self, context: AnalysisContext, cycle_id: str):
        """The complete Stage 1 evidence for one cycle of this context.

        Resolution is **context-local**: the id is looked up in this context's
        own analysis and nowhere else.  There is no global cycle registry, so a
        `cycle_id` produced by one workspace simply does not exist in another
        and comes back as absent — isolation is a property of the lookup, not a
        check bolted on top of it.

        Returns ``None`` when the analysis completed and holds no such cycle,
        which the route reports as 404.  An *unavailable* analysis is a
        different answer and returns the ``UNAVAILABLE`` envelope instead: it
        does not claim the cycle is absent, because it does not know.
        """
        unavailable = self._cycle_envelope(context, {"cycle_id": cycle_id, "cycle": None})
        if unavailable is not None:
            return unavailable

        analysis = context.cycles
        cycle = analysis.get(cycle_id)
        if cycle is None:
            return None

        return {
            "workspace_id": context.workspace_id,
            "cycle_id": cycle_id,
            "state": CYCLE_STATE_COMPLETE,
            "cycle": {
                **_serialize_cycle(cycle),
                # Declared, deliberately empty. See CYCLE_MATCHED_RULES_NOTE.
                "matched_rules": [],
                "matched_rules_note": CYCLE_MATCHED_RULES_NOTE,
            },
            # The bounds this cycle was found under travel with it, so a single
            # cycle fetched on its own is still qualifiable.
            "metadata": _serialize_metadata(analysis.metadata),
            "unavailable": None,
        }

    def get_cycles(
        self,
        context: AnalysisContext,
        entity_id=None,
        limit: int = None,
        offset: int = 0,
    ) -> dict:
        """A compact, paged index of this context's cycles.

        This is **investigator evidence, not model reasoning**.  It carries no
        risk score, no "cycle risk" number, no suspiciousness judgement and no
        threshold this layer invented.  Nothing here re-reads, re-derives or
        reinterprets ``risk_score`` or ``predicted_class``; the frozen model's
        own threshold remains the only model-derived cut-off in the system.

        Rows are summaries.  The full per-leg evidence for any one cycle is
        fetched from :data:`CYCLE_DETAIL_ROUTE`; what a row omits is named in
        ``detail.omitted_fields``, so the reduction is stated rather than
        silent.  On the Reference Dataset this is the difference between a
        ~47 KB page and a ~19.3 MB response.

        Parameters
        ----------
        context:
            A context already resolved (and cached) by
            ``WorkspaceAnalysisResolver``.  Cycle analysis is read off it; it
            is never recomputed here.
        entity_id:
            Optional filter.  When given, only cycles that entity participates
            in are indexed — as **whole** cycles.  Filtering selects, it never
            re-roots or truncates: a row for ``B`` of ``A -> B -> C -> A``
            lists ``A -> B -> C -> A`` under its unchanged canonical
            ``cycle_id``.
        limit, offset:
            Transport paging over the selection.  Defaults to
            ``UIConfig.CYCLE_PAGE_SIZE``.  The slice in force is always
            reported in ``page``.

        Returns
        -------
        dict
            ``state`` is ``COMPLETE`` or ``UNAVAILABLE``.

            * ``COMPLETE`` — ``cycles`` is a list of summary rows (possibly
              empty), ``summary`` aggregates the **whole selection** before
              paging, ``page`` states the slice, and ``metadata`` describes the
              analysis. An empty list is a genuine finding.
            * ``UNAVAILABLE`` — ``cycles``, ``summary``, ``page`` and
              ``metadata`` are ``None``, and ``unavailable`` states the reason.
              This can never be mistaken for an empty result.

            Three counts are kept separate and are never conflated:
            ``page.returned`` is this response, ``summary.total_cycles`` is the
            selection after any entity filter, and
            ``metadata.counts.cycles_retained`` is the whole analysis.
        """
        unavailable = self._cycle_envelope(
            context,
            {
                "entity_id": entity_id,
                "filtered_by_entity": entity_id is not None,
                "cycles": None,
                "summary": None,
                "page": None,
            },
        )
        if unavailable is not None:
            return unavailable

        analysis = context.cycles

        if entity_id is None:
            selected = analysis.cycles
        else:
            # entity_index maps an entity to the ids of the cycles it is in;
            # the cycles themselves are then indexed whole, in the analysis's
            # own deterministic order.
            wanted = set(analysis.entity_index.get(entity_id, ()))
            selected = tuple(c for c in analysis.cycles if c.cycle_id in wanted)

        if limit is None:
            limit = UIConfig.CYCLE_PAGE_SIZE
        sliced = _paginate(selected, limit, offset)

        return {
            "workspace_id": context.workspace_id,
            "entity_id": entity_id,
            "filtered_by_entity": entity_id is not None,
            "state": CYCLE_STATE_COMPLETE,
            "summary": _summarize_selection(selected),
            "page": sliced["page"],
            "cycles": [_serialize_cycle_summary(c) for c in sliced["window"]],
            "detail": {
                "route": CYCLE_DETAIL_ROUTE,
                "omitted_fields": list(CYCLE_DETAIL_ONLY_FIELDS),
            },
            "metadata": _serialize_metadata(analysis.metadata),
            "unavailable": None,
        }
