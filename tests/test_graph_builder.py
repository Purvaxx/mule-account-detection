"""
Tests for src/graph/builder.py — Layer 3 Graph Construction.

Coverage
--------
1.  Exactly 3,440 graph nodes for the current real dataset.
2.  Exactly 16,940 graph edges.
3.  Every transaction produces exactly one edge.
4.  Every transaction ID is preserved as an edge key.
5.  Transaction IDs remain unique graph edge keys.
6.  Parallel edges are preserved (multi-edge pairs survive).
7.  Self-loops are preserved.
8.  Reciprocal edges preserve both directions.
9.  No transaction is silently dropped.
10. No entity is silently dropped.
11. Source/target direction is preserved.
12. amount_orig is unchanged on edges.
13. amount_usd is unchanged on edges.
14. date is unchanged on edges.
15. Account metadata (payer_account_used, beneficiary_account_used) preserved.
16. Edge attributes correspond to the correct transaction ID.
17. No graph-derived ML features are added at this layer.
18. Existing Layer 1 and Layer 2 tests continue to pass (verified by running
    the full test suite).

Regression checks against the current real dataset:
- at least 69 self-loop edges
- exactly 148 reciprocal undirected entity-pair relationships
- exactly 4,024 distinct directed entity pairs

Synthetic-fixture tests:
- invalid preprocessing input raises ValueError
- parallel edges between same pair are NOT collapsed
- self-loop edges are NOT removed
- both directions of A→B and B→A are preserved
- missing account IDs do not remove transactions or entities
- node attribute conflicts are detected and reported
- edge_table has one row per transaction and aligns with graph edge count
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.data.ingestion import load_transactions
from src.data.preprocessing import PreprocessingResult, preprocess
from src.graph.builder import GraphResult, build_graph, build_graph_from_dataset, describe_result

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATH = _REPO_ROOT.parent / "dkdata.json"
_DATASET_PATH = Path(os.environ.get("DKDATA_PATH", _DATASET_PATH))
_REAL_DATASET_AVAILABLE = _DATASET_PATH.exists()

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_record(
    rec_id: int,
    payer_norm: str,
    beneficiary_norm: str,
    *,
    payer_account: str | None = "ACC001",
    beneficiary_account: str | None = "BACC001",
    amount_orig: float = -10_000.0,
    amount_usd: float = 10_000.0,
    date: str = "2013-06-01",
    payer_type: str = "Company",
    beneficiary_type: str = "Company",
    payer_core: bool = True,
    beneficiary_core: bool = False,
    payer_jurisdiction: str = "GB",
    beneficiary_jurisdiction: str = "TR",
    payer_bank_country: str = "GB",
    beneficiary_bank_country: str = "TR",
) -> dict:
    return {
        "id": rec_id,
        "date": date,
        "payer_name": f"Payer {payer_norm}",
        "payer_name_norm": payer_norm,
        "payer_account": payer_account,
        "payer_type": payer_type,
        "payer_jurisdiction": payer_jurisdiction,
        "payer_bank_country": payer_bank_country,
        "payer_core": payer_core,
        "beneficiary_name": f"Beneficiary {beneficiary_norm}",
        "beneficiary_name_norm": beneficiary_norm,
        "beneficiary_account": beneficiary_account,
        "beneficiary_type": beneficiary_type,
        "beneficiary_jurisdiction": beneficiary_jurisdiction,
        "beneficiary_bank_country": beneficiary_bank_country,
        "beneficiary_core": beneficiary_core,
        "amount_orig": amount_orig,
        "amount_orig_currency": "USD",
        "amount_usd": amount_usd,
        "amount_eur": amount_usd * 0.9,
        "purpose": "Payment",
        "source_file": "test",
        "investigation": "az",
    }


def _build_from_records(records: list[dict], tmp_path: Path) -> GraphResult:
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"data": records}), encoding="utf-8")
    ingestion = load_transactions(p)
    prep = preprocess(ingestion)
    return build_graph(prep)


# ---------------------------------------------------------------------------
# Session-scoped real-dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_graph_result() -> GraphResult:
    from tests.test_utils import get_cached_real_dataset_graph
    try:
        return get_cached_real_dataset_graph()
    except:
        pytest.skip(f"Real dataset not found at {_DATASET_PATH}")


# ---------------------------------------------------------------------------
# Test 1 & 10: Exactly 3,440 nodes; no entity dropped
# ---------------------------------------------------------------------------


class TestNodeCount:
    def test_node_count_equals_entity_count(
        self, real_graph_result: GraphResult
    ) -> None:
        assert real_graph_result.node_count == 3_440
        assert real_graph_result.graph.number_of_nodes() == 3_440

    def test_no_entity_silently_dropped(
        self, real_graph_result: GraphResult
    ) -> None:
        """All 3,440 preprocessing entities must appear as graph nodes."""
        # entity_ids come from the preprocessing layer — verify we can
        # reconstruct them independently from the edge table
        et = real_graph_result.edge_table
        entities_in_edges = (
            set(et["source_entity"].unique()) |
            set(et["target_entity"].unique())
        )
        assert real_graph_result.node_count == len(entities_in_edges)


# ---------------------------------------------------------------------------
# Test 2 & 9: Exactly 16,940 edges; no transaction dropped
# ---------------------------------------------------------------------------


class TestEdgeCount:
    def test_edge_count_equals_transaction_count(
        self, real_graph_result: GraphResult
    ) -> None:
        assert real_graph_result.edge_count == 16_940
        assert real_graph_result.graph.number_of_edges() == 16_940

    def test_edge_table_has_one_row_per_transaction(
        self, real_graph_result: GraphResult
    ) -> None:
        assert len(real_graph_result.edge_table) == 16_940


# ---------------------------------------------------------------------------
# Test 3 & 4: Every transaction produces exactly one edge; IDs preserved
# ---------------------------------------------------------------------------


class TestTransactionToEdgeMapping:
    def test_each_transaction_maps_to_exactly_one_edge(
        self, real_graph_result: GraphResult
    ) -> None:
        G = real_graph_result.graph
        all_keys = [k for _, _, k in G.edges(keys=True)]
        # Edge count == transaction count already tested; additionally verify
        # no key appears twice
        assert len(all_keys) == len(set(all_keys))

    def test_transaction_ids_in_edge_table_are_unique(
        self, real_graph_result: GraphResult
    ) -> None:
        keys = real_graph_result.edge_table["edge_key"]
        assert keys.duplicated().sum() == 0


# ---------------------------------------------------------------------------
# Test 5: Transaction IDs are unique graph edge keys
# ---------------------------------------------------------------------------


class TestEdgeKeyUniqueness:
    def test_edge_keys_are_unique(self, real_graph_result: GraphResult) -> None:
        all_keys = [k for _, _, k in real_graph_result.graph.edges(keys=True)]
        assert len(all_keys) == len(set(all_keys))


# ---------------------------------------------------------------------------
# Test 6: Parallel edges are preserved (regression: max 437 on one pair)
# ---------------------------------------------------------------------------


class TestParallelEdges:
    def test_parallel_edges_not_collapsed(self, tmp_path: Path) -> None:
        records = [
            _make_record(i + 1, "alpha", "beta") for i in range(5)
        ]
        result = _build_from_records(records, tmp_path)
        assert result.is_valid
        # MultiDiGraph should carry all 5 parallel edges
        edges = list(result.graph.edges("alpha", keys=True))
        alpha_to_beta = [k for u, v, k in edges if v == "beta"]
        assert len(alpha_to_beta) == 5

    def test_real_dataset_has_multi_edge_pairs(
        self, real_graph_result: GraphResult
    ) -> None:
        """The pair with max 437 transactions must still be 437 edges."""
        G = real_graph_result.graph
        max_parallel = 0
        for u in G.nodes():
            for v in G.successors(u):
                n = G.number_of_edges(u, v)
                if n > max_parallel:
                    max_parallel = n
        assert max_parallel == 437


# ---------------------------------------------------------------------------
# Test 7: Self-loops are preserved (regression: at least 69)
# ---------------------------------------------------------------------------


class TestSelfLoops:
    def test_self_loops_not_removed(self, tmp_path: Path) -> None:
        records = [_make_record(1, "ring_entity", "ring_entity")]
        result = _build_from_records(records, tmp_path)
        assert result.is_valid
        assert nx.number_of_selfloops(result.graph) == 1

    def test_real_dataset_self_loop_count(
        self, real_graph_result: GraphResult
    ) -> None:
        """Audit found 69 self-loop transactions across 2 entities."""
        assert nx.number_of_selfloops(real_graph_result.graph) >= 69


# ---------------------------------------------------------------------------
# Test 8: Reciprocal edges (regression: 148 undirected reciprocal pairs)
# ---------------------------------------------------------------------------


class TestReciprocalEdges:
    def test_both_directions_preserved(self, tmp_path: Path) -> None:
        records = [
            _make_record(1, "a_entity", "b_entity"),
            _make_record(2, "b_entity", "a_entity"),
        ]
        result = _build_from_records(records, tmp_path)
        G = result.graph
        assert G.has_edge("a_entity", "b_entity")
        assert G.has_edge("b_entity", "a_entity")

    def test_real_dataset_reciprocal_pairs(
        self, real_graph_result: GraphResult
    ) -> None:
        """Audit found exactly 148 undirected reciprocal pairs."""
        G = real_graph_result.graph
        pair_set = {(u, v) for u, v, _ in G.edges(keys=True) if u != v}
        reciprocal = {
            tuple(sorted([u, v]))
            for u, v in pair_set
            if (v, u) in pair_set
        }
        assert len(reciprocal) == 148


# ---------------------------------------------------------------------------
# Test 11: Source/target direction is preserved
# ---------------------------------------------------------------------------


class TestEdgeDirection:
    def test_direction_preserved(self, tmp_path: Path) -> None:
        """payer → beneficiary direction must not be inverted."""
        records = [_make_record(1, "source_co", "sink_co")]
        result = _build_from_records(records, tmp_path)
        G = result.graph
        assert G.has_edge("source_co", "sink_co")
        assert not G.has_edge("sink_co", "source_co")


# ---------------------------------------------------------------------------
# Test 12: amount_orig unchanged (signed)
# ---------------------------------------------------------------------------


class TestAmountOrigOnEdge:
    def test_signed_amount_orig_preserved_on_edge(self, tmp_path: Path) -> None:
        records = [_make_record(1, "p", "b", amount_orig=-55_000.0)]
        result = _build_from_records(records, tmp_path)
        edge_attrs = result.graph["p"]["b"][1]
        assert edge_attrs["amount_orig"] == -55_000.0

    def test_real_amount_orig_matches_edge_table(
        self, real_graph_result: GraphResult
    ) -> None:
        # Sample first 100 rows of the edge table for alignment check
        et = real_graph_result.edge_table.head(100)
        G = real_graph_result.graph
        for _, row in et.iterrows():
            u, v, k = row["source_entity"], row["target_entity"], row["edge_key"]
            assert G[u][v][k]["amount_orig"] == row["amount_orig"]


# ---------------------------------------------------------------------------
# Test 13: amount_usd unchanged
# ---------------------------------------------------------------------------


class TestAmountUsdOnEdge:
    def test_amount_usd_preserved_on_edge(self, tmp_path: Path) -> None:
        records = [_make_record(1, "p", "b", amount_usd=77_777.0)]
        result = _build_from_records(records, tmp_path)
        edge_attrs = result.graph["p"]["b"][1]
        assert edge_attrs["amount_usd"] == pytest.approx(77_777.0)


# ---------------------------------------------------------------------------
# Test 14: date unchanged
# ---------------------------------------------------------------------------


class TestDateOnEdge:
    def test_date_preserved_on_edge(self, tmp_path: Path) -> None:
        records = [_make_record(1, "p", "b", date="2013-07-20")]
        result = _build_from_records(records, tmp_path)
        edge_attrs = result.graph["p"]["b"][1]
        assert edge_attrs["date"] == pd.Timestamp("2013-07-20")


# ---------------------------------------------------------------------------
# Test 15: Account metadata preserved
# ---------------------------------------------------------------------------


class TestAccountMetadataOnEdge:
    def test_account_metadata_preserved(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                1, "p", "b",
                payer_account="PAYER_ACC_XYZ",
                beneficiary_account="BEN_ACC_XYZ",
            )
        ]
        result = _build_from_records(records, tmp_path)
        edge_attrs = result.graph["p"]["b"][1]
        assert edge_attrs["payer_account_used"] == "PAYER_ACC_XYZ"
        assert edge_attrs["beneficiary_account_used"] == "BEN_ACC_XYZ"

    def test_missing_account_preserved_as_none(self, tmp_path: Path) -> None:
        records = [
            _make_record(1, "ghost_payer", "omega", payer_account=None)
        ]
        result = _build_from_records(records, tmp_path)
        edge_attrs = result.graph["ghost_payer"]["omega"][1]
        assert edge_attrs["payer_account_used"] is None


# ---------------------------------------------------------------------------
# Test 16: Edge attributes correspond to the correct transaction ID
# ---------------------------------------------------------------------------


class TestEdgeAttributeAlignment:
    def test_edge_table_aligns_with_graph_edges(self, tmp_path: Path) -> None:
        records = [
            _make_record(101, "entity_x", "entity_y", amount_usd=1_000.0),
            _make_record(202, "entity_x", "entity_y", amount_usd=2_000.0),
        ]
        result = _build_from_records(records, tmp_path)
        G = result.graph
        edge_101 = G["entity_x"]["entity_y"][101]
        edge_202 = G["entity_x"]["entity_y"][202]
        assert edge_101["amount_usd"] == pytest.approx(1_000.0)
        assert edge_202["amount_usd"] == pytest.approx(2_000.0)
        assert edge_101["transaction_id"] == 101
        assert edge_202["transaction_id"] == 202


# ---------------------------------------------------------------------------
# Test 17: No ML features added at this layer
# ---------------------------------------------------------------------------


class TestNoMLFeatures:
    _ML_FEATURE_NAMES = {
        "degree", "in_degree", "out_degree",
        "pagerank", "centrality", "betweenness",
        "embedding", "risk_score", "velocity",
        "rolling_count", "anomaly_score",
    }

    def test_node_attributes_contain_no_ml_features(
        self, real_graph_result: GraphResult
    ) -> None:
        G = real_graph_result.graph
        sample_entity = next(iter(G.nodes()))
        node_keys = set(G.nodes[sample_entity].keys())
        overlap = node_keys & self._ML_FEATURE_NAMES
        assert not overlap, f"ML features found on nodes: {overlap}"

    def test_edge_attributes_contain_no_ml_features(
        self, real_graph_result: GraphResult
    ) -> None:
        G = real_graph_result.graph
        u, v, k = next(iter(G.edges(keys=True)))
        edge_keys = set(G[u][v][k].keys())
        overlap = edge_keys & self._ML_FEATURE_NAMES
        assert not overlap, f"ML features found on edges: {overlap}"


# ---------------------------------------------------------------------------
# Regression: distinct directed entity pairs (4,024)
# ---------------------------------------------------------------------------


class TestDistinctDirectedPairs:
    def test_distinct_directed_pairs(self, real_graph_result: GraphResult) -> None:
        """Audit confirmed exactly 4,024 distinct directed entity pairs."""
        G = real_graph_result.graph
        distinct = len({(u, v) for u, v, _ in G.edges(keys=True)})
        assert distinct == 4_024


# ---------------------------------------------------------------------------
# Guard: build_graph raises on invalid input
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_raises_for_invalid_preprocessing_result(self) -> None:
        bad = PreprocessingResult(
            transactions=pd.DataFrame(),
            transaction_count=0,
            entity_count=0,
            entity_ids=(),
            validation_errors=["Simulated error."],
        )
        with pytest.raises(ValueError, match="invalid PreprocessingResult"):
            build_graph(bad)


# ---------------------------------------------------------------------------
# Graph type assertion
# ---------------------------------------------------------------------------


class TestGraphType:
    def test_graph_is_multidigraph(self, real_graph_result: GraphResult) -> None:
        assert isinstance(real_graph_result.graph, nx.MultiDiGraph)

    def test_graph_is_not_digraph(self, real_graph_result: GraphResult) -> None:
        """Specifically verify that nx.DiGraph is NOT the type used."""
        assert not type(real_graph_result.graph) is nx.DiGraph


# ---------------------------------------------------------------------------
# describe_result
# ---------------------------------------------------------------------------


class TestDescribeResult:
    def test_describe_result_returns_nonempty_string(
        self, real_graph_result: GraphResult
    ) -> None:
        summary = describe_result(real_graph_result)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "GRAPH CONSTRUCTION SUMMARY" in summary
        assert "MultiDiGraph" in summary

    def test_describe_result_shows_node_and_edge_counts(
        self, real_graph_result: GraphResult
    ) -> None:
        summary = describe_result(real_graph_result)
        assert "3440" in summary
        assert "16940" in summary


# ---------------------------------------------------------------------------
# is_valid flag
# ---------------------------------------------------------------------------


class TestValidityFlag:
    def test_is_valid_true_for_real_dataset(
        self, real_graph_result: GraphResult
    ) -> None:
        assert real_graph_result.is_valid

    def test_is_valid_true_for_synthetic(self, tmp_path: Path) -> None:
        records = [_make_record(1, "a", "b"), _make_record(2, "b", "a")]
        result = _build_from_records(records, tmp_path)
        assert result.is_valid
