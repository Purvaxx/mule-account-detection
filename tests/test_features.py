"""
Layer 4 — Feature Engineering Tests
=====================================

Tests cover three feature groups (baseline, temporal, multigraph) plus
integrity checks that confirm Layers 1–3 are untouched.

Numbering aligns with the 26 required checks in the specification:

BASELINE  (1–8)
TEMPORAL  (9–16)
MULTIGRAPH (17–21)
INTEGRITY  (22–26)
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Test fixtures — tiny synthetic graph used for deterministic assertions
# ---------------------------------------------------------------------------

def _make_tiny_graph() -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """Minimal synthetic fixture with 4 entities.

    Entities
    --------
    ALICE  — Company; bank_country EE; sends to BOB and CAROL
    BOB    — Person;  bank_country LV (and EE in one tx); sends to ALICE (reciprocal)
    CAROL  — Company; bank_country CH; only receives
    DAVID  — Invalid; no bank_country; self-loop only

    Transactions (8 total)
    ----------------------
    1. ALICE → BOB,   amount_usd=100, date=2013-01-01
    2. ALICE → BOB,   amount_usd=200, date=2013-01-10  (parallel edge)
    3. ALICE → BOB,   amount_usd=150, date=2013-01-15  (third parallel edge)
    4. ALICE → CAROL, amount_usd=300, date=2013-02-01
    5. BOB   → ALICE, amount_usd= 50, date=2013-01-20  (reciprocal)
    6. BOB   → ALICE, amount_usd= 75, date=2013-01-25  (reciprocal, same-day as tx7 for BOB)
    7. CAROL → CAROL, amount_usd=  0, date=2013-01-25  (self-loop)
    8. DAVID → DAVID, amount_usd=  0, date=2013-03-01  (self-loop, no bank_country)
    """
    G = nx.MultiDiGraph()

    # Entity types & bank countries on nodes
    G.add_node("ALICE", entity_type="Company", bank_country="EE",
               jurisdiction="EE", core=False)
    G.add_node("BOB",   entity_type="Person",  bank_country="LV",
               jurisdiction="LV", core=True)
    G.add_node("CAROL", entity_type="Company", bank_country="CH",
               jurisdiction="CH", core=False)
    G.add_node("DAVID", entity_type="Invalid", bank_country=None,
               jurisdiction=None, core=False)

    rows = [
        # (src, tgt, tx_id, amount_usd, date, pbc, bbc)
        ("ALICE","BOB",  "T1", 100.0, "2013-01-01", "EE", "LV"),
        ("ALICE","BOB",  "T2", 200.0, "2013-01-10", "EE", "LV"),
        ("ALICE","BOB",  "T3", 150.0, "2013-01-15", "EE", "EE"),  # BOB gets EE too
        ("ALICE","CAROL","T4", 300.0, "2013-02-01", "EE", "CH"),
        ("BOB",  "ALICE","T5",  50.0, "2013-01-20", "LV", "EE"),
        ("BOB",  "ALICE","T6",  75.0, "2013-01-25", "LV", "EE"),
        ("CAROL","CAROL","T7",   0.0, "2013-01-25", "CH", "CH"),
        ("DAVID","DAVID","T8",   0.0, "2013-03-01", None, None),
    ]

    edge_rows = []
    for (src, tgt, tx_id, amt, date, pbc, bbc) in rows:
        G.add_edge(src, tgt, key=tx_id,
                   transaction_id=tx_id,
                   amount_usd=amt,
                   amount_orig=amt,
                   amount_orig_currency="USD",
                   amount_eur=str(amt),
                   date=pd.Timestamp(date),
                   purpose="test",
                   payer_account_used=None,
                   beneficiary_account_used=None,
                   payer_type=G.nodes[src]["entity_type"],
                   beneficiary_type=G.nodes[tgt]["entity_type"],
                   payer_jurisdiction=G.nodes[src]["jurisdiction"],
                   beneficiary_jurisdiction=G.nodes[tgt]["jurisdiction"],
                   payer_bank_country=pbc,
                   beneficiary_bank_country=bbc,
                   payer_core=G.nodes[src]["core"],
                   beneficiary_core=G.nodes[tgt]["core"],
                   source_file="test",
                   investigation="test")
        edge_rows.append({
            "source_entity": src,
            "target_entity": tgt,
            "edge_key": tx_id,
            "transaction_id": tx_id,
            "amount_usd": amt,
            "amount_orig": amt,
            "amount_orig_currency": "USD",
            "amount_eur": str(amt),
            "date": pd.Timestamp(date),
            "purpose": "test",
            "payer_account_used": None,
            "beneficiary_account_used": None,
            "payer_bank_country": pbc,
            "beneficiary_bank_country": bbc,
            "payer_core": G.nodes[src]["core"],
            "beneficiary_core": G.nodes[tgt]["core"],
            "source_file": "test",
            "investigation": "test",
        })

    edge_table = pd.DataFrame(edge_rows)
    return G, edge_table


@pytest.fixture
def tiny() -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    return _make_tiny_graph()


# ===========================================================================
# BASELINE TESTS  (1–8)
# ===========================================================================

class TestBaseline:

    def test_01_node_count(self, tiny):
        """(1) Baseline feature rows = number of graph nodes."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        assert result.entity_count == G.number_of_nodes()
        assert len(result.features) == G.number_of_nodes()

    def test_02_feature_dimensions(self, tiny):
        """(2) Feature matrix has the expected number of columns."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        # 5 scalar + entity type cols + country cols
        assert result.features.shape[1] == len(result.feature_names)
        assert "log_in_degree" in result.feature_names
        assert "log_out_degree" in result.feature_names
        assert "log_total_incoming_usd" in result.feature_names
        assert "log_total_outgoing_usd" in result.feature_names
        assert "flow_imbalance" in result.feature_names

    def test_03_entity_type_encoding(self, tiny):
        """(3) One-hot entity type columns are mutually exclusive and sum to 1."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        type_cols = [c for c in result.feature_names if c.startswith("entity_type_")]
        assert len(type_cols) >= 3, "Expected at least Company, Person, Invalid columns"
        row_sums = result.features[type_cols].sum(axis=1)
        # Every entity should have exactly one type set
        assert (row_sums == 1.0).all(), f"Entity type row sums not all 1: {row_sums}"

    def test_04_multi_country_entities(self, tiny):
        """(4) Multi-country entities retain multiple country indicators."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        country_cols = [c for c in result.feature_names if c.startswith("bank_country_")]
        # BOB is associated with EE (from T3: beneficiary_bank_country=EE)
        # and LV (from T1, T2 as beneficiary)
        bob_row = result.features.loc["BOB", country_cols]
        assert "bank_country_EE" in country_cols
        assert "bank_country_LV" in country_cols
        assert result.features.loc["BOB", "bank_country_EE"] == 1.0
        assert result.features.loc["BOB", "bank_country_LV"] == 1.0

    def test_05_log_degree_calculations(self, tiny):
        """(5) log-degree values are computed correctly."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        # ALICE: out_degree=4 (T1,T2,T3,T4), in_degree=2 (T5,T6)
        assert abs(result.features.loc["ALICE", "log_out_degree"] - math.log10(4 + 1)) < 1e-9
        assert abs(result.features.loc["ALICE", "log_in_degree"]  - math.log10(2 + 1)) < 1e-9
        # CAROL: out_degree=1 (T7 self-loop), in_degree=2 (T4 + T7 self-loop)
        assert abs(result.features.loc["CAROL", "log_out_degree"] - math.log10(1 + 1)) < 1e-9

    def test_06_log_usd_calculations(self, tiny):
        """(6) log-USD values match manual totals."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        # ALICE outgoing: T1(100)+T2(200)+T3(150)+T4(300)=750
        assert abs(result.features.loc["ALICE", "log_total_outgoing_usd"] - math.log10(750 + 1)) < 1e-6
        # ALICE incoming: T5(50)+T6(75)=125
        assert abs(result.features.loc["ALICE", "log_total_incoming_usd"] - math.log10(125 + 1)) < 1e-6

    def test_07_flow_imbalance(self, tiny):
        """(7) Flow imbalance is in [-1, 1] and directionally correct."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        fi = result.features["flow_imbalance"]
        assert ((fi >= -1.0) & (fi <= 1.0)).all()
        # CAROL: only receives T4 (300) + self-loop (0, same in/out); net inflow bias
        assert result.features.loc["CAROL", "flow_imbalance"] >= 0.0
        # ALICE: sends 750, receives 125 → net outflow bias → negative imbalance
        assert result.features.loc["ALICE", "flow_imbalance"] < 0.0

    def test_08_no_labels_in_baseline(self, tiny):
        """(8) Baseline feature matrix contains no label columns."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        result = compute_baseline(G, et)
        forbidden = {"core", "payer_core", "beneficiary_core",
                     "source_file", "investigation"}
        for col in result.feature_names:
            assert col not in forbidden, f"Forbidden column found: {col}"
            assert "account" not in col.lower()


# ===========================================================================
# TEMPORAL TESTS  (9–16)
# ===========================================================================

class TestTemporal:

    def _entity_ids(self, G):
        return sorted(G.nodes())

    def test_09_activity_7d(self, tiny):
        """(9) 7-day activity counts transactions in the trailing 7-day window."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        # Use 2013-01-20 as as_of; trailing 7 days = [2013-01-14 .. 2013-01-20]
        # T3 (ALICE→BOB, 2013-01-15) and T5 (BOB→ALICE, 2013-01-20) qualify
        result = compute_temporal(et, entities, as_of=pd.Timestamp("2013-01-20"))
        # ALICE sent T3 (in 7d window) and received T5 → 2 tx in 7d
        assert result.features.loc["ALICE", "activity_7d"] == 2.0
        # BOB received T3 and sent T5 → 2 tx in 7d
        assert result.features.loc["BOB",   "activity_7d"] == 2.0

    def test_10_activity_30d(self, tiny):
        """(10) 30-day activity covers a wider window."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        # as_of=2013-01-25; trailing 30 days = [2012-12-26 .. 2013-01-25]
        # All Jan transactions qualify
        result = compute_temporal(et, entities, as_of=pd.Timestamp("2013-01-25"))
        # ALICE: T1,T2,T3 sent + T5,T6 received = 5 tx in 30d
        assert result.features.loc["ALICE", "activity_30d"] == 5.0

    def test_11_inter_transaction_time(self, tiny):
        """(11) Mean IAT is computed from sorted dates, including both roles."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        result = compute_temporal(et, entities)
        # BOB's transactions (all roles): T1(Jan1), T2(Jan10), T3(Jan15), T5(Jan20), T6(Jan25)
        # Diffs: 9, 5, 5, 5 → mean = 6.0
        assert abs(result.features.loc["BOB", "mean_iat"] - 6.0) < 1e-6

    def test_12_burstiness(self, tiny):
        """(12) Burstiness (CV of IAT) is finite for entities with >= 2 tx."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        result = compute_temporal(et, entities)
        # DAVID has only 1 transaction (self-loop) → NaN burstiness
        assert np.isnan(result.features.loc["DAVID", "burstiness"])
        # ALICE has >= 2 transactions → finite burstiness
        assert np.isfinite(result.features.loc["ALICE", "burstiness"])

    def test_13_active_day_ratio(self, tiny):
        """(13) Active-day ratio is in (0, 1]."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        result = compute_temporal(et, entities)
        adr = result.features["active_day_ratio"]
        valid = adr.dropna()
        assert (valid > 0).all()
        assert (valid <= 1.0 + 1e-9).all()
        # DAVID: single date → span=0 → ratio=1.0
        assert result.features.loc["DAVID", "active_day_ratio"] == pytest.approx(1.0)

    def test_14_same_day_cooccurrence(self, tiny):
        """(14) Same-day co-occurrence is 1 only when entity is both payer AND beneficiary on the same date.

        CAROL has a self-loop (T7) on Jan 25: it appears as both payer and beneficiary on that
        exact date, so same_day_cooccurrence must be 1.  This does NOT imply causality; both
        roles exist on the same calendar day at day-level precision.
        """
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        result = compute_temporal(et, entities)
        # BOB sends T5 (Jan 20) and T6 (Jan 25); receives T1(Jan1), T2(Jan10), T3(Jan15)
        # No date in payer role overlaps with a date in beneficiary role → 0
        assert result.features.loc["BOB",   "same_day_cooccurrence"] == 0.0
        # CAROL has self-loop T7 on Jan 25; it is both payer and beneficiary on that date → 1
        assert result.features.loc["CAROL", "same_day_cooccurrence"] == 1.0
        # ALICE: sends on Jan1, Jan10, Jan15, Feb1; receives on Jan20, Jan25 — no overlap → 0
        assert result.features.loc["ALICE", "same_day_cooccurrence"] == 0.0

    def test_15_no_intraday_ordering(self, tiny):
        """(15) Temporal feature produces same result for two transactions on the same day — no intra-day claims."""
        from src.features.temporal import compute_temporal
        # Craft a scenario: entity sends and receives on the exact same date
        G2, et2 = _make_tiny_graph()
        # Add an extra transaction where ALICE receives from CAROL on Feb 1
        # (ALICE already sends T4 to CAROL on Feb 1)
        et2 = pd.concat([et2, pd.DataFrame([{
            "source_entity": "CAROL",
            "target_entity": "ALICE",
            "edge_key": "T9",
            "transaction_id": "T9",
            "amount_usd": 10.0,
            "amount_orig": 10.0,
            "amount_orig_currency": "USD",
            "amount_eur": "10.0",
            "date": pd.Timestamp("2013-02-01"),
            "purpose": "test",
            "payer_account_used": None,
            "beneficiary_account_used": None,
            "payer_bank_country": "CH",
            "beneficiary_bank_country": "EE",
            "payer_core": False,
            "beneficiary_core": False,
            "source_file": "test",
            "investigation": "test",
        }])], ignore_index=True)
        G2.add_edge("CAROL", "ALICE", key="T9", transaction_id="T9",
                    amount_usd=10.0, date=pd.Timestamp("2013-02-01"),
                    payer_bank_country="CH", beneficiary_bank_country="EE")

        entities = sorted(G2.nodes())
        result = compute_temporal(et2, entities)
        # ALICE now sends T4 to CAROL AND receives T9 from CAROL on the same day
        # same_day_cooccurrence should be 1; but this does NOT mean inflow preceded outflow
        assert result.features.loc["ALICE", "same_day_cooccurrence"] == 1.0

    def test_16_causal_mode_excludes_future(self, tiny):
        """(16) Causal as_of mode returns 0 activity for transactions after the cut-off."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        entities = self._entity_ids(G)
        # as_of=2013-01-01: only T1 exists
        result = compute_temporal(et, entities, as_of=pd.Timestamp("2013-01-01"))
        # ALICE: sent T1 (Jan 1), no future tx included
        assert result.features.loc["ALICE", "activity_7d"]  == 1.0
        assert result.features.loc["ALICE", "activity_30d"] == 1.0
        # T4 (Feb 1) must NOT appear
        result_full = compute_temporal(et, entities)
        alice_30d_full = result_full.features.loc["ALICE", "activity_30d"]
        # Full dataset should have more activity than the causal cut-off
        assert alice_30d_full >= result.features.loc["ALICE", "activity_30d"]


# ===========================================================================
# MULTIGRAPH TESTS  (17–21)
# ===========================================================================

class TestMultigraph:

    def test_17_repeated_counterparty_ratio(self, tiny):
        """(17) Repeated-counterparty ratio is total_edges / unique_counterparties."""
        from src.features.multigraph import compute_multigraph
        G, et = tiny
        result = compute_multigraph(G, et)
        # ALICE out: 4 edges to 2 unique targets (BOB x3, CAROL x1)
        # out_repeated_counterparty_ratio = 4/2 = 2.0
        assert abs(result.features.loc["ALICE", "out_repeated_counterparty_ratio"] - 2.0) < 1e-9
        # BOB out: 2 edges to 1 unique target (ALICE x2) → ratio=2.0
        assert abs(result.features.loc["BOB",   "out_repeated_counterparty_ratio"] - 2.0) < 1e-9

    def test_18_parallel_edge_count(self, tiny):
        """(18) max_parallel_edge_count reflects the most repeated directed pair."""
        from src.features.multigraph import compute_multigraph
        G, et = tiny
        result = compute_multigraph(G, et)
        # ALICE → BOB has 3 parallel edges; ALICE → CAROL has 1
        # ALICE's max_parallel_edge_count = 3
        assert result.features.loc["ALICE", "max_parallel_edge_count"] == 3.0

    def test_19_parallel_temporal_dispersion(self, tiny):
        """(19) Temporal dispersion is > 0 for pairs with multiple transactions on different dates."""
        from src.features.multigraph import compute_multigraph
        G, et = tiny
        result = compute_multigraph(G, et)
        # ALICE → BOB: T1(Jan1), T2(Jan10), T3(Jan15) → std is non-zero
        assert result.features.loc["ALICE", "parallel_temporal_dispersion"] > 0.0
        # DAVID has only a single self-loop → dispersion = 0.0 (no multi-edge pairs)
        assert result.features.loc["DAVID", "parallel_temporal_dispersion"] == 0.0

    def test_20_reciprocal_behaviour(self, tiny):
        """(20) Reciprocal counts detect A→B and B→A pairs."""
        from src.features.multigraph import compute_multigraph
        G, et = tiny
        result = compute_multigraph(G, et)
        # ALICE → BOB and BOB → ALICE both exist → reciprocal count = 1
        assert result.features.loc["ALICE", "reciprocal_out_count"] == 1.0
        assert result.features.loc["BOB",   "reciprocal_out_count"] == 1.0
        # CAROL → CAROL self-loop only → no A→B and B→A (well, self-loop is reciprocal with itself)
        # DAVID has no successors other than itself → 0 or 1 (self-loop)
        # Key: CAROL's only successor is CAROL (self-loop); CAROL is in predecessors too
        # so reciprocal_out_count for CAROL = 1 (self-loop)
        # This is correct: self-loop is a valid reciprocal by definition.
        # But DAVID also has a self-loop.
        assert result.features.loc["ALICE", "reciprocal_in_count"] == 1.0

    def test_21_parallel_edges_preserved(self, tiny):
        """(21) Multigraph features do NOT collapse parallel edges; count > 1 possible."""
        from src.features.multigraph import compute_multigraph
        G, et = tiny
        result = compute_multigraph(G, et)
        # Verify that max_parallel_edge_count can be > 1 (proves multi-edges preserved)
        assert result.features["max_parallel_edge_count"].max() > 1.0


# ===========================================================================
# INTEGRITY TESTS  (22–26)
# ===========================================================================

class TestIntegrity:

    def test_22_transaction_count_unchanged(self, tiny):
        """(22) Baseline computation does not drop or add transactions."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        original_count = len(et)
        compute_baseline(G, et)
        assert len(et) == original_count

    def test_23_graph_unchanged(self, tiny):
        """(23) Graph node/edge count is unchanged after feature computation."""
        from src.features.baseline import compute_baseline
        from src.features.temporal import compute_temporal
        from src.features.multigraph import compute_multigraph
        G, et = tiny
        n_before = G.number_of_nodes()
        e_before = G.number_of_edges()
        compute_baseline(G, et)
        compute_temporal(et, sorted(G.nodes()))
        compute_multigraph(G, et)
        assert G.number_of_nodes() == n_before
        assert G.number_of_edges() == e_before

    def test_24_raw_amounts_unchanged(self, tiny):
        """(24) amount_usd and amount_orig are not altered by feature computation."""
        from src.features.baseline import compute_baseline
        G, et = tiny
        orig_usd  = et["amount_usd"].copy()
        orig_orig = et["amount_orig"].copy()
        compute_baseline(G, et)
        pd.testing.assert_series_equal(et["amount_usd"],  orig_usd)
        pd.testing.assert_series_equal(et["amount_orig"], orig_orig)

    def test_25_raw_dates_unchanged(self, tiny):
        """(25) dates are not altered by feature computation."""
        from src.features.temporal import compute_temporal
        G, et = tiny
        orig_dates = et["date"].copy()
        compute_temporal(et, sorted(G.nodes()))
        pd.testing.assert_series_equal(et["date"], orig_dates)

    def test_26_existing_test_suites_not_broken(self):
        """(26) Layer 4 feature modules can be imported without breaking Layer 1–3 imports."""
        # If any of these imports fail, the test fails.
        from src.data.ingestion     import load_transactions   # noqa: F401
        from src.data.preprocessing import preprocess          # noqa: F401
        from src.graph.builder      import build_graph         # noqa: F401
        from src.features.baseline  import compute_baseline    # noqa: F401
        from src.features.temporal  import compute_temporal    # noqa: F401
        from src.features.multigraph import compute_multigraph  # noqa: F401


# ===========================================================================
# Integration smoke test against the real dataset
# (skipped if dkdata.json is not present — CI-friendly)
# ===========================================================================

import os

@pytest.mark.skipif(
    not os.path.exists("../dkdata.json") and not os.path.exists("data/dkdata.json") and not os.path.exists("dkdata.json"),
    reason="dkdata.json not available",
)
class TestRealDataset:

    @pytest.fixture(scope="class")
    def real_result(self):
        from tests.test_utils import get_cached_real_dataset_graph
        return get_cached_real_dataset_graph()

    # Each of these is the same computation over the same reference graph that
    # tests/test_utils.py already memoizes for the rest of the suite, so they
    # take it from there and get their own copy rather than recomputing it.
    @pytest.fixture(scope="class")
    def real_baseline(self, real_result):
        from tests.test_utils import get_cached_baseline
        return get_cached_baseline()

    @pytest.fixture(scope="class")
    def real_temporal(self, real_result):
        from tests.test_utils import get_cached_temporal
        return get_cached_temporal()

    @pytest.fixture(scope="class")
    def real_multigraph(self, real_result):
        from tests.test_utils import get_cached_multigraph
        return get_cached_multigraph()

    def test_real_baseline_node_count(self, real_baseline):
        assert real_baseline.entity_count == 3440

    def test_real_baseline_no_inf(self, real_baseline):
        assert real_baseline.inf_count == 0

    def test_real_temporal_no_inf(self, real_temporal):
        assert real_temporal.inf_count == 0

    def test_real_multigraph_no_inf(self, real_multigraph):
        assert real_multigraph.inf_count == 0

    def test_real_edge_count_preserved(self, real_result):
        assert real_result.graph.number_of_edges() == 16940

    def test_real_node_count_preserved(self, real_result):
        assert real_result.graph.number_of_nodes() == 3440
