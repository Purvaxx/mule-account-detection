"""
Tests for the scalable AML feature families (``src.experiments.aml_features``).

Every case is a hand-built transaction table whose correct answer is arithmetic
that can be checked by reading the fixture.  Grouped by the property protected:

* arithmetic       - each column computes what its name and docstring claim
* undefined policy - a statistic with no meaning is ``NaN``, never 0, never a
                     sentinel
* temporal safety  - a family given a truncated edge table cannot describe a
                     transaction outside it, and the ``as_of``-dependent columns
                     move with ``as_of`` while the rest do not
* separability     - the families are independently registered so an ablation
                     can turn one off
* reuse            - families sharing a request share the expensive
                     intermediates rather than recomputing them
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.features import (
    FeatureRequest,
    available_feature_sets,
    build_feature_matrix,
    get_feature_set,
)

pytestmark = pytest.mark.fast

AML_FAMILIES = [
    "aml_txn_v1",
    "aml_temporal_v1",
    "aml_flow_v1",
    "aml_network_v1",
    "aml_cycle_v1",
    "aml_context_v1",
]

BASE = pd.Timestamp("2024-01-01")


def _edges(rows: list[tuple]) -> pd.DataFrame:
    """Build an edge table from ``(src, dst, amount, day_offset_hours)`` tuples."""
    frame = pd.DataFrame(rows, columns=["source_entity", "target_entity", "amount", "hours"])
    frame["date"] = BASE + pd.to_timedelta(frame.pop("hours"), unit="h")
    frame["edge_key"] = np.arange(len(frame))
    return frame


def _request(edges: pd.DataFrame, entities: list[str], **kwargs) -> FeatureRequest:
    return FeatureRequest(
        entity_ids=entities, edge_table=edges, graph=None, **kwargs
    )


def _matrix(edges: pd.DataFrame, entities: list[str], names: list[str], **kwargs):
    return build_feature_matrix(_request(edges, entities, **kwargs), names).values


# ---------------------------------------------------------------------------
# A fixture whose answers are known by hand
# ---------------------------------------------------------------------------
#
#   PASS receives 100 from SRC at hour 0 and sends 100 to DST at hour 2.
#   It is the textbook pass-through: one in, one out, two-hour turnaround,
#   a matching amount, exactly two counterparties.
#   LOOP1 -> LOOP2 -> LOOP3 -> LOOP1 is a directed 3-cycle.
#   MUTUAL_A and MUTUAL_B pay each other: one reciprocal pair.
#   QUIET sends one 500 payment and does nothing else.

_HAND_ROWS = [
    ("SRC", "PASS", 100.0, 0),
    ("PASS", "DST", 100.0, 2),
    ("LOOP1", "LOOP2", 10.0, 10),
    ("LOOP2", "LOOP3", 10.0, 20),
    ("LOOP3", "LOOP1", 10.0, 30),
    ("MUTUAL_A", "MUTUAL_B", 50.0, 40),
    ("MUTUAL_B", "MUTUAL_A", 50.0, 50),
    ("QUIET", "DST", 500.0, 60),
]
_HAND_ENTITIES = [
    "DST", "LOOP1", "LOOP2", "LOOP3", "MUTUAL_A", "MUTUAL_B", "PASS", "QUIET", "SRC",
]


@pytest.fixture
def hand() -> tuple[pd.DataFrame, list[str]]:
    return _edges(_HAND_ROWS), list(_HAND_ENTITIES)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_every_family_is_registered_separately(self):
        registered = available_feature_sets()
        for name in AML_FAMILIES:
            assert name in registered, f"{name} is not registered"

    def test_families_declare_distinct_information_sources(self):
        """The ablation only means something if the families are not synonyms."""
        declared = {name: get_feature_set(name).family for name in AML_FAMILIES}
        assert declared["aml_txn_v1"] == "transaction"
        assert declared["aml_temporal_v1"] == "temporal"
        assert declared["aml_flow_v1"] == "flow"
        assert declared["aml_network_v1"] == "network"
        assert declared["aml_cycle_v1"] == "structure"

    def test_no_family_requires_a_networkx_graph(self):
        """A graph object over millions of edges is what these exist to avoid."""
        for name in AML_FAMILIES:
            assert get_feature_set(name).requires_graph is False

    def test_column_names_are_prefixed_with_the_family(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_txn_v1", "aml_flow_v1"])
        assert all("::" in str(c) for c in values.columns)
        prefixes = {str(c).split("::")[0] for c in values.columns}
        assert prefixes == {"aml_txn_v1", "aml_flow_v1"}

    def test_legacy_aml_core_v1_is_untouched(self, hand):
        """aml_core_v1 is pinned by existing artifacts and must keep its columns."""
        edges, entities = hand
        edges = edges.rename(columns={"amount": "amount_usd"})
        values = _matrix(edges, entities, ["aml_core_v1"])
        assert values.shape[1] == 29
        assert "aml_core_v1::min_in_to_out_gap_days" in values.columns


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestTransactionFamily:
    def test_counts_and_sums(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_txn_v1"])
        assert values.loc["PASS", "aml_txn_v1::txn_in"] == 1
        assert values.loc["PASS", "aml_txn_v1::txn_out"] == 1
        assert values.loc["PASS", "aml_txn_v1::txn_total"] == 2
        assert values.loc["DST", "aml_txn_v1::txn_in"] == 2
        assert values.loc["DST", "aml_txn_v1::txn_out"] == 0
        assert values.loc["DST", "aml_txn_v1::amount_in_sum"] == pytest.approx(600.0)

    def test_outgoing_sum_is_nan_for_a_receive_only_entity(self, hand):
        """DST never sends; its outgoing total is undefined, not zero."""
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_txn_v1"])
        assert np.isnan(values.loc["DST", "aml_txn_v1::amount_out_sum"])
        assert np.isnan(values.loc["DST", "aml_txn_v1::amount_out_mean"])

    def test_dispersion_survives_large_near_constant_amounts(self):
        """The structuring case is exactly where a one-pass variance collapses.

        Twelve transactions of 5,000,000 +/- 1: the sum of squares reaches 3e14,
        and ``E[X^2] - E[X]^2`` loses most of its significant digits there. The
        true population standard deviation is 1.0.
        """
        amounts = [5_000_000.0 + (1.0 if i % 2 else -1.0) for i in range(12)]
        edges = _edges([("A", "B", a, i) for i, a in enumerate(amounts)])
        values = _matrix(edges, ["A", "B"], ["aml_txn_v1"])
        assert values.loc["A", "aml_txn_v1::amount_std"] == pytest.approx(1.0, rel=1e-6)
        assert values.loc["A", "aml_txn_v1::amount_cv"] == pytest.approx(
            1.0 / 5_000_000.0, rel=1e-6
        )

    def test_dispersion_matches_numpy_on_ordinary_values(self):
        import numpy as np

        amounts = [10.0, 200.0, 35.0, 4000.0, 7.0]
        edges = _edges([("A", "B", a, i) for i, a in enumerate(amounts)])
        values = _matrix(edges, ["A", "B"], ["aml_txn_v1"])
        assert values.loc["A", "aml_txn_v1::amount_std"] == pytest.approx(
            float(np.std(amounts))
        )

    def test_dispersion_is_undefined_for_a_single_transaction(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_txn_v1"])
        assert values.loc["QUIET", "aml_txn_v1::txn_total"] == 1
        assert np.isnan(values.loc["QUIET", "aml_txn_v1::amount_std"])

    def test_round_amount_share(self):
        edges = _edges(
            [("A", "B", 100.0, 0), ("A", "B", 250.0, 1), ("A", "B", 133.7, 2), ("A", "B", 400.0, 3)]
        )
        values = _matrix(edges, ["A", "B"], ["aml_txn_v1"])
        # 100, 400 are multiples of 100; 250 and 133.7 are not.
        assert values.loc["A", "aml_txn_v1::round_amount_share"] == pytest.approx(0.5)

    def test_top1_share(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_txn_v1"])
        # DST receives 100 and 500; the largest is 500 of 600.
        assert values.loc["DST", "aml_txn_v1::amount_top1_share"] == pytest.approx(
            500.0 / 600.0
        )


class TestTemporalFamily:
    def test_inter_arrival_statistics(self):
        edges = _edges(
            [("A", "B", 1.0, 0), ("A", "B", 1.0, 24), ("A", "B", 1.0, 48)]
        )
        values = _matrix(edges, ["A", "B"], ["aml_temporal_v1"])
        assert values.loc["A", "aml_temporal_v1::mean_iat_hours"] == pytest.approx(24.0)
        assert values.loc["A", "aml_temporal_v1::std_iat_hours"] == pytest.approx(0.0)
        assert values.loc["A", "aml_temporal_v1::active_days"] == 3
        assert values.loc["A", "aml_temporal_v1::span_days"] == pytest.approx(2.0)

    def test_burstiness_sign(self):
        """Perfectly regular timing is -1; clustered timing is positive."""
        regular = _edges([("A", "B", 1.0, h) for h in (0, 24, 48, 72, 96)])
        bursty = _edges([("A", "B", 1.0, h) for h in (0, 1, 2, 3, 400)])
        regular_value = _matrix(regular, ["A", "B"], ["aml_temporal_v1"]).loc[
            "A", "aml_temporal_v1::burstiness"
        ]
        bursty_value = _matrix(bursty, ["A", "B"], ["aml_temporal_v1"]).loc[
            "A", "aml_temporal_v1::burstiness"
        ]
        assert regular_value == pytest.approx(-1.0)
        assert bursty_value > 0.0

    def test_inter_arrival_is_undefined_with_one_event(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_temporal_v1"])
        assert np.isnan(values.loc["QUIET", "aml_temporal_v1::mean_iat_hours"])
        assert np.isnan(values.loc["QUIET", "aml_temporal_v1::burstiness"])

    def test_second_half_share_detects_a_dormant_to_active_shift(self):
        early = _edges([("A", "B", 1.0, h) for h in (0, 1, 2, 3)] + [("A", "B", 1.0, 1000)])
        late = _edges([("A", "B", 1.0, 0)] + [("A", "B", 1.0, h) for h in (997, 998, 999, 1000)])
        early_share = _matrix(early, ["A", "B"], ["aml_temporal_v1"]).loc[
            "A", "aml_temporal_v1::second_half_share"
        ]
        late_share = _matrix(late, ["A", "B"], ["aml_temporal_v1"]).loc[
            "A", "aml_temporal_v1::second_half_share"
        ]
        assert late_share > early_share

    def test_weekend_share_is_saturday_and_sunday(self):
        """Off by one here silently reports Friday-and-Saturday as the weekend."""
        # BASE is 2024-01-01, a Monday. Hours 0..144 walk Monday to Sunday.
        days = {
            "Mon": 0, "Tue": 24, "Wed": 48, "Thu": 72,
            "Fri": 96, "Sat": 120, "Sun": 144,
        }
        for name, hours in days.items():
            edges = _edges([("A", "B", 1.0, hours)])
            share = _matrix(edges, ["A", "B"], ["aml_temporal_v1"]).loc[
                "A", "aml_temporal_v1::weekend_share"
            ]
            expected = 1.0 if name in ("Sat", "Sun") else 0.0
            assert share == pytest.approx(expected), f"{name} classified wrongly"

    def test_recency_follows_as_of(self, hand):
        """Only the two 'now'-relative columns may move when as_of moves."""
        edges, entities = hand
        early = _matrix(edges, entities, ["aml_temporal_v1"])
        later = _matrix(
            edges, entities, ["aml_temporal_v1"], as_of=BASE + pd.Timedelta(days=30)
        )
        moved = {
            str(c)
            for c in early.columns
            if not np.allclose(
                early[c].to_numpy(), later[c].to_numpy(), equal_nan=True
            )
        }
        assert moved == {
            "aml_temporal_v1::recency_days",
            "aml_temporal_v1::tenure_days",
        }


class TestFlowFamily:
    def test_fan_and_pass_through(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_flow_v1"])
        assert values.loc["PASS", "aml_flow_v1::fan_in"] == 1
        assert values.loc["PASS", "aml_flow_v1::fan_out"] == 1
        # 100 in and 100 out is a perfect pass-through and zero imbalance.
        assert values.loc["PASS", "aml_flow_v1::pass_through_ratio"] == pytest.approx(1.0)
        assert values.loc["PASS", "aml_flow_v1::flow_imbalance"] == pytest.approx(0.0)

    def test_turnaround_latency(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_flow_v1"])
        assert values.loc["PASS", "aml_flow_v1::min_in_to_out_hours"] == pytest.approx(2.0)
        assert values.loc["PASS", "aml_flow_v1::rapid_turnover_share"] == pytest.approx(1.0)

    def test_instant_and_rapid_turnaround_are_different_bands(self):
        """PASS turns money around in two hours: rapid, but not instant."""
        edges = _edges([("X", "M", 100.0, 0), ("M", "Y", 100.0, 2)])
        values = _matrix(edges, ["M", "X", "Y"], ["aml_flow_v1"])
        assert values.loc["M", "aml_flow_v1::rapid_turnover_share"] == pytest.approx(1.0)
        assert values.loc["M", "aml_flow_v1::instant_turnaround_share"] == pytest.approx(0.0)

        immediate = _edges([("X", "M", 100.0, 0), ("M", "Y", 100.0, 0)])
        values = _matrix(immediate, ["M", "X", "Y"], ["aml_flow_v1"])
        assert values.loc["M", "aml_flow_v1::instant_turnaround_share"] == pytest.approx(1.0)

    def test_turnaround_is_undefined_without_a_preceding_inflow(self, hand):
        """SRC sends before it ever receives, so it has no turnaround at all."""
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_flow_v1"])
        assert np.isnan(values.loc["SRC", "aml_flow_v1::min_in_to_out_hours"])

    def test_turnaround_does_not_match_across_entities(self):
        """An inflow to one account must never satisfy an outflow from another."""
        edges = _edges([("X", "A", 100.0, 0), ("B", "Y", 100.0, 1)])
        values = _matrix(edges, ["A", "B", "X", "Y"], ["aml_flow_v1"])
        assert np.isnan(values.loc["B", "aml_flow_v1::min_in_to_out_hours"])

    def test_flow_imbalance_signs_the_direction(self, hand):
        """+1 for a pure receiver, -1 for a pure sender."""
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_flow_v1"])
        assert values.loc["SRC", "aml_flow_v1::flow_imbalance"] == pytest.approx(-1.0)
        assert values.loc["DST", "aml_flow_v1::flow_imbalance"] == pytest.approx(1.0)

    def test_fan_counts_ignore_a_self_loop(self):
        """A self-loop must not make an account its own counterparty."""
        edges = _edges([("A", "A", 1.0, 0), ("A", "B", 1.0, 1)])
        values = _matrix(edges, ["A", "B"], ["aml_flow_v1", "aml_network_v1"])
        assert values.loc["A", "aml_flow_v1::fan_out"] == 1
        assert values.loc["A", "aml_flow_v1::fan_in"] == 0
        assert (
            values.loc["A", "aml_network_v1::counterparty_count"]
            == values.loc["A", "aml_flow_v1::fan_out"]
        )


class TestNetworkFamily:
    def test_counterparty_and_reciprocity(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_network_v1"])
        assert values.loc["PASS", "aml_network_v1::counterparty_count"] == 2
        assert values.loc["MUTUAL_A", "aml_network_v1::reciprocal_partners"] == 1
        assert values.loc["MUTUAL_A", "aml_network_v1::reciprocal_ratio"] == pytest.approx(1.0)
        assert values.loc["PASS", "aml_network_v1::reciprocal_partners"] == 0

    def test_component_size(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_network_v1"])
        # SRC - PASS - DST - QUIET is one component of four.
        assert values.loc["PASS", "aml_network_v1::component_size"] == 4
        # The three-node loop is its own component.
        assert values.loc["LOOP1", "aml_network_v1::component_size"] == 3
        assert values.loc["MUTUAL_A", "aml_network_v1::component_size"] == 2

    def test_repeated_pair_concentration(self):
        edges = _edges([("A", "B", 1.0, h) for h in range(9)] + [("A", "C", 1.0, 20)])
        values = _matrix(edges, ["A", "B", "C"], ["aml_network_v1"])
        assert values.loc["A", "aml_network_v1::top_counterparty_share"] == pytest.approx(0.9)
        assert values.loc["A", "aml_network_v1::counterparty_count"] == 2

    def test_neighbour_aggregation_summarises_counterparties(self):
        """A quiet account next to a busy one must look different from one that is not."""
        edges = _edges(
            [("HUB", f"L{i}", 1.0, i) for i in range(10)] + [("LONE_A", "LONE_B", 1.0, 50)]
        )
        entities = ["HUB", "LONE_A", "LONE_B"] + [f"L{i}" for i in range(10)]
        values = _matrix(edges, entities, ["aml_network_v1"])
        assert values.loc["L0", "aml_network_v1::neighbour_txn_mean"] == 10
        assert values.loc["LONE_B", "aml_network_v1::neighbour_txn_mean"] == 1


class TestCycleFamily:
    def test_directed_three_cycle_is_counted(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_cycle_v1"])
        for node in ("LOOP1", "LOOP2", "LOOP3"):
            assert values.loc[node, "aml_cycle_v1::cycle3_count"] == 1
        assert values.loc["PASS", "aml_cycle_v1::cycle3_count"] == 0

    def test_no_column_appears_in_two_families(self, hand):
        """A duplicated column inflates its own importance in every model."""
        edges, entities = hand
        values = _matrix(edges, entities, AML_FAMILIES)
        short = [str(c).split("::")[-1] for c in values.columns]
        duplicates = {name for name in short if short.count(name) > 1}
        assert duplicates == set(), f"duplicated short names: {sorted(duplicates)}"

    def test_two_path_through_counts_pass_through_position(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_cycle_v1"])
        assert values.loc["PASS", "aml_cycle_v1::two_path_through"] == 1
        assert values.loc["SRC", "aml_cycle_v1::two_path_through"] == 0

    def test_layering_events_respect_time_order_and_window(self):
        """Money leaving before it arrived is not layering; too late is not either."""
        prompt = _edges([("X", "M", 100.0, 0), ("M", "Y", 100.0, 5)])
        stale = _edges([("X", "M", 100.0, 0), ("M", "Y", 100.0, 5000)])
        reversed_order = _edges([("M", "Y", 100.0, 0), ("X", "M", 100.0, 5)])
        entities = ["M", "X", "Y"]
        assert _matrix(prompt, entities, ["aml_cycle_v1"]).loc[
            "M", "aml_cycle_v1::layering_events"
        ] == 1
        assert _matrix(stale, entities, ["aml_cycle_v1"]).loc[
            "M", "aml_cycle_v1::layering_events"
        ] == 0
        assert _matrix(reversed_order, entities, ["aml_cycle_v1"]).loc[
            "M", "aml_cycle_v1::layering_events"
        ] == 0

    def test_self_loops_do_not_close_a_cycle(self):
        """An account paying itself is not a counterparty and not a triangle."""
        edges = _edges(
            [("A", "A", 1.0, 0), ("A", "B", 1.0, 1), ("B", "A", 1.0, 2)]
        )
        values = _matrix(edges, ["A", "B"], ["aml_cycle_v1", "aml_network_v1"])
        assert values.loc["A", "aml_cycle_v1::cycle3_count"] == 0
        assert values.loc["A", "aml_network_v1::reciprocal_partners"] == 1
        assert values.loc["A", "aml_network_v1::counterparty_count"] == 1

    def test_amount_match_distinguishes_the_same_money(self):
        same = _edges([("X", "M", 100.0, 0), ("M", "Y", 98.0, 5)])
        different = _edges([("X", "M", 100.0, 0), ("M", "Y", 5.0, 5)])
        entities = ["M", "X", "Y"]
        assert _matrix(same, entities, ["aml_cycle_v1"]).loc[
            "M", "aml_cycle_v1::layering_amount_match_share"
        ] == pytest.approx(1.0)
        assert _matrix(different, entities, ["aml_cycle_v1"]).loc[
            "M", "aml_cycle_v1::layering_amount_match_share"
        ] == pytest.approx(0.0)


class TestContextFamily:
    def test_absent_columns_produce_nan_not_an_invented_value(self, hand):
        edges, entities = hand
        values = _matrix(edges, entities, ["aml_context_v1"])
        assert values.shape[1] == 6
        assert values["aml_context_v1::cross_border_share"].isna().all()

    def test_ibm_column_spellings_resolve(self, hand):
        """IBM spells these with spaces; they matched nothing until 2026-09-06.

        The family fails *silently* when it matches nothing - six all-NaN
        columns, no error - so every supported spelling is pinned here. IBM
        HI-Small ran an entire benchmark with its context features dead, and
        Payment Format is one of that dataset's strongest label correlates.
        """
        from src.experiments.aml_features import (
            _CONTEXT_CHANNEL_COLUMNS,
            _CONTEXT_CURRENCY_PAIR,
            _first_pair,
            _first_present,
        )

        edges, entities = hand
        edges = edges.copy()
        edges["Payment Format"] = ["ACH"] * len(edges)
        edges["Payment Currency"] = ["US Dollar"] * len(edges)
        edges["Receiving Currency"] = ["Euro"] * len(edges)

        assert _first_present(edges, _CONTEXT_CHANNEL_COLUMNS) == "Payment Format"
        assert _first_pair(edges, _CONTEXT_CURRENCY_PAIR) == (
            "Payment Currency",
            "Receiving Currency",
        )
        values = _matrix(edges, entities, ["aml_context_v1"])
        assert values["aml_context_v1::n_payment_types"].notna().any()
        assert values["aml_context_v1::currency_change_share"].notna().any()

    def test_unresolved_context_is_recorded_not_merely_silent(self, hand):
        """A family that resolves nothing must leave a trace to find it by."""
        edges, entities = hand
        request = _request(edges, entities)
        build_feature_matrix(request, ["aml_context_v1"])
        resolved = request.cache["aml_context_v1_resolved"]
        assert set(resolved) == {"channel", "currency", "location"}
        assert all(v is None for v in resolved.values())

    def test_present_columns_are_used(self, hand):
        edges, entities = hand
        edges = edges.copy()
        edges["Sender_bank_location"] = ["UK"] * len(edges)
        edges["Receiver_bank_location"] = ["UK", "UAE"] + ["UK"] * (len(edges) - 2)
        edges["Payment_type"] = ["ACH"] * len(edges)
        values = _matrix(edges, entities, ["aml_context_v1"])
        # PASS is on row 0 (domestic, as receiver) and row 1 (cross-border, as sender).
        assert values.loc["PASS", "aml_context_v1::cross_border_share"] == pytest.approx(0.5)
        assert values.loc["PASS", "aml_context_v1::n_payment_types"] == 1


# ---------------------------------------------------------------------------
# Temporal safety and reuse
# ---------------------------------------------------------------------------

class TestTemporalSafety:
    def test_a_truncated_edge_table_cannot_describe_a_later_transaction(self, hand):
        """The window is enforced by what the runner passes, and honoured here."""
        edges, entities = hand
        cutoff = BASE + pd.Timedelta(hours=35)
        visible = edges[edges["date"] < cutoff]
        windowed = _matrix(visible, entities, AML_FAMILIES, as_of=cutoff)
        full = _matrix(edges, entities, AML_FAMILIES)
        # MUTUAL_A and MUTUAL_B transact after the cutoff and must be invisible.
        assert windowed.loc["MUTUAL_A", "aml_txn_v1::txn_total"] == 0
        assert full.loc["MUTUAL_A", "aml_txn_v1::txn_total"] == 2
        assert windowed.loc["MUTUAL_A", "aml_network_v1::reciprocal_partners"] == 0
        assert full.loc["MUTUAL_A", "aml_network_v1::reciprocal_partners"] == 1

    def test_an_empty_window_yields_all_nan_not_an_exception(self, hand):
        edges, entities = hand
        empty = edges.iloc[0:0]
        values = _matrix(empty, entities, AML_FAMILIES, as_of=BASE)
        assert values.shape[0] == len(entities)
        assert values.isna().to_numpy().all()

    def test_families_share_one_event_view_per_request(self, hand):
        """Ablation runs and the full run must not pay for the flattening twice."""
        edges, entities = hand
        request = _request(edges, entities)
        build_feature_matrix(request, AML_FAMILIES)
        assert "event_view" in request.cache
        assert "simple_adjacency" in request.cache

    def test_a_new_request_does_not_inherit_another_window(self, hand):
        """Two folds must not share a cache; the window is part of the request."""
        edges, entities = hand
        cutoff = BASE + pd.Timedelta(hours=35)
        first = _request(edges[edges["date"] < cutoff], entities, as_of=cutoff)
        build_feature_matrix(first, ["aml_txn_v1"])
        second = _request(edges, entities)
        assert second.cache == {}
        values = build_feature_matrix(second, ["aml_txn_v1"]).values
        assert values.loc["MUTUAL_A", "aml_txn_v1::txn_total"] == 2


class TestAmountHandling:
    def test_amount_usd_is_preferred_over_native_amount(self, hand):
        edges, entities = hand
        edges = edges.copy()
        edges["amount_usd"] = edges["amount"] * 10.0
        values = _matrix(edges, entities, ["aml_txn_v1"])
        assert values.loc["DST", "aml_txn_v1::amount_in_sum"] == pytest.approx(6000.0)

    def test_no_amount_column_leaves_amount_features_undefined(self, hand):
        edges, entities = hand
        edges = edges.drop(columns=["amount"])
        values = _matrix(edges, entities, ["aml_txn_v1", "aml_flow_v1"])
        assert values["aml_txn_v1::amount_in_sum"].isna().all()
        assert values["aml_flow_v1::flow_imbalance"].isna().all()
        # Counts do not need amounts and must still be right.
        assert values.loc["DST", "aml_txn_v1::txn_in"] == 2
