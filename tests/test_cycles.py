"""
Layer 10 — Cycle & structural evidence detection tests
======================================================

Every synthetic graph below is built by hand and is fully deterministic: no
randomness, no dataset dependency, no model.  The reference-dataset tests are
marked ``reference`` and skip when ``dkdata.json`` is not present, matching the
convention in ``tests/conftest.py``.

The final section asserts the architectural invariant this layer exists under:
cycle analysis is downstream of, and independent from, the frozen 167-feature
representation and the Experiment 3 checkpoint.
"""

import time

import networkx as nx
import pandas as pd
import pytest

from src.detection.cycles import (
    AMOUNT_ABSENT,
    AMOUNT_COMPLETE,
    AMOUNT_PARTIAL,
    DEFAULT_CYCLE_CONFIG,
    PRECISION_DAY,
    PRECISION_NONE,
    PRECISION_SUB_DAY,
    TEMPORAL_DAY_LEVEL,
    TEMPORAL_INCOMPLETE,
    TEMPORAL_NO_EVIDENCE,
    TEMPORAL_NO_ORDER,
    TEMPORAL_SELF_LOOP,
    TEMPORAL_STRICT_ORDER,
    TEMPORAL_TIED,
    TRUNCATION_MAX_CYCLES,
    CycleAnalysisConfig,
    analyze_cycles,
    canonical_rotation,
    cycle_identity,
    describe_analysis,
    reciprocal_cross_reference,
    summarize_entity_cycles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph(edges, *, day_level=True):
    """Build a MultiDiGraph from ``(source, target, day, amount)`` tuples.

    ``day`` may be an int (interpreted as day-of-month at midnight -> day-level
    precision), a string, a Timestamp, or ``None``.  ``amount`` may be ``None``
    to model a missing amount.
    """
    graph = nx.MultiDiGraph()
    for index, (source, target, day, amount) in enumerate(edges):
        if day is None:
            stamp = None
        elif isinstance(day, int):
            stamp = pd.Timestamp(f"2024-01-{day:02d}")
        else:
            stamp = pd.Timestamp(day)
        graph.add_node(source)
        graph.add_node(target)
        graph.add_edge(
            source,
            target,
            key=f"t{index}",
            transaction_id=f"t{index}",
            date=stamp,
            amount_usd=amount,
        )
    return graph


def _ring(nodes, *, parallel=1, day=1, amount=100.0):
    """A directed ring over ``nodes`` with ``parallel`` transactions per leg."""
    edges = []
    size = len(nodes)
    for index, source in enumerate(nodes):
        target = nodes[(index + 1) % size]
        for _ in range(parallel):
            edges.append((source, target, day, amount))
    return _graph(edges)


def _ids(analysis):
    return [cycle.cycle_id for cycle in analysis.cycles]


def _entity_sets(analysis):
    return [tuple(cycle.entities) for cycle in analysis.cycles]


# ---------------------------------------------------------------------------
# 1-4. Degenerate graphs
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_01_empty_graph():
    analysis = analyze_cycles(nx.MultiDiGraph())

    assert analysis.cycles == ()
    assert analysis.entity_index == {}
    assert analysis.metadata.node_count == 0
    assert analysis.metadata.transaction_count == 0
    assert analysis.metadata.unique_directed_pair_count == 0
    assert analysis.metadata.truncated is False
    assert analysis.metadata.enumeration_complete is True
    assert analysis.metadata.temporal_precision == PRECISION_NONE
    assert analysis.metadata.elapsed_seconds >= 0.0


@pytest.mark.fast
def test_02_single_node_no_edges():
    graph = nx.MultiDiGraph()
    graph.add_node("A")

    analysis = analyze_cycles(graph)

    assert analysis.cycles == ()
    assert analysis.metadata.node_count == 1
    assert analysis.metadata.transaction_count == 0
    assert summarize_entity_cycles(analysis, "A").cycle_count == 0


@pytest.mark.fast
def test_03_disconnected_graph_finds_only_the_component_with_a_cycle():
    # Component 1: A->B->C->A.  Component 2: D->E (no cycle).  Isolated: F.
    graph = _graph(
        [
            ("A", "B", 1, 10.0),
            ("B", "C", 2, 10.0),
            ("C", "A", 3, 10.0),
            ("D", "E", 4, 10.0),
        ]
    )
    graph.add_node("F")

    analysis = analyze_cycles(graph)

    assert len(analysis.cycles) == 1
    assert set(analysis.cycles[0].entities) == {"A", "B", "C"}
    assert "D" not in analysis.entity_index
    assert "F" not in analysis.entity_index


@pytest.mark.fast
def test_04_no_cycle_graph():
    # A pure DAG: A->B, B->C, A->C.
    analysis = analyze_cycles(
        _graph([("A", "B", 1, 5.0), ("B", "C", 2, 5.0), ("A", "C", 3, 5.0)])
    )

    assert analysis.cycles == ()
    assert analysis.metadata.unique_directed_pair_count == 3
    assert analysis.metadata.cycles_enumerated == 0


# ---------------------------------------------------------------------------
# 5. Self-loop
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_05a_self_loop_is_excluded_by_default_but_recorded():
    analysis = analyze_cycles(_graph([("A", "A", 1, 50.0)]))

    assert analysis.cycles == ()
    # Excluded, never silently discarded.
    assert analysis.metadata.cycles_enumerated == 1
    assert analysis.metadata.cycles_excluded_by_min_length == 1
    assert analysis.metadata.self_loop_pair_count == 1
    assert analysis.metadata.self_loop_entities == frozenset({"A"})
    assert summarize_entity_cycles(analysis, "A").has_self_loop is True


@pytest.mark.fast
def test_05b_self_loop_retained_when_min_length_is_one():
    analysis = analyze_cycles(
        _graph([("A", "A", 1, 50.0), ("A", "A", 2, 70.0)]),
        config=CycleAnalysisConfig(min_cycle_length=1),
    )

    assert len(analysis.cycles) == 1
    cycle = analysis.cycles[0]
    assert cycle.length == 1
    assert cycle.is_self_loop is True
    assert cycle.is_reciprocal_pair is False
    assert len(cycle.legs) == 1
    assert cycle.legs[0].source == cycle.legs[0].target == "A"
    assert cycle.legs[0].transaction_count == 2
    # A single-leg cycle has no traversal order to establish.
    assert cycle.is_time_respecting is None
    assert cycle.temporal_basis == TEMPORAL_SELF_LOOP


# ---------------------------------------------------------------------------
# 6-8. 2-, 3- and 4-cycles
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_06_two_cycle():
    analysis = analyze_cycles(_graph([("A", "B", 1, 100.0), ("B", "A", 2, 90.0)]))

    assert len(analysis.cycles) == 1
    cycle = analysis.cycles[0]
    assert cycle.entities == ("A", "B")
    assert cycle.length == 2
    assert cycle.participant_count == 2
    assert cycle.is_reciprocal_pair is True
    assert cycle.transaction_count == 2
    assert [(leg.source, leg.target) for leg in cycle.legs] == [("A", "B"), ("B", "A")]
    assert cycle.leg_amount_totals == (100.0, 90.0)
    assert cycle.leg_amount_balance == pytest.approx(0.9)


@pytest.mark.fast
def test_07_three_cycle():
    analysis = analyze_cycles(
        _graph([("A", "B", 1, 100.0), ("B", "C", 2, 100.0), ("C", "A", 3, 100.0)])
    )

    assert len(analysis.cycles) == 1
    cycle = analysis.cycles[0]
    assert cycle.entities == ("A", "B", "C")
    assert cycle.length == 3
    assert [(leg.source, leg.target) for leg in cycle.legs] == [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
    ]
    assert cycle.leg_amount_balance == pytest.approx(1.0)


@pytest.mark.fast
def test_08_four_cycle():
    analysis = analyze_cycles(_ring(["A", "B", "C", "D"]))

    assert len(analysis.cycles) == 1
    assert analysis.cycles[0].entities == ("A", "B", "C", "D")
    assert analysis.cycles[0].length == 4
    assert set(analysis.entity_index) == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# 9. Direction matters
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_09_reverse_direction_is_a_different_cycle():
    forward = analyze_cycles(_ring(["A", "B", "C"]))
    reverse = analyze_cycles(_ring(["A", "C", "B"]))

    assert forward.cycles[0].entities == ("A", "B", "C")
    assert reverse.cycles[0].entities == ("A", "C", "B")
    assert forward.cycles[0].cycle_id != reverse.cycles[0].cycle_id

    # Both directions present in one graph -> two distinct cycles.
    both = nx.MultiDiGraph()
    for source, target in [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("A", "C"),
        ("C", "B"),
        ("B", "A"),
    ]:
        both.add_edge(source, target, key=f"{source}{target}", date=None, amount_usd=None)

    analysis = analyze_cycles(both)
    triples = [c.entities for c in analysis.cycles if c.length == 3]
    assert sorted(triples) == [("A", "B", "C"), ("A", "C", "B")]


# ---------------------------------------------------------------------------
# 10. Rotation invariance
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_10a_canonical_rotation_and_identity_are_rotation_invariant():
    rotations = [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]]

    canonical = {canonical_rotation(rotation) for rotation in rotations}
    assert canonical == {("A", "B", "C")}

    identities = {cycle_identity(rotation) for rotation in rotations}
    assert len(identities) == 1

    # ...but not reversal-invariant: direction is preserved.
    assert cycle_identity(["A", "B", "C"]) != cycle_identity(["A", "C", "B"])


@pytest.mark.fast
def test_10b_cycle_id_is_independent_of_the_bounds_used():
    graph = _ring(["A", "B", "C"])

    wide = analyze_cycles(graph, config=CycleAnalysisConfig(max_cycle_length=6))
    narrow = analyze_cycles(graph, config=CycleAnalysisConfig(max_cycle_length=3))

    assert _ids(wide) == _ids(narrow)


@pytest.mark.fast
def test_10c_cycle_id_is_stable_across_node_insertion_order():
    first = _ring(["A", "B", "C"])

    second = nx.MultiDiGraph()
    for source, target in [("C", "A"), ("B", "C"), ("A", "B")]:
        second.add_edge(source, target, key=f"{source}{target}", date=None, amount_usd=None)

    assert _ids(analyze_cycles(first)) == _ids(analyze_cycles(second))


# ---------------------------------------------------------------------------
# 11. Parallel edges are collapsed, never multiplied out
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_11_fifty_parallel_edges_per_leg_yield_exactly_one_cycle():
    graph = _ring(["A", "B", "C"], parallel=50, amount=10.0)
    assert graph.number_of_edges() == 150

    analysis = analyze_cycles(graph)

    # 1 entity-level cycle, 3 legs -- not 50^3 = 125,000 combinations.
    assert len(analysis.cycles) == 1
    cycle = analysis.cycles[0]
    assert cycle.length == 3
    assert len(cycle.legs) == 3
    assert [leg.transaction_count for leg in cycle.legs] == [50, 50, 50]
    assert cycle.transaction_count == 150

    assert analysis.metadata.transaction_count == 150
    assert analysis.metadata.unique_directed_pair_count == 3


@pytest.mark.fast
def test_11b_leg_transaction_id_samples_are_bounded_and_flagged():
    analysis = analyze_cycles(
        _ring(["A", "B"], parallel=20),
        config=CycleAnalysisConfig(max_transaction_samples_per_leg=3),
    )

    for leg in analysis.cycles[0].legs:
        assert len(leg.transaction_id_sample) == 3
        assert leg.transaction_id_sample_truncated is True
        assert leg.transaction_count == 20


# ---------------------------------------------------------------------------
# 12. Cycle-length bound
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_12_cycle_length_bound_excludes_longer_cycles():
    graph = _ring(["A", "B", "C", "D", "E"])

    assert len(analyze_cycles(graph, config=CycleAnalysisConfig(max_cycle_length=5)).cycles) == 1

    bounded = analyze_cycles(graph, config=CycleAnalysisConfig(max_cycle_length=4))
    assert bounded.cycles == ()
    assert bounded.metadata.config.max_cycle_length == 4
    assert bounded.metadata.truncated is False  # a length bound is not a truncation


@pytest.mark.fast
def test_12b_default_bounds_are_the_documented_ones_and_are_visible():
    assert DEFAULT_CYCLE_CONFIG.max_cycle_length == 6
    assert DEFAULT_CYCLE_CONFIG.min_cycle_length == 2
    assert DEFAULT_CYCLE_CONFIG.max_cycles == 10000

    meta = analyze_cycles(_ring(["A", "B", "C"])).metadata
    assert meta.config == DEFAULT_CYCLE_CONFIG


@pytest.mark.fast
def test_12c_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        CycleAnalysisConfig(max_cycle_length=2, min_cycle_length=3)
    with pytest.raises(ValueError):
        CycleAnalysisConfig(max_cycles=0)
    with pytest.raises(ValueError):
        CycleAnalysisConfig(min_cycle_length=0)


# ---------------------------------------------------------------------------
# 13. Retention bound / truncation
# ---------------------------------------------------------------------------

def _complete_digraph_multi(size):
    """A complete directed graph on ``size`` nodes: many overlapping cycles."""
    graph = nx.MultiDiGraph()
    names = [f"N{i:02d}" for i in range(size)]
    for source in names:
        for target in names:
            if source != target:
                graph.add_edge(
                    source, target, key=f"{source}->{target}", date=None, amount_usd=None
                )
    return graph


@pytest.mark.fast
def test_13_retention_bound_truncates_and_says_so():
    graph = _complete_digraph_multi(6)

    unbounded = analyze_cycles(graph, config=CycleAnalysisConfig(max_cycle_length=4))
    assert unbounded.metadata.truncated is False
    assert unbounded.metadata.enumeration_complete is True
    total = unbounded.metadata.cycles_retained
    assert total > 10

    bounded = analyze_cycles(
        graph, config=CycleAnalysisConfig(max_cycle_length=4, max_cycles=10)
    )
    assert bounded.metadata.truncated is True
    assert bounded.metadata.truncation_reason == TRUNCATION_MAX_CYCLES
    assert bounded.metadata.enumeration_complete is False
    assert bounded.metadata.cycles_retained == 10
    assert bounded.metadata.cycles_retained < total


@pytest.mark.fast
def test_13b_truncation_is_deterministic_in_which_cycles_survive():
    graph = _complete_digraph_multi(6)
    config = CycleAnalysisConfig(max_cycle_length=4, max_cycles=12)

    assert _ids(analyze_cycles(graph, config=config)) == _ids(
        analyze_cycles(graph, config=config)
    )


@pytest.mark.fast
def test_13c_truncation_marks_entity_summaries_as_lower_bounds():
    graph = _complete_digraph_multi(6)
    analysis = analyze_cycles(
        graph, config=CycleAnalysisConfig(max_cycle_length=4, max_cycles=5)
    )

    summary = summarize_entity_cycles(analysis, analysis.cycles[0].entities[0])
    assert summary.truncated is True
    # A reconciliation that cannot be made is None, never False.
    assert reciprocal_cross_reference(summary, 3.0)["consistent"] is None


# ---------------------------------------------------------------------------
# 14. Missing amount evidence
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_14a_absent_amounts_are_none_never_zero():
    analysis = analyze_cycles(
        _graph([("A", "B", 1, None), ("B", "A", 2, None)])
    )
    cycle = analysis.cycles[0]

    assert cycle.amount_evidence == AMOUNT_ABSENT
    assert cycle.leg_amount_totals == (None, None)
    assert cycle.leg_amount_balance is None
    assert cycle.amount_retention is None
    for leg in cycle.legs:
        assert leg.amount_usd_total is None
        assert leg.amount_usd_min is None
        assert leg.amount_usd_max is None
        assert leg.transactions_with_amount == 0
        assert leg.amount_evidence == AMOUNT_ABSENT


@pytest.mark.fast
def test_14b_partial_amounts_are_flagged_and_never_completed():
    # Leg A->B: one amount observed of two transactions.  Leg B->A: complete.
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", 1, 100.0),
                ("A", "B", 2, None),
                ("B", "A", 3, 40.0),
            ]
        )
    )
    cycle = analysis.cycles[0]

    ab = next(leg for leg in cycle.legs if leg.source == "A")
    ba = next(leg for leg in cycle.legs if leg.source == "B")

    assert ab.amount_evidence == AMOUNT_PARTIAL
    assert ab.transaction_count == 2
    assert ab.transactions_with_amount == 1
    assert ab.amount_usd_total == 100.0  # a sum over what was observed
    assert ba.amount_evidence == AMOUNT_COMPLETE

    assert cycle.amount_evidence == AMOUNT_PARTIAL
    assert cycle.legs_with_complete_amounts == 1
    # Derived amount properties require complete evidence.
    assert cycle.leg_amount_balance is None
    assert cycle.amount_retention is None


@pytest.mark.fast
def test_14c_nan_amount_is_treated_as_missing():
    analysis = analyze_cycles(
        _graph([("A", "B", 1, float("nan")), ("B", "A", 2, 10.0)])
    )
    ab = next(leg for leg in analysis.cycles[0].legs if leg.source == "A")

    assert ab.transactions_with_amount == 0
    assert ab.amount_usd_total is None
    assert analysis.cycles[0].amount_evidence == AMOUNT_PARTIAL


@pytest.mark.fast
def test_14d_amount_aggregates_over_a_parallel_bundle():
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", 1, 10.0),
                ("A", "B", 2, 30.0),
                ("A", "B", 3, 20.0),
                ("B", "A", 4, 60.0),
            ]
        )
    )
    ab = next(leg for leg in analysis.cycles[0].legs if leg.source == "A")

    assert ab.amount_usd_total == pytest.approx(60.0)
    assert ab.amount_usd_min == pytest.approx(10.0)
    assert ab.amount_usd_max == pytest.approx(30.0)
    assert ab.transactions_with_amount == 3
    assert analysis.cycles[0].leg_amount_balance == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. Day-level temporal precision -> None
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_15a_day_level_precision_never_claims_ordering():
    # Legs on three different days: ordering would look establishable, but at
    # day-level precision no ordering claim is made.
    analysis = analyze_cycles(
        _graph([("A", "B", 1, 10.0), ("B", "C", 5, 10.0), ("C", "A", 9, 10.0)])
    )

    assert analysis.metadata.temporal_precision == PRECISION_DAY
    assert "midnight" in analysis.metadata.temporal_precision_basis

    cycle = analysis.cycles[0]
    assert cycle.is_time_respecting is None
    assert cycle.temporal_basis == TEMPORAL_DAY_LEVEL
    assert "intra-day" in cycle.temporal_explanation.lower()
    assert cycle.amount_retention is None  # no traversal -> no retention basis
    assert cycle.time_span_days == pytest.approx(8.0)


@pytest.mark.fast
def test_15b_no_timestamps_at_all():
    analysis = analyze_cycles(
        _graph([("A", "B", None, 10.0), ("B", "A", None, 10.0)])
    )

    assert analysis.metadata.temporal_precision == PRECISION_NONE
    assert analysis.metadata.timestamped_transaction_count == 0
    cycle = analysis.cycles[0]
    assert cycle.is_time_respecting is None
    assert cycle.temporal_basis == TEMPORAL_NO_EVIDENCE
    assert cycle.first_transaction_at is None
    assert cycle.time_span_days is None


# ---------------------------------------------------------------------------
# 16. Sub-day temporal precision -> correct boolean
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_16a_sub_day_precision_in_order_is_true():
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", "2024-01-01 09:00:00", 100.0),
                ("B", "C", "2024-01-01 10:00:00", 90.0),
                ("C", "A", "2024-01-01 11:00:00", 80.0),
            ]
        )
    )

    assert analysis.metadata.temporal_precision == PRECISION_SUB_DAY
    cycle = analysis.cycles[0]
    assert cycle.is_time_respecting is True
    assert cycle.temporal_basis == TEMPORAL_STRICT_ORDER
    # Retention is computed along the established traversal: 80 back out of 100.
    assert cycle.amount_retention == pytest.approx(0.8)
    assert cycle.amount_retention_basis == "TIME_RESPECTING_TRAVERSAL"


@pytest.mark.fast
def test_16b_sub_day_precision_out_of_order_is_false():
    # A->B at 12:00, B->C at 11:00, C->A at 10:00.  No rotation is
    # non-decreasing: strictly decreasing round the ring.
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", "2024-01-01 12:00:00", 10.0),
                ("B", "C", "2024-01-01 11:00:00", 10.0),
                ("C", "A", "2024-01-01 10:00:00", 10.0),
            ]
        )
    )
    cycle = analysis.cycles[0]

    assert cycle.is_time_respecting is False
    assert cycle.temporal_basis == TEMPORAL_NO_ORDER
    assert cycle.amount_retention is None


@pytest.mark.fast
def test_16c_rotation_is_searched_for_a_time_respecting_traversal():
    # In canonical order (A,B,C) the timestamps are 11:00, 12:00, 10:00 --
    # not increasing.  Starting from the C->A leg they are 10, 11, 12.
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", "2024-01-01 11:00:00", 10.0),
                ("B", "C", "2024-01-01 12:00:00", 10.0),
                ("C", "A", "2024-01-01 10:00:00", 10.0),
            ]
        )
    )

    assert analysis.cycles[0].is_time_respecting is True


@pytest.mark.fast
def test_16d_identical_timestamps_are_not_establishable_not_false():
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", "2024-01-01 09:00:00", 10.0),
                ("B", "C", "2024-01-01 09:00:00", 10.0),
                ("C", "A", "2024-01-01 09:00:00", 10.0),
            ]
        )
    )
    cycle = analysis.cycles[0]

    assert cycle.is_time_respecting is None
    assert cycle.temporal_basis == TEMPORAL_TIED
    assert cycle.amount_retention is None


@pytest.mark.fast
def test_16e_a_leg_with_no_timestamp_is_not_establishable():
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", "2024-01-01 09:00:00", 10.0),
                ("B", "C", "2024-01-01 10:00:00", 10.0),
                ("C", "A", None, 10.0),
            ]
        )
    )
    cycle = analysis.cycles[0]

    assert analysis.metadata.temporal_precision == PRECISION_SUB_DAY
    assert cycle.is_time_respecting is None
    assert cycle.temporal_basis == TEMPORAL_INCOMPLETE


@pytest.mark.fast
def test_16f_parallel_transactions_offer_alternative_traversals():
    # The only A->B transaction is at 15:00, but B->A has two candidates and
    # the later one completes the traversal.
    analysis = analyze_cycles(
        _graph(
            [
                ("A", "B", "2024-01-01 15:00:00", 10.0),
                ("B", "A", "2024-01-01 09:00:00", 10.0),
                ("B", "A", "2024-01-01 16:00:00", 10.0),
            ]
        )
    )

    assert analysis.cycles[0].is_time_respecting is True


@pytest.mark.fast
def test_16g_sub_day_precision_still_orders_across_days():
    graph = _graph(
        [
            ("A", "B", "2024-01-01 09:30:00", 10.0),
            ("B", "A", "2024-03-05 00:00:00", 10.0),
        ]
    )
    analysis = analyze_cycles(graph)

    assert analysis.metadata.temporal_precision == PRECISION_SUB_DAY
    assert analysis.cycles[0].is_time_respecting is True
    assert analysis.cycles[0].time_span_days == pytest.approx(63.60416667, abs=1e-6)


# ---------------------------------------------------------------------------
# 17-18. Determinism and stable ordering
# ---------------------------------------------------------------------------

def _mixed_graph():
    graph = nx.MultiDiGraph()
    edges = [
        ("D", "E", "2024-02-01 08:00:00", 5.0),
        ("E", "D", "2024-02-02 08:00:00", 5.0),
        ("A", "B", "2024-01-01 08:00:00", 10.0),
        ("B", "C", "2024-01-02 08:00:00", 20.0),
        ("C", "A", "2024-01-03 08:00:00", 30.0),
        ("C", "D", "2024-01-04 08:00:00", 40.0),
        ("D", "A", "2024-01-05 08:00:00", 50.0),
    ]
    return _graph(edges)


@pytest.mark.fast
def test_17_output_is_deterministic_across_repeated_runs():
    graph = _mixed_graph()

    runs = [analyze_cycles(graph) for _ in range(3)]
    signatures = [
        [
            (
                c.cycle_id,
                c.entities,
                c.transaction_count,
                c.leg_amount_totals,
                c.is_time_respecting,
                c.temporal_basis,
            )
            for c in run.cycles
        ]
        for run in runs
    ]

    assert signatures[0] == signatures[1] == signatures[2]


_SUBPROCESS_DETERMINISM = """
import hashlib
import sys

import networkx as nx

sys.path.insert(0, {repo!r})
from src.detection.cycles import CycleAnalysisConfig, analyze_cycles

graph = nx.MultiDiGraph()
names = ["entity-%02d" % i for i in range(7)]
for source in names:
    for target in names:
        if source != target:
            graph.add_edge(source, target, key=source + target, date=None, amount_usd=None)

analysis = analyze_cycles(
    graph, config=CycleAnalysisConfig(max_cycle_length=5, max_cycles=40)
)
assert analysis.metadata.truncated is True
print(hashlib.sha256("|".join(c.cycle_id for c in analysis.cycles).encode()).hexdigest())
"""


@pytest.mark.fast
def test_17b_truncated_output_is_reproducible_across_processes():
    """A truncated run must retain the same cycles in a fresh interpreter.

    ``nx.simple_cycles`` iterates internal sets, and set iteration order for
    strings varies with the per-process hash seed.  Without the integer
    relabelling in ``analyze_cycles`` a truncated result differs between
    processes, so the same dataset would hand two investigators two different
    cycle lists.  Asserted here against actually-differing hash seeds, which is
    only observable across processes.
    """
    import os
    import pathlib
    import subprocess
    import sys

    repo = str(pathlib.Path(__file__).resolve().parents[1])
    script = _SUBPROCESS_DETERMINISM.format(repo=repo)

    digests = []
    for seed in ("0", "1", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repo,
            env=env,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        digests.append(result.stdout.strip())

    assert len(set(digests)) == 1, f"truncated retention varied by hash seed: {digests}"


@pytest.mark.fast
def test_18a_cycles_are_ordered_by_length_then_entity_sequence():
    analysis = analyze_cycles(_mixed_graph())
    keys = [(c.length, tuple(str(e) for e in c.entities)) for c in analysis.cycles]

    assert keys == sorted(keys)
    assert len(analysis.cycles) >= 2


@pytest.mark.fast
def test_18b_ordering_is_independent_of_edge_insertion_order():
    forward = _mixed_graph()

    reversed_graph = nx.MultiDiGraph()
    for source, target, key, data in reversed(list(forward.edges(keys=True, data=True))):
        reversed_graph.add_edge(source, target, key=key, **data)

    assert _entity_sets(analyze_cycles(forward)) == _entity_sets(
        analyze_cycles(reversed_graph)
    )


@pytest.mark.fast
def test_18c_entity_index_matches_the_retained_cycles():
    analysis = analyze_cycles(_mixed_graph())

    rebuilt: dict[str, set[str]] = {}
    for cycle in analysis.cycles:
        for entity in cycle.entities:
            rebuilt.setdefault(entity, set()).add(cycle.cycle_id)

    assert {k: set(v) for k, v in analysis.entity_index.items()} == rebuilt


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_19_a_non_multigraph_is_refused():
    graph = nx.DiGraph()
    graph.add_edge("A", "B")

    with pytest.raises(TypeError, match="MultiDiGraph"):
        analyze_cycles(graph)


@pytest.mark.fast
def test_20_the_input_graph_is_not_modified():
    graph = _ring(["A", "B", "C"], parallel=3)
    before = (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        sorted(graph.edges(keys=True)),
    )

    analyze_cycles(graph)

    assert (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        sorted(graph.edges(keys=True)),
    ) == before


# ---------------------------------------------------------------------------
# Reciprocal cross-reference against the frozen model feature
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_21a_two_cycles_reconcile_with_reciprocal_out_count():
    # X has reciprocal relationships with Y and Z, and a one-way edge to W.
    graph = _graph(
        [
            ("X", "Y", 1, 10.0),
            ("Y", "X", 2, 10.0),
            ("X", "Z", 3, 10.0),
            ("Z", "X", 4, 10.0),
            ("X", "W", 5, 10.0),
        ]
    )
    summary = summarize_entity_cycles(analyze_cycles(graph), "X")

    assert summary.two_cycle_counterparties == ("Y", "Z")

    reference = reciprocal_cross_reference(summary, 2.0)
    assert reference["model_feature"] == "reciprocal_out_count"
    assert reference["consistent"] is True
    assert reference["self_loop_offset"] == 0


@pytest.mark.fast
def test_21b_self_loop_offset_is_stated_not_left_as_a_discrepancy():
    # multigraph.reciprocal_out_count counts X's self-loop as a reciprocal
    # relationship with itself; a self-loop is a length-1 cycle here.
    graph = _graph(
        [
            ("X", "Y", 1, 10.0),
            ("Y", "X", 2, 10.0),
            ("X", "X", 3, 10.0),
        ]
    )
    summary = summarize_entity_cycles(analyze_cycles(graph), "X")

    assert summary.has_self_loop is True
    assert summary.two_cycle_counterparties == ("Y",)

    reference = reciprocal_cross_reference(summary, 2.0)
    assert reference["self_loop_offset"] == 1
    assert reference["consistent"] is True
    assert "self-loop" in reference["note"]


@pytest.mark.fast
def test_21c_reciprocal_cross_reference_matches_the_real_feature():
    from src.features.multigraph import compute_multigraph

    graph = _graph(
        [
            ("X", "Y", 1, 10.0),
            ("Y", "X", 2, 10.0),
            ("X", "Z", 3, 10.0),
            ("Z", "X", 4, 10.0),
            ("X", "W", 5, 10.0),
            ("W", "Y", 6, 10.0),
        ]
    )
    edge_table = pd.DataFrame(
        [
            {"source_entity": u, "target_entity": v, "date": d["date"]}
            for u, v, d in graph.edges(data=True)
        ]
    )
    features = compute_multigraph(graph, edge_table)
    analysis = analyze_cycles(graph)

    for entity in graph.nodes():
        summary = summarize_entity_cycles(analysis, entity)
        value = features.features.at[entity, "reciprocal_out_count"]
        assert reciprocal_cross_reference(summary, value)["consistent"] is True


@pytest.mark.fast
def test_21d_missing_feature_value_is_not_establishable():
    summary = summarize_entity_cycles(analyze_cycles(_ring(["A", "B"])), "A")

    assert reciprocal_cross_reference(summary, None)["consistent"] is None


@pytest.mark.fast
def test_22_summary_of_an_entity_outside_every_cycle():
    analysis = analyze_cycles(_graph([("A", "B", 1, 10.0), ("B", "A", 2, 10.0)]))
    summary = summarize_entity_cycles(analysis, "GHOST")

    assert summary.cycle_count == 0
    assert summary.cycle_ids == ()
    assert summary.length_distribution == {}
    assert summary.shortest_cycle_length is None
    assert summary.longest_cycle_length is None
    assert summary.has_self_loop is False


@pytest.mark.fast
def test_23_describe_analysis_renders():
    text = describe_analysis(analyze_cycles(_mixed_graph()))

    assert "CYCLE ANALYSIS SUMMARY" in text
    assert "not model input" in text
    assert "Temporal precision" in text


# ---------------------------------------------------------------------------
# Reference dataset
# ---------------------------------------------------------------------------

@pytest.mark.reference
@pytest.mark.integration
def test_24_reference_dataset_cycle_analysis(reference_dataset):
    graph = reference_dataset.graph

    started = time.perf_counter()
    analysis = analyze_cycles(graph)
    elapsed = time.perf_counter() - started

    meta = analysis.metadata

    # The graph the analysis actually saw.
    assert meta.node_count == 3440
    assert meta.transaction_count == 16940
    assert meta.unique_directed_pair_count == 4024

    # Parallel transactions were collapsed, not multiplied out.
    assert meta.unique_directed_pair_count < meta.transaction_count

    # The default retention bound does NOT truncate the Reference Dataset: the
    # complete enumeration is 5,589 cycles at length <= 6 and costs ~2 s, so the
    # normal result is complete rather than a sample.  The bound remains a cap
    # for pathological graphs, and this test is what would catch it starting to
    # bite on the reference data.
    assert meta.truncated is False
    assert meta.truncation_reason is None
    assert meta.enumeration_complete is True
    assert meta.cycles_retained == 5589
    assert meta.cycles_retained < DEFAULT_CYCLE_CONFIG.max_cycles

    # Only the two self-loops fall under min_cycle_length, and they are counted
    # rather than silently dropped.
    assert meta.self_loop_pair_count == 2
    assert meta.cycles_excluded_by_min_length == 2
    assert meta.cycles_enumerated == meta.cycles_retained + 2

    by_length: dict[int, int] = {}
    for cycle in analysis.cycles:
        by_length[cycle.length] = by_length.get(cycle.length, 0) + 1
    assert by_length == {2: 148, 3: 53, 4: 571, 5: 403, 6: 4414}

    # The Reference Dataset carries day-level precision only, so no cycle
    # claims a traversal order.  docs/research.md Finding 3.3.
    assert meta.temporal_precision == PRECISION_DAY
    assert all(cycle.is_time_respecting is None for cycle in analysis.cycles)
    assert all(cycle.temporal_basis == TEMPORAL_DAY_LEVEL for cycle in analysis.cycles)

    # Every retained cycle is within the configured bounds.
    for cycle in analysis.cycles:
        assert DEFAULT_CYCLE_CONFIG.min_cycle_length <= cycle.length
        assert cycle.length <= DEFAULT_CYCLE_CONFIG.max_cycle_length
        assert len(cycle.legs) == cycle.length
        assert cycle.participant_count == cycle.length

    # Cycle identities are unique and the ordering is the stable one.
    ids = _ids(analysis)
    assert len(ids) == len(set(ids))
    keys = [(c.length, tuple(str(e) for e in c.entities)) for c in analysis.cycles]
    assert keys == sorted(keys)

    # Bounded cost.  The measured figure is recorded in docs/research.md; this
    # bound is deliberately loose so the test is not a performance benchmark.
    assert elapsed < 60.0
    assert meta.elapsed_seconds > 0.0


@pytest.mark.reference
@pytest.mark.integration
def test_25_reference_two_cycles_reconcile_with_the_frozen_feature(reference_dataset):
    """2-cycle evidence must agree with the model's `reciprocal_out_count`.

    This is the cross-reference the layer promises: the same structure at two
    resolutions.  A disagreement would mean one of the two is wrong.

    Reads the multigraph features from the session cache in ``tests.test_utils``
    rather than the ``reference_features`` fixture, which additionally computes
    temporal and TGA features this test does not need.
    """
    from tests.test_utils import get_cached_multigraph

    graph = reference_dataset.graph
    multigraph = get_cached_multigraph().features

    # Length bound of 2 keeps the enumeration complete, so the reconciliation
    # is not weakened by truncation.
    analysis = analyze_cycles(
        graph, config=CycleAnalysisConfig(max_cycle_length=2, max_cycles=1_000_000)
    )
    assert analysis.metadata.truncated is False

    checked = 0
    for entity in analysis.entity_index:
        summary = summarize_entity_cycles(analysis, entity)
        value = multigraph.at[entity, "reciprocal_out_count"]
        assert reciprocal_cross_reference(summary, value)["consistent"] is True, entity
        checked += 1

    assert checked > 0


@pytest.mark.reference
@pytest.mark.integration
def test_26_reference_analysis_is_deterministic(reference_dataset):
    config = CycleAnalysisConfig(max_cycle_length=4)
    graph = reference_dataset.graph

    first = analyze_cycles(graph, config=config)
    second = analyze_cycles(graph, config=config)

    assert _ids(first) == _ids(second)
    assert [c.transaction_count for c in first.cycles] == [
        c.transaction_count for c in second.cycles
    ]


# ---------------------------------------------------------------------------
# Architectural invariant: cycle analysis is downstream and independent
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_27_module_does_not_import_the_model_or_feature_layers():
    """Layer 10 must not be able to reach the frozen model or its features.

    An import is the cheapest way for cycle evidence to leak into inference, so
    the absence of one is asserted directly on the module source rather than
    left to review.
    """
    import inspect

    import src.detection.cycles as cycles_module

    source = inspect.getsource(cycles_module)
    for forbidden in (
        "import torch",
        "from src.features",
        "import src.features",
        "from src.models",
        "import src.models",
        "from src.detection.inference",
        "BaselineGraphSAGE",
    ):
        assert forbidden not in source, f"cycles.py must not reference {forbidden!r}"


@pytest.mark.reference
@pytest.mark.integration
def test_28_cycle_analysis_does_not_disturb_the_167_feature_representation(
    reference_dataset,
):
    """The feature matrix must be value-identical before and after analysis."""
    from src.features.baseline import compute_baseline
    from src.features.multigraph import compute_multigraph

    graph = reference_dataset.graph
    edge_table = reference_dataset.edge_table

    before_baseline = compute_baseline(graph, edge_table)
    before_multi = compute_multigraph(graph, edge_table)

    analyze_cycles(graph)

    after_baseline = compute_baseline(graph, edge_table)
    after_multi = compute_multigraph(graph, edge_table)

    assert before_baseline.feature_names == after_baseline.feature_names
    assert before_multi.feature_names == after_multi.feature_names
    assert (
        len(before_baseline.feature_names) + len(before_multi.feature_names) == 167
    )
    pd.testing.assert_frame_equal(before_baseline.features, after_baseline.features)
    pd.testing.assert_frame_equal(before_multi.features, after_multi.features)
