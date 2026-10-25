"""
Layer 10 — Stage 2: shared AnalysisContext integration
======================================================

Covers the *integration* only.  The 52 core-algorithm tests live in
``tests/test_cycles.py`` and are not repeated here; what is asserted below is
that cycle evidence reaches both analysis paths off the graph that path already
built, participates correctly in the existing context cache, respects workspace
identity strategy, stays isolated between workspaces, and cannot disturb model
output when it fails.

The workspace fixtures build genuinely small READY workspaces (40 rows) through
the real path — ingest, profile, automatic mapping, canonicalize, validate,
predict — so nothing here mocks the pipeline it is meant to be testing.
"""

import copy
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

import src.detection.cycles as cycles_module
import src.ui.resolver as ui_resolver
from src.datasets.canonicalizer import canonicalize_and_validate
from src.datasets.inference import predict_workspace
from src.datasets.manifest import CurrencyPolicy, IdentitySource
from src.datasets.mapper import MappingManifest
from src.datasets.workspace_manager import WorkspaceManager
from src.detection.cycles import (
    UNAVAILABLE_ANALYSIS_FAILED,
    CycleAnalysis,
    CycleAnalysisMetadata,
    CycleAnalysisUnavailable,
    analyze_cycles,
    analyze_cycles_safely,
)
from src.ui.context import AnalysisContext
from src.ui.resolver import WorkspaceAnalysisResolver


# ---------------------------------------------------------------------------
# Fixtures — small, real workspaces
# ---------------------------------------------------------------------------

#: Three entities in a directed ring, each reachable by name and by account.
#: Under NAME identity the entities are the names; under ACCOUNT_ID they are the
#: account identifiers.  Both produce one 3-cycle, with different identities.
_NAMES = ["alice", "bob", "carol"]
_ACCOUNTS = {"alice": "ACC-A1", "bob": "ACC-B", "carol": "ACC-C"}


def _ring_frame(rows=40, amount_base=100.0):
    records = []
    for index in range(rows):
        sender = _NAMES[index % 3]
        receiver = _NAMES[(index + 1) % 3]
        records.append(
            {
                "sender_name": sender,
                "receiver_name": receiver,
                "sender_account": _ACCOUNTS[sender],
                "receiver_account": _ACCOUNTS[receiver],
                "amount": amount_base + index,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            }
        )
    return pd.DataFrame(records)


def _chain_frame(rows=40):
    """A pure DAG: real entities, real predictions, genuinely no cycle."""
    records = []
    for index in range(rows):
        records.append(
            {
                "sender_name": f"src-{index % 4}",
                "receiver_name": f"dst-{index % 5}",
                "sender_account": f"S{index % 4}",
                "receiver_account": f"R{index % 5}",
                "amount": 250.0 + index,
                "date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=index),
            }
        )
    return pd.DataFrame(records)


def _make_ready_workspace(manager, root, name, frame, identity_source):
    """Drive one frame through the real workspace path to COMPLETE."""
    path = Path(root) / f"{name}.csv"
    frame.to_csv(path, index=False)

    workspace = manager.create_workspace(name)
    workspace_id = workspace.manifest.workspace_id
    manager.ingest_file(workspace_id, str(path))
    manager.profile_workspace(workspace_id)
    manager.map_workspace(workspace_id)

    config = manager.get_workspace(workspace_id).manifest.dataset_config
    config.identity_source = identity_source
    config.currency_policy = CurrencyPolicy.NATIVE_USD
    manager.update_workspace(workspace_id, dataset_config=config)

    # Confirm the mapping actually landed, so a failure here is reported as a
    # fixture problem rather than as a cycle-integration problem.
    manifest = MappingManifest.from_dict(manager.get_mapping(workspace_id))
    assert manifest.mappings

    report = canonicalize_and_validate(workspace_id, manager)
    assert report.status == "READY", report.missing_required_fields
    assert report.feature_dimension == 167
    assert report.feature_order_valid

    predict_workspace(workspace_id, manager)
    return workspace_id


@pytest.fixture
def store(tmp_path):
    return WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))


@pytest.fixture
def bound_resolver(store, monkeypatch):
    """A resolver reading ``store`` instead of the real workspace directory."""
    monkeypatch.setattr(ui_resolver, "WorkspaceManager", lambda *a, **k: store)
    return WorkspaceAnalysisResolver(base_dir=str(store.base_dir))


@pytest.fixture
def ring_workspace(store, tmp_path):
    return _make_ready_workspace(
        store, tmp_path, "ring_name", _ring_frame(), IdentitySource.NAME
    )


@pytest.fixture
def spy_on_cycles(monkeypatch):
    """Count real calls into cycle analysis without replacing the behaviour."""
    calls = []
    real = cycles_module.analyze_cycles_safely

    def counted(graph, config=None):
        calls.append(graph)
        return real(graph, config)

    monkeypatch.setattr(ui_resolver, "analyze_cycles_safely", counted)
    return calls


def _stub_reference(resolver, entity="REF_ONLY"):
    """Attach a cheap sentinel Reference context, as the existing tests do."""
    graph = nx.MultiDiGraph()
    graph.add_node(entity)
    resolver.reference_context = AnalysisContext(
        workspace_id=None,
        df=pd.DataFrame(
            {"entity_id": [entity], "risk_score": [0.99], "predicted_class": [1]}
        ),
        graph=graph,
    )
    return resolver.reference_context


# ---------------------------------------------------------------------------
# 1. AnalysisContext surface
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_01_context_carries_cycles_and_defaults_to_none():
    """The field is additive: existing three-argument construction still works."""
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="t0")
    context = AnalysisContext(
        workspace_id=None, df=pd.DataFrame({"entity_id": ["A"]}), graph=graph
    )

    assert hasattr(context, "cycles")
    assert context.cycles is None


@pytest.mark.fast
def test_02_context_surface_is_exactly_four_fields():
    """Stage 2 adds one field and no others."""
    import dataclasses

    assert [f.name for f in dataclasses.fields(AnalysisContext)] == [
        "workspace_id",
        "df",
        "graph",
        "cycles",
    ]


@pytest.mark.fast
def test_03_three_cycle_states_are_distinguishable():
    """None, empty-analysis and unavailable must never collapse into each other.

    This is the guarantee that stops "analysis failed" being read downstream as
    "this dataset has no cycles".
    """
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="t0", date=None, amount_usd=None)

    not_attempted = None
    empty = analyze_cycles(graph)
    unavailable = analyze_cycles_safely("not a graph")

    assert not_attempted is None

    assert isinstance(empty, CycleAnalysis)
    assert empty.cycles == ()
    assert isinstance(empty.metadata, CycleAnalysisMetadata)
    assert empty.metadata.node_count == 2  # it really looked at a real graph

    assert isinstance(unavailable, CycleAnalysisUnavailable)
    assert unavailable.reason == UNAVAILABLE_ANALYSIS_FAILED
    assert unavailable.error_type == "TypeError"
    assert not isinstance(unavailable, CycleAnalysis)
    assert not hasattr(unavailable, "cycles")


# ---------------------------------------------------------------------------
# 2. Reference path
# ---------------------------------------------------------------------------

@pytest.mark.reference
@pytest.mark.integration
def test_04_reference_context_receives_cycle_analysis():
    resolver = WorkspaceAnalysisResolver()
    context = resolver.get_analysis_context(None)

    assert isinstance(context.cycles, CycleAnalysis)

    meta = context.cycles.metadata
    # Computed from the context's own graph, not a graph rebuilt for it.
    assert meta.node_count == context.graph.number_of_nodes()
    assert meta.transaction_count == context.graph.number_of_edges()
    assert meta.cycles_retained == 5589
    assert meta.truncated is False

    # Every cycle participant is a node of that graph and an entity of that
    # prediction frame.
    entities = set(context.df["entity_id"])
    for entity in context.cycles.entity_index:
        assert entity in context.graph
        assert entity in entities


@pytest.mark.reference
@pytest.mark.integration
def test_05_reference_cycle_analysis_is_computed_once_per_resolver(spy_on_cycles):
    resolver = WorkspaceAnalysisResolver()

    first = resolver.get_analysis_context(None)
    second = resolver.get_analysis_context(None)
    third = resolver.get_analysis_context("default")

    assert len(spy_on_cycles) == 1, "cycle analysis recomputed on a cached context"
    assert first is second is third
    assert first.cycles is second.cycles is third.cycles


# ---------------------------------------------------------------------------
# 3. Workspace path
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_06_workspace_context_receives_cycle_analysis(bound_resolver, ring_workspace):
    context = bound_resolver.get_analysis_context(ring_workspace)

    assert isinstance(context.cycles, CycleAnalysis)
    assert context.cycles.metadata.node_count == context.graph.number_of_nodes()

    # The ring frame contains exactly one 3-cycle over the three entities.
    assert len(context.cycles.cycles) == 1
    cycle = context.cycles.cycles[0]
    assert cycle.length == 3
    assert set(cycle.entities) == set(_NAMES)
    assert cycle.transaction_count == 40


@pytest.mark.integration
def test_07_workspace_cycle_analysis_is_cached_with_its_context(
    bound_resolver, ring_workspace, spy_on_cycles
):
    first = bound_resolver.get_analysis_context(ring_workspace)
    second = bound_resolver.get_analysis_context(ring_workspace)

    assert len(spy_on_cycles) == 1
    assert first is second
    assert first.cycles is second.cycles


@pytest.mark.integration
def test_08_invalidate_recomputes_cycles_deterministically(
    bound_resolver, ring_workspace, spy_on_cycles
):
    before = bound_resolver.get_analysis_context(ring_workspace)
    assert len(spy_on_cycles) == 1

    bound_resolver.invalidate(ring_workspace)
    after = bound_resolver.get_analysis_context(ring_workspace)

    assert len(spy_on_cycles) == 2, "invalidation did not drop the cycle evidence"
    assert after is not before
    assert after.cycles is not before.cycles
    # Recomputation is deterministic: same graph, same evidence.
    assert [c.cycle_id for c in after.cycles.cycles] == [
        c.cycle_id for c in before.cycles.cycles
    ]


@pytest.mark.integration
def test_09_invalidate_all_drops_workspace_cycles_but_not_the_reference(
    bound_resolver, ring_workspace, spy_on_cycles
):
    reference = _stub_reference(bound_resolver)
    bound_resolver.get_analysis_context(ring_workspace)

    bound_resolver.invalidate()

    assert bound_resolver.cache == {}
    assert bound_resolver.get_analysis_context(None) is reference


# ---------------------------------------------------------------------------
# 4. Identity strategy
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_10_cycle_identity_follows_the_workspace_identity_strategy(
    store, tmp_path, bound_resolver
):
    """NAME and ACCOUNT_ID resolve different entities, so different cycles.

    Nothing in the resolver special-cases this: entity ids *are* the graph's
    nodes, and the graph was already built under the manifest's strategy.
    """
    by_name = _make_ready_workspace(
        store, tmp_path, "ident_name", _ring_frame(), IdentitySource.NAME
    )
    by_account = _make_ready_workspace(
        store, tmp_path, "ident_account", _ring_frame(), IdentitySource.ACCOUNT_ID
    )

    name_cycles = bound_resolver.get_analysis_context(by_name).cycles
    account_cycles = bound_resolver.get_analysis_context(by_account).cycles

    name_entities = set(name_cycles.cycles[0].entities)
    account_entities = set(account_cycles.cycles[0].entities)

    assert name_entities == set(_NAMES)
    assert account_entities == set(_ACCOUNTS.values())
    assert name_entities.isdisjoint(account_entities)

    # Different entities therefore mean different stable cycle identities.
    assert name_cycles.cycles[0].cycle_id != account_cycles.cycles[0].cycle_id


# ---------------------------------------------------------------------------
# 5. Workspace isolation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_11_two_workspaces_hold_independent_cycle_evidence(
    store, tmp_path, bound_resolver
):
    with_cycle = _make_ready_workspace(
        store, tmp_path, "iso_ring", _ring_frame(), IdentitySource.NAME
    )
    without_cycle = _make_ready_workspace(
        store, tmp_path, "iso_chain", _chain_frame(), IdentitySource.NAME
    )

    ring = bound_resolver.get_analysis_context(with_cycle)
    chain = bound_resolver.get_analysis_context(without_cycle)

    assert ring.cycles is not chain.cycles
    assert len(ring.cycles.cycles) == 1
    # A real analysis of a real acyclic graph: empty, not unavailable.
    assert isinstance(chain.cycles, CycleAnalysis)
    assert chain.cycles.cycles == ()
    assert chain.cycles.metadata.node_count > 0
    assert chain.cycles.entity_index == {}

    assert set(ring.cycles.entity_index).isdisjoint(set(chain.graph.nodes()))


@pytest.mark.integration
def test_12_switching_between_reference_and_a_workspace_keeps_evidence_attached(
    bound_resolver, ring_workspace
):
    reference = _stub_reference(bound_resolver)
    assert reference.cycles is None  # the stub never attempted analysis

    workspace = bound_resolver.get_analysis_context(ring_workspace)
    assert isinstance(workspace.cycles, CycleAnalysis)

    # Reference -> workspace -> Reference: the stub is untouched.
    back = bound_resolver.get_analysis_context(None)
    assert back is reference
    assert back.cycles is None
    assert workspace.cycles is not None


@pytest.mark.integration
def test_13_workspace_a_then_b_then_a_never_returns_stale_evidence(
    store, tmp_path, bound_resolver
):
    a = _make_ready_workspace(
        store, tmp_path, "abba_ring", _ring_frame(), IdentitySource.NAME
    )
    b = _make_ready_workspace(
        store, tmp_path, "abba_chain", _chain_frame(), IdentitySource.NAME
    )

    first_a = bound_resolver.get_analysis_context(a)
    only_b = bound_resolver.get_analysis_context(b)
    second_a = bound_resolver.get_analysis_context(a)

    assert second_a is first_a
    assert second_a.cycles is first_a.cycles
    assert second_a.workspace_id == a
    assert len(second_a.cycles.cycles) == 1
    assert only_b.cycles.cycles == ()

    # And after invalidating only B, A's evidence is still A's.
    bound_resolver.invalidate(b)
    assert bound_resolver.get_analysis_context(a).cycles is first_a.cycles


# ---------------------------------------------------------------------------
# 6. Failure and degradation semantics
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_14_cycle_failure_leaves_workspace_predictions_untouched(
    bound_resolver, ring_workspace, monkeypatch
):
    healthy = bound_resolver.get_analysis_context(ring_workspace)
    healthy_df = healthy.df.copy(deep=True)
    bound_resolver.invalidate(ring_workspace)

    def exploding(graph, config=None):
        raise TypeError("simulated enumeration failure")

    monkeypatch.setattr(cycles_module, "analyze_cycles", exploding)
    degraded = bound_resolver.get_analysis_context(ring_workspace)

    # The enrichment failed, explicitly.
    assert isinstance(degraded.cycles, CycleAnalysisUnavailable)
    assert degraded.cycles.reason == UNAVAILABLE_ANALYSIS_FAILED
    assert "simulated enumeration failure" in degraded.cycles.detail

    # The model side is byte-for-byte what it was.
    pd.testing.assert_frame_equal(degraded.df, healthy_df)
    assert degraded.graph.number_of_nodes() == healthy.graph.number_of_nodes()
    assert degraded.graph.number_of_edges() == healthy.graph.number_of_edges()


@pytest.mark.reference
@pytest.mark.integration
def test_15_cycle_failure_leaves_reference_risk_scores_untouched(monkeypatch):
    healthy = WorkspaceAnalysisResolver().get_analysis_context(None)
    healthy_df = healthy.df.copy(deep=True)

    def exploding(graph, config=None):
        raise TypeError("simulated enumeration failure")

    monkeypatch.setattr(cycles_module, "analyze_cycles", exploding)
    degraded = WorkspaceAnalysisResolver().get_analysis_context(None)

    assert isinstance(degraded.cycles, CycleAnalysisUnavailable)
    pd.testing.assert_frame_equal(degraded.df, healthy_df)
    assert list(degraded.df["risk_score"]) == list(healthy_df["risk_score"])
    assert list(degraded.df["predicted_class"]) == list(healthy_df["predicted_class"])


@pytest.mark.integration
def test_16_workspace_status_and_trust_are_unaffected_by_cycle_failure(
    store, bound_resolver, ring_workspace, monkeypatch
):
    """Enrichment failure must not touch READY / DEGRADED / BLOCKED or trust."""
    import json

    workspace = store.get_workspace(ring_workspace)
    compatibility = json.loads(
        (workspace.directory / "compatibility.json").read_text(encoding="utf-8")
    )
    trust = json.loads((workspace.directory / "trust.json").read_text(encoding="utf-8"))

    def exploding(graph, config=None):
        raise TypeError("simulated enumeration failure")

    monkeypatch.setattr(cycles_module, "analyze_cycles", exploding)
    context = bound_resolver.get_analysis_context(ring_workspace)
    assert isinstance(context.cycles, CycleAnalysisUnavailable)

    after = store.get_workspace(ring_workspace)
    assert after.manifest.prediction_status == "COMPLETE"
    assert json.loads(
        (after.directory / "compatibility.json").read_text(encoding="utf-8")
    ) == compatibility
    assert json.loads(
        (after.directory / "trust.json").read_text(encoding="utf-8")
    ) == trust


@pytest.mark.fast
def test_17_unexpected_exceptions_are_not_swallowed(monkeypatch):
    """The failure boundary is narrow: only enumerated errors are recovered."""

    def unexpected(graph, config=None):
        raise KeyError("a defect, not a property of the graph")

    monkeypatch.setattr(cycles_module, "analyze_cycles", unexpected)

    with pytest.raises(KeyError):
        analyze_cycles_safely(nx.MultiDiGraph())


@pytest.mark.fast
def test_18_every_recoverable_error_becomes_an_explicit_state(monkeypatch):
    for error in (TypeError, ValueError, MemoryError, RecursionError):

        def raising(graph, config=None, _error=error):
            raise _error("boom")

        monkeypatch.setattr(cycles_module, "analyze_cycles", raising)
        result = analyze_cycles_safely(nx.MultiDiGraph())

        assert isinstance(result, CycleAnalysisUnavailable)
        assert result.error_type == error.__name__
        assert result.reason == UNAVAILABLE_ANALYSIS_FAILED


# ---------------------------------------------------------------------------
# 7. Frozen model invariants after integration
# ---------------------------------------------------------------------------

@pytest.mark.reference
@pytest.mark.integration
def test_19_reference_predictions_still_match_the_pinned_regression_anchors():
    """The Stage 2 integration must not move the frozen checkpoint's output.

    ``tests/test_reference_regression.py`` pins these through ``run_inference``;
    this pins the same numbers as seen *through the resolver*, which is the code
    path Stage 2 actually changed.
    """
    import math

    context = WorkspaceAnalysisResolver().get_analysis_context(None)
    ranked = context.df.sort_values(
        by="risk_score", ascending=False
    ).reset_index(drop=True)

    assert len(context.df) == 3440
    for position, (entity, score) in enumerate(
        [
            ("HILUX SERVICES LP", 0.56275397),
            ("LCM ALLIANCE LLP", 0.55135107),
            ("METASTAR INVEST LLP", 0.55129605),
        ]
    ):
        assert ranked.iloc[position]["entity_id"] == entity
        assert math.isclose(ranked.iloc[position]["risk_score"], score, abs_tol=1e-5)
        assert ranked.iloc[position]["predicted_class"] == 1

    # And the prediction frame carries no cycle-derived column.
    assert not any("cycle" in column.lower() for column in context.df.columns)


@pytest.mark.fast
def test_20_checkpoint_contract_is_untouched():
    """167 features, in exactly one order, pinned by digest.

    A digest over the joined names fails on any addition, removal or reorder —
    including one that keeps the count at 167.
    """
    import hashlib

    import torch

    checkpoint = torch.load(
        "artifacts/models/experiment3_graphsage.pt", map_location="cpu", weights_only=False
    )
    metadata = checkpoint["metadata"]
    names = metadata["feature_names"]

    assert metadata["input_dim"] == 167
    assert len(names) == 167
    assert len(set(names)) == 167
    assert (
        hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()
        == "6555c2cda28f8b3b71f592c3672bb75bce090b10cf18557dc8250118fcff8c58"
    )

    assert names[:5] == [
        "log_in_degree",
        "log_out_degree",
        "log_total_incoming_usd",
        "log_total_outgoing_usd",
        "flow_imbalance",
    ]
    assert names[-7:] == [
        "out_repeated_counterparty_ratio",
        "in_repeated_counterparty_ratio",
        "max_parallel_edge_count",
        "mean_parallel_edge_count",
        "parallel_temporal_dispersion",
        "reciprocal_out_count",
        "reciprocal_in_count",
    ]
    # Layer 10 contributes nothing to the model's input vocabulary.
    assert not any("cycle" in name.lower() for name in names)


@pytest.mark.fast
def test_21_resolver_does_not_rebuild_a_graph_for_cycle_analysis(monkeypatch):
    """Cycle analysis reads the context's graph, never a second one.

    A rebuilt graph could silently diverge from the one the predictions were
    produced from — different identity strategy, different node set, evidence
    that does not correspond to the scores beside it.
    """
    seen = {}

    class FakeState:
        def __init__(self, graph):
            self.graph = graph

    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="t0", date=None, amount_usd=None)
    graph.add_edge("B", "A", key="t1", date=None, amount_usd=None)
    state = FakeState(graph)

    monkeypatch.setattr(ui_resolver, "prepare_detection_data", lambda path: state)
    monkeypatch.setattr(
        ui_resolver,
        "predict_from_prepared_data",
        lambda s, ckpt: pd.DataFrame({"entity_id": ["A", "B"]}),
    )

    def recording(g, config=None):
        seen["graph"] = g
        return analyze_cycles(g, config)

    monkeypatch.setattr(ui_resolver, "analyze_cycles_safely", recording)

    context = WorkspaceAnalysisResolver().get_analysis_context(None)

    assert seen["graph"] is state.graph
    assert context.graph is state.graph
    assert len(context.cycles.cycles) == 1


# ---------------------------------------------------------------------------
# 8. Compatibility with the existing stub-context tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_22_existing_stub_context_construction_still_works():
    """The shapes used by test_api_workspace_dashboard / test_inference_contract."""
    graph = nx.MultiDiGraph()
    graph.add_edge("REF1", "REF2")

    positional = AnalysisContext(None, pd.DataFrame({"entity_id": ["REF1"]}), graph)
    keyword = AnalysisContext(
        workspace_id="ws1", df=pd.DataFrame({"entity_id": ["WS1"]}), graph=graph
    )

    assert positional.cycles is None
    assert keyword.cycles is None
    assert keyword.workspace_id == "ws1"


@pytest.mark.fast
def test_23_a_context_can_be_deep_copied_with_its_evidence():
    """Contexts are handed around and copied; the evidence must survive that."""
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="t0", date=None, amount_usd=None)
    graph.add_edge("B", "A", key="t1", date=None, amount_usd=None)

    context = AnalysisContext(
        workspace_id=None,
        df=pd.DataFrame({"entity_id": ["A", "B"]}),
        graph=graph,
        cycles=analyze_cycles(graph),
    )
    clone = copy.deepcopy(context)

    assert [c.cycle_id for c in clone.cycles.cycles] == [
        c.cycle_id for c in context.cycles.cycles
    ]
    assert clone.cycles.metadata.config == context.cycles.metadata.config
