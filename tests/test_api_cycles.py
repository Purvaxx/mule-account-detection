"""
Layer 10 — Stage 3 / 4A: the cycle evidence API
===============================================

Exercises the endpoint over a real ``APIHandler`` on a real socket, against
resolver contexts built from hand-made graphs, following the pattern already
used by ``tests/test_api_workspace_dashboard.py``.

Not repeated here: the Stage 1 algorithm tests (``tests/test_cycles.py``) and
the Stage 2 resolver-cache tests (``tests/test_cycles_context.py``).  What is
asserted below is the API surface — the two response states, entity filtering,
JSON safety, workspace isolation, status codes, and that none of it touches the
frozen model's output — plus (Stage 4A) the index/detail split that keeps the
initial payload small without hiding anything.
"""

import json
import threading
import time
from socketserver import TCPServer
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import requests

from src.detection.cycles import (
    CycleAnalysisConfig,
    CycleAnalysisUnavailable,
    analyze_cycles,
    analyze_cycles_safely,
)
from src.ui.adapter import (
    CYCLE_DETAIL_ONLY_FIELDS,
    CYCLE_DETAIL_ROUTE,
    CYCLE_REASON_NOT_ATTEMPTED,
    CYCLE_STATE_COMPLETE,
    CYCLE_STATE_UNAVAILABLE,
    DetectionAdapter,
)
from src.ui.config import UIConfig
from src.ui.app import APIHandler
from src.ui.context import (
    AnalysisContext,
    InvalidWorkspaceError,
    WorkspaceNotFoundError,
    WorkspaceNotReadyError,
)


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

def _edge(graph, source, target, key, when, amount):
    graph.add_edge(
        source,
        target,
        key=key,
        transaction_id=key,
        date=None if when is None else pd.Timestamp(when),
        amount_usd=amount,
    )


def _reference_like_graph():
    """A 3-cycle at day-level precision, plus an acyclic tail.

    ``REF_D`` transacts but is in no cycle, which is what makes the "valid
    entity, zero cycles" case testable.  Amounts are deliberately incomplete on
    one leg so ``None`` survives serialization.
    """
    graph = nx.MultiDiGraph()
    _edge(graph, "REF_A", "REF_B", "t1", "2024-01-01", 100.0)
    _edge(graph, "REF_A", "REF_B", "t2", "2024-01-02", None)      # missing amount
    _edge(graph, "REF_B", "REF_C", "t3", "2024-01-03", 90.0)
    _edge(graph, "REF_C", "REF_A", "t4", "2024-01-04", 80.0)
    _edge(graph, "REF_C", "REF_D", "t5", "2024-01-05", 10.0)      # acyclic tail
    return graph


def _workspace_graph():
    """A different 3-cycle with sub-day precision and numpy-typed ids."""
    graph = nx.MultiDiGraph()
    _edge(graph, "WS_X", "WS_Y", np.int64(901), "2024-05-01 09:00:00", np.float64(500.0))
    _edge(graph, "WS_Y", "WS_Z", np.int64(902), "2024-05-01 10:00:00", np.float64(450.0))
    _edge(graph, "WS_Z", "WS_X", np.int64(903), "2024-05-01 11:00:00", np.float64(400.0))
    return graph


def _acyclic_graph():
    graph = nx.MultiDiGraph()
    _edge(graph, "NC_1", "NC_2", "n1", "2024-02-01", 5.0)
    _edge(graph, "NC_2", "NC_3", "n2", "2024-02-02", 5.0)
    return graph


def _frame(entity_ids, scores):
    return pd.DataFrame(
        {
            "entity_id": entity_ids,
            "risk_score": scores,
            "predicted_class": [1 if s >= 0.45 else 0 for s in scores],
            "reciprocal_out": [0] * len(entity_ids),
            "reciprocal_in": [0] * len(entity_ids),
            "out_repeated_ratio": [0.0] * len(entity_ids),
            "in_repeated_ratio": [0.0] * len(entity_ids),
            "max_parallel_edge": [1] * len(entity_ids),
            "flow_imbalance": [0.5] * len(entity_ids),
            "in_degree": [1] * len(entity_ids),
            "out_degree": [1] * len(entity_ids),
        }
    )


@pytest.fixture
def contexts():
    reference_graph = _reference_like_graph()
    workspace_graph = _workspace_graph()
    acyclic_graph = _acyclic_graph()
    broken_graph = _reference_like_graph()

    return {
        "default": AnalysisContext(
            workspace_id=None,
            df=_frame(["REF_A", "REF_B", "REF_C", "REF_D"], [0.9, 0.8, 0.7, 0.1]),
            graph=reference_graph,
            cycles=analyze_cycles(reference_graph),
        ),
        "ws1": AnalysisContext(
            workspace_id="ws1",
            df=_frame(["WS_X", "WS_Y", "WS_Z"], [0.95, 0.5, 0.05]),
            graph=workspace_graph,
            cycles=analyze_cycles(workspace_graph),
        ),
        "ws_empty": AnalysisContext(
            workspace_id="ws_empty",
            df=_frame(["NC_1", "NC_2", "NC_3"], [0.2, 0.3, 0.4]),
            graph=acyclic_graph,
            cycles=analyze_cycles(acyclic_graph),
        ),
        "ws_broken": AnalysisContext(
            workspace_id="ws_broken",
            df=_frame(["REF_A", "REF_B", "REF_C", "REF_D"], [0.9, 0.8, 0.7, 0.1]),
            graph=broken_graph,
            cycles=analyze_cycles_safely("not a graph"),
        ),
        "ws_stub": AnalysisContext(
            workspace_id="ws_stub",
            df=_frame(["REF_A"], [0.4]),
            graph=_reference_like_graph(),
            # cycles left at its default: never attempted.
        ),
    }


@pytest.fixture
def server(contexts):
    """A real APIHandler over a real socket, resolving to ``contexts``."""
    resolve_calls = []

    with patch("src.ui.app.resolver.get_analysis_context") as mock_resolve:

        def side_effect(workspace_id):
            resolve_calls.append(workspace_id)
            if not workspace_id or workspace_id == "default":
                return contexts["default"]
            if workspace_id in contexts:
                return contexts[workspace_id]
            if workspace_id == "missing":
                raise WorkspaceNotFoundError("Not found")
            if workspace_id == "incomplete":
                raise WorkspaceNotReadyError("Not ready")
            if workspace_id == "invalid!":
                raise InvalidWorkspaceError("Invalid ID")
            return contexts["default"]

        mock_resolve.side_effect = side_effect

        httpd = TCPServer(("127.0.0.1", 0), APIHandler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, args=(0.01,))
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        yield port, resolve_calls

        httpd.shutdown()
        httpd.server_close()


def _get(server, query=""):
    """GET the cycle index."""
    port, _ = server
    return requests.get(f"http://127.0.0.1:{port}/api/cycles{query}")


def _detail(server, cycle_id, query=""):
    """GET one cycle's complete evidence."""
    port, _ = server
    return requests.get(f"http://127.0.0.1:{port}/api/cycles/{cycle_id}{query}")


def _only_cycle(server, query=""):
    """The full evidence for the single cycle an index response lists."""
    index = _get(server, query).json()
    assert index["page"]["returned"] == 1, index["page"]
    suffix = query.replace("?", "&", 1) if query.startswith("?") else query
    workspace = ""
    if "workspace_id=" in suffix:
        workspace = "?workspace_id=" + suffix.split("workspace_id=")[1].split("&")[0]
    return _detail(server, index["cycles"][0]["cycle_id"], workspace).json()["cycle"]


# ---------------------------------------------------------------------------
# 1. Complete analysis
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_01_complete_analysis_returns_cycles_and_metadata(server):
    response = _get(server)
    assert response.status_code == 200
    body = response.json()

    assert body["state"] == CYCLE_STATE_COMPLETE
    assert body["workspace_id"] is None
    assert body["entity_id"] is None
    assert body["filtered_by_entity"] is False
    assert body["unavailable"] is None

    assert body["summary"]["total_cycles"] == 1
    assert body["page"] == {
        "limit": 100, "offset": 0, "returned": 1, "total": 1, "has_more": False,
    }

    row = body["cycles"][0]
    assert row["entities"] == ["REF_A", "REF_B", "REF_C"]
    assert row["length"] == 3
    assert row["transaction_count"] == 4
    assert row["has_detail"] is True

    # An index row is a summary: the per-leg evidence lives on the detail route
    # and the row says which fields it does not carry.
    assert "legs" not in row
    assert "legs" in body["detail"]["omitted_fields"]
    assert body["detail"]["route"] == "/api/cycles/{cycle_id}"

    cycle = _detail(server, row["cycle_id"]).json()["cycle"]
    assert [(leg["source"], leg["target"]) for leg in cycle["legs"]] == [
        ("REF_A", "REF_B"),
        ("REF_B", "REF_C"),
        ("REF_C", "REF_A"),
    ]


@pytest.mark.integration
def test_02_metadata_carries_the_bounds_and_counts_for_reproducibility(server):
    metadata = _get(server).json()["metadata"]

    assert metadata["bounds"] == {
        "max_cycle_length": 6,
        "min_cycle_length": 2,
        "max_cycles": 10000,
        "max_transaction_samples_per_leg": 5,
    }
    assert metadata["temporal_precision"] == "DAY"
    assert metadata["temporal_precision_basis"]
    assert metadata["enumeration_complete"] is True
    assert metadata["truncated"] is False
    assert metadata["truncation_reason"] is None
    assert isinstance(metadata["elapsed_seconds"], float)

    counts = metadata["counts"]
    assert counts["entities"] == 4
    assert counts["transactions"] == 5
    assert counts["unique_directed_pairs"] == 4
    assert counts["cycles_retained"] == 1
    assert counts["transactions_with_amount"] == 4  # one is genuinely missing
    assert counts["timestamped_transactions"] == 5


@pytest.mark.integration
def test_03_no_internal_implementation_detail_is_exposed(server):
    body = _get(server).json()
    text = json.dumps(body)

    for leaked in (
        "Traceback",
        "src.detection.cycles",
        "src/ui",
        "networkx",
        "MultiDiGraph",
        "_PairBundle",
        "self_loop_entities",
    ):
        assert leaked not in text, f"response exposes {leaked!r}"


# ---------------------------------------------------------------------------
# 2. Empty result — distinct from unavailable
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_04_acyclic_dataset_is_complete_with_an_empty_list(server):
    body = _get(server, "?workspace_id=ws_empty").json()

    assert body["state"] == CYCLE_STATE_COMPLETE
    assert body["cycles"] == []
    assert body["summary"]["total_cycles"] == 0
    assert body["summary"]["by_length"] == {}
    assert body["summary"]["shortest_cycle_length"] is None
    assert body["page"]["returned"] == 0
    assert body["page"]["has_more"] is False
    assert body["unavailable"] is None
    # A real analysis really looked at a real graph.
    assert body["metadata"]["counts"]["entities"] == 3
    assert body["metadata"]["counts"]["cycles_retained"] == 0


@pytest.mark.integration
def test_05_entity_in_no_cycle_is_complete_and_empty_not_an_error(server):
    response = _get(server, "?entity_id=REF_D")
    assert response.status_code == 200

    body = response.json()
    assert body["state"] == CYCLE_STATE_COMPLETE
    assert body["cycles"] == []
    assert body["summary"]["total_cycles"] == 0
    assert body["page"]["total"] == 0
    assert body["entity_id"] == "REF_D"
    assert body["filtered_by_entity"] is True
    # The analysis itself did find a cycle; this entity is not in it.
    assert body["metadata"]["counts"]["cycles_retained"] == 1


# ---------------------------------------------------------------------------
# 3. Unavailable analysis
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_06_failed_analysis_is_unavailable_never_an_empty_list(server):
    response = _get(server, "?workspace_id=ws_broken")
    # A well-formed request whose evidence does not exist is not a client error.
    assert response.status_code == 200

    body = response.json()
    assert body["state"] == CYCLE_STATE_UNAVAILABLE
    assert body["cycles"] is None
    assert body["cycles"] != []
    assert body["metadata"] is None
    # Every count-bearing field is null too, so nothing reads as "zero cycles".
    assert body["summary"] is None
    assert body["page"] is None

    assert body["unavailable"]["reason"] == "CYCLE_ANALYSIS_FAILED"
    assert body["unavailable"]["error_type"] == "TypeError"
    assert "not a finding that the dataset has no cycles" in body["unavailable"]["message"]


@pytest.mark.integration
def test_07_never_attempted_is_unavailable_with_its_own_reason(server):
    body = _get(server, "?workspace_id=ws_stub").json()

    assert body["state"] == CYCLE_STATE_UNAVAILABLE
    assert body["cycles"] is None
    assert body["unavailable"]["reason"] == CYCLE_REASON_NOT_ATTEMPTED
    assert body["unavailable"]["error_type"] is None


@pytest.mark.integration
def test_08_unavailable_response_carries_no_exception_text(server):
    body = _get(server, "?workspace_id=ws_broken").json()
    text = json.dumps(body["unavailable"])

    # analyze_cycles_safely stores the interpreter message in `detail`; it is
    # logged, never served.
    assert "detail" not in body["unavailable"]
    assert "analyze_cycles requires" not in text
    assert "Traceback" not in text
    assert "File \"" not in text


@pytest.mark.integration
def test_09_unavailable_is_scoped_to_its_own_workspace(server):
    broken = _get(server, "?workspace_id=ws_broken").json()
    healthy = _get(server, "?workspace_id=ws1").json()
    reference = _get(server).json()

    assert broken["state"] == CYCLE_STATE_UNAVAILABLE
    assert healthy["state"] == CYCLE_STATE_COMPLETE
    assert reference["state"] == CYCLE_STATE_COMPLETE
    assert healthy["summary"]["total_cycles"] == 1
    assert reference["summary"]["total_cycles"] == 1


# ---------------------------------------------------------------------------
# 4-5. Entity filtering and canonical identity
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_10_entity_filter_selects_only_that_entitys_cycles(server):
    unfiltered = _get(server).json()
    filtered = _get(server, "?entity_id=REF_B").json()

    assert filtered["state"] == CYCLE_STATE_COMPLETE
    assert filtered["summary"]["total_cycles"] == 1
    assert all("REF_B" in row["entities"] for row in filtered["cycles"])
    # Three counts, three meanings, never conflated.
    assert filtered["page"]["returned"] == 1
    assert filtered["metadata"]["counts"]["cycles_retained"] == 1
    assert unfiltered["summary"]["total_cycles"] == 1


@pytest.mark.integration
def test_11_filtering_returns_whole_cycles_never_re_rooted(server):
    """Querying B of A->B->C->A returns A->B->C->A, not B->C->A."""
    unfiltered = _get(server).json()["cycles"][0]
    filtered = _get(server, "?entity_id=REF_B").json()["cycles"][0]

    assert filtered["entities"] == ["REF_A", "REF_B", "REF_C"]
    assert filtered["entities"][0] != "REF_B"
    assert filtered["length"] == 3
    # Byte-for-byte the same row the unfiltered index returns.
    assert filtered == unfiltered

    # And the same holds for the full evidence behind it.
    detail = _detail(server, filtered["cycle_id"]).json()["cycle"]
    assert detail["entities"] == ["REF_A", "REF_B", "REF_C"]
    assert len(detail["legs"]) == 3
    assert detail["legs"][0]["source"] == "REF_A"


@pytest.mark.integration
def test_12_canonical_cycle_id_is_unchanged_by_filtering(server):
    unfiltered = _get(server).json()["cycles"][0]["cycle_id"]

    for entity in ("REF_A", "REF_B", "REF_C"):
        body = _get(server, f"?entity_id={entity}").json()
        assert body["cycles"][0]["cycle_id"] == unfiltered


@pytest.mark.integration
def test_13_entity_filter_is_scoped_to_the_requested_workspace(server):
    """A Reference entity id must not resolve inside a workspace."""
    response = _get(server, "?workspace_id=ws1&entity_id=REF_B")
    assert response.status_code == 404

    inside = _get(server, "?workspace_id=ws1&entity_id=WS_Y").json()
    assert inside["state"] == CYCLE_STATE_COMPLETE
    assert inside["cycles"][0]["entities"] == ["WS_X", "WS_Y", "WS_Z"]


# ---------------------------------------------------------------------------
# 6-8. Serialization
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_14_response_is_json_native_throughout(server):
    body = _get(server, "?workspace_id=ws1").json()

    def walk(node, path="$"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert isinstance(key, str), path
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        else:
            assert node is None or isinstance(node, (str, int, float, bool)), (
                f"{path} is {type(node).__name__}"
            )
            assert not isinstance(node, (np.generic, pd.Timestamp)), path

    walk(body)
    # Round-trips without a custom encoder, which is all send_json has.
    assert json.loads(json.dumps(body)) == body


@pytest.mark.integration
def test_15_datetimes_are_iso_8601_strings(server):
    row = _get(server, "?workspace_id=ws1").json()["cycles"][0]
    cycle = _only_cycle(server, "?workspace_id=ws1")

    # Summary row and full evidence agree, and both are ISO 8601.
    assert row["first_transaction_at"] == "2024-05-01T09:00:00"
    assert cycle["first_transaction_at"] == row["first_transaction_at"]
    assert cycle["last_transaction_at"] == "2024-05-01T11:00:00"
    assert cycle["legs"][0]["first_timestamp"] == "2024-05-01T09:00:00"
    for value in (cycle["first_transaction_at"], cycle["last_transaction_at"]):
        assert isinstance(value, str)
        assert pd.Timestamp(value)  # parses back


@pytest.mark.integration
def test_16_missing_amounts_serialize_as_null_never_zero(server):
    cycle = _only_cycle(server)
    leg = next(l for l in cycle["legs"] if l["source"] == "REF_A")

    # Two transactions, one without an amount.
    assert leg["transaction_count"] == 2
    assert leg["transactions_with_amount"] == 1
    assert leg["amount_evidence"] == "PARTIAL"
    assert leg["amount_usd_total"] == 100.0  # the sum over what was observed

    # Cycle-level derived amount properties need complete evidence, and stay
    # null rather than being computed over a partial bundle.
    assert cycle["amount_evidence"] == "PARTIAL"
    assert cycle["leg_amount_balance"] is None
    assert cycle["amount_retention"] is None
    assert cycle["amount_retention_basis"] is None

    cycle_id = _get(server).json()["cycles"][0]["cycle_id"]
    raw = _detail(server, cycle_id).text
    assert '"leg_amount_balance": null' in raw or '"leg_amount_balance":null' in raw


@pytest.mark.integration
def test_17_three_valued_temporal_verdict_survives_serialization(server):
    day_level = _only_cycle(server)
    sub_day = _only_cycle(server, "?workspace_id=ws1")

    # Not establishable stays null — it must never arrive as false.
    assert day_level["is_time_respecting"] is None
    assert day_level["is_time_respecting"] is not False
    assert day_level["temporal_basis"] == "DAY_LEVEL_PRECISION"

    assert sub_day["is_time_respecting"] is True
    assert sub_day["temporal_basis"] == "STRICT_ORDER_ESTABLISHED"

    # Literal null on the wire, on both the index row and the full evidence.
    index_raw = _get(server).text
    detail_raw = _detail(server, _get(server).json()["cycles"][0]["cycle_id"]).text
    for raw in (index_raw, detail_raw):
        assert '"is_time_respecting": null' in raw or '"is_time_respecting":null' in raw


@pytest.mark.integration
def test_18_numpy_typed_ids_and_amounts_become_native(server):
    leg = _only_cycle(server, "?workspace_id=ws1")["legs"][0]

    assert leg["transaction_id_sample"] == [901]
    assert type(leg["transaction_id_sample"][0]) is int
    assert leg["amount_usd_total"] == 500.0
    assert type(leg["amount_usd_total"]) is float


# ---------------------------------------------------------------------------
# 9. Workspace isolation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_19_workspace_a_never_returns_workspace_b_cycles(server):
    reference = _get(server).json()
    workspace = _get(server, "?workspace_id=ws1").json()

    reference_entities = {e for c in reference["cycles"] for e in c["entities"]}
    workspace_entities = {e for c in workspace["cycles"] for e in c["entities"]}

    assert reference_entities == {"REF_A", "REF_B", "REF_C"}
    assert workspace_entities == {"WS_X", "WS_Y", "WS_Z"}
    assert reference_entities.isdisjoint(workspace_entities)

    assert reference["cycles"][0]["cycle_id"] != workspace["cycles"][0]["cycle_id"]
    assert reference["workspace_id"] is None
    assert workspace["workspace_id"] == "ws1"


@pytest.mark.integration
def test_20_switching_workspace_ids_produces_the_right_result_each_time(server):
    sequence = [None, "ws1", "ws_empty", "ws1", None, "ws_broken", "ws1"]
    seen = []
    for workspace_id in sequence:
        query = "" if workspace_id is None else f"?workspace_id={workspace_id}"
        body = _get(server, query).json()
        total = None if body["summary"] is None else body["summary"]["total_cycles"]
        seen.append((body["state"], total))

    assert seen == [
        (CYCLE_STATE_COMPLETE, 1),
        (CYCLE_STATE_COMPLETE, 1),
        (CYCLE_STATE_COMPLETE, 0),
        (CYCLE_STATE_COMPLETE, 1),
        (CYCLE_STATE_COMPLETE, 1),
        (CYCLE_STATE_UNAVAILABLE, None),
        (CYCLE_STATE_COMPLETE, 1),
    ]


# ---------------------------------------------------------------------------
# 10. Cached context reuse
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_21_the_endpoint_reuses_the_resolved_context_and_never_recomputes(
    server, monkeypatch
):
    """Repeated requests must not re-enumerate cycles.

    The resolver owns the cache; the endpoint must read through it and do no
    analysis of its own.  A spy on the domain entry point proves the request
    path never calls it.
    """
    import src.detection.cycles as cycles_module

    calls = []
    real = cycles_module.analyze_cycles

    def counted(graph, config=None):
        calls.append(graph)
        return real(graph, config)

    monkeypatch.setattr(cycles_module, "analyze_cycles", counted)

    for _ in range(5):
        assert _get(server).status_code == 200
        assert _get(server, "?entity_id=REF_B").status_code == 200
        assert _get(server, "?workspace_id=ws1").status_code == 200

    assert calls == [], "the API recomputed cycle analysis per request"

    # And the responses are identical across repeats, not merely successful.
    first = _get(server).json()
    last = _get(server).json()
    assert first["cycles"] == last["cycles"]
    assert first["metadata"]["counts"] == last["metadata"]["counts"]


@pytest.mark.integration
def test_22_each_request_resolves_through_the_shared_resolver(server):
    _, resolve_calls = server
    before = len(resolve_calls)

    _get(server, "?workspace_id=ws1")
    _get(server, "?workspace_id=ws_empty")

    assert resolve_calls[before:] == ["ws1", "ws_empty"]


# ---------------------------------------------------------------------------
# 11. Client errors
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize(
    "workspace_id,status",
    [("missing", 404), ("incomplete", 409), ("invalid!", 400)],
)
def test_23_workspace_errors_use_the_existing_status_contract(
    server, workspace_id, status
):
    import urllib.parse

    encoded = urllib.parse.quote(workspace_id)
    assert _get(server, f"?workspace_id={encoded}").status_code == status


@pytest.mark.integration
def test_24_unknown_entity_is_404_like_the_entity_endpoint(server):
    port, _ = server

    cycles = requests.get(f"http://127.0.0.1:{port}/api/cycles?entity_id=NOBODY")
    entity = requests.get(f"http://127.0.0.1:{port}/api/entity/NOBODY")

    assert cycles.status_code == 404
    assert entity.status_code == 404


@pytest.mark.integration
def test_25_unknown_entity_is_404_even_when_analysis_is_unavailable(server):
    """A client error is reported as one regardless of the evidence state."""
    unknown = _get(server, "?workspace_id=ws_broken&entity_id=NOBODY")
    assert unknown.status_code == 404

    known = _get(server, "?workspace_id=ws_broken&entity_id=REF_A")
    assert known.status_code == 200
    assert known.json()["state"] == CYCLE_STATE_UNAVAILABLE


@pytest.mark.integration
def test_26_entity_ids_needing_url_encoding_round_trip(server):
    import urllib.parse

    graph = nx.MultiDiGraph()
    _edge(graph, "A/B CO., LTD", "X&Y LP", "e1", "2024-01-01", 10.0)
    _edge(graph, "X&Y LP", "A/B CO., LTD", "e2", "2024-01-02", 10.0)

    with patch("src.ui.app.resolver.get_analysis_context") as mock_resolve:
        mock_resolve.return_value = AnalysisContext(
            workspace_id=None,
            df=_frame(["A/B CO., LTD", "X&Y LP"], [0.9, 0.8]),
            graph=graph,
            cycles=analyze_cycles(graph),
        )
        encoded = urllib.parse.quote("A/B CO., LTD", safe="")
        body = _get(server, f"?entity_id={encoded}").json()

    assert body["state"] == CYCLE_STATE_COMPLETE
    assert body["entity_id"] == "A/B CO., LTD"
    assert body["summary"]["total_cycles"] == 1
    assert set(body["cycles"][0]["entities"]) == {"A/B CO., LTD", "X&Y LP"}


# ---------------------------------------------------------------------------
# 12-13. The frozen model is untouched
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_27_predictions_are_identical_before_and_after_querying_cycles(
    server, contexts
):
    port, _ = server
    before = requests.get(f"http://127.0.0.1:{port}/api/overview").json()

    _get(server)
    _get(server, "?entity_id=REF_A")
    _get(server, "?workspace_id=ws_broken")

    after = requests.get(f"http://127.0.0.1:{port}/api/overview").json()
    assert before == after

    entity = requests.get(f"http://127.0.0.1:{port}/api/entity/REF_A").json()
    assert entity["risk_score"] == 0.9
    assert entity["predicted_class"] == 1


@pytest.mark.integration
def test_28_the_cycles_response_carries_no_model_output(server):
    body = _get(server).json()
    text = json.dumps(body)

    for forbidden in (
        "risk_score",
        "predicted_class",
        "cycle_risk",
        "suspicious",
        "threshold",
        "probability",
    ):
        assert forbidden not in text, f"cycle evidence exposes {forbidden!r}"


@pytest.mark.integration
def test_29_no_cycle_field_leaks_into_the_prediction_view_models(server):
    port, _ = server

    overview = requests.get(f"http://127.0.0.1:{port}/api/overview").json()
    entity = requests.get(f"http://127.0.0.1:{port}/api/entity/REF_A").json()
    network = requests.get(f"http://127.0.0.1:{port}/api/network/REF_A").json()

    for payload in (overview, entity, network):
        assert "cycle" not in json.dumps(payload).lower()


# ---------------------------------------------------------------------------
# Adapter-level checks that do not need a socket
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_30_adapter_does_not_mutate_the_context(contexts):
    adapter = DetectionAdapter()
    context = contexts["default"]

    before_ids = [c.cycle_id for c in context.cycles.cycles]
    before_frame = context.df.copy(deep=True)

    adapter.get_cycles(context)
    adapter.get_cycles(context, entity_id="REF_B")

    assert [c.cycle_id for c in context.cycles.cycles] == before_ids
    pd.testing.assert_frame_equal(context.df, before_frame)


@pytest.mark.fast
def test_31_truncation_is_reported_through_the_api_shape():
    """A truncated analysis must say so in the metadata it returns."""
    adapter = DetectionAdapter()

    graph = nx.MultiDiGraph()
    names = [f"N{i:02d}" for i in range(6)]
    for source in names:
        for target in names:
            if source != target:
                _edge(graph, source, target, f"{source}{target}", "2024-01-01", 1.0)

    analysis = analyze_cycles(
        graph, CycleAnalysisConfig(max_cycle_length=4, max_cycles=10)
    )
    context = AnalysisContext(
        workspace_id=None,
        df=_frame(names, [0.1] * len(names)),
        graph=graph,
        cycles=analysis,
    )

    metadata = adapter.get_cycles(context)["metadata"]
    assert metadata["truncated"] is True
    assert metadata["truncation_reason"] == "MAX_CYCLES_REACHED"
    assert metadata["enumeration_complete"] is False
    assert metadata["bounds"]["max_cycles"] == 10
    assert json.loads(json.dumps(metadata)) == metadata


# ===========================================================================
# Stage 4A — index / detail split
# ===========================================================================

def _many_cycles_context(rings=12):
    """A context with many disjoint 3-cycles, for paging."""
    graph = nx.MultiDiGraph()
    names = []
    for index in range(rings):
        a, b, c = f"P{index:02d}_A", f"P{index:02d}_B", f"P{index:02d}_C"
        names += [a, b, c]
        _edge(graph, a, b, f"{index}-1", "2024-01-01", 10.0)
        _edge(graph, b, c, f"{index}-2", "2024-01-02", 10.0)
        _edge(graph, c, a, f"{index}-3", "2024-01-03", 10.0)
    return AnalysisContext(
        workspace_id="ws_many",
        df=_frame(names, [0.1] * len(names)),
        graph=graph,
        cycles=analyze_cycles(graph),
    )


@pytest.fixture
def paged_server(contexts):
    contexts["ws_many"] = _many_cycles_context()
    with patch("src.ui.app.resolver.get_analysis_context") as mock_resolve:
        mock_resolve.side_effect = lambda wid: contexts[wid or "default"]
        httpd = TCPServer(("127.0.0.1", 0), APIHandler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True).start()
        time.sleep(0.1)
        yield port, []
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Summary index
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_32_index_aggregates_the_whole_selection_not_the_page(paged_server):
    body = _get(paged_server, "?workspace_id=ws_many&limit=5").json()

    # Aggregates describe all 12 cycles; the page carries 5.
    assert body["summary"]["total_cycles"] == 12
    assert body["summary"]["by_length"] == {"3": 12}
    assert body["summary"]["shortest_cycle_length"] == 3
    assert body["summary"]["longest_cycle_length"] == 3
    assert body["summary"]["participating_entity_count"] == 36
    assert body["summary"]["reciprocal_pair_count"] == 0

    assert body["page"] == {
        "limit": 5, "offset": 0, "returned": 5, "total": 12, "has_more": True,
    }
    assert len(body["cycles"]) == 5


@pytest.mark.integration
def test_33_paging_walks_the_selection_without_gaps_or_repeats(paged_server):
    seen, offset = [], 0
    while True:
        body = _get(paged_server, f"?workspace_id=ws_many&limit=5&offset={offset}").json()
        seen += [row["cycle_id"] for row in body["cycles"]]
        if not body["page"]["has_more"]:
            break
        offset += body["page"]["returned"]

    everything = [
        row["cycle_id"]
        for row in _get(paged_server, "?workspace_id=ws_many&limit=500").json()["cycles"]
    ]
    assert seen == everything
    assert len(seen) == len(set(seen)) == 12


@pytest.mark.integration
def test_34_offset_past_the_end_is_an_empty_page_not_an_error(paged_server):
    response = _get(paged_server, "?workspace_id=ws_many&offset=999")
    assert response.status_code == 200

    body = response.json()
    assert body["state"] == CYCLE_STATE_COMPLETE
    assert body["cycles"] == []
    assert body["page"]["returned"] == 0
    assert body["page"]["has_more"] is False
    # The selection is still reported in full.
    assert body["page"]["total"] == 12
    assert body["summary"]["total_cycles"] == 12


@pytest.mark.integration
def test_35_the_page_limit_is_explicit_never_silent(paged_server):
    body = _get(paged_server, "?workspace_id=ws_many&limit=3").json()

    # Everything a reader needs to know the response was reduced, and by how
    # much, is in the response itself.
    assert body["page"]["limit"] == 3
    assert body["page"]["returned"] == 3
    assert body["page"]["total"] == 12
    assert body["page"]["has_more"] is True

    assert body["detail"]["route"] == CYCLE_DETAIL_ROUTE
    assert body["detail"]["omitted_fields"] == list(CYCLE_DETAIL_ONLY_FIELDS)
    assert set(body["detail"]["omitted_fields"]) & set(body["cycles"][0]) == set()


@pytest.mark.integration
def test_36_default_page_size_comes_from_the_project_configuration(paged_server):
    body = _get(paged_server, "?workspace_id=ws_many").json()
    assert body["page"]["limit"] == UIConfig.CYCLE_PAGE_SIZE


@pytest.mark.integration
@pytest.mark.parametrize(
    "query",
    ["?limit=abc", "?limit=0", "?limit=-1", "?offset=-5", "?offset=x",
     "?limit=" + str(UIConfig.CYCLE_PAGE_MAX_SIZE + 1)],
)
def test_37_invalid_paging_is_a_client_error_not_a_server_fault(paged_server, query):
    assert _get(paged_server, query).status_code == 400


@pytest.mark.integration
def test_38_entity_filtered_summary_aggregates_only_that_entitys_cycles(server):
    filtered = _get(server, "?entity_id=REF_B").json()

    assert filtered["filtered_by_entity"] is True
    assert filtered["summary"]["total_cycles"] == 1
    assert filtered["summary"]["by_length"] == {"3": 1}
    assert filtered["summary"]["participating_entity_count"] == 3
    # The analysis-wide count is untouched by the filter.
    assert filtered["metadata"]["counts"]["cycles_retained"] == 1

    absent = _get(server, "?entity_id=REF_D").json()
    assert absent["summary"]["total_cycles"] == 0
    assert absent["summary"]["by_length"] == {}
    assert absent["summary"]["participating_entity_count"] == 0


@pytest.mark.integration
def test_39_index_rows_carry_no_per_leg_evidence(server):
    row = _get(server).json()["cycles"][0]

    for omitted in CYCLE_DETAIL_ONLY_FIELDS:
        assert omitted not in row, f"summary row still carries {omitted!r}"
    assert row["has_detail"] is True


# ---------------------------------------------------------------------------
# Detail route
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_40_detail_returns_the_complete_stage_1_evidence(server):
    row = _get(server).json()["cycles"][0]
    response = _detail(server, row["cycle_id"])
    assert response.status_code == 200

    body = response.json()
    assert body["state"] == CYCLE_STATE_COMPLETE
    assert body["cycle_id"] == row["cycle_id"]
    assert body["workspace_id"] is None
    assert body["unavailable"] is None

    cycle = body["cycle"]
    for field in (
        "cycle_id", "entities", "length", "participant_count", "transaction_count",
        "legs", "amount_evidence", "legs_with_complete_amounts",
        "leg_amount_totals", "leg_amount_balance",
        "amount_retention", "amount_retention_basis",
        "first_transaction_at", "last_transaction_at", "time_span_days",
        "is_time_respecting", "temporal_basis", "temporal_explanation",
        "is_self_loop", "is_reciprocal_pair", "matched_rules",
    ):
        assert field in cycle, f"detail is missing {field!r}"

    leg = cycle["legs"][0]
    for field in (
        "source", "target", "transaction_count", "amount_usd_total",
        "amount_usd_min", "amount_usd_max", "transactions_with_amount",
        "amount_evidence", "first_timestamp", "last_timestamp",
        "transactions_with_timestamp", "transaction_id_sample",
        "transaction_id_sample_truncated",
    ):
        assert field in leg, f"leg is missing {field!r}"

    # The effective bounds travel with a cycle fetched on its own.
    assert body["metadata"]["bounds"]["max_cycle_length"] == 6
    assert body["metadata"]["temporal_precision"] == "DAY"


@pytest.mark.integration
def test_41_detail_agrees_with_its_index_row_field_for_field(server):
    row = _get(server).json()["cycles"][0]
    cycle = _detail(server, row["cycle_id"]).json()["cycle"]

    shared = set(row) & set(cycle)
    assert "has_detail" not in shared
    for field in shared:
        assert row[field] == cycle[field], f"{field} disagrees between index and detail"
    assert shared >= {
        "cycle_id", "entities", "length", "transaction_count",
        "amount_evidence", "leg_amount_balance", "is_time_respecting",
    }


@pytest.mark.integration
def test_42_named_rules_are_a_declared_empty_seam_not_an_invented_verdict(server):
    row = _get(server).json()["cycles"][0]
    cycle = _detail(server, row["cycle_id"]).json()["cycle"]

    assert cycle["matched_rules"] == []
    assert "no named selection rules" in cycle["matched_rules_note"].lower()
    assert "invents no threshold" in cycle["matched_rules_note"]


@pytest.mark.integration
def test_43_unknown_cycle_id_is_404(server):
    assert _detail(server, "cyc_0000000000000000").status_code == 404
    assert _detail(server, "not-a-cycle-id").status_code == 404


@pytest.mark.integration
def test_44_detail_of_an_unavailable_analysis_is_not_a_missing_cycle(server):
    """UNAVAILABLE must not be reported as 404: the server does not know."""
    response = _detail(server, "cyc_0000000000000000", "?workspace_id=ws_broken")
    assert response.status_code == 200

    body = response.json()
    assert body["state"] == CYCLE_STATE_UNAVAILABLE
    assert body["cycle"] is None
    assert body["metadata"] is None
    assert body["unavailable"]["reason"] == "CYCLE_ANALYSIS_FAILED"
    assert "detail" not in body["unavailable"]


@pytest.mark.integration
def test_45_detail_of_an_empty_analysis_is_404_because_the_cycle_is_absent(server):
    """An analysis that completed and found nothing genuinely lacks the id."""
    index = _get(server, "?workspace_id=ws_empty").json()
    assert index["state"] == CYCLE_STATE_COMPLETE
    assert index["summary"]["total_cycles"] == 0

    assert _detail(
        server, "cyc_0000000000000000", "?workspace_id=ws_empty"
    ).status_code == 404


# ---------------------------------------------------------------------------
# Workspace isolation of cycle ids
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_46_a_cycle_id_from_one_workspace_does_not_resolve_in_another(server):
    reference_id = _get(server).json()["cycles"][0]["cycle_id"]
    workspace_id = _get(server, "?workspace_id=ws1").json()["cycles"][0]["cycle_id"]
    assert reference_id != workspace_id

    # Each resolves in its own context...
    assert _detail(server, reference_id).status_code == 200
    assert _detail(server, workspace_id, "?workspace_id=ws1").status_code == 200

    # ...and in no other.
    assert _detail(server, workspace_id).status_code == 404
    assert _detail(server, reference_id, "?workspace_id=ws1").status_code == 404
    assert _detail(server, reference_id, "?workspace_id=ws_empty").status_code == 404


@pytest.mark.integration
def test_47_detail_never_returns_another_workspaces_entities(server):
    reference_id = _get(server).json()["cycles"][0]["cycle_id"]
    cycle = _detail(server, reference_id).json()["cycle"]

    assert set(cycle["entities"]) == {"REF_A", "REF_B", "REF_C"}
    assert not set(cycle["entities"]) & {"WS_X", "WS_Y", "WS_Z", "NC_1"}

    workspace_cycle = _detail(
        server,
        _get(server, "?workspace_id=ws1").json()["cycles"][0]["cycle_id"],
        "?workspace_id=ws1",
    ).json()["cycle"]
    assert set(workspace_cycle["entities"]) == {"WS_X", "WS_Y", "WS_Z"}


@pytest.mark.integration
def test_48_lookup_is_context_local_with_no_global_registry(server, contexts):
    """The id is resolved against the addressed context's own analysis.

    Asserted at the adapter, where the lookup happens: the same id returns
    evidence from one context and nothing from another, with no shared table
    consulted in between.
    """
    adapter = DetectionAdapter()
    reference_id = adapter.get_cycles(contexts["default"])["cycles"][0]["cycle_id"]

    assert adapter.get_cycle_detail(contexts["default"], reference_id) is not None
    assert adapter.get_cycle_detail(contexts["ws1"], reference_id) is None
    assert adapter.get_cycle_detail(contexts["ws_empty"], reference_id) is None


# ---------------------------------------------------------------------------
# Canonical identity, serialization, and model isolation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_49_canonical_id_is_identical_across_index_filter_and_detail(server):
    unfiltered = _get(server).json()["cycles"][0]["cycle_id"]

    for entity in ("REF_A", "REF_B", "REF_C"):
        row = _get(server, "?entity_id=" + entity).json()["cycles"][0]
        assert row["cycle_id"] == unfiltered
        assert _detail(server, row["cycle_id"]).json()["cycle"]["cycle_id"] == unfiltered

    # Paging does not alter identity either.
    assert _get(server, "?limit=1&offset=0").json()["cycles"][0]["cycle_id"] == unfiltered


@pytest.mark.integration
def test_50_both_responses_are_json_native_throughout(server):
    def walk(node, path="$"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert isinstance(key, str), path
                walk(value, path + "." + key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + "[" + str(index) + "]")
        else:
            assert node is None or isinstance(node, (str, int, float, bool)), (
                path + " is " + type(node).__name__
            )

    index = _get(server, "?workspace_id=ws1").json()
    detail = _detail(
        server, index["cycles"][0]["cycle_id"], "?workspace_id=ws1"
    ).json()

    for body in (index, detail):
        walk(body)
        assert json.loads(json.dumps(body)) == body


#: Field names that would mean cycle evidence had acquired a model output or a
#: score of its own.  Checked as *keys*, and as substrings of values, with one
#: enumerated exemption below.
_MODEL_TOKENS = (
    "risk_score", "predicted_class", "cycle_risk",
    "suspicious", "threshold", "probability",
)

#: The only fields allowed to *mention* those words, because their whole job is
#: to say the layer does none of it.  Enumerated so the exemption cannot widen
#: silently: a new prose field would fail the test until it is added here on
#: purpose.
_PROSE_EXEMPT_FIELDS = ("matched_rules_note", "temporal_explanation",
                        "temporal_precision_basis", "message")


@pytest.mark.integration
def test_51_no_model_output_leaks_into_either_response(server):
    """No cycle response may carry a model output or a score of its own.

    Checked structurally — every key, and every value that is not one of the
    enumerated prose fields — rather than by grepping the whole body, so that a
    note whose *content* is a denial ("invents no threshold") cannot be
    mistaken for the thing it denies.
    """
    index = _get(server).json()
    detail = _detail(server, index["cycles"][0]["cycle_id"]).json()

    offences = []

    def walk(node, path="$", key=None):
        if isinstance(node, dict):
            for child_key, value in node.items():
                for token in _MODEL_TOKENS:
                    if token in child_key:
                        offences.append(f"{path}.{child_key} is a {token} field")
                walk(value, f"{path}.{child_key}", child_key)
        elif isinstance(node, list):
            for position, value in enumerate(node):
                walk(value, f"{path}[{position}]", key)
        elif isinstance(node, str) and key not in _PROSE_EXEMPT_FIELDS:
            for token in _MODEL_TOKENS:
                assert token not in node, f"{path} mentions {token!r}"

    for body in (index, detail):
        walk(body)

    assert offences == []

    # The exemption is real prose, and it is a denial rather than a verdict.
    note = detail["cycle"]["matched_rules_note"]
    assert "invents no threshold" in note
    assert detail["cycle"]["matched_rules"] == []


@pytest.mark.integration
def test_52_the_index_and_detail_routes_do_not_recompute_anything(server, monkeypatch):
    import src.detection.cycles as cycles_module

    calls = []
    real = cycles_module.analyze_cycles

    def counted(graph, config=None):
        calls.append(graph)
        return real(graph, config)

    monkeypatch.setattr(cycles_module, "analyze_cycles", counted)

    for _ in range(5):
        index = _get(server).json()
        _detail(server, index["cycles"][0]["cycle_id"])
        _get(server, "?entity_id=REF_B&limit=10")

    assert calls == [], "the API recomputed cycle analysis per request"
