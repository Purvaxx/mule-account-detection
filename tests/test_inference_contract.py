"""
Inference contract vs. training/evaluation contract
====================================================

These tests pin the architectural decision that

    inference-required fields  !=  training/evaluation/label metadata

Specifically: the absence of ``payer_core`` / ``beneficiary_core`` /
``investigation`` must NOT prevent an unlabelled dataset from reaching READY
when every genuine inference dependency is satisfied.

They also cover the end-to-end workspace path:
    READY -> Run Analysis -> predict_workspace() -> persisted predictions
and the isolation and ground-truth-separation guarantees around it.

All fixtures here are tiny and deterministic; the expensive Reference Dataset
is exercised separately by tests/test_reference_regression.py.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import networkx as nx
import pandas as pd
import pytest
import torch

from src.data.ingestion import (
    IDENTITY_REQUIRED_COLUMNS,
    INFERENCE_DEGRADABLE_COLUMNS,
    INFERENCE_REQUIRED_BASE,
    LABEL_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    REQUIRED_COLUMNS,
    inference_required_columns,
)
from src.datasets.canonicalizer import canonicalize_and_validate
from src.datasets.inference import predict_workspace
from src.datasets.manifest import CurrencyPolicy, IdentitySource, WorkspaceStatus
from src.datasets.workspace_manager import WorkspaceManager
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph

CHECKPOINT = "artifacts/models/experiment3_graphsage.pt"


# ---------------------------------------------------------------------------
# Fixtures — tiny, deterministic, no labels unless explicitly stated
# ---------------------------------------------------------------------------

def _unlabelled_frame(n=12, prefix="A", accounts=4):
    """An account-keyed dataset with NO names and NO ground truth at all."""
    return pd.DataFrame({
        "txn_time": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "from_acct": [f"{prefix}{i % accounts}" for i in range(n)],
        "to_acct": [f"{prefix}{(i + 1) % accounts}" for i in range(n)],
        "usd_amount": [100.0 + i for i in range(n)],
    })


_UNLABELLED_MAP = {
    "txn_time": "TIMESTAMP",
    "from_acct": "SENDER_ACCOUNT",
    "to_acct": "RECEIVER_ACCOUNT",
    "usd_amount": "USD_AMOUNT",
}


def _apply_mapping(ws, wanted):
    path = ws.directory / "mapping.json"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    for entry in m["mappings"]:
        if entry["source_column"] in wanted:
            entry["selected_concept"] = wanted[entry["source_column"]]
            entry["status"] = "AUTO_ACCEPTED"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f)


def _build_workspace(manager, tmp_path, frame, wanted, name="ws",
                     identity=IdentitySource.ACCOUNT_ID,
                     currency=CurrencyPolicy.NATIVE_USD):
    csv = tmp_path / f"{name}.csv"
    frame.to_csv(csv, index=False)

    ws = manager.create_workspace(name)
    wid = ws.manifest.workspace_id
    manager.ingest_file(wid, str(csv))
    manager.profile_workspace(wid)
    manager.map_workspace(wid)

    ws = manager.get_workspace(wid)
    ws.manifest.dataset_config.identity_source = identity
    ws.manifest.dataset_config.currency_policy = currency
    ws.update()

    _apply_mapping(ws, wanted)
    return wid


@pytest.fixture
def manager(tmp_path):
    return WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))


# ---------------------------------------------------------------------------
# 1. The contract itself
# ---------------------------------------------------------------------------

def test_contract_partitions_the_canonical_schema():
    """Every canonical column belongs to exactly one requirement class."""
    union = (
        INFERENCE_REQUIRED_BASE
        | INFERENCE_DEGRADABLE_COLUMNS
        | LABEL_COLUMNS
        | OPTIONAL_METADATA_COLUMNS
        | IDENTITY_REQUIRED_COLUMNS["NAME"]
        | IDENTITY_REQUIRED_COLUMNS["ACCOUNT_ID"]
    )
    assert union == set(REQUIRED_COLUMNS)

    # The classes must not overlap.
    classes = [
        INFERENCE_REQUIRED_BASE,
        INFERENCE_DEGRADABLE_COLUMNS,
        LABEL_COLUMNS,
        OPTIONAL_METADATA_COLUMNS,
    ]
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            assert not (a & b), f"overlapping requirement classes: {a & b}"


def test_labels_are_never_inference_required():
    """payer_core / beneficiary_core / investigation are label-only."""
    assert LABEL_COLUMNS == {"payer_core", "beneficiary_core", "investigation"}
    for source in (IdentitySource.NAME, IdentitySource.ACCOUNT_ID):
        required = inference_required_columns(source)
        assert not (required & LABEL_COLUMNS), (
            f"{source} wrongly requires label columns: {required & LABEL_COLUMNS}"
        )


def test_identity_strategy_selects_its_own_identifier_columns():
    name = inference_required_columns(IdentitySource.NAME)
    acct = inference_required_columns(IdentitySource.ACCOUNT_ID)
    assert {"payer_name_norm", "beneficiary_name_norm"} <= name
    assert not ({"payer_account", "beneficiary_account"} & name)
    assert {"payer_account", "beneficiary_account"} <= acct
    assert not ({"payer_name_norm", "beneficiary_name_norm"} & acct)


def test_unknown_identity_source_is_rejected():
    with pytest.raises(ValueError):
        inference_required_columns("SOMETHING_ELSE")


# ---------------------------------------------------------------------------
# 2. Exact 167-feature contract from the minimal field set
# ---------------------------------------------------------------------------

def test_167_features_from_minimal_fields_only():
    """The full 167-vector is computable with no labels and no metadata.

    This is the empirical basis for the contract split: only identity, date and
    amount_usd are structurally required.  Bank country and entity type degrade
    to a zeroed block / unknown rather than failing.
    """
    G = nx.MultiDiGraph()
    G.add_node("A")          # deliberately NO entity_type attribute
    G.add_node("B")
    G.add_edge("A", "B", key=0)
    G.add_edge("B", "A", key=0)

    edge_table = pd.DataFrame({
        "source_entity": ["A", "B"],
        "target_entity": ["B", "A"],
        "amount_usd": [100.0, 50.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
    })

    baseline = compute_baseline(G, edge_table)
    multi = compute_multigraph(G, edge_table)

    generated = list(baseline.feature_names) + list(multi.feature_names)
    expected = torch.load(CHECKPOINT, map_location="cpu")["metadata"]["feature_names"]

    assert len(generated) == 167
    assert generated == expected, "feature order must match the frozen checkpoint exactly"

    # Documented degradation, not silent breakage.
    country_cols = [c for c in baseline.features.columns if c.startswith("bank_country_")]
    assert (baseline.features[country_cols].values == 0).all()
    assert (baseline.features["entity_type_unknown"] == 1).all()


# ---------------------------------------------------------------------------
# 3. Unlabelled dataset reaches READY
# ---------------------------------------------------------------------------

def test_unlabelled_dataset_reaches_ready(manager, tmp_path):
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY", (
        f"unlabelled dataset blocked: missing={report.missing_required_fields} "
        f"graph={report.graph_requirements} dtype={report.datatype_errors}"
    )
    assert report.missing_required_fields == []
    assert report.feature_dimension == 167
    assert report.feature_order_valid is True
    assert report.labels_present == []


def test_label_absence_does_not_appear_as_missing_required(manager, tmp_path):
    """The specific regression: labels must not gate READY."""
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    report = canonicalize_and_validate(wid, manager)

    for label in LABEL_COLUMNS:
        assert label not in report.missing_required_fields

    by_name = {f["canonical_name"]: f for f in report.field_statuses}
    for label in LABEL_COLUMNS:
        assert by_name[label]["requirement"] == "LABEL_ONLY"
        assert by_name[label]["state"] == "NOT_SUPPLIED"


def test_absent_optional_signal_is_reported_but_not_blocking(manager, tmp_path):
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert set(report.degraded_features) == set(INFERENCE_DEGRADABLE_COLUMNS)
    assert any("reduced signal" in w for w in report.value_quality_warnings)
    assert any("no ground-truth labels" in w for w in report.value_quality_warnings)


def test_out_of_vocabulary_country_is_warned_not_silent(manager, tmp_path):
    """Mapped-but-unrecognised country codes must be reported, not silently zeroed."""
    frame = _unlabelled_frame()
    # Country *names*, not the ISO codes in the frozen vocabulary.
    frame["from_country"] = ["Wakanda"] * len(frame)
    frame["to_country"] = ["Atlantis"] * len(frame)
    wanted = dict(_UNLABELLED_MAP,
                  from_country="SENDER_BANK_COUNTRY",
                  to_country="RECEIVER_BANK_COUNTRY")

    wid = _build_workspace(manager, tmp_path, frame, wanted, name="oov")
    report = canonicalize_and_validate(wid, manager)

    # Structurally fine: the vector is still exactly 167 in the right order.
    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid is True
    # But the operator is told the signal is not reaching the model.
    assert any("country vocabulary" in w for w in report.value_quality_warnings)


def test_genuinely_required_field_still_blocks(manager, tmp_path):
    """Removing amount_usd (structurally required) must still be INCOMPATIBLE."""
    frame = _unlabelled_frame().drop(columns=["usd_amount"])
    wanted = {k: v for k, v in _UNLABELLED_MAP.items() if k != "usd_amount"}
    wid = _build_workspace(manager, tmp_path, frame, wanted, name="no_amount")
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "INCOMPATIBLE"
    assert "amount_usd" in report.missing_required_fields


def test_labels_preserved_when_present(manager, tmp_path):
    """Supplied ground truth is recorded, never discarded or fabricated."""
    frame = _unlabelled_frame()
    frame["is_fraud"] = [1, 0] * (len(frame) // 2)
    wanted = dict(_UNLABELLED_MAP, is_fraud="INVESTIGATION")

    wid = _build_workspace(manager, tmp_path, frame, wanted, name="labelled")
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert "investigation" in report.labels_present


# ---------------------------------------------------------------------------
# 4. Prediction: persistence, restart, ground-truth separation
# ---------------------------------------------------------------------------

def test_prediction_persists_to_disk(manager, tmp_path):
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    assert canonicalize_and_validate(wid, manager).status == "READY"

    df = predict_workspace(wid, manager)
    ws = manager.get_workspace(wid)

    assert (ws.directory / "predictions.parquet").exists()
    assert ws.manifest.prediction_status == "COMPLETE"
    assert ws.manifest.prediction_timestamp is not None
    assert ws.manifest.model_version is not None
    assert len(df) > 0
    assert {"entity_id", "risk_score", "predicted_class"} <= set(df.columns)


def test_predictions_survive_restart_without_rerunning(manager, tmp_path, monkeypatch):
    """Artifacts survive a fresh process; inference is NOT re-run on load."""
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    canonicalize_and_validate(wid, manager)
    original = predict_workspace(wid, manager)

    # Simulate a restart: brand-new manager reading the same store.
    fresh = WorkspaceManager(base_dir=str(manager.base_dir))
    ws = fresh.get_workspace(wid)

    assert ws.manifest.prediction_status == "COMPLETE"
    reloaded = pd.read_parquet(ws.directory / "predictions.parquet")
    # check_dtype=False: entity_id is a pandas StringDtype in memory and plain
    # object after a parquet round-trip.  The values must match exactly; the
    # storage dtype is not part of the persistence contract.
    pd.testing.assert_frame_equal(
        original.reset_index(drop=True), reloaded.reset_index(drop=True),
        check_dtype=False,
    )

    # Nothing on the read path may invoke the model.
    import src.detection.inference as det

    def _boom(*a, **k):
        raise AssertionError("inference must not re-run on restart/load")

    monkeypatch.setattr(det, "predict_from_prepared_data", _boom)
    again = WorkspaceManager(base_dir=str(manager.base_dir)).get_workspace(wid)
    assert again.manifest.prediction_status == "COMPLETE"
    assert pd.read_parquet(again.directory / "predictions.parquet").shape == reloaded.shape


def test_ground_truth_is_separate_from_model_prediction(manager, tmp_path):
    """Dataset truth and model output never overwrite one another."""
    frame = _unlabelled_frame()
    # Deliberately adversarial: truth that disagrees with any plausible output.
    frame["is_fraud"] = [1] * len(frame)
    wanted = dict(_UNLABELLED_MAP, is_fraud="INVESTIGATION")

    wid = _build_workspace(manager, tmp_path, frame, wanted, name="truthsep")
    report = canonicalize_and_validate(wid, manager)
    assert report.status == "READY"
    assert "investigation" in report.labels_present

    preds = predict_workspace(wid, manager)

    # The prediction artifact carries model output only.
    assert "predicted_class" in preds.columns
    assert "risk_score" in preds.columns
    for label in ("investigation", "payer_core", "beneficiary_core", "is_fraud"):
        assert label not in preds.columns, (
            f"ground-truth column {label!r} leaked into the prediction artifact"
        )

    # And the source truth is still intact in the canonical dataset.
    canonical = pd.read_parquet(manager.get_workspace(wid).directory / "canonical.parquet")
    assert canonical["investigation"].notna().all()
    assert (canonical["investigation"] == 1).all()


# ---------------------------------------------------------------------------
# 5. Workspace isolation
# ---------------------------------------------------------------------------

def test_two_external_workspaces_do_not_leak(manager, tmp_path):
    """Distinct datasets keep distinct graphs, entities and risk scores."""
    wid_a = _build_workspace(
        manager, tmp_path, _unlabelled_frame(n=12, prefix="A", accounts=4),
        _UNLABELLED_MAP, name="alpha")
    wid_b = _build_workspace(
        manager, tmp_path, _unlabelled_frame(n=20, prefix="B", accounts=5),
        _UNLABELLED_MAP, name="beta")

    assert canonicalize_and_validate(wid_a, manager).status == "READY"
    assert canonicalize_and_validate(wid_b, manager).status == "READY"

    preds_a = predict_workspace(wid_a, manager)
    preds_b = predict_workspace(wid_b, manager)

    ents_a = set(preds_a["entity_id"])
    ents_b = set(preds_b["entity_id"])
    assert ents_a == {"A0", "A1", "A2", "A3"}
    assert ents_b == {"B0", "B1", "B2", "B3", "B4"}
    assert not (ents_a & ents_b), "entity namespaces leaked between workspaces"

    # Separate artifacts on disk, neither overwritten by the other.
    ws_a = manager.get_workspace(wid_a)
    ws_b = manager.get_workspace(wid_b)
    assert ws_a.directory != ws_b.directory
    assert len(pd.read_parquet(ws_a.directory / "predictions.parquet")) == 4
    assert len(pd.read_parquet(ws_b.directory / "predictions.parquet")) == 5


def test_resolver_isolates_reference_from_workspaces(manager, tmp_path, monkeypatch):
    """The Reference context and workspace contexts never share state."""
    from src.ui.resolver import WorkspaceAnalysisResolver

    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    canonicalize_and_validate(wid, manager)
    predict_workspace(wid, manager)

    resolver = WorkspaceAnalysisResolver(base_dir=str(manager.base_dir))
    monkeypatch.setattr(
        "src.ui.resolver.WorkspaceManager", lambda *a, **k: manager
    )

    # Inject a sentinel reference context so this test stays cheap.
    sentinel_df = pd.DataFrame({
        "entity_id": ["REF_ONLY"], "risk_score": [0.99], "predicted_class": [1],
    })
    sentinel_graph = nx.MultiDiGraph()
    sentinel_graph.add_node("REF_ONLY")
    from src.ui.context import AnalysisContext
    resolver.reference_context = AnalysisContext(
        workspace_id=None, df=sentinel_df, graph=sentinel_graph
    )

    ref = resolver.get_analysis_context(None)
    ws_ctx = resolver.get_analysis_context(wid)

    assert ref.workspace_id is None
    assert ws_ctx.workspace_id == wid
    assert set(ref.df["entity_id"]) == {"REF_ONLY"}
    assert "REF_ONLY" not in set(ws_ctx.df["entity_id"])
    assert ref.graph is not ws_ctx.graph

    # The reference context must be unchanged after resolving a workspace.
    assert set(resolver.get_analysis_context(None).df["entity_id"]) == {"REF_ONLY"}


def test_invalidate_clears_only_the_named_workspace(manager, tmp_path):
    from src.ui.resolver import WorkspaceAnalysisResolver

    resolver = WorkspaceAnalysisResolver(base_dir=str(manager.base_dir))
    resolver.cache["a"] = "ctx-a"
    resolver.cache["b"] = "ctx-b"

    resolver.invalidate("a")
    assert "a" not in resolver.cache
    assert resolver.cache["b"] == "ctx-b"

    resolver.invalidate()
    assert resolver.cache == {}


# ---------------------------------------------------------------------------
# 6. Prediction API
# ---------------------------------------------------------------------------

@pytest.fixture
def api_server(manager, monkeypatch):
    """Serve the real APIHandler against an isolated workspace store."""
    import socketserver

    from src.ui.app import APIHandler
    from src.ui.config import UIConfig

    monkeypatch.setattr("src.ui.app.WorkspaceManager", lambda *a, **k: manager)

    server = socketserver.TCPServer((UIConfig.HOST, 0), APIHandler)
    port = server.server_address[1]
    # poll_interval bounds how long shutdown() blocks (default 0.5 s).
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    time.sleep(0.3)
    try:
        yield f"http://{UIConfig.HOST}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _post(url):
    req = urllib.request.Request(url, data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_predict_api_runs_and_persists(api_server, manager, tmp_path):
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    assert canonicalize_and_validate(wid, manager).status == "READY"

    status, body = _post(f"{api_server}/api/workspaces/{wid}/predict")

    assert status == 200
    assert body["status"] == "success"
    assert body["workspace_id"] == wid
    assert body["prediction_status"] == "COMPLETE"
    assert body["entity_count"] == 4

    ws = manager.get_workspace(wid)
    assert (ws.directory / "predictions.parquet").exists()
    assert ws.manifest.prediction_status == "COMPLETE"


def test_predict_api_rejects_non_ready_workspace(api_server, manager, tmp_path):
    """A workspace that has not passed compatibility must be refused."""
    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP,
                           name="notready")
    # Deliberately skip canonicalize_and_validate: no compatibility.json.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{api_server}/api/workspaces/{wid}/predict")
    assert exc.value.code == 409

    ws = manager.get_workspace(wid)
    assert not (ws.directory / "predictions.parquet").exists()


def test_predict_api_rejects_incompatible_workspace(api_server, manager, tmp_path):
    frame = _unlabelled_frame().drop(columns=["usd_amount"])
    wanted = {k: v for k, v in _UNLABELLED_MAP.items() if k != "usd_amount"}
    wid = _build_workspace(manager, tmp_path, frame, wanted, name="incompat")
    assert canonicalize_and_validate(wid, manager).status == "INCOMPATIBLE"

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{api_server}/api/workspaces/{wid}/predict")
    assert exc.value.code == 409


def test_workspace_summary_distinguishes_ready_from_complete(manager, tmp_path, monkeypatch):
    """READY must not imply that predictions already exist."""
    from src.ui.workspaces import get_workspaces_summary

    monkeypatch.setattr("src.ui.workspaces.WorkspaceManager", lambda: manager)

    wid = _build_workspace(manager, tmp_path, _unlabelled_frame(), _UNLABELLED_MAP)
    canonicalize_and_validate(wid, manager)

    row = next(r for r in get_workspaces_summary() if r["id"] == wid)
    assert row["compatibility"] == "READY"
    assert row["can_predict"] is True
    assert row["has_predictions"] is False
    assert row["prediction_status"] == "NOT_RUN"

    predict_workspace(wid, manager)

    row = next(r for r in get_workspaces_summary() if r["id"] == wid)
    assert row["has_predictions"] is True
    assert row["prediction_status"] == "COMPLETE"
