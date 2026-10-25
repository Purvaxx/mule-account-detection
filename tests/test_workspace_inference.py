import atexit
import shutil
import tempfile
import uuid

import pytest
import pandas as pd
import json
from pathlib import Path

from src.datasets.workspace_manager import WorkspaceManager
from src.datasets.canonicalizer import canonicalize_and_validate
from src.datasets.inference import predict_workspace, prepare_workspace_detection_data
from src.detection.inference import run_inference

@pytest.fixture
def manager(tmp_path):
    return WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))

@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    import os
    p = "tests/test_data/sample.csv" if os.path.exists("tests/test_data/sample.csv") else "../../tests/test_data/sample.csv"
    pd.read_csv(p).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# READY workspace template
# ---------------------------------------------------------------------------
# ``_build_ready_workspace`` drives the real path — ingest, profile, map,
# canonicalize + validate — over the full 16,954-row sample.  Validation alone
# builds the graph and the 167-feature matrix, so a single call costs ~12 s and
# the tests below need ten of them.
#
# A workspace is a self-contained directory whose only reference to its own
# identity is the ``workspace_id`` field in ``manifest.json``.  So the template
# is built once per session and each test gets a byte-for-byte copy under a
# fresh UUID inside its own ``tmp_path`` store: every test still owns a private
# workspace on disk, and every artifact in it is one the production code wrote.

_TEMPLATE_ROOT = None
_TEMPLATE_DIR = None


def _ready_workspace_template() -> Path:
    """The directory of a canonicalized, validated READY workspace."""
    global _TEMPLATE_ROOT, _TEMPLATE_DIR
    if _TEMPLATE_DIR is None:
        import os
        _TEMPLATE_ROOT = Path(tempfile.mkdtemp(prefix="ready_ws_template_"))
        atexit.register(shutil.rmtree, str(_TEMPLATE_ROOT), True)

        source = ("tests/test_data/sample.csv"
                  if os.path.exists("tests/test_data/sample.csv")
                  else "../../tests/test_data/sample.csv")
        csv_path = _TEMPLATE_ROOT / "sample.csv"
        pd.read_csv(source).to_csv(csv_path, index=False)

        template_manager = WorkspaceManager(
            base_dir=str(_TEMPLATE_ROOT / ".workspaces"))
        _TEMPLATE_DIR = _build_ready_workspace(
            template_manager, csv_path).directory
    return _TEMPLATE_DIR


def create_ready_workspace(manager, sample_csv):
    """A READY workspace of the sample dataset, private to ``manager``."""
    template = _ready_workspace_template()

    workspace_id = str(uuid.uuid4())
    destination = manager.base_dir / workspace_id
    shutil.copytree(template, destination)

    manifest_path = destination / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["workspace_id"] = workspace_id
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manager.get_workspace(workspace_id)


def _build_ready_workspace(manager, sample_csv):
    ws = manager.create_workspace("Ready WS")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)

    manager.map_workspace(ws.manifest.workspace_id)
    
    # FORCE MAP EXACT MATCHES for testing Stage 5!
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f:
        m = json.load(f)
    
    col_to_concept = {
        "id": "TRANSACTION_ID",
        "date": "TIMESTAMP",
        "payer_name": "SENDER_NAME",
        "beneficiary_name": "RECEIVER_NAME",
        "payer_account": "SENDER_ACCOUNT",
        "beneficiary_account": "RECEIVER_ACCOUNT",
        "amount_orig": "TRANSACTION_AMOUNT",
        "amount_orig_currency": "ORIGINAL_CURRENCY",
        "amount_usd": "USD_AMOUNT",
        "amount_eur": "EUR_AMOUNT",
        "payer_type": "SENDER_TYPE",
        "beneficiary_type": "RECEIVER_TYPE",
        "payer_jurisdiction": "SENDER_JURISDICTION",
        "beneficiary_jurisdiction": "RECEIVER_JURISDICTION",
        "payer_bank_country": "SENDER_BANK_COUNTRY",
        "beneficiary_bank_country": "RECEIVER_BANK_COUNTRY",
        "purpose": "PURPOSE",
        "source_file": "SOURCE_FILE",
        "investigation": "INVESTIGATION",
        "payer_core": "SENDER_CORE",
        "beneficiary_core": "RECEIVER_CORE",
        "payer_name_norm": "SENDER_NAME",
        "beneficiary_name_norm": "RECEIVER_NAME"
    }
    
    for mapping in m["mappings"]:
        sc = mapping["source_column"]
        if sc in col_to_concept:
            mapping["selected_concept"] = col_to_concept[sc]
            mapping["status"] = "AUTO_ACCEPTED"
            
    with open(map_path, "w") as f:
        json.dump(m, f)
        
    canonicalize_and_validate(ws.manifest.workspace_id, manager)

    return ws

def test_ready_workspace_can_invoke_inference(manager, sample_csv):
    ws = create_ready_workspace(manager, sample_csv)
    df = predict_workspace(ws.manifest.workspace_id, manager)
    assert not df.empty
    assert (ws.directory / "predictions.parquet").exists()
    ws = manager.get_workspace(ws.manifest.workspace_id)
    assert ws.manifest.prediction_status == "COMPLETE"
    assert ws.manifest.model_version is not None

def test_empty_workspace_cannot_invoke_inference(manager):
    ws = manager.create_workspace("Empty WS")
    with pytest.raises(ValueError, match="compatibility not evaluated"):
        predict_workspace(ws.manifest.workspace_id, manager)

def test_needs_mapping_cannot_invoke_inference(manager, sample_csv):
    ws = manager.create_workspace("Needs Mapping")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    with pytest.raises(ValueError, match="compatibility not evaluated"):
        predict_workspace(ws.manifest.workspace_id, manager)

def test_incompatible_cannot_invoke_inference(manager, tmp_path):
    ws = manager.create_workspace("Incompat WS")
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({
        "payer_name": ["A"],
        "beneficiary_name": ["B"],
        "id": [1]
        # missing date and amount
    }).to_csv(bad_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(bad_csv))
    manager.profile_workspace(ws.manifest.workspace_id)

    manager.map_workspace(ws.manifest.workspace_id)
    
    # FORCE MAP EXACT MATCHES for testing Stage 5!
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f:
        m = json.load(f)
    
    col_to_concept = {
        "id": "TRANSACTION_ID",
        "date": "TIMESTAMP",
        "payer_name": "SENDER_NAME",
        "beneficiary_name": "RECEIVER_NAME",
        "payer_account": "SENDER_ACCOUNT",
        "beneficiary_account": "RECEIVER_ACCOUNT",
        "amount_orig": "TRANSACTION_AMOUNT",
        "amount_orig_currency": "ORIGINAL_CURRENCY",
        "amount_usd": "USD_AMOUNT",
        "amount_eur": "EUR_AMOUNT",
        "payer_type": "SENDER_TYPE",
        "beneficiary_type": "RECEIVER_TYPE",
        "payer_jurisdiction": "SENDER_JURISDICTION",
        "beneficiary_jurisdiction": "RECEIVER_JURISDICTION",
        "payer_bank_country": "SENDER_BANK_COUNTRY",
        "beneficiary_bank_country": "RECEIVER_BANK_COUNTRY",
        "purpose": "PURPOSE",
        "source_file": "SOURCE_FILE",
        "investigation": "INVESTIGATION",
        "payer_core": "SENDER_CORE",
        "beneficiary_core": "RECEIVER_CORE",
        "payer_name_norm": "SENDER_NAME",
        "beneficiary_name_norm": "RECEIVER_NAME"
    }
    
    for mapping in m["mappings"]:
        sc = mapping["source_column"]
        if sc in col_to_concept:
            mapping["selected_concept"] = col_to_concept[sc]
            mapping["status"] = "AUTO_ACCEPTED"
            
    with open(map_path, "w") as f:
        json.dump(m, f)
        
    canonicalize_and_validate(ws.manifest.workspace_id, manager)

    with pytest.raises(ValueError, match="Cannot run inference on workspace with status: INCOMPATIBLE"):
        predict_workspace(ws.manifest.workspace_id, manager)

def test_failed_cannot_invoke_inference(manager, tmp_path):
    ws = manager.create_workspace("Failed WS")
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"id": []}).to_csv(bad_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(bad_csv))
    manager.profile_workspace(ws.manifest.workspace_id)

    manager.map_workspace(ws.manifest.workspace_id)
    
    # FORCE MAP EXACT MATCHES for testing Stage 5!
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f:
        m = json.load(f)
    
    col_to_concept = {
        "id": "TRANSACTION_ID",
        "date": "TIMESTAMP",
        "payer_name": "SENDER_NAME",
        "beneficiary_name": "RECEIVER_NAME",
        "payer_account": "SENDER_ACCOUNT",
        "beneficiary_account": "RECEIVER_ACCOUNT",
        "amount_orig": "TRANSACTION_AMOUNT",
        "amount_orig_currency": "ORIGINAL_CURRENCY",
        "amount_usd": "USD_AMOUNT",
        "amount_eur": "EUR_AMOUNT",
        "payer_type": "SENDER_TYPE",
        "beneficiary_type": "RECEIVER_TYPE",
        "payer_jurisdiction": "SENDER_JURISDICTION",
        "beneficiary_jurisdiction": "RECEIVER_JURISDICTION",
        "payer_bank_country": "SENDER_BANK_COUNTRY",
        "beneficiary_bank_country": "RECEIVER_BANK_COUNTRY",
        "purpose": "PURPOSE",
        "source_file": "SOURCE_FILE",
        "investigation": "INVESTIGATION",
        "payer_core": "SENDER_CORE",
        "beneficiary_core": "RECEIVER_CORE",
        "payer_name_norm": "SENDER_NAME",
        "beneficiary_name_norm": "RECEIVER_NAME"
    }
    
    for mapping in m["mappings"]:
        sc = mapping["source_column"]
        if sc in col_to_concept:
            mapping["selected_concept"] = col_to_concept[sc]
            mapping["status"] = "AUTO_ACCEPTED"
            
    with open(map_path, "w") as f:
        json.dump(m, f)
        
    canonicalize_and_validate(ws.manifest.workspace_id, manager)

    with pytest.raises(ValueError, match="Cannot run inference on workspace with status: FAILED"):
        predict_workspace(ws.manifest.workspace_id, manager)

def test_prediction_schema_matches(manager, sample_csv):
    ws = create_ready_workspace(manager, sample_csv)
    df = predict_workspace(ws.manifest.workspace_id, manager)
    expected_cols = ["entity_id", "risk_score", "predicted_class", "in_degree", "out_degree", 
                     "total_incoming_usd", "total_outgoing_usd", "flow_imbalance", 
                     "in_repeated_ratio", "out_repeated_ratio", "max_parallel_edge", 
                     "mean_parallel_edge", "reciprocal_in", "reciprocal_out"]
    for c in expected_cols:
        assert c in df.columns

def test_detection_state_contract(manager, sample_csv):
    ws = create_ready_workspace(manager, sample_csv)
    from src.datasets.inference import prepare_workspace_detection_data
    from src.detection.inference import DetectionState
    
    state = prepare_workspace_detection_data(ws.directory)
    assert isinstance(state, DetectionState)
    
    import networkx as nx
    from src.features.baseline import BaselineFeatures
    from src.features.multigraph import MultigraphFeatures
    from src.models.dataset import PyGData

    assert isinstance(state.graph, nx.MultiDiGraph)
    assert isinstance(state.baseline, BaselineFeatures)
    assert isinstance(state.multi, MultigraphFeatures)
    assert isinstance(state.pyg, PyGData)
    
    # Check fields in PyG exactly
    assert hasattr(state.pyg.data, "x")
    assert hasattr(state.pyg.data, "edge_index")
    assert state.pyg.feature_dim == 167
    assert state.graph.number_of_nodes() == state.pyg.num_nodes
    assert state.graph.number_of_edges() == state.pyg.num_edges
    
    # Contract is valid if predict_from_prepared_data works
    from src.detection.inference import predict_from_prepared_data
    from src.ui.config import UIConfig
    df = predict_from_prepared_data(state, UIConfig.CHECKPOINT_PATH)
    ws = create_ready_workspace(manager, sample_csv)
    predict_workspace(ws.manifest.workspace_id, manager)
    
    ws_reopened = manager.get_workspace(ws.manifest.workspace_id)
    assert ws_reopened.manifest.prediction_status == "COMPLETE"
    assert (ws_reopened.directory / "predictions.parquet").exists()

def test_missing_checkpoint_handled(manager, sample_csv):
    ws = create_ready_workspace(manager, sample_csv)
    with pytest.raises(FileNotFoundError):
        predict_workspace(ws.manifest.workspace_id, manager, checkpoint_path="missing.pt")

def test_no_ml_behavior_outside_existing_path(manager, sample_csv, monkeypatch):
    ws = create_ready_workspace(manager, sample_csv)

    invocations = []
    
    def fake_prepare(*args, **kwargs):
        invocations.append("prepare")
        raise RuntimeError("Stop before heavy compute")
        
    monkeypatch.setattr("src.datasets.inference.prepare_workspace_detection_data", fake_prepare)
    
    with pytest.raises(RuntimeError, match="Stop before heavy compute"):
        predict_workspace(ws.manifest.workspace_id, manager)
        
    assert "prepare" in invocations


def test_workspace_isolation(manager, sample_csv, tmp_path):
    ws1 = create_ready_workspace(manager, sample_csv)
    ws2 = create_ready_workspace(manager, sample_csv)
    
    predict_workspace(ws1.manifest.workspace_id, manager)
    assert (ws1.directory / "predictions.parquet").exists()
    assert not (ws2.directory / "predictions.parquet").exists()
    
    import pandas as pd
    df1 = pd.read_parquet(ws1.directory / "predictions.parquet")
    predict_workspace(ws2.manifest.workspace_id, manager)
    df2 = pd.read_parquet(ws2.directory / "predictions.parquet")
    
    assert ws1.manifest.workspace_id != ws2.manifest.workspace_id
    assert len(df1) > 0
    assert len(df2) > 0
    # Also verify metadata change doesn't leak
    manager.update_workspace(ws1.manifest.workspace_id, name="Renamed")
    ws2_reloaded = manager.get_workspace(ws2.manifest.workspace_id)
    assert ws2_reloaded.manifest.display_name != "Renamed"




def test_analyzing_cannot_invoke_inference(manager):
    ws = manager.create_workspace("Analyzing WS")
    from src.datasets.manifest import WorkspaceStatus
    ws.update(status=WorkspaceStatus.ANALYZING)
    import json
    with open(ws.directory / "compatibility.json", "w") as f: json.dump({"status": "ANALYZING"}, f)
    import pytest
    from src.datasets.inference import predict_workspace
    with pytest.raises(ValueError, match="Cannot run inference on workspace with status: ANALYZING"):
        predict_workspace(ws.manifest.workspace_id, manager)

def test_partial_cannot_invoke_inference(manager):
    ws = manager.create_workspace("Partial WS")
    from src.datasets.manifest import WorkspaceStatus
    ws.update(status=WorkspaceStatus.PARTIAL)
    import json
    with open(ws.directory / "compatibility.json", "w") as f: json.dump({"status": "PARTIAL"}, f)
    import pytest
    from src.datasets.inference import predict_workspace
    with pytest.raises(ValueError, match="Cannot run inference on workspace with status: PARTIAL"):
        predict_workspace(ws.manifest.workspace_id, manager)

def test_adapter_caching_and_isolation(manager, sample_csv, monkeypatch):
    ws1 = create_ready_workspace(manager, sample_csv)
    ws2 = create_ready_workspace(manager, sample_csv)
    predict_workspace(ws1.manifest.workspace_id, manager)
    predict_workspace(ws2.manifest.workspace_id, manager)
    
    from src.ui.adapter import DetectionAdapter
    from src.ui.resolver import WorkspaceAnalysisResolver
    adapter = DetectionAdapter()
    resolver = WorkspaceAnalysisResolver(base_dir=manager.base_dir)
    
    # Spy on prepare_workspace_detection_data
    call_counts = {}
    import src.ui.resolver
    import src.datasets.inference
    original_prepare = src.datasets.inference.prepare_workspace_detection_data
    def mock_prepare(workspace_dir, *args, **kwargs):
        # Forward transparently: the real function also takes an
        # ``identity_source`` argument, which the resolver supplies from the
        # workspace manifest.
        call_counts[str(workspace_dir)] = call_counts.get(str(workspace_dir), 0) + 1
        return original_prepare(workspace_dir, *args, **kwargs)
    monkeypatch.setattr(src.ui.resolver, "prepare_workspace_detection_data", mock_prepare)
    
    # Mock WorkspaceManager inside adapter
    import src.datasets.workspace_manager
    monkeypatch.setattr(src.ui.resolver, "WorkspaceManager", lambda *args, **kwargs: manager)
    
    # A
    ctx1 = resolver.get_analysis_context(ws1.manifest.workspace_id)
    df1, g1 = ctx1.df, ctx1.graph
    assert call_counts[str(ws1.directory)] == 1
    
    # B
    ctx2 = resolver.get_analysis_context(ws2.manifest.workspace_id)
    df2, g2 = ctx2.df, ctx2.graph
    assert call_counts[str(ws2.directory)] == 1
    
    # A again
    ctx1_again = resolver.get_analysis_context(ws1.manifest.workspace_id)
    assert call_counts[str(ws1.directory)] == 1 # Still 1!
    
    assert ctx1_again is ctx1
    assert ctx1_again.df.equals(ctx1.df)
    assert ctx1_again.graph is ctx1.graph

    # Context B remains distinct from A
    assert ctx2 is not ctx1
    assert ctx2.df is not ctx1.df
    assert ctx2.graph is not ctx1.graph
