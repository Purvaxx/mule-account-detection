import json
import urllib.request
import urllib.parse
import threading
import time
from pathlib import Path
from src.datasets.workspace_manager import WorkspaceManager
from src.datasets.manifest import WorkspaceStatus
from src.ui.app import run_server, APIHandler
from src.ui.config import UIConfig

def test_semantic_mapping_review_flow(tmp_path, monkeypatch):
    class MockAdapter:
        def initialize(self): pass
    monkeypatch.setattr("src.ui.app.adapter", MockAdapter())
    
    manager = WorkspaceManager(base_dir=str(tmp_path))
    monkeypatch.setattr("src.ui.app.WorkspaceManager", lambda: manager)
    
    import socketserver
    server = socketserver.TCPServer((UIConfig.HOST, 0), APIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    time.sleep(0.5)
    
    try:
        # Create a workspace and ingest data
        ws = manager.create_workspace("Test Mapping")
        csv_path = Path("tests/fixtures/data.csv")
        ws = manager.ingest_file(ws.manifest.workspace_id, str(csv_path))
        manager.profile_workspace(ws.manifest.workspace_id)
        
        ws_id = ws.manifest.workspace_id
        
        # 1. Fetch mapping (should generate mapping.json initially)
        url = f"http://{UIConfig.HOST}:{port}/api/workspaces/{ws_id}/mapping"
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        assert response.status == 200
        
        data = json.loads(response.read().decode('utf-8'))
        assert "mapping" in data
        assert "manifest" in data
        assert "profile" in data
        
        mappings = data["mapping"]["mappings"]
        assert len(mappings) > 0
        
        # 2. Existing mapper candidates are displayed & confidence values match
        tx_amount_m = next((m for m in mappings if m["source_column"] == "amount_orig"), None)
        assert tx_amount_m is not None
        # The mapper should have candidates for 'Amount'
        assert len(tx_amount_m["candidates"]) > 0
        assert "evidence" in tx_amount_m["candidates"][0]
        
        # 3. Confirming a mapping persists it
        # We will post an update to change a status
        update_url = f"http://{UIConfig.HOST}:{port}/api/workspaces/{ws_id}/mapping"
        payload = json.dumps({
            "source_column": "amount_orig",
            "selected_concept": "TRANSACTION_AMOUNT",
            "status": "AUTO_ACCEPTED"
        }).encode('utf-8')
        
        req = urllib.request.Request(update_url, data=payload, method="POST")
        req.add_header('Content-Length', str(len(payload)))
        response = urllib.request.urlopen(req)
        assert response.status == 200
        
        # Verify persistence
        persisted = manager.get_mapping(ws_id)
        persisted_m = next((m for m in persisted["mappings"] if m["source_column"] == "amount_orig"), None)
        assert persisted_m["status"] == "AUTO_ACCEPTED"
        
        # 4. Leaving unmapped persists correctly
        payload = json.dumps({
            "source_column": "date",
            "selected_concept": None,
            "status": "UNMAPPED"
        }).encode('utf-8')
        req = urllib.request.Request(update_url, data=payload, method="POST")
        req.add_header('Content-Length', str(len(payload)))
        urllib.request.urlopen(req)
        
        persisted = manager.get_mapping(ws_id)
        persisted_m = next((m for m in persisted["mappings"] if m["source_column"] == "date"), None)
        assert persisted_m["status"] == "UNMAPPED"
        assert persisted_m["selected_concept"] is None
        
        # 5. Validation is triggered only after explicit user action
        val_url = f"http://{UIConfig.HOST}:{port}/api/workspaces/{ws_id}/validate"
        req = urllib.request.Request(val_url, method="POST")
        response = urllib.request.urlopen(req)
        assert response.status == 200
        val_data = json.loads(response.read().decode('utf-8'))
        
        assert "manifest_status" in val_data
        assert "compatibility" in val_data
        
        # Since we only mapped one field, it's definitely INCOMPATIBLE
        assert val_data["manifest_status"] == "INCOMPATIBLE"
        assert len(val_data["compatibility"]["missing_required_fields"]) > 0
        
        # Verify manifest is updated
        ws = manager.get_workspace(ws_id)
        assert ws.manifest.status == WorkspaceStatus.INCOMPATIBLE
        
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Route matching
# ---------------------------------------------------------------------------
# Routes are matched on the whole, exactly-sized segment list.  The previous
# startswith/endswith ladder matched any path merely *ending* in the right
# word, so an extra segment silently changed which workspace was addressed.

import pytest


def test_route_matching_captures_the_workspace_id():
    handler, params, _ = APIHandler.match_route(
        "GET", "/api/workspaces/abc-123/config"
    )
    assert handler == "get_workspace_config"
    assert params == {"workspace_id": "abc-123"}


def test_route_matching_rejects_an_extra_path_segment():
    """Regression: `/api/workspaces/<a>/<b>/config` matched the config route
    and read `<a>` as the workspace id."""
    handler, params, _ = APIHandler.match_route(
        "GET", "/api/workspaces/abc-123/extra/config"
    )
    assert handler is None


def test_route_matching_is_method_specific():
    assert APIHandler.match_route("GET", "/api/workspaces/x/predict")[0] is None
    assert APIHandler.match_route("POST", "/api/workspaces/x/predict")[0] == \
        "post_workspace_predict"


def test_upload_route_is_not_shadowed_by_the_workspace_id_routes():
    handler, params, _ = APIHandler.match_route("POST", "/api/workspaces/upload")
    assert handler == "post_upload"
    assert params == {}


def test_entity_id_segment_is_url_decoded():
    _, params, _ = APIHandler.match_route("GET", "/api/entity/HILUX%20SERVICES%20LP")
    assert params["entity_id"] == "HILUX SERVICES LP"


# ---------------------------------------------------------------------------
# Mapping overrides are owned by WorkspaceManager
# ---------------------------------------------------------------------------

def _mapped_workspace(tmp_path):
    manager = WorkspaceManager(base_dir=str(tmp_path))
    ws = manager.create_workspace("Override Test")
    ws = manager.ingest_file(ws.manifest.workspace_id, str(Path("tests/fixtures/data.csv")))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    return manager, ws.manifest.workspace_id


def test_mapping_override_persists_atomically(tmp_path):
    manager, ws_id = _mapped_workspace(tmp_path)
    column = manager.get_mapping(ws_id)["mappings"][0]["source_column"]

    manager.apply_mapping_overrides(ws_id, [{
        "source_column": column,
        "selected_concept": "TRANSACTION_AMOUNT",
        "status": "AUTO_ACCEPTED",
    }])

    ws_dir = manager.get_workspace(ws_id).directory
    reloaded = manager.get_mapping(ws_id)
    applied = [m for m in reloaded["mappings"] if m["source_column"] == column][0]
    assert applied["selected_concept"] == "TRANSACTION_AMOUNT"
    assert applied["status"] == "AUTO_ACCEPTED"
    # The atomic write leaves no partial file behind.
    assert not (ws_dir / "mapping.json.tmp").exists()


def test_mapping_override_rejects_an_unknown_column(tmp_path):
    """Silently ignoring an unknown column reports success for an override
    that never happened."""
    manager, ws_id = _mapped_workspace(tmp_path)
    with pytest.raises(ValueError, match="Unknown source column"):
        manager.apply_mapping_overrides(
            ws_id, [{"source_column": "no_such_column", "status": "AUTO_ACCEPTED"}]
        )


def test_mapping_override_requires_an_existing_mapping(tmp_path):
    manager = WorkspaceManager(base_dir=str(tmp_path))
    ws = manager.create_workspace("Unmapped")
    with pytest.raises(ValueError, match="no mapping to override"):
        manager.apply_mapping_overrides(
            ws.manifest.workspace_id, [{"source_column": "x"}]
        )
