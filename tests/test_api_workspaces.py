import pytest
import json
from src.ui.workspaces import get_workspaces_summary
from src.datasets.workspace_manager import WorkspaceManager
from src.datasets.manifest import WorkspaceStatus
import pandas as pd

def test_get_workspaces_summary(tmp_path, monkeypatch):
    manager = WorkspaceManager(base_dir=str(tmp_path))
    monkeypatch.setattr("src.ui.workspaces.WorkspaceManager", lambda: manager)

    # 1. Empty workspace
    ws1 = manager.create_workspace("Test WS 1")

    # 2. Uploaded workspace
    ws2 = manager.create_workspace("Test WS 2")
    ws2.update(status=WorkspaceStatus.UPLOADED)
    from src.datasets.profiler import DatasetProfile
    prof = DatasetProfile(row_estimate=123, is_exact_count=True, columns=[])
    with open(ws2.directory / "profile.json", "w") as f:
        json.dump(prof.to_dict(), f)

    # 3. READY workspace
    ws3 = manager.create_workspace("Test WS 3")
    ws3.update(status=WorkspaceStatus.READY)
    with open(ws3.directory / "profile.json", "w") as f:
        json.dump(DatasetProfile(row_estimate=456, is_exact_count=True, columns=[]).to_dict(), f)

    summary = get_workspaces_summary()
    assert len(summary) == 3

    # Sort or find
    ws1_summary = next(s for s in summary if s["name"] == "Test WS 1")
    ws2_summary = next(s for s in summary if s["name"] == "Test WS 2")
    ws3_summary = next(s for s in summary if s["name"] == "Test WS 3")

    assert ws1_summary["records"] == 0 # Or None depending on implementation
    assert ws2_summary["records"] == 123
    assert ws3_summary["records"] == 456

    # The get_workspaces_summary returns prediction_status implicitly if it uses NOT_RUN
    # we just verify it exists
    assert "status" in ws1_summary
    assert "status" in ws2_summary
    assert "status" in ws3_summary

import urllib.request
import urllib.parse
import threading
import time
from pathlib import Path
from src.ui.app import run_server
from src.ui.config import UIConfig

def test_dataset_hub_upload_api(tmp_path, monkeypatch):
    from src.ui.adapter import DetectionAdapter

    class MockAdapter:
        def initialize(self): pass
    monkeypatch.setattr("src.ui.app.adapter", MockAdapter())

    manager = WorkspaceManager(base_dir=str(tmp_path))
    monkeypatch.setattr("src.ui.app.WorkspaceManager", lambda: manager)

    import http.server
    import socketserver
    from src.ui.app import APIHandler

    server = socketserver.TCPServer((UIConfig.HOST, 0), APIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    time.sleep(0.5)

    try:
        # Test 10, 11: empty upload is rejected
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://{UIConfig.HOST}:{port}/api/workspaces/upload", method="POST"))
            assert False, "Should have failed with 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400

        # Unsupported extension rejected
        url = f"http://{UIConfig.HOST}:{port}/api/workspaces/upload?name=Test&filename=test.txt"
        req = urllib.request.Request(url, data=b"dummy", method="POST")
        try:
            urllib.request.urlopen(req)
            assert False, "Should have failed with 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400

        import tempfile
        import os

        # Test 1, 2, 3, 4, 5, 6, 7, 8, 12: Test all formats
        for fmt, fixture in [("CSV", "data.csv"), ("XLSX", "data.xlsx"), ("XLS", "data.xls")]:
            fixture_path = Path(f"tests/fixtures/{fixture}")
            with open(fixture_path, "rb") as f:
                file_data = f.read()

            url = f"http://{UIConfig.HOST}:{port}/api/workspaces/upload?name={fmt}_Workspace&filename={fixture}"
            url = url.replace(' ', '%20')
            req = urllib.request.Request(url, data=file_data, method="POST")
            req.add_header('Content-Length', str(len(file_data)))

            # Before request, count temp files to verify cleanup
            temp_dir = tempfile.gettempdir()
            files_before = set(os.listdir(temp_dir))

            response = urllib.request.urlopen(req)
            assert response.status == 200
            result = json.loads(response.read().decode('utf-8'))
            assert result["status"] == "success"
            ws_id = result["workspace_id"]

            files_after = set(os.listdir(temp_dir))
            # Test 12: temporary file is cleaned up afterward
            # Assert no new .csv, .xlsx, .xls file was left behind
            new_files = files_after - files_before
            for nf in new_files:
                assert not nf.endswith(fixture_path.suffix), f"Temp file {nf} leaked!"

            ws = manager.get_workspace(ws_id)
            # Test 6: original_filename is correct
            assert ws.manifest.original_filename == fixture
            # Test 8: successful upload does not remain ANALYZING
            assert ws.manifest.status.value == "NEEDS_MAPPING"
            # Test 5: Workspace source file is persisted
            assert ws.manifest.source_file.endswith(fixture_path.suffix)
            assert (ws.directory / ws.manifest.source_file).exists()
            # Test 7: profile.json is created
            assert (ws.directory / "profile.json").exists()

        # Test 9: failed upload results in FAILED
        # Uploading a multi-sheet excel file causes safe_ingest to fail
        multi_path = Path("tests/fixtures/multi_data.xlsx")
        with open(multi_path, "rb") as f:
            multi_data = f.read()

        url = f"http://{UIConfig.HOST}:{port}/api/workspaces/upload?name=Multi_Workspace&filename=multi.xlsx"
        req = urllib.request.Request(url, data=multi_data, method="POST")
        req.add_header('Content-Length', str(len(multi_data)))

        try:
            urllib.request.urlopen(req)
            assert False, "Should have failed with 500"
        except urllib.error.HTTPError as e:
            assert e.code == 500

        # Verify it went to FAILED
        workspaces = manager.list_workspaces()
        multi_ws = [w for w in workspaces if w.manifest.display_name == "Multi_Workspace"][0]
        assert multi_ws.manifest.status.value == "FAILED"

        # Test 10, 13: Multiple uploads isolation
        ws_a_url = f"http://{UIConfig.HOST}:{port}/api/workspaces/upload?name=IBM_AML_A&filename=data.xlsx"
        ws_b_url = f"http://{UIConfig.HOST}:{port}/api/workspaces/upload?name=IBM_AML_B&filename=data.xlsx"

        req_a = urllib.request.Request(ws_a_url, data=file_data, method="POST")
        req_a.add_header('Content-Length', str(len(file_data)))
        res_a = json.loads(urllib.request.urlopen(req_a).read().decode('utf-8'))
        ws_id_a = res_a["workspace_id"]

        req_b = urllib.request.Request(ws_b_url, data=file_data, method="POST")
        req_b.add_header('Content-Length', str(len(file_data)))
        res_b = json.loads(urllib.request.urlopen(req_b).read().decode('utf-8'))
        ws_id_b = res_b["workspace_id"]

        ws_a = manager.get_workspace(ws_id_a)
        ws_b = manager.get_workspace(ws_id_b)

        # Test 13: A's raw file belongs to A, B's raw file belongs to B
        assert str(ws_a.directory) in str((ws_a.directory / ws_a.manifest.source_file).resolve())
        assert str(ws_b.directory) in str((ws_b.directory / ws_b.manifest.source_file).resolve())
        assert ws_a.directory != ws_b.directory

        # And ensure GraphSAGE hasn't been triggered. (Since it's not in the code path).

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

def test_upload_with_datetime_and_numpy_types(tmp_path, monkeypatch):
    import numpy as np
    from src.ui.config import UIConfig
    import urllib.request
    import urllib.parse
    import threading
    import time
    from pathlib import Path

    class MockAdapter:
        def initialize(self): pass
    monkeypatch.setattr("src.ui.app.adapter", MockAdapter())

    manager = WorkspaceManager(base_dir=str(tmp_path))
    monkeypatch.setattr("src.ui.app.WorkspaceManager", lambda: manager)

    import http.server
    import socketserver
    from src.ui.app import APIHandler

    server = socketserver.TCPServer((UIConfig.HOST, 0), APIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    time.sleep(0.5)

    try:
        df = pd.DataFrame({
            "date_col": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "int_col": [np.int64(1), np.int64(2)],
            "float_col": [np.float64(1.5), np.nan],
            "inf_col": [np.inf, np.float64(3.14)],
            "bool_col": [np.bool_(True), pd.NA]
        })
        test_csv_path = tmp_path / "test_types.csv"
        df.to_csv(test_csv_path, index=False)

        with open(test_csv_path, "rb") as f:
            file_data = f.read()

        url = f"http://{UIConfig.HOST}:{port}/api/workspaces/upload?name=Test_Types&filename=test_types.csv"
        req = urllib.request.Request(url, data=file_data, method="POST")
        req.add_header('Content-Length', str(len(file_data)))

        response = urllib.request.urlopen(req)
        assert response.status == 200
        result = json.loads(response.read().decode('utf-8'))
        assert result["status"] == "success"

        ws_id = result["workspace_id"]
        ws = manager.get_workspace(ws_id)

        assert ws.manifest.status.value == "NEEDS_MAPPING"

        profile_path = ws.directory / "profile.json"
        assert profile_path.exists()

        with open(profile_path, "r", encoding="utf-8") as pf:
            profile_data = json.load(pf)

        assert len(profile_data["columns"]) == 5
        date_col = next(c for c in profile_data["columns"] if c["original_name"] == "date_col")
        assert date_col["inferred_type"] == "datetime"
        assert date_col["samples"][0].startswith("2023-01-01")

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

def test_restart_reconciles_workspace(tmp_path):
    manager = WorkspaceManager(base_dir=str(tmp_path))

    # 1. Valid profile.json with status ANALYZING should reconcile to NEEDS_MAPPING
    ws1 = manager.create_workspace("Test Reconcile Valid")
    ws1.update(status=WorkspaceStatus.ANALYZING)
    # create valid profile.json
    prof1 = {"row_estimate": 1000, "columns": []}
    with open(ws1.directory / "profile.json", "w") as f:
        json.dump(prof1, f)

    # Restart simulation: Reload workspace
    ws1_reloaded = manager.get_workspace(ws1.manifest.workspace_id)
    assert ws1_reloaded.manifest.status.value == "NEEDS_MAPPING"
    assert ws1_reloaded.manifest.profiler_version == "1.0"

    # 2. Missing profile.json with status ANALYZING should NOT reconcile
    ws2 = manager.create_workspace("Test Reconcile Missing")
    ws2.update(status=WorkspaceStatus.ANALYZING)

    # Restart simulation
    ws2_reloaded = manager.get_workspace(ws2.manifest.workspace_id)
    assert ws2_reloaded.manifest.status.value == "ANALYZING"

    # 3. Invalid profile.json with status ANALYZING should NOT reconcile
    ws3 = manager.create_workspace("Test Reconcile Invalid")
    ws3.update(status=WorkspaceStatus.ANALYZING)
    with open(ws3.directory / "profile.json", "w") as f:
        f.write("{invalid json")

    # Restart simulation
    ws3_reloaded = manager.get_workspace(ws3.manifest.workspace_id)
    assert ws3_reloaded.manifest.status.value == "ANALYZING"

    # 4. A genuine failed workspace remains FAILED
    ws4 = manager.create_workspace("Test Failed")
    ws4.update(status=WorkspaceStatus.FAILED)

    # Restart simulation
    ws4_reloaded = manager.get_workspace(ws4.manifest.workspace_id)
    assert ws4_reloaded.manifest.status.value == "FAILED"
