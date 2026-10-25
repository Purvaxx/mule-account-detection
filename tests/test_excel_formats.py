import pytest
import pandas as pd
from pathlib import Path
from src.datasets.workspace_manager import WorkspaceManager
import json

def test_formats_supported(tmp_path):
    manager = WorkspaceManager(base_dir=str(tmp_path))
    
    csv_path = Path("tests/fixtures/data.csv")
    xlsx_path = Path("tests/fixtures/data.xlsx")
    xls_path = Path("tests/fixtures/data.xls")
    
    # We will track the semantic profiles to ensure parity
    profiles = {}
    
    for path in [csv_path, xlsx_path, xls_path]:
        ws = manager.create_workspace(f"WS_{path.suffix}")
        
        # 1. Ingest
        manager.ingest_file(ws.manifest.workspace_id, str(path))
        
        # 2. Profile
        manager.profile_workspace(ws.manifest.workspace_id)
        
        ws = manager.get_workspace(ws.manifest.workspace_id)
        assert ws.manifest.status.value == "NEEDS_MAPPING"
        
        with open(ws.directory / "profile.json", "r") as f:
            prof = json.load(f)
            
        profiles[path.suffix] = prof
        
        # 3. Map
        manager.map_workspace(ws.manifest.workspace_id)
        with open(ws.directory / "mapping.json", "r") as f:
            m = json.load(f)
            assert len(m["mappings"]) > 5
            
    # Verify parity across formats
    csv_prof = profiles[".csv"]
    xlsx_prof = profiles[".xlsx"]
    xls_prof = profiles[".xls"]
    
    assert csv_prof["row_estimate"] == xlsx_prof["row_estimate"] == xls_prof["row_estimate"] == 2
    
    # Assert column parity (names and approximate inferred types)
    csv_cols = {c["original_name"]: c["inferred_type"] for c in csv_prof["columns"]}
    xlsx_cols = {c["original_name"]: c["inferred_type"] for c in xlsx_prof["columns"]}
    xls_cols = {c["original_name"]: c["inferred_type"] for c in xls_prof["columns"]}
    
    assert csv_cols.keys() == xlsx_cols.keys() == xls_cols.keys()
    assert "amount_orig" in csv_cols
    assert "type" in csv_cols
    
    # Values sanity check
    assert csv_cols["id"] in ["numeric", "integer", "float"]
    assert xlsx_cols["id"] in ["numeric", "integer", "float"]
    assert xls_cols["id"] in ["numeric", "integer", "float"]

def test_multisheet_rejected(tmp_path):
    multi_xlsx_path = Path("tests/fixtures/multi_data.xlsx")
    manager = WorkspaceManager(base_dir=str(tmp_path))
    ws = manager.create_workspace("WS_MULTI")
    
    with pytest.raises(ValueError, match="Multi-sheet Excel files are not supported"):
        manager.ingest_file(ws.manifest.workspace_id, str(multi_xlsx_path))
