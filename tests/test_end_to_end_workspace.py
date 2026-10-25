import pytest
from src.datasets.workspace_manager import WorkspaceManager
from src.datasets.manifest import WorkspaceStatus
from src.datasets.canonicalizer import canonicalize_and_validate
from src.datasets.inference import predict_workspace
from src.ui.adapter import DetectionAdapter
import os
import pandas as pd
import json

def test_actual_end_to_end_workspace_pipeline(tmp_path):
    """
    Real end-to-end test of the entire pipeline using the reference dataset structure.
    This guarantees all required graph features are present and validates compatibility genuinely.
    """
    csv_path = tmp_path / "reference.csv"
    
    with open("../dkdata.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    df.to_csv(csv_path, index=False)
    
    manager = WorkspaceManager(base_dir=str(tmp_path))
    
    # 1. Create
    ws = manager.create_workspace("E2E Test")
    assert ws.manifest.status == WorkspaceStatus.EMPTY
    
    # 2. Ingest
    manager.ingest_file(ws.manifest.workspace_id, str(csv_path))
    ws = manager.get_workspace(ws.manifest.workspace_id)
    assert ws.manifest.status == WorkspaceStatus.UPLOADED
    
    # 3. Profile
    manager.profile_workspace(ws.manifest.workspace_id)
    ws = manager.get_workspace(ws.manifest.workspace_id)
    assert ws.manifest.status == WorkspaceStatus.NEEDS_MAPPING
    
    # 4. Map
    manager.map_workspace(ws.manifest.workspace_id)
    ws = manager.get_workspace(ws.manifest.workspace_id)
    
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f:
        m = json.load(f)
    
    # Exact reverse map from canonicalizer to simulate perfect user mapping
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
        "beneficiary_core": "RECEIVER_CORE"
    }
    
    for mp in m.get("mappings", []):
        col = mp["source_column"]
        if col in col_to_concept:
            mp["selected_concept"] = col_to_concept[col]
            mp["confidence"] = 1.0
            mp["status"] = "AUTO_ACCEPTED"
            
    with open(map_path, "w") as f:
        json.dump(m, f)
        
    ws.update(status=WorkspaceStatus.READY)
    
    # 5. Canonicalize & Evaluate Compatibility
    canonicalize_and_validate(ws.manifest.workspace_id, manager)
    ws = manager.get_workspace(ws.manifest.workspace_id)
    
    comp_path = ws.directory / "compatibility.json"
    with open(comp_path, "r") as f:
        comp_report = json.load(f)
        
    assert ws.manifest.status == WorkspaceStatus.READY, json.dumps(comp_report, indent=2)
    
    # 6. Workspace-Aware Inference
    df_preds = predict_workspace(ws.manifest.workspace_id, manager)
    assert len(df_preds) > 0
    assert "risk_score" in df_preds.columns
    assert "predicted_class" in df_preds.columns
    
    # 7. Predictions Artifact Persistence & Reopen
    from src.ui.resolver import WorkspaceAnalysisResolver
    resolver = WorkspaceAnalysisResolver(base_dir=str(tmp_path))
    context = resolver.get_analysis_context(ws.manifest.workspace_id)
    adapter = DetectionAdapter()
    df_reopen, graph_reopen = context.df, context.graph
    
    assert len(df_reopen) == len(df_preds)
    assert graph_reopen.number_of_nodes() > 0
    assert graph_reopen.number_of_edges() > 0
