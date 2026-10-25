import pytest
import pandas as pd
import torch
from src.datasets.mapper import CanonicalConcept, MappingManifest, ColumnMapping, MappingStatus
from src.datasets.canonicalizer import canonicalize_and_validate
from src.datasets.workspace_manager import WorkspaceManager
from src.features.baseline import compute_baseline

def test_feature_contract(tmp_path):
    ckpt = torch.load('artifacts/models/experiment3_graphsage.pt', map_location='cpu')
    expected_features = ckpt['metadata']['feature_names']
    
    import networkx as nx
    
    G = nx.MultiDiGraph()
    G.add_node("EntityA", entity_type="Company")
    G.add_node("EntityB", entity_type="WeirdType") # Unknown type
    G.add_node("EntityC", entity_type=None)
    
    edge_table = pd.DataFrame({
        "source_entity": ["EntityA", "EntityB", "EntityC"],
        "target_entity": ["EntityB", "EntityC", "EntityA"],
        "amount_usd": [100.0, 200.0, 300.0],
        "payer_bank_country": ["GB", "NEW_COUNTRY", None],
        "beneficiary_bank_country": ["DE", "GB", "ZZZ"]
    })
    
    baseline = compute_baseline(G, edge_table)
    
    assert baseline.feature_names == expected_features[:160]
    assert baseline.features.shape[1] == 160
    
    assert "bank_country_NEW_COUNTRY" not in baseline.feature_names
    assert "bank_country_ZZZ" not in baseline.feature_names
    
    gb_idx = baseline.feature_names.index("bank_country_GB")
    de_idx = baseline.feature_names.index("bank_country_DE")
    fr_idx = baseline.feature_names.index("bank_country_FR")
    
    # EntityA has GB as payer country and ZZZ as beneficiary country (from C -> A). Wait, C -> A has beneficiary ZZZ.
    # EntityA is source for A->B (GB). Target for C->A (ZZZ).
    # So EntityA is associated with GB and ZZZ.
    # GB should be 1, ZZZ should be ignored. FR should be 0.
    
    # EntityA is index 0
    assert baseline.features.iloc[0, gb_idx] == 1.0
    assert baseline.features.iloc[0, fr_idx] == 0.0
    
    assert "entity_type_Company" in baseline.feature_names
    assert "entity_type_WeirdType" not in baseline.feature_names
    assert "entity_type_unknown" in baseline.feature_names
    
    unknown_idx = baseline.feature_names.index("entity_type_unknown")
    assert baseline.features.iloc[1, unknown_idx] == 1.0
    assert baseline.features.iloc[2, unknown_idx] == 1.0

