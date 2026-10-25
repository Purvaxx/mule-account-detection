import pytest
import numpy as np
import torch
import pandas as pd
import networkx as nx

from src.features.baseline import BaselineFeatures
from src.features.temporal import TemporalFeatures
from src.models.dataset import build_pyg_data

def test_experiment_2_preprocessing_policy():
    # Mock graph
    graph = nx.MultiDiGraph()
    graph.add_node("A", core=True)
    graph.add_node("B", core=False)
    graph.add_node("C", core=False)
    graph.add_edge("A", "B", key=0)
    
    # Mock baseline
    base_df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])
    baseline = BaselineFeatures(
        features=base_df, 
        entity_count=3, 
        feature_names=["feat1"], 
        nan_count=0, 
        inf_count=0, 
        country_codes=[], 
        entity_type_categories=[]
    )
    
    # Mock temporal matching the 6 features
    temp_df = pd.DataFrame({
        "activity_7d": [0.0, 10.0, np.nan],
        "activity_30d": [5.0, 100.0, np.nan],
        "mean_iat": [0.0, 2.5, np.nan],
        "burstiness": [0.0, 5.0, np.nan],
        "active_day_ratio": [0.0, 0.5, np.nan],
        "same_day_cooccurrence": [0.0, 1.0, np.nan],
    }, index=["A", "B", "C"])
    
    temporal = TemporalFeatures(
        features=temp_df, 
        entity_count=3, 
        feature_names=["activity_7d", "activity_30d", "mean_iat", "burstiness", "active_day_ratio", "same_day_cooccurrence"], 
        as_of=None, 
        nan_count=3, 
        inf_count=0
    )
    
    pyg = build_pyg_data(graph, baseline, temporal=temporal)
    
    assert pyg.num_nodes == 3
    assert pyg.feature_dim == 7  # 1 baseline + 6 temporal
    assert pyg.data.x.shape == (3, 7)
    
    # No NaNs or infs
    assert not torch.isnan(pyg.data.x).any().item()
    assert not torch.isinf(pyg.data.x).any().item()
    
    # Check Node A (Zeros)
    a_idx = 0
    # log1p(0.0) = 0.0
    assert np.isclose(pyg.data.x[a_idx, 1].item(), 0.0)  # activity_7d
    assert np.isclose(pyg.data.x[a_idx, 2].item(), np.log1p(5.0))  # activity_30d
    assert np.isclose(pyg.data.x[a_idx, 3].item(), 0.0)  # mean_iat
    assert np.isclose(pyg.data.x[a_idx, 4].item(), 0.0)  # burstiness
    
    # Check Node B (Valid numbers)
    b_idx = 1
    assert np.isclose(pyg.data.x[b_idx, 1].item(), np.log1p(10.0))  # activity_7d
    assert np.isclose(pyg.data.x[b_idx, 2].item(), np.log1p(100.0)) # activity_30d
    assert np.isclose(pyg.data.x[b_idx, 3].item(), np.log1p(2.5))   # mean_iat
    assert np.isclose(pyg.data.x[b_idx, 4].item(), 5.0)             # burstiness (unscaled)
    assert np.isclose(pyg.data.x[b_idx, 5].item(), 0.5)             # active_day_ratio (unscaled)
    assert np.isclose(pyg.data.x[b_idx, 6].item(), 1.0)             # same_day_cooc (unscaled)
    
    # Check Node C (NaNs) -> should be -1.0 for all temporal
    c_idx = 2
    for i in range(1, 7):
        assert np.isclose(pyg.data.x[c_idx, i].item(), -1.0)
