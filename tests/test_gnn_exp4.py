import pytest
import numpy as np
import torch
import pandas as pd
import networkx as nx

from src.features.baseline import BaselineFeatures
from src.features.temporal import TemporalFeatures
from src.features.multigraph import MultigraphFeatures
from src.models.dataset import build_pyg_data

def test_experiment_4_preprocessing_policy():
    # Mock graph
    graph = nx.MultiDiGraph()
    graph.add_node("A", core=True)
    graph.add_node("B", core=False)
    graph.add_node("C", core=False)
    graph.add_edge("A", "B", key=0)
    
    # Mock baseline (1 feature for test)
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
        "activity_7d": [10.0, np.nan, np.nan],
        "activity_30d": [30.0, np.nan, np.nan],
        "mean_iat": [5.0, np.nan, np.nan],
        "burstiness": [1.2, np.nan, np.nan],
        "active_day_ratio": [0.5, np.nan, np.nan],
        "same_day_cooccurrence": [0.1, np.nan, np.nan],
    }, index=["A", "B", "C"])
    temporal = TemporalFeatures(
        features=temp_df,
        entity_count=3,
        feature_names=["activity_7d", "activity_30d", "mean_iat", "burstiness", "active_day_ratio", "same_day_cooccurrence"],
        nan_count=10,
        inf_count=0,
        as_of=None
    )

    # Mock multigraph matching the 7 features
    mg_df = pd.DataFrame({
        "out_repeated_counterparty_ratio": [1.0, 10.0, np.nan],
        "in_repeated_counterparty_ratio": [1.0, 5.0, np.nan],
        "max_parallel_edge_count": [1.0, 100.0, np.nan],
        "mean_parallel_edge_count": [1.0, 50.0, np.nan],
        "parallel_temporal_dispersion": [0.0, 25.0, np.nan],
        "reciprocal_out_count": [0.0, 2.0, np.nan],
        "reciprocal_in_count": [0.0, 2.0, np.nan],
    }, index=["A", "B", "C"])
    
    multigraph = MultigraphFeatures(
        features=mg_df, 
        entity_count=3, 
        feature_names=["out_repeated_counterparty_ratio", "in_repeated_counterparty_ratio", 
                       "max_parallel_edge_count", "mean_parallel_edge_count", 
                       "parallel_temporal_dispersion", "reciprocal_out_count", "reciprocal_in_count"], 
        nan_count=7, 
        inf_count=0
    )
    
    pyg = build_pyg_data(graph, baseline, temporal=temporal, multigraph=multigraph)
    
    assert pyg.num_nodes == 3
    assert pyg.feature_dim == 1 + 6 + 7  # 1 baseline + 6 temporal + 7 multigraph = 14 total
    assert pyg.data.x.shape == (3, 14)
    
    # No NaNs or infs
    assert not torch.isnan(pyg.data.x).any().item()
    assert not torch.isinf(pyg.data.x).any().item()
    
    # Check Node C (NaNs) -> should be -1.0 for all temporal and multigraph features
    c_idx = 2
    for i in range(1, 14):
        assert np.isclose(pyg.data.x[c_idx, i].item(), -1.0)
