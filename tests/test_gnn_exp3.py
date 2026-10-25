import pytest
import numpy as np
import torch
import pandas as pd
import networkx as nx

from src.features.baseline import BaselineFeatures
from src.features.multigraph import MultigraphFeatures
from src.models.dataset import build_pyg_data

def test_experiment_3_preprocessing_policy():
    # Mock graph
    graph = nx.MultiDiGraph()
    graph.add_node("A", core=True)
    graph.add_node("B", core=False)
    graph.add_node("C", core=False)
    graph.add_edge("A", "B", key=0)
    
    # Mock baseline (160 features is too big to mock fully, just mock a small one to check concatenation)
    # But the test just cares about dimension additions.
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
    
    pyg = build_pyg_data(graph, baseline, temporal=None, multigraph=multigraph)
    
    assert pyg.num_nodes == 3
    assert pyg.feature_dim == 8  # 1 baseline + 7 multigraph
    assert pyg.data.x.shape == (3, 8)
    
    # No NaNs or infs
    assert not torch.isnan(pyg.data.x).any().item()
    assert not torch.isinf(pyg.data.x).any().item()
    
    # Check Node A (1.0 or 0.0 values)
    a_idx = 0
    assert np.isclose(pyg.data.x[a_idx, 1].item(), np.log1p(1.0))  # out_repeated_counterparty_ratio
    assert np.isclose(pyg.data.x[a_idx, 2].item(), np.log1p(1.0))  # in_repeated_counterparty_ratio
    assert np.isclose(pyg.data.x[a_idx, 3].item(), np.log1p(1.0))  # max_parallel_edge_count
    assert np.isclose(pyg.data.x[a_idx, 4].item(), np.log1p(1.0))  # mean_parallel_edge_count
    assert np.isclose(pyg.data.x[a_idx, 5].item(), np.log1p(0.0))  # parallel_temporal_dispersion
    assert np.isclose(pyg.data.x[a_idx, 6].item(), np.log1p(0.0))  # reciprocal_out_count
    assert np.isclose(pyg.data.x[a_idx, 7].item(), np.log1p(0.0))  # reciprocal_in_count
    
    # Check Node B (Valid larger numbers)
    b_idx = 1
    assert np.isclose(pyg.data.x[b_idx, 1].item(), np.log1p(10.0))
    assert np.isclose(pyg.data.x[b_idx, 2].item(), np.log1p(5.0))
    assert np.isclose(pyg.data.x[b_idx, 3].item(), np.log1p(100.0))
    assert np.isclose(pyg.data.x[b_idx, 4].item(), np.log1p(50.0))
    assert np.isclose(pyg.data.x[b_idx, 5].item(), np.log1p(25.0))
    assert np.isclose(pyg.data.x[b_idx, 6].item(), np.log1p(2.0))
    assert np.isclose(pyg.data.x[b_idx, 7].item(), np.log1p(2.0))
    
    # Check Node C (NaNs) -> should be -1.0 for all multigraph
    c_idx = 2
    for i in range(1, 8):
        assert np.isclose(pyg.data.x[c_idx, i].item(), -1.0)

