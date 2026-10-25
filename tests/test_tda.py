import pytest
import pandas as pd
import networkx as nx
from src.features.tda import compute_tga

def create_edge(u, v, k, d):
    return (u, v, k, {"date": pd.Timestamp(d)})

def test_tga_cases():
    # Helper to test a single graph topology around 'A'
    def evaluate(edges, target_node='A'):
        G = nx.MultiDiGraph()
        for i, e in enumerate(edges):
            if len(e) == 3:
                u, v, date = e
                k = 0
            else:
                u, v, k, date = e
            G.add_edge(u, v, key=k, date=pd.Timestamp(date))
        
        # Ensure 'A' is in the graph
        if not G.has_node(target_node):
            G.add_node(target_node)
            
        res = compute_tga(G)
        df = res.features
        return df.loc[target_node]

    # A. A -> B -> C
    res = evaluate([
        ('A', 'B', '2023-01-01'),
        ('B', 'C', '2023-01-02')
    ])
    assert res['temporal_reciprocal_count'] == 0
    assert res['temporal_cycle_count'] == 0
    assert res['earliest_cycle_completion_days'] == -1.0

    # B. A -> B -> C -> A
    res = evaluate([
        ('A', 'B', '2023-01-01'),
        ('B', 'C', '2023-01-02'),
        ('C', 'A', '2023-01-03')
    ])
    assert res['temporal_cycle_count'] == 1
    assert res['earliest_cycle_completion_days'] == 2.0 # day 3 - day 1

    # C. A -> B, B -> A
    res = evaluate([
        ('A', 'B', '2023-01-01'),
        ('B', 'A', '2023-01-02')
    ])
    assert res['temporal_reciprocal_count'] == 1
    assert res['temporal_cycle_count'] == 0
    assert res['earliest_cycle_completion_days'] == 1.0

    # D. A -> B, A -> B (parallel)
    res = evaluate([
        ('A', 'B', 0, '2023-01-01'),
        ('A', 'B', 1, '2023-01-02')
    ])
    assert res['temporal_reciprocal_count'] == 0
    assert res['temporal_cycle_count'] == 0

    # E. A -> B, B -> C, A -> C
    res = evaluate([
        ('A', 'B', '2023-01-01'),
        ('B', 'C', '2023-01-02'),
        ('A', 'C', '2023-01-03')
    ])
    assert res['temporal_cycle_count'] == 0

    # F. Causal temporal cycle
    res = evaluate([
        ('A', 'B', '2023-01-01'),
        ('B', 'C', '2023-01-02'),
        ('C', 'A', '2023-01-03')
    ])
    assert res['temporal_cycle_count'] == 1

    # G. Non-causal loop
    res = evaluate([
        ('A', 'B', '2023-01-03'),
        ('B', 'C', '2023-01-02'),
        ('C', 'A', '2023-01-01')
    ])
    assert res['temporal_cycle_count'] == 0

    # H. Same day reciprocal
    res = evaluate([
        ('A', 'B', '2023-01-01'),
        ('B', 'A', '2023-01-01')
    ])
    assert res['temporal_reciprocal_count'] == 1

    # I. Two parallel A->B plus B->A
    res = evaluate([
        ('A', 'B', 0, '2023-01-01'),
        ('A', 'B', 1, '2023-01-02'),
        ('B', 'A', 0, '2023-01-03')
    ])
    # The B->A can complete a cycle with either A->B(0) or A->B(1)
    # Both are time respecting. So 2 reciprocal cycles.
    assert res['temporal_reciprocal_count'] == 2

    # J. Check cycle count doesn't count rotations
    # Handled by B being 1.
