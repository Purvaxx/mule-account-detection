import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List

@dataclass
class TGAFeatures:
    features: pd.DataFrame
    feature_names: List[str]

def compute_tga(graph: nx.MultiDiGraph, edge_table: pd.DataFrame = None) -> TGAFeatures:
    """
    Computes Temporal Graph Analysis (TGA) features for each node.
    Extracts 1-hop directed multigraph ego-networks and finds time-respecting directed cycles.
    """
    nodes = list(graph.nodes())
    
    reciprocal_counts = []
    cycle_counts = []
    earliest_completions = []
    
    for v in nodes:
        # Extract 1-hop ego network
        neighbors = set(graph.predecessors(v)) | set(graph.successors(v)) | {v}
        ego = graph.subgraph(neighbors)
        
        dates = [d['date'] for _, _, d in ego.edges(data=True) if 'date' in d]
        min_date = min(dates) if dates else pd.Timestamp.max
        
        # Build G_simple
        G_simple = nx.DiGraph()
        G_simple.add_nodes_from(ego.nodes())
        for u, w in ego.edges():
            if not G_simple.has_edge(u, w):
                G_simple.add_edge(u, w)
                
        # Find vertex cycles involving v
        v_cycles = []
        vc_calls = [0]
        def dfs_vc(curr, path, visited):
            if len(v_cycles) >= 100:
                return
            vc_calls[0] += 1
            if vc_calls[0] > 10000: # hard limit to prevent explosion
                return
            if len(path) > 6: # cap length to prevent combinatorial explosion on dense graphs
                return
            for w in G_simple.successors(curr):
                if w == v:
                    v_cycles.append(path + [w])
                elif w not in visited:
                    dfs_vc(w, path + [w], visited | {w})
        dfs_vc(v, [v], {v})
        
        rec_count = 0
        long_count = 0
        min_days = -1.0
        
        for vc in v_cycles:
            # Extract parallel edge bundles for this cycle
            k = len(vc) - 1
            L_list = []
            for i in range(k):
                u, w = vc[i], vc[i+1]
                # list of (date, id)
                bundle = []
                for key, d in ego[u][w].items():
                    if 'date' in d:
                        bundle.append((d['date'], u, w, str(key)))
                L_list.append(bundle)
                
            # DP to count temporal cycles
            total_cycles = 0
            
            for start_idx in range(k):
                shifted = L_list[start_idx:] + L_list[:start_idx]
                for e_start in shifted[0]:
                    dp = {e_start[0]: 1}
                    for j in range(1, k):
                        next_dp = {}
                        
                        valid_counts = {}
                        for e_next in shifted[j]:
                            if e_next > e_start:
                                valid_counts[e_next[0]] = valid_counts.get(e_next[0], 0) + 1
                                
                        for prev_date, count in dp.items():
                            for next_date, mult in valid_counts.items():
                                if next_date >= prev_date:
                                    next_dp[next_date] = next_dp.get(next_date, 0) + count * mult
                        dp = next_dp
                        if not dp:
                            break
                    
                    for comp_date, count in dp.items():
                        total_cycles += count
                        days = (comp_date - min_date).days
                        if min_days == -1.0 or days < min_days:
                            min_days = float(days)
                            
            if k == 2:
                rec_count += total_cycles
            elif k >= 3:
                long_count += total_cycles
                
        reciprocal_counts.append(rec_count)
        cycle_counts.append(long_count)
        earliest_completions.append(min_days)
        
    df = pd.DataFrame({
        "temporal_reciprocal_count": reciprocal_counts,
        "temporal_cycle_count": cycle_counts,
        "earliest_cycle_completion_days": earliest_completions
    }, index=nodes)
    
    return TGAFeatures(
        features=df,
        feature_names=["temporal_reciprocal_count", "temporal_cycle_count", "earliest_cycle_completion_days"]
    )
