import pandas as pd
import numpy as np

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.temporal import compute_temporal
from src.features.multigraph import compute_multigraph
from src.models.dataset import build_pyg_data

def audit_exp4():
    ing_res = load_transactions("../dkdata.json")
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    graph = graph_res.graph
    edge_table = graph_res.edge_table
    
    baseline = compute_baseline(graph, edge_table)
    temporal = compute_temporal(edge_table, list(graph.nodes()))
    multigraph = compute_multigraph(graph, edge_table)
    
    # Run the exact same processing dataset.py uses for Exp 4
    pyg_4 = build_pyg_data(graph, baseline, temporal=temporal, multigraph=multigraph)
    
    core_nodes = [n for n, attr in graph.nodes(data=True) if attr.get("core")]
    
    # Verify exact feature dimensions
    print("\n" + "="*50)
    print("FEATURE MATRIX INTEGRITY")
    print("="*50)
    print(f"Nodes: {pyg_4.num_nodes} | Edges: {pyg_4.num_edges}")
    print(f"Total Feature Dim: {pyg_4.feature_dim}")
    print(f"Expected: 160 (baseline) + 6 (temporal) + 7 (multigraph) = 173")
    
    # -------------------------------------------------------------------
    # AUDIT 3 & 4 & 6 & 7 — TEMPORAL DISTRIBUTION & SENTINEL ANALYSIS
    # -------------------------------------------------------------------
    print("\n" + "="*50)
    print("TEMPORAL FEATURE DISTRIBUTIONS: CORE VS NON-CORE")
    print("="*50)
    
    nodes = list(sorted(graph.nodes()))
    temp_df = temporal.features.reindex(nodes).copy()
    temp_df = temp_df.fillna(-1.0)
    for col in ["activity_7d", "activity_30d", "mean_iat"]:
        mask = temp_df[col] >= 0
        temp_df.loc[mask, col] = np.log1p(temp_df.loc[mask, col])
        
    core_df = temp_df.loc[core_nodes]
    non_core_df = temp_df.drop(index=core_nodes)
    
    stats = []
    for col in temporal.feature_names:
        c_vals = core_df[col].values
        nc_series = non_core_df[col]
        all_series = temp_df[col]
        
        stat = {
            "Feature": col,
            "Core Values": list(c_vals),
            "Overall Min": float(all_series.min()),
            "Overall Max": float(all_series.max()),
            "Overall Mean": float(all_series.mean()),
            "Overall Std": float(all_series.std()),
            "Zeros Count": (all_series == 0.0).sum(),
            "-1 Count": (all_series == -1.0).sum(),
            "Core Min": float(np.min(c_vals)),
            "Core Max": float(np.max(c_vals)),
            "NC Min": float(nc_series.min()),
            "NC Max": float(nc_series.max()),
            "NC Median": float(nc_series.median()),
            "Core Sentinels (-1)": (core_df[col] == -1.0).sum(),
            "NC Sentinels (-1)": (non_core_df[col] == -1.0).sum()
        }
        stats.append(stat)
        
    stats_df = pd.DataFrame(stats).set_index("Feature")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats_df)
    
    # -------------------------------------------------------------------
    # Verify Multigraph Signal is Preserved
    # -------------------------------------------------------------------
    mg_df = multigraph.features.reindex(nodes).copy()
    mg_df = mg_df.fillna(-1.0)
    for col in multigraph.feature_names:
        mask = mg_df[col] >= 0
        mg_df.loc[mask, col] = np.log1p(mg_df.loc[mask, col])
        
    core_mg = mg_df.loc[core_nodes]
    non_core_mg = mg_df.drop(index=core_nodes)
    
    rec_out = "reciprocal_out_count"
    print("\n" + "="*50)
    print("MULTIGRAPH SIGNAL VERIFICATION (After Preprocessing)")
    print("="*50)
    print(f"{rec_out} Core Values (log1p): {core_mg[rec_out].values}")
    print(f"{rec_out} Core Min: {core_mg[rec_out].min():.4f}")
    print(f"{rec_out} NC Max: {non_core_mg[rec_out].max():.4f}")
    
    # Let's inspect the order of concatenation in PyG tensor
    # x = pd.concat([baseline, temporal, multigraph]) ... actually dataset.py does:
    # feat_df = pd.concat([baseline, temp_df], axis=1)
    # feat_df = pd.concat([feat_df, mg_df], axis=1)
    # So indices:
    # 0-159 = baseline
    # 160-165 = temporal
    # 166-172 = multigraph
    print(f"\nVerifying actual Tensor X...")
    x = pyg_4.data.x
    col_idx = 166 + multigraph.feature_names.index(rec_out)
    core_indices = [nodes.index(c) for c in core_nodes]
    nc_indices = [nodes.index(c) for c in nodes if c not in core_nodes]
    
    c_tensor_vals = x[core_indices, col_idx]
    nc_tensor_vals = x[nc_indices, col_idx]
    print(f"Tensor Core Min (Col {col_idx}): {c_tensor_vals.min().item():.4f}")
    print(f"Tensor NC Max (Col {col_idx}): {nc_tensor_vals.max().item():.4f}")

if __name__ == "__main__":
    audit_exp4()
