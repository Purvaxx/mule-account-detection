import sys
import pandas as pd
import numpy as np

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph

def audit_exp3():
    ing_res = load_transactions("../dkdata.json")
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    graph = graph_res.graph
    edge_table = graph_res.edge_table
    
    baseline = compute_baseline(graph, edge_table)
    mg = compute_multigraph(graph, edge_table)
    
    core_nodes = [n for n, attr in graph.nodes(data=True) if attr.get("core")]
    
    # -------------------------------------------------------------------
    # AUDIT 3 — FEATURE DISTRIBUTIONS (Core vs Non-Core)
    # -------------------------------------------------------------------
    print("\n" + "="*50)
    print("MULTIGRAPH FEATURE DISTRIBUTIONS: CORE VS NON-CORE")
    print("="*50)
    
    df = mg.features
    
    core_df = df.loc[core_nodes]
    non_core_df = df.drop(index=core_nodes)
    
    stats = []
    
    for col in df.columns:
        c_vals = core_df[col].dropna().values
        nc_series = non_core_df[col].dropna()
        
        stat = {
            "Feature": col,
            "Core Values": list(c_vals),
            "Core Min": float(np.min(c_vals)),
            "Core Max": float(np.max(c_vals)),
            "NC Min": float(nc_series.min()) if len(nc_series) > 0 else np.nan,
            "NC Max": float(nc_series.max()) if len(nc_series) > 0 else np.nan,
            "NC Median": float(nc_series.median()) if len(nc_series) > 0 else np.nan,
            "NC 95th": float(np.percentile(nc_series, 95)) if len(nc_series) > 0 else np.nan
        }
        stats.append(stat)
        
    stats_df = pd.DataFrame(stats).set_index("Feature")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats_df)
    
    # -------------------------------------------------------------------
    # AUDIT 5 — CORRELATION WITH BASELINE
    # -------------------------------------------------------------------
    print("\n" + "="*50)
    print("SPEARMAN RANK CORRELATION: MULTIGRAPH VS BASELINE")
    print("="*50)
    
    base_df = baseline.features
    combined = pd.concat([df, base_df], axis=1)
    
    baseline_cols = ["log_in_degree", "log_out_degree", "log_total_incoming_usd", "log_total_outgoing_usd", "flow_imbalance"]
    
    corr = combined[list(df.columns) + baseline_cols].corr(method='spearman')
    
    print(corr.loc[list(df.columns), baseline_cols])

if __name__ == "__main__":
    audit_exp3()
