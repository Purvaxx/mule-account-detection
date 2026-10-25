import sys
import pandas as pd
import numpy as np

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.temporal import compute_temporal

def audit_temporal():
    ing_res = load_transactions("../dkdata.json")
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    graph = graph_res.graph
    edge_table = graph_res.edge_table
    
    baseline = compute_baseline(graph, edge_table)
    temporal = compute_temporal(edge_table, list(graph.nodes()))
    
    # 2. TEMPORAL FEATURE VALUES
    print("\n" + "="*50)
    print("TEMPORAL FEATURE DISTRIBUTIONS (3440 entities)")
    print("="*50)
    
    df = temporal.features
    stats = []
    
    for col in df.columns:
        series = df[col]
        non_nan = series.dropna()
        
        stat = {
            "Feature": col,
            "Min": float(non_nan.min()) if len(non_nan) > 0 else np.nan,
            "Max": float(non_nan.max()) if len(non_nan) > 0 else np.nan,
            "Mean": float(non_nan.mean()) if len(non_nan) > 0 else np.nan,
            "Median": float(non_nan.median()) if len(non_nan) > 0 else np.nan,
            "Std": float(non_nan.std()) if len(non_nan) > 0 else np.nan,
            "NaN Count": series.isna().sum(),
            "Zero Count": (series == 0).sum(),
            "Unique Count": series.nunique()
        }
        stats.append(stat)
        
    stats_df = pd.DataFrame(stats).set_index("Feature")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(stats_df)
    
    # Baseline comparison (scaling mismatch)
    print("\n" + "="*50)
    print("BASELINE FEATURE SCALING (Comparison)")
    print("="*50)
    
    base_df = baseline.features
    base_summary = {
        "log_in_degree": {"max": base_df["log_in_degree"].max(), "mean": base_df["log_in_degree"].mean()},
        "log_out_degree": {"max": base_df["log_out_degree"].max(), "mean": base_df["log_out_degree"].mean()},
        "log_usd_in": {"max": base_df["log_total_incoming_usd"].max(), "mean": base_df["log_total_incoming_usd"].mean()},
        "log_usd_out": {"max": base_df["log_total_outgoing_usd"].max(), "mean": base_df["log_total_outgoing_usd"].mean()},
    }
    for k, v in base_summary.items():
        print(f"{k}: Max={v['max']:.4f}, Mean={v['mean']:.4f}")
        
    print("\nCategorical baseline features are 0/1.")

if __name__ == "__main__":
    audit_temporal()
