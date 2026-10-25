import sys
import pandas as pd
import numpy as np

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph

def audit_multigraph():
    ing_res = load_transactions("../dkdata.json")
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    graph = graph_res.graph
    edge_table = graph_res.edge_table
    
    baseline = compute_baseline(graph, edge_table)
    mg = compute_multigraph(graph, edge_table)
    
    # AUDIT 1 — DISTRIBUTIONS
    print("\n" + "="*50)
    print("MULTIGRAPH FEATURE DISTRIBUTIONS (3440 entities)")
    print("="*50)
    
    df = mg.features
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
            "Inf Count": np.isinf(series).sum(),
            "Zero Count": (series == 0).sum(),
            "Unique Count": series.nunique()
        }
        stats.append(stat)
        
    stats_df = pd.DataFrame(stats).set_index("Feature")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats_df)
    
if __name__ == "__main__":
    audit_multigraph()
