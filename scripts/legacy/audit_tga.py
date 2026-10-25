import time
import pandas as pd
import numpy as np

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.tda import compute_tga

def audit_tga():
    ing_res = load_transactions("../dkdata.json")
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    graph = graph_res.graph
    
    core_nodes = [n for n, attr in graph.nodes(data=True) if attr.get("core")]
    
    print("\n" + "="*50)
    print("STARTING TGA AUDIT ON REAL DATASET")
    print("="*50)
    
    start_time = time.time()
    tga_res = compute_tga(graph)
    end_time = time.time()
    
    df = tga_res.features
    
    print(f"Total nodes: {len(df)}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    
    rec_mask = df["temporal_reciprocal_count"] > 0
    long_mask = df["temporal_cycle_count"] > 0
    
    print(f"Nodes with reciprocal temporal cycles: {rec_mask.sum()}")
    print(f"Nodes with >=3-edge temporal cycles: {long_mask.sum()}")
    
    print(f"Max reciprocal cycle count: {df['temporal_reciprocal_count'].max()}")
    print(f"Max longer-cycle count: {df['temporal_cycle_count'].max()}")
    
    earliest = df[df["earliest_cycle_completion_days"] >= 0]["earliest_cycle_completion_days"]
    print("\nDistribution of earliest cycle completion (in days, for nodes with cycles):")
    if not earliest.empty:
        print(earliest.describe())
    else:
        print("No cycles found.")
        
    print("\n" + "="*50)
    print("CORE VS NON-CORE FEATURE DISTRIBUTIONS")
    print("="*50)
    
    core_df = df.loc[core_nodes]
    nc_df = df.drop(index=core_nodes)
    
    for col in df.columns:
        print(f"\nFeature: {col}")
        print(f"Core values: {core_df[col].values}")
        print(f"Core Min: {core_df[col].min()}, Core Max: {core_df[col].max()}")
        print(f"Non-Core Min: {nc_df[col].min()}, Non-Core Max: {nc_df[col].max()}")
        
    print("\n" + "="*50)
    print("REDUNDANCY WITH MULTIGRAPH FEATURES")
    print("="*50)
    # The previous multigraph feature "reciprocal_out_count" had Core Min = 8, NC Max = 3.
    # Let's see if "temporal_reciprocal_count" is identical or different due to temporal constraints.
    print("If temporal_reciprocal_count matches reciprocal_out_count exactly, TGA is redundant.")
    print("Core temporal_reciprocal_count values:", core_df["temporal_reciprocal_count"].values)

if __name__ == "__main__":
    audit_tga()
