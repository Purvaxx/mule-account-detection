import sys
import pandas as pd
import numpy as np

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline

def investigate_core_nodes():
    print("Loading and preparing data...")
    ing_res = load_transactions("../dkdata.json")
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    graph = graph_res.graph
    edge_table = graph_res.edge_table
    baseline = compute_baseline(graph, edge_table)
    
    nodes = list(graph.nodes(data=True))
    core_nodes = [n for n, d in nodes if d.get('core', False)]
    
    print(f"\nFound {len(core_nodes)} core entities.")
    
    for core_node in core_nodes:
        print("="*60)
        print(f"Entity ID: {core_node}")
        
        # Look up baseline features
        row = baseline.features.loc[core_node]
        in_deg = np.exp(row['log_in_degree']) - 1
        out_deg = np.exp(row['log_out_degree']) - 1
        in_usd = np.exp(row['log_total_incoming_usd']) - 1
        out_usd = np.exp(row['log_total_outgoing_usd']) - 1
        imbalance = row['flow_imbalance']
        
        print(f"log_in_degree: {row['log_in_degree']:.4f} (Actual: {in_deg:.1f})")
        print(f"log_out_degree: {row['log_out_degree']:.4f} (Actual: {out_deg:.1f})")
        print(f"log_total_incoming_usd: {row['log_total_incoming_usd']:.4f} (Actual: ${in_usd:.2f})")
        print(f"log_total_outgoing_usd: {row['log_total_outgoing_usd']:.4f} (Actual: ${out_usd:.2f})")
        print(f"flow_imbalance: {imbalance:.4f}")
        
        entity_types = [c for c in baseline.feature_names if c.startswith("entity_type_") and row[c] == 1]
        print(f"Entity Type: {entity_types}")
        
        bank_countries = [c for c in baseline.feature_names if c.startswith("bank_country_") and row[c] == 1]
        print(f"Bank Countries: {bank_countries}")
        
        # Graph neighborhood
        in_edges = list(graph.in_edges(core_node, data=True))
        out_edges = list(graph.out_edges(core_node, data=True))
        print(f"\nIncoming edges ({len(in_edges)}):")
        for u, v, d in in_edges[:5]:
            print(f"  {u} -> {v} | USD: {d['amount_usd']} | key: {d.get('key')}")
        if len(in_edges) > 5: print("  ...")
            
        print(f"\nOutgoing edges ({len(out_edges)}):")
        for u, v, d in out_edges[:5]:
            print(f"  {u} -> {v} | USD: {d['amount_usd']} | key: {d.get('key')}")
        if len(out_edges) > 5: print("  ...")

if __name__ == "__main__":
    investigate_core_nodes()
