import pandas as pd
import numpy as np
import networkx as nx
from src.graph.builder import build_graph
from src.features.tda import compute_tga

from src.data.preprocessing import load_and_preprocess

def main():
    # Load graph
    prep = load_and_preprocess('../dkdata.json')
    G_res = build_graph(prep)
    G = G_res.graph
    
    # 1. Base Features
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
    # max parallel edge count
    max_parallel = {}
    for u in G.nodes():
        mp = 0
        for w in G.successors(u):
            c = G.number_of_edges(u, w)
            if c > mp: mp = c
        max_parallel[u] = mp
        
    # reciprocal_out_count
    rec_out = {}
    for u in G.nodes():
        count = 0
        for w in G.successors(u):
            if G.has_edge(w, u):
                count += 1
        rec_out[u] = count
        
    # total USD
    usd = {}
    for u in G.nodes():
        total = 0.0
        for _, _, d in G.in_edges(u, data=True):
            total += d.get('amount_usd', 0)
        for _, _, d in G.out_edges(u, data=True):
            total += d.get('amount_usd', 0)
        usd[u] = total
        
    # 2. Compute TGA
    tga_res = compute_tga(G)
    tga_df = tga_res.features
    
    # Combine
    df = pd.DataFrame({
        'in_degree': pd.Series(in_deg),
        'out_degree': pd.Series(out_deg),
        'max_parallel': pd.Series(max_parallel),
        'reciprocal_out_count': pd.Series(rec_out),
        'total_usd': pd.Series(usd),
        'temporal_rec_count': tga_df['temporal_reciprocal_count'],
        'temporal_cycle_count': tga_df['temporal_cycle_count'],
        'earliest_comp_days': tga_df['earliest_cycle_completion_days']
    })
    
    # Log transforms
    df['log_in_degree'] = np.log1p(df['in_degree'])
    df['log_out_degree'] = np.log1p(df['out_degree'])
    df['log_total_usd'] = np.log1p(df['total_usd'])
    df['log_temp_rec_count'] = np.log1p(df['temporal_rec_count'])
    df['log_temp_cycle_count'] = np.log1p(df['temporal_cycle_count'])
    
    # Correlation
    cols = ['log_in_degree', 'log_out_degree', 'log_total_usd', 'reciprocal_out_count', 'max_parallel']
    tga_cols = ['temporal_rec_count', 'temporal_cycle_count', 'earliest_comp_days', 'log_temp_rec_count', 'log_temp_cycle_count']
    
    print("========== CORRELATIONS (Spearman) ==========")
    corr = df[cols + tga_cols].corr(method='spearman')
    for tc in tga_cols:
        print(f"\nCorrelations for {tc}:")
        for c in cols:
            print(f"  {c}: {corr.loc[tc, c]:.4f}")
            
    print("\n========== SUMMARY STATS ==========")
    print(df[['temporal_rec_count', 'temporal_cycle_count', 'earliest_comp_days']].describe())
    
    print("\n========== 4TH CORE NODE INVESTIGATION ==========")
    # Find the core nodes
    cores = [n for n, d in G.nodes(data=True) if d.get('is_core', False)]
    for c in cores:
        tc = df.loc[c, 'temporal_rec_count']
        print(f"Core: {c}, temporal_rec_count: {tc}")
        if tc == 0:
            print(f"  INVESTIGATING WHY Core {c} has 0 temporal reciprocal cycles:")
            # List reciprocal edges
            for w in set(G.successors(c)).intersection(set(G.predecessors(c))):
                print(f"  Counterparty: {w}")
                out_dates = sorted([d['date'] for u, v, d in G.edges(c, data=True) if v == w])
                in_dates = sorted([d['date'] for u, v, d in G.edges(w, data=True) if v == c])
                print(f"    Out edges (Core -> w) dates: {out_dates}")
                print(f"    In edges (w -> Core) dates: {in_dates}")
                
                # Manual causality check
                valid = [ (d1, d2) for d1 in out_dates for d2 in in_dates if d1 <= d2 ]
                print(f"    Valid causal pairs (Out <= In): {len(valid)}")

if __name__ == '__main__':
    main()
