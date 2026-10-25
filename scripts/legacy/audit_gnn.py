import sys, logging
import pandas as pd
import networkx as nx
import numpy as np

sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

from src.graph.builder import build_graph_from_dataset

# 1. Load the graph
result = build_graph_from_dataset('../dkdata.json')
G = result.graph

# 2. Label Distribution
nodes = list(G.nodes(data=True))
core_nodes = [n for n, attr in nodes if attr.get('core') is True]
non_core_nodes = [n for n, attr in nodes if attr.get('core') is False]

num_core = len(core_nodes)
num_non_core = len(non_core_nodes)
total = len(nodes)
pos_pct = num_core / total * 100
imbalance_ratio = num_non_core / num_core if num_core > 0 else float('inf')

print("=== 1. LABEL DISTRIBUTION ===")
print(f"Total canonical entities: {total}")
print(f"Core entities (positive): {num_core}")
print(f"Non-core entities (negative): {num_non_core}")
print(f"Positive percentage: {pos_pct:.2f}%")
print(f"Class imbalance ratio (Negative:Positive): {imbalance_ratio:.2f}:1")

# Verify core label consistency
# Wait, builder.py already validates it, but we can verify node_conflict_report
print(f"Node conflict report for core: {[k for k, v in result.node_conflict_report.items() if 'core' in v]}")

# 3. Graph Connectivity
# Convert to simple directed graph for connectivity metrics
G_simple = nx.DiGraph(G)
# Convert to undirected for weak connectivity
G_undirected = G_simple.to_undirected()

wcc = list(nx.weakly_connected_components(G_simple))
scc = list(nx.strongly_connected_components(G_simple))
isolated = list(nx.isolates(G_simple))

print("\n=== 2. GRAPH CONNECTIVITY ===")
print(f"Weakly connected components: {len(wcc)}")
print(f"Largest WCC size: {max(len(c) for c in wcc) if wcc else 0}")
print(f"Strongly connected components: {len(scc)}")
print(f"Largest SCC size: {max(len(c) for c in scc) if scc else 0}")
print(f"Isolated nodes: {len(isolated)}")

in_degrees = [d for n, d in G.in_degree()]
out_degrees = [d for n, d in G.out_degree()]
print(f"In-degree: max={max(in_degrees)}, mean={np.mean(in_degrees):.2f}")
print(f"Out-degree: max={max(out_degrees)}, mean={np.mean(out_degrees):.2f}")

# Graph density (using multi-edges)
# Density = E / (V * (V-1))
density = G.number_of_edges() / (total * (total - 1))
print(f"Graph density (MultiDiGraph): {density:.6f}")
