import sys, logging
sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

from src.graph.builder import build_graph_from_dataset
from src.features.baseline import compute_baseline
from src.features.temporal import compute_temporal
from src.features.multigraph import compute_multigraph

r = build_graph_from_dataset('../dkdata.json')
entities = sorted(r.graph.nodes())

b = compute_baseline(r.graph, r.edge_table)
t = compute_temporal(r.edge_table, entities)
m = compute_multigraph(r.graph, r.edge_table)

print("=== LAYER 4 VERIFICATION ===")
print(f"Nodes:        {r.graph.number_of_nodes():,}")
print(f"Edges:        {r.graph.number_of_edges():,}")
print()
print(f"BASELINE:  {b.entity_count} entities × {b.features.shape[1]:3d} features  | NaN={b.nan_count}  inf={b.inf_count}")
print(f"           Entity type categories: {b.entity_type_categories}")
print(f"           Country codes: {len(b.country_codes)} distinct raw codes")
print()
print(f"TEMPORAL:  {t.entity_count} entities × {t.features.shape[1]:3d} features  | NaN={t.nan_count}  inf={t.inf_count}")
print(f"           Features: {t.feature_names}")
print()
print(f"MULTIGRAPH:{m.entity_count} entities × {m.features.shape[1]:3d} features  | NaN={m.nan_count}  inf={m.inf_count}")
print(f"           Features: {m.feature_names}")
print()
print(f"Max parallel edge count (global):  {m.features['max_parallel_edge_count'].max():.0f}")
print(f"Entities with reciprocal behaviour:{int((m.features['reciprocal_out_count'] > 0).sum())}")
print(f"Temporal NaNs (expected, entities with 1 tx): {t.nan_count}")
