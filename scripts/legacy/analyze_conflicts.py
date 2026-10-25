import sys, logging
from collections import defaultdict
import pandas as pd

sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

from src.graph.builder import build_graph_from_dataset
from src.data.preprocessing import load_and_preprocess

# Load preprocessed data and graph
prep = load_and_preprocess('../dkdata.json')
result = build_graph_from_dataset('../dkdata.json')
conflicts = result.node_conflict_report

bank_country_conflicts = {}
jurisdiction_conflicts = {}

for entity_id, c in conflicts.items():
    if 'bank_country' in c:
        bank_country_conflicts[entity_id] = c['bank_country']
    if 'jurisdiction' in c:
        jurisdiction_conflicts[entity_id] = c['jurisdiction']

print("=== CONFLICT ANALYSIS ===")
print(f"Entities with bank_country conflicts: {len(bank_country_conflicts)}")
print(f"Entities with jurisdiction conflicts: {len(jurisdiction_conflicts)}\n")

print("--- Jurisdiction Conflict ---")
for e, c in jurisdiction_conflicts.items():
    print(f"Entity: {e}, Values: {c}")

print("\n--- Bank Country Conflict Pairs ---")
pair_counts = defaultdict(int)
for e, c in bank_country_conflicts.items():
    pair = tuple(sorted([str(x) for x in c]))
    pair_counts[pair] += 1

for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
    print(f"{pair}: {count}")

print("\n--- Unique Values in Bank Country Conflicts ---")
all_values = set()
for c in bank_country_conflicts.values():
    all_values.update(c)
print(sorted([str(x) for x in all_values]))
