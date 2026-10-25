import sys, logging
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

from src.data.preprocessing import load_and_preprocess

prep = load_and_preprocess('../dkdata.json')
df = prep.transactions

# To find out if 33 = EE, we can look at the conflicts.
# The previous script gave us:
# ('33', 'EE'): 27
# ('90', 'TR'): 2
# ('.D', 'DE'): 2
# ('19', 'CZ'): 1
# Let's list entities with ('33', 'EE', 'LV')
ent_33_ee_lv = []
for entity, c in prep.transactions.groupby('payer_entity_id')['payer_bank_country'].apply(lambda x: set(x.dropna())).items():
    if len(c) > 1:
        pass # we can do this via graph builder node_conflict_report

from src.graph.builder import build_graph_from_dataset
result = build_graph_from_dataset('../dkdata.json')
conflicts = result.node_conflict_report

print("--- Checking '33' vs 'EE' vs 'LV' ---")
for ent, c in conflicts.items():
    if 'bank_country' in c and set(c['bank_country']) == {'33', 'EE', 'LV'}:
        print(f"Entity: {ent}")
        rows = df[(df['payer_entity_id'] == ent) | (df['beneficiary_entity_id'] == ent)]
        for _, row in rows.iterrows():
            bc = row['payer_bank_country'] if row['payer_entity_id'] == ent else row['beneficiary_bank_country']
            acc = row['payer_account'] if row['payer_entity_id'] == ent else row['beneficiary_account']
            print(f"  BankCountry: {bc}, Account: {acc}, Source: {row['source_file']}")
