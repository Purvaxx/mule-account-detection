import sys, logging
from collections import defaultdict
import pandas as pd

sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

from src.data.preprocessing import load_and_preprocess

# We will inspect the dataframe to find ALL occurrences of numeric/ISO codes to build a mapping.
prep = load_and_preprocess('../dkdata.json')
df = prep.transactions

# Combine all bank_country values with other data to infer mappings
country_codes = pd.concat([df['payer_bank_country'], df['beneficiary_bank_country']]).dropna().unique()

print(f"All unique country codes in dataset: {sorted([str(c) for c in country_codes])}")

# Let's see if we can map numbers to ISOs using the entity conflicts
# If an entity has both '33' and 'EE', it's strong evidence '33'='EE'
# But '33' and 'LV'?
# Let's print out the full rows for some conflicting entities to see the source_file or other columns
# Maybe '33' came from one source file, and 'EE' from another?

target_entities = ['SISCON MEDICAL Gmbh', 'GEISTLICH PHARMA']

for ent in target_entities:
    print(f"\n--- Entity: {ent} ---")
    rows = df[(df['payer_entity_id'] == ent) | (df['beneficiary_entity_id'] == ent)]
    for _, row in rows.iterrows():
        bc = row['payer_bank_country'] if row['payer_entity_id'] == ent else row['beneficiary_bank_country']
        jur = row['payer_jurisdiction'] if row['payer_entity_id'] == ent else row['beneficiary_jurisdiction']
        print(f"Date: {row['date'].strftime('%Y-%m-%d')}, BankCountry: {bc}, Jur: {jur}, Source: {row['source_file']}")
