import pytest
import pandas as pd
from src.data.ingestion import IngestionResult
from src.data.preprocessing import preprocess, IdentitySource
from src.graph.builder import build_graph

def test_account_id_identity_mode():
    df = pd.DataFrame({
        "id": ["T1", "T2"],
        "date": ["2024-01-01", "2024-01-02"],
        "payer_name_norm": ["John Doe", "John Doe"],
        "beneficiary_name_norm": ["Alice", "Bob"],
        "payer_name": ["John Doe", "John Doe"],
        "beneficiary_name": ["Alice", "Bob"],
        "payer_account": ["ACC1", "ACC2"],
        "beneficiary_account": ["B_ACC1", "B_ACC2"],
        "payer_type": ["Company", "Company"],
        "beneficiary_type": ["Person", "Person"],
        "payer_bank_country": ["US", "US"],
        "beneficiary_bank_country": ["GB", "GB"],
        "payer_jurisdiction": ["US", "US"],
        "beneficiary_jurisdiction": ["GB", "GB"],
        "payer_core": [False, False],
        "beneficiary_core": [False, False],
        "amount": [100.0, 200.0],
        "amount_usd": [100.0, 200.0],
        "amount_orig": [100.0, 200.0],
        "amount_orig_currency": ["USD", "USD"],
        "amount_eur": [90.0, 180.0],
        "currency": ["USD", "USD"],
        "purpose": ["test", "test"],
        "source_file": ["test.csv", "test.csv"],
        "investigation": [False, False]
    })
    
    ingest = IngestionResult(transactions=df, record_count=len(df))
    
    prep_name = preprocess(ingest, identity_source=IdentitySource.NAME)
    assert not prep_name.validation_errors
    g_name = build_graph(prep_name)
    assert "John Doe" in g_name.graph.nodes()
    assert g_name.graph.out_degree("John Doe") == 2
    
    prep_acct = preprocess(ingest, identity_source=IdentitySource.ACCOUNT_ID)
    assert not prep_acct.validation_errors
    g_acct = build_graph(prep_acct)
    
    assert "ACC1" in g_acct.graph.nodes()
    assert "ACC2" in g_acct.graph.nodes()
    assert "John Doe" not in g_acct.graph.nodes()
    
    assert g_acct.graph.out_degree("ACC1") == 1
    assert g_acct.graph.out_degree("ACC2") == 1
    
    from src.features.baseline import compute_baseline
    baseline = compute_baseline(g_acct.graph, g_acct.edge_table)
    assert baseline.features.shape[1] == 160
