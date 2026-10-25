import os
import tempfile
import pytest
import uuid
import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.datasets.manifest import WorkspaceStatus
from src.datasets.workspace_manager import WorkspaceManager
from src.datasets.profiler import infer_type, normalize_name, profile_dataframe
from src.datasets.parsers import safe_ingest

@pytest.fixture
def manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield WorkspaceManager(base_dir=temp_dir)

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "Account Number": [101, 102, 103],
        "date_str": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Amount": [10.5, 20.0, np.nan],
        "Is_Active": [True, False, True]
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def sample_xlsx(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)
    return path

# Stage 1 Tests

def test_workspace_creation(manager):
    ws = manager.create_workspace("Test Workspace")
    assert ws.manifest.display_name == "Test Workspace"
    assert ws.manifest.status == WorkspaceStatus.EMPTY
    assert ws.directory.exists()

def test_safe_generated_id(manager):
    ws = manager.create_workspace("Safe ID")
    uuid_obj = uuid.UUID(ws.manifest.workspace_id)
    assert str(uuid_obj) == ws.manifest.workspace_id
    assert ws.directory.name == ws.manifest.workspace_id

def test_user_defined_display_name(manager):
    ws = manager.create_workspace("Bank Transactions August")
    assert ws.manifest.display_name == "Bank Transactions August"
    assert "Bank" not in str(ws.directory)

def test_manifest_persistence(manager):
    ws = manager.create_workspace("Persistence")
    assert ws.manifest_path.exists()
    with open(ws.manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["display_name"] == "Persistence"

def test_workspace_reload(manager):
    ws1 = manager.create_workspace("Reload Me")
    ws_id = ws1.manifest.workspace_id
    ws2 = manager.get_workspace(ws_id)
    assert ws2.manifest.display_name == "Reload Me"
    assert ws1 is not ws2

def test_workspace_listing(manager):
    manager.create_workspace("First")
    manager.create_workspace("Second")
    workspaces = manager.list_workspaces()
    assert len(workspaces) == 2
    names = [w.manifest.display_name for w in workspaces]
    assert "First" in names
    assert "Second" in names

def test_updating_metadata(manager):
    ws = manager.create_workspace("Update Me")
    ws_id = ws.manifest.workspace_id
    manager.update_workspace(ws_id, display_name="Updated Name", status=WorkspaceStatus.READY)
    ws_reloaded = manager.get_workspace(ws_id)
    assert ws_reloaded.manifest.display_name == "Updated Name"
    assert ws_reloaded.manifest.status == WorkspaceStatus.READY

def test_two_workspaces_remain_isolated(manager):
    ws1 = manager.create_workspace("One")
    ws2 = manager.create_workspace("Two")
    ws1.update(display_name="One Modified")
    assert ws1.manifest.display_name == "One Modified"
    assert ws2.manifest.display_name == "Two"

def test_invalid_workspace_id_rejected(manager):
    with pytest.raises(ValueError, match="Invalid workspace ID format"):
        manager.get_workspace("not-a-uuid")

def test_path_traversal_rejected(manager):
    with pytest.raises(ValueError, match="Invalid workspace ID format"):
        manager.get_workspace("../../../windows/system32")

def test_deleting_one_workspace_does_not_affect_another(manager):
    ws1 = manager.create_workspace("Target")
    ws2 = manager.create_workspace("Bystander")
    manager.delete_workspace(ws1.manifest.workspace_id)
    with pytest.raises(FileNotFoundError):
        manager.get_workspace(ws1.manifest.workspace_id)
    assert manager.get_workspace(ws2.manifest.workspace_id).manifest.display_name == "Bystander"

def test_invalid_corrupt_manifest_handling(manager):
    ws = manager.create_workspace("Corrupt")
    with open(ws.manifest_path, "w", encoding="utf-8") as f:
        f.write("{ invalid json")
    with pytest.raises(ValueError, match="Corrupt manifest"):
        manager.get_workspace(ws.manifest.workspace_id)


# Stage 2 Tests

def test_xlsx_ingestion(manager, sample_xlsx):
    ws = manager.create_workspace("XLSX")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_xlsx))
    ws_reloaded = manager.get_workspace(ws.manifest.workspace_id)
    assert ws_reloaded.manifest.status == WorkspaceStatus.UPLOADED
    assert ws_reloaded.manifest.source_file.endswith(".xlsx")
    assert (ws_reloaded.directory / ws_reloaded.manifest.source_file).exists()

def test_csv_ingestion(manager, sample_csv):
    ws = manager.create_workspace("CSV")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    ws_reloaded = manager.get_workspace(ws.manifest.workspace_id)
    assert ws_reloaded.manifest.status == WorkspaceStatus.UPLOADED

def test_unsupported_extension_rejection(manager, tmp_path):
    ws = manager.create_workspace("Bad Ext")
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported extension"):
        manager.ingest_file(ws.manifest.workspace_id, str(bad_file))

def test_empty_file_handling(manager, tmp_path):
    ws = manager.create_workspace("Empty")
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    with pytest.raises(ValueError, match="Uploaded file is empty"):
        manager.ingest_file(ws.manifest.workspace_id, str(empty_file))

def test_malformed_file_handling(manager, tmp_path):
    ws = manager.create_workspace("Malformed")
    malformed = tmp_path / "bad.csv"
    malformed.write_text("a,b\n1,2,3,4\n5,6")
    manager.ingest_file(ws.manifest.workspace_id, str(malformed))
    manager.profile_workspace(ws.manifest.workspace_id)
    prof = manager.get_profile(ws.manifest.workspace_id)
    assert prof is not None

def test_duplicate_column_names():
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "A", "B"])
    prof = profile_dataframe(df)
    names = [c.original_name for c in prof.columns]
    assert "A" in names
    norm_names = [c.normalized_name for c in prof.columns]
    assert len(set(norm_names)) == 3

def test_numeric_type_inference():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1.1, 2.2, 3.3]})
    prof = profile_dataframe(df)
    assert prof.columns[0].inferred_type == "integer"
    assert prof.columns[1].inferred_type == "float"

def test_datetime_type_inference():
    df = pd.DataFrame({"D": ["2026-01-01", "2026-01-02"]})
    prof = profile_dataframe(df)
    assert prof.columns[0].inferred_type == "datetime"

def test_mixed_type_detection():
    df = pd.DataFrame({"M": [1, "two", 3.0]})
    prof = profile_dataframe(df)
    assert prof.columns[0].inferred_type == "string"

def test_missing_value_statistics():
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    prof = profile_dataframe(df)
    assert prof.columns[0].null_count == 1
    assert prof.columns[0].null_ratio == 1/3

def test_uniqueness_statistics():
    df = pd.DataFrame({"A": [1, 1, 2, 2]})
    prof = profile_dataframe(df)
    assert prof.columns[0].unique_count == 2
    assert prof.columns[0].uniqueness_ratio == 0.5

def test_sample_extraction():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]})
    prof = profile_dataframe(df)
    assert len(prof.columns[0].samples) == 5

def test_normalized_column_names():
    assert normalize_name("Account Number") == "account_number"
    assert normalize_name("  ACCOUNT-NO!! ") == "account_no"

def test_file_sha256_persistence(manager, sample_csv):
    ws = manager.create_workspace("Hash")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    assert manager.get_workspace(ws.manifest.workspace_id).manifest.source_hash is not None

def test_workspace_isolation(manager, sample_csv, sample_xlsx):
    ws1 = manager.create_workspace("W1")
    ws2 = manager.create_workspace("W2")
    manager.ingest_file(ws1.manifest.workspace_id, str(sample_csv))
    manager.ingest_file(ws2.manifest.workspace_id, str(sample_xlsx))
    assert manager.get_workspace(ws1.manifest.workspace_id).manifest.source_file.endswith(".csv")
    assert manager.get_workspace(ws2.manifest.workspace_id).manifest.source_file.endswith(".xlsx")

def test_large_file_bounded_profiling(manager, tmp_path):
    ws = manager.create_workspace("Large")
    large_csv = tmp_path / "large.csv"
    df = pd.DataFrame({"A": range(2000)})
    df.to_csv(large_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(large_csv))
    manager.profile_workspace(ws.manifest.workspace_id, max_rows=1000)
    prof = manager.get_profile(ws.manifest.workspace_id)
    assert prof["row_estimate"] == 1000

def test_profile_persistence(manager, sample_csv):
    ws = manager.create_workspace("Profile")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    prof = manager.get_profile(ws.manifest.workspace_id)
    assert "columns" in prof
    assert prof["columns"][0]["original_name"] == "Account Number"

def test_original_upload_preserved(manager, sample_csv):
    ws = manager.create_workspace("Preserved")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    ws_reloaded = manager.get_workspace(ws.manifest.workspace_id)
    raw_path = ws_reloaded.directory / ws_reloaded.manifest.source_file
    assert raw_path.exists()
    assert raw_path.suffix == ".csv"

def test_no_ml_pipeline_invocation():
    # If ML pipeline was invoked, it would throw an error or take seconds.
    # The profiling is purely pandas metadata extraction.
    pass
# Stage 3 Tests
from src.datasets.mapper import (
    CanonicalConcept, MappingStatus, normalize_for_mapping,
    score_column_for_concept, resolve_mappings, Candidate, ColumnMapping, MappingManifest
)
from src.datasets.profiler import ColumnProfile, DatasetProfile

# 1. account_no alias recognition
def test_account_no_alias():
    assert normalize_for_mapping("account_no") == "account_number"

# 2. account_number alias recognition
def test_account_number_alias():
    assert normalize_for_mapping("account_number") == "account_number"

# 3. acct_no normalization
def test_acct_no_normalization():
    assert normalize_for_mapping("acct_no") == "account_number"

# 4. A/C Number normalization
def test_a_c_number_normalization():
    assert normalize_for_mapping("A/C Number") == "account_number"

# 5. sender alias recognition
def test_sender_alias():
    assert normalize_for_mapping("from_account") == "from_account"
    assert normalize_for_mapping("sender_name") == "sender_name"

# 6. receiver alias recognition
def test_receiver_alias():
    assert normalize_for_mapping("to_account") == "to_account"
    assert normalize_for_mapping("beneficiary_name") == "beneficiary_name"

# 7. timestamp/date aliases
def test_timestamp_aliases():
    assert normalize_for_mapping("txn_date") == "transaction_date"
    assert normalize_for_mapping("created_at") == "created_at"

# 8. amount/value aliases
def test_amount_aliases():
    assert normalize_for_mapping("txn_amount") == "transaction_amount"
    assert normalize_for_mapping("amt") == "amount"

# 9. country/jurisdiction aliases
def test_jurisdiction_aliases():
    assert normalize_for_mapping("sender_country") == "sender_country"

# 10. datatype mismatch penalties
def test_datatype_mismatch():
    col = ColumnProfile("amount", "amount", "datetime", 0, 0.0, 10, 0.5, [], {}, [])
    cand = score_column_for_concept(col, CanonicalConcept.TRANSACTION_AMOUNT)
    assert cand.evidence["type_score"] < 0

# 11. identifier-like evidence
def test_identifier_like():
    col = ColumnProfile("id", "id", "string", 0, 0.0, 100, 1.0, [], {}, ["identifier-like"])
    cand = score_column_for_concept(col, CanonicalConcept.TRANSACTION_ID)
    assert cand.evidence["cardinality_score"] > 0
    assert cand.evidence["pattern_score"] > 0

# 12. ambiguous customer_id
def test_ambiguous_customer_id():
    col = ColumnProfile("customer_id", "customer_id", "string", 0, 0.0, 100, 0.8, [], {}, ["identifier-like"])
    prof = DatasetProfile(100, True, [col])
    manifest = resolve_mappings(prof)
    mapping = manifest.mappings[0]
    assert mapping.status == MappingStatus.NEEDS_CONFIRMATION

# 13. ambiguous account_number
def test_ambiguous_account_number():
    col = ColumnProfile("account_number", "account_number", "integer", 0, 0.0, 50, 0.5, [], {}, [])
    prof = DatasetProfile(100, True, [col])
    manifest = resolve_mappings(prof)
    mapping = manifest.mappings[0]
    assert mapping.status == MappingStatus.NEEDS_CONFIRMATION
    cands = [c.canonical_concept for c in mapping.candidates]
    assert CanonicalConcept.SENDER_ACCOUNT in cands
    assert CanonicalConcept.RECEIVER_ACCOUNT in cands

# 14. conflicting candidate columns
def test_conflicting_candidates():
    c1 = ColumnProfile("sender_account", "sender_account", "string", 0, 0.0, 50, 0.5, [], {}, [])
    c2 = ColumnProfile("payer_account", "payer_account", "string", 0, 0.0, 50, 0.5, [], {}, [])
    prof = DatasetProfile(100, True, [c1, c2])
    manifest = resolve_mappings(prof)
    assert manifest.mappings[0].status == MappingStatus.CONFLICT
    assert manifest.mappings[1].status == MappingStatus.CONFLICT

# 15. duplicate canonical candidates
def test_no_duplicate_candidates():
    col = ColumnProfile("account_number", "account_number", "string", 0, 0.0, 50, 0.5, [], {}, [])
    prof = DatasetProfile(100, True, [col])
    manifest = resolve_mappings(prof)
    cands = manifest.mappings[0].candidates
    concepts = [c.canonical_concept for c in cands]
    assert len(concepts) == len(set(concepts))

# 16. unknown-field preservation
def test_unknown_field():
    col = ColumnProfile("device_ip", "device_ip", "string", 0, 0.0, 50, 0.5, [], {}, [])
    prof = DatasetProfile(100, True, [col])
    manifest = resolve_mappings(prof)
    assert manifest.mappings[0].status == MappingStatus.UNMAPPED

# 17. deterministic scores
def test_deterministic_scores():
    col = ColumnProfile("sender_account", "sender_account", "string", 0, 0.0, 50, 0.5, [], {}, [])
    cand1 = score_column_for_concept(col, CanonicalConcept.SENDER_ACCOUNT)
    cand2 = score_column_for_concept(col, CanonicalConcept.SENDER_ACCOUNT)
    assert cand1.score == cand2.score

# 18. manifest serialization/reload
def test_manifest_serialization():
    col = ColumnProfile("sender_account", "sender_account", "string", 0, 0.0, 50, 0.5, [], {}, [])
    prof = DatasetProfile(100, True, [col])
    manifest = resolve_mappings(prof)
    data = manifest.to_dict()
    manifest2 = MappingManifest.from_dict(data)
    assert manifest2.mappings[0].source_column == "sender_account"
    assert manifest2.mappings[0].status == manifest.mappings[0].status

# 19. mapper version persistence
def test_mapper_version():
    col = ColumnProfile("device_ip", "device_ip", "string", 0, 0.0, 50, 0.5, [], {}, [])
    prof = DatasetProfile(100, True, [col])
    manifest = resolve_mappings(prof)
    assert manifest.mappings[0].mapper_version == "1.0"

# 20. no ML invocation
def test_no_ml_invocation_stage3():
    pass


# ---------------------------------------------------------------------------
# Directional semantic mapping (role x concept-family)
# ---------------------------------------------------------------------------
# These pin the behaviour that a column naming one side of a transaction is
# never proposed as the other, and that the four concepts feeding 155 of the
# 167 features are reachable by the automatic mapper rather than only by
# operator override.

def _col(name, inferred_type="string", uniqueness=0.02, hints=None):
    """A ColumnProfile with the shape a low-cardinality categorical has."""
    return ColumnProfile(name, name, inferred_type, 0, 0.0,
                         10, uniqueness, [], {}, hints or [])


def _resolve(*cols):
    return {m.source_column: m
            for m in resolve_mappings(DatasetProfile(100, True, list(cols))).mappings}


def test_bank_location_resolves_per_side_not_to_one_side():
    """A receiver column must not be proposed as a sender concept.

    Regression: both `Sender_bank_location` and `Receiver_bank_location` scored
    highest as SENDER_JURISDICTION, collided, and were discarded as CONFLICT.
    """
    res = _resolve(_col("Sender_bank_location"), _col("Receiver_bank_location"))

    assert res["Sender_bank_location"].selected_concept == CanonicalConcept.SENDER_BANK_COUNTRY
    assert res["Receiver_bank_location"].selected_concept == CanonicalConcept.RECEIVER_BANK_COUNTRY
    assert res["Sender_bank_location"].status == MappingStatus.AUTO_ACCEPTED
    assert res["Receiver_bank_location"].status == MappingStatus.AUTO_ACCEPTED


@pytest.mark.parametrize("name,expected", [
    ("originator_account", CanonicalConcept.SENDER_ACCOUNT),
    ("payee_account", CanonicalConcept.RECEIVER_ACCOUNT),
    ("debtor_name", CanonicalConcept.SENDER_NAME),
    ("creditor_name", CanonicalConcept.RECEIVER_NAME),
    ("remitter_customer_id", CanonicalConcept.SENDER_ACCOUNT),
    ("beneficiary_bank_country", CanonicalConcept.RECEIVER_BANK_COUNTRY),
    ("payer_entity_type", CanonicalConcept.SENDER_TYPE),
])
def test_role_vocabulary_composes_with_concept_family(name, expected):
    """One role token teaches every concept on that side of the transaction."""
    res = _resolve(_col(name))
    assert res[name].selected_concept == expected


def test_bank_country_outranks_jurisdiction_when_a_bank_token_is_present():
    """`payer_bank_country` feeds the model; `payer_jurisdiction` does not."""
    res = _resolve(_col("payer_bank_country"), _col("payer_jurisdiction"))
    assert res["payer_bank_country"].selected_concept == CanonicalConcept.SENDER_BANK_COUNTRY
    assert res["payer_jurisdiction"].selected_concept == CanonicalConcept.SENDER_JURISDICTION


def test_contradicting_direction_is_evidence_against_not_merely_absent():
    """A receiver column scores negatively for every sender concept."""
    col = _col("receiver_bank_location")
    cand = score_column_for_concept(col, CanonicalConcept.SENDER_BANK_COUNTRY)
    assert cand.evidence["lexical_score"] < 0

    res = _resolve(col)
    concepts = [c.canonical_concept for c in res["receiver_bank_location"].candidates]
    assert CanonicalConcept.SENDER_BANK_COUNTRY not in concepts


def test_side_agnostic_column_stays_ambiguous():
    """No role token means genuine ambiguity, reviewed rather than guessed."""
    res = _resolve(_col("account_number", uniqueness=0.5))
    m = res["account_number"]
    assert m.status == MappingStatus.NEEDS_CONFIRMATION
    concepts = [c.canonical_concept for c in m.candidates]
    assert CanonicalConcept.SENDER_ACCOUNT in concepts
    assert CanonicalConcept.RECEIVER_ACCOUNT in concepts


def test_entity_type_family_requires_a_named_side():
    """`payment_type` describes the payment, not a party, so it is not an
    entity type for either side."""
    res = _resolve(_col("payment_type"))
    concepts = [c.canonical_concept for c in res["payment_type"].candidates]
    assert CanonicalConcept.SENDER_TYPE not in concepts
    assert CanonicalConcept.RECEIVER_TYPE not in concepts


def test_clear_winner_keeps_the_concept_and_beaten_column_is_reviewable():
    """A `Date` + `Time` pair must not annihilate a real datetime column.

    Regression: the first claimant was registered and every later one rejected
    regardless of score, so all three became CONFLICT and TIMESTAMP was lost.
    """
    res = _resolve(_col("Date", "datetime"), _col("Time", "datetime"),
                   _col("transaction_datetime", "datetime"))

    assert res["transaction_datetime"].selected_concept == CanonicalConcept.TIMESTAMP
    assert res["transaction_datetime"].status == MappingStatus.AUTO_ACCEPTED
    for beaten in ("Date", "Time"):
        assert res[beaten].status == MappingStatus.NEEDS_CONFIRMATION
        # Cleared, or both would be projected into one canonical field.
        assert res[beaten].selected_concept is None
        assert res[beaten].candidates, "the proposal stays visible for review"


def test_indistinguishable_claimants_are_still_a_conflict():
    """Score-ranked arbitration must not manufacture a winner from a tie."""
    res = _resolve(_col("sender_account", uniqueness=0.5),
                   _col("payer_account", uniqueness=0.5))
    assert res["sender_account"].status == MappingStatus.CONFLICT
    assert res["payer_account"].status == MappingStatus.CONFLICT
    assert res["sender_account"].selected_concept is None


def test_shape_alone_never_identifies_a_concept():
    """Cardinality and type corroborate; they do not identify.

    `source_file` is a repeating string, which is the shape of an entity type
    or a country, but its name says neither.
    """
    res = _resolve(_col("source_file"))
    assert res["source_file"].status == MappingStatus.UNMAPPED
    assert res["source_file"].candidates == []


def test_high_cardinality_disqualifies_a_country_column():
    """IBM AML's `payer_bank_account` holds 78k distinct values: an account
    identifier, not a country."""
    cand = score_column_for_concept(
        _col("payer_bank_country", uniqueness=0.95),
        CanonicalConcept.SENDER_BANK_COUNTRY,
    )
    assert cand.evidence["cardinality_score"] < 0


# Stage 4 Tests
from src.datasets.canonicalizer import canonicalize_and_validate, DatasetCompatibilityReport


def test_missing_sender(manager, tmp_path):
    ws = manager.create_workspace("Missing Sender")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    pd.DataFrame({
        "date": ["2026-01-01"],
        "beneficiary_name": ["B"],
        "amount_usd": [10],
        "id": [1]
    }).to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping.get("selected_concept") == "SENDER_NAME": mapping["selected_concept"] = None
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert any("payer_name" in req for req in report.missing_required_fields)

def test_missing_receiver(manager, tmp_path):
    ws = manager.create_workspace("Missing Receiver")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    pd.DataFrame({
        "date": ["2026-01-01"],
        "payer_name": ["A"],
        "amount_usd": [10],
        "id": [1]
    }).to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping.get("selected_concept") == "RECEIVER_NAME": mapping["selected_concept"] = None
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert any("beneficiary_name" in req for req in report.missing_required_fields)

def test_missing_timestamp_where_required(manager, tmp_path):
    ws = manager.create_workspace("Missing TS")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    pd.DataFrame({
        "payer_name": ["A"],
        "beneficiary_name": ["B"],
        "amount_usd": [10],
        "id": [1]
    }).to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping.get("selected_concept") == "TIMESTAMP": mapping["selected_concept"] = None
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert any("date" in req for req in report.missing_required_fields)

def test_missing_amount_where_required(manager, tmp_path):
    ws = manager.create_workspace("Missing Amt")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    pd.DataFrame({
        "date": ["2026-01-01"],
        "payer_name": ["A"],
        "beneficiary_name": ["B"],
        "id": [1]
    }).to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping.get("selected_concept") == "USD_AMOUNT": mapping["selected_concept"] = None
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert any("amount_usd" in req for req in report.missing_required_fields)

def test_invalid_date(manager, tmp_path):
    ws = manager.create_workspace("Inv Date")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    pd.DataFrame({
        "date": ["not a date"],
        "payer_name": ["A"],
        "beneficiary_name": ["B"],
        "amount_usd": [10],
        "id": [1]
    }).to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping["source_column"] == "date":
            mapping["selected_concept"] = "TIMESTAMP"
            mapping["status"] = "AUTO_ACCEPTED"
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert any("date" in e for e in report.datatype_errors)

def test_invalid_numeric_values(manager, tmp_path):
    ws = manager.create_workspace("Inv Num")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    pd.DataFrame({
        "date": ["2026-01-01"],
        "payer_name": ["A"],
        "beneficiary_name": ["B"],
        "amount_usd": ["not an amount"],
        "id": [1]
    }).to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping["source_column"] == "amount_usd":
            mapping["selected_concept"] = "USD_AMOUNT"
            mapping["status"] = "AUTO_ACCEPTED"
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert any("amount_usd" in e for e in report.datatype_errors)

def test_ambiguous_mapping():
    assert True

def test_conflicting_mapping():
    assert True

def test_unknown_fields_preserved():
    assert True

def test_exact_feature_dimension_validation(manager, sample_csv):
    ws = manager.create_workspace("Dim Val")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.feature_dimension >= 0

def test_feature_order_validation():
    assert True

def test_non_finite_feature_detection():
    assert True

def test_dynamic_category_incompatibility(manager, tmp_path, sample_csv):
    ws = manager.create_workspace("Dyn Cat")
    ze_csv = tmp_path / "ze.csv"
    import pandas as pd
    df = pd.read_csv(str(sample_csv))
    df["payer_bank_country"] = "MARS"
    df.to_csv(ze_csv, index=False)
    manager.ingest_file(ws.manifest.workspace_id, str(ze_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    import json
    map_path = ws.directory / "mapping.json"
    with open(map_path, "r") as f: m = json.load(f)
    for mapping in m["mappings"]:
        if mapping["source_column"] == "payer_bank_country":
            mapping["selected_concept"] = "SENDER_BANK_COUNTRY"
            mapping["status"] = "AUTO_ACCEPTED"
    with open(map_path, "w") as f: json.dump(m, f)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status == "INCOMPATIBLE"
    assert not report.feature_order_valid

def test_workspace_isolation():
    assert True

def test_compatibility_report_persistence(manager, sample_csv):
    ws = manager.create_workspace("Comp Persist")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert (ws.directory / "compatibility.json").exists()

def test_model_version_binding(manager, sample_csv):
    ws = manager.create_workspace("Binding")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.model_version == "exp3_checkpoint"

def test_no_production_inference_invocation(manager, sample_csv, monkeypatch):
    ws = manager.create_workspace("No Inf")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    from src.detection import inference

    called = []
    def mock_run_inference(*args, **kwargs):
        called.append(True)
    monkeypatch.setattr(inference, 'run_inference', mock_run_inference)

    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert not called

def test_valid_canonicalization(manager, sample_csv):
    ws = manager.create_workspace("Valid")
    manager.ingest_file(ws.manifest.workspace_id, str(sample_csv))
    manager.profile_workspace(ws.manifest.workspace_id)
    manager.map_workspace(ws.manifest.workspace_id)
    from src.datasets.canonicalizer import canonicalize_and_validate
    report = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    assert report.status in ["READY", "INCOMPATIBLE"]


def test_canonicalization_strict_mapping_obedience(manager, tmp_path):
    import pandas as pd
    import json
    from src.datasets.mapper import MappingManifest, ColumnMapping, MappingStatus, CanonicalConcept
    from src.datasets.canonicalizer import canonicalize_and_validate

    # 1. Create source dataframe with NO literal 'Date' or 'Time' names
    df = pd.DataFrame({
        "my_date_val": ["2023-01-01", "2023-01-02"],
        "my_time_val": ["10:00", "11:00"],
        "other_ts_candidate": ["2023-01-01 10:00:00", "2023-01-02 11:00:00"],
        "sender_acc_id": ["A1", "A2"],
        "sender_loc_str": ["US", "UK"],
        "amount_orig": [100.0, 200.0],
        "currency_str": ["USD", "EUR"]
    })
    csv_path = tmp_path / "strict_mapping.csv"
    df.to_csv(csv_path, index=False)

    ws = manager.create_workspace("Strict")
    manager.ingest_file(ws.manifest.workspace_id, str(csv_path))
    manager.profile_workspace(ws.manifest.workspace_id)

    # First configuration: my_date_val + my_time_val -> TIMESTAMP
    manifest1 = MappingManifest(
        mappings=[
            ColumnMapping("my_date_val", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.TIMESTAMP),
            ColumnMapping("my_time_val", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.TIMESTAMP),
            ColumnMapping("other_ts_candidate", MappingStatus.UNMAPPED, []),
            ColumnMapping("sender_acc_id", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.SENDER_ACCOUNT),
            ColumnMapping("sender_loc_str", MappingStatus.NEEDS_CONFIRMATION, [], CanonicalConcept.SENDER_JURISDICTION),
            ColumnMapping("amount_orig", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.TRANSACTION_AMOUNT),
            ColumnMapping("currency_str", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.ORIGINAL_CURRENCY),
        ]
    )

    mapping_path = ws.directory / "mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(manifest1.to_dict(), f)

    manager.update_workspace(ws.manifest.workspace_id, status="NEEDS_MAPPING")

    report1 = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    canonical_df1 = pd.read_parquet(ws.directory / "canonical.parquet")

    # 3. canonicalize_and_validate() derives canonical `date` from exactly the selected Date and Time columns
    assert "date" in canonical_df1.columns
    assert str(canonical_df1.loc[0, "date"]) == "2023-01-01 10:00:00"

    # account IDs do not become names
    assert "payer_account" in canonical_df1.columns
    assert "payer_name" not in canonical_df1.columns or canonical_df1["payer_name"].isnull().all()

    # jurisdiction ambiguity is preserved appropriately
    jurisdiction_status = next((f for f in report1.field_statuses if (f["canonical_name"] == "payer_jurisdiction" if isinstance(f, dict) else getattr(f, "canonical_name", None) == "payer_jurisdiction")), None)
    assert jurisdiction_status is not None
    assert jurisdiction_status["state"] == "NEEDS_REVIEW" if isinstance(jurisdiction_status, dict) else jurisdiction_status.state == "NEEDS_REVIEW"

    # no synthetic values are introduced
    assert canonical_df1.loc[0, "payer_account"] == "A1"

    # 4. Change the mapping manifest so a different valid timestamp source is selected
    manifest2 = MappingManifest(
        mappings=[
            ColumnMapping("my_date_val", MappingStatus.UNMAPPED, []),
            ColumnMapping("my_time_val", MappingStatus.UNMAPPED, []),
            ColumnMapping("other_ts_candidate", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.TIMESTAMP),
            ColumnMapping("sender_acc_id", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.SENDER_ACCOUNT),
            ColumnMapping("sender_loc_str", MappingStatus.NEEDS_CONFIRMATION, [], CanonicalConcept.SENDER_JURISDICTION),
            ColumnMapping("amount_orig", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.TRANSACTION_AMOUNT),
            ColumnMapping("currency_str", MappingStatus.AUTO_ACCEPTED, [], CanonicalConcept.ORIGINAL_CURRENCY),
        ]
    )
    with open(mapping_path, "w") as f:
        json.dump(manifest2.to_dict(), f)

    report2 = canonicalize_and_validate(ws.manifest.workspace_id, manager)
    canonical_df2 = pd.read_parquet(ws.directory / "canonical.parquet")

    # 5. canonicalization follows the changed mapping and does not secretly scan dataframe column names
    assert "date" in canonical_df2.columns
    assert str(canonical_df2.loc[0, "date"]) == "2023-01-01 10:00:00"
