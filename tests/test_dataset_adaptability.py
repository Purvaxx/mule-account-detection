"""
Universal dataset adaptability
==============================

These fixtures are deliberately *not* copies of any dataset the repository has
seen.  Each one expresses a legitimate AML/payments schema in a different
dialect, and the point of the suite is to demonstrate that the mapper reaches
the right canonical concepts from **semantics** — role, concept family, value
type, cardinality, and the relationship between columns — rather than by
recognising a finite list of column names.

The three outcomes under test are the ones an operator has to act on:

``FULL``      every critical inference input available, no optional signal absent
``DEGRADED``  optional signals missing; inference remains mathematically valid
``BLOCKED``   genuinely required information unavailable

Nothing here relaxes the frozen contract: every dataset that reaches inference
still produces exactly 167 features in the checkpoint's order.
"""

import tempfile

import pandas as pd
import pytest

from src.datasets.canonicalizer import (
    READINESS_BLOCKED,
    READINESS_DEGRADED,
    canonicalize_and_validate,
)
from src.datasets.manifest import CurrencyPolicy, GeographyPolicy, IdentitySource
from src.datasets.mapper import CanonicalConcept, MappingManifest, MappingStatus
from src.datasets.workspace_manager import WorkspaceManager


@pytest.fixture
def manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield WorkspaceManager(base_dir=temp_dir)


def _prepare(manager, tmp_path, name, df, **config):
    """Upload a frame through the real workspace path and map it automatically.

    Returns ``(workspace_id, {source_column: ColumnMapping})``.  No mapping
    override is ever applied here: every assertion below is about what the
    automatic mapper proposed on its own.
    """
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)

    ws = manager.create_workspace(name)
    wid = ws.manifest.workspace_id
    manager.ingest_file(wid, str(path))
    manager.profile_workspace(wid)
    manager.map_workspace(wid)

    if config:
        current = manager.get_workspace(wid).manifest.dataset_config
        for key, value in config.items():
            setattr(current, key, value)
        manager.update_workspace(wid, dataset_config=current)

    manifest = MappingManifest.from_dict(manager.get_mapping(wid))
    return wid, {m.source_column: m for m in manifest.mappings}


#: Rows per fixture. Enough for the profiler's cardinality and coverage
#: statistics to be meaningful, and for a graph with repeated counterparties.
ROWS = 40


# ---------------------------------------------------------------------------
# Fixture A — party words with no qualifier
# ---------------------------------------------------------------------------

def _fixture_a():
    n = ROWS
    return pd.DataFrame({
        "payer": [f"Acme Holdings {i % 7}" for i in range(n)],
        "payee": [f"Northwind Ltd {i % 5}" for i in range(n)],
        "transaction_amount": [100.0 + i for i in range(n)],
        "transaction_timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
    })


def test_fixture_a_unqualified_party_words_reach_identity(manager, tmp_path):
    """`payer` / `payee` name a party without saying which attribute they hold.

    Both identity readings must be on the table, and the *values* — strings of
    labels, not identifiers — must select the name reading.  Neither column
    name appears in any alias table.
    """
    _wid, mappings = _prepare(manager, tmp_path, "fixture_a", _fixture_a())

    assert mappings["payer"].selected_concept == CanonicalConcept.SENDER_NAME
    assert mappings["payee"].selected_concept == CanonicalConcept.RECEIVER_NAME

    # The account reading is retained as a candidate, not discarded: which
    # attribute an unqualified party word holds is genuinely ambiguous.
    payer_candidates = [c.canonical_concept for c in mappings["payer"].candidates]
    assert CanonicalConcept.SENDER_ACCOUNT in payer_candidates
    assert mappings["payer"].status == MappingStatus.NEEDS_CONFIRMATION

    assert mappings["transaction_amount"].selected_concept == \
        CanonicalConcept.TRANSACTION_AMOUNT
    assert mappings["transaction_timestamp"].selected_concept == \
        CanonicalConcept.TIMESTAMP


def test_fixture_a_reaches_inference_under_name_identity(manager, tmp_path):
    wid, _ = _prepare(manager, tmp_path, "fixture_a_run", _fixture_a(),
                      identity_source=IdentitySource.NAME,
                      currency_policy=CurrencyPolicy.NATIVE_USD)
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid
    # No entity type and no geography were supplied, which is a real reduction
    # in signal and must be reported as one.
    assert report.readiness == READINESS_DEGRADED
    assert "payer_type" in report.degraded_features


def test_fixture_a_party_column_of_identifiers_reads_as_an_account(manager, tmp_path):
    """The same column name, holding integers, is an account rather than a name.

    This is the evidence dimension the fixture above pins from the other side:
    the decision comes from the values, not from the spelling.
    """
    n = ROWS
    df = pd.DataFrame({
        "payer": [900000 + (i % 9) for i in range(n)],
        "payee": [800000 + (i % 6) for i in range(n)],
        "amount": [10.0 * i for i in range(n)],
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
    })
    _wid, mappings = _prepare(manager, tmp_path, "fixture_a_ids", df)

    assert mappings["payer"].selected_concept == CanonicalConcept.SENDER_ACCOUNT
    assert mappings["payee"].selected_concept == CanonicalConcept.RECEIVER_ACCOUNT


# ---------------------------------------------------------------------------
# Fixture B — a split instant
# ---------------------------------------------------------------------------

def _fixture_b():
    n = ROWS
    stamps = pd.date_range("2023-03-01 08:00:00", periods=n, freq="37min")
    return pd.DataFrame({
        "sender_account": [f"ACC{i % 8:03d}" for i in range(n)],
        "receiver_account": [f"ACC{(i + 3) % 8:03d}" for i in range(n)],
        "amount": [250.0 + i for i in range(n)],
        "date": stamps.strftime("%Y-%m-%d"),
        "time": stamps.strftime("%H:%M:%S"),
    })


def test_fixture_b_date_and_time_compose_rather_than_conflict(manager, tmp_path):
    """A calendar column and a clock column are halves of one fact.

    Two columns claiming TIMESTAMP is normally a conflict.  When one names the
    date part and the other the time part, they are complementary: both keep
    the concept and canonicalization composes them.
    """
    _wid, mappings = _prepare(manager, tmp_path, "fixture_b", _fixture_b())

    assert mappings["date"].selected_concept == CanonicalConcept.TIMESTAMP
    assert mappings["time"].selected_concept == CanonicalConcept.TIMESTAMP
    assert mappings["date"].status != MappingStatus.CONFLICT
    assert mappings["time"].status != MappingStatus.CONFLICT


def test_fixture_b_composed_instant_is_a_real_timestamp(manager, tmp_path):
    wid, _ = _prepare(manager, tmp_path, "fixture_b_run", _fixture_b(),
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD)
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    # Not a midnight-truncated date and not a string paste: the clock half is
    # actually carried into the instant.
    assert canonical["date"].notna().all()
    assert canonical["date"].dt.hour.nunique() > 1

    date_status = next(f for f in report.field_statuses
                       if f["canonical_name"] == "date")
    assert date_status["state"] == "DERIVABLE"
    assert "calendar" in date_status["rule"] and "clock" in date_status["rule"]
    assert sorted(date_status["source_columns"]) == ["date", "time"]


def test_a_lone_calendar_column_is_still_a_direct_mapping(manager, tmp_path):
    """Composition is for a *pair*. One date column maps straight through."""
    n = ROWS
    df = pd.DataFrame({
        "sender_account": [f"A{i % 5}" for i in range(n)],
        "receiver_account": [f"B{i % 4}" for i in range(n)],
        "amount": [5.0 * i for i in range(n)],
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    })
    wid, mappings = _prepare(manager, tmp_path, "lone_date", df,
                             identity_source=IdentitySource.ACCOUNT_ID,
                             currency_policy=CurrencyPolicy.NATIVE_USD)
    assert mappings["date"].status == MappingStatus.AUTO_ACCEPTED
    report = canonicalize_and_validate(wid, manager)
    date_status = next(f for f in report.field_statuses
                       if f["canonical_name"] == "date")
    assert date_status["state"] == "AVAILABLE"


# ---------------------------------------------------------------------------
# Fixture C — role words qualified by an id token
# ---------------------------------------------------------------------------

def _fixture_c():
    n = ROWS
    return pd.DataFrame({
        "originator_id": [f"O{i % 9}" for i in range(n)],
        "beneficiary_id": [f"B{i % 6}" for i in range(n)],
        "value": [75.0 + i for i in range(n)],
        "currency": ["USD"] * n,
        "event_time": pd.date_range("2023-05-01", periods=n, freq="2h"),
        "country": ["GB"] * n,
    })


def test_fixture_c_role_plus_id_identifies_the_party_not_the_transaction(
        manager, tmp_path):
    """`originator_id` identifies the originator, not the transaction.

    Regression: `id` matched TRANSACTION_ID by substring, so an entire schema
    dialect's two identity columns were read as the transaction's own id and
    the dataset had no sender or receiver at all.
    """
    _wid, mappings = _prepare(manager, tmp_path, "fixture_c", _fixture_c())

    assert mappings["originator_id"].selected_concept == \
        CanonicalConcept.SENDER_ACCOUNT
    assert mappings["beneficiary_id"].selected_concept == \
        CanonicalConcept.RECEIVER_ACCOUNT
    for column in ("originator_id", "beneficiary_id"):
        concepts = [c.canonical_concept for c in mappings[column].candidates]
        assert CanonicalConcept.TRANSACTION_ID not in concepts

    assert mappings["value"].selected_concept == CanonicalConcept.TRANSACTION_AMOUNT
    assert mappings["currency"].selected_concept == CanonicalConcept.ORIGINAL_CURRENCY
    assert mappings["event_time"].selected_concept == CanonicalConcept.TIMESTAMP


def test_fixture_c_side_agnostic_country_is_never_assigned_a_side(
        manager, tmp_path):
    """A bare `country` column does not say whose country it is.

    Both sides score identically, so neither is selected. Copying one generic
    location into both sender and receiver geography would assert a fact the
    dataset does not contain.
    """
    _wid, mappings = _prepare(manager, tmp_path, "fixture_c_geo", _fixture_c())

    country = mappings["country"]
    assert country.selected_concept is None
    assert country.status == MappingStatus.NEEDS_CONFIRMATION
    concepts = [c.canonical_concept for c in country.candidates]
    assert CanonicalConcept.SENDER_JURISDICTION in concepts
    assert CanonicalConcept.RECEIVER_JURISDICTION in concepts


def test_fixture_c_runs_under_account_identity(manager, tmp_path):
    wid, _ = _prepare(manager, tmp_path, "fixture_c_run", _fixture_c(),
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD)
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    # Account identifiers are never rewritten as names.
    assert canonical["payer_account"].iloc[0] == "O0"
    assert canonical["payer_name"].isna().all()


# ---------------------------------------------------------------------------
# Fixture D — names that appear in no alias table at all
# ---------------------------------------------------------------------------

def test_fixture_d_arbitrary_names_with_valid_semantic_structure(
        manager, tmp_path):
    """Nothing here is an alias; every mapping comes from composition.

    `remitting_party_identifier`, `creditor_wallet`, `booking_day` and
    `ordering_institution_country` are invented spellings. Each is reached by
    composing a role token with a concept-family token, which is what makes the
    mapper adaptive rather than a lookup table.
    """
    n = ROWS
    df = pd.DataFrame({
        "remitting_party_identifier": [f"R{i % 7}" for i in range(n)],
        "creditor_wallet": [f"C{i % 5}" for i in range(n)],
        "settled_value": [42.0 + i for i in range(n)],
        "booking_day": pd.date_range("2023-07-01", periods=n, freq="D"),
        "ordering_institution_country": ["DE"] * n,
        "creditor_institution_country": ["FR"] * n,
        "remitter_classification": ["Company"] * n,
        "creditor_classification": ["Person"] * n,
    })
    wid, mappings = _prepare(manager, tmp_path, "fixture_d", df,
                             identity_source=IdentitySource.ACCOUNT_ID,
                             currency_policy=CurrencyPolicy.NATIVE_USD,
                             geography_policy=(
                                 GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE))

    assert mappings["remitting_party_identifier"].selected_concept == \
        CanonicalConcept.SENDER_ACCOUNT
    assert mappings["creditor_wallet"].selected_concept == \
        CanonicalConcept.RECEIVER_ACCOUNT
    assert mappings["booking_day"].selected_concept == CanonicalConcept.TIMESTAMP
    assert mappings["ordering_institution_country"].selected_concept == \
        CanonicalConcept.SENDER_BANK_COUNTRY
    assert mappings["creditor_institution_country"].selected_concept == \
        CanonicalConcept.RECEIVER_BANK_COUNTRY
    assert mappings["remitter_classification"].selected_concept == \
        CanonicalConcept.SENDER_TYPE
    assert mappings["creditor_classification"].selected_concept == \
        CanonicalConcept.RECEIVER_TYPE

    report = canonicalize_and_validate(wid, manager)
    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid
    # Every optional signal supplied and resolvable: nothing degraded.
    assert report.degraded_features == []
    assert report.readiness == "FULL"


# ---------------------------------------------------------------------------
# Fixture E — no entity type
# ---------------------------------------------------------------------------

def test_fixture_e_missing_entity_type_degrades_and_says_so(manager, tmp_path):
    """An absent optional signal is a degradation, not a failure — and is named."""
    n = ROWS
    df = pd.DataFrame({
        "sender_account": [f"S{i % 6}" for i in range(n)],
        "receiver_account": [f"T{i % 4}" for i in range(n)],
        "amount": [12.0 * (i + 1) for i in range(n)],
        "timestamp": pd.date_range("2023-02-01", periods=n, freq="h"),
        "sender_bank_country": ["GB"] * n,
        "receiver_bank_country": ["ES"] * n,
    })
    wid, _ = _prepare(manager, tmp_path, "fixture_e", df,
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD,
                      geography_policy=(
                          GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE))
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert report.readiness == READINESS_DEGRADED
    assert report.degraded_features == ["beneficiary_type", "payer_type"]
    assert any("payer_type" in reason for reason in report.readiness_reasons)

    # Degraded, not diminished: the vector is still exactly the right width.
    assert report.feature_dimension == 167
    assert report.feature_order_valid

    type_status = next(f for f in report.field_statuses
                       if f["canonical_name"] == "payer_type")
    assert type_status["state"] == "NOT_SUPPLIED"
    assert type_status["requirement"] == "OPTIONAL_SIGNAL"


# ---------------------------------------------------------------------------
# Fixture F — no geography
# ---------------------------------------------------------------------------

def test_fixture_f_missing_geography_degrades_and_is_never_filled_in(
        manager, tmp_path):
    """No country is invented, and the empty block is stated rather than implied."""
    n = ROWS
    df = pd.DataFrame({
        "sender_account": [f"S{i % 6}" for i in range(n)],
        "receiver_account": [f"T{i % 4}" for i in range(n)],
        "amount": [30.0 + i for i in range(n)],
        "timestamp": pd.date_range("2023-04-01", periods=n, freq="h"),
        "sender_entity_type": ["Company"] * n,
        "receiver_entity_type": ["Person"] * n,
    })
    wid, _ = _prepare(manager, tmp_path, "fixture_f", df,
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD)
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert report.readiness == READINESS_DEGRADED
    assert report.degraded_features == [
        "beneficiary_bank_country", "payer_bank_country"]
    assert report.feature_dimension == 167

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    # Absence stays absence: not "US", not a bucket, not the other side's value.
    assert canonical["payer_bank_country"].isna().all()
    assert canonical["beneficiary_bank_country"].isna().all()


# ---------------------------------------------------------------------------
# Fixture G — no counterparty identity at all
# ---------------------------------------------------------------------------

def test_fixture_g_missing_receiver_identity_is_blocked(manager, tmp_path):
    """Without one side of every edge there is no graph, and no honest fallback."""
    n = ROWS
    df = pd.DataFrame({
        "sender_account": [f"S{i % 6}" for i in range(n)],
        "amount": [15.0 + i for i in range(n)],
        "timestamp": pd.date_range("2023-06-01", periods=n, freq="h"),
        "narrative": [f"payment {i}" for i in range(n)],
    })
    wid, _ = _prepare(manager, tmp_path, "fixture_g", df,
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD)
    report = canonicalize_and_validate(wid, manager)

    assert report.status != "READY"
    assert report.readiness == READINESS_BLOCKED
    assert "beneficiary_account" in report.missing_required_fields
    assert report.readiness_reasons
    assert any("beneficiary_account" in r for r in report.readiness_reasons)


def test_fixture_g_missing_sender_identity_is_blocked(manager, tmp_path):
    n = ROWS
    df = pd.DataFrame({
        "beneficiary_account": [f"B{i % 6}" for i in range(n)],
        "amount": [15.0 + i for i in range(n)],
        "timestamp": pd.date_range("2023-06-01", periods=n, freq="h"),
    })
    wid, _ = _prepare(manager, tmp_path, "fixture_g_sender", df,
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD)
    report = canonicalize_and_validate(wid, manager)

    assert report.status != "READY"
    assert report.readiness == READINESS_BLOCKED
    assert "payer_account" in report.missing_required_fields


# ---------------------------------------------------------------------------
# Cross-cutting: the mapper is not matching a list of dataset names
# ---------------------------------------------------------------------------

def test_no_fixture_column_name_appears_in_the_alias_tables(manager, tmp_path):
    """The names the fixtures above rely on are not enumerated anywhere.

    If any of these were an alias, the corresponding fixture would prove
    nothing about adaptability. This asserts the fixtures are actually testing
    composition.
    """
    from src.datasets.mapper import (
        GENERIC_ALIASES, SPECIFIC_ALIASES, normalize_for_mapping,
    )

    enumerated = set(GENERIC_ALIASES)
    for phrases in SPECIFIC_ALIASES.values():
        enumerated.update(phrases)

    composed_only = [
        "payee", "originator_id", "beneficiary_id",
        "remitting_party_identifier", "creditor_wallet", "booking_day",
        "ordering_institution_country", "creditor_institution_country",
        "remitter_classification", "creditor_classification",
        "sender_bank_country", "receiver_bank_country",
        "sender_entity_type", "receiver_entity_type", "target_account",
    ]
    for name in composed_only:
        assert normalize_for_mapping(name) not in enumerated, name


# ---------------------------------------------------------------------------
# Derivation: the rules, tested without building a workspace
# ---------------------------------------------------------------------------

def test_compose_instant_reads_every_shape_a_clock_column_takes():
    """A spreadsheet writes a time column in at least three different shapes."""
    import datetime

    from src.datasets.derivation import compose_instant

    dates = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])

    seconds = compose_instant(dates, pd.Series(["10:35:19", "23:00:01", "00:00:00"]))
    assert list(seconds.dt.hour) == [10, 23, 0]

    minutes = compose_instant(dates, pd.Series(["10:00", "23:15", "06:45"]))
    assert list(minutes.dt.hour) == [10, 23, 6]
    assert list(minutes.dt.minute) == [0, 15, 45]

    objects = compose_instant(dates, pd.Series([
        datetime.time(1, 2, 3), datetime.time(14, 0, 0), datetime.time(9, 30, 0)]))
    assert list(objects.dt.hour) == [1, 14, 9]

    # Excel's own representation: a time on a placeholder date.
    excel = compose_instant(dates, pd.Series(pd.to_datetime(
        ["1899-12-30 08:15:00", "1899-12-30 17:45:00", "1899-12-30 12:00:00"])))
    assert list(excel.dt.hour) == [8, 17, 12]
    assert list(excel.dt.year) == [2023, 2023, 2023]


def test_compose_instant_never_invents_a_date_and_never_drops_one():
    from src.datasets.derivation import compose_instant

    composed = compose_instant(
        pd.Series(["2023-01-01", None, "2023-01-03"]),
        pd.Series(["10:00:00", "10:00:00", "not a time"]),
    )
    # A missing date has no instant to be placed at.
    assert pd.isna(composed.iloc[1])
    # A missing time does not discard the calendar fact that was supplied.
    assert composed.iloc[2] == pd.Timestamp("2023-01-03 00:00:00")


@pytest.mark.parametrize("columns", [
    ["transaction_date"],
    ["value_date", "booking_date"],
    ["event_time", "posted_time"],
    ["transaction_datetime", "event_time"],
    ["date", "time", "timestamp"],
])
def test_instant_pair_is_only_one_calendar_and_one_clock(columns):
    """Composition is not a general merge: anything else is left to arbitration."""
    from src.datasets.derivation import instant_pair
    assert instant_pair(columns) is None


# ---------------------------------------------------------------------------
# Currency: a declaration is checked against the file, or refused
# ---------------------------------------------------------------------------

def test_ambiguous_currency_column_blocks_a_native_usd_declaration(
        manager, tmp_path):
    """Two tied currency columns are unresolved evidence, not absent evidence.

    Regression: an unresolved ORIGINAL_CURRENCY claim left the concept unmapped,
    the contradiction check found nothing to look at and passed, and a
    multi-currency book reached READY with euros recorded as amount_usd.
    """
    n = 60
    df = pd.DataFrame({
        "sender_account": [f"S{i % 5}" for i in range(n)],
        "receiver_account": [f"R{i % 4}" for i in range(n)],
        "amount": [100.0 + i for i in range(n)],
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "settlement_ccy": ["EUR" if i % 2 else "USD" for i in range(n)],
        "instructed_ccy": ["EUR" if i % 2 else "USD" for i in range(n)],
    })
    wid, mappings = _prepare(manager, tmp_path, "ambiguous_ccy", df,
                             identity_source=IdentitySource.ACCOUNT_ID,
                             currency_policy=CurrencyPolicy.NATIVE_USD)

    assert mappings["settlement_ccy"].status == MappingStatus.CONFLICT
    assert mappings["instructed_ccy"].status == MappingStatus.CONFLICT

    report = canonicalize_and_validate(wid, manager)
    assert report.currency_status == "INCOMPATIBLE"
    assert report.status == "INCOMPATIBLE"
    assert report.readiness == READINESS_BLOCKED

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    assert canonical["amount_usd"].isna().all()


def test_an_unstated_currency_reports_what_the_file_says_without_deciding(
        manager, tmp_path):
    """Evidence is surfaced; the declaration stays the operator's to make."""
    n = 60
    df = pd.DataFrame({
        "sender_account": [f"S{i % 5}" for i in range(n)],
        "receiver_account": [f"R{i % 4}" for i in range(n)],
        "amount": [100.0 + i for i in range(n)],
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "currency": ["USD"] * n,
    })
    wid, _ = _prepare(manager, tmp_path, "unstated_ccy", df,
                      identity_source=IdentitySource.ACCOUNT_ID)
    report = canonicalize_and_validate(wid, manager)

    currency = report.configuration_assessment["currency"]
    assert currency["selection"] == "UNKNOWN"
    assert currency["status"] == "NEEDS_REVIEW"
    assert currency["evidence"]["supports_native_usd"] is True
    assert "would be supported by the file itself" in currency["reason"]

    # Reported, not acted on: no amount_usd was produced from an undeclared book.
    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    assert canonical["amount_usd"].isna().all()


# ---------------------------------------------------------------------------
# Geography: a country the checkpoint has no column for
# ---------------------------------------------------------------------------

def test_a_recognised_country_outside_the_vocabulary_is_named_not_guessed(
        manager, tmp_path):
    """`United States` is a country. The checkpoint has no `US` column.

    Those are two different facts and the report states both, rather than
    filing the value alongside genuinely unreadable data — and rather than
    substituting a nearby code that would place the transaction elsewhere.
    """
    n = 60
    df = pd.DataFrame({
        "sender_account": [f"S{i % 5}" for i in range(n)],
        "receiver_account": [f"R{i % 4}" for i in range(n)],
        "amount": [100.0 + i for i in range(n)],
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "sender_bank_country": (["United States"] * (n // 2)
                                + ["Spain"] * (n - n // 2)),
        "receiver_bank_country": ["Atlantis"] * (n // 2) + ["GB"] * (n - n // 2),
    })
    wid, _ = _prepare(manager, tmp_path, "oov_geo", df,
                      identity_source=IdentitySource.ACCOUNT_ID,
                      currency_policy=CurrencyPolicy.NATIVE_USD,
                      geography_policy=(
                          GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE))
    report = canonicalize_and_validate(wid, manager)

    payer = next(g for g in report.geography_normalization
                 if g["column"] == "payer_bank_country")
    assert payer["recognized_out_of_vocabulary"] == {"United States": "US"}
    assert payer["unresolved_values"] == ["United States"]

    beneficiary = next(g for g in report.geography_normalization
                       if g["column"] == "beneficiary_bank_country")
    assert beneficiary["recognized_out_of_vocabulary"] == {}
    assert beneficiary["unresolved_values"] == ["Atlantis"]

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    values = set(canonical["payer_bank_country"].dropna())
    # Resolved values are rewritten; the unresolvable one is left exactly as
    # supplied and is never turned into a code.
    assert values == {"United States", "ES"}
