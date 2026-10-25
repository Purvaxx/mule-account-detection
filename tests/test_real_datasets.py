"""
Real, unmodified external datasets
==================================

The fixtures in ``test_dataset_adaptability.py`` are constructed to isolate one
semantic question each.  These are not: they are three real financial-crime
datasets, each pushed through the actual workspace path — upload, profile,
automatic mapping, canonicalization, compatibility validation — with **no
mapping override and no enrichment**.

Nothing here forces a dataset to ``READY``.  Two of the three are genuinely
blocked, for a reason the repository can state precisely, and the tests assert
the reason rather than the absence of one.

The files are runtime data (``.workspaces/`` is git-ignored, and the Reference
Dataset's source workbook lives above the repository), so every test skips when
its dataset is not present, following the convention in ``tests/conftest.py``.

**Bounded reads.** Each dataset is read to a row bound rather than in full.  A
bound truncates; it does not alter — no value is changed, no column is added or
removed by it, and the schema under test is the file's own.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.canonicalizer import (
    READINESS_BLOCKED,
    canonicalize_and_validate,
)
from src.datasets.manifest import CurrencyPolicy, GeographyPolicy, IdentitySource
from src.datasets.mapper import CanonicalConcept, MappingManifest
from src.datasets.workspace_manager import WorkspaceManager

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_WORKSPACES = _REPO_ROOT / ".workspaces"

#: The dataset fixtures below are session-scoped: each workbook is tens of
#: megabytes and reading it once per test dominates the suite's runtime. They
#: are read-only — every test writes its own CSV copy into its own workspace.

#: Rows read from each real dataset. Large enough that cardinality, currency
#: mix and country coverage are all exercised; small enough to stay a test.
ROW_BOUND = 20000

#: SAML-D as published: twelve columns, a separate calendar and clock column,
#: no USD amount and no entity type.  The copy in this repository's runtime
#: workspaces was *enriched* with this project's own canonical columns
#: (``amount_usd``, ``payer_core``, ``transaction_datetime``, ...), so the raw
#: dataset is recovered by projecting onto the columns SAML-D actually ships.
#: That is de-enrichment: no value is modified and nothing is added.
SAMLD_PUBLISHED_COLUMNS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type", "Is_laundering",
    "Laundering_type",
]

IBM_AML_COLUMNS = [
    "payer_account", "beneficiary_account", "amount_orig", "currency_orig",
    "amount_dest", "currency_dest",
]

LAUNDROMAT_WORKBOOK = _REPO_ROOT.parent / "Azerbaijani_Laundromat_Raw_Data.xlsx"


def _find_runtime_upload(required_columns):
    """The runtime workspace whose upload has all of ``required_columns``.

    Located by reading each workspace's cheap ``profile.json`` rather than by
    hard-coding a workspace UUID, which is generated per upload.
    """
    if not _RUNTIME_WORKSPACES.exists():
        return None
    for profile_path in sorted(_RUNTIME_WORKSPACES.glob("*/profile.json")):
        try:
            with open(profile_path, "r", encoding="utf-8") as handle:
                profile = json.load(handle)
        except (OSError, ValueError):
            continue
        names = {c.get("original_name") for c in profile.get("columns", [])}
        if not set(required_columns) <= names:
            continue
        for upload in sorted(profile_path.parent.glob("raw_data.*")):
            return upload
    return None


#: The workspace store is session-scoped for the same reason the frames are:
#: every ``_run`` below creates its own workspace inside it under a fresh UUID,
#: so the tests remain isolated from one another while the store — and the
#: workspaces already validated in it — survive for the whole module.
@pytest.fixture(scope="session")
def manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield WorkspaceManager(base_dir=temp_dir)


#: Several tests below assert different properties of the *same* run: the three
#: SAML-D tests, and two of the three each for IBM AML and the Laundromat, drive
#: an identical (dataset, configuration) pair through the identical path. Each
#: such run costs 7-25 s, almost all of it in ``canonicalize_and_validate``.
#: The cache below is keyed on what actually determines the outcome — the
#: frame's schema and size, and the declared configuration — so a differing
#: declaration (``ibm_usd``'s NATIVE_USD, ``laundromat_labels``' geography) is a
#: different key and still runs on its own.
_RUN_CACHE: dict = {}


def _run(manager, tmp_path, name, df, **config):
    """Upload and validate through the real path, with no mapping override."""
    key = (
        tuple(str(c) for c in df.columns),
        len(df.index),
        tuple(sorted((k, str(v)) for k, v in config.items())),
    )
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]

    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)

    wid = manager.create_workspace(name).manifest.workspace_id
    manager.ingest_file(wid, str(path))
    manager.profile_workspace(wid)
    manager.map_workspace(wid)

    if config:
        current = manager.get_workspace(wid).manifest.dataset_config
        for key_name, value in config.items():
            setattr(current, key_name, value)
        manager.update_workspace(wid, dataset_config=current)

    mapping = MappingManifest.from_dict(manager.get_mapping(wid))
    report = canonicalize_and_validate(wid, manager)
    result = (wid, {m.source_column: m for m in mapping.mappings}, report)
    _RUN_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# SAML-D, as published
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def samld_raw():
    upload = _find_runtime_upload(SAMLD_PUBLISHED_COLUMNS)
    if upload is None:
        pytest.skip("SAML-D upload not present in .workspaces/")
    frame = pd.read_excel(upload, nrows=ROW_BOUND)
    return frame[SAMLD_PUBLISHED_COLUMNS].copy()


def test_samld_raw_is_mapped_entirely_without_operator_help(
        manager, tmp_path, samld_raw):
    """Every SAML-D column reaches its correct concept automatically.

    Including the split instant: SAML-D ships ``Date`` and ``Time`` as separate
    columns and no combined one, so the two must compose rather than annihilate
    each other as rival TIMESTAMP claimants.
    """
    _wid, mappings, _report = _run(
        manager, tmp_path, "samld_raw", samld_raw,
        identity_source=IdentitySource.ACCOUNT_ID,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)

    assert mappings["Sender_account"].selected_concept == \
        CanonicalConcept.SENDER_ACCOUNT
    assert mappings["Receiver_account"].selected_concept == \
        CanonicalConcept.RECEIVER_ACCOUNT
    assert mappings["Amount"].selected_concept == CanonicalConcept.TRANSACTION_AMOUNT
    assert mappings["Payment_currency"].selected_concept == \
        CanonicalConcept.ORIGINAL_CURRENCY
    assert mappings["Sender_bank_location"].selected_concept == \
        CanonicalConcept.SENDER_BANK_COUNTRY
    assert mappings["Receiver_bank_location"].selected_concept == \
        CanonicalConcept.RECEIVER_BANK_COUNTRY
    assert mappings["Date"].selected_concept == CanonicalConcept.TIMESTAMP
    assert mappings["Time"].selected_concept == CanonicalConcept.TIMESTAMP

    # `Received_currency` is what arrived, not what was sent, and the canonical
    # schema has no column for it. It must not beat `Payment_currency`.
    assert mappings["Received_currency"].selected_concept is None


def test_samld_raw_is_blocked_only_by_its_missing_usd_amount(
        manager, tmp_path, samld_raw):
    """The blocker is real: SAML-D amounts are in several currencies.

    Every structural requirement is satisfied — the graph builds and the feature
    matrix is exactly 167 wide in the checkpoint's order. What is missing is a
    USD amount, and producing one would need dated FX rates this repository does
    not have and will not invent.
    """
    _wid, _mappings, report = _run(
        manager, tmp_path, "samld_blocked", samld_raw,
        identity_source=IdentitySource.ACCOUNT_ID,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)

    assert report.status == "INCOMPATIBLE"
    assert report.readiness == READINESS_BLOCKED
    assert report.missing_required_fields == ["amount_usd"]
    assert report.identity_status == "READY"
    # The rest of the contract is genuinely satisfied.
    assert report.feature_dimension == 167
    assert report.feature_order_valid
    assert report.graph_requirements == []


def test_samld_country_names_are_resolved_and_the_gap_is_explained(
        manager, tmp_path, samld_raw):
    """SAML-D writes countries as names. Most resolve; the rest are explained.

    ``UK`` / ``Spain`` / ``Germany`` reach the frozen vocabulary. ``USA`` /
    ``Japan`` / ``Morocco`` do not — not because they are unreadable, but
    because the Experiment 3 checkpoint has no column for those countries. That
    distinction is now reported rather than both being filed as "unresolved".
    """
    _wid, _mappings, report = _run(
        manager, tmp_path, "samld_geo", samld_raw,
        identity_source=IdentitySource.ACCOUNT_ID,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)

    assert report.geography_normalization
    for column_report in report.geography_normalization:
        assert column_report["resolved_ratio"] > 0.9
        assert column_report["rules_applied"].get("ISO_NAME", 0) > 0
        recognized = column_report["recognized_out_of_vocabulary"]
        assert recognized, "USA/Japan/Morocco are countries, not unreadable data"
        assert recognized.get("USA") == "US"
        # Recognised is not resolved: nothing is substituted into the data.
        assert set(recognized) <= set(column_report["unresolved_values"])


# ---------------------------------------------------------------------------
# IBM AML
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ibm_aml_raw():
    upload = _find_runtime_upload(IBM_AML_COLUMNS)
    if upload is None:
        pytest.skip("IBM AML upload not present in .workspaces/")
    return pd.read_excel(upload, nrows=ROW_BOUND)


def test_ibm_aml_bank_identifiers_are_not_read_as_party_accounts(
        manager, tmp_path, ibm_aml_raw):
    """`payer_bank_account` identifies the bank, not the payer.

    It holds tens of thousands of distinct values. Reading it as the payer's own
    account would replace the entity population with banks, and reading it as a
    country would produce the same nonsense one level down.
    """
    _wid, mappings, _report = _run(
        manager, tmp_path, "ibm_banks", ibm_aml_raw,
        identity_source=IdentitySource.ACCOUNT_ID)

    assert mappings["payer_account"].selected_concept == \
        CanonicalConcept.SENDER_ACCOUNT
    assert mappings["beneficiary_account"].selected_concept == \
        CanonicalConcept.RECEIVER_ACCOUNT
    for column in ("payer_bank_account", "beneficiary_bank_account"):
        assert mappings[column].selected_concept is None

    # The sent currency wins over the received one, so the currency check below
    # has something to look at instead of two tied columns and a blind spot.
    assert mappings["currency_orig"].selected_concept == \
        CanonicalConcept.ORIGINAL_CURRENCY
    assert mappings["currency_dest"].selected_concept is None


def test_ibm_aml_native_usd_is_contradicted_by_its_own_currency_column(
        manager, tmp_path, ibm_aml_raw):
    """Declaring "already USD" over a multi-currency book is refused.

    The declaration is the operator's to make; whether the file supports it is
    not. Here it does not, so ``amount_usd`` is never produced by copying euros
    and yen into it.
    """
    wid, _mappings, report = _run(
        manager, tmp_path, "ibm_usd", ibm_aml_raw,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD)

    assert report.currency_status == "INCOMPATIBLE"
    assert report.status == "INCOMPATIBLE"
    assert report.readiness == READINESS_BLOCKED

    currency = report.configuration_assessment["currency"]
    assert currency["evidence"]["distinct_currencies"] > 1
    assert currency["evidence"]["usd_share"] < 0.999

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet")
    assert canonical["amount_usd"].isna().all(), "no amount was reinterpreted as USD"


def test_ibm_aml_geography_is_left_unspecified_rather_than_guessed(
        manager, tmp_path, ibm_aml_raw):
    """IBM AML has no country column, so it gets none — not a duplicated one."""
    _wid, _mappings, report = _run(
        manager, tmp_path, "ibm_geo", ibm_aml_raw,
        identity_source=IdentitySource.ACCOUNT_ID)

    assert "payer_bank_country" in report.degraded_features
    assert "beneficiary_bank_country" in report.degraded_features
    assert report.geography_normalization == []


# ---------------------------------------------------------------------------
# Azerbaijani Laundromat (OCCRP) — the Reference Dataset's own source workbook
# ---------------------------------------------------------------------------
# Included as a third real dataset with a genuinely different shape: named
# counterparties rather than account ids, an explicit USD amount column, and
# numeric bank-country codes.  It is *not* independent evidence of model
# validity — the checkpoint was trained on this data — but it is a real test of
# whether the external path can read a real schema, and of one specific failure:
# its raw workbook carries both `payer_name` and `payer_name_norm`, which tied
# for SENDER_NAME and annihilated each other, leaving the dataset with no
# counterparty identity at all.

@pytest.fixture(scope="session")
def laundromat_raw():
    if not LAUNDROMAT_WORKBOOK.exists():
        pytest.skip(f"Source workbook not available at {LAUNDROMAT_WORKBOOK}")
    return pd.read_excel(LAUNDROMAT_WORKBOOK, nrows=ROW_BOUND)


def test_laundromat_normalized_name_does_not_annihilate_the_name(
        manager, tmp_path, laundromat_raw):
    """A normalization of a fact ranks below the fact, rather than tying with it.

    `payer_name` and `payer_name_norm` both denote the payer's name. As equal
    claimants they were both discarded as a CONFLICT and the dataset reached
    validation with no identity. The plain column now wins, and the pipeline
    re-derives its own `payer_name_norm` under the entity-resolution rule the
    checkpoint was trained with.
    """
    _wid, mappings, report = _run(
        manager, tmp_path, "laundromat_names", laundromat_raw,
        identity_source=IdentitySource.NAME,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)

    assert mappings["payer_name"].selected_concept == CanonicalConcept.SENDER_NAME
    assert mappings["beneficiary_name"].selected_concept == \
        CanonicalConcept.RECEIVER_NAME
    assert mappings["payer_name_norm"].selected_concept is None
    assert mappings["beneficiary_name_norm"].selected_concept is None

    norm_status = next(f for f in report.field_statuses
                       if f["canonical_name"] == "payer_name_norm")
    assert norm_status["state"] == "DERIVABLE"
    assert norm_status["rule"] == "Lowercase and alphanumeric strip from payer_name"


def test_laundromat_reaches_the_full_contract_under_name_identity(
        manager, tmp_path, laundromat_raw):
    """A real named-counterparty schema runs end to end on the frozen contract."""
    _wid, _mappings, report = _run(
        manager, tmp_path, "laundromat_run", laundromat_raw,
        identity_source=IdentitySource.NAME,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)

    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid
    assert report.missing_required_fields == []
    assert report.degraded_features == []
    # Its own numeric country codes are already frozen-vocabulary members and
    # pass through untouched.
    for column_report in report.geography_normalization:
        assert column_report["normalized_examples"] == {}
        assert column_report["rules_applied"] == {"EXACT": column_report["distinct_raw"]}


def test_laundromat_labels_are_preserved_but_never_required(
        manager, tmp_path, laundromat_raw):
    """Ground truth is carried through and kept out of the feature path."""
    _wid, _mappings, report = _run(
        manager, tmp_path, "laundromat_labels", laundromat_raw,
        identity_source=IdentitySource.NAME,
        currency_policy=CurrencyPolicy.NATIVE_USD)

    assert set(report.labels_present) == {
        "payer_core", "beneficiary_core", "investigation"}
    for label in report.labels_present:
        assert label not in report.missing_required_fields
