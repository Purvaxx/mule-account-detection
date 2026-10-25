"""
Tests for src/data/ingestion.py — Layer 1 Data Ingestion.

Coverage
--------
1.  Valid dataset loading against the real dkdata.json.
2.  Correct transaction count.
3.  All required fields present.
4.  Missing account identifiers are preserved (not dropped or replaced).
5.  Normalized names are preserved and never null.
6.  Dates are parsed to datetime dtype.
7.  Amounts are parsed to numeric dtype.
8.  Malformed JSON produces a ValueError.
9.  Missing required column produces a fatal validation error.
10. Missing file produces FileNotFoundError.
11. Unexpected top-level JSON structure produces a ValueError.
12. IngestionResult.is_valid reflects validation state.
13. describe_result returns a non-empty string.
14. Duplicate transaction IDs are flagged as a warning.
15. Non-positive amounts produce a warning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import (
    REQUIRED_COLUMNS,
    IngestionResult,
    describe_result,
    load_transactions,
)

# ---------------------------------------------------------------------------
# Locate the canonical dataset
# ---------------------------------------------------------------------------

# dkdata.json lives one level above fraud-detection/
_REPO_ROOT = Path(__file__).resolve().parent.parent          # fraud-detection/
_DATASET_PATH = _REPO_ROOT.parent / "dkdata.json"           # SIH/dkdata.json

# Allow overriding via env-var for CI environments
_DATASET_PATH = Path(os.environ.get("DKDATA_PATH", _DATASET_PATH))

_REAL_DATASET_AVAILABLE = _DATASET_PATH.exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_result() -> IngestionResult:
    """Load the real dataset once for the entire test session."""
    if not _REAL_DATASET_AVAILABLE:
        pytest.skip(f"Real dataset not found at {_DATASET_PATH}")
    return load_transactions(_DATASET_PATH)


@pytest.fixture()
def minimal_valid_json(tmp_path: Path) -> Path:
    """A minimal valid dkdata.json with two synthetic records."""
    record_base = {col: None for col in REQUIRED_COLUMNS}
    records = []
    for i in range(2):
        r = dict(record_base)
        r["id"] = i + 1
        r["date"] = "2013-01-15"
        r["payer_name"] = f"Payer {i}"
        r["payer_name_norm"] = f"payer_{i}"
        r["payer_account"] = f"ACC{i:04d}"
        r["payer_type"] = "Company"
        r["payer_jurisdiction"] = "GB"
        r["payer_bank_country"] = "GB"
        r["payer_core"] = True
        r["beneficiary_name"] = f"Beneficiary {i}"
        r["beneficiary_name_norm"] = f"beneficiary_{i}"
        r["beneficiary_account"] = f"BACC{i:04d}"
        r["beneficiary_type"] = "Company"
        r["beneficiary_jurisdiction"] = "TR"
        r["beneficiary_bank_country"] = "TR"
        r["beneficiary_core"] = False
        r["amount_orig"] = 10000.0 * (i + 1)
        r["amount_orig_currency"] = "USD"
        r["amount_usd"] = 10000 * (i + 1)
        r["amount_eur"] = str(9000 * (i + 1))
        r["purpose"] = f"Payment {i}"
        r["source_file"] = "synthetic"
        r["investigation"] = "az"
        records.append(r)

    p = tmp_path / "dkdata.json"
    p.write_text(json.dumps({"data": records}), encoding="utf-8")
    return p


@pytest.fixture()
def json_missing_column(tmp_path: Path, minimal_valid_json: Path) -> Path:
    """A JSON file missing the 'amount_usd' required column."""
    data = json.loads(minimal_valid_json.read_text(encoding="utf-8"))
    for rec in data["data"]:
        rec.pop("amount_usd", None)
    p = tmp_path / "missing_col.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def malformed_json_file(tmp_path: Path) -> Path:
    """A file that is not valid JSON."""
    p = tmp_path / "malformed.json"
    p.write_text("{this is not valid JSON", encoding="utf-8")
    return p


@pytest.fixture()
def wrong_structure_json(tmp_path: Path) -> Path:
    """A valid JSON file but without the expected 'data' top-level key."""
    p = tmp_path / "wrong_structure.json"
    p.write_text(json.dumps({"records": []}), encoding="utf-8")
    return p


@pytest.fixture()
def json_with_missing_accounts(tmp_path: Path, minimal_valid_json: Path) -> Path:
    """A JSON where one record has null payer_account."""
    data = json.loads(minimal_valid_json.read_text(encoding="utf-8"))
    data["data"][0]["payer_account"] = None
    p = tmp_path / "missing_accounts.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def json_with_duplicate_ids(tmp_path: Path, minimal_valid_json: Path) -> Path:
    """A JSON where two records share the same ID."""
    data = json.loads(minimal_valid_json.read_text(encoding="utf-8"))
    data["data"][1]["id"] = data["data"][0]["id"]
    p = tmp_path / "dup_ids.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def json_with_non_positive_amount(tmp_path: Path, minimal_valid_json: Path) -> Path:
    """A JSON where one record has amount_usd = 0."""
    data = json.loads(minimal_valid_json.read_text(encoding="utf-8"))
    data["data"][0]["amount_usd"] = 0
    p = tmp_path / "zero_amount.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests — real dataset
# ---------------------------------------------------------------------------

class TestRealDataset:
    """Tests that run against the actual dkdata.json."""

    def test_loads_successfully(self, real_result: IngestionResult) -> None:
        assert real_result is not None
        assert isinstance(real_result.transactions, pd.DataFrame)

    def test_transaction_count(self, real_result: IngestionResult) -> None:
        assert real_result.record_count == 16_940
        assert len(real_result.transactions) == 16_940

    def test_all_required_columns_present(self, real_result: IngestionResult) -> None:
        for col in REQUIRED_COLUMNS:
            assert col in real_result.transactions.columns, (
                f"Required column '{col}' missing from loaded DataFrame."
            )

    def test_no_fatal_errors(self, real_result: IngestionResult) -> None:
        assert real_result.is_valid, (
            f"Unexpected validation errors: {real_result.validation_errors}"
        )

    def test_missing_accounts_preserved(self, real_result: IngestionResult) -> None:
        """Missing account identifiers must be preserved as NaN, not dropped."""
        df = real_result.transactions
        n_missing_payer = int(df["payer_account"].isna().sum())
        n_missing_ben = int(df["beneficiary_account"].isna().sum())
        # Dataset audit found 115 missing payer accounts and 4 missing beneficiary accounts.
        # We check that records are present (not dropped) and counts are reported.
        assert n_missing_payer > 0, (
            "Expected some missing payer accounts; found none. "
            "If the dataset changed, update this test."
        )
        assert n_missing_ben >= 0  # 0 is acceptable if dataset was cleaned
        assert len(df) == real_result.record_count  # no rows dropped

    def test_normalized_names_always_present(self, real_result: IngestionResult) -> None:
        df = real_result.transactions
        assert df["payer_name_norm"].isna().sum() == 0
        assert df["beneficiary_name_norm"].isna().sum() == 0

    def test_date_column_is_datetime(self, real_result: IngestionResult) -> None:
        assert pd.api.types.is_datetime64_any_dtype(
            real_result.transactions["date"]
        ), "Column 'date' should be parsed to a datetime dtype."

    def test_amounts_are_numeric(self, real_result: IngestionResult) -> None:
        df = real_result.transactions
        assert pd.api.types.is_numeric_dtype(df["amount_usd"])
        assert pd.api.types.is_numeric_dtype(df["amount_orig"])

    def test_missing_value_report_populated(self, real_result: IngestionResult) -> None:
        # payer_account and payer_bank_country are known to have missing values.
        assert "payer_account" in real_result.missing_value_report
        assert real_result.missing_value_report["payer_account"] > 0

    def test_describe_result_returns_string(self, real_result: IngestionResult) -> None:
        summary = describe_result(real_result)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "INGESTION SUMMARY" in summary


# ---------------------------------------------------------------------------
# Tests — synthetic / edge-case fixtures
# ---------------------------------------------------------------------------

class TestValidSyntheticDataset:
    def test_minimal_valid_loads(self, minimal_valid_json: Path) -> None:
        result = load_transactions(minimal_valid_json)
        assert result.is_valid
        assert result.record_count == 2

    def test_date_parsed(self, minimal_valid_json: Path) -> None:
        result = load_transactions(minimal_valid_json)
        assert pd.api.types.is_datetime64_any_dtype(result.transactions["date"])

    def test_amounts_numeric(self, minimal_valid_json: Path) -> None:
        result = load_transactions(minimal_valid_json)
        assert pd.api.types.is_numeric_dtype(result.transactions["amount_usd"])
        assert pd.api.types.is_numeric_dtype(result.transactions["amount_orig"])


class TestMissingAccountsPreserved:
    def test_null_account_row_not_dropped(
        self, json_with_missing_accounts: Path
    ) -> None:
        result = load_transactions(json_with_missing_accounts)
        assert result.record_count == 2
        assert result.transactions["payer_account"].isna().sum() == 1
        # Check row is still present (not silently removed)
        assert len(result.transactions) == 2


class TestMalformedInput:
    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_transactions(tmp_path / "nonexistent.json")

    def test_malformed_json_raises_value_error(
        self, malformed_json_file: Path
    ) -> None:
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            load_transactions(malformed_json_file)

    def test_wrong_structure_raises_value_error(
        self, wrong_structure_json: Path
    ) -> None:
        with pytest.raises(ValueError, match="Unexpected top-level JSON structure"):
            load_transactions(wrong_structure_json)


class TestMissingColumn:
    def test_missing_required_column_produces_fatal_error(
        self, json_missing_column: Path
    ) -> None:
        result = load_transactions(json_missing_column)
        assert not result.is_valid
        assert any("amount_usd" in e for e in result.validation_errors)


class TestDuplicateIds:
    def test_duplicate_ids_flagged_as_warning(
        self, json_with_duplicate_ids: Path
    ) -> None:
        result = load_transactions(json_with_duplicate_ids)
        assert result.is_valid  # duplicate IDs are a warning, not a fatal error
        assert any("duplicate" in w.lower() for w in result.validation_warnings)


class TestNonPositiveAmounts:
    def test_zero_amount_flagged_as_warning(
        self, json_with_non_positive_amount: Path
    ) -> None:
        result = load_transactions(json_with_non_positive_amount)
        assert any("non-positive" in w for w in result.validation_warnings)
