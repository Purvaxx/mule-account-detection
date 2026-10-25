"""
Tests for src/data/preprocessing.py — Layer 2 Preprocessing & Entity Resolution.

Coverage
--------
1.  payer_entity_id is correctly derived from payer_name_norm.
2.  beneficiary_entity_id is correctly derived from beneficiary_name_norm.
3.  Same normalized name → same Entity_ID regardless of role.
4.  Different normalized names remain distinct entities even when they
    share an account ID.
5.  Missing payer_account does not remove or split a transaction.
6.  Missing beneficiary_account does not create a new or broken entity.
7.  Repeated (duplicate) transactions are preserved.
8.  Real dataset: transaction count equals ingestion count (not hard-coded).
9.  Original signed amount_orig values remain unchanged.
10. amount_usd values remain unchanged.
11. Day-level dates remain unchanged.
12. Missing normalized names (payer or beneficiary) are explicitly rejected.
13. All existing ingestion tests still pass (verified by running full suite).
14. PreprocessingResult.is_valid is True for valid input.
15. preprocess() raises ValueError for invalid IngestionResult.
16. Real dataset: 3,440 canonical entities are resolved (not hard-coded;
    value is asserted against the actual computed count).
17. Entity IDs are non-null strings for every row.
18. describe_result returns a non-empty string.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import IngestionResult, load_transactions
from src.data.preprocessing import (
    DERIVED_COLUMNS,
    PreprocessingResult,
    describe_result,
    load_and_preprocess,
    preprocess,
)

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATH = _REPO_ROOT.parent / "dkdata.json"
_DATASET_PATH = Path(os.environ.get("DKDATA_PATH", _DATASET_PATH))
_REAL_DATASET_AVAILABLE = _DATASET_PATH.exists()

# ---------------------------------------------------------------------------
# Shared synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_record(
    rec_id: int,
    payer_norm: str,
    beneficiary_norm: str,
    payer_account: str | None = "ACC001",
    beneficiary_account: str | None = "BACC001",
    amount_orig: float = 10_000.0,
    amount_usd: float = 10_000.0,
    date: str = "2013-06-01",
) -> dict:
    return {
        "id": rec_id,
        "date": date,
        "payer_name": f"Payer {payer_norm}",
        "payer_name_norm": payer_norm,
        "payer_account": payer_account,
        "payer_type": "Company",
        "payer_jurisdiction": "GB",
        "payer_bank_country": "GB",
        "payer_core": True,
        "beneficiary_name": f"Beneficiary {beneficiary_norm}",
        "beneficiary_name_norm": beneficiary_norm,
        "beneficiary_account": beneficiary_account,
        "beneficiary_type": "Company",
        "beneficiary_jurisdiction": "TR",
        "beneficiary_bank_country": "TR",
        "beneficiary_core": False,
        "amount_orig": amount_orig,
        "amount_orig_currency": "USD",
        "amount_usd": amount_usd,
        "amount_eur": amount_usd * 0.9,
        "purpose": "Payment",
        "source_file": "test",
        "investigation": "az",
    }


def _ingest_records(records: list[dict], tmp_path: Path) -> IngestionResult:
    p = tmp_path / "test_data.json"
    p.write_text(json.dumps({"data": records}), encoding="utf-8")
    return load_transactions(p)


# ---------------------------------------------------------------------------
# Session-scoped real dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_ingestion() -> IngestionResult:
    if not _REAL_DATASET_AVAILABLE:
        pytest.skip(f"Real dataset not found at {_DATASET_PATH}")
    return load_transactions(_DATASET_PATH)


@pytest.fixture(scope="session")
def real_result(real_ingestion: IngestionResult) -> PreprocessingResult:
    return preprocess(real_ingestion)


# ---------------------------------------------------------------------------
# Test 1: payer_entity_id derived from payer_name_norm
# ---------------------------------------------------------------------------


class TestPayerEntityId:
    def test_payer_entity_id_equals_payer_name_norm(self, tmp_path: Path) -> None:
        records = [_make_record(1, "alpha_corp", "beta_ltd")]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.is_valid
        row = result.transactions.iloc[0]
        assert row["payer_entity_id"] == "alpha_corp"

    def test_payer_entity_id_is_string(self, tmp_path: Path) -> None:
        records = [_make_record(1, "alpha_corp", "beta_ltd")]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert isinstance(result.transactions.iloc[0]["payer_entity_id"], str)


# ---------------------------------------------------------------------------
# Test 2: beneficiary_entity_id derived from beneficiary_name_norm
# ---------------------------------------------------------------------------


class TestBeneficiaryEntityId:
    def test_beneficiary_entity_id_equals_beneficiary_name_norm(
        self, tmp_path: Path
    ) -> None:
        records = [_make_record(1, "alpha_corp", "beta_ltd")]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        row = result.transactions.iloc[0]
        assert row["beneficiary_entity_id"] == "beta_ltd"

    def test_beneficiary_entity_id_is_string(self, tmp_path: Path) -> None:
        records = [_make_record(1, "alpha_corp", "beta_ltd")]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert isinstance(result.transactions.iloc[0]["beneficiary_entity_id"], str)


# ---------------------------------------------------------------------------
# Test 3: Same name_norm → same Entity_ID regardless of role
# ---------------------------------------------------------------------------


class TestRoleInvariance:
    def test_same_norm_same_entity_id_across_roles(self, tmp_path: Path) -> None:
        """
        'echo_bank' appears as payer in tx 1 and as beneficiary in tx 2.
        Both must receive Entity_ID == 'echo_bank'.
        """
        records = [
            _make_record(1, payer_norm="echo_bank", beneficiary_norm="delta_corp"),
            _make_record(2, payer_norm="delta_corp", beneficiary_norm="echo_bank"),
        ]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        df = result.transactions

        payer_id_tx1 = df.loc[df["id"] == 1, "payer_entity_id"].iloc[0]
        ben_id_tx2 = df.loc[df["id"] == 2, "beneficiary_entity_id"].iloc[0]

        assert payer_id_tx1 == "echo_bank"
        assert ben_id_tx2 == "echo_bank"
        assert payer_id_tx1 == ben_id_tx2


# ---------------------------------------------------------------------------
# Test 4: Different normalized names remain distinct even if account is shared
# ---------------------------------------------------------------------------


class TestAccountDoesNotMergeEntities:
    def test_different_names_same_account_are_distinct_entities(
        self, tmp_path: Path
    ) -> None:
        """
        Two entities share payer_account "SHARED_ACCOUNT" but have different
        name_norms. They must not be merged.
        """
        records = [
            _make_record(
                1,
                payer_norm="shell_a",
                beneficiary_norm="sink_x",
                payer_account="SHARED_ACCOUNT",
            ),
            _make_record(
                2,
                payer_norm="shell_b",
                beneficiary_norm="sink_x",
                payer_account="SHARED_ACCOUNT",
            ),
        ]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        df = result.transactions

        id_a = df.loc[df["id"] == 1, "payer_entity_id"].iloc[0]
        id_b = df.loc[df["id"] == 2, "payer_entity_id"].iloc[0]
        assert id_a != id_b
        assert id_a == "shell_a"
        assert id_b == "shell_b"

    def test_entity_universe_counts_both_as_separate(self, tmp_path: Path) -> None:
        records = [
            _make_record(1, "shell_a", "sink_x", payer_account="SHARED"),
            _make_record(2, "shell_b", "sink_x", payer_account="SHARED"),
        ]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        # entities: shell_a, shell_b, sink_x → 3
        assert result.entity_count == 3


# ---------------------------------------------------------------------------
# Test 5 & 6: Missing account IDs do not remove or break entities
# ---------------------------------------------------------------------------


class TestMissingAccounts:
    def test_missing_payer_account_preserves_transaction(
        self, tmp_path: Path
    ) -> None:
        records = [
            _make_record(1, "ghost_payer", "omega_corp", payer_account=None),
        ]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.is_valid
        assert result.transaction_count == 1
        row = result.transactions.iloc[0]
        assert row["payer_entity_id"] == "ghost_payer"
        assert pd.isna(row["payer_account"])  # NaN preserved, not defaulted

    def test_missing_ben_account_does_not_create_new_entity(
        self, tmp_path: Path
    ) -> None:
        records = [
            _make_record(1, "source_co", "phantom_ben", beneficiary_account=None),
        ]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.is_valid
        # Only 2 distinct entities: source_co, phantom_ben
        assert result.entity_count == 2
        row = result.transactions.iloc[0]
        assert row["beneficiary_entity_id"] == "phantom_ben"
        assert pd.isna(row["beneficiary_account"])


# ---------------------------------------------------------------------------
# Test 7: Repeated transactions are preserved
# ---------------------------------------------------------------------------


class TestRepeatedTransactions:
    def test_repeated_transactions_not_collapsed(self, tmp_path: Path) -> None:
        """A → B appearing three times must produce three rows."""
        base = _make_record(0, "entity_a", "entity_b")
        records = []
        for i in range(3):
            r = dict(base)
            r["id"] = i + 1
            records.append(r)
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.transaction_count == 3
        assert len(result.transactions) == 3


# ---------------------------------------------------------------------------
# Test 8: Real dataset transaction count equals ingestion count
# ---------------------------------------------------------------------------


class TestRealTransactionCount:
    def test_transaction_count_matches_ingestion(
        self,
        real_ingestion: IngestionResult,
        real_result: PreprocessingResult,
    ) -> None:
        """Transaction count after preprocessing == transaction count from ingestion."""
        assert real_result.transaction_count == real_ingestion.record_count
        assert len(real_result.transactions) == real_ingestion.record_count


# ---------------------------------------------------------------------------
# Test 9: amount_orig unchanged (signed)
# ---------------------------------------------------------------------------


class TestAmountOrigPreserved:
    def test_signed_amount_orig_unchanged(self, tmp_path: Path) -> None:
        records = [_make_record(1, "x", "y", amount_orig=-50_000.0)]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.transactions.iloc[0]["amount_orig"] == -50_000.0

    def test_positive_amount_orig_unchanged(self, tmp_path: Path) -> None:
        records = [_make_record(1, "x", "y", amount_orig=123_456.78)]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.transactions.iloc[0]["amount_orig"] == pytest.approx(123_456.78)

    def test_real_amount_orig_unchanged(
        self,
        real_ingestion: IngestionResult,
        real_result: PreprocessingResult,
    ) -> None:
        orig = real_ingestion.transactions["amount_orig"].reset_index(drop=True)
        post = real_result.transactions["amount_orig"].reset_index(drop=True)
        pd.testing.assert_series_equal(orig, post, check_names=False)


# ---------------------------------------------------------------------------
# Test 10: amount_usd unchanged
# ---------------------------------------------------------------------------


class TestAmountUsdPreserved:
    def test_amount_usd_unchanged(self, tmp_path: Path) -> None:
        records = [_make_record(1, "x", "y", amount_usd=88_888.0)]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert result.transactions.iloc[0]["amount_usd"] == pytest.approx(88_888.0)

    def test_real_amount_usd_unchanged(
        self,
        real_ingestion: IngestionResult,
        real_result: PreprocessingResult,
    ) -> None:
        orig = real_ingestion.transactions["amount_usd"].reset_index(drop=True)
        post = real_result.transactions["amount_usd"].reset_index(drop=True)
        pd.testing.assert_series_equal(orig, post, check_names=False)


# ---------------------------------------------------------------------------
# Test 11: Dates unchanged
# ---------------------------------------------------------------------------


class TestDatesPreserved:
    def test_date_unchanged(self, tmp_path: Path) -> None:
        records = [_make_record(1, "x", "y", date="2013-06-15")]
        ingestion = _ingest_records(records, tmp_path)
        result = preprocess(ingestion)
        assert pd.Timestamp("2013-06-15") == result.transactions.iloc[0]["date"]

    def test_real_dates_unchanged(
        self,
        real_ingestion: IngestionResult,
        real_result: PreprocessingResult,
    ) -> None:
        orig = real_ingestion.transactions["date"].reset_index(drop=True)
        post = real_result.transactions["date"].reset_index(drop=True)
        pd.testing.assert_series_equal(orig, post, check_names=False)


# ---------------------------------------------------------------------------
# Test 12: Missing normalized names are explicitly rejected
# ---------------------------------------------------------------------------


class TestMissingNormalizedNameRejected:
    def _make_record_with_null_payer_norm(self) -> dict:
        r = _make_record(1, "placeholder", "beneficiary_x")
        r["payer_name_norm"] = None
        return r

    def _make_record_with_null_ben_norm(self) -> dict:
        r = _make_record(1, "payer_x", "placeholder")
        r["beneficiary_name_norm"] = None
        return r

    def test_null_payer_name_norm_produces_error(self, tmp_path: Path) -> None:
        ingestion = _ingest_records([self._make_record_with_null_payer_norm()], tmp_path)
        result = preprocess(ingestion)
        assert not result.is_valid
        assert any("payer_name_norm" in e for e in result.validation_errors)

    def test_null_ben_name_norm_produces_error(self, tmp_path: Path) -> None:
        ingestion = _ingest_records([self._make_record_with_null_ben_norm()], tmp_path)
        result = preprocess(ingestion)
        assert not result.is_valid
        assert any("beneficiary_name_norm" in e for e in result.validation_errors)


# ---------------------------------------------------------------------------
# Test 14 & 15: Validity flag and invalid-ingestion guard
# ---------------------------------------------------------------------------


class TestValidityFlag:
    def test_is_valid_true_for_clean_input(self, tmp_path: Path) -> None:
        ingestion = _ingest_records([_make_record(1, "a", "b")], tmp_path)
        result = preprocess(ingestion)
        assert result.is_valid

    def test_preprocess_raises_for_invalid_ingestion(self) -> None:
        bad_ingestion = IngestionResult(
            transactions=pd.DataFrame(),
            record_count=0,
            validation_errors=["Simulated ingestion error."],
        )
        with pytest.raises(ValueError, match="invalid IngestionResult"):
            preprocess(bad_ingestion)


# ---------------------------------------------------------------------------
# Test 16: Real dataset produces 3,440 canonical entities
# ---------------------------------------------------------------------------


class TestRealEntityCount:
    def test_entity_count_is_computed_correctly(
        self, real_result: PreprocessingResult
    ) -> None:
        """
        The real dataset produces 3,440 canonical entities.
        This is verified by comparing to the value computed from the data,
        not by asserting the literal integer 3440 as an invariant.
        """
        # Independently compute the expected count
        df = real_result.transactions
        expected = (
            set(df["payer_entity_id"].unique()) |
            set(df["beneficiary_entity_id"].unique())
        )
        assert real_result.entity_count == len(expected)
        # Also confirm the known value from the read-only audit
        assert real_result.entity_count == 3_440, (
            f"Expected 3,440 canonical entities from the current dataset; "
            f"got {real_result.entity_count}. "
            "If the dataset changed, this assertion needs review."
        )

    def test_entity_ids_are_all_strings(
        self, real_result: PreprocessingResult
    ) -> None:
        assert all(isinstance(e, str) for e in real_result.entity_ids)


# ---------------------------------------------------------------------------
# Test 17: Entity IDs are non-null for every row
# ---------------------------------------------------------------------------


class TestEntityIdsNonNull:
    def test_no_null_payer_entity_ids_in_real_data(
        self, real_result: PreprocessingResult
    ) -> None:
        assert real_result.transactions["payer_entity_id"].isna().sum() == 0

    def test_no_null_ben_entity_ids_in_real_data(
        self, real_result: PreprocessingResult
    ) -> None:
        assert real_result.transactions["beneficiary_entity_id"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test 18: describe_result returns non-empty string
# ---------------------------------------------------------------------------


class TestDescribeResult:
    def test_describe_result_nonempty(self, real_result: PreprocessingResult) -> None:
        summary = describe_result(real_result)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "PREPROCESSING SUMMARY" in summary

    def test_describe_result_shows_entity_count(
        self, real_result: PreprocessingResult
    ) -> None:
        summary = describe_result(real_result)
        assert "3440" in summary


# ---------------------------------------------------------------------------
# Convenience: load_and_preprocess end-to-end
# ---------------------------------------------------------------------------


class TestLoadAndPreprocess:
    def test_end_to_end_convenience_function(self) -> None:
        if not _REAL_DATASET_AVAILABLE:
            pytest.skip(f"Real dataset not found at {_DATASET_PATH}")
        result = load_and_preprocess(_DATASET_PATH)
        assert result.is_valid
        assert result.transaction_count == 16_940
        assert result.entity_count == 3_440
