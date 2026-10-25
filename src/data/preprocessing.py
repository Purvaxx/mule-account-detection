"""
Layer 2 — Data Preprocessing and Entity Resolution
====================================================

Receives the validated output of Layer 1 (:mod:`src.data.ingestion`) and
produces a clean, preprocessed transaction DataFrame suitable for the graph
construction layer (Layer 4).

Responsibilities
----------------
- Accept a validated :class:`~src.data.ingestion.IngestionResult`.
- Apply the canonical entity-resolution policy:
      Entity_ID = name_norm
  for both payer and beneficiary roles.
- Validate that no normalized name is missing (missing names are rejected
  explicitly; an identity is never invented).
- Produce derived columns:
      payer_entity_id      = payer_name_norm
      beneficiary_entity_id = beneficiary_name_norm
- Preserve every original transaction field intact.
- Report a summary of the preprocessing results via
  :class:`PreprocessingResult`.

Non-responsibilities (handled by later layers)
----------------------------------------------
- Degree / centrality / PageRank       → src/features/
- Velocity / rolling-window features   → src/features/
- TDA features                         → src/tda/
- Graph construction                   → src/graph/
- ML model features / risk scores      → src/models/ and src/detection/

Entity-resolution policy (docs/architecture.md §7)
--------------------------------------------------------------
The identifier used for entity resolution is selected explicitly by the caller
via :class:`IdentitySource`.  It is never inferred from the data.

``IdentitySource.NAME`` (default, and the Reference Dataset policy):
- Entity_ID = name_norm
- Account IDs are NOT entity identifiers.
- The same normalized name always maps to the same canonical entity.

``IdentitySource.ACCOUNT_ID`` (external datasets with stable identifiers and
no usable names):
- Entity_ID = payer_account / beneficiary_account
- Names are NOT entity identifiers.
- The same account always maps to the same canonical entity.

Invariant under both policies:
- Entities are NOT split by role (payer vs. beneficiary).
- Fuzzy matching is NOT applied.
- An identity is never invented; rows missing the selected identifier are
  rejected rather than filled.

Amount semantics
----------------
- ``amount_orig`` is signed in the source dataset (negative = debit).
  It is preserved exactly as ingested and is NEVER converted to an
  unsigned value by this layer.
- ``amount_usd`` is the absolute USD equivalent and is also preserved
  exactly as supplied.

Temporal semantics
------------------
- ``date`` carries day-level precision only.
- No intra-day timestamps are invented.
- No temporal ML features are created here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.ingestion import IngestionResult, load_transactions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns that must be present in the ingestion output before preprocessing
#: can proceed.
_REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "date",
    "payer_name_norm",
    "beneficiary_name_norm",
    "payer_name",
    "beneficiary_name",
    "payer_account",
    "beneficiary_account",
    "payer_type",
    "beneficiary_type",
    "payer_jurisdiction",
    "beneficiary_jurisdiction",
    "payer_bank_country",
    "beneficiary_bank_country",
    "payer_core",
    "beneficiary_core",
    "amount_orig",
    "amount_orig_currency",
    "amount_usd",
    "amount_eur",
    "purpose",
    "source_file",
    "investigation",
)

#: Columns present in the preprocessed output that did not exist in the raw
#: ingestion output (derived by this layer).
DERIVED_COLUMNS: tuple[str, ...] = (
    "payer_entity_id",
    "beneficiary_entity_id",
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingResult:
    """Container returned by :func:`preprocess`.

    Attributes
    ----------
    transactions:
        Preprocessed DataFrame.  One row per original transaction.
        All original columns are preserved; ``payer_entity_id`` and
        ``beneficiary_entity_id`` are added.
    transaction_count:
        Number of rows in the preprocessed DataFrame.
    entity_count:
        Number of distinct canonical entities resolved across both payer
        and beneficiary roles.
    entity_ids:
        Sorted tuple of all distinct ``Entity_ID`` values.
    validation_warnings:
        Non-fatal observations.  Processing completes even when warnings
        are present.
    validation_errors:
        Fatal issues.  When non-empty, callers must NOT pass the result to
        downstream layers.
    """

    transactions: pd.DataFrame
    transaction_count: int
    entity_count: int
    entity_ids: tuple[str, ...]
    validation_warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no fatal validation errors were detected."""
        return len(self.validation_errors) == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


from enum import Enum


class IdentitySource(str, Enum):
    """Which column supplies the canonical Entity_ID.

    This is the single authoritative definition of the identity strategy.
    Entity resolution is a Layer-2 (data) concern, and the project dependency
    direction is data -> graph -> features -> models -> detection ->
    application (docs/architecture.md §35), so the enum lives at the lowest
    layer that uses it.  The workspace layer re-exports it from
    :mod:`src.datasets.manifest`; it must not be redeclared there.
    """

    NAME = 'NAME'
    ACCOUNT_ID = 'ACCOUNT_ID'

def preprocess(ingestion_result: IngestionResult, identity_source: IdentitySource = IdentitySource.NAME) -> PreprocessingResult:
    """Apply entity resolution and field validation to ingested transactions.

    Parameters
    ----------
    ingestion_result:
        A validated :class:`~src.data.ingestion.IngestionResult` returned by
        :func:`~src.data.ingestion.load_transactions`.

    Returns
    -------
    PreprocessingResult
        Always returns a result.  Callers must inspect ``result.is_valid``
        and ``result.validation_errors`` before passing the output to
        downstream layers.

    Raises
    ------
    ValueError
        If ``ingestion_result`` itself is invalid (i.e. the ingestion layer
        already flagged fatal errors).  Preprocessing must not be run on
        invalid ingestion output.
    """
    if not ingestion_result.is_valid:
        raise ValueError(
            "Cannot preprocess an invalid IngestionResult. "
            f"Ingestion errors: {ingestion_result.validation_errors}"
        )

    df: pd.DataFrame = ingestion_result.transactions.copy()
    warnings: list[str] = []
    errors: list[str] = []

    logger.info(
        "Starting preprocessing on %d ingested transactions.", len(df)
    )

    # ------------------------------------------------------------------
    # 1. Verify required input columns are present
    # ------------------------------------------------------------------
    missing_cols = [c for c in _REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(
            f"Required input columns missing from ingestion output: {missing_cols}"
        )
        return PreprocessingResult(
            transactions=df,
            transaction_count=len(df),
            entity_count=0,
            entity_ids=(),
            validation_errors=errors,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2. Guard: reject missing identifier based on identity_source
    # ------------------------------------------------------------------
    if identity_source == IdentitySource.NAME:
        n_missing_payer_id = int(df["payer_name_norm"].isna().sum())
        n_missing_ben_id = int(df["beneficiary_name_norm"].isna().sum())
        payer_col = "payer_name_norm"
        ben_col = "beneficiary_name_norm"
    else:
        n_missing_payer_id = int(df["payer_account"].isna().sum())
        n_missing_ben_id = int(df["beneficiary_account"].isna().sum())
        payer_col = "payer_account"
        ben_col = "beneficiary_account"

    if n_missing_payer_id > 0:
        errors.append(
            f"'{payer_col}' is missing in {n_missing_payer_id} row(s). "
            f"Cannot assign a canonical Entity_ID without {identity_source.value}."
        )
    if n_missing_ben_id > 0:
        errors.append(
            f"'{ben_col}' is missing in {n_missing_ben_id} row(s). "
            f"Cannot assign a canonical Entity_ID without {identity_source.value}."
        )

    if errors:
        return PreprocessingResult(
            transactions=df,
            transaction_count=len(df),
            entity_count=0,
            entity_ids=(),
            validation_errors=errors,
        )

    # ------------------------------------------------------------------
    # 3. Apply entity-resolution policy
    # ------------------------------------------------------------------
    df["payer_entity_id"] = df[payer_col].astype(str)
    df["beneficiary_entity_id"] = df[ben_col].astype(str)

    logger.info(f"Entity IDs assigned from {identity_source.value} for all transactions.")

    # ------------------------------------------------------------------
    # 4. Validate generated Entity_IDs
    # ------------------------------------------------------------------
    n_null_payer_id = int(df["payer_entity_id"].isna().sum())
    n_null_ben_id = int(df["beneficiary_entity_id"].isna().sum())

    if n_null_payer_id > 0:
        errors.append(
            f"payer_entity_id is null for {n_null_payer_id} row(s) after assignment. "
            "This is an internal consistency error."
        )
    if n_null_ben_id > 0:
        errors.append(
            f"beneficiary_entity_id is null for {n_null_ben_id} row(s) after "
            "assignment. This is an internal consistency error."
        )

    # ------------------------------------------------------------------
    # 5. Resolve canonical entity universe
    #    Union of payer and beneficiary entity IDs across all transactions.
    # ------------------------------------------------------------------
    all_entity_ids: set[str] = (
        set(df["payer_entity_id"].unique()) |
        set(df["beneficiary_entity_id"].unique())
    )
    entity_ids_sorted: tuple[str, ...] = tuple(sorted(all_entity_ids))
    entity_count = len(entity_ids_sorted)

    logger.info("Canonical entity count: %d", entity_count)

    # ------------------------------------------------------------------
    # 6. Verify original amounts are unchanged (signed preservation check)
    #    amount_orig must not have been altered by entity resolution steps.
    # ------------------------------------------------------------------
    original_amount_orig = ingestion_result.transactions["amount_orig"].copy()
    if not df["amount_orig"].equals(original_amount_orig):
        errors.append(
            "amount_orig values have been modified during preprocessing. "
            "This is an internal consistency error."
        )

    original_amount_usd = ingestion_result.transactions["amount_usd"].copy()
    if not df["amount_usd"].equals(original_amount_usd):
        errors.append(
            "amount_usd values have been modified during preprocessing. "
            "This is an internal consistency error."
        )

    # ------------------------------------------------------------------
    # 7. Verify date values are unchanged
    # ------------------------------------------------------------------
    original_dates = ingestion_result.transactions["date"].copy()
    if not df["date"].equals(original_dates):
        errors.append(
            "date values have been modified during preprocessing. "
            "This is an internal consistency error."
        )

    # ------------------------------------------------------------------
    # 8. Verify transaction count is preserved
    # ------------------------------------------------------------------
    if len(df) != ingestion_result.record_count:
        errors.append(
            f"Transaction count changed during preprocessing: "
            f"expected {ingestion_result.record_count}, got {len(df)}."
        )

    # ------------------------------------------------------------------
    # 9. Warn about missing account identifiers (expected; not an error)
    # ------------------------------------------------------------------
    n_missing_payer_account = int(df["payer_account"].isna().sum())
    n_missing_ben_account = int(df["beneficiary_account"].isna().sum())

    if n_missing_payer_account > 0:
        warnings.append(
            f"payer_account is missing in {n_missing_payer_account} transaction(s). "
            "These rows retain their entity ID via payer_name_norm. "
            "Account metadata will be NaN for these rows."
        )
    if n_missing_ben_account > 0:
        warnings.append(
            f"beneficiary_account is missing in {n_missing_ben_account} "
            "transaction(s). Entity ID is derived from beneficiary_name_norm."
        )

    # ------------------------------------------------------------------
    # 10. Log summary
    # ------------------------------------------------------------------
    logger.info(
        "Preprocessing complete. Transactions: %d | Entities: %d | "
        "Warnings: %d | Errors: %d",
        len(df),
        entity_count,
        len(warnings),
        len(errors),
    )
    for w in warnings:
        logger.warning("Preprocessing warning: %s", w)
    for e in errors:
        logger.error("Preprocessing error: %s", e)

    return PreprocessingResult(
        transactions=df,
        transaction_count=len(df),
        entity_count=entity_count,
        entity_ids=entity_ids_sorted,
        validation_warnings=warnings,
        validation_errors=errors,
    )


def load_and_preprocess(dataset_path: str | Path, identity_source: IdentitySource = IdentitySource.NAME) -> PreprocessingResult:
    """Convenience function: ingest + preprocess in one call.

    Parameters
    ----------
    dataset_path:
        Path to ``dkdata.json``.

    Returns
    -------
    PreprocessingResult
    """
    ingestion_result = load_transactions(dataset_path)
    return preprocess(ingestion_result, identity_source=identity_source)


def describe_result(result: PreprocessingResult) -> str:
    """Return a human-readable summary of a :class:`PreprocessingResult`.

    Parameters
    ----------
    result:
        Result returned by :func:`preprocess`.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: list[str] = [
        "=" * 60,
        "PREPROCESSING SUMMARY",
        "=" * 60,
        f"Transaction count : {result.transaction_count}",
        f"Canonical entities: {result.entity_count}",
        f"Validation status : {'VALID' if result.is_valid else 'INVALID'}",
        "",
    ]

    if result.validation_warnings:
        lines.append(f"Warnings ({len(result.validation_warnings)}):")
        for w in result.validation_warnings:
            lines.append(f"  [WARN] {w}")
        lines.append("")

    if result.validation_errors:
        lines.append(f"Errors ({len(result.validation_errors)}):")
        for e in result.validation_errors:
            lines.append(f"  [ERROR] {e}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
