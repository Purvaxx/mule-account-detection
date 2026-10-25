"""
Layer 1 — Data Ingestion
========================

Loads the canonical ``dkdata.json`` dataset and converts it into a
validated internal transaction representation (a pandas DataFrame).

Responsibilities
----------------
- Parse the raw JSON file safely.
- Validate the expected top-level structure.
- Validate that all required columns are present.
- Report data-quality findings (missing values, type issues).
- Preserve every original transaction record intact.
- Return validation metadata alongside the DataFrame.

Non-responsibilities (handled by later layers)
----------------------------------------------
- Entity resolution  → src/data/preprocessing.py
- Graph construction → src/graph/
- Feature engineering → src/features/
- ML / modelling     → src/models/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical field definitions (as documented in docs/architecture.md §5)
# ---------------------------------------------------------------------------

#: Fields that MUST be present in the raw dataset.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "date",
    "payer_name",
    "payer_name_norm",
    "payer_account",
    "payer_type",
    "payer_jurisdiction",
    "payer_bank_country",
    "payer_core",
    "beneficiary_name",
    "beneficiary_name_norm",
    "beneficiary_account",
    "beneficiary_type",
    "beneficiary_jurisdiction",
    "beneficiary_bank_country",
    "beneficiary_core",
    "amount_orig",
    "amount_orig_currency",
    "amount_usd",
    "amount_eur",
    "purpose",
    "source_file",
    "investigation",
)

#: Fields expected to be non-null for every row.
#: Any field NOT in this set is allowed to have missing values.
ALWAYS_PRESENT_COLUMNS: frozenset[str] = frozenset({
    "id",
    "date",
    "payer_name",
    "payer_name_norm",
    "payer_type",
    "payer_jurisdiction",
    "payer_core",
    "beneficiary_name",
    "beneficiary_name_norm",
    "beneficiary_type",
    "beneficiary_jurisdiction",
    "beneficiary_core",
    "amount_orig",
    "amount_orig_currency",
    "amount_usd",
    "amount_eur",
    "purpose",
    "source_file",
    "investigation",
})

# ---------------------------------------------------------------------------
# Inference contract vs. training/evaluation contract
# ---------------------------------------------------------------------------
# ``REQUIRED_COLUMNS`` / ``ALWAYS_PRESENT_COLUMNS`` above describe the schema of
# the Reference Dataset, which is a *labelled training* dataset.  They are the
# right contract for training and evaluation and are left unchanged.
#
# They are NOT the right contract for inference.  An unlabelled external
# dataset legitimately has no ground truth and no analyst metadata, yet the
# frozen Experiment 3 model consumes neither: the 167 features are built from a
# strictly smaller set of fields.  The sets below make that distinction
# explicit so that a workspace can reach READY on what inference actually
# needs.
#
# Empirically established (tests/test_inference_contract.py): all 167 features
# are computable from the edge columns
#   source_entity, target_entity, amount_usd,
#   payer_bank_country, beneficiary_bank_country, date
# plus the ``entity_type`` node attribute, which is derived from
# payer_type / beneficiary_type.

#: Ground-truth labels and analyst metadata.  Used for training, evaluation and
#: display.  NEVER consumed by the 167-model features, and therefore never
#: required for inference.  Preserved verbatim when the dataset supplies them.
LABEL_COLUMNS: frozenset[str] = frozenset({
    "payer_core",
    "beneficiary_core",
    "investigation",
})

#: Columns STRUCTURALLY required for graph construction and the 167 features,
#: regardless of identity strategy.  Without these the feature matrix cannot be
#: built at all.
INFERENCE_REQUIRED_BASE: frozenset[str] = frozenset({
    "date",        # multigraph parallel-edge temporal dispersion
    "amount_usd",  # baseline volume + flow-imbalance features
})

#: Signal-bearing but structurally OPTIONAL.  When absent the 167-feature
#: vector is still produced in exactly the right shape and order, but the
#: corresponding block is degraded:
#:   - bank country absent  -> all 151 bank_country_* columns are 0
#:   - entity type absent   -> entity_type_unknown = 1
#: This is a documented, deliberate degradation (docs/architecture.md §12.1.1),
#: not a failure.  Their absence is reported as a quality warning so the
#: operator knows the model is seeing less signal — it does not block READY.
INFERENCE_DEGRADABLE_COLUMNS: frozenset[str] = frozenset({
    "payer_type",
    "beneficiary_type",
    "payer_bank_country",
    "beneficiary_bank_country",
})

#: Identity columns required per :class:`~src.data.preprocessing.IdentitySource`.
#: Keyed by the enum's *value* so this module does not need to import the enum
#: (``src.data.preprocessing`` imports this module, not the other way around).
IDENTITY_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "NAME": frozenset({"payer_name_norm", "beneficiary_name_norm"}),
    "ACCOUNT_ID": frozenset({"payer_account", "beneficiary_account"}),
}


def inference_required_columns(identity_source: str = "NAME") -> frozenset[str]:
    """Return the columns genuinely required to run inference.

    Parameters
    ----------
    identity_source:
        The ``IdentitySource`` value (``"NAME"`` or ``"ACCOUNT_ID"``).  Accepts
        the enum directly as well, since it is a ``str`` enum.
    """
    key = getattr(identity_source, "value", identity_source)
    if key not in IDENTITY_REQUIRED_COLUMNS:
        raise ValueError(
            f"Unknown identity source {key!r}. "
            f"Expected one of {sorted(IDENTITY_REQUIRED_COLUMNS)}."
        )
    return INFERENCE_REQUIRED_BASE | IDENTITY_REQUIRED_COLUMNS[key]


#: Everything in the canonical schema that is neither inference-required under
#: some identity strategy nor a label.  Carried through when present (it feeds
#: edge attributes, explainability and the UI) but never required.
OPTIONAL_METADATA_COLUMNS: frozenset[str] = (
    frozenset(REQUIRED_COLUMNS)
    - INFERENCE_REQUIRED_BASE
    - INFERENCE_DEGRADABLE_COLUMNS
    - LABEL_COLUMNS
    - IDENTITY_REQUIRED_COLUMNS["NAME"]
    - IDENTITY_REQUIRED_COLUMNS["ACCOUNT_ID"]
)


#: Columns whose values must be castable to a numeric type.
NUMERIC_COLUMNS: tuple[str, ...] = ("amount_orig", "amount_usd")

#: Columns whose values encode boolean flags (True / False).
BOOLEAN_COLUMNS: tuple[str, ...] = ("payer_core", "beneficiary_core")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Container returned by :func:`load_transactions`.

    Attributes
    ----------
    transactions:
        Validated DataFrame of all transaction records (one row per
        transaction).  All original columns are preserved; no records are
        silently dropped.
    record_count:
        Total number of rows loaded (before any validation filtering).
    missing_value_report:
        Mapping of column name -> count of null/NaN values in that column.
    validation_warnings:
        Non-fatal issues detected during validation.  Callers should log or
        surface these.  The DataFrame is still returned when warnings are
        present.
    validation_errors:
        Fatal issues that prevent further processing.  When this list is
        non-empty the caller should treat the result as invalid.
    """

    transactions: pd.DataFrame
    record_count: int
    missing_value_report: dict[str, int] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no fatal validation errors were detected."""
        return len(self.validation_errors) == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_transactions(path: str | Path) -> IngestionResult:
    """Load and validate the canonical ``dkdata.json`` dataset.

    Parameters
    ----------
    path:
        Absolute or relative path to ``dkdata.json``.

    Returns
    -------
    IngestionResult
        Always returns a result object.  Callers must inspect
        ``result.is_valid`` and ``result.validation_errors`` before
        proceeding to downstream processing.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as JSON or has unexpected structure.
    """
    path = Path(path)
    logger.info("Loading dataset from: %s", path)

    # ------------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # ------------------------------------------------------------------
    # 2. JSON parsing
    # ------------------------------------------------------------------
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # 3. Top-level structure — expect {"data": [...]}
    # ------------------------------------------------------------------
    if not isinstance(raw, dict) or "data" not in raw:
        raise ValueError(
            f"Unexpected top-level JSON structure in {path}. "
            "Expected an object with a 'data' key."
        )

    records = raw["data"]
    if not isinstance(records, list):
        raise ValueError(
            f"Expected 'data' to be a list of records; got {type(records).__name__}."
        )

    logger.info("Parsed %d raw records from JSON.", len(records))

    # ------------------------------------------------------------------
    # 4. Convert to DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    record_count = len(df)

    warnings: list[str] = []
    errors: list[str] = []

    # ------------------------------------------------------------------
    # 5. Required column presence
    # ------------------------------------------------------------------
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(
            f"Required columns are absent from the dataset: {missing_columns}"
        )
        # Cannot safely proceed with further validation.
        return IngestionResult(
            transactions=df,
            record_count=record_count,
            validation_errors=errors,
        )

    # ------------------------------------------------------------------
    # 6. Date parsing
    # ------------------------------------------------------------------
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to parse 'date' column: {exc}")

    # ------------------------------------------------------------------
    # 7. Numeric type coercion
    # ------------------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            original_non_null = int(df[col].notna().sum())
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coercion_failures = original_non_null - int(df[col].notna().sum())
            if coercion_failures > 0:
                warnings.append(
                    f"Column '{col}': {coercion_failures} value(s) could not be "
                    "coerced to numeric and were set to NaN."
                )

    # ------------------------------------------------------------------
    # 8. Boolean flag validation
    # ------------------------------------------------------------------
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            invalid_mask = ~df[col].isin([True, False, None])
            n_invalid = int(invalid_mask.sum())
            if n_invalid > 0:
                warnings.append(
                    f"Column '{col}': {n_invalid} row(s) contain unexpected "
                    f"non-boolean values: "
                    f"{df.loc[invalid_mask, col].unique().tolist()}"
                )

    # ------------------------------------------------------------------
    # 9. Missing value report (all columns)
    # ------------------------------------------------------------------
    missing_value_report: dict[str, int] = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().any()
    }

    # ------------------------------------------------------------------
    # 10. Always-present column null check
    # ------------------------------------------------------------------
    for col in ALWAYS_PRESENT_COLUMNS:
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                warnings.append(
                    f"Column '{col}' is expected to be non-null for every row, "
                    f"but {n_null} null value(s) were found."
                )

    # ------------------------------------------------------------------
    # 11. Amount sign check (all amounts should be positive)
    # ------------------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            n_non_positive = int((df[col].dropna() <= 0).sum())
            if n_non_positive > 0:
                warnings.append(
                    f"Column '{col}': {n_non_positive} row(s) have a "
                    "non-positive value. Review these records."
                )

    # ------------------------------------------------------------------
    # 12. Unique transaction ID check
    # ------------------------------------------------------------------
    if "id" in df.columns:
        n_duplicate_ids = int(df["id"].duplicated().sum())
        if n_duplicate_ids > 0:
            warnings.append(
                f"Column 'id': {n_duplicate_ids} duplicate transaction ID(s) detected."
            )

    # ------------------------------------------------------------------
    # 13. Log summary
    # ------------------------------------------------------------------
    logger.info(
        "Ingestion complete. Records: %d | Warnings: %d | Errors: %d",
        record_count,
        len(warnings),
        len(errors),
    )
    if missing_value_report:
        logger.info("Missing value counts: %s", missing_value_report)
    for w in warnings:
        logger.warning("Validation warning: %s", w)
    for e in errors:
        logger.error("Validation error: %s", e)

    return IngestionResult(
        transactions=df,
        record_count=record_count,
        missing_value_report=missing_value_report,
        validation_warnings=warnings,
        validation_errors=errors,
    )


def describe_result(result: IngestionResult) -> str:
    """Return a human-readable summary of an :class:`IngestionResult`.

    Parameters
    ----------
    result:
        Result returned by :func:`load_transactions`.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: list[str] = [
        "=" * 60,
        "INGESTION SUMMARY",
        "=" * 60,
        f"Record count      : {result.record_count}",
        f"Validation status : {'VALID' if result.is_valid else 'INVALID'}",
        "",
    ]

    if result.missing_value_report:
        lines.append("Missing values per column:")
        for col, count in sorted(result.missing_value_report.items()):
            pct = 100.0 * count / result.record_count if result.record_count else 0
            lines.append(f"  {col:<35} {count:>6} ({pct:.2f}%)")
        lines.append("")

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
