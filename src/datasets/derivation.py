"""
Deterministic derivation of canonical fields
============================================

Some canonical columns are not mapped from a source column: they are *computed*
from columns that were mapped, or from workspace metadata, under a rule that can
be stated in one sentence.  ``payer_name_norm`` is a normalization of
``payer_name``; ``date`` may be the composition of a calendar column and a clock
column; ``amount_usd`` may be ``amount_orig`` when — and only when — the operator
has declared the book is already USD.

This module owns **which derivations are possible for a given dataset, and under
what rule**.  Before it existed each rule lived inline in
``canonicalize_and_validate`` as one arm of an ``elif`` ladder interleaved with
the reporting of field statuses, plus two more blocks after the loop, so there
was no single place to read what the system can derive and no seam at which a
derivation could be tested without building a workspace on disk.

Two invariants hold for every rule here:

* **A derivation is evidence-bearing, not gap-filling.** Each rule fires only
  when its inputs are actually present, and every one records the source columns
  it read.  Nothing is defaulted to ``0``, ``"US"``, ``False`` or any other
  substantive value, no FX rate is invented, and no value is copied from one
  side of a transaction to the other.
* **Ambiguity is not a derivation.** A rule fires only where exactly one reading
  is possible.  Where the evidence supports more than one answer, no rule fires
  and the field stays missing for review — which is the honest outcome, not a
  failure to try harder.

Derivations are planned against a canonical frame that already holds the
directly-mapped columns, and applied by the caller.  Planning is pure: it reads
frames and configuration and returns descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.datasets.manifest import CurrencyPolicy, GeographyPolicy, WorkspaceConfig
from src.datasets.mapper import timestamp_part

__all__ = [
    "Derivation",
    "DerivationContext",
    "compose_instant",
    "instant_pair",
    "plan_derivations",
]


@dataclass(frozen=True)
class Derivation:
    """One canonical column the system can compute, and the rule it computed it by.

    ``values`` is whatever pandas will accept as a column assignment: a Series
    aligned to the canonical frame, or a scalar to broadcast.
    """

    canonical_name: str
    rule: str
    source_columns: Tuple[str, ...]
    values: Any

    def to_field_status(self) -> Dict[str, Any]:
        """The compatibility-report row describing this derivation."""
        return {
            "canonical_name": self.canonical_name,
            "state": "DERIVABLE",
            "source_columns": list(self.source_columns),
            "confidence": "HIGH",
            "rule": self.rule,
        }


@dataclass(frozen=True)
class DerivationContext:
    """Everything the rules are allowed to read.

    ``frame`` is the canonical frame after direct projection; ``raw`` is the
    uploaded file, needed only where a derivation composes *source* columns that
    were deliberately not projected individually.
    """

    frame: pd.DataFrame
    raw: pd.DataFrame
    #: canonical column name -> the source columns mapped to it.
    mapped_sources: Mapping[str, Sequence[str]]
    config: WorkspaceConfig
    upload_filename: Optional[str] = None
    #: Whether the declared currency policy survived assessment against the
    #: file (``src.datasets.policy``).  A declaration the dataset contradicts
    #: is not a licence to derive from it — see :func:`_derive_amount_usd`.
    currency_supported: bool = True

    def has(self, canonical_name: str) -> bool:
        return canonical_name in self.frame.columns

    def mapped(self, canonical_name: str) -> List[str]:
        return list(self.mapped_sources.get(canonical_name, []))


# ---------------------------------------------------------------------------
# Instant composition
# ---------------------------------------------------------------------------

def instant_pair(source_columns: Sequence[str]) -> Optional[Tuple[str, str]]:
    """``(date_column, time_column)`` when two columns are halves of one instant.

    Returns ``None`` for anything else — one column, three columns, two calendar
    columns, or a column that names a whole instant on its own.  The reading is
    the mapper's (:func:`src.datasets.mapper.timestamp_part`), so it is the same
    semantic vocabulary that proposed the mapping in the first place, not a
    second private list of column names.
    """
    if len(source_columns) != 2:
        return None
    parts = {timestamp_part(c): c for c in source_columns}
    if parts.keys() != {"date", "time"}:
        return None
    return parts["date"], parts["time"]


def _time_of_day(values: "pd.Series") -> "pd.Series":
    """A clock column as a timedelta since midnight.

    Handles the three shapes a spreadsheet produces for a time column: a string
    (``"10:35:19"``), a ``datetime.time``, and a full timestamp on a placeholder
    date (Excel's own representation).  Anything unparseable becomes ``NaT``.
    """
    as_text = values.astype(str)
    elapsed = pd.to_timedelta(as_text, errors="coerce")
    if elapsed.notna().any():
        return elapsed

    # Not clock-shaped text ("10:00" without seconds, or Excel's placeholder
    # date): read it as a datetime and keep only the time of day.  ``mixed``
    # matches the profiler's own parsing so a column typed as datetime there is
    # parsed the same way here.
    parsed = pd.to_datetime(values, format="mixed", errors="coerce")
    return pd.to_timedelta(
        parsed.dt.hour.astype("Float64") * 3600
        + parsed.dt.minute.astype("Float64") * 60
        + parsed.dt.second.astype("Float64"),
        unit="s",
    )


def compose_instant(date_values: "pd.Series", time_values: "pd.Series") -> "pd.Series":
    """Compose a calendar column and a clock column into one timestamp column.

    A row whose time is unreadable keeps its date at midnight rather than being
    discarded: the calendar fact was supplied and is not in doubt, and dropping
    the whole instant would silently remove the row from the graph.  A row whose
    *date* is unreadable stays ``NaT`` — there is no instant to place it at, and
    inventing one would be fabrication.
    """
    days = pd.to_datetime(date_values, errors="coerce").dt.normalize()
    elapsed = _time_of_day(time_values).fillna(pd.Timedelta(0))
    return days + elapsed


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _normalized_name(values: "pd.Series") -> "pd.Series":
    """Lowercase, alphanumeric-only entity key. The Reference Dataset's own rule."""
    return (values.astype(str)
            .replace(["nan", "None", "<NA>"], pd.NA)
            .str.lower()
            .str.replace(r'[^a-z0-9]', '', regex=True))


def _derive_instant(ctx: DerivationContext) -> List[Derivation]:
    pair = instant_pair(ctx.mapped("date"))
    if pair is None:
        return []
    date_col, time_col = pair
    if date_col not in ctx.raw.columns or time_col not in ctx.raw.columns:
        return []
    return [Derivation(
        canonical_name="date",
        rule=(f"Composed from the calendar column {date_col!r} and the clock "
              f"column {time_col!r}"),
        source_columns=(date_col, time_col),
        values=compose_instant(ctx.raw[date_col], ctx.raw[time_col]),
    )]


def _derive_payer_name_norm(ctx: DerivationContext) -> List[Derivation]:
    if ctx.has("payer_name_norm") or not ctx.has("payer_name"):
        return []
    return [Derivation(
        canonical_name="payer_name_norm",
        rule="Lowercase and alphanumeric strip from payer_name",
        source_columns=tuple(ctx.mapped("payer_name")),
        values=_normalized_name(ctx.frame["payer_name"]),
    )]


def _derive_beneficiary_name_norm(ctx: DerivationContext) -> List[Derivation]:
    if ctx.has("beneficiary_name_norm") or not ctx.has("beneficiary_name"):
        return []
    return [Derivation(
        canonical_name="beneficiary_name_norm",
        rule="Lowercase and alphanumeric strip from beneficiary_name",
        source_columns=tuple(ctx.mapped("beneficiary_name")),
        values=_normalized_name(ctx.frame["beneficiary_name"]),
    )]


def _derive_source_file(ctx: DerivationContext) -> List[Derivation]:
    if ctx.mapped("source_file"):
        return []
    return [Derivation(
        canonical_name="source_file",
        rule="Extracted from workspace upload metadata",
        source_columns=(),
        values=ctx.upload_filename,
    )]


def _derive_row_id(ctx: DerivationContext) -> List[Derivation]:
    if ctx.mapped("id"):
        return []
    return [Derivation(
        canonical_name="id",
        rule="Deterministic row identifier generation",
        source_columns=(),
        values=[f"tx_{i}" for i in range(len(ctx.raw))],
    )]


def _derive_amount_usd(ctx: DerivationContext) -> List[Derivation]:
    """``amount_orig`` is USD only because the operator declared it so.

    No amount of evidence in the file *makes* this declaration: a currency
    column reading "USD" corroborates it and a column reading otherwise
    contradicts it (``src.datasets.policy``), but the assertion that untagged
    numbers are dollars is the operator's to make.
    """
    if ctx.config.currency_policy != CurrencyPolicy.NATIVE_USD:
        return []
    if not ctx.currency_supported:
        # The declaration is contradicted by the dataset's own currency column,
        # or cannot be checked against it.  Copying the native amounts into
        # amount_usd would be an invented FX rate of exactly 1.0, and writing it
        # into canonical.parquet would leave euros on disk labelled as dollars
        # even though the compatibility gate stops the model from reading them.
        # Refuse at the point of derivation as well as at the gate.
        return []
    if ctx.has("amount_usd") or not ctx.has("amount_orig"):
        return []
    return [Derivation(
        canonical_name="amount_usd",
        rule="NATIVE_USD policy copies amount_orig to amount_usd",
        source_columns=("amount_orig",),
        values=ctx.frame["amount_orig"],
    )]


def _derive_single_country(ctx: DerivationContext) -> List[Derivation]:
    """Fill both bank-country columns from one declared domestic country.

    This is the only place a country reaches both sides of a transaction, and it
    is permitted only because the operator stated that every transaction in the
    dataset is domestic to that country.  It is never inferred from a
    single-valued geography column, and one side's value is never copied to the
    other.
    """
    if ctx.config.geography_policy != GeographyPolicy.SINGLE_DOMESTIC_COUNTRY:
        return []
    code = (ctx.config.single_country_code or "").strip()
    if not code:
        return []
    out = []
    for column in ("payer_bank_country", "beneficiary_bank_country"):
        if ctx.has(column):
            continue
        out.append(Derivation(
            canonical_name=column,
            rule=(f"SINGLE_DOMESTIC_COUNTRY policy fills every row with the "
                  f"declared country {code!r}"),
            source_columns=(),
            values=code,
        ))
    return out


#: Ordered.  Each rule sees the frame as the rules before it left it, so a rule
#: may depend on an earlier derivation (``payer_name_norm`` on a projected
#: ``payer_name``) but never on a later one.  Every rule has one shape — it
#: returns the derivations it supports, possibly none — so the loop below needs
#: no special case for the rule that produces two columns.
_RULES = (
    _derive_instant,
    _derive_payer_name_norm,
    _derive_beneficiary_name_norm,
    _derive_source_file,
    _derive_row_id,
    _derive_amount_usd,
    _derive_single_country,
)


def plan_derivations(ctx: DerivationContext) -> List[Derivation]:
    """Every canonical column this dataset supports deriving, in application order.

    Pure: the caller's context is not mutated and no file is touched.  The caller
    applies the results, which is also what makes the plan reportable — the
    compatibility report lists exactly the rules that ran.
    """
    derivations: List[Derivation] = []

    for rule in _RULES:
        for derivation in rule(ctx):
            derivations.append(derivation)
            # The next rule sees the column this one produced.
            ctx = replace(ctx, frame=ctx.frame.assign(
                **{derivation.canonical_name: derivation.values}))

    return derivations
