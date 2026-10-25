"""
Dataset configuration policy — assessment against the actual upload
===================================================================

``WorkspaceConfig`` records three decisions the system cannot reliably infer and
must not guess: how entities are identified, what currency the amounts are in,
and how geography is expressed.  Recording a decision is not the same as being
able to honour it, and this module owns that second question:

    *Does the uploaded dataset actually support what the operator declared?*

It answers per dimension with a status, a stated reason, and the evidence the
answer rests on.  Three consumers share that one answer:

* :mod:`src.datasets.canonicalizer` — decides ``READY`` / ``INCOMPATIBLE``;
* the ``/api/workspaces/<id>/config`` route — validates before persisting;
* the configuration UI — renders status, reason and evidence.

Before this module each of those either re-derived the answer, skipped it, or
did not ask.

Evidence is drawn from the **real uploaded file** (a bounded sample, the same
bound the profiler uses), never from the declaration itself.  Where no evidence
is available the assessment says so rather than assuming the declaration holds.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.datasets.fx import fx_availability
from src.datasets.geography import resolve_country
from src.datasets.mapper import CanonicalConcept, MappingStatus
from src.datasets.manifest import (
    CurrencyPolicy,
    GeographyPolicy,
    IdentitySource,
    WorkspaceConfig,
)

__all__ = [
    "DimensionStatus",
    "DimensionAssessment",
    "ConfigurationAssessment",
    "DatasetEvidence",
    "collect_dataset_evidence",
    "unresolved_concept_claims",
    "assess_configuration",
    "configuration_options",
]


class DimensionStatus:
    """Status of one configuration dimension.

    ``INCOMPATIBLE`` blocks inference; ``NEEDS_REVIEW`` does not.  That split is
    load-bearing and pre-existing: ``CurrencyPolicy.UNKNOWN`` and
    ``GeographyPolicy.UNSPECIFIED`` are the defaults for any workspace whose
    configuration was never set, so treating ``NEEDS_REVIEW`` as a blocker would
    demote every existing workspace.
    """

    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    INCOMPATIBLE = "INCOMPATIBLE"


#: How many distinct values an evidence summary retains per column.  Evidence is
#: persisted and rendered, so it is bounded.
_EVIDENCE_VALUE_LIMIT = 20

#: Share of sampled rows whose currency must be USD before a ``NATIVE_USD``
#: declaration is contradicted by the dataset's own currency column.  A dataset
#: that states its currency and states something other than USD is not USD; the
#: threshold exists only to tolerate stray nulls and casing, not mixed books.
_NATIVE_USD_MIN_SHARE = 0.999

#: Non-null share an identity column must reach in the sample before the chosen
#: identity strategy is considered supported without review.
_IDENTITY_MIN_COVERAGE = 0.95

#: Minimum candidate score for a column to count as *claiming* a concept.
#: Matches the mapper's own NEEDS_CONFIRMATION bar, so "claimed" here means the
#: same thing it means in the mapping review screen.
_CLAIM_THRESHOLD = 0.3

#: Accepted spellings of "US dollar" in a currency column.
_USD_TOKENS = frozenset({"USD", "US DOLLAR", "US DOLLARS", "USDOLLAR",
                         "DOLLAR", "DOLLARS", "US$", "$", "840"})


@dataclass
class ColumnEvidence:
    """Observed facts about one mapped source column, from the real upload."""

    concept: str
    source_columns: List[str]
    sampled_rows: int
    non_null: int
    distinct: int
    #: value -> occurrences, most frequent first, bounded.
    top_values: Dict[str, int] = field(default_factory=dict)

    @property
    def non_null_ratio(self) -> float:
        return (self.non_null / self.sampled_rows) if self.sampled_rows else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["non_null_ratio"] = self.non_null_ratio
        return d


@dataclass
class DatasetEvidence:
    """Bounded, observed evidence about the columns the configuration depends on.

    ``sampled_rows`` is the number of rows actually inspected, not the dataset
    size: every ratio below is a property of the sample and is presented as such.
    """

    sampled_rows: int
    is_exact: bool
    columns: Dict[str, ColumnEvidence] = field(default_factory=dict)

    def get(self, concept: CanonicalConcept) -> Optional[ColumnEvidence]:
        return self.columns.get(concept.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampled_rows": self.sampled_rows,
            "is_exact": self.is_exact,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }


#: Concepts whose actual values the configuration assessment needs to see.
_EVIDENCE_CONCEPTS = (
    CanonicalConcept.SENDER_NAME,
    CanonicalConcept.RECEIVER_NAME,
    CanonicalConcept.SENDER_ACCOUNT,
    CanonicalConcept.RECEIVER_ACCOUNT,
    CanonicalConcept.TRANSACTION_AMOUNT,
    CanonicalConcept.USD_AMOUNT,
    CanonicalConcept.ORIGINAL_CURRENCY,
    CanonicalConcept.SENDER_BANK_COUNTRY,
    CanonicalConcept.RECEIVER_BANK_COUNTRY,
)


def mapped_concepts(mapping_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """concept -> source columns, for the mappings canonicalization will project.

    Mirrors the canonicalizer's own admission rule (``AUTO_ACCEPTED`` or
    ``NEEDS_CONFIRMATION``) so the assessment and the canonicalizer can never
    disagree about which concepts are available.
    """
    out: Dict[str, List[str]] = {}
    for m in (mapping_data or {}).get("mappings", []):
        status = m.get("status")
        concept = m.get("selected_concept")
        if not concept:
            continue
        if status in (MappingStatus.AUTO_ACCEPTED.value,
                      MappingStatus.NEEDS_CONFIRMATION.value):
            out.setdefault(concept, []).append(m["source_column"])
    return out


def unresolved_concept_claims(mapping_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """concept -> source columns that claimed it but were not selected for it.

    A concept can be *claimed* without being *mapped*: two columns tie for it
    and both drop to ``CONFLICT``, or one is beaten by a better claimant and has
    its selection cleared.  :func:`mapped_concepts` correctly omits those, and
    for projection that is right — an unresolved claim must not become a
    canonical column.

    For assessment it is not enough.  "No currency column is mapped" and "the
    dataset states its currency in two columns and the mapping has not decided
    which" are different situations, and treating the second as the first is how
    a fifteen-currency book passes a ``NATIVE_USD`` declaration unchallenged:
    the contradicting evidence exists in the file and is simply not being
    looked at.
    """
    mapped = mapped_concepts(mapping_data)
    out: Dict[str, List[str]] = {}
    for m in (mapping_data or {}).get("mappings", []):
        column = m.get("source_column")
        for candidate in m.get("candidates", []):
            concept = candidate.get("canonical_concept")
            if not concept or concept in mapped:
                continue
            # Only claims strong enough to have been selectable.  A weaker
            # candidate is retained for operator context, not evidence that the
            # dataset states the fact.
            if float(candidate.get("score", 0.0)) < _CLAIM_THRESHOLD:
                continue
            if column not in out.setdefault(concept, []):
                out[concept].append(column)
    return {k: v for k, v in out.items() if v}


def collect_dataset_evidence(
    sample: "pd.DataFrame",
    concept_sources: Dict[str, List[str]],
    is_exact: bool = False,
) -> DatasetEvidence:
    """Summarise the mapped columns of a real (bounded) dataset sample."""
    evidence = DatasetEvidence(sampled_rows=int(len(sample)), is_exact=is_exact)

    for concept in _EVIDENCE_CONCEPTS:
        cols = [c for c in concept_sources.get(concept.value, []) if c in sample.columns]
        if not cols:
            continue

        series = sample[cols[0]] if len(cols) == 1 else (
            sample[cols].astype(str).agg(" ".join, axis=1)
        )
        non_null = series.dropna()
        counts = non_null.astype(str).value_counts()

        evidence.columns[concept.value] = ColumnEvidence(
            concept=concept.value,
            source_columns=cols,
            sampled_rows=int(len(sample)),
            non_null=int(len(non_null)),
            distinct=int(counts.size),
            top_values={str(k): int(v)
                        for k, v in counts.head(_EVIDENCE_VALUE_LIMIT).items()},
        )

    return evidence


@dataclass
class DimensionAssessment:
    """One dimension's verdict, with the reason and the evidence behind it."""

    dimension: str
    selection: str
    status: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status == DimensionStatus.INCOMPATIBLE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["blocking"] = self.blocking
        return d


@dataclass
class ConfigurationAssessment:
    identity: DimensionAssessment
    currency: DimensionAssessment
    geography: DimensionAssessment
    #: True when evidence came from the real upload rather than the declaration
    #: alone.  A False here is why an assessment may be weaker than it looks.
    evidence_available: bool = False
    sampled_rows: int = 0

    @property
    def dimensions(self) -> List[DimensionAssessment]:
        return [self.identity, self.currency, self.geography]

    @property
    def blocking(self) -> bool:
        return any(d.blocking for d in self.dimensions)

    @property
    def status(self) -> str:
        if self.blocking:
            return DimensionStatus.INCOMPATIBLE
        if any(d.status == DimensionStatus.NEEDS_REVIEW for d in self.dimensions):
            return DimensionStatus.NEEDS_REVIEW
        return DimensionStatus.READY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "blocking": self.blocking,
            "evidence_available": self.evidence_available,
            "sampled_rows": self.sampled_rows,
            "identity": self.identity.to_dict(),
            "currency": self.currency.to_dict(),
            "geography": self.geography.to_dict(),
        }


# ---------------------------------------------------------------------------
# Per-dimension assessment
# ---------------------------------------------------------------------------

def _identity_assessment(
    config: WorkspaceConfig,
    concept_sources: Dict[str, List[str]],
    evidence: Optional[DatasetEvidence],
) -> DimensionAssessment:
    source = config.identity_source
    if source == IdentitySource.NAME:
        required = (CanonicalConcept.SENDER_NAME, CanonicalConcept.RECEIVER_NAME)
        label = "counterparty names"
        concept_label = "counterparty name"
    else:
        required = (CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT)
        label = "account identifiers"
        concept_label = "account identifier"

    missing = [c.value for c in required if c.value not in concept_sources]
    if missing:
        return DimensionAssessment(
            dimension="identity",
            selection=source.value,
            status=DimensionStatus.INCOMPATIBLE,
            reason=(
                f"Resolving entities by {label} requires {' and '.join(missing)} "
                f"to be mapped. The current mapping does not supply "
                f"{'them' if len(missing) > 1 else 'it'}, so entities cannot be "
                f"resolved."
            ),
            evidence={"missing_concepts": missing},
        )

    detail: Dict[str, Any] = {
        "required_concepts": [c.value for c in required],
        "source_columns": {c.value: concept_sources[c.value] for c in required},
    }

    if evidence is None:
        return DimensionAssessment(
            dimension="identity",
            selection=source.value,
            status=DimensionStatus.READY,
            reason=(
                f"Both {concept_label} concepts are mapped. Their contents have not "
                f"been inspected, so coverage is unverified."
            ),
            evidence=detail,
        )

    thin = []
    for c in required:
        col = evidence.get(c)
        if col is None:
            continue
        detail[c.value] = {
            "non_null_ratio": round(col.non_null_ratio, 4),
            "distinct": col.distinct,
        }
        if col.non_null_ratio < _IDENTITY_MIN_COVERAGE:
            thin.append((c.value, col.non_null_ratio))

    if thin:
        parts = ", ".join(f"{n} is populated in {r:.0%} of sampled rows"
                          for n, r in thin)
        return DimensionAssessment(
            dimension="identity",
            selection=source.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                f"{parts}. Rows with a missing identifier are dropped during "
                f"preprocessing, so part of the dataset will not reach the graph."
            ),
            evidence=detail,
        )

    return DimensionAssessment(
        dimension="identity",
        selection=source.value,
        status=DimensionStatus.READY,
        reason=(
            f"Both {concept_label} concepts are mapped and populated across the "
            f"sampled rows."
        ),
        evidence=detail,
    )


def _usd_share(col: ColumnEvidence) -> Optional[float]:
    """Share of sampled non-null currency values that denote US dollars."""
    if not col.top_values or col.non_null == 0:
        return None
    usd = sum(n for v, n in col.top_values.items()
              if str(v).strip().upper() in _USD_TOKENS)
    return usd / col.non_null


def _currency_observation(
    evidence: Optional[DatasetEvidence],
) -> Tuple[Dict[str, Any], Optional[float]]:
    """What the dataset's own currency column says, and how much of it is USD.

    Returns ``({}, None)`` when the dataset states no currency at all, which is
    a different situation from stating one this code could not read.  Both
    branches of the currency assessment ask the same question of the same
    column, so they ask it in one place.
    """
    col = evidence.get(CanonicalConcept.ORIGINAL_CURRENCY) if evidence else None
    if col is None or col.non_null == 0:
        return {}, None

    detail: Dict[str, Any] = {
        "currency_column": col.source_columns,
        "distinct_currencies": col.distinct,
        "observed_currencies": col.top_values,
    }
    share = _usd_share(col)
    if share is not None:
        detail["usd_share"] = round(share, 4)
        detail["supports_native_usd"] = share >= _NATIVE_USD_MIN_SHARE
    return detail, share


def _currency_assessment(
    config: WorkspaceConfig,
    concept_sources: Dict[str, List[str]],
    evidence: Optional[DatasetEvidence],
    unresolved_claims: Optional[Dict[str, List[str]]] = None,
) -> DimensionAssessment:
    policy = config.currency_policy
    unresolved_claims = unresolved_claims or {}

    if policy == CurrencyPolicy.EXTERNAL_FX_REQUIRED:
        availability = fx_availability()
        return DimensionAssessment(
            dimension="currency",
            selection=policy.value,
            status=(DimensionStatus.READY if availability["available"]
                    else DimensionStatus.INCOMPATIBLE),
            reason=availability["reason"] or (
                f"FX conversion available via {availability['provider']}."
            ),
            evidence={"fx": availability},
        )

    if policy == CurrencyPolicy.UNKNOWN:
        # The file may already answer this question.  Saying what it says is
        # not the same as answering for the operator: the declaration stays
        # theirs to make (no evidence ever *makes* a
        # declaration), but leaving them to discover a single-currency USD
        # column for themselves is withholding an answer we already have.
        detail, share = _currency_observation(evidence)
        observed = ""
        if share is not None:
            if detail["supports_native_usd"]:
                observed = (
                    f" The dataset's own currency column reports USD for "
                    f"{share:.0%} of sampled rows, so declaring 'Already "
                    f"USD' would be supported by the file itself."
                )
            else:
                observed = (
                    f" The dataset's own currency column reports "
                    f"{detail['distinct_currencies']} distinct currencies, only "
                    f"{share:.0%} of them USD, so 'Already USD' would not "
                    f"be supported by the file: converting these amounts "
                    f"needs dated, attributable FX rates."
                )
        return DimensionAssessment(
            dimension="currency",
            selection=policy.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                "The currency of the amounts has not been declared. The model "
                "consumes amount_usd; until this is stated the amounts are taken "
                "at face value without any assertion that they are USD." + observed
            ),
            evidence=detail,
        )

    # NATIVE_USD
    has_amount = (
        CanonicalConcept.TRANSACTION_AMOUNT.value in concept_sources
        or CanonicalConcept.USD_AMOUNT.value in concept_sources
    )
    if not has_amount:
        return DimensionAssessment(
            dimension="currency",
            selection=policy.value,
            status=DimensionStatus.INCOMPATIBLE,
            reason=(
                "Declaring amounts already USD requires an amount column to be "
                "mapped (TRANSACTION_AMOUNT or USD_AMOUNT). None is."
            ),
            evidence={"missing_concepts": [CanonicalConcept.TRANSACTION_AMOUNT.value,
                                           CanonicalConcept.USD_AMOUNT.value]},
        )

    # A dataset that supplies its own USD column is not relying on the
    # declaration at all: canonicalization projects USD_AMOUNT straight into
    # amount_usd and never copies amount_orig.  ORIGINAL_CURRENCY then describes
    # amount_orig -- a different quantity, which the model does not consume --
    # so it cannot contradict anything.  The declaration is load-bearing only
    # when amount_usd has to be DERIVED from the native amount.
    supplies_usd_column = CanonicalConcept.USD_AMOUNT.value in concept_sources

    detail: Dict[str, Any] = {
        "amount_concepts": [c for c in (CanonicalConcept.USD_AMOUNT.value,
                                        CanonicalConcept.TRANSACTION_AMOUNT.value)
                            if c in concept_sources],
        "usd_amount_supplied": supplies_usd_column,
    }

    if supplies_usd_column:
        return DimensionAssessment(
            dimension="currency",
            selection=policy.value,
            status=DimensionStatus.READY,
            reason=(
                "The dataset supplies its own USD amount column, which is used "
                "directly. No conversion is performed and no native amount is "
                "reinterpreted as USD."
            ),
            evidence=detail,
        )

    # Otherwise amount_usd is derived by copying amount_orig, and the dataset's
    # own currency column is the one piece of evidence that can contradict that.
    # If it says the book is mixed, the declaration is wrong regardless of how
    # confidently it was made.
    #
    # That check is only as good as its ability to *see* the currency column.
    # When two columns tie for ORIGINAL_CURRENCY and both drop to CONFLICT, the
    # concept is unmapped and the contradiction test silently passes -- a
    # fifteen-currency book then reaches READY with euros and yen recorded as
    # amount_usd.  An unresolved claim is not an absent one: the dataset does
    # state a currency, and until the mapping says which column states it, the
    # declaration cannot be checked and must not be relied on.
    unresolved_currency = unresolved_claims.get(
        CanonicalConcept.ORIGINAL_CURRENCY.value, [])
    if unresolved_currency:
        return DimensionAssessment(
            dimension="currency",
            selection=policy.value,
            status=DimensionStatus.INCOMPATIBLE,
            reason=(
                f"amount_usd would be a copy of the native amounts, but the "
                f"dataset states a currency of its own in "
                f"{', '.join(repr(c) for c in unresolved_currency)} and the "
                f"mapping has not resolved which column that is. The "
                f"declaration cannot be checked against the file until it does. "
                f"Confirm the currency column in schema mapping, then re-validate."
            ),
            evidence=dict(detail, unresolved_currency_claims=unresolved_currency),
        )

    observation, share = _currency_observation(evidence)
    detail.update(observation)

    if share is not None and share < _NATIVE_USD_MIN_SHARE:
        return DimensionAssessment(
            dimension="currency",
            selection=policy.value,
            status=DimensionStatus.INCOMPATIBLE,
            reason=(
                f"The dataset's own currency column reports "
                f"{detail['distinct_currencies']} distinct currencies and only "
                f"{share:.0%} of sampled rows are USD, which contradicts "
                f"the declaration that amounts are already USD. "
                f"amount_usd would be a copy of those native amounts. "
                f"Converting them needs dated, attributable FX rates; "
                f"they will not be fabricated."
            ),
            evidence=detail,
        )

    return DimensionAssessment(
        dimension="currency",
        selection=policy.value,
        status=DimensionStatus.READY,
        reason=(
            "Amounts are declared to be USD and a mapped amount column supplies "
            "them."
            + ("" if observation else
               " The dataset states no currency of its own, so this rests on the "
               "declaration.")
        ),
        evidence=detail,
    )


def _geography_assessment(
    config: WorkspaceConfig,
    concept_sources: Dict[str, List[str]],
    evidence: Optional[DatasetEvidence],
) -> DimensionAssessment:
    policy = config.geography_policy

    if policy == GeographyPolicy.UNSPECIFIED:
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                "No geography is declared. All 151 bank_country features are "
                "zero for every entity: the model runs, but sees no geographic "
                "signal. Nothing is invented to fill the gap."
            ),
            evidence={},
        )

    if policy == GeographyPolicy.SINGLE_DOMESTIC_COUNTRY:
        code = (config.single_country_code or "").strip()
        if not code:
            return DimensionAssessment(
                dimension="geography",
                selection=policy.value,
                status=DimensionStatus.NEEDS_REVIEW,
                reason=(
                    "A single domestic country is declared but no country code "
                    "was supplied, so nothing can be filled in."
                ),
                evidence={},
            )
        res = resolve_country(code)
        if not res.resolved:
            return DimensionAssessment(
                dimension="geography",
                selection=policy.value,
                status=DimensionStatus.NEEDS_REVIEW,
                reason=(
                    f"'{code}' is not in the frozen Experiment 3 country "
                    f"vocabulary and could not be resolved into it. It will be "
                    f"recorded, but sets no feature column, so the geographic "
                    f"block stays zero."
                ),
                evidence={"single_country_code": code, "resolution": res.to_dict()},
            )
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.READY,
            reason=(
                f"Both bank-country columns will be filled with '{res.code}' "
                f"(resolved from '{code}' by rule {res.rule}), which is in the "
                f"frozen vocabulary."
            ),
            evidence={"single_country_code": code, "resolution": res.to_dict()},
        )

    # PAYER_AND_BENEFICIARY_SEPARATE
    required = (CanonicalConcept.SENDER_BANK_COUNTRY,
                CanonicalConcept.RECEIVER_BANK_COUNTRY)
    missing = [c.value for c in required if c.value not in concept_sources]
    if missing:
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.INCOMPATIBLE,
            reason=(
                f"Distinct sender and receiver geography is declared, but "
                f"{' and '.join(missing)} is not mapped. One side's geography is "
                f"never copied onto the other and no country is invented, so this "
                f"declaration cannot be honoured."
            ),
            evidence={"missing_concepts": missing},
        )

    detail: Dict[str, Any] = {
        "source_columns": {c.value: concept_sources[c.value] for c in required},
    }

    if evidence is None:
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.READY,
            reason=(
                "Both bank-country concepts are mapped. Their values have not "
                "been inspected, so how many resolve into the frozen vocabulary "
                "is unverified."
            ),
            evidence=detail,
        )

    # ``top_values`` is bounded (_EVIDENCE_VALUE_LIMIT), so it can cover fewer
    # rows than the column actually has.  Divide by the column's full non-null
    # count, not by the counts we happened to inspect: an uninspected tail then
    # lowers the share instead of being invisible.  This is the same
    # conservative direction _usd_share takes.
    resolvable = 0
    total = 0
    inspected = 0
    fully_inspected = True
    unresolved: List[str] = []
    #: Values that name a real country the frozen vocabulary has no column for.
    #: Not resolvable either, but for a different reason, and the operator can
    #: act on the difference: unreadable data can be fixed, a missing
    #: checkpoint column cannot.
    out_of_vocabulary: Dict[str, str] = {}
    for c in required:
        col = evidence.get(c)
        if col is None:
            continue
        total += col.non_null
        if col.distinct > len(col.top_values):
            fully_inspected = False
        for value, count in col.top_values.items():
            inspected += count
            resolution = resolve_country(value)
            if resolution.resolved:
                resolvable += count
                continue
            if value not in unresolved:
                unresolved.append(value)
            if resolution.recognized_but_unsupported:
                out_of_vocabulary[value] = resolution.recognized_as

    share = (resolvable / total) if total else 0.0
    detail["resolvable_share"] = round(share, 4)
    detail["inspected_values"] = inspected
    detail["sampled_values"] = total
    detail["fully_inspected"] = fully_inspected
    detail["unresolved_examples"] = unresolved[:10]
    detail["recognized_out_of_vocabulary"] = dict(list(out_of_vocabulary.items())[:10])

    if total == 0:
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                "Both bank-country concepts are mapped but the sampled rows "
                "contain no values for them."
            ),
            evidence=detail,
        )

    # Everything below distinguishes what was *inspected* from what was
    # *sampled*.  Only a column whose every distinct value was inspected can
    # support a statement about the whole sample.
    unverified = "" if fully_inspected else (
        f" Only the {inspected:,} most frequent of {total:,} sampled values "
        f"were inspected here; validation re-checks every value against the "
        f"full file."
    )

    if out_of_vocabulary:
        named = ", ".join(f"{raw!r} ({code})"
                          for raw, code in list(out_of_vocabulary.items())[:3])
        unverified += (
            f" {len(out_of_vocabulary)} of the unresolved values name a real "
            f"country the Experiment 3 checkpoint has no feature column for "
            f"({named}); that is a limit of the frozen vocabulary, not a "
            f"problem with the data, and nothing is substituted for them."
        )

    if share == 0.0:
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                f"None of the {len(unresolved)} inspected geography values "
                f"resolve into the frozen Experiment 3 country vocabulary "
                f"(e.g. {', '.join(repr(u) for u in unresolved[:3])}). "
                f"The 151 bank_country features stay zero and no geographic "
                f"signal reaches the model." + unverified
            ),
            evidence=detail,
        )

    if unresolved:
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                f"{share:.0%} of sampled geography values resolve into the frozen "
                f"country vocabulary. Values that do not "
                f"(e.g. {', '.join(repr(u) for u in unresolved[:3])}) set no "
                f"feature column; they are left as-is rather than bucketed."
                + unverified
            ),
            evidence=detail,
        )

    if not fully_inspected:
        # Every value we looked at resolved, but we did not look at all of
        # them.  Calling that "every sampled value resolves" would assert more
        # than was observed.
        return DimensionAssessment(
            dimension="geography",
            selection=policy.value,
            status=DimensionStatus.NEEDS_REVIEW,
            reason=(
                f"Every geography value inspected resolves into the frozen "
                f"country vocabulary, covering {share:.0%} of sampled rows."
                + unverified
            ),
            evidence=detail,
        )

    return DimensionAssessment(
        dimension="geography",
        selection=policy.value,
        status=DimensionStatus.READY,
        reason=(
            "Both bank-country concepts are mapped and every sampled value "
            "resolves into the frozen country vocabulary."
        ),
        evidence=detail,
    )


def assess_configuration(
    config: WorkspaceConfig,
    concept_sources: Dict[str, List[str]],
    evidence: Optional[DatasetEvidence] = None,
    unresolved_claims: Optional[Dict[str, List[str]]] = None,
) -> ConfigurationAssessment:
    """Assess a declared configuration against what the dataset actually supplies.

    Parameters
    ----------
    config:
        The operator's declaration.
    concept_sources:
        ``CanonicalConcept`` value -> source columns, for mappings that
        canonicalization will project (see :func:`mapped_concepts`).
    evidence:
        Observed values from a bounded sample of the real upload.  ``None`` is
        allowed and is reported: the assessment then rests on the mapping alone
        and says so, rather than assuming the declaration holds.
    unresolved_claims:
        ``CanonicalConcept`` value -> columns that claimed it without being
        selected for it (see :func:`unresolved_concept_claims`).  A dimension
        whose evidence is *ambiguous* is not a dimension with no evidence, and
        the difference is load-bearing for currency.
    """
    return ConfigurationAssessment(
        identity=_identity_assessment(config, concept_sources, evidence),
        currency=_currency_assessment(config, concept_sources, evidence,
                                      unresolved_claims),
        geography=_geography_assessment(config, concept_sources, evidence),
        evidence_available=evidence is not None,
        sampled_rows=evidence.sampled_rows if evidence else 0,
    )


def configuration_options() -> Dict[str, Any]:
    """Selectable values and their operator-facing meaning, for the UI.

    The UI must not hard-code policy vocabularies or the country list: both are
    derived from the frozen contract and would silently drift.
    """
    from src.datasets.geography import selectable_country_codes

    return {
        "identity_source": [
            {
                "value": IdentitySource.NAME.value,
                "label": "Counterparty name",
                "description": (
                    "One entity per normalized name. Two payments from the same "
                    "named party collapse to a single node."
                ),
                "requires": [CanonicalConcept.SENDER_NAME.value,
                             CanonicalConcept.RECEIVER_NAME.value],
            },
            {
                "value": IdentitySource.ACCOUNT_ID.value,
                "label": "Account identifier",
                "description": (
                    "One entity per account. Use when the dataset has stable "
                    "account ids and no usable names."
                ),
                "requires": [CanonicalConcept.SENDER_ACCOUNT.value,
                             CanonicalConcept.RECEIVER_ACCOUNT.value],
            },
        ],
        "currency_policy": [
            {
                "value": CurrencyPolicy.NATIVE_USD.value,
                "label": "Already USD",
                "description": (
                    "Amounts are USD as supplied; amount_orig is copied to "
                    "amount_usd. Contradicted if the dataset's own currency "
                    "column says otherwise."
                ),
                "requires": [CanonicalConcept.TRANSACTION_AMOUNT.value],
            },
            {
                "value": CurrencyPolicy.EXTERNAL_FX_REQUIRED.value,
                "label": "Needs FX conversion",
                "description": (
                    "Amounts are in other currencies. Blocked: no "
                    "provenance-backed FX provider is configured, and rates are "
                    "not fabricated."
                ),
                "requires": [],
            },
            {
                "value": CurrencyPolicy.UNKNOWN.value,
                "label": "Not stated",
                "description": "No currency claim is made. Reported for review.",
                "requires": [],
            },
        ],
        "geography_policy": [
            {
                "value": GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE.value,
                "label": "Separate sender / receiver country",
                "description": (
                    "The dataset supplies a bank country for each side. Values "
                    "are resolved into the frozen vocabulary where a stated rule "
                    "applies."
                ),
                "requires": [CanonicalConcept.SENDER_BANK_COUNTRY.value,
                             CanonicalConcept.RECEIVER_BANK_COUNTRY.value],
            },
            {
                "value": GeographyPolicy.SINGLE_DOMESTIC_COUNTRY.value,
                "label": "Single domestic country",
                "description": (
                    "Every transaction is domestic to one country, which you "
                    "state explicitly. Both sides are filled with it."
                ),
                "requires": [],
            },
            {
                "value": GeographyPolicy.UNSPECIFIED.value,
                "label": "Not stated",
                "description": (
                    "No geography. All 151 bank_country features stay zero; "
                    "nothing is invented."
                ),
                "requires": [],
            },
        ],
        "country_codes": selectable_country_codes(),
        "fx": fx_availability(),
    }
