"""
Country resolution for external datasets
========================================

The Experiment 3 checkpoint encodes bank country against a **frozen** 151-entry
vocabulary (``src.features.baseline._KNOWN_COUNTRY_CODES``).  That vocabulary is
part of the model contract: it is not derived from the dataset being processed,
it is matched by exact string equality, and an unrecognised value sets no column
(there is no "other" bucket).

An external dataset frequently expresses the same fact in a different notation —
``"gb"``, ``"GBR"``, ``"United Kingdom"`` all denote the country the checkpoint
knows as ``"GB"``.  Under exact matching every one of those produces a 151-wide
block of zeros, so the dataset contributes no geographic signal at all.

This module resolves a raw value to a frozen-vocabulary code under a small set
of **named, semantically safe** rules, and reports every value it could not
resolve.  It is deliberately conservative:

* it never invents a country — an unresolvable value stays unresolved;
* it never resolves to a code outside the frozen vocabulary, because such a code
  has no column in the checkpoint and would be indistinguishable from nothing;
* it never copies one side's geography onto the other, and never derives a
  sender or receiver country from a non-geographic column.

It does **not** touch the frozen vocabulary, ``src/features/baseline.py``, the
checkpoint, or the Reference Dataset — which does not pass through the workspace
canonicalization path at all.  Resolution happens strictly upstream, on
external-dataset values, before they are handed to the unchanged feature code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field as dataclass_field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.features.baseline import _KNOWN_COUNTRY_CODES

__all__ = [
    "FROZEN_COUNTRY_CODES",
    "ResolutionRule",
    "CountryResolution",
    "GeographyNormalizationReport",
    "resolve_country",
    "normalize_country_series",
    "selectable_country_codes",
]

#: The model's country vocabulary, re-exported read-only.  Single source of
#: truth: this module never declares a second copy.
FROZEN_COUNTRY_CODES: Tuple[str, ...] = tuple(_KNOWN_COUNTRY_CODES)

_FROZEN_SET = frozenset(FROZEN_COUNTRY_CODES)


class ResolutionRule:
    """The rule under which a raw value was resolved. Recorded, never inferred."""

    EXACT = "EXACT"
    #: Whitespace stripped and upper-cased, then matched exactly.
    CASE_NORMALIZED = "CASE_NORMALIZED"
    #: ISO 3166-1 alpha-3 code mapped to its alpha-2 equivalent.
    ISO_ALPHA3 = "ISO_ALPHA3"
    #: ISO 3166-1 English country name (or a widely used short form) mapped to
    #: its alpha-2 code.
    ISO_NAME = "ISO_NAME"
    UNRESOLVED = "UNRESOLVED"


# ---------------------------------------------------------------------------
# Alpha-2 -> (alpha-3, accepted names)
# ---------------------------------------------------------------------------
# ONE table of ISO 3166-1 countries, split below by whether the frozen
# vocabulary actually has a column for each.  Both halves are needed and they
# are used for opposite purposes -- one resolves, the other only explains -- so
# keeping them as two hand-maintained tables would make "which half does this
# country belong in?" a question a maintainer could get wrong.  Membership of
# the frozen vocabulary answers it, and is not a judgement call.
#
# Several alpha-looking members of the frozen vocabulary (JH, OS, TA, TX, G8, R0,
# T1) are not ISO codes; they are Reference-Dataset-specific values whose meaning
# the repository has never established.  They are matched exactly and by case,
# and are deliberately given no name aliases.
#
# The table is not exhaustive.  It covers the jurisdictions that actually occur
# in AML and payments data; a country absent from it is simply unrecognised,
# which is honest and costs nothing.
_ISO_3166: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "AE": ("ARE", ("united arab emirates", "uae")),
    "AF": ("AFG", ("afghanistan",)),
    "AL": ("ALB", ("albania",)),
    "AR": ("ARG", ("argentina",)),
    "AT": ("AUT", ("austria",)),
    "AU": ("AUS", ("australia",)),
    "AZ": ("AZE", ("azerbaijan",)),
    "BD": ("BGD", ("bangladesh",)),
    "BE": ("BEL", ("belgium",)),
    "BG": ("BGR", ("bulgaria",)),
    "BH": ("BHR", ("bahrain",)),
    "BR": ("BRA", ("brazil",)),
    "BS": ("BHS", ("bahamas", "the bahamas")),
    "BY": ("BLR", ("belarus",)),
    "CA": ("CAN", ("canada",)),
    "CH": ("CHE", ("switzerland",)),
    "CL": ("CHL", ("chile",)),
    "CN": ("CHN", ("china", "peoples republic of china")),
    "CO": ("COL", ("colombia",)),
    "CU": ("CUB", ("cuba",)),
    "CY": ("CYP", ("cyprus",)),
    "CZ": ("CZE", ("czechia", "czech republic")),
    "DE": ("DEU", ("germany",)),
    "DK": ("DNK", ("denmark",)),
    "EE": ("EST", ("estonia",)),
    "EG": ("EGY", ("egypt",)),
    "ES": ("ESP", ("spain",)),
    "FI": ("FIN", ("finland",)),
    "FR": ("FRA", ("france",)),
    "GB": ("GBR", ("united kingdom", "uk", "great britain",
                   "united kingdom of great britain and northern ireland")),
    "GE": ("GEO", ("georgia",)),
    "GH": ("GHA", ("ghana",)),
    "GR": ("GRC", ("greece",)),
    "HK": ("HKG", ("hong kong",)),
    "HR": ("HRV", ("croatia",)),
    "HU": ("HUN", ("hungary",)),
    "ID": ("IDN", ("indonesia",)),
    "IE": ("IRL", ("ireland",)),
    "IL": ("ISR", ("israel",)),
    "IN": ("IND", ("india",)),
    "IQ": ("IRQ", ("iraq",)),
    "IR": ("IRN", ("iran", "islamic republic of iran")),
    "IS": ("ISL", ("iceland",)),
    "IT": ("ITA", ("italy",)),
    "JO": ("JOR", ("jordan",)),
    "JP": ("JPN", ("japan",)),
    "KE": ("KEN", ("kenya",)),
    "KH": ("KHM", ("cambodia",)),
    "KP": ("PRK", ("north korea", "democratic peoples republic of korea")),
    "KR": ("KOR", ("south korea", "korea", "republic of korea")),
    "KW": ("KWT", ("kuwait",)),
    "KY": ("CYM", ("cayman islands",)),
    "KZ": ("KAZ", ("kazakhstan",)),
    "LB": ("LBN", ("lebanon",)),
    "LI": ("LIE", ("liechtenstein",)),
    "LK": ("LKA", ("sri lanka",)),
    "LT": ("LTU", ("lithuania",)),
    "LU": ("LUX", ("luxembourg",)),
    "LV": ("LVA", ("latvia",)),
    "LY": ("LBY", ("libya",)),
    "MA": ("MAR", ("morocco",)),
    "MD": ("MDA", ("moldova", "republic of moldova")),
    "ME": ("MNE", ("montenegro",)),
    "MF": ("MAF", ("saint martin", "saint martin french part")),
    "MK": ("MKD", ("north macedonia", "macedonia")),
    "MM": ("MMR", ("myanmar", "burma")),
    "MT": ("MLT", ("malta",)),
    "MU": ("MUS", ("mauritius",)),
    "MX": ("MEX", ("mexico",)),
    "MY": ("MYS", ("malaysia",)),
    "NG": ("NGA", ("nigeria",)),
    "NL": ("NLD", ("netherlands", "the netherlands", "holland")),
    "NO": ("NOR", ("norway",)),
    "NR": ("NRU", ("nauru",)),
    "NZ": ("NZL", ("new zealand",)),
    "OM": ("OMN", ("oman",)),
    "PA": ("PAN", ("panama",)),
    "PE": ("PER", ("peru",)),
    "PH": ("PHL", ("philippines", "the philippines")),
    "PK": ("PAK", ("pakistan",)),
    "PL": ("POL", ("poland",)),
    "PT": ("PRT", ("portugal",)),
    "QA": ("QAT", ("qatar",)),
    "RO": ("ROU", ("romania",)),
    "RS": ("SRB", ("serbia",)),
    "RU": ("RUS", ("russia", "russian federation")),
    "SA": ("SAU", ("saudi arabia",)),
    "SC": ("SYC", ("seychelles",)),
    "SE": ("SWE", ("sweden",)),
    "SG": ("SGP", ("singapore",)),
    "SI": ("SVN", ("slovenia",)),
    "SK": ("SVK", ("slovakia",)),
    "SY": ("SYR", ("syria", "syrian arab republic")),
    "TH": ("THA", ("thailand",)),
    "TR": ("TUR", ("turkey", "turkiye", "türkiye")),
    "TW": ("TWN", ("taiwan",)),
    "TZ": ("TZA", ("tanzania", "united republic of tanzania")),
    "UA": ("UKR", ("ukraine",)),
    "UG": ("UGA", ("uganda",)),
    "US": ("USA", ("united states", "united states of america", "usa", "us",
                   "u s a", "america")),
    "UY": ("URY", ("uruguay",)),
    "VE": ("VEN", ("venezuela",)),
    "VG": ("VGB", ("british virgin islands", "virgin islands british")),
    "VN": ("VNM", ("vietnam", "viet nam")),
    "YE": ("YEM", ("yemen",)),
    "ZA": ("ZAF", ("south africa",)),
    "ZW": ("ZWE", ("zimbabwe",)),
}

# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------
# IN VOCABULARY: the checkpoint has a feature column for these, so resolving a
# notation to one of them makes a real difference to the model input.
#
# OUT OF VOCABULARY: real countries the checkpoint has no column for.  "United
# States", "USA" and "US" all denote a country the frozen 151-entry vocabulary
# does not contain.  Left to the table above alone, every one of them would be
# reported exactly like "Atlantis" or a corrupted cell -- and those are not the
# same fact.  *"We know what this says and the model has no column for it"* is a
# limit of the checkpoint; *"this is not a country we recognise"* is a problem
# with the data, and only one of them is worth an operator's time to fix.
#
# So an out-of-vocabulary value is **recognised** and reported as such, and
# otherwise treated exactly as before: code stays None, rule stays UNRESOLVED,
# resolved stays False, the value is left in the data byte-identical, and it
# sets no feature column.  Nothing here is a mapping into the vocabulary: there
# is no nearest-neighbour, no regional bucket and no substitution, because
# inventing one would place a transaction in a country it is not in.
_IN_VOCABULARY = {c: v for c, v in _ISO_3166.items() if c in _FROZEN_SET}
_OUT_OF_VOCABULARY = {c: v for c, v in _ISO_3166.items() if c not in _FROZEN_SET}


def _alpha_index(table):
    """Alpha-3 (and alpha-2, for the out-of-vocabulary half) -> alpha-2."""
    return {a3: a2 for a2, (a3, _names) in table.items()}


def _name_index(table):
    """Every accepted country name -> alpha-2."""
    return {name: a2 for a2, (_a3, names) in table.items() for name in names}


_ALPHA3_TO_ALPHA2 = _alpha_index(_IN_VOCABULARY)
_NAME_TO_ALPHA2 = _name_index(_IN_VOCABULARY)

_OOV_ALPHA_TO_ALPHA2 = dict(_alpha_index(_OUT_OF_VOCABULARY),
                            **{a2: a2 for a2 in _OUT_OF_VOCABULARY})
_OOV_NAME_TO_ALPHA2 = _name_index(_OUT_OF_VOCABULARY)


def _name_key(value: str) -> str:
    """Fold a country name for lookup: lowercase, punctuation and spacing removed."""
    folded = value.strip().lower()
    folded = re.sub(r"[.,'`’]", "", folded)
    folded = re.sub(r"\s+", " ", folded)
    return folded


@dataclass(frozen=True)
class CountryResolution:
    """The outcome of resolving one raw value. ``code`` is None when unresolved."""

    raw: str
    code: Optional[str]
    rule: str
    #: The ISO 3166-1 alpha-2 code this value denotes when it names a real
    #: country the **checkpoint has no column for** — ``"US"`` for
    #: ``"United States"``.  Purely explanatory: it is never written into the
    #: data, never used as a feature key, and never substituted for ``code``.
    #: ``None`` when the value resolved, or when it names nothing recognisable.
    recognized_as: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.code is not None

    @property
    def recognized_but_unsupported(self) -> bool:
        """A real country the frozen 151-entry vocabulary does not contain."""
        return self.code is None and self.recognized_as is not None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recognized_but_unsupported"] = self.recognized_but_unsupported
        return d


def resolve_country(value: Any) -> CountryResolution:
    """Resolve one raw geography value to a frozen-vocabulary country code.

    Rules are tried in order of decreasing certainty and the winning rule is
    recorded.  Anything that does not match a rule is returned ``UNRESOLVED``
    with ``code=None``; no fallback, no "other" bucket, no guess.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return CountryResolution(raw="", code=None, rule=ResolutionRule.UNRESOLVED)

    raw = str(value)
    stripped = raw.strip()
    if not stripped:
        return CountryResolution(raw=raw, code=None, rule=ResolutionRule.UNRESOLVED)

    # 1. Exact match against the frozen vocabulary. Numeric and symbolic
    #    Reference-Dataset codes (".D", "33") only ever match here.
    if raw in _FROZEN_SET:
        return CountryResolution(raw=raw, code=raw, rule=ResolutionRule.EXACT)

    # 2. Case/whitespace normalization. "gb" and " ES " denote the same country
    #    as "GB" and "ES" under any reading.
    if stripped in _FROZEN_SET:
        return CountryResolution(
            raw=raw, code=stripped, rule=ResolutionRule.CASE_NORMALIZED
        )
    upper = stripped.upper()
    if upper in _FROZEN_SET:
        return CountryResolution(raw=raw, code=upper, rule=ResolutionRule.CASE_NORMALIZED)

    # 3. ISO 3166-1 alpha-3.
    if upper in _ALPHA3_TO_ALPHA2:
        return CountryResolution(
            raw=raw, code=_ALPHA3_TO_ALPHA2[upper], rule=ResolutionRule.ISO_ALPHA3
        )

    # 4. ISO 3166-1 English name or a widely used short form.
    key = _name_key(stripped)
    if key in _NAME_TO_ALPHA2:
        return CountryResolution(
            raw=raw, code=_NAME_TO_ALPHA2[key], rule=ResolutionRule.ISO_NAME
        )

    # 5. The value names a real country the checkpoint has no column for.  This
    #    changes nothing about the outcome — still unresolved, still left as
    #    supplied, still sets no feature — but it records *which* fact this is,
    #    so "United States" is not reported the same way as "Atlantis".
    recognized = _OOV_ALPHA_TO_ALPHA2.get(upper) or _OOV_NAME_TO_ALPHA2.get(key)
    if recognized is not None:
        return CountryResolution(
            raw=raw, code=None, rule=ResolutionRule.UNRESOLVED,
            recognized_as=recognized,
        )

    return CountryResolution(raw=raw, code=None, rule=ResolutionRule.UNRESOLVED)


@dataclass
class GeographyNormalizationReport:
    """What normalization actually did to a column. Every count is observed."""

    column: str
    total_values: int
    resolved_values: int
    distinct_raw: int
    #: raw value -> resolved code, only for values a rule changed.
    normalized_examples: Dict[str, str]
    #: raw values no rule could resolve. These set no feature column.
    unresolved_values: List[str]
    #: rule name -> number of distinct raw values resolved under it.
    rules_applied: Dict[str, int]
    #: The subset of ``unresolved_values`` that names a real country the frozen
    #: vocabulary has no column for: raw value -> the ISO code it denotes.
    #: Reported so an operator can tell a checkpoint limitation ("United
    #: States") apart from unreadable data ("Atlantis"). Nothing is substituted.
    recognized_out_of_vocabulary: Dict[str, str] = dataclass_field(
        default_factory=dict
    )

    @property
    def resolved_ratio(self) -> float:
        return (self.resolved_values / self.total_values) if self.total_values else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolved_ratio"] = self.resolved_ratio
        return d


#: How many distinct unresolved values / normalization examples to retain in a
#: report.  Reports are persisted and rendered; an unbounded list of a
#: high-cardinality column would be neither.
_REPORT_SAMPLE_LIMIT = 25


def normalize_country_series(
    series: "pd.Series", column: str = "", limit: int = _REPORT_SAMPLE_LIMIT
) -> Tuple["pd.Series", GeographyNormalizationReport]:
    """Resolve a whole geography column, returning the new column and a report.

    Resolution is computed over the **distinct** values and then mapped back, so
    cost scales with cardinality rather than row count.

    Values that resolve are replaced by their frozen-vocabulary code.  Values
    that do not are left **exactly as they were** — not blanked, not bucketed —
    so that the downstream all-zero behaviour is unchanged for them and the raw
    value remains visible for review.
    """
    non_null = series.dropna()
    distinct = pd.Series(non_null.astype(str).unique())

    mapping: Dict[str, str] = {}
    changed: Dict[str, str] = {}
    unresolved: List[str] = []
    recognized_oov: Dict[str, str] = {}
    rules: Dict[str, int] = {}

    for raw in distinct:
        res = resolve_country(raw)
        if res.resolved:
            mapping[raw] = res.code
            rules[res.rule] = rules.get(res.rule, 0) + 1
            if res.code != raw:
                changed[raw] = res.code
        else:
            unresolved.append(raw)
            if res.recognized_but_unsupported:
                # Still unresolved, still untouched in the data.  Recorded only
                # so the report can say *why* it sets no feature column.
                recognized_oov[raw] = res.recognized_as

    # Nulls are passed through as the caller's own sentinel: "no value" is not
    # something this module has anything to say about, and rewriting it would
    # change how downstream code reads absence.
    normalized = series.map(
        lambda v: v if pd.isna(v) else mapping.get(str(v), v)
    )

    resolved_count = int(
        non_null.astype(str).isin(list(mapping.keys())).sum()
    )

    report = GeographyNormalizationReport(
        column=column or (series.name if series.name is not None else ""),
        total_values=int(len(non_null)),
        resolved_values=resolved_count,
        distinct_raw=int(len(distinct)),
        normalized_examples=dict(list(changed.items())[:limit]),
        unresolved_values=sorted(unresolved)[:limit],
        rules_applied=rules,
        recognized_out_of_vocabulary=dict(
            sorted(recognized_oov.items())[:limit]
        ),
    )
    return normalized, report


def selectable_country_codes() -> List[Dict[str, str]]:
    """The country codes a UI may legitimately offer for explicit selection.

    Only the ISO members of the frozen vocabulary are offered with a label: the
    remaining entries are Reference-Dataset-specific codes whose meaning the
    repository has never established, and offering them as a "country" in a
    picker would assert knowledge that does not exist.  They remain accepted by
    the API when supplied explicitly.
    """
    return [
        {"code": code, "name": names[0].title()}
        for code, (_a3, names) in sorted(
            _IN_VOCABULARY.items(), key=lambda kv: kv[1][1][0]
        )
    ]
