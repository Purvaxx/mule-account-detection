"""
Layer 10 — Bounded Cycle & Structural Evidence Detection
=========================================================

Enumerates bounded directed cycles on the **entity-level** transaction graph and
reports what was actually observed on each cycle: which entities, in which
direction, how many transactions on each leg, what amounts are known, and what
the timestamps do or do not establish.

Position in the architecture
----------------------------
This module is **post-hoc investigator evidence**. It is downstream of, and
independent from, model inference::

    Graph
      |
      +--> frozen 167-feature representation
      |        +--> Experiment 3 GraphSAGE
      |                +--> risk_score / predicted_class
      |
      +--> cycle / structural analysis        (this module)
               +--> investigator evidence

Nothing here feeds the model. No output of this module participates in
`risk_score`, `predicted_class`, the 167-feature contract, feature ordering,
trust assessment, or out-of-distribution reporting. The module reads a graph and
returns facts about it. See `docs/architecture.md` §22.

Cycle semantics
---------------
The authoritative graph is a ``networkx.MultiDiGraph`` carrying one edge per
transaction. Enumerating cycles on it directly would multiply out parallel
transactions: 50 transactions on each leg of a 3-cycle would present as 125,000
"cycles". That is a transaction-combination count, not a structural finding.

Therefore:

1. Parallel transactions are collapsed into unique ordered ``(source, target)``
   pairs.
2. Cycles are enumerated on that collapsed **directed simple** graph.
3. The original transaction bundles are reattached afterwards as per-leg
   evidence.

So ``A->B`` (50 tx), ``B->C`` (50 tx), ``C->A`` (50 tx) yields **one** cycle with
**three** legs carrying 50 transactions each.

**Direction matters.** ``A->B->C->A`` and ``A->C->B->A`` are different cycles and
receive different identities.

**Rotation does not matter.** ``A->B->C->A``, ``B->C->A->B`` and ``C->A->B->C``
are the same cycle. The canonical representation is the lexicographically
smallest rotation of the entity sequence (by ``str``); the cycle identity is a
SHA-256 digest of that sequence and is stable across runs, across bound
settings, and across datasets.

Structural facts, not verdicts
------------------------------
A ``CycleEvidence`` states what exists. It contains no fraud judgement, no
composite "cycle risk score", and no threshold that this module invented. A
cycle is not evidence of fraud; participation in one is a structural
observation. The frozen model's own ``predicted_class`` threshold remains the
only model-derived cut-off in the system, and this module does not touch it.

Relationship to ``reciprocal_out_count``
----------------------------------------
The frozen model feature ``reciprocal_out_count``
(``src/features/multigraph.py``) answers *"how many reciprocal relationships
does this entity have?"* — a scalar the model consumes.

2-cycles here answer a different question: *"which relationships are they, and
what happened in each direction?"* They are the same underlying structure viewed
at different resolution, **not** two competing counts.
:func:`reciprocal_cross_reference` states the relationship explicitly, including
the one systematic difference: ``reciprocal_out_count`` counts a self-loop
(``A->A``) as a reciprocal relationship with ``A`` itself, whereas a self-loop is
a length-1 cycle here and is excluded by the default ``min_cycle_length`` of 2.

Amount evidence
---------------
A missing amount is ``None``, never ``0.0``. Every aggregate is accompanied by
the count of transactions that actually carried an amount and by an
``amount_evidence`` state (``COMPLETE`` / ``PARTIAL`` / ``ABSENT``), so a total
computed over a partially-observed bundle can never be mistaken for a complete
one. No FX conversion is performed and no currency assumption is made: this
module reads ``amount_usd`` exactly as the graph carries it.

Temporal semantics
------------------
Whether a cycle can be traversed in time order depends on the precision the
source data actually has, which differs between datasets. Precision is measured
from the graph, never declared:

``SUB_DAY``
    At least one transaction timestamp carries a non-zero time-of-day
    component.
``DAY``
    Timestamps exist, but every one falls exactly at midnight. Intra-day
    ordering cannot be established, so ``is_time_respecting`` is ``None`` for
    every cycle. This is the Reference Dataset's situation
    (``docs/research.md`` Finding 3.3) and the rule is applied conservatively:
    no ordering claim is made even for a cycle that happens to span several
    days.
``NONE``
    No usable timestamp anywhere in the graph.

Under ``SUB_DAY`` precision a per-cycle verdict is computed over the actual
transaction bundles, and it is three-valued:

``True``
    Some rotation admits a selection of one transaction per leg with
    **strictly increasing** timestamps.
``False``
    No rotation admits such a selection even when equal timestamps are allowed.
    The cycle definitively cannot be traversed in time order.
``None``
    A non-decreasing traversal exists but a strictly increasing one does not:
    the ordering turns on transactions sharing an identical timestamp, so
    precedence is not establishable. Also returned when a leg carries no
    timestamped transaction at all.

"Not establishable" is never reported as ``False``. Every verdict carries a
``temporal_basis`` code and a plain-language ``temporal_explanation``.

Bounds
------
Enumeration is bounded structurally, by :class:`CycleAnalysisConfig`, and the
configuration in force is echoed into :attr:`CycleAnalysis.metadata` so the
bounds are visible on every result. When the retention bound stops enumeration,
``metadata.truncated`` is ``True`` and ``metadata.truncation_reason`` says why;
cycles are never silently discarded. Cycles excluded for falling under
``min_cycle_length`` are counted in
``metadata.cycles_excluded_by_min_length`` rather than dropped unrecorded.

No amount threshold and no time-window threshold exist here, because the
repository has no evidence from which to derive one.

Cost
----
The protections that matter are the length bound, the retention bound, and
collapsing parallel edges. No wall-clock timeout is implemented: a reliable,
cross-platform, interrupt-safe timeout around a C-level generator is
disproportionate to the risk once those three bounds hold. Elapsed time is
measured and reported in ``metadata.elapsed_seconds`` instead.

Not implemented here
--------------------
- UI / API integration                    -> Stage 2
- Suspicious-subgraph detection beyond cycles
- Any contribution to model input, risk score, or trust assessment
"""

from __future__ import annotations

import hashlib
import logging
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence-state vocabularies
# ---------------------------------------------------------------------------

#: Amount evidence states.  ``COMPLETE`` = every transaction carried an amount;
#: ``PARTIAL`` = some did; ``ABSENT`` = none did.
AMOUNT_COMPLETE = "COMPLETE"
AMOUNT_PARTIAL = "PARTIAL"
AMOUNT_ABSENT = "ABSENT"

#: Temporal precision of the graph, measured from its timestamps.
PRECISION_SUB_DAY = "SUB_DAY"
PRECISION_DAY = "DAY"
PRECISION_NONE = "NONE"

#: Basis codes for the ``is_time_respecting`` verdict.
TEMPORAL_NO_EVIDENCE = "NO_TEMPORAL_EVIDENCE"
TEMPORAL_DAY_LEVEL = "DAY_LEVEL_PRECISION"
TEMPORAL_INCOMPLETE = "INCOMPLETE_TEMPORAL_EVIDENCE"
TEMPORAL_STRICT_ORDER = "STRICT_ORDER_ESTABLISHED"
TEMPORAL_TIED = "TIED_TIMESTAMPS"
TEMPORAL_NO_ORDER = "NO_ORDERED_TRAVERSAL"
TEMPORAL_SELF_LOOP = "SELF_LOOP_NOT_APPLICABLE"

_TEMPORAL_EXPLANATIONS: dict[str, str] = {
    TEMPORAL_NO_EVIDENCE: (
        "No transaction on this cycle carries a usable timestamp, so no "
        "ordering can be established."
    ),
    TEMPORAL_DAY_LEVEL: (
        "The source data carries day-level precision only. Intra-day ordering "
        "cannot be established, so whether this cycle was traversed in time "
        "order is not determinable from the recorded timestamps."
    ),
    TEMPORAL_INCOMPLETE: (
        "At least one leg of this cycle carries no timestamped transaction, so "
        "a traversal cannot be checked end to end."
    ),
    TEMPORAL_STRICT_ORDER: (
        "One transaction per leg can be selected, in cycle order, with strictly "
        "increasing timestamps: a time-respecting traversal exists."
    ),
    TEMPORAL_TIED: (
        "A traversal exists only if transactions sharing an identical timestamp "
        "are assumed to be ordered. Their precedence is not recorded, so "
        "whether the cycle is time-respecting cannot be established."
    ),
    TEMPORAL_NO_ORDER: (
        "No selection of one transaction per leg, from any starting leg, is "
        "non-decreasing in time. This cycle cannot be traversed in time order."
    ),
    TEMPORAL_SELF_LOOP: (
        "A single-leg cycle has no traversal order to establish."
    ),
}

#: Basis for a computed ``amount_retention``.
RETENTION_TIME_RESPECTING = "TIME_RESPECTING_TRAVERSAL"

#: Reason codes for enumeration truncation.
TRUNCATION_MAX_CYCLES = "MAX_CYCLES_REACHED"

#: The frozen model feature this module cross-references but never replaces.
MODEL_RECIPROCAL_FEATURE = "reciprocal_out_count"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleAnalysisConfig:
    """Structural bounds for a cycle analysis run.

    Follows the project's established configuration convention (a frozen
    settings object passed in, as with ``src/ui/config.py``) rather than
    introducing a new configuration file format.  Every field is echoed into
    :attr:`CycleAnalysisMetadata.config` so the bounds in force are visible on
    the result itself.

    Attributes
    ----------
    max_cycle_length:
        Longest cycle to enumerate, in entities.  Cycles longer than this are
        never enumerated; this is the primary cost protection.
    min_cycle_length:
        Shortest cycle to retain.  The default of 2 excludes self-loops
        (length-1 cycles), which are counted in
        :attr:`CycleAnalysisMetadata.cycles_excluded_by_min_length` and whose
        participating entities are listed in
        :attr:`CycleAnalysisMetadata.self_loop_entities` rather than discarded
        unrecorded.  Set to 1 to retain them as cycles.
    max_cycles:
        Hard safety cap on the number of cycles retained.  On reaching it,
        enumeration stops and ``truncated`` is set; the total number of cycles
        in the graph is then *not* known and is not guessed.

        The default of 10,000 is set above what a complete enumeration of the
        Reference Dataset costs — 5,589 cycles at ``max_cycle_length=6``, in
        about 2 s — so the normal result is complete rather than a truncated
        sample of a computation that was cheap anyway.  It remains a cap, not a
        target: a pathological graph is still bounded, and the bound is never
        unlimited.
    max_transaction_samples_per_leg:
        Upper bound on transaction identifiers sampled per leg for
        investigator drill-down.  A display bound, not a detection threshold.

    Notes
    -----
    There is deliberately **no** minimum amount and **no** time-window bound.
    The repository holds no evidence from which either could be derived, and
    inventing one would make an arbitrary number look like a finding.
    """

    max_cycle_length: int = 6
    min_cycle_length: int = 2
    max_cycles: int = 10000
    max_transaction_samples_per_leg: int = 5

    def __post_init__(self) -> None:
        if self.min_cycle_length < 1:
            raise ValueError(
                f"min_cycle_length must be >= 1, got {self.min_cycle_length}."
            )
        if self.max_cycle_length < self.min_cycle_length:
            raise ValueError(
                f"max_cycle_length ({self.max_cycle_length}) must be >= "
                f"min_cycle_length ({self.min_cycle_length})."
            )
        if self.max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {self.max_cycles}.")
        if self.max_transaction_samples_per_leg < 0:
            raise ValueError(
                "max_transaction_samples_per_leg must be >= 0, got "
                f"{self.max_transaction_samples_per_leg}."
            )


DEFAULT_CYCLE_CONFIG = CycleAnalysisConfig()


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleLeg:
    """One directed ``source -> target`` step of a cycle, with its evidence.

    A leg bundles **every** transaction observed on that ordered pair; parallel
    transactions are never multiplied out into separate cycles.

    Attributes
    ----------
    source, target:
        The ordered entity pair.  ``source == target`` only for a self-loop.
    transaction_count:
        Number of transactions observed on this ordered pair.
    amount_usd_total, amount_usd_min, amount_usd_max:
        Aggregates over the transactions that **carried** an amount.  ``None``
        when none did.  A missing amount is never read as zero, so when
        ``amount_evidence`` is ``PARTIAL`` the total is a sum over part of the
        bundle and must be read with ``transactions_with_amount``.
    transactions_with_amount:
        How many of ``transaction_count`` carried a usable ``amount_usd``.
    amount_evidence:
        ``COMPLETE`` / ``PARTIAL`` / ``ABSENT``.
    first_timestamp, last_timestamp:
        Earliest and latest observed transaction time on this leg; ``None``
        when no transaction carried a timestamp.
    transactions_with_timestamp:
        How many of ``transaction_count`` carried a usable timestamp.
    transaction_id_sample:
        Bounded, deterministic sample of transaction identifiers (earliest
        first).  Bounded by
        :attr:`CycleAnalysisConfig.max_transaction_samples_per_leg`.
    transaction_id_sample_truncated:
        True when the leg carries more transactions than the sample shows.
    """

    source: str
    target: str
    transaction_count: int
    amount_usd_total: Optional[float]
    amount_usd_min: Optional[float]
    amount_usd_max: Optional[float]
    transactions_with_amount: int
    amount_evidence: str
    first_timestamp: Optional[pd.Timestamp]
    last_timestamp: Optional[pd.Timestamp]
    transactions_with_timestamp: int
    transaction_id_sample: tuple[Any, ...]
    transaction_id_sample_truncated: bool


@dataclass(frozen=True)
class CycleEvidence:
    """Structural evidence for one entity-level directed cycle.

    This is a description of an observed structure.  It carries no fraud
    verdict, no risk score and no threshold.

    Attributes
    ----------
    cycle_id:
        Stable identity derived from the canonical entity sequence.  Rotation
        invariant, direction sensitive, and independent of the bounds used.
    entities:
        The cycle's entities in canonical rotation order.  The cycle's edges
        are ``entities[i] -> entities[(i + 1) % length]``.
    length:
        Number of legs, equal to the number of distinct entities.
    participant_count:
        Number of distinct entities.  Equal to ``length``; reported separately
        because it is the quantity an investigator asks for.
    legs:
        One :class:`CycleLeg` per step, aligned with ``entities``.
    transaction_count:
        Total transactions across all legs.
    amount_evidence:
        ``COMPLETE`` when every leg is complete, ``ABSENT`` when no leg carries
        any amount, ``PARTIAL`` otherwise.
    legs_with_complete_amounts:
        How many legs have ``amount_evidence == COMPLETE``.
    leg_amount_totals:
        Per-leg totals in canonical order; an entry is ``None`` where the leg
        carried no amount at all.
    leg_amount_balance:
        ``min(leg totals) / max(leg totals)`` — 1.0 when every leg moves the
        same total, approaching 0 when one leg dominates.  ``None`` unless
        amount evidence is ``COMPLETE`` and every leg total is positive.
        A derived observation, not a score.
    amount_retention:
        Total on the closing leg divided by the total on the opening leg **of
        an established time-respecting traversal**.  ``None`` unless
        ``is_time_respecting`` is ``True`` and amount evidence is ``COMPLETE``,
        because a cycle has no canonical starting point without a temporal
        anchor.
    amount_retention_basis:
        Names the traversal the retention was computed along, or ``None``.
    first_transaction_at, last_transaction_at:
        Earliest and latest observed transaction time across the cycle.
    time_span_days:
        ``last - first`` in days, or ``None`` when fewer than one timestamp is
        observed.
    is_time_respecting:
        Three-valued.  See the module docstring: ``None`` means *not
        establishable*, never *false*.
    temporal_basis, temporal_explanation:
        Why the verdict is what it is.
    is_self_loop:
        True for a length-1 cycle.
    is_reciprocal_pair:
        True for a length-2 cycle — the structure the frozen model feature
        ``reciprocal_out_count`` counts as a scalar.
    """

    cycle_id: str
    entities: tuple[str, ...]
    length: int
    participant_count: int
    legs: tuple[CycleLeg, ...]
    transaction_count: int
    amount_evidence: str
    legs_with_complete_amounts: int
    leg_amount_totals: tuple[Optional[float], ...]
    leg_amount_balance: Optional[float]
    amount_retention: Optional[float]
    amount_retention_basis: Optional[str]
    first_transaction_at: Optional[pd.Timestamp]
    last_transaction_at: Optional[pd.Timestamp]
    time_span_days: Optional[float]
    is_time_respecting: Optional[bool]
    temporal_basis: str
    temporal_explanation: str
    is_self_loop: bool
    is_reciprocal_pair: bool


@dataclass(frozen=True)
class CycleAnalysisMetadata:
    """What was analysed, under which bounds, and what the bounds cost.

    Attributes
    ----------
    node_count, transaction_count:
        Size of the input graph.
    unique_directed_pair_count:
        Distinct ordered ``(source, target)`` pairs — the size of the collapsed
        graph cycles were enumerated on.  The ratio against
        ``transaction_count`` is what collapsing parallel edges saved.
    self_loop_pair_count, self_loop_entities:
        Self-loops observed, recorded whether or not the configuration retains
        them as cycles.
    config:
        The bounds in force for this run.
    cycles_enumerated:
        Cycles the enumerator produced before it stopped.  Equals the total
        number of cycles within ``max_cycle_length`` **only when**
        ``enumeration_complete`` is True.
    cycles_retained:
        Cycles present in :attr:`CycleAnalysis.cycles`.
    cycles_excluded_by_min_length:
        Cycles seen but not retained because they are shorter than
        ``min_cycle_length``.  Counted, not silently dropped.
    truncated, truncation_reason, enumeration_complete:
        Whether the retention bound stopped enumeration.  When it did, the
        total number of cycles in the graph is unknown and is not estimated.
    temporal_precision, temporal_precision_basis:
        Measured from the graph's own timestamps; see the module docstring.
    timestamped_transaction_count, unparseable_timestamp_count,
    amount_present_transaction_count:
        Evidence completeness across the whole graph.
    elapsed_seconds:
        Wall-clock cost of this analysis.
    """

    node_count: int
    transaction_count: int
    unique_directed_pair_count: int
    self_loop_pair_count: int
    self_loop_entities: frozenset[str]
    config: CycleAnalysisConfig
    cycles_enumerated: int
    cycles_retained: int
    cycles_excluded_by_min_length: int
    truncated: bool
    truncation_reason: Optional[str]
    enumeration_complete: bool
    temporal_precision: str
    temporal_precision_basis: str
    timestamped_transaction_count: int
    unparseable_timestamp_count: int
    amount_present_transaction_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CycleAnalysis:
    """Result of :func:`analyze_cycles`.

    Attributes
    ----------
    cycles:
        Retained cycles, ordered deterministically by ``(length, entities)``.
    entity_index:
        ``entity_id -> tuple of cycle_id`` for every entity appearing in a
        retained cycle.
    metadata:
        See :class:`CycleAnalysisMetadata`.
    """

    cycles: tuple[CycleEvidence, ...]
    entity_index: Mapping[str, tuple[str, ...]]
    metadata: CycleAnalysisMetadata

    def get(self, cycle_id: str) -> Optional[CycleEvidence]:
        """Return the cycle with ``cycle_id``, or ``None``."""
        for cycle in self.cycles:
            if cycle.cycle_id == cycle_id:
                return cycle
        return None


@dataclass(frozen=True)
class CycleAnalysisUnavailable:
    """Cycle analysis was attempted for this graph and did not produce a result.

    This exists so that *"analysis failed"* can never be read as *"there are no
    cycles"*.  Three states must stay distinguishable wherever cycle evidence is
    carried:

    ============================================  ===========================
    Value                                         Meaning
    ============================================  ===========================
    ``None``                                      never attempted (a stub or
                                                  lightweight context)
    :class:`CycleAnalysis` with ``cycles == ()``  attempted, completed, the
                                                  graph genuinely has none
    :class:`CycleAnalysisUnavailable`             attempted and failed; what is
                                                  known is the reason, not the
                                                  cycle count
    ============================================  ===========================

    Attributes
    ----------
    reason:
        Stable machine-readable code.  See :data:`UNAVAILABLE_ANALYSIS_FAILED`.
    error_type:
        The exception class name, e.g. ``"TypeError"``.
    detail:
        The exception message, for a log or an operator.
    """

    reason: str
    error_type: str
    detail: str


#: ``CycleAnalysisUnavailable.reason`` when enumeration raised.
UNAVAILABLE_ANALYSIS_FAILED = "CYCLE_ANALYSIS_FAILED"

#: What a caller may find on a carrier of cycle evidence.
CycleEvidenceState = Union[CycleAnalysis, CycleAnalysisUnavailable]

#: Exceptions :func:`analyze_cycles_safely` converts into an explicit
#: unavailable state instead of propagating.  Deliberately enumerated rather
#: than a bare ``Exception``: these are the ways a graph can defeat enumeration,
#: and anything else is a defect that must surface.
#:
#: * ``TypeError``      — not a MultiDiGraph, or timestamps that cannot be
#:                        compared to one another (mixed tz-aware / tz-naive).
#: * ``ValueError``     — an invalid configuration or unusable attribute value.
#: * ``MemoryError``    — a graph whose cycle structure exceeds available memory
#:                        despite the bounds.
#: * ``RecursionError`` — pathological nesting inside the enumerator.
RECOVERABLE_CYCLE_ERRORS = (TypeError, ValueError, MemoryError, RecursionError)


@dataclass(frozen=True)
class EntityCycleSummary:
    """Tier-1 / Tier-2 structural facts about one entity's cycle participation.

    Contains no verdict.  ``"Entity participates in 4 directed cycles"`` is a
    fact; whether that matters is an investigator's judgement.

    Attributes
    ----------
    entity_id:
        The entity described.
    cycle_count:
        Number of retained cycles the entity participates in.
    cycle_ids:
        Their identities, in the analysis's own order.
    length_distribution:
        ``cycle length -> count``.
    two_cycle_counterparties:
        Distinct counterparties reached through a retained 2-cycle.
    shortest_cycle_length, longest_cycle_length:
        ``None`` when the entity participates in no retained cycle.
    counterparties:
        Every distinct other entity sharing a retained cycle with this one.
    has_self_loop:
        Whether the graph carries an ``entity -> entity`` transaction, recorded
        independently of whether the configuration retains self-loops as
        cycles.
    truncated:
        Carried through from the analysis: when True these counts are lower
        bounds, because enumeration stopped early.
    """

    entity_id: str
    cycle_count: int
    cycle_ids: tuple[str, ...]
    length_distribution: Mapping[int, int]
    two_cycle_counterparties: tuple[str, ...]
    shortest_cycle_length: Optional[int]
    longest_cycle_length: Optional[int]
    counterparties: tuple[str, ...]
    has_self_loop: bool
    truncated: bool


# ---------------------------------------------------------------------------
# Internal: per-pair transaction bundles
# ---------------------------------------------------------------------------

@dataclass
class _PairBundle:
    """Aggregated evidence for one ordered ``(source, target)`` pair."""

    transaction_count: int = 0
    amount_total: float = 0.0
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    amount_count: int = 0
    #: Sorted ascending; only transactions that carried a usable timestamp.
    timestamps: list[pd.Timestamp] = field(default_factory=list)
    #: ``(sort_key, transaction_id)`` for deterministic bounded sampling.
    sample_source: list[tuple[Any, Any]] = field(default_factory=list)


def _coerce_timestamp(value: Any) -> tuple[Optional[pd.Timestamp], bool]:
    """Coerce an edge ``date`` attribute to a Timestamp.

    Returns ``(timestamp, unparseable)``.  ``(None, False)`` means the value was
    genuinely absent; ``(None, True)`` means a value was present but could not
    be read as a time, which is reported rather than silently ignored.
    """
    if value is None:
        return None, False
    if isinstance(value, pd.Timestamp):
        return (None, False) if pd.isna(value) else (value, False)
    try:
        if pd.isna(value):
            return None, False
    except (TypeError, ValueError):
        pass
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None, True
    if pd.isna(ts):
        return None, True
    return ts, False


def _coerce_amount(value: Any) -> Optional[float]:
    """Coerce an edge ``amount_usd`` attribute to a float, or ``None``.

    A missing amount is ``None``.  It is never converted to ``0.0``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if amount != amount:  # NaN
        return None
    return amount


def _build_pair_bundles(
    graph: nx.MultiDiGraph,
) -> tuple[dict[tuple[Any, Any], _PairBundle], int, int, int, int]:
    """Collapse the multigraph into per-ordered-pair evidence bundles.

    Returns
    -------
    bundles:
        ``(source, target) -> _PairBundle``.
    transaction_count, timestamped_count, unparseable_count, amount_count:
        Graph-wide evidence completeness counters.
    """
    bundles: dict[tuple[Any, Any], _PairBundle] = {}
    transaction_count = 0
    timestamped = 0
    unparseable = 0
    with_amount = 0

    for source, target, key, data in graph.edges(keys=True, data=True):
        transaction_count += 1
        bundle = bundles.get((source, target))
        if bundle is None:
            bundle = _PairBundle()
            bundles[(source, target)] = bundle

        bundle.transaction_count += 1

        tx_id = data.get("transaction_id", key)

        amount = _coerce_amount(data.get("amount_usd"))
        if amount is not None:
            with_amount += 1
            bundle.amount_count += 1
            bundle.amount_total += amount
            if bundle.amount_min is None or amount < bundle.amount_min:
                bundle.amount_min = amount
            if bundle.amount_max is None or amount > bundle.amount_max:
                bundle.amount_max = amount

        timestamp, bad = _coerce_timestamp(data.get("date"))
        if bad:
            unparseable += 1
        if timestamp is not None:
            timestamped += 1
            bundle.timestamps.append(timestamp)
            # Timestamped transactions sort first, then by time, then by id.
            bundle.sample_source.append(((0, timestamp, str(tx_id)), tx_id))
        else:
            bundle.sample_source.append(((1, pd.Timestamp.min, str(tx_id)), tx_id))

    for bundle in bundles.values():
        bundle.timestamps.sort()
        bundle.sample_source.sort(key=lambda item: item[0])

    return bundles, transaction_count, timestamped, unparseable, with_amount


def _determine_precision(
    bundles: Mapping[tuple[Any, Any], _PairBundle],
    timestamped_count: int,
) -> tuple[str, str]:
    """Measure the graph's temporal precision from its own timestamps."""
    if timestamped_count == 0:
        return (
            PRECISION_NONE,
            "No transaction in the graph carries a usable timestamp.",
        )

    sub_day = 0
    for bundle in bundles.values():
        for ts in bundle.timestamps:
            if (
                ts.hour
                or ts.minute
                or ts.second
                or ts.microsecond
                or getattr(ts, "nanosecond", 0)
            ):
                sub_day += 1

    if sub_day:
        return (
            PRECISION_SUB_DAY,
            f"{sub_day} of {timestamped_count} transaction timestamp(s) carry a "
            "non-zero time-of-day component, so transactions within a day can "
            "be ordered.",
        )

    return (
        PRECISION_DAY,
        f"All {timestamped_count} transaction timestamp(s) fall exactly at "
        "midnight, so the source carries day-level precision only and "
        "intra-day ordering cannot be established.",
    )


# ---------------------------------------------------------------------------
# Internal: canonical identity
# ---------------------------------------------------------------------------

def canonical_rotation(entities: Sequence[Any]) -> tuple[Any, ...]:
    """Return the rotation-invariant canonical ordering of a cycle.

    Direction is preserved (the sequence is never reversed); only the starting
    point is normalised, to the rotation whose ``str`` sequence is
    lexicographically smallest.  Cycle entities are distinct, so the minimum is
    unique.
    """
    nodes = list(entities)
    size = len(nodes)
    if size <= 1:
        return tuple(nodes)
    best = min(
        range(size),
        key=lambda start: tuple(str(nodes[(start + off) % size]) for off in range(size)),
    )
    return tuple(nodes[(best + off) % size] for off in range(size))


def cycle_identity(entities: Sequence[Any]) -> str:
    """Return a stable identity for the cycle through ``entities``.

    Rotation invariant, direction sensitive, and independent of the analysis
    bounds — the same structure receives the same identity in every run and on
    every dataset.  The separator is ASCII unit separator (``\\x1f``), which
    cannot occur in an entity identifier.
    """
    canonical = canonical_rotation(entities)
    digest = hashlib.sha256(
        "\x1f".join(str(node) for node in canonical).encode("utf-8")
    ).hexdigest()
    return f"cyc_{digest[:16]}"


# ---------------------------------------------------------------------------
# Internal: temporal traversal
# ---------------------------------------------------------------------------

def _greedy_traversal(
    leg_timestamps: Sequence[Sequence[pd.Timestamp]],
    start: int,
    strict: bool,
) -> Optional[list[pd.Timestamp]]:
    """Earliest-feasible traversal from ``start``, or ``None`` if none exists.

    Picking the earliest admissible timestamp at each step never forecloses a
    later step, so the greedy result is optimal for a fixed starting leg.
    """
    size = len(leg_timestamps)
    chosen: list[pd.Timestamp] = []
    previous: Optional[pd.Timestamp] = None

    for offset in range(size):
        options = leg_timestamps[(start + offset) % size]
        if not options:
            return None
        if previous is None:
            candidate = options[0]
        else:
            index = (
                bisect_right(options, previous)
                if strict
                else bisect_left(options, previous)
            )
            if index >= len(options):
                return None
            candidate = options[index]
        chosen.append(candidate)
        previous = candidate

    return chosen


def _assess_temporal(
    leg_timestamps: Sequence[Sequence[pd.Timestamp]],
    precision: str,
    is_self_loop: bool,
) -> tuple[Optional[bool], str, Optional[int]]:
    """Decide whether the cycle admits a time-respecting traversal.

    Returns ``(verdict, basis, start_leg_index)``.  ``start_leg_index`` is the
    opening leg of the established strictly-increasing traversal, and is
    ``None`` unless the verdict is ``True``.
    """
    if is_self_loop:
        return None, TEMPORAL_SELF_LOOP, None
    if precision == PRECISION_NONE:
        return None, TEMPORAL_NO_EVIDENCE, None
    if precision == PRECISION_DAY:
        # Deliberately conservative: no ordering claim is made at day-level
        # precision, even for a cycle spanning several days.
        return None, TEMPORAL_DAY_LEVEL, None
    if any(len(options) == 0 for options in leg_timestamps):
        return None, TEMPORAL_INCOMPLETE, None

    for start in range(len(leg_timestamps)):
        if _greedy_traversal(leg_timestamps, start, strict=True) is not None:
            return True, TEMPORAL_STRICT_ORDER, start

    for start in range(len(leg_timestamps)):
        if _greedy_traversal(leg_timestamps, start, strict=False) is not None:
            return None, TEMPORAL_TIED, None

    return False, TEMPORAL_NO_ORDER, None


# ---------------------------------------------------------------------------
# Internal: evidence assembly
# ---------------------------------------------------------------------------

def _build_leg(
    source: Any,
    target: Any,
    bundle: _PairBundle,
    config: CycleAnalysisConfig,
) -> CycleLeg:
    if bundle.amount_count == 0:
        evidence = AMOUNT_ABSENT
        total = None
        low = None
        high = None
    else:
        total = bundle.amount_total
        low = bundle.amount_min
        high = bundle.amount_max
        evidence = (
            AMOUNT_COMPLETE
            if bundle.amount_count == bundle.transaction_count
            else AMOUNT_PARTIAL
        )

    limit = config.max_transaction_samples_per_leg
    sample = tuple(tx_id for _, tx_id in bundle.sample_source[:limit])

    return CycleLeg(
        source=source,
        target=target,
        transaction_count=bundle.transaction_count,
        amount_usd_total=total,
        amount_usd_min=low,
        amount_usd_max=high,
        transactions_with_amount=bundle.amount_count,
        amount_evidence=evidence,
        first_timestamp=bundle.timestamps[0] if bundle.timestamps else None,
        last_timestamp=bundle.timestamps[-1] if bundle.timestamps else None,
        transactions_with_timestamp=len(bundle.timestamps),
        transaction_id_sample=sample,
        transaction_id_sample_truncated=len(sample) < bundle.transaction_count,
    )


def _build_cycle(
    entities: tuple[Any, ...],
    bundles: Mapping[tuple[Any, Any], _PairBundle],
    precision: str,
    config: CycleAnalysisConfig,
) -> CycleEvidence:
    size = len(entities)
    is_self_loop = size == 1

    legs: list[CycleLeg] = []
    leg_timestamps: list[list[pd.Timestamp]] = []
    for index, source in enumerate(entities):
        target = entities[(index + 1) % size]
        bundle = bundles[(source, target)]
        legs.append(_build_leg(source, target, bundle, config))
        leg_timestamps.append(bundle.timestamps)

    transaction_count = sum(leg.transaction_count for leg in legs)

    complete_legs = sum(1 for leg in legs if leg.amount_evidence == AMOUNT_COMPLETE)
    absent_legs = sum(1 for leg in legs if leg.amount_evidence == AMOUNT_ABSENT)
    if complete_legs == size:
        amount_evidence = AMOUNT_COMPLETE
    elif absent_legs == size:
        amount_evidence = AMOUNT_ABSENT
    else:
        amount_evidence = AMOUNT_PARTIAL

    totals = tuple(leg.amount_usd_total for leg in legs)

    balance: Optional[float] = None
    if amount_evidence == AMOUNT_COMPLETE and all(
        total is not None and total > 0 for total in totals
    ):
        balance = float(min(totals)) / float(max(totals))

    verdict, basis, start = _assess_temporal(leg_timestamps, precision, is_self_loop)

    retention: Optional[float] = None
    retention_basis: Optional[str] = None
    if (
        verdict is True
        and start is not None
        and amount_evidence == AMOUNT_COMPLETE
        and totals[start] is not None
        and totals[start] > 0
    ):
        closing = totals[(start + size - 1) % size]
        if closing is not None:
            retention = float(closing) / float(totals[start])
            retention_basis = RETENTION_TIME_RESPECTING

    observed = [ts for options in leg_timestamps for ts in options]
    first_at = min(observed) if observed else None
    last_at = max(observed) if observed else None
    span: Optional[float] = None
    if first_at is not None and last_at is not None:
        span = (last_at - first_at).total_seconds() / 86400.0

    return CycleEvidence(
        cycle_id=cycle_identity(entities),
        entities=entities,
        length=size,
        participant_count=len(set(entities)),
        legs=tuple(legs),
        transaction_count=transaction_count,
        amount_evidence=amount_evidence,
        legs_with_complete_amounts=complete_legs,
        leg_amount_totals=totals,
        leg_amount_balance=balance,
        amount_retention=retention,
        amount_retention_basis=retention_basis,
        first_transaction_at=first_at,
        last_transaction_at=last_at,
        time_span_days=span,
        is_time_respecting=verdict,
        temporal_basis=basis,
        temporal_explanation=_TEMPORAL_EXPLANATIONS[basis],
        is_self_loop=is_self_loop,
        is_reciprocal_pair=size == 2,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_cycles(
    graph: nx.MultiDiGraph,
    config: Optional[CycleAnalysisConfig] = None,
) -> CycleAnalysis:
    """Enumerate bounded entity-level directed cycles and describe each one.

    Parameters
    ----------
    graph:
        The authoritative ``networkx.MultiDiGraph`` produced by Layer 4
        (``src.graph.builder.build_graph``), for either the Reference Dataset
        or a workspace.  **Not modified.**  Edge attributes read: ``amount_usd``
        (optional), ``date`` (optional), ``transaction_id`` (falling back to the
        multigraph edge key).
    config:
        Structural bounds.  Defaults to :data:`DEFAULT_CYCLE_CONFIG`
        (length <= 6, at most 10,000 retained cycles).

    Returns
    -------
    CycleAnalysis
        Retained cycles, an entity index, and metadata recording the bounds in
        force, what they excluded, the graph's measured temporal precision, and
        the elapsed time.

    Raises
    ------
    TypeError
        If ``graph`` is not a ``networkx.MultiDiGraph``.  The multigraph is the
        authoritative representation throughout this project; accepting a
        collapsed ``DiGraph`` would silently discard the per-leg transaction
        evidence that makes a cycle investigable.

    Notes
    -----
    This function is a pure read.  It contributes nothing to the 167-feature
    representation, the frozen checkpoint, ``risk_score`` or
    ``predicted_class``.
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(
            "analyze_cycles requires the authoritative networkx.MultiDiGraph "
            f"(Layer 4 output); got {type(graph).__name__}."
        )

    config = config or DEFAULT_CYCLE_CONFIG
    started = time.perf_counter()

    bundles, transaction_count, timestamped, unparseable, with_amount = (
        _build_pair_bundles(graph)
    )
    precision, precision_basis = _determine_precision(bundles, timestamped)

    self_loop_pairs = [pair for pair in bundles if pair[0] == pair[1]]
    self_loop_entities = frozenset(pair[0] for pair in self_loop_pairs)

    # Collapse to the simple directed graph, relabelled to integers.
    #
    # Two reasons, both about reproducibility rather than speed.  Nodes are
    # inserted in sorted entity order so the enumerator walks the graph the same
    # way on every run.  They are relabelled to their position in that order
    # because ``nx.simple_cycles`` iterates internal sets, and set iteration
    # order for *strings* depends on the interpreter's per-process hash seed —
    # so on a run that hits the retention bound, *which* cycles survived would
    # differ between processes.  Integer keys hash to themselves, so the walk is
    # identical everywhere and a truncated result is reproducible.
    ordered_nodes = sorted(graph.nodes(), key=str)
    node_index = {node: position for position, node in enumerate(ordered_nodes)}

    collapsed = nx.DiGraph()
    collapsed.add_nodes_from(range(len(ordered_nodes)))
    collapsed.add_edges_from(
        (node_index[source], node_index[target])
        for source, target in sorted(
            bundles.keys(), key=lambda pair: (str(pair[0]), str(pair[1]))
        )
    )

    logger.info(
        "Cycle analysis: %d entities, %d transactions collapsed to %d unique "
        "directed pair(s); enumerating cycles of length %d-%d, retaining at "
        "most %d.",
        graph.number_of_nodes(),
        transaction_count,
        len(bundles),
        config.min_cycle_length,
        config.max_cycle_length,
        config.max_cycles,
    )

    enumerated = 0
    excluded_short = 0
    truncated = False
    retained: list[tuple[Any, ...]] = []

    for found in nx.simple_cycles(collapsed, length_bound=config.max_cycle_length):
        enumerated += 1
        if len(found) < config.min_cycle_length:
            excluded_short += 1
            continue
        retained.append(canonical_rotation([ordered_nodes[position] for position in found]))
        if len(retained) >= config.max_cycles:
            truncated = True
            break

    cycles = [_build_cycle(entities, bundles, precision, config) for entities in retained]
    cycles.sort(key=lambda c: (c.length, tuple(str(e) for e in c.entities)))

    index: dict[str, list[str]] = {}
    for cycle in cycles:
        for entity in cycle.entities:
            index.setdefault(entity, []).append(cycle.cycle_id)

    elapsed = time.perf_counter() - started

    if truncated:
        logger.warning(
            "Cycle enumeration truncated at the retention bound of %d; the "
            "total number of cycles within length %d is not known.",
            config.max_cycles,
            config.max_cycle_length,
        )

    logger.info(
        "Cycle analysis complete: %d retained, %d excluded by min length, "
        "truncated=%s, precision=%s, %.3fs.",
        len(cycles),
        excluded_short,
        truncated,
        precision,
        elapsed,
    )

    metadata = CycleAnalysisMetadata(
        node_count=graph.number_of_nodes(),
        transaction_count=transaction_count,
        unique_directed_pair_count=len(bundles),
        self_loop_pair_count=len(self_loop_pairs),
        self_loop_entities=self_loop_entities,
        config=config,
        cycles_enumerated=enumerated,
        cycles_retained=len(cycles),
        cycles_excluded_by_min_length=excluded_short,
        truncated=truncated,
        truncation_reason=TRUNCATION_MAX_CYCLES if truncated else None,
        enumeration_complete=not truncated,
        temporal_precision=precision,
        temporal_precision_basis=precision_basis,
        timestamped_transaction_count=timestamped,
        unparseable_timestamp_count=unparseable,
        amount_present_transaction_count=with_amount,
        elapsed_seconds=elapsed,
    )

    return CycleAnalysis(
        cycles=tuple(cycles),
        entity_index={key: tuple(value) for key, value in index.items()},
        metadata=metadata,
    )


def analyze_cycles_safely(
    graph: nx.MultiDiGraph,
    config: Optional[CycleAnalysisConfig] = None,
) -> CycleEvidenceState:
    """:func:`analyze_cycles`, with enumeration failure as a value not a raise.

    Cycle evidence is **investigative enrichment**, not a prerequisite for model
    inference.  A graph that defeats enumeration must not cost the caller a
    valid ``risk_score``, ``predicted_class``, compatibility status or trust
    assessment — so this wrapper exists for callers that are assembling those
    alongside cycle evidence.

    Returns
    -------
    CycleAnalysis
        On success — including the successful finding that a graph has no
        cycles, which is an empty ``cycles`` tuple and **not** a failure.
    CycleAnalysisUnavailable
        When enumeration raised one of :data:`RECOVERABLE_CYCLE_ERRORS`.  The
        reason is carried, so nothing downstream can read the failure as
        "no cycles found".

    Notes
    -----
    The catch is deliberately narrow.  Only the enumerated recoverable errors
    are converted; every other exception propagates, because it would be a
    defect in this module rather than a property of the graph.  Failures are
    logged with a traceback so a degraded run is never silent.
    """
    try:
        return analyze_cycles(graph, config)
    except RECOVERABLE_CYCLE_ERRORS as error:
        logger.warning(
            "Cycle analysis unavailable (%s: %s). Model output is unaffected; "
            "this is investigator enrichment only.",
            type(error).__name__,
            error,
            exc_info=True,
        )
        return CycleAnalysisUnavailable(
            reason=UNAVAILABLE_ANALYSIS_FAILED,
            error_type=type(error).__name__,
            detail=str(error),
        )


def summarize_entity_cycles(
    analysis: CycleAnalysis,
    entity_id: str,
) -> EntityCycleSummary:
    """Summarise one entity's participation in the retained cycles.

    Returns Tier-1 structural facts and Tier-2 derived properties only.  An
    entity that participates in no retained cycle gets a zero summary, not an
    error — absence of a cycle is a fact too.
    """
    wanted = set(analysis.entity_index.get(entity_id, ()))
    members = [cycle for cycle in analysis.cycles if cycle.cycle_id in wanted]

    lengths: dict[int, int] = {}
    counterparties: set[str] = set()
    two_cycle_partners: set[str] = set()

    for cycle in members:
        lengths[cycle.length] = lengths.get(cycle.length, 0) + 1
        others = [node for node in cycle.entities if node != entity_id]
        counterparties.update(others)
        if cycle.is_reciprocal_pair:
            two_cycle_partners.update(others)

    return EntityCycleSummary(
        entity_id=entity_id,
        cycle_count=len(members),
        cycle_ids=tuple(cycle.cycle_id for cycle in members),
        length_distribution=dict(sorted(lengths.items())),
        two_cycle_counterparties=tuple(sorted(two_cycle_partners, key=str)),
        shortest_cycle_length=min(lengths) if lengths else None,
        longest_cycle_length=max(lengths) if lengths else None,
        counterparties=tuple(sorted(counterparties, key=str)),
        has_self_loop=entity_id in analysis.metadata.self_loop_entities,
        truncated=analysis.metadata.truncated,
    )


def reciprocal_cross_reference(
    summary: EntityCycleSummary,
    reciprocal_out_count: Optional[float],
) -> dict[str, Any]:
    """Relate an entity's 2-cycles to the frozen model feature that counts them.

    ``reciprocal_out_count`` is a **model input**: a scalar answering *how many*
    reciprocal relationships an entity has.  The 2-cycles here answer *which*
    ones, and what happened in each direction.  They are the same structure at
    two resolutions and must not be presented as independent competing counts.

    The two differ systematically in exactly one way: ``reciprocal_out_count``
    counts a self-loop as a reciprocal relationship with the entity itself,
    whereas a self-loop is a length-1 cycle here and is excluded by the default
    ``min_cycle_length``.  That offset is applied explicitly below rather than
    left to look like a discrepancy.

    Parameters
    ----------
    summary:
        Output of :func:`summarize_entity_cycles`.
    reciprocal_out_count:
        The entity's value from ``src/features/multigraph.py``, or ``None``
        when unavailable.

    Returns
    -------
    dict
        ``consistent`` is ``None`` — not ``False`` — whenever the comparison
        cannot be made (no feature value, or a truncated enumeration whose
        counts are lower bounds).
    """
    self_loop_offset = 1 if summary.has_self_loop else 0
    observed = len(summary.two_cycle_counterparties)

    consistent: Optional[bool]
    if reciprocal_out_count is None or summary.truncated:
        consistent = None
    else:
        consistent = int(round(float(reciprocal_out_count))) == observed + self_loop_offset

    note = (
        f"'{MODEL_RECIPROCAL_FEATURE}' is the model input counting how many "
        "reciprocal relationships this entity has. The 2-cycles listed here are "
        "those same relationships, named, with the transactions observed in "
        "each direction. They are one structure at two resolutions, not two "
        "independent counts."
    )
    if self_loop_offset:
        note += (
            f" '{MODEL_RECIPROCAL_FEATURE}' also counts this entity's self-loop "
            "as a reciprocal relationship with itself; a self-loop is a "
            "length-1 cycle here and is not listed among the counterparties."
        )
    if summary.truncated:
        note += (
            " Cycle enumeration was truncated, so the listed 2-cycles are a "
            "lower bound and cannot be reconciled against the feature value."
        )

    return {
        "model_feature": MODEL_RECIPROCAL_FEATURE,
        "model_feature_value": reciprocal_out_count,
        "two_cycle_counterparty_count": observed,
        "two_cycle_counterparties": summary.two_cycle_counterparties,
        "self_loop_offset": self_loop_offset,
        "consistent": consistent,
        "note": note,
    }


def describe_analysis(analysis: CycleAnalysis) -> str:
    """Return a human-readable summary of a :class:`CycleAnalysis`."""
    meta = analysis.metadata
    lines = [
        "=" * 62,
        "CYCLE ANALYSIS SUMMARY (investigator evidence; not model input)",
        "=" * 62,
        f"Entities                 : {meta.node_count}",
        f"Transactions             : {meta.transaction_count}",
        f"Unique directed pairs    : {meta.unique_directed_pair_count}",
        f"Self-loop pairs          : {meta.self_loop_pair_count}",
        "",
        f"Bounds                   : length {meta.config.min_cycle_length}-"
        f"{meta.config.max_cycle_length}, max {meta.config.max_cycles} retained",
        f"Cycles enumerated        : {meta.cycles_enumerated}",
        f"Cycles retained          : {meta.cycles_retained}",
        f"Excluded by min length   : {meta.cycles_excluded_by_min_length}",
        f"Enumeration complete     : {meta.enumeration_complete}",
        f"Truncated                : {meta.truncated}"
        + (f" ({meta.truncation_reason})" if meta.truncation_reason else ""),
        "",
        f"Temporal precision       : {meta.temporal_precision}",
        f"  {meta.temporal_precision_basis}",
        f"Timestamped transactions : {meta.timestamped_transaction_count}",
        f"Unparseable timestamps   : {meta.unparseable_timestamp_count}",
        f"Transactions with amount : {meta.amount_present_transaction_count}",
        "",
        f"Elapsed                  : {meta.elapsed_seconds:.3f}s",
    ]

    if analysis.cycles:
        by_length: dict[int, int] = {}
        for cycle in analysis.cycles:
            by_length[cycle.length] = by_length.get(cycle.length, 0) + 1
        lines.append("")
        lines.append("Retained cycles by length:")
        for length in sorted(by_length):
            lines.append(f"  length {length}: {by_length[length]}")

    lines.append("=" * 62)
    return "\n".join(lines)
