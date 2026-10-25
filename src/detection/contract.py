"""
Risk Assessment Contract
========================

The model-independent output of the ML risk engine: what every consumer
downstream of scoring - persistence, the investigation layer, explanation, the
analyst UI - is allowed to rely on.  It delivers the seam that
``docs/architecture.md`` §45.9 declared and ``docs/system_architecture_decision.md``
adopts.

Why it exists
-------------
The shipped inference path returns a frame shaped by one model: GraphSAGE's
sigmoid output as ``risk_score`` and eleven columns named after the legacy
167-feature blocks.  Anything that reads those names is bound to that model.
The contract fixes *what a score means and where it came from*, so the model
behind it can be replaced without redesigning what consumes it.

What an assessment carries
--------------------------
``model``      :class:`ModelIdentity` - which model, which version, which
               feature version, which task it was evaluated for, what kind of
               number the score is, and the benchmark artifact that supports it.
``window``     :class:`AnalysisWindow` - the transactions the scores summarise.
``scores``     one row per scored entity: ``entity_id``, ``risk_score``,
               ``rank`` (1 = highest), ``decision`` (at a declared threshold, or
               null) and ``ground_truth`` (null where unknown - never 0).
``evidence``   observed quantities per entity, by name.  Evidence is what an
               investigator can check, not a model internal.

Semantics that are enforced, not assumed
----------------------------------------
* ``score_kind`` states whether the score is a calibrated probability, a
  probability-shaped model output, or a rank aggregate.  ``risk_score`` is never
  silently treated as a probability.
* ``task`` states the question the model was evaluated on.  The project's
  measured distinction between *retrospective detection* and *forward
  forecasting* (``docs/calendar_forward_benchmark.md``) travels with every
  score, so a retrospective number cannot be presented as a forecast.
* ``rank`` is a permutation of 1..n consistent with the score; ties are broken
  by ``entity_id`` so the ranking is deterministic.
* ``decision`` is present only with a threshold *and* the basis on which that
  threshold was chosen.
* Evidence may not carry a label or investigation column: ground truth has its
  own field.

Nothing here computes a score.  It describes one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CONTRACT_VERSION",
    "TASK_RETROSPECTIVE",
    "TASK_FORECAST",
    "TASKS",
    "SCORE_CALIBRATED",
    "SCORE_MODEL_PROBABILITY",
    "SCORE_RANK_AGGREGATE",
    "SCORE_KINDS",
    "EVIDENCE_FEATURE",
    "EVIDENCE_CYCLE",
    "EVIDENCE_COUNTERPARTY",
    "EVIDENCE_CYCLE_ANALYSIS",
    "ContractViolation",
    "AnalysisWindow",
    "ModelIdentity",
    "EvidenceReference",
    "RiskAssessment",
    "build_assessment",
    "entity_evidence",
]

#: Bump when the shape of an assessment changes.
CONTRACT_VERSION = "1.0"

#: "Rank entities by suspicious behaviour observed in an analysis window."
TASK_RETROSPECTIVE = "retrospective_detection"
#: "Predict suspicious activity after the window from behaviour before it."
TASK_FORECAST = "forward_forecasting"
TASKS: tuple[str, ...] = (TASK_RETROSPECTIVE, TASK_FORECAST)

#: A probability fitted to match observed frequencies on held-out data.
SCORE_CALIBRATED = "calibrated_probability"
#: A number on [0, 1] from a probabilistic model, not calibrated.
SCORE_MODEL_PROBABILITY = "model_probability"
#: An ordering-only score (e.g. the label-free rank-sum floor).
SCORE_RANK_AGGREGATE = "rank_aggregate"
SCORE_KINDS: tuple[str, ...] = (SCORE_CALIBRATED, SCORE_MODEL_PROBABILITY, SCORE_RANK_AGGREGATE)
_BOUNDED_KINDS = frozenset({SCORE_CALIBRATED, SCORE_MODEL_PROBABILITY})

EVIDENCE_FEATURE = "feature"
EVIDENCE_CYCLE = "cycle"
EVIDENCE_COUNTERPARTY = "counterparty"
EVIDENCE_CYCLE_ANALYSIS = "cycle_analysis"

_SCORE_COLUMNS: tuple[str, ...] = ("entity_id", "risk_score", "rank", "decision", "ground_truth")

#: Column names that are labels or post-investigation bookkeeping.  Kept in
#: step with ``src.experiments.dataset.FORBIDDEN_FEATURE_COLUMNS``; restated
#: here so the contract does not depend on the experiment layer.
_FORBIDDEN_EVIDENCE = frozenset(
    {"core", "payer_core", "beneficiary_core", "label", "y", "target", "investigation", "source_file"}
)


class ContractViolation(ValueError):
    """An assessment that does not satisfy the contract.  Never repaired silently."""


# ---------------------------------------------------------------------------
# Identity and window
# ---------------------------------------------------------------------------


def _ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return None if pd.isna(stamp) else stamp


@dataclass(frozen=True)
class AnalysisWindow:
    """The transactions an assessment summarises.

    ``start`` / ``end`` are the *declared* bounds (inclusive / exclusive);
    ``None`` means unbounded on that side.  ``observed_start`` /
    ``observed_end`` are *measured* from the rows actually scored, so a window
    declared wider than the data is visible as such.
    """

    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    observed_start: Optional[pd.Timestamp] = None
    observed_end: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        for name in ("start", "end", "observed_start", "observed_end"):
            object.__setattr__(self, name, _ts(getattr(self, name)))
        if self.start is not None and self.end is not None and not self.start < self.end:
            raise ContractViolation(f"analysis window start {self.start} is not before end {self.end}")

    @classmethod
    def observed(
        cls,
        dates: Any,
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> "AnalysisWindow":
        """A window whose observed bounds are measured from ``dates``."""
        parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
        return cls(
            start=start,
            end=end,
            observed_start=parsed.min() if len(parsed) else None,
            observed_end=parsed.max() if len(parsed) else None,
        )

    def to_dict(self) -> dict[str, Optional[str]]:
        return {k: None if v is None else str(v) for k, v in (
            ("start", self.start),
            ("end", self.end),
            ("observed_start", self.observed_start),
            ("observed_end", self.observed_end),
        )}


@dataclass(frozen=True)
class ModelIdentity:
    """Which model produced a score, and what that score may be taken to mean.

    Attributes
    ----------
    model_id:
        Stable name of the model line, e.g. ``"experiment3_graphsage"``.
    model_version:
        The exact fitted artefact: a checkpoint digest, or a run id.
    family:
        Candidate family (``"gradient_boosting"``, ``"legacy_graphsage"``, ...).
    feature_version:
        The feature representation the model consumed, with versions.
    task:
        One of :data:`TASKS` - the question the model was evaluated on.
    score_kind:
        One of :data:`SCORE_KINDS`.
    evaluation_reference:
        Where the evidence for this model lives (artifact path or document
        section).  ``None`` means no evaluation supports it, and that is stated.
    evaluation_regime:
        One sentence: dataset, regime and what the evaluation can support.
    """

    model_id: str
    model_version: str
    family: str
    feature_version: str
    task: str
    score_kind: str
    evaluation_reference: Optional[str] = None
    evaluation_regime: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("model_id", "model_version", "family", "feature_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractViolation(f"ModelIdentity.{name} must be a non-empty string")
        if self.task not in TASKS:
            raise ContractViolation(f"unknown task {self.task!r}; expected one of {TASKS}")
        if self.score_kind not in SCORE_KINDS:
            raise ContractViolation(
                f"unknown score_kind {self.score_kind!r}; expected one of {SCORE_KINDS}"
            )

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "family": self.family,
            "feature_version": self.feature_version,
            "task": self.task,
            "score_kind": self.score_kind,
            "evaluation_reference": self.evaluation_reference,
            "evaluation_regime": self.evaluation_regime,
        }


@dataclass(frozen=True)
class EvidenceReference:
    """One checkable fact behind an entity's position in the ranking.

    ``kind`` is :data:`EVIDENCE_FEATURE` (an observed quantity, ``ref`` its
    name, ``value`` the number), :data:`EVIDENCE_CYCLE` (``ref`` a cycle id
    resolvable through the investigation layer), :data:`EVIDENCE_COUNTERPARTY`
    or :data:`EVIDENCE_CYCLE_ANALYSIS` (the state of the analysis itself).
    """

    kind: str
    ref: str
    value: Any = None
    detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "value": self.value, "detail": self.detail}


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@dataclass
class RiskAssessment:
    """A validated set of entity risk scores plus their provenance.

    Construct with :func:`build_assessment`, which derives ``rank`` and
    ``decision``; direct construction is validated the same way.
    """

    model: ModelIdentity
    window: AnalysisWindow
    scores: pd.DataFrame
    evidence: Optional[pd.DataFrame] = None
    threshold: Optional[float] = None
    threshold_basis: Optional[str] = None
    produced_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: list[str] = field(default_factory=list)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        self.validate()

    # -- validation -------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ContractViolation` on the first rule this breaks."""
        s = self.scores
        missing = [c for c in _SCORE_COLUMNS if c not in s.columns]
        if missing:
            raise ContractViolation(f"scores frame is missing columns {missing}")
        if s["entity_id"].isna().any():
            raise ContractViolation("entity_id contains nulls")
        if s["entity_id"].duplicated().any():
            dupes = s.loc[s["entity_id"].duplicated(), "entity_id"].head(5).tolist()
            raise ContractViolation(f"entity_id is not unique, e.g. {dupes}")

        score = pd.to_numeric(s["risk_score"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(score).all():
            raise ContractViolation("risk_score contains a non-finite value")
        if self.model.score_kind in _BOUNDED_KINDS and len(score) and (
            score.min() < 0.0 or score.max() > 1.0
        ):
            raise ContractViolation(
                f"score_kind {self.model.score_kind!r} requires risk_score in [0, 1]; "
                f"observed [{score.min()}, {score.max()}]"
            )

        rank = pd.to_numeric(s["rank"], errors="coerce").to_numpy()
        n = len(s)
        if n and not np.array_equal(np.sort(rank), np.arange(1, n + 1)):
            raise ContractViolation("rank must be a permutation of 1..n")
        if n > 1:
            by_rank = score[np.argsort(rank)]
            if np.any(np.diff(by_rank) > 0):
                raise ContractViolation("rank is not consistent with descending risk_score")

        if self.threshold is None:
            if s["decision"].notna().any():
                raise ContractViolation("decision is set but no threshold was declared")
            if self.threshold_basis is not None:
                raise ContractViolation("threshold_basis given without a threshold")
        else:
            if not math.isfinite(float(self.threshold)):
                raise ContractViolation("threshold must be finite")
            if not self.threshold_basis:
                raise ContractViolation(
                    "a threshold needs a threshold_basis saying how it was chosen"
                )
            expected = (score >= float(self.threshold)).astype(np.int64)
            decision = pd.to_numeric(s["decision"], errors="coerce")
            if decision.isna().any() or not np.array_equal(decision.to_numpy(dtype=np.int64), expected):
                raise ContractViolation("decision does not equal risk_score >= threshold")

        truth = pd.to_numeric(s["ground_truth"], errors="coerce").dropna()
        if len(truth) and not set(np.unique(truth.to_numpy())).issubset({0, 1}):
            raise ContractViolation("ground_truth must be 0, 1 or null")

        if self.evidence is not None:
            ev = self.evidence
            if ev.index.duplicated().any():
                raise ContractViolation("evidence index is not unique")
            unknown = pd.Index(ev.index).difference(pd.Index(s["entity_id"]))
            if len(unknown):
                raise ContractViolation(
                    f"evidence describes {len(unknown)} entities that were not scored"
                )
            leaked = sorted(
                c for c in ev.columns if str(c).split("::")[-1].lower() in _FORBIDDEN_EVIDENCE
            )
            if leaked:
                raise ContractViolation(
                    f"evidence may not carry label or investigation columns: {leaked}"
                )

    # -- access -----------------------------------------------------------

    @property
    def n_entities(self) -> int:
        return int(len(self.scores))

    @property
    def evidence_columns(self) -> list[str]:
        return [] if self.evidence is None else [str(c) for c in self.evidence.columns]

    def ranked(self) -> pd.DataFrame:
        """Scores in rank order."""
        return self.scores.sort_values("rank").reset_index(drop=True)

    def top(self, k: int) -> pd.DataFrame:
        return self.ranked().head(int(k))

    def evidence_for(self, entity_id: Any) -> list[EvidenceReference]:
        """Observed feature evidence for one entity.  ``KeyError`` if not scored."""
        if not (self.scores["entity_id"] == entity_id).any():
            raise KeyError(entity_id)
        if self.evidence is None or entity_id not in self.evidence.index:
            return []
        row = self.evidence.loc[entity_id]
        refs = []
        for column, value in row.items():
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            refs.append(
                EvidenceReference(
                    kind=EVIDENCE_FEATURE,
                    ref=str(column),
                    value=None if pd.isna(number) else float(number),
                )
            )
        return refs

    def to_frame(self) -> pd.DataFrame:
        """Scores joined with evidence, in rank order."""
        ranked = self.ranked()
        if self.evidence is None:
            return ranked
        return ranked.join(self.evidence, on="entity_id")

    def metadata(self) -> dict[str, Any]:
        """Everything about the assessment except its per-entity rows.

        This is what a run persists beside its predictions: small, and enough
        to say which model produced the scores, on which window, under which
        feature version, for which task, at which threshold.
        """
        score = self.scores["risk_score"].astype(float)
        decided = self.scores["decision"].dropna()
        return {
            "contract_version": self.contract_version,
            "produced_at": self.produced_at,
            "model": self.model.to_dict(),
            "window": self.window.to_dict(),
            "n_entities": self.n_entities,
            "score_summary": {
                "min": float(score.min()) if len(score) else None,
                "max": float(score.max()) if len(score) else None,
                "mean": float(score.mean()) if len(score) else None,
            },
            "threshold": None if self.threshold is None else float(self.threshold),
            "threshold_basis": self.threshold_basis,
            "n_decision_positive": int(decided.astype(int).sum()) if self.threshold is not None else None,
            "has_ground_truth": bool(self.scores["ground_truth"].notna().any()),
            "evidence_columns": self.evidence_columns,
            "notes": list(self.notes),
        }


def build_assessment(
    entity_ids: Sequence[Any],
    risk_scores: Sequence[float],
    *,
    model: ModelIdentity,
    window: AnalysisWindow,
    threshold: Optional[float] = None,
    threshold_basis: Optional[str] = None,
    evidence: Optional[pd.DataFrame] = None,
    ground_truth: Optional[Sequence[Any]] = None,
    notes: Sequence[str] = (),
) -> RiskAssessment:
    """Assemble and validate an assessment, deriving ``rank`` and ``decision``.

    Rank 1 is the highest score; equal scores are ordered by ``entity_id`` so
    two runs over the same scores give the same ranking.
    """
    ids = list(entity_ids)
    score = np.asarray(risk_scores, dtype=np.float64)
    if len(ids) != len(score):
        raise ContractViolation(f"{len(ids)} entity ids but {len(score)} scores")
    # Refuse before ranking: a null id cannot take part in the deterministic
    # tie-break, and the failure must be a contract violation, not a sort error.
    if any(i is None or (isinstance(i, float) and math.isnan(i)) for i in ids):
        raise ContractViolation("entity_id contains nulls")
    frame = pd.DataFrame({"entity_id": ids, "risk_score": score})
    order = np.lexsort((frame["entity_id"].astype(str).to_numpy(), -score)) if len(ids) else np.array([], dtype=np.int64)
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(1, len(ids) + 1)
    frame["rank"] = rank
    if threshold is None:
        frame["decision"] = pd.array([pd.NA] * len(ids), dtype="Int8")
    else:
        frame["decision"] = pd.array((score >= float(threshold)).astype(np.int8), dtype="Int8")
    if ground_truth is None:
        frame["ground_truth"] = pd.array([pd.NA] * len(ids), dtype="Int8")
    else:
        frame["ground_truth"] = pd.array(list(ground_truth), dtype="Int8")
    return RiskAssessment(
        model=model,
        window=window,
        scores=frame,
        evidence=evidence,
        threshold=None if threshold is None else float(threshold),
        threshold_basis=threshold_basis,
        notes=list(notes),
    )


def entity_evidence(
    assessment: RiskAssessment,
    entity_id: Any,
    cycle_analysis: Any = None,
) -> list[EvidenceReference]:
    """Feature evidence from the assessment plus investigation-layer references.

    ``cycle_analysis`` is a Layer 10 result (``src.detection.cycles``):
    ``CycleAnalysis``, ``CycleAnalysisUnavailable`` or ``None`` (not attempted).
    Cycles are *post-prediction* evidence: they are looked up for an entity the
    model has already scored and never change its score or rank.
    """
    from src.detection.cycles import CycleAnalysis, CycleAnalysisUnavailable, summarize_entity_cycles

    refs = assessment.evidence_for(entity_id)
    if isinstance(cycle_analysis, CycleAnalysis):
        summary = summarize_entity_cycles(cycle_analysis, entity_id)
        for cycle_id in summary.cycle_ids:
            cycle = cycle_analysis.get(cycle_id)
            refs.append(
                EvidenceReference(
                    kind=EVIDENCE_CYCLE,
                    ref=cycle_id,
                    value=None if cycle is None else int(cycle.length),
                    detail="directed cycle the entity participates in (length in value)",
                )
            )
        for counterparty in summary.counterparties:
            refs.append(
                EvidenceReference(
                    kind=EVIDENCE_COUNTERPARTY,
                    ref=str(counterparty),
                    detail="shares at least one retained cycle with the entity",
                )
            )
        if summary.truncated:
            refs.append(
                EvidenceReference(
                    kind=EVIDENCE_CYCLE_ANALYSIS,
                    ref="truncated",
                    detail="cycle enumeration stopped at its bound; counts are lower bounds",
                )
            )
    elif isinstance(cycle_analysis, CycleAnalysisUnavailable):
        refs.append(
            EvidenceReference(
                kind=EVIDENCE_CYCLE_ANALYSIS,
                ref="unavailable",
                detail=f"{cycle_analysis.reason}: {cycle_analysis.error_type}",
            )
        )
    return refs
