"""
Prediction trust — what a scored external dataset does and does not establish
============================================================================

The frozen Experiment 3 checkpoint was trained on the Reference Dataset alone.
Running it over an arbitrary uploaded dataset produces a number for every entity
regardless of whether that dataset resembles anything the model has seen, and
the dashboard renders that number identically either way.  On SAML-D roughly
70% of entities score above the checkpoint's 0.45 threshold — a rate no usable
alerting system produces, and a strong indication the score distribution does
not transfer.

This module turns **observable facts about the run** into signals with stated
rules, so the UI can show what the numbers rest on.  Every field below is
measured from the run itself:

* the share of entities the model flagged;
* which feature blocks were identically zero or entirely "unknown", i.e. how
  much of the 167-dimensional input the dataset actually exercised;
* whether ground truth exists at all, i.e. whether the output could even in
  principle be checked.

Deliberately absent: any comparison against a training-distribution statistic.
The checkpoint records no class prevalence and no feature moments, so such a
comparison would have to be invented.  Thresholds here are **operational
heuristics with their rule written out**, in the same shape as the existing
analyst-facing heuristic flags — never presented as a statistical test.

Nothing in this module changes a prediction.  It never re-scores, re-thresholds,
suppresses or reorders anything: ``risk_score`` and ``predicted_class`` are
exactly what the frozen model produced.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "TrustLevel",
    "SignalLevel",
    "TrustSignal",
    "PredictionTrustAssessment",
    "feature_block_coverage",
    "assess_prediction_trust",
]


class SignalLevel:
    INFO = "INFO"
    CAUTION = "CAUTION"
    CRITICAL = "CRITICAL"


class TrustLevel:
    """How much the run's own evidence supports acting on its scores.

    There is deliberately no "HIGH".  Nothing available at inference time can
    establish that a frozen model transfers to an unseen dataset, so the ceiling
    is "nothing observed argues against it".
    """

    #: Nothing observed in this run argues against the scores; still untested
    #: against the training distribution.
    UNVERIFIED = "UNVERIFIED"
    #: At least one observed property of the run limits what the scores mean.
    LIMITED = "LIMITED"
    #: An observed property makes the scores unsafe to read as an alert list.
    LOW = "LOW"


#: Share of entities flagged above which the output stops being usable as an
#: alert population.  An operational heuristic, stated as such wherever shown.
_FLAG_RATE_CAUTION = 0.25
_FLAG_RATE_CRITICAL = 0.50

#: Share of a geography column resolving into the frozen vocabulary below which
#: the residue is treated as limiting rather than merely notable.  Any
#: unresolved value is always reported; this only sets the signal's level.
#: An operational heuristic, stated as such, not a statistical test.
_GEOGRAPHY_RESOLVED_FLOOR = 0.95

#: Feature blocks of the frozen 167-vector, by column prefix.  Used to report
#: which parts of the model's input the dataset actually exercised.
_FEATURE_BLOCKS = (
    ("geography", "bank_country_", "geographic"),
    ("entity_type", "entity_type_", "entity-type"),
)


@dataclass(frozen=True)
class TrustSignal:
    """One observed property of the run.

    Same shape as the analyst-facing heuristic flags: what was observed, why it
    matters, and the explicit rule that produced it.
    """

    key: str
    title: str
    level: str
    value: str
    observed: str
    why_it_matters: str
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionTrustAssessment:
    level: str
    headline: str
    entity_count: int
    flagged_count: int
    flag_rate: float
    labels_available: bool
    feature_dimension: int
    exercised_features: int
    zero_blocks: List[str] = field(default_factory=list)
    signals: List[TrustSignal] = field(default_factory=list)
    model_version: Optional[str] = None
    assessed_at: Optional[str] = None
    trust_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["signals"] = [s.to_dict() for s in self.signals]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionTrustAssessment":
        data = dict(data)
        data["signals"] = [TrustSignal(**s) for s in data.get("signals", [])]
        return cls(**data)


def feature_block_coverage(features: "pd.DataFrame") -> Dict[str, Any]:
    """How much of the model's input the dataset actually exercised.

    Returns the number of feature columns that are non-zero for at least one
    entity, and which named blocks are identically zero across every entity.
    An identically-zero block means that part of the trained input space carried
    no information for this dataset.
    """
    if features is None or features.empty:
        return {"feature_dimension": 0, "exercised": 0, "zero_blocks": [], "blocks": {}}

    values = features.to_numpy(dtype=float, na_value=0.0)
    non_zero_cols = np.abs(values).sum(axis=0) > 0

    blocks: Dict[str, Any] = {}
    zero_blocks: List[str] = []
    for name, prefix, _label in _FEATURE_BLOCKS:
        idx = [i for i, c in enumerate(features.columns) if str(c).startswith(prefix)]
        if not idx:
            continue
        exercised = int(non_zero_cols[idx].sum())
        blocks[name] = {"width": len(idx), "exercised": exercised}
        if exercised == 0:
            zero_blocks.append(name)

    # entity_type is a one-hot with an "unknown" column: it is never all-zero,
    # but a dataset supplying no type at all lands entirely in that column,
    # which carries the same "no signal" meaning.
    unknown_col = "entity_type_unknown"
    if unknown_col in features.columns:
        known = [c for c in features.columns
                 if str(c).startswith("entity_type_") and c != unknown_col]
        if known and float(features[known].to_numpy(dtype=float).sum()) == 0.0:
            if "entity_type" not in zero_blocks:
                zero_blocks.append("entity_type")
            if "entity_type" in blocks:
                blocks["entity_type"]["all_unknown"] = True

    return {
        "feature_dimension": int(features.shape[1]),
        "exercised": int(non_zero_cols.sum()),
        "zero_blocks": zero_blocks,
        "blocks": blocks,
    }


_BLOCK_LABELS = {
    "geography": (
        "no geographic signal",
        "Every bank_country feature is zero for every entity, so the 151 "
        "geographic dimensions the model was trained on carry no information "
        "here.",
    ),
    "entity_type": (
        "no entity-type signal",
        "Every entity is entity_type_unknown, so the entity-type dimensions the "
        "model was trained on carry no information here.",
    ),
}


def _geography_signals(reports: List[Dict[str, Any]]) -> List[TrustSignal]:
    """One signal per geography column that reached the model only in part.

    The zero-block check catches a column where *nothing* resolved.  A column
    where some values resolved and others did not exercises the block, so it
    produces no zero-block signal at all -- yet every unresolved row carries an
    all-zero country vector while looking, in the raw data, exactly like a row
    that worked.  Counts come from the canonicalizer's own tally for this run.
    """
    signals: List[TrustSignal] = []
    for report in reports:
        total = int(report.get("total_values", 0) or 0)
        resolved = int(report.get("resolved_values", 0) or 0)
        if not total or resolved >= total:
            # Nothing went unresolved: there is no observation to report.
            continue

        # Derived here from the two counts rather than read back from the
        # report's own ratio field, so the share and the counts in the message
        # cannot disagree.
        unresolved_count = total - resolved
        resolved_ratio = resolved / total

        # Any unresolved value is stated, so the fact is never hidden.  Its
        # *level* follows the magnitude: a small residue is recorded without
        # downgrading the run, a large one limits what the scores mean.  A hard
        # cutoff would instead make thousands of countryless rows invisible for
        # being a small enough share.
        signal_level = (SignalLevel.INFO
                        if resolved_ratio >= _GEOGRAPHY_RESOLVED_FLOOR
                        else SignalLevel.CAUTION)

        column = report.get("column", "geography")
        unresolved = [str(v) for v in (report.get("unresolved_values") or [])]
        examples = ", ".join(repr(v) for v in unresolved[:5])
        signals.append(TrustSignal(
            key=f"geography_unresolved_{column}",
            title="GEOGRAPHY PARTIALLY UNRESOLVED",
            level=signal_level,
            value=f"{resolved_ratio:.1%} resolved",
            observed=(
                f"{column}: {unresolved_count:,} of {total:,} values "
                f"({1 - resolved_ratio:.1%}) resolved into no country in the "
                f"frozen vocabulary"
                + (f"; these include {examples}." if examples else ".")
            ),
            why_it_matters=(
                "Values the vocabulary does not contain set no country feature, "
                "so those entities carry an all-zero geographic vector while "
                "appearing, in the source data, to have a country. The "
                "geographic signal is uneven across the population rather than "
                "uniformly absent."
            ),
            rule=(
                f"Counted during canonicalization. Any unresolved value is "
                f"reported; below {_GEOGRAPHY_RESOLVED_FLOOR:.0%} resolved it "
                f"is raised to a caution. Not a statistical test."
            ),
        ))
    return signals


def _degraded_features_signal(columns: List[str]) -> Optional[TrustSignal]:
    """Which optional signal columns the dataset never supplied.

    Recorded at validation time and otherwise never seen again beside the
    scores, so an empty feature block appears without its reason.
    """
    if not columns:
        return None
    return TrustSignal(
        key="degraded_features",
        title="OPTIONAL SIGNALS ABSENT",
        level=SignalLevel.CAUTION,
        value=f"{len(columns)} absent",
        observed=("The dataset supplies no "
                  + ", ".join(str(c) for c in columns) + "."),
        why_it_matters=(
            "The model still receives a vector of exactly the trained width "
            "and order, but the corresponding features are zero or unknown "
            "for every entity. Scores rest on the remaining signal only."
        ),
        rule=("Compatibility report lists these columns as degraded: "
              "absent, and defaulted to no signal rather than to a value."),
    )


def assess_prediction_trust(
    predictions: "pd.DataFrame",
    coverage: Optional[Dict[str, Any]] = None,
    labels_present: Optional[List[str]] = None,
    model_version: Optional[str] = None,
    assessed_at: Optional[str] = None,
    geography_normalization: Optional[List[Dict[str, Any]]] = None,
    degraded_features: Optional[List[str]] = None,
) -> PredictionTrustAssessment:
    """Assess what a completed prediction run does and does not establish.

    Parameters
    ----------
    predictions:
        The persisted prediction frame; needs ``predicted_class``.
    coverage:
        Output of :func:`feature_block_coverage` for the same run.
    labels_present:
        Ground-truth label columns the dataset supplied, from the compatibility
        report.  Empty or ``None`` means the output cannot be checked at all.
    geography_normalization:
        The compatibility report's per-column record of which raw geography
        values resolved into the frozen vocabulary.  A column that resolved
        only partly still exercises the geography block, so the all-zero check
        above cannot see it.
    degraded_features:
        Optional signal columns the dataset never supplied, from the
        compatibility report.
    """
    coverage = coverage or {}
    labels_present = labels_present or []
    geography_normalization = geography_normalization or []
    degraded_features = degraded_features or []

    entity_count = int(len(predictions)) if predictions is not None else 0
    # Whether the alert rate could be measured at all.  A run whose predictions
    # carry no predicted_class has not been observed to be fine; it has not been
    # observed.  Those two must never collapse into the same verdict, because
    # the reassuring one is the default.
    measurable = bool(
        entity_count
        and predictions is not None
        and "predicted_class" in predictions.columns
    )
    flagged = int((predictions["predicted_class"] == 1).sum()) if measurable else 0
    flag_rate = (flagged / entity_count) if measurable else 0.0

    signals: List[TrustSignal] = []

    # 1. Alert rate — the single most legible symptom of a score distribution
    #    that has not transferred.
    if not measurable:
        level = SignalLevel.CRITICAL
        observed = (
            "The alert rate could not be measured: this run produced "
            f"{entity_count:,} scored entities and no predicted_class column."
        )
        why = (
            "Every other statement here describes what was observed about the "
            "run. Without the model's own classification there is nothing to "
            "observe, so this run establishes nothing at all about its scores."
        )
    elif flag_rate >= _FLAG_RATE_CRITICAL:
        level = SignalLevel.CRITICAL
        observed = (
            f"{flagged:,} of {entity_count:,} entities ({flag_rate:.0%}) score "
            f"above the checkpoint's decision threshold."
        )
        why = (
            "A majority of the population reading as suspicious is not a usable "
            "alert list. It indicates the score distribution learned on the "
            "Reference Dataset does not transfer to this dataset, not that the "
            "dataset is mostly fraudulent."
        )
    elif flag_rate >= _FLAG_RATE_CAUTION:
        level = SignalLevel.CAUTION
        observed = (
            f"{flagged:,} of {entity_count:,} entities ({flag_rate:.0%}) score "
            f"above the checkpoint's decision threshold."
        )
        why = (
            "This alert rate is high enough that the ranking is more informative "
            "than the SUSPICIOUS/NORMAL split. Treat the scores as an ordering "
            "for review, not as a classification."
        )
    else:
        level = SignalLevel.INFO
        observed = (
            f"{flagged:,} of {entity_count:,} entities ({flag_rate:.0%}) score "
            f"above the checkpoint's decision threshold."
        )
        why = (
            "The alert rate is within a range that can be reviewed. It is not "
            "evidence that the predictions are correct."
        )

    signals.append(TrustSignal(
        key="flag_rate",
        title="ALERT RATE",
        level=level,
        value=f"{flag_rate:.1%}" if measurable else "not measurable",
        observed=observed,
        why_it_matters=why,
        rule=(
            f"Operational heuristic: flagged share >= {_FLAG_RATE_CRITICAL:.0%} "
            f"is critical, >= {_FLAG_RATE_CAUTION:.0%} is caution. Not a "
            f"statistical test against the training distribution."
        ),
    ))

    # 2. Feature blocks the dataset left empty.
    for block in coverage.get("zero_blocks", []):
        short, why = _BLOCK_LABELS.get(
            block, (f"no {block} signal", f"The {block} feature block is empty.")
        )
        width = coverage.get("blocks", {}).get(block, {}).get("width", 0)
        signals.append(TrustSignal(
            key=f"zero_block_{block}",
            title=f"{block.replace('_', ' ').upper()} BLOCK EMPTY",
            level=SignalLevel.CAUTION,
            value=f"0 / {width} active",
            observed=(
                f"The dataset exercised none of the {width} {block} features "
                f"the model was trained with — {short}."
            ),
            why_it_matters=(
                why + " The model still produces a score, but from a narrower "
                "input than it was trained on."
            ),
            rule=f"Every {block} feature column is zero across all entities.",
        ))

    signals.extend(_geography_signals(geography_normalization))

    degraded_signal = _degraded_features_signal(degraded_features)
    if degraded_signal:
        signals.append(degraded_signal)

    exercised = int(coverage.get("exercised", 0))
    dimension = int(coverage.get("feature_dimension", 0))

    # 3. Whether the output can be checked at all.
    if labels_present:
        signals.append(TrustSignal(
            key="labels",
            title="GROUND TRUTH SUPPLIED",
            level=SignalLevel.INFO,
            value=", ".join(labels_present),
            observed=(
                "The dataset supplied ground-truth columns, kept strictly "
                "separate from model output."
            ),
            why_it_matters=(
                "Predictions can be compared against these labels. No such "
                "comparison has been run here; the model was not retrained or "
                "recalibrated on them."
            ),
            rule="Compatibility report lists one or more label columns.",
        ))
    else:
        signals.append(TrustSignal(
            key="labels",
            title="NO GROUND TRUTH",
            level=SignalLevel.CAUTION,
            value="none",
            observed="The dataset supplies no ground-truth label column.",
            why_it_matters=(
                "Nothing in this workspace can measure whether these predictions "
                "are right. Accuracy, precision and recall are unknown, not "
                "assumed."
            ),
            rule="Compatibility report lists no label columns.",
        ))

    # 4. The standing limitation, always shown.
    signals.append(TrustSignal(
        key="distribution_untested",
        title="DISTRIBUTION UNTESTED",
        level=SignalLevel.CAUTION,
        value=f"{exercised} / {dimension} features active" if dimension else "n/a",
        observed=(
            "This dataset has not been statistically tested against the "
            "distribution the frozen model was trained on."
        ),
        why_it_matters=(
            "The checkpoint was trained on the Reference Dataset only. Scores on "
            "any other dataset are out-of-sample and are not evidence of fraud."
        ),
        rule="Always reported for a workspace dataset.",
    ))

    if any(s.level == SignalLevel.CRITICAL for s in signals):
        level = TrustLevel.LOW
        if measurable:
            headline = (
                "These scores should not be read as an alert list. The model "
                f"flagged {flag_rate:.0%} of entities, which indicates its "
                "decision threshold does not transfer to this dataset. Use the "
                "ranking to explore structure; do not treat a flag as a finding."
            )
        else:
            headline = (
                "This run cannot be assessed: it carries no model "
                "classification, so its alert rate was never measured. Do not "
                "read these scores as an alert list."
            )
    elif any(s.level == SignalLevel.CAUTION and s.key != "distribution_untested"
             for s in signals):
        level = TrustLevel.LIMITED
        headline = (
            "Scores are usable for ranking and review, with stated limitations. "
            "They are model output on an out-of-sample dataset, not evidence of "
            "fraud."
        )
    else:
        level = TrustLevel.UNVERIFIED
        headline = (
            "Nothing observed in this run argues against the scores, but the "
            "dataset remains untested against the training distribution. Scores "
            "are model output, not evidence of fraud."
        )

    return PredictionTrustAssessment(
        level=level,
        headline=headline,
        entity_count=entity_count,
        flagged_count=flagged,
        flag_rate=flag_rate,
        labels_available=bool(labels_present),
        feature_dimension=dimension,
        exercised_features=exercised,
        zero_blocks=list(coverage.get("zero_blocks", [])),
        signals=signals,
        model_version=model_version,
        assessed_at=assessed_at,
    )
