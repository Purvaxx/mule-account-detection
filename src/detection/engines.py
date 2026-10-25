"""
Risk Engines
============

The ML risk engine is the one component of the system that turns observed
behaviour into a ranked list.  Everything after it - persistence, the
investigation graph, cycle and flow analysis, explanation, the analyst UI -
consumes a :class:`~src.detection.contract.RiskAssessment` and nothing else, so
the model behind an engine can be replaced without redesigning what reads it.

Two engines exist, and they deliberately share an *output* but not an *input*:

``LegacyGraphSAGEEngine``
    Wraps the frozen Experiment 3 checkpoint.  Its input is the canonical
    workspace / Reference Dataset path (Layers 1-5, the 167-feature vector),
    which the frozen model contract fixes.  It does not change a single number the
    checkpoint produces; it only states what those numbers are.
``CandidateRiskEngine``
    Wraps any fitted candidate from ``src/experiments`` that can score a feature
    matrix (logistic regression, gradient boosting, the label-free floor) over
    the ``aml_*`` feature families, computed from the transactions inside a
    declared :class:`~src.detection.contract.AnalysisWindow`.  This is the path
    the benchmarks measured.

Unifying the *input* representation - one normalized transaction table feeding
both - is recorded as follow-on work in ``docs/system_architecture_decision.md``;
the frozen checkpoint cannot consume the ``aml_*`` representation and must not
be made to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.detection.contract import (
    SCORE_MODEL_PROBABILITY,
    TASK_RETROSPECTIVE,
    AnalysisWindow,
    ContractViolation,
    ModelIdentity,
    RiskAssessment,
    build_assessment,
)

__all__ = [
    "RiskEngine",
    "TransactionWindow",
    "LegacyGraphSAGEEngine",
    "CandidateRiskEngine",
    "LEGACY_EVIDENCE_COLUMNS",
    "legacy_model_identity",
    "legacy_assessment",
    "feature_version_for",
]

#: The observed quantities the frozen inference frame carries beside its score.
#: They are evidence (inverted from the 167-feature blocks for display), not
#: inputs any other model shares.
LEGACY_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_incoming_usd",
    "total_outgoing_usd",
    "flow_imbalance",
    "in_repeated_ratio",
    "out_repeated_ratio",
    "max_parallel_edge",
    "mean_parallel_edge",
    "reciprocal_in",
    "reciprocal_out",
)

_LEGACY_REGIME = (
    "Reference Dataset (3,440 entities, 4 positives): a regression reference "
    "only. Ranking by raw transaction count alone gives AP 1.0 there, so it is "
    "not an estimate of detection performance."
)


class RiskEngine(ABC):
    """Anything that produces a :class:`RiskAssessment`."""

    @property
    @abstractmethod
    def identity(self) -> ModelIdentity:
        """The model behind this engine, and what its scores mean."""


# ---------------------------------------------------------------------------
# Legacy: the frozen Experiment 3 checkpoint
# ---------------------------------------------------------------------------


def legacy_model_identity(checkpoint_path: str | Path) -> tuple[ModelIdentity, float]:
    """Identity of the frozen checkpoint, and its decision threshold.

    ``model_version`` is the checkpoint's metadata version plus a SHA-256
    prefix of the file, so two different files can never share an identity.
    """
    import torch

    from src.experiments.candidate import checkpoint_identity

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    metadata = torch.load(path, weights_only=False).get("metadata", {})
    digest = checkpoint_identity(path)["sha256"]
    identity = ModelIdentity(
        model_id="experiment3_graphsage",
        model_version=f"{metadata.get('version', 'unversioned')}+sha256:{digest[:12]}",
        family="legacy_graphsage",
        feature_version=f"legacy_exp3_v1({int(metadata.get('input_dim', 0))})",
        task=TASK_RETROSPECTIVE,
        score_kind=SCORE_MODEL_PROBABILITY,
        evaluation_reference="docs/model_selection.md §2",
        evaluation_regime=_LEGACY_REGIME,
    )
    return identity, float(metadata["threshold"])


def _graph_dates(graph: Any) -> list[Any]:
    return [data.get("date") for _, _, data in graph.edges(data=True)]


def legacy_assessment(
    predictions: pd.DataFrame,
    *,
    identity: ModelIdentity,
    threshold: float,
    window: AnalysisWindow,
) -> RiskAssessment:
    """State a frozen-checkpoint prediction frame in contract terms.

    No value changes.  ``decision`` is recomputed from the checkpoint threshold
    and must equal the frame's ``predicted_class``; a mismatch means the frame
    did not come from this checkpoint and is refused.
    """
    missing = [c for c in ("entity_id", "risk_score", "predicted_class", *LEGACY_EVIDENCE_COLUMNS) if c not in predictions.columns]
    if missing:
        raise ContractViolation(f"legacy prediction frame is missing {missing}")
    evidence = predictions.set_index("entity_id")[list(LEGACY_EVIDENCE_COLUMNS)]
    assessment = build_assessment(
        predictions["entity_id"].tolist(),
        predictions["risk_score"].to_numpy(dtype=np.float64),
        model=identity,
        window=window,
        threshold=threshold,
        threshold_basis="frozen checkpoint metadata threshold (Experiment 3 training)",
        evidence=evidence,
        notes=[
            "risk_score is the frozen GraphSAGE sigmoid output, unchanged; it is "
            "not calibrated.",
        ],
    )
    decided = assessment.scores.set_index("entity_id")["decision"].astype(int)
    shipped = predictions.set_index("entity_id")["predicted_class"].astype(int)
    if not decided.sort_index().equals(shipped.sort_index()):
        raise ContractViolation(
            "predicted_class disagrees with the checkpoint threshold; this frame "
            "was not produced by the checkpoint it is attributed to"
        )
    return assessment


class LegacyGraphSAGEEngine(RiskEngine):
    """The frozen Experiment 3 checkpoint behind the contract.

    ``score`` runs ``src.detection.inference.predict_from_prepared_data``
    exactly as the shipped path does and restates its output.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self._identity, self.threshold = legacy_model_identity(checkpoint_path)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def assess(self, predictions: pd.DataFrame, graph: Any) -> RiskAssessment:
        """Restate a prediction frame already produced from ``graph``."""
        return legacy_assessment(
            predictions,
            identity=self._identity,
            threshold=self.threshold,
            window=AnalysisWindow.observed(_graph_dates(graph)),
        )

    def score(self, state: Any) -> RiskAssessment:
        """Score a prepared ``DetectionState``."""
        from src.detection.inference import predict_from_prepared_data

        predictions = predict_from_prepared_data(state, self.checkpoint_path)
        return self.assess(predictions, state.graph)


# ---------------------------------------------------------------------------
# Candidate: a fitted experiment-framework model over the aml_* families
# ---------------------------------------------------------------------------


def feature_version_for(feature_sets: Sequence[str]) -> str:
    """``name@version`` for each family, in order - the feature version string."""
    from src.experiments.features import get_feature_set

    return "+".join(f"{name}@{get_feature_set(name).version}" for name in feature_sets)


@dataclass(frozen=True)
class TransactionWindow:
    """Normalized transactions, the entity population, and the analysis window.

    ``edge_table`` uses the experiment layer's schema: ``source_entity``,
    ``target_entity``, ``date`` and ``amount`` or ``amount_usd``, plus any
    context columns the families read.  Only rows inside ``window`` are
    summarised; when the window is bounded, undated rows are excluded because
    they cannot be shown to fall inside it.
    """

    entity_ids: list[str]
    edge_table: pd.DataFrame
    window: AnalysisWindow
    attributes: Optional[pd.DataFrame] = None

    def rows_in_window(self) -> pd.DataFrame:
        if self.window.start is None and self.window.end is None:
            return self.edge_table
        dates = pd.to_datetime(self.edge_table["date"], errors="coerce")
        mask = dates.notna()
        if self.window.start is not None:
            mask &= dates >= self.window.start
        if self.window.end is not None:
            mask &= dates < self.window.end
        return self.edge_table.loc[mask.to_numpy()]


class CandidateRiskEngine(RiskEngine):
    """A fitted candidate scoring the ``aml_*`` families over one window.

    Parameters
    ----------
    candidate:
        A fitted :class:`~src.experiments.candidate.Candidate` implementing
        ``score_features``.
    feature_sets:
        The families it was fitted on, in the same order.
    identity:
        Its identity; ``identity.feature_version`` must equal
        :func:`feature_version_for` of ``feature_sets``.
    threshold, threshold_basis:
        Optional frozen decision threshold and how it was chosen (for the
        benchmarks: an alert budget on validation).
    evidence_columns:
        Feature columns reported as evidence; default all of them.
    """

    def __init__(
        self,
        candidate: Any,
        feature_sets: Sequence[str],
        identity: ModelIdentity,
        *,
        threshold: Optional[float] = None,
        threshold_basis: Optional[str] = None,
        evidence_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if not getattr(candidate, "fitted", False):
            raise ValueError("CandidateRiskEngine needs a fitted candidate")
        expected = feature_version_for(feature_sets)
        if identity.feature_version != expected:
            raise ContractViolation(
                f"identity.feature_version {identity.feature_version!r} does not match "
                f"the feature sets supplied ({expected!r})"
            )
        self.candidate = candidate
        self.feature_sets = tuple(feature_sets)
        self._identity = identity
        self.threshold = threshold
        self.threshold_basis = threshold_basis
        self.evidence_columns = None if evidence_columns is None else list(evidence_columns)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def score(self, window_input: TransactionWindow) -> RiskAssessment:
        """Score every entity with at least one transaction in the window."""
        from src.experiments.features import FeatureRequest, build_feature_matrix
        from src.experiments.sources import entity_codes

        rows = window_input.rows_in_window()
        request = FeatureRequest(
            entity_ids=list(window_input.entity_ids),
            edge_table=rows,
            graph=None,
            # recency / tenure are measured from the window's end when it has
            # one, so a score never depends on data after the window.
            as_of=window_input.window.end,
            attributes=window_input.attributes,
        )
        matrix = build_feature_matrix(request, self.feature_sets).values

        src, dst = entity_codes(rows, window_input.entity_ids)
        active = np.zeros(len(window_input.entity_ids), dtype=bool)
        active[np.asarray(src, dtype=np.int64)] = True
        active[np.asarray(dst, dtype=np.int64)] = True
        scored = matrix.iloc[np.flatnonzero(active)]

        scores = self.candidate.score_features(scored)
        evidence = scored if self.evidence_columns is None else scored[self.evidence_columns]
        notes = [
            f"{int((~active).sum())} entities have no transaction in the window and "
            f"are not scored.",
        ]
        return build_assessment(
            scored.index.tolist(),
            scores,
            model=self._identity,
            window=AnalysisWindow.observed(
                rows["date"], start=window_input.window.start, end=window_input.window.end
            ),
            threshold=self.threshold,
            threshold_basis=self.threshold_basis,
            evidence=evidence,
            notes=notes,
        )
