"""
Layer 6 — Experiment Specification
====================================

A single, machine-readable description of one experiment.

The specification is the *only* place experiment-specific constants live.  A
run is fully described by ``ExperimentSpec`` plus the dataset it names, and two
runs of the same spec on the same dataset must produce the same artifact
(modulo the framework/hardware non-determinism recorded in the artifact's
environment block).

Design notes
------------
Nothing here knows what a model is.  ``ModelSpec.family`` is a registry key
resolved at run time by :mod:`src.experiments.candidate`, so a logistic
regression, a gradient-boosted tree ensemble, the legacy GraphSAGE, or a future
edge-aware temporal GNN are all named the same way.

``FeatureSpec.scope`` is a *scientific* setting, not a performance one:

``"global"``
    Feature functions see the whole graph / whole edge table, including rows
    belonging to validation and test entities.  This is the regime the legacy
    Experiment 1-5 runs used.  It is legitimate for transductive research
    framing but it must be declared, because structural features computed this
    way carry information about held-out entities.

``"train_window"``
    Feature functions see only the transactions inside the fold's training
    window (requires a time-aware split).  This is the regime an operational
    deployment actually faces.

``"origin_window"``
    Each scored snapshot sees only the transactions strictly before *its own*
    prediction origin (and, with a lookback, at or after the lookback start).
    Only the ``calendar_forward`` split defines origins, and it accepts no
    other scope; see :mod:`src.experiments.calendar`.

``"detection_window"``
    Features are computed from the transactions *inside* the detection window
    they are scored for, which is also the window the label is drawn from -
    retrospective detection, where the period is complete and the question is
    which entities in it to investigate. Only the ``detection_transfer`` split
    defines detection windows; see :mod:`src.experiments.detection_transfer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

__all__ = [
    "DatasetSpec",
    "FeatureSpec",
    "SplitSpec",
    "ModelSpec",
    "ThresholdSpec",
    "CalibrationSpec",
    "EvaluationSpec",
    "ExperimentSpec",
    "FEATURE_SCOPES",
]

#: Permitted values of :attr:`FeatureSpec.scope`.
FEATURE_SCOPES: tuple[str, ...] = (
    "global",
    "train_window",
    "origin_window",
    "detection_window",
)


def _params_dict(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a plain dict copy of ``params`` (``None`` becomes an empty dict)."""
    return dict(params) if params else {}


@dataclass(frozen=True)
class DatasetSpec:
    """Identity and provenance of the data an experiment consumes.

    Attributes
    ----------
    name:
        Stable short name, e.g. ``"reference_dkdata"``.
    version:
        Version string for *this* snapshot of the dataset.  Bump it whenever
        the underlying rows change; results carrying different versions are not
        comparable.
    provenance:
        Where the data came from, in enough detail to find it again
        (a path, a workspace UUID, ``"synthetic:<generator>@<seed>"``).
    label_definition:
        Exactly what the positive class means.  Free text, but it must state
        the rule, e.g. ``"graph node attribute 'core' is True"``.
    synthetic:
        ``True`` when the rows were generated rather than observed.  Results
        computed on synthetic data must never be reported as evidence about
        real-world performance; the flag is carried into every artifact so that
        distinction survives.
    notes:
        Free-text caveats about the dataset.
    """

    name: str
    version: str
    provenance: str
    label_definition: str
    synthetic: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            name=d["name"],
            version=d["version"],
            provenance=d["provenance"],
            label_definition=d["label_definition"],
            synthetic=bool(d.get("synthetic", False)),
            notes=tuple(d.get("notes", ())),
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Which feature families are used, and what they are allowed to see.

    Attributes
    ----------
    sets:
        Ordered registry names from :mod:`src.experiments.features`.  Ordering
        is significant: it fixes the column order of the feature matrix.
    scope:
        ``"global"`` or ``"train_window"``.  See the module docstring.
    """

    sets: tuple[str, ...] = ()
    scope: str = "global"

    def __post_init__(self) -> None:
        if self.scope not in FEATURE_SCOPES:
            raise ValueError(
                f"FeatureSpec.scope must be one of {FEATURE_SCOPES}; got {self.scope!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"sets": list(self.sets), "scope": self.scope}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureSpec":
        return cls(sets=tuple(d.get("sets", ())), scope=d.get("scope", "global"))


@dataclass(frozen=True)
class SplitSpec:
    """How the dataset is partitioned into train / validation / test.

    ``strategy`` is a registry key resolved by :mod:`src.experiments.splits`.
    ``seed`` is recorded separately from the experiment seed because a split
    must be reproducible independently of model initialisation.
    """

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "params": dict(self.params),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SplitSpec":
        return cls(
            strategy=d["strategy"],
            params=_params_dict(d.get("params")),
            seed=int(d.get("seed", 42)),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Which candidate to build, and with what hyperparameters.

    Attributes
    ----------
    family:
        Registry key from :mod:`src.experiments.candidate`.
    params:
        Hyperparameters passed to the candidate constructor.
    imbalance:
        Class-imbalance strategy name, recorded so that runs using different
        strategies are not silently compared.  Interpretation is the
        candidate's; ``"none"``, ``"class_weight"`` and ``"pos_weight"`` are the
        values currently in use.
    """

    family: str
    params: dict[str, Any] = field(default_factory=dict)
    imbalance: str = "class_weight"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "params": dict(self.params),
            "imbalance": self.imbalance,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            family=d["family"],
            params=_params_dict(d.get("params")),
            imbalance=d.get("imbalance", "class_weight"),
        )


@dataclass(frozen=True)
class ThresholdSpec:
    """How the decision threshold is chosen - always on validation, never test."""

    strategy: str = "max_f1"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ThresholdSpec":
        return cls(
            strategy=d.get("strategy", "max_f1"),
            params=_params_dict(d.get("params")),
        )


@dataclass(frozen=True)
class CalibrationSpec:
    """Score to probability mapping, fitted on validation only.

    Attributes
    ----------
    method:
        ``"none"``, ``"platt"`` or ``"isotonic"``.
    min_val_positives:
        Below this many validation positives the calibrator is *not* fitted;
        the run falls back to the identity map and records why.  Fitting a
        calibration curve on a handful of positives produces a
        confident-looking but meaningless probability.
    """

    method: str = "none"
    min_val_positives: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "min_val_positives": self.min_val_positives}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationSpec":
        return cls(
            method=d.get("method", "none"),
            min_val_positives=int(d.get("min_val_positives", 10)),
        )


@dataclass(frozen=True)
class EvaluationSpec:
    """Which ranking operating points the report covers.

    ``k_values`` are absolute alert-list sizes; ``alert_rates`` are fractional
    alert budgets (0.01 = alert on the top 1 % of scored entities).  Both are
    reported because an AML team's constraint is usually "how many alerts can
    we work per day", not a probability threshold.
    """

    k_values: tuple[int, ...] = (10, 20, 50, 100)
    alert_rates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)
    calibration_bins: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_values": list(self.k_values),
            "alert_rates": list(self.alert_rates),
            "calibration_bins": self.calibration_bins,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvaluationSpec":
        return cls(
            k_values=tuple(d.get("k_values", (10, 20, 50, 100))),
            alert_rates=tuple(d.get("alert_rates", (0.001, 0.005, 0.01, 0.05))),
            calibration_bins=int(d.get("calibration_bins", 10)),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """The complete, serializable description of one experiment."""

    name: str
    dataset: DatasetSpec
    split: SplitSpec
    model: ModelSpec
    features: FeatureSpec = field(default_factory=FeatureSpec)
    threshold: ThresholdSpec = field(default_factory=ThresholdSpec)
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    seed: int = 42
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "seed": self.seed,
            "dataset": self.dataset.to_dict(),
            "features": self.features.to_dict(),
            "split": self.split.to_dict(),
            "model": self.model.to_dict(),
            "threshold": self.threshold.to_dict(),
            "calibration": self.calibration.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentSpec":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            tags=tuple(d.get("tags", ())),
            seed=int(d.get("seed", 42)),
            dataset=DatasetSpec.from_dict(d["dataset"]),
            features=FeatureSpec.from_dict(d.get("features", {})),
            split=SplitSpec.from_dict(d["split"]),
            model=ModelSpec.from_dict(d["model"]),
            threshold=ThresholdSpec.from_dict(d.get("threshold", {})),
            calibration=CalibrationSpec.from_dict(d.get("calibration", {})),
            evaluation=EvaluationSpec.from_dict(d.get("evaluation", {})),
        )
