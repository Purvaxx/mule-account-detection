"""
Layer 6 - Candidate Model Contract
====================================

The one interface every model family implements.  It is deliberately small:
four required methods and no assumption about what the model is made of.

    fit(ctx)                     train on ctx.train_idx only
    predict_scores(ctx, index)   score the entities at those positional indices
    config()                     hyperparameters, for the artifact
    emits_probabilities          whether the score is already on [0, 1]

A tabular model reads ``ctx.features``; a GNN reads ``ctx.dataset.graph`` and
the same feature matrix as node features; a heuristic reads named feature
columns and never looks at labels.  A future edge-aware temporal GNN reads
``ctx.train_edges``.  None of them require a change to the runner, the metrics,
the thresholds or the artifact format - which is the point of this layer.

Leakage boundary
----------------
``FitContext`` exposes labels **only** for the training partition
(``ctx.y_train``).  Validation labels reach threshold selection and calibration
through the runner, not through the model.  Test labels are not in the context
at all.  A candidate therefore cannot accidentally fit on data it should not
see; it would have to reach back into ``ctx.dataset.labels``, which is a
deliberate act and is what the leakage tests check for.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.experiments.dataset import ExperimentDataset
from src.experiments.splits import SplitFold

logger = logging.getLogger(__name__)

__all__ = [
    "FitContext",
    "Candidate",
    "register_candidate",
    "build_candidate",
    "available_candidates",
    "checkpoint_identity",
]


@dataclass
class FitContext:
    """Everything a candidate is allowed to see for one fold.

    Attributes
    ----------
    dataset:
        The full population.  Structure and edges are visible; ``labels`` is
        present but a candidate must not read it - use ``y_train``.
    features:
        Feature matrix indexed by entity id, row-aligned to
        ``dataset.entity_ids``.  ``None`` for candidates that build their own
        representation from the graph.
    fold:
        The partition being run.
    train_idx / val_idx:
        Positional indices.  ``val_idx`` is exposed so a model may use it for
        early stopping; using validation *labels* for that is legitimate and
        recorded, using test data is not possible from here.
    y_train:
        Training labels.  The only labels handed to a model.
    y_val:
        Validation labels, for early stopping only.  May be all-zero; a
        candidate must handle that rather than assume a positive exists.
    train_edges:
        Edge rows visible to this fold under the experiment's feature scope.
    seed:
        Fold-derived seed.
    imbalance:
        The configured class-imbalance strategy name.
    """

    dataset: ExperimentDataset
    features: Optional[pd.DataFrame]
    fold: SplitFold
    train_idx: np.ndarray
    val_idx: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    train_edges: Optional[pd.DataFrame] = None
    seed: int = 42
    imbalance: str = "class_weight"
    notes: list[str] = field(default_factory=list)

    def feature_values(self, index: np.ndarray) -> np.ndarray:
        """Feature rows for ``index`` as a float array."""
        if self.features is None:
            raise ValueError(
                "This candidate requires a feature matrix, but the experiment "
                "specified no feature sets."
            )
        return self.features.to_numpy(dtype=np.float64)[index]

    def pos_weight(self) -> Optional[float]:
        """``n_negative / n_positive`` on the training partition.

        ``None`` when the training partition has no positives - a caller must
        decide what to do rather than receive a silent infinity.
        """
        n_pos = float(self.y_train.sum())
        if n_pos == 0.0:
            return None
        n_neg = float(len(self.y_train) - n_pos)
        return n_neg / n_pos


class Candidate(ABC):
    """Base class for every model family in the benchmark."""

    #: Registry key.  Set by the ``@register_candidate`` decorator.
    family: str = "abstract"

    #: Whether ``predict_scores`` output is already a probability in [0, 1].
    emits_probabilities: bool = False

    #: Whether the runner must build a feature matrix for this candidate.
    requires_features: bool = True

    #: Whether the candidate may be fitted on one time snapshot and scored on
    #: another (``calendar_forward``).  True only for a candidate that reads
    #: nothing but ``ctx.features``, ``ctx.y_train`` and ``ctx.y_val``: under
    #: that split ``ctx.dataset.labels`` is the whole-file label, which includes
    #: the future, and no per-origin graph exists.  Opt in explicitly.
    calendar_compatible: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self._fitted: bool = False
        #: Free-form information the candidate wants in the experiment artifact
        #: (epochs run, chosen sub-model, convergence warnings, ...).
        self.fit_report: dict[str, Any] = {}

    # -- required interface ------------------------------------------------

    @abstractmethod
    def fit(self, ctx: FitContext) -> "Candidate":
        """Train on ``ctx.train_idx``.  Must not read validation or test labels
        except through ``ctx.y_val`` for early stopping."""

    @abstractmethod
    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        """Return one score per entity in ``index``, higher meaning more suspicious."""

    # -- optional interface ------------------------------------------------

    def score_features(self, features: pd.DataFrame) -> np.ndarray:
        """Score a feature matrix directly, one score per row.

        The production scoring path (``src/detection/engines.py``): a fitted
        model applied to the feature matrix of an analysis window, with no fold
        and no labels in sight.  A candidate that cannot score without its fit
        context - one that needs a graph, for instance - does not implement it,
        and that is reported rather than approximated.
        """
        raise NotImplementedError(
            f"candidate family {self.family!r} does not score a feature matrix "
            f"outside its fit context"
        )

    def config(self) -> dict[str, Any]:
        """Hyperparameters, recorded verbatim in the artifact."""
        return dict(self.params)

    def save(self, path: Path) -> Optional[Path]:
        """Persist the fitted model.  Returning ``None`` means "not persistable".

        Candidates that can be checkpointed should override this; the runner
        hashes whatever file comes back and records the digest as the
        checkpoint identity.
        """
        return None

    @property
    def fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Candidate]] = {}


def register_candidate(name: str) -> Callable[[type[Candidate]], type[Candidate]]:
    """Class decorator registering a candidate family under ``name``."""

    def decorator(cls: type[Candidate]) -> type[Candidate]:
        if name in _REGISTRY:
            raise ValueError(f"Candidate family {name!r} is already registered.")
        cls.family = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_candidates() -> list[str]:
    """Registered families, importing the shipped ones first.

    Without the import the list is empty until something calls
    ``build_candidate``, which makes the registry look broken to anyone
    introspecting it.
    """
    _load_builtin_candidates()
    return sorted(_REGISTRY)


def build_candidate(family: str, params: Optional[dict[str, Any]] = None) -> Candidate:
    """Instantiate a registered candidate.

    Importing this module does not import every model backend.  The baseline
    and legacy modules register themselves on import, so the runner imports
    them lazily; a torch-free environment can still run the tabular baselines.
    """
    if family not in _REGISTRY:
        _load_builtin_candidates()
    if family not in _REGISTRY:
        raise KeyError(
            f"Unknown candidate family {family!r}. Available: {available_candidates()}"
        )
    return _REGISTRY[family](**(params or {}))


def _load_builtin_candidates() -> None:
    """Import the shipped candidate modules, tolerating missing optional deps."""
    for module_name in ("src.experiments.baselines", "src.experiments.legacy_graphsage"):
        try:
            __import__(module_name)
        except ImportError as exc:
            logger.info("Candidate module %s unavailable: %s", module_name, exc)


def checkpoint_identity(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """SHA-256 digest and size of a saved checkpoint, or ``None``."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(p),
        "sha256": digest.hexdigest(),
        "bytes": p.stat().st_size,
    }


def config_digest(config: dict[str, Any]) -> str:
    """Stable digest of a configuration dict, for comparing runs at a glance."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
