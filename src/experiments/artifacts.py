"""
Layer 6 - Experiment Artifacts
================================

One run produces one JSON file that is sufficient, on its own, to say what was
done and what came out.  The comparison table in a later session is built from
these files, not from console scrollback.

An artifact records:

    dataset identity, version, provenance, label definition, synthetic flag
    feature sets and their versions, and the scope they were computed under
    split strategy, parameters, seed, regime and per-fold partition sizes
    model family, hyperparameters, class-imbalance strategy
    seeds, and what determinism was and was not achieved
    leakage findings
    per-fold validation and test metrics, with explicit nulls
    threshold decision and its provenance
    calibration method, whether it was fitted, and why not when it was not
    runtime, checkpoint digests, package versions, git commit
    a timestamp

Nothing is written that was not measured.  Missing metrics are ``null`` with a
reason in the fold's ``undefined`` block.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.experiments.spec import ExperimentSpec

logger = logging.getLogger(__name__)

__all__ = ["FoldOutcome", "ExperimentResult", "save_result", "load_result", "git_commit"]

#: Schema version of the artifact format.  Bump when the shape changes so old
#: files stay interpretable.
ARTIFACT_SCHEMA_VERSION = "1.0"


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Current commit hash, or ``None`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass
class FoldOutcome:
    """Everything measured for one fold."""

    fold_idx: int
    counts: dict[str, int]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    threshold: dict[str, Any]
    calibration: dict[str, Any]
    leakage: list[dict[str, Any]] = field(default_factory=list)
    fit_report: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: dict[str, float] = field(default_factory=dict)
    checkpoint: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    #: Test metrics on a declared sub-population, scored with the same frozen
    #: threshold and calibrator.  Empty unless the split defines one.
    subgroup_test_metrics: dict[str, Any] = field(default_factory=dict)
    #: Measurements about the fold that are not metrics of a partition: drift
    #: between train and test, retention ratios, the risk assessment's identity.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """The complete artifact for one experiment run."""

    spec: ExperimentSpec
    dataset_summary: dict[str, Any]
    split_summary: dict[str, Any]
    feature_summary: dict[str, Any]
    folds: list[FoldOutcome]
    aggregate: dict[str, Any]
    environment: dict[str, Any]
    determinism: dict[str, Any]
    validation_aggregate: dict[str, Any] = field(default_factory=dict)
    leakage: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    total_runtime_seconds: float = 0.0
    git_commit: Optional[str] = None
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": self.spec.to_dict(),
            "dataset": self.dataset_summary,
            "features": self.feature_summary,
            "split": self.split_summary,
            "leakage": list(self.leakage),
            "folds": [f.to_dict() for f in self.folds],
            "aggregate": self.aggregate,
            "validation_aggregate": self.validation_aggregate,
            "environment": self.environment,
            "determinism": self.determinism,
            "notes": list(self.notes),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_runtime_seconds": self.total_runtime_seconds,
            "git_commit": self.git_commit,
        }

    def headline(self) -> str:
        """A short, honest text summary.

        Deliberately terse: it states the regime, the primary metric with its
        defined-fold count, and any error-severity leakage finding.  It does not
        rank models or use the word "best".
        """
        ap = self.aggregate.get("average_precision", {})
        lift = self.aggregate.get("ap_lift", {})
        lines = [
            f"{self.spec.name}  [{self.spec.model.family}]",
            f"  dataset : {self.spec.dataset.name} v{self.spec.dataset.version}"
            + ("  (SYNTHETIC)" if self.spec.dataset.synthetic else ""),
            f"  features: {', '.join(self.spec.features.sets) or 'none'} "
            f"(scope={self.spec.features.scope})",
            f"  split   : {self.split_summary.get('strategy')} "
            f"[{self.split_summary.get('regime')}], "
            f"{self.split_summary.get('n_folds')} fold(s)",
        ]
        if ap.get("mean") is None:
            lines.append("  test AP : undefined in every fold")
        else:
            lines.append(
                f"  test AP : {ap['mean']:.4f} +/- {ap['std']:.4f} "
                f"(defined in {ap['n_defined']}/{ap['n_folds']} folds)"
            )
            if lift.get("mean") is not None:
                lines.append(f"  AP lift : {lift['mean']:.2f}x prevalence")
        warnings = [f for f in self.leakage if f.get("severity") == "warning"]
        if warnings:
            lines.append(f"  warnings: {len(warnings)} leakage/regime warning(s)")
        return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_result(result: ExperimentResult, output_dir: Path | str) -> Path:
    """Write the artifact as ``<name>__<timestamp>.json`` and return the path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (
        result.finished_at.replace(":", "").replace("-", "").replace(".", "")[:15]
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    )
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in result.spec.name)
    path = directory / f"{safe_name}__{stamp}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=_json_default)
    logger.info("Experiment artifact written to %s", path)
    return path


def load_result(path: Path | str) -> dict[str, Any]:
    """Read an artifact back as a plain dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
