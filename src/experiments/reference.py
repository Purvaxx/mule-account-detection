"""
Layer 6 - Reference Dataset Loader
====================================

Builds an :class:`~src.experiments.dataset.ExperimentDataset` from the
Reference Dataset by running the existing Layer 1-3 pipeline unchanged
(ingestion -> preprocessing -> graph construction).

This is the only place the framework knows how the project's own data is
loaded.  Everything above it works on ``ExperimentDataset`` and does not care
where the rows came from, which is what lets a future workspace-backed dataset
plug in by adding one more loader here.

Provenance
----------
The dataset version is derived from the source file's size and modification
time rather than asserted.  If the file changes, the version string changes,
and artifacts produced before and after are visibly not comparable.  Nothing
about the label definition is inferred: the positive class is the ``core`` node
attribute that Layer 3 sets, and that rule is written into the spec.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from src.experiments.dataset import ExperimentDataset, dataset_from_graph_result
from src.experiments.spec import DatasetSpec

logger = logging.getLogger(__name__)

__all__ = ["load_reference_dataset", "DEFAULT_REFERENCE_PATH"]

#: The Reference Dataset is not vendored into the repository; it sits one level
#: above the project directory.  Resolved from this file so the loader does not
#: depend on the working directory.
DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parents[2].parent / "dkdata.json"


def _content_version(path: Path) -> str:
    """A short version string derived from the file itself.

    Hashing 13 MB on every run is wasteful, so this uses size plus modification
    time, which is enough to notice that the file is not the one a previous
    artifact was produced from.  It is a change detector, not a content hash,
    and the artifact records it as such.
    """
    stat = path.stat()
    digest = hashlib.sha256(f"{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()
    return f"size{stat.st_size}-{digest[:8]}"


def load_reference_dataset(
    path: Optional[Path | str] = None,
    *,
    name: str = "reference_dkdata",
) -> ExperimentDataset:
    """Run Layers 1-3 on the Reference Dataset and adapt the result.

    Raises
    ------
    FileNotFoundError
        When the dataset file is absent.  Callers should skip rather than
        substitute anything.
    RuntimeError
        When ingestion, preprocessing or graph construction reports errors.
        The framework never proceeds on a partially valid pipeline result.
    """
    from src.data.ingestion import load_transactions
    from src.data.preprocessing import preprocess
    from src.graph.builder import build_graph

    dataset_path = Path(path) if path is not None else DEFAULT_REFERENCE_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {dataset_path}")

    ingestion = load_transactions(str(dataset_path))
    if not ingestion.is_valid:
        raise RuntimeError(f"Ingestion failed: {ingestion.validation_errors}")

    preprocessed = preprocess(ingestion)
    if not preprocessed.is_valid:
        raise RuntimeError(f"Preprocessing failed: {preprocessed.validation_errors}")

    graph_result = build_graph(preprocessed)
    if not graph_result.is_valid:
        raise RuntimeError(f"Graph construction failed: {graph_result.validation_errors}")

    spec = DatasetSpec(
        name=name,
        version=_content_version(dataset_path),
        provenance=f"file:{dataset_path}",
        label_definition=(
            "positive = graph node attribute 'core' is True, set by Layer 3 from "
            "the payer_core / beneficiary_core flags carried by the source "
            "records; it marks entities named as core to the published "
            "investigation, not a supervised fraud adjudication"
        ),
        synthetic=False,
        notes=(
            "Version is derived from file size and mtime, not a content hash.",
            "The positive class is extremely rare; see the split plan's notes "
            "for which folds can support label-dependent metrics.",
        ),
    )
    return dataset_from_graph_result(graph_result, spec, label_attr="core")
