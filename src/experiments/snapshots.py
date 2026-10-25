"""
Layer 6 - Window Feature Snapshots
===================================

One feature matrix computed from the transactions inside one time window, with
the caching that makes a batch of experiments over the same windows affordable.

Both calendar-time regimes need exactly this and must not implement it twice:

* ``src/experiments/calendar.py`` (forecasting): the window is
  ``[origin - lookback, origin)`` and the label comes from after the origin.
* ``src/experiments/detection_transfer.py`` (retrospective detection): the
  window *is* the detection window, and the label comes from inside it.

The semantics are identical - "summarise exactly the rows dated in this
half-open interval, measuring recency from a stated instant" - so the mask, the
provenance and the cache live here.

Caching
-------
Each family's block depends only on the rows in the window and on ``as_of``, so
a block computed once is the block any later spec would compute. The cache is
keyed by ``(window, as_of, family)`` and is dropped whenever a different dataset
object arrives, so it can never serve a block computed from other rows. The
owner is held by weak reference rather than ``id()``: an id can be reused by a
new object once the old one is collected, and would then match a stranger. A
dataset is treated as immutable once built; mutating its edge table in place
between runs would defeat this, as it would every other reuse in the layer.
"""

from __future__ import annotations

import weakref
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.dataset import ExperimentDataset
from src.experiments.features import FeatureRequest, build_feature_matrix, get_feature_set

__all__ = [
    "edge_dates",
    "window_mask",
    "window_snapshot",
    "clear_snapshot_cache",
    "snapshot_cache_size",
]

_SNAPSHOT_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}
_CACHE_OWNER: Optional[weakref.ref] = None


def clear_snapshot_cache() -> None:
    """Drop every cached snapshot block."""
    global _CACHE_OWNER
    _SNAPSHOT_CACHE.clear()
    _CACHE_OWNER = None


def snapshot_cache_size() -> int:
    """Number of cached blocks - for tests and diagnostics."""
    return len(_SNAPSHOT_CACHE)


def edge_dates(dataset: ExperimentDataset) -> np.ndarray:
    """The edge table's dates as ``datetime64``, NaT where unparseable."""
    return pd.to_datetime(dataset.edge_table["date"], errors="coerce").to_numpy()


def window_mask(
    dates: np.ndarray,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> np.ndarray:
    """Rows dated in ``[start, end)``.

    Undated rows are never included: an undated transaction cannot be shown to
    fall inside the window, and assuming it does would admit evidence the
    window is not entitled to.
    """
    mask = ~np.isnat(dates)
    if start is not None:
        mask &= dates >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        mask &= dates < np.datetime64(pd.Timestamp(end))
    return mask


def window_snapshot(
    dataset: ExperimentDataset,
    sets: Sequence[str],
    *,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    as_of: Optional[pd.Timestamp],
    dates: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], pd.DataFrame]:
    """Feature matrix over every entity from the rows in ``[start, end)``.

    Returns ``(frame, provenance, reused_sets, window_edges)``. ``frame`` has one
    row per entity of ``dataset`` in its order, whether or not the entity is
    active in the window.
    """
    global _CACHE_OWNER
    if _CACHE_OWNER is None or _CACHE_OWNER() is not dataset:
        _SNAPSHOT_CACHE.clear()
        _CACHE_OWNER = weakref.ref(dataset)

    if dates is None:
        dates = edge_dates(dataset)
    key_window = (
        None if start is None else pd.Timestamp(start),
        None if end is None else pd.Timestamp(end),
        None if as_of is None else pd.Timestamp(as_of),
    )
    edges = dataset.edge_table.loc[window_mask(dates, start, end)]
    names = tuple(sets)
    missing = [s for s in names if (key_window, s) not in _SNAPSHOT_CACHE]
    if missing:
        request = FeatureRequest(
            entity_ids=list(dataset.entity_ids),
            edge_table=edges,
            graph=None,
            as_of=None if as_of is None else pd.Timestamp(as_of),
            attributes=dataset.attributes,
        )
        matrix = build_feature_matrix(request, missing)
        for name in missing:
            prefix = f"{name}::"
            _SNAPSHOT_CACHE[(key_window, name)] = matrix.values[
                [c for c in matrix.values.columns if str(c).startswith(prefix)]
            ]
    blocks = [_SNAPSHOT_CACHE[(key_window, s)] for s in names]
    frame = pd.concat(blocks, axis=1)
    provenance = [
        get_feature_set(s).to_dict(columns=list(b.columns)) for s, b in zip(names, blocks)
    ]
    reused = [s for s in names if s not in missing]
    return frame, provenance, reused, edges
