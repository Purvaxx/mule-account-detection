"""
Layer 6 - Distribution Drift
=============================

One definition of the project's drift metric, so that a number reported by the
calendar design audit and a number reported by a cross-period experiment mean
the same thing.

Population Stability Index, as used throughout this project:

* ten quantile bins fitted on the **reference** (training) sample;
* duplicate bin edges collapsed, outer edges opened to infinity;
* undefined values (``NaN``, ``inf``) excluded from both samples;
* bin shares floored at 1e-6 so an empty bin cannot produce an infinity;
* ``PSI = sum (p_ref - p_cmp) * log(p_ref / p_cmp)``.

Conventional reading: below 0.10 stable, 0.10-0.25 moderate, above 0.25 a major
shift. Those bands are convention, not a property of this data.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = ["psi", "feature_psi", "summarise_psi"]

MAJOR_SHIFT = 0.25
MODERATE_SHIFT = 0.10


def psi(reference: Any, comparison: Any, bins: int = 10) -> Optional[float]:
    """PSI of ``comparison`` against ``reference``'s quantile bins.

    ``None`` when either sample has no finite value, or the reference has too
    few distinct values to define a bin edge - undefined, never 0.0.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(comparison, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    edges = np.unique(np.quantile(a, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    p_cmp = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, None)
    return float(np.sum((p_ref - p_cmp) * np.log(p_ref / p_cmp)))


def feature_psi(reference: pd.DataFrame, comparison: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Per-column PSI with each sample's median, for the shared columns.

    Columns whose PSI is undefined are omitted and counted by
    :func:`summarise_psi` as such.
    """
    rows = []
    for column in reference.columns:
        if column not in comparison.columns:
            continue
        a = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(comparison[column], errors="coerce").to_numpy(dtype=np.float64)
        value = psi(a, b, bins=bins)
        if value is None:
            continue
        rows.append(
            {
                "feature": str(column),
                "psi": round(value, 4),
                "reference_median": None if not np.isfinite(a).any() else float(np.nanmedian(a)),
                "comparison_median": None if not np.isfinite(b).any() else float(np.nanmedian(b)),
            }
        )
    return pd.DataFrame(rows, columns=["feature", "psi", "reference_median", "comparison_median"])


def summarise_psi(table: pd.DataFrame, top: int = 10) -> dict[str, Any]:
    """Counts, median, per-family median and the largest shifts."""
    if not len(table):
        return {"n_defined": 0, "n_above_0_25": 0, "n_above_0_10": 0, "median": None,
                "median_by_family": {}, "top": []}
    families = table.assign(family=table["feature"].str.split("::").str[0])
    return {
        "n_defined": int(len(table)),
        "n_above_0_25": int((table["psi"] > MAJOR_SHIFT).sum()),
        "n_above_0_10": int((table["psi"] > MODERATE_SHIFT).sum()),
        "median": float(table["psi"].median()),
        "median_by_family": families.groupby("family")["psi"].median().round(4).to_dict(),
        "top": table.sort_values("psi", ascending=False).head(top).to_dict(orient="records"),
    }
