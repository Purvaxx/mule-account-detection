"""
Layer 4 — Feature Engineering
==============================

Sub-packages
------------
baseline   Baseline feature set: entity type, bank country (multi-valued),
           log-degree, log-USD volume, flow imbalance.
temporal   Temporal upgrade: rolling windows, inter-transaction time,
           burstiness, active-day ratio, same-day co-occurrence.
multigraph Directed-multigraph upgrade: repeated-counterparty ratio,
           parallel-edge count, temporal dispersion, reciprocal behaviour.

These sub-modules never modify the authoritative MultiDiGraph or the
preprocessed transaction DataFrame; they produce new DataFrames containing
only the derived features.
"""

from src.features.baseline import BaselineFeatures, compute_baseline
from src.features.temporal import TemporalFeatures, compute_temporal
from src.features.multigraph import MultigraphFeatures, compute_multigraph

__all__ = [
    "BaselineFeatures",
    "compute_baseline",
    "TemporalFeatures",
    "compute_temporal",
    "MultigraphFeatures",
    "compute_multigraph",
]
