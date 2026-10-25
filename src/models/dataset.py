"""
Layer 5 — PyG Dataset Construction
====================================

Converts the authoritative Layer 3 NetworkX MultiDiGraph and the Layer 4
baseline feature matrix into a single PyTorch Geometric ``Data`` object,
and constructs the stratified 4-fold cross-validation masks.

Design decisions
----------------
Node ordering:
    Nodes are sorted deterministically (``sorted(graph.nodes())``) so that
    row *i* of the feature matrix ``x`` always corresponds to the same entity
    across every run.  The integer ↔ entity-ID mapping is stored in
    ``node_to_idx`` / ``idx_to_node``.

edge_index:
    Every directed transaction edge is preserved as a separate column in
    ``edge_index`` — parallel edges between the same (source, target) pair
    appear as multiple columns, NOT collapsed.  ``edge_index.shape == (2, E)``
    where E is the total transaction count (16,940 for the real dataset).

Label tensor y:
    Binary: 1 for core entities, 0 for non-core.  The ``core`` node attribute
    from Layer 3 is used.  The label is NEVER used as a node feature.

4-fold CV mask construction:
    With exactly 4 positive entities the folds are constructed so that each
    fold assigns:
        train  : 2 positives  +  stratified sample of negatives
        val    : 1 positive   +  stratified sample of negatives
        test   : 1 positive   +  remaining negatives
    The graph topology (edge_index, x) is IDENTICAL in every fold.
    Only the boolean train/val/test masks differ.

Leakage controls:
    - ``core`` / ``payer_core`` / ``beneficiary_core`` are never added to x.
    - ``source_file``, ``investigation``, raw names and account IDs are excluded.
    - val/test labels are never used to compute class weights (pos_weight).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.features.baseline import BaselineFeatures
from src.features.temporal import TemporalFeatures
from src.features.multigraph import MultigraphFeatures
from src.features.tda import TGAFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PyGData:
    """PyTorch Geometric representation of the transaction graph.

    Attributes
    ----------
    data:
        The ``torch_geometric.data.Data`` object containing ``x``, ``y``,
        and ``edge_index``.  No masks are attached here; masks live in each
        :class:`CVFold`.
    node_to_idx:
        Deterministic mapping ``entity_id → integer node index``.
    idx_to_node:
        Inverse mapping ``integer index → entity_id``.
    num_nodes:
        Total number of nodes.
    num_edges:
        Total number of directed transaction edges.
    num_positive:
        Number of positive (core) entities.
    num_negative:
        Number of negative (non-core) entities.
    feature_dim:
        Number of baseline features (columns of x).
    """

    data: Data
    node_to_idx: dict[str, int]
    idx_to_node: dict[int, str]
    num_nodes: int
    num_edges: int
    num_positive: int
    num_negative: int
    feature_dim: int


@dataclass
class CVFold:
    """One fold of the 4-fold cross-validation.

    Attributes
    ----------
    fold_idx:
        Zero-based fold index (0–3).
    train_mask:
        Boolean tensor of shape ``(num_nodes,)``.
    val_mask:
        Boolean tensor of shape ``(num_nodes,)``.
    test_mask:
        Boolean tensor of shape ``(num_nodes,)``.
    pos_weight:
        ``neg_train / pos_train`` computed from the training mask only.
        A scalar ``torch.Tensor`` of shape ``(1,)`` suitable for
        ``BCEWithLogitsLoss(pos_weight=...)``.
    n_train_pos:
        Number of positive nodes in training.
    n_val_pos:
        Number of positive nodes in validation.
    n_test_pos:
        Number of positive nodes in test.
    """

    fold_idx: int
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    pos_weight: torch.Tensor
    n_train_pos: int
    n_val_pos: int
    n_test_pos: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pyg_data(
    graph: nx.MultiDiGraph,
    baseline: BaselineFeatures,
    temporal: Optional[TemporalFeatures] = None,
    multigraph: Optional[MultigraphFeatures] = None,
    tga: Optional[TGAFeatures] = None,
) -> PyGData:
    """Build a PyG ``Data`` object from Layer 3 + Layer 4 outputs.

    Parameters
    ----------
    graph:
        Authoritative ``networkx.MultiDiGraph`` from Layer 3.
        **Not modified.**
    baseline:
        Baseline features from Layer 4 (``compute_baseline``).
        ``baseline.features`` must be indexed by entity_id.
        **Not modified.**
    temporal:
        Optional temporal features from Layer 4 (``compute_temporal``).
        If provided, concatenated to baseline features.
    multigraph:
        Optional multigraph features from Layer 4 (``compute_multigraph``).
        If provided, concatenated to baseline features (with log1p preprocessing).
    tga:
        Optional Temporal Graph Topology features (``compute_tga``).
        If provided, concatenated to baseline features (with log1p preprocessing
        for counts; sentinel -1.0 preserved for earliest_cycle_completion_days).

    Returns
    -------
    PyGData
    """
    # ------------------------------------------------------------------
    # 1. Deterministic node ordering
    # ------------------------------------------------------------------
    nodes: list[str] = sorted(graph.nodes())
    n = len(nodes)
    node_to_idx: dict[str, int] = {e: i for i, e in enumerate(nodes)}
    idx_to_node: dict[int, str] = {i: e for i, e in enumerate(nodes)}

    logger.info("Building PyG data object: %d nodes.", n)

    # ------------------------------------------------------------------
    # 2. Feature matrix x  (3440 × 160 for the real dataset)
    # ------------------------------------------------------------------
    # Reindex the baseline DataFrame to match the sorted node order so that
    # row i corresponds to nodes[i].  fill_value=0.0 for any missing rows.
    feat_df = baseline.features.reindex(nodes).fillna(0.0)

    feature_dim = len(baseline.feature_names)

    if temporal is not None:
        temp_df = temporal.features.reindex(nodes).copy()
        
        # Handle NaN values explicitly as per Section 28.1 Experiment 2 Preprocessing Policy
        # Explicitly distinguish undefined temporal statistics by filling with -1.0
        temp_df = temp_df.fillna(-1.0)
        
        # Apply deterministic log1p transformation to counts and times for defined values (x >= 0)
        for col in ["activity_7d", "activity_30d", "mean_iat"]:
            mask = temp_df[col] >= 0
            temp_df.loc[mask, col] = np.log1p(temp_df.loc[mask, col])
            
        # burstiness, active_day_ratio, same_day_cooccurrence are left unscaled
        
        feat_df = pd.concat([feat_df, temp_df], axis=1)
        feature_dim += len(temporal.feature_names)

    if multigraph is not None:
        mg_df = multigraph.features.reindex(nodes).copy()
        
        # Handle NaN explicitly as per Section 28.2 Experiment 3 Preprocessing Policy
        mg_df = mg_df.fillna(-1.0)
        
        # Apply deterministic log1p transformation to all multigraph features for defined values (x >= 0)
        for col in multigraph.feature_names:
            mask = mg_df[col] >= 0
            mg_df.loc[mask, col] = np.log1p(mg_df.loc[mask, col])
            
        feat_df = pd.concat([feat_df, mg_df], axis=1)
        feature_dim += len(multigraph.feature_names)

    if tga is not None:
        tga_df = tga.features.reindex(nodes).copy()

        # Handle NaN explicitly — fill with -1.0 sentinel
        tga_df = tga_df.fillna(-1.0)

        # Ensure float dtype for log1p assignment (TGA may return int columns)
        for col in tga_df.columns:
            tga_df[col] = tga_df[col].astype(np.float64)

        # Apply deterministic log1p to count features for defined values (x >= 0)
        for col in ["temporal_reciprocal_count", "temporal_cycle_count"]:
            mask = tga_df[col] >= 0
            tga_df.loc[mask, col] = np.log1p(tga_df.loc[mask, col])

        # earliest_cycle_completion_days: log1p for x >= 0, preserve -1.0 sentinel
        col = "earliest_cycle_completion_days"
        mask = tga_df[col] >= 0
        tga_df.loc[mask, col] = np.log1p(tga_df.loc[mask, col])
        # -1.0 sentinel remains untouched (distinct from log1p(0) = 0.0)

        feat_df = pd.concat([feat_df, tga_df], axis=1)
        feature_dim += len(tga.feature_names)

    # Safety check: ensure core / raw label columns are not present in x.
    forbidden = {"core", "payer_core", "beneficiary_core",
                 "source_file", "investigation"}
    present_forbidden = forbidden & set(feat_df.columns)
    if present_forbidden:
        raise ValueError(
            f"Target/metadata columns found in feature matrix: {present_forbidden}. "
            "These must never enter x."
        )

    x = torch.tensor(feat_df.values, dtype=torch.float32)
    assert x.shape == (n, feature_dim), (
        f"Feature matrix shape mismatch: got {x.shape}, "
        f"expected ({n}, {feature_dim})"
    )

    # ------------------------------------------------------------------
    # 3. Label tensor y  (binary: 1 = core, 0 = non-core)
    # ------------------------------------------------------------------
    labels = []
    for node in nodes:
        attr = graph.nodes[node]
        core_val = attr.get("core", False)
        labels.append(1 if core_val is True or core_val is np.bool_(True) else 0)

    y = torch.tensor(labels, dtype=torch.float32)
    num_positive = int(y.sum().item())
    num_negative = n - num_positive

    logger.info(
        "Labels: %d positive / %d negative (%.3f%% positive).",
        num_positive, num_negative,
        100.0 * num_positive / n if n > 0 else 0.0,
    )

    # ------------------------------------------------------------------
    # 4. edge_index  — preserves ALL parallel directed edges
    # ------------------------------------------------------------------
    # Iterate over every edge in the MultiDiGraph (each transaction is a
    # separate entry).  For a MultiDiGraph, G.edges() yields (u, v, key)
    # tuples; we only need (u, v).
    src_list: list[int] = []
    tgt_list: list[int] = []

    for u, v, _ in graph.edges(keys=True):
        if u in node_to_idx and v in node_to_idx:
            src_list.append(node_to_idx[u])
            tgt_list.append(node_to_idx[v])
        else:
            logger.warning(
                "Edge (%s, %s) references a node not in node_to_idx. Skipped.", u, v
            )

    edge_index = torch.tensor(
        [src_list, tgt_list], dtype=torch.long
    )  # shape (2, E)
    num_edges = edge_index.shape[1]

    logger.info(
        "edge_index shape: %s  (parallel edges preserved).", tuple(edge_index.shape)
    )

    # ------------------------------------------------------------------
    # 5. Assemble PyG Data (no masks — masks are per-fold)
    # ------------------------------------------------------------------
    data = Data(x=x, y=y, edge_index=edge_index)
    data.num_nodes = n  # explicit for PyG bookkeeping

    return PyGData(
        data=data,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        num_nodes=n,
        num_edges=num_edges,
        num_positive=num_positive,
        num_negative=num_negative,
        feature_dim=feature_dim,
    )


def build_cv_folds(
    pyg: PyGData,
    seed: int = 42,
) -> list[CVFold]:
    """Construct the 4-fold stratified cross-validation masks.

    With exactly 4 positive entities the fold assignment is deterministic:
    all 4! = 24 permutations are possible; we choose the first 4 in a fixed
    order seeded by ``seed``.

    Each fold:
        train  : 2 positives  + stratified negatives (≈50 %)
        val    : 1 positive   + stratified negatives (≈25 %)
        test   : 1 positive   + remaining negatives  (≈25 %)

    The train/val/test masks are non-overlapping boolean tensors of shape
    ``(num_nodes,)``.  The graph topology is identical across all folds.

    Parameters
    ----------
    pyg:
        Populated :class:`PyGData` returned by :func:`build_pyg_data`.
    seed:
        Random seed for reproducible negative-node assignment.

    Returns
    -------
    list of 4 :class:`CVFold` objects.
    """
    rng = np.random.default_rng(seed)
    y_np = pyg.data.y.numpy()

    pos_indices = np.where(y_np == 1)[0]
    neg_indices = np.where(y_np == 0)[0]

    assert len(pos_indices) == 4, (
        f"Expected exactly 4 positive entities; found {len(pos_indices)}."
    )

    # Shuffle negatives for reproducible random split
    neg_shuffled = rng.permutation(neg_indices)
    n_neg = len(neg_shuffled)

    # Split negatives into 4 roughly equal groups for val/test (25% each)
    # and the remaining 50% for train in each fold.
    # We assign each negative to one of 4 "held-out" groups; in each fold
    # that group is held out as test, an adjacent group as val, the rest as train.
    neg_groups = np.array_split(neg_shuffled, 4)

    # Positive permutation for folds: each positive appears exactly once in
    # each of the 4 roles (test, val, train[0], train[1]).
    # We use a fixed assignment: positive i goes to fold i as the test positive.
    pos_perm = rng.permutation(pos_indices)  # reproducible shuffle of positives

    folds: list[CVFold] = []

    for fold_idx in range(4):
        # Positive assignments for this fold
        test_pos  = pos_perm[fold_idx]
        val_pos   = pos_perm[(fold_idx + 1) % 4]
        train_pos = [pos_perm[(fold_idx + 2) % 4], pos_perm[(fold_idx + 3) % 4]]

        # Negative assignments for this fold
        test_neg  = neg_groups[fold_idx]
        val_neg   = neg_groups[(fold_idx + 1) % 4]
        train_neg = np.concatenate([
            neg_groups[(fold_idx + 2) % 4],
            neg_groups[(fold_idx + 3) % 4],
        ])

        # Build masks
        n_nodes = pyg.num_nodes
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_pos] = True
        train_mask[train_neg] = True

        val_mask[val_pos] = True
        val_mask[val_neg] = True

        test_mask[test_pos] = True
        test_mask[test_neg] = True

        # Sanity: no overlap
        assert not (train_mask & val_mask).any(), "Train/Val overlap in fold!"
        assert not (train_mask & test_mask).any(), "Train/Test overlap in fold!"
        assert not (val_mask & test_mask).any(), "Val/Test overlap in fold!"
        assert (train_mask | val_mask | test_mask).all(), "Some nodes unassigned!"

        # pos_weight from training mask ONLY
        y_train = pyg.data.y[train_mask]
        n_train_pos = int(y_train.sum().item())
        n_train_neg = int((y_train == 0).sum().item())
        if n_train_pos == 0:
            raise ValueError(f"Fold {fold_idx}: no positive training nodes.")
        pw = torch.tensor([n_train_neg / n_train_pos], dtype=torch.float32)

        n_val_pos  = int(pyg.data.y[val_mask].sum().item())
        n_test_pos = int(pyg.data.y[test_mask].sum().item())

        logger.info(
            "Fold %d: train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d), "
            "pos_weight=%.2f",
            fold_idx,
            train_mask.sum().item(), n_train_pos,
            val_mask.sum().item(), n_val_pos,
            test_mask.sum().item(), n_test_pos,
            pw.item(),
        )

        folds.append(CVFold(
            fold_idx=fold_idx,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            pos_weight=pw,
            n_train_pos=n_train_pos,
            n_val_pos=n_val_pos,
            n_test_pos=n_test_pos,
        ))

    return folds
