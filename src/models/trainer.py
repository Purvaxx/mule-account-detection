"""
Layer 5 — Training, Evaluation, and 4-Fold Cross-Validation
==============================================================

Implements:
    - Training loop with early stopping (monitor validation PR-AUC).
    - Evaluation function producing all required metrics.
    - 4-fold cross-validation orchestrator.

Training protocol (docs/architecture.md §27.5)
-----------------------------------------------
    Optimizer  : Adam (lr=0.01, weight_decay=5e-4)
    Loss       : BCEWithLogitsLoss(pos_weight = neg_train / pos_train)
    Max epochs : 200
    Early stop : patience=50, monitor=val PR-AUC (higher is better)
    Checkpoint : best-val checkpoint restored before test evaluation

Threshold selection
-------------------
    The classification threshold is chosen on the VALIDATION set by
    maximising F1.  That threshold is frozen and applied unchanged to
    the TEST set.  The test set is NEVER used to select a threshold.

Evaluation metrics (§27.5)
---------------------------
    Primary  : PR-AUC
    Also     : ROC-AUC, Precision, Recall, F1, Confusion Matrix

Statistical caveat
------------------
    With only 1 positive per val/test fold, metrics have extreme variance.
    Mean ± std across 4 folds are reported but should not be over-interpreted.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from src.models.baseline_gnn import BaselineGraphSAGE
from src.models.dataset import CVFold, PyGData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    """All evaluation metrics for a single fold."""

    fold_idx: int
    threshold: float          # chosen on val set, applied to test
    # Test-set metrics
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion: list           # [[TN, FP], [FN, TP]]
    # Validation PR-AUC (best checkpoint)
    best_val_pr_auc: float
    best_epoch: int
    # Counts
    n_test_pos: int
    n_test_neg: int


@dataclass
class CVResults:
    """Aggregated 4-fold cross-validation results."""

    folds: list[FoldMetrics]

    def _vals(self, attr: str) -> np.ndarray:
        return np.array([getattr(f, attr) for f in self.folds])

    @property
    def mean_pr_auc(self) -> float:
        return float(self._vals("pr_auc").mean())

    @property
    def std_pr_auc(self) -> float:
        return float(self._vals("pr_auc").std())

    @property
    def mean_roc_auc(self) -> float:
        return float(self._vals("roc_auc").mean())

    @property
    def std_roc_auc(self) -> float:
        return float(self._vals("roc_auc").std())

    @property
    def mean_f1(self) -> float:
        return float(self._vals("f1").mean())

    @property
    def std_f1(self) -> float:
        return float(self._vals("f1").std())

    @property
    def mean_precision(self) -> float:
        return float(self._vals("precision").mean())

    @property
    def std_precision(self) -> float:
        return float(self._vals("precision").std())

    @property
    def mean_recall(self) -> float:
        return float(self._vals("recall").mean())

    @property
    def std_recall(self) -> float:
        return float(self._vals("recall").std())

    def summary(self) -> str:
        """Human-readable fold-by-fold + mean ± std table.

        ``CVResults`` is experiment-agnostic — it holds whatever folds it was
        given and has no way to know which experiment produced them.  The title
        is therefore neutral.  It previously read "Experiment 1 — Baseline GNN
        (GraphSAGE)", which mislabelled the captured output of every other
        experiment that calls this method.
        """
        lines = [
            "=" * 72,
            "GraphSAGE   4-Fold CV Results",
            "=" * 72,
            f"{'Fold':<6}{'PR-AUC':>10}{'ROC-AUC':>10}{'F1':>8}"
            f"{'Prec':>8}{'Recall':>8}{'Val-PR':>10}{'Ep':>5}",
            "-" * 72,
        ]
        for fm in self.folds:
            lines.append(
                f"{fm.fold_idx:<6}"
                f"{fm.pr_auc:>10.4f}"
                f"{fm.roc_auc:>10.4f}"
                f"{fm.f1:>8.4f}"
                f"{fm.precision:>8.4f}"
                f"{fm.recall:>8.4f}"
                f"{fm.best_val_pr_auc:>10.4f}"
                f"{fm.best_epoch:>5}"
            )
        lines.append("-" * 72)
        lines.append(
            f"{'Mean':<6}"
            f"{self.mean_pr_auc:>10.4f}"
            f"{self.mean_roc_auc:>10.4f}"
            f"{self.mean_f1:>8.4f}"
            f"{self.mean_precision:>8.4f}"
            f"{self.mean_recall:>8.4f}"
        )
        lines.append(
            f"{'±Std':<6}"
            f"{self.std_pr_auc:>10.4f}"
            f"{self.std_roc_auc:>10.4f}"
            f"{self.std_f1:>8.4f}"
            f"{self.std_precision:>8.4f}"
            f"{self.std_recall:>8.4f}"
        )
        lines.append("=" * 72)
        lines.append(
            "[WARNING] With only 1 positive per test fold, metrics have "
            "high variance and should not be over-interpreted."
        )
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_probs(
    model: BaselineGraphSAGE,
    data: Data,
    mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) numpy arrays for nodes in mask."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = data.y.cpu().numpy()
    return y_true[mask.numpy()], probs[mask.numpy()]


def _best_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
) -> float:
    """Find the threshold that maximises F1 on the given set.

    Searches ``n_thresholds`` evenly-spaced values in [0, 1].
    In case of a tie the lower threshold is returned (more recall-friendly).
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    best_f1, best_thr = -1.0, 0.5
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

def evaluate(
    model: BaselineGraphSAGE,
    data: Data,
    mask: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Evaluate ``model`` on nodes selected by ``mask``.

    Parameters
    ----------
    model:
        Trained ``BaselineGraphSAGE``.
    data:
        Full ``PyG Data`` object (graph topology + all node features/labels).
    mask:
        Boolean tensor selecting which nodes to evaluate.
    threshold:
        Pre-selected classification threshold (chosen on val, not test).

    Returns
    -------
    dict with keys: pr_auc, roc_auc, precision, recall, f1, confusion.
    """
    y_true, y_prob = _get_probs(model, data, mask)
    preds = (y_prob >= threshold).astype(int)

    # Guard against single-class subsets (possible with 1 positive)
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "Only one class present in evaluation subset — "
            "AUC metrics are undefined; returning 0.0."
        )
        pr_auc  = 0.0
        roc_auc = 0.0
    else:
        pr_auc  = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "pr_auc":    pr_auc,
        "roc_auc":   roc_auc,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "confusion": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }


def train_fold(
    pyg: PyGData,
    fold: CVFold,
    *,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 50,
    seed: int = 42,
) -> FoldMetrics:
    """Train one fold and return test-set metrics.

    Parameters
    ----------
    pyg:
        Pre-built ``PyGData`` (graph + features + labels).
    fold:
        ``CVFold`` containing masks and pre-computed ``pos_weight``.
    hidden_dim, dropout, lr, weight_decay, max_epochs, patience:
        Training hyperparameters (spec §27).
    seed:
        Random seed for model weight initialisation.

    Returns
    -------
    FoldMetrics
    """
    torch.manual_seed(seed + fold.fold_idx)
    np.random.seed(seed + fold.fold_idx)

    data = pyg.data
    model = BaselineGraphSAGE(
        input_dim=pyg.feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=fold.pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, max_epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[fold.train_mask], data.y[fold.train_mask])
        loss.backward()
        optimizer.step()

        # Validation: PR-AUC
        y_val_true, y_val_prob = _get_probs(model, data, fold.val_mask)

        if len(np.unique(y_val_true)) >= 2:
            val_pr_auc = float(average_precision_score(y_val_true, y_val_prob))
        else:
            # Only 1 class in val (possible in tiny synthetic tests)
            val_pr_auc = 0.0

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                "Fold %d: early stopping at epoch %d (best val PR-AUC=%.4f "
                "at epoch %d).",
                fold.fold_idx, epoch, best_val_pr_auc, best_epoch,
            )
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    logger.info(
        "Fold %d: training complete. Best epoch=%d, val PR-AUC=%.4f.",
        fold.fold_idx, best_epoch, best_val_pr_auc,
    )

    # Select threshold on VALIDATION set (never on test)
    y_val_true, y_val_prob = _get_probs(model, data, fold.val_mask)
    threshold = _best_threshold_by_f1(y_val_true, y_val_prob)
    logger.info(
        "Fold %d: threshold selected on validation set = %.4f.",
        fold.fold_idx, threshold,
    )

    # Evaluate on TEST set with frozen threshold
    test_metrics = evaluate(model, data, fold.test_mask, threshold)

    n_test_pos = int(data.y[fold.test_mask].sum().item())
    n_test_neg = int(fold.test_mask.sum().item()) - n_test_pos

    return FoldMetrics(
        fold_idx=fold.fold_idx,
        threshold=threshold,
        pr_auc=test_metrics["pr_auc"],
        roc_auc=test_metrics["roc_auc"],
        precision=test_metrics["precision"],
        recall=test_metrics["recall"],
        f1=test_metrics["f1"],
        confusion=test_metrics["confusion"],
        best_val_pr_auc=best_val_pr_auc,
        best_epoch=best_epoch,
        n_test_pos=n_test_pos,
        n_test_neg=n_test_neg,
    )


def run_cross_validation(
    pyg: PyGData,
    folds: list[CVFold],
    *,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 50,
    seed: int = 42,
) -> CVResults:
    """Run the full 4-fold cross-validation and return aggregated results.

    Parameters
    ----------
    pyg:
        ``PyGData`` object (shared across all folds).
    folds:
        List of 4 ``CVFold`` objects from :func:`~src.models.dataset.build_cv_folds`.
    hidden_dim, dropout, lr, weight_decay, max_epochs, patience, seed:
        Training hyperparameters.

    Returns
    -------
    CVResults
    """
    assert len(folds) == 4, f"Expected 4 folds, got {len(folds)}."
    fold_results: list[FoldMetrics] = []

    for fold in folds:
        logger.info("=" * 60)
        logger.info("Starting fold %d / 4", fold.fold_idx + 1)
        logger.info("=" * 60)
        fm = train_fold(
            pyg, fold,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )
        fold_results.append(fm)
        logger.info(
            "Fold %d results: PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  "
            "Prec=%.4f  Recall=%.4f",
            fm.fold_idx, fm.pr_auc, fm.roc_auc, fm.f1,
            fm.precision, fm.recall,
        )

    return CVResults(folds=fold_results)
