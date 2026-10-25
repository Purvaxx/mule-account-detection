"""
Layer 5 — Graph Representation Learning (Baseline GNN)
=======================================================

Sub-modules
-----------
dataset        Converts Layer 3 (NetworkX MultiDiGraph) + Layer 4 (baseline
               feature matrix) into a single PyTorch Geometric ``Data`` object
               and constructs the 4-fold cross-validation masks.
baseline_gnn   Two-layer GraphSAGE model for transductive node classification.
trainer        Training loop, early stopping, evaluation, and the 4-fold CV
               orchestrator.

Experiment boundary
-------------------
Experiment 1 (this module) uses ONLY:
  - Baseline node features (3,440 × 160)
  - Full directed multigraph topology (edge_index shape 2 × 16,940)

Excluded from Experiment 1:
  - Temporal upgrade features
  - Directed-multigraph upgrade features
  - TDA features
  - Target labels as model input
  - Edge attributes
"""

from src.models.dataset import build_pyg_data, build_cv_folds
from src.models.baseline_gnn import BaselineGraphSAGE
from src.models.trainer import train_fold, evaluate, run_cross_validation

__all__ = [
    "build_pyg_data",
    "build_cv_folds",
    "BaselineGraphSAGE",
    "train_fold",
    "evaluate",
    "run_cross_validation",
]
