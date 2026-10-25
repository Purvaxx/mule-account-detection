"""
Layer 5 — Baseline GraphSAGE Model
=====================================

Implements the 2-layer GraphSAGE architecture specified in
``docs/architecture.md`` §27.3.

Architecture
------------
    Input  :  x  ∈  ℝ^(N × 160)        (baseline features)
    Layer 1:  SAGEConv(160 → 64) + ReLU + Dropout(0.5)
    Layer 2:  SAGEConv(64  → 64) + ReLU + Dropout(0.5)
    Output :  Linear(64 → 1)            (single logit per node)

The output is a raw logit (NOT a probability).  The caller applies
``torch.sigmoid`` for inference or ``BCEWithLogitsLoss`` for training.

Boundary
--------
This model is strictly Experiment 1 (Baseline GNN).
It receives ONLY the baseline feature matrix + directed graph topology.
Temporal / multigraph / TDA features must never be concatenated here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BaselineGraphSAGE(nn.Module):
    """Two-layer GraphSAGE node classifier.

    Parameters
    ----------
    input_dim:
        Number of input features per node (160 for the baseline).
    hidden_dim:
        Size of the hidden representation (default 64).
    dropout:
        Dropout probability applied after each hidden activation (default 0.5).
    """

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Node feature matrix, shape ``(N, input_dim)``.
        edge_index:
            Graph connectivity, shape ``(2, E)``.  Parallel edges are
            separate columns — they are NOT collapsed.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(N,)``.  Apply ``torch.sigmoid`` for
            probabilities.
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Classification head: (N, 64) → (N, 1) → (N,)
        out = self.classifier(h).squeeze(-1)
        return out
