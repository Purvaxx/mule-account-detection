"""
Layer 6 - Legacy GraphSAGE Candidate
======================================

Exposes the Experiment 3 architecture through the new experiment contract, so
the historical model can be benchmarked against new candidates on identical
splits, metrics and thresholds.

What this module is
-------------------
An adapter.  It imports ``src.models.baseline_gnn.BaselineGraphSAGE`` and the
training recipe from ``docs/architecture.md`` 27.5 unchanged, and drives them
through :class:`~src.experiments.candidate.Candidate`.  It does not modify the
legacy modules, does not touch the frozen checkpoint at
``artifacts/models/experiment3_graphsage.pt``, and does not re-run or restate
any historical result.

What this module is not
-----------------------
It is *not* the frozen checkpoint.  Training here produces a new model from the
same architecture; the frozen checkpoint remains the inference contract that
``src/detection/inference.py`` enforces and that
``tests/test_reference_regression.py`` pins.  A number produced by this
candidate is a benchmark of the architecture, never a restatement of the
Experiment 3 result.

Feature independence
--------------------
The candidate takes whatever feature matrix the experiment configured.  Running
it on ``legacy_exp3_v1`` reproduces the historical 167-column input; running it
on ``aml_core_v1`` asks a different and equally legitimate question.  The 167
columns are not required by anything here.

Transductivity
--------------
GraphSAGE aggregates over neighbours, so the forward pass touches held-out
entities' features whenever they are graph neighbours of training entities.
Under a ``global`` feature scope that is the historical transductive regime;
under ``train_window`` the graph itself is truncated.  Either way the runner's
leakage block states which applied.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.experiments.candidate import Candidate, FitContext, register_candidate

logger = logging.getLogger(__name__)

__all__ = ["LegacyGraphSAGECandidate"]


@register_candidate("legacy_graphsage")
class LegacyGraphSAGECandidate(Candidate):
    """Two-layer GraphSAGE node classifier, trained per fold.

    Parameters mirror the legacy defaults (``hidden_dim=64``, ``dropout=0.5``,
    Adam ``lr=0.01``/``weight_decay=5e-4``, 200 epochs, patience 50) so that a
    run under this contract is comparable with the historical protocol.

    Early stopping monitors validation average precision.  When the validation
    partition has no positives - which the framework permits and the legacy
    trainer did not - AP is undefined, so the candidate falls back to
    monitoring training loss and records that substitution in the fit report
    rather than silently scoring the fold as 0.0.
    """

    emits_probabilities = True
    requires_features = True

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        max_epochs: int = 200,
        patience: int = 50,
        **params: Any,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            **params,
        )
        self._model = None
        self._x = None
        self._edge_index = None

    # -- tensor construction ------------------------------------------------

    def _build_tensors(self, ctx: FitContext) -> None:
        import torch

        if ctx.features is None:
            raise ValueError("legacy_graphsage requires a feature matrix.")
        values = ctx.features.to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values), values, 0.0)
        self._x = torch.tensor(values, dtype=torch.float32)

        edges = ctx.train_edges
        entity_index = {e: i for i, e in enumerate(ctx.dataset.entity_ids)}
        if edges is None or len(edges) == 0:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            return
        src = [entity_index[e] for e in edges["source_entity"] if e in entity_index]
        dst = [entity_index[e] for e in edges["target_entity"] if e in entity_index]
        self._edge_index = torch.tensor([src, dst], dtype=torch.long)

    # -- Candidate interface ------------------------------------------------

    def fit(self, ctx: FitContext) -> "LegacyGraphSAGECandidate":
        import torch
        import torch.nn as nn
        from sklearn.metrics import average_precision_score

        from src.models.baseline_gnn import BaselineGraphSAGE

        torch.manual_seed(ctx.seed)
        np.random.seed(ctx.seed % (2**32))

        self._build_tensors(ctx)
        model = BaselineGraphSAGE(
            input_dim=int(self._x.shape[1]),
            hidden_dim=int(self.params["hidden_dim"]),
            dropout=float(self.params["dropout"]),
        )

        y_all = torch.tensor(ctx.dataset.labels.astype(np.float32))
        train_idx = torch.tensor(ctx.train_idx, dtype=torch.long)
        val_idx = torch.tensor(ctx.val_idx, dtype=torch.long)

        pos_weight = ctx.pos_weight() if ctx.imbalance != "none" else None
        if pos_weight is None:
            criterion = nn.BCEWithLogitsLoss()
            imbalance_note = (
                "no positive weighting: training partition has no positives"
                if ctx.pos_weight() is None
                else "imbalance strategy 'none'"
            )
        else:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            )
            imbalance_note = f"pos_weight={pos_weight:.2f}"

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        y_val_np = ctx.y_val
        monitor = "val_average_precision" if (len(y_val_np) and y_val_np.sum() > 0) else "train_loss"

        best_score = -np.inf
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        max_epochs = int(self.params["max_epochs"])
        patience = int(self.params["patience"])

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(self._x, self._edge_index)
            loss = criterion(logits[train_idx], y_all[train_idx])
            loss.backward()
            optimizer.step()

            if monitor == "val_average_precision":
                model.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(model(self._x, self._edge_index))
                score = float(
                    average_precision_score(y_val_np, probs[val_idx].cpu().numpy())
                )
            else:
                score = -float(loss.item())

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._fitted = True
        self.fit_report = {
            "status": "fitted",
            "architecture": "BaselineGraphSAGE (2-layer, mean aggregator)",
            "input_dim": int(self._x.shape[1]),
            "n_edges_visible": int(self._edge_index.shape[1]),
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "early_stopping_monitor": monitor,
            "best_monitor_value": None if not np.isfinite(best_score) else float(best_score),
            "imbalance": imbalance_note,
            "n_train": int(len(ctx.train_idx)),
            "n_train_positive": int(ctx.y_train.sum()),
        }
        if monitor == "train_loss":
            self.fit_report["monitor_note"] = (
                "validation partition has no positives, so validation average "
                "precision is undefined; early stopping monitored training loss "
                "instead"
            )
        return self

    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        import torch

        if len(index) == 0:
            return np.array([], dtype=np.float64)
        if self._model is None:
            raise RuntimeError("legacy_graphsage.predict_scores called before fit.")
        with torch.no_grad():
            probs = torch.sigmoid(self._model(self._x, self._edge_index))
        return probs[torch.tensor(index, dtype=torch.long)].cpu().numpy().astype(np.float64)

    def save(self, path: Path) -> Optional[Path]:
        import torch

        if self._model is None:
            return None
        target = Path(f"{path}.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "config": self.config(),
                "fit_report": self.fit_report,
                "note": (
                    "Benchmark checkpoint produced by the experiment framework. "
                    "This is NOT the frozen Experiment 3 inference checkpoint."
                ),
            },
            target,
        )
        return target
