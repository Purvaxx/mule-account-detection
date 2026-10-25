"""
Final Inference Module
======================

Runs the end-to-end inference pipeline for Experiment 3 (Baseline + Multigraph).
Trains the BaselineGraphSAGE on the full graph (using a standard fold to select
the threshold and early stopping) and returns predicted probabilities for all nodes.
"""

import copy
import logging

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph
from src.models.dataset import build_pyg_data, build_cv_folds, PyGData, CVFold
from src.models.baseline_gnn import BaselineGraphSAGE
from src.models.trainer import _get_probs, _best_threshold_by_f1

logger = logging.getLogger(__name__)


def train_inference_model(
    pyg: PyGData,
    fold: CVFold,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 50,
    seed: int = 42,
) -> tuple[BaselineGraphSAGE, float]:
    """Train the model identically to Experiment 3, returning model and threshold."""
    torch.manual_seed(seed + fold.fold_idx)
    np.random.seed(seed + fold.fold_idx)

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
        model.train()
        optimizer.zero_grad()
        logits = model(pyg.data.x, pyg.data.edge_index)
        loss = criterion(logits[fold.train_mask], pyg.data.y[fold.train_mask])
        loss.backward()
        optimizer.step()

        y_val_true, y_val_prob = _get_probs(model, pyg.data, fold.val_mask)

        if len(np.unique(y_val_true)) >= 2:
            val_pr_auc = float(average_precision_score(y_val_true, y_val_prob))
        else:
            val_pr_auc = 0.0

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    y_val_true, y_val_prob = _get_probs(model, pyg.data, fold.val_mask)
    threshold = _best_threshold_by_f1(y_val_true, y_val_prob)

    return model, threshold


from dataclasses import dataclass
from typing import Any

@dataclass
class DetectionState:
    """Immutable structured state for both inference and UI."""
    graph: nx.MultiDiGraph
    pyg: PyGData
    baseline: Any
    multi: Any

def prepare_detection_data(dataset_path: str) -> DetectionState:
    """Prepares the detection data, returning a shared state object."""
    # 2. Ingestion & Preprocessing
    ing_res = load_transactions(dataset_path)
    if not ing_res.is_valid:
        raise ValueError("Ingestion failed")

    pre_res = preprocess(ing_res)
    if not pre_res.is_valid:
        raise ValueError("Preprocessing failed")

    # 3. Graph Construction
    graph_res = build_graph(pre_res)
    if not graph_res.is_valid:
        raise ValueError("Graph construction failed")

    # 4. Features (Baseline + Multigraph)
    baseline = compute_baseline(graph_res.graph, graph_res.edge_table)
    multi = compute_multigraph(graph_res.graph, graph_res.edge_table)

    # 5. PyG Data
    pyg = build_pyg_data(graph_res.graph, baseline, multigraph=multi)

    return DetectionState(
        graph=graph_res.graph,
        pyg=pyg,
        baseline=baseline,
        multi=multi
    )

def predict_from_prepared_data(state: DetectionState, checkpoint_path: str) -> pd.DataFrame:
    """Runs inference on prepared data using the checkpoint."""
    if checkpoint_path is None:
        raise ValueError(
            "A checkpoint path must be provided for inference. "
            "On-the-fly training is no longer the default."
        )

    # 1. Load Checkpoint and Verify
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    metadata = checkpoint.get("metadata", {})

    if metadata.get("input_dim") != 167:
        raise ValueError(f"Incompatible checkpoint: expected input_dim=167, got {metadata.get('input_dim')}")

    # 6. Verify feature dimension matches checkpoint
    pyg = state.pyg
    baseline = state.baseline
    multi = state.multi

    if pyg.feature_dim != metadata["input_dim"]:
        raise ValueError(f"Feature dimension mismatch: checkpoint expects {metadata['input_dim']}, data has {pyg.feature_dim}")

    # Verify feature ordering
    current_features = list(baseline.features.columns) + list(multi.features.columns)
    if current_features != metadata.get("feature_names", []):
        raise ValueError("Feature ordering mismatch between current data and checkpoint.")

    # 7. Load model
    model = BaselineGraphSAGE(
        input_dim=metadata["input_dim"],
        hidden_dim=metadata["hidden_dim"],
        dropout=metadata["training_config"]["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = metadata["threshold"]

    # 7. Predict for all nodes
    model.eval()
    with torch.no_grad():
        logits = model(pyg.data.x, pyg.data.edge_index)
        all_probs = torch.sigmoid(logits).cpu().numpy()

    # Map back to entity IDs and raw features
    results = []
    for idx, prob in enumerate(all_probs):
        entity_id = pyg.idx_to_node[idx]

        # Raw features before transformations
        b_df = baseline.features
        m_df = multi.features

        # Invert log10 for display values: 10^v - 1
        in_usd = (10 ** b_df.at[entity_id, "log_total_incoming_usd"]) - 1
        out_usd = (10 ** b_df.at[entity_id, "log_total_outgoing_usd"]) - 1
        in_deg = int(round((10 ** b_df.at[entity_id, "log_in_degree"]) - 1))
        out_deg = int(round((10 ** b_df.at[entity_id, "log_out_degree"]) - 1))
        imbalance = b_df.at[entity_id, "flow_imbalance"]

        rep_out = m_df.at[entity_id, "out_repeated_counterparty_ratio"]
        rep_in = m_df.at[entity_id, "in_repeated_counterparty_ratio"]
        max_parallel = int(m_df.at[entity_id, "max_parallel_edge_count"])
        mean_parallel = m_df.at[entity_id, "mean_parallel_edge_count"]
        recip_in = int(m_df.at[entity_id, "reciprocal_in_count"])
        recip_out = int(m_df.at[entity_id, "reciprocal_out_count"])

        pred_class = 1 if prob >= threshold else 0

        results.append({
            "entity_id": entity_id,
            "risk_score": float(prob),
            "predicted_class": pred_class,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_incoming_usd": float(in_usd),
            "total_outgoing_usd": float(out_usd),
            "flow_imbalance": float(imbalance),
            "in_repeated_ratio": float(rep_in),
            "out_repeated_ratio": float(rep_out),
            "max_parallel_edge": max_parallel,
            "mean_parallel_edge": float(mean_parallel),
            "reciprocal_in": recip_in,
            "reciprocal_out": recip_out,
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)
    return df

def run_inference(dataset_path: str, checkpoint_path: str = None) -> pd.DataFrame:
    """End-to-end inference producing entity scores and structural features."""
    state = prepare_detection_data(dataset_path)
    return predict_from_prepared_data(state, checkpoint_path)
