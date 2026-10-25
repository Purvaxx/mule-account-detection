"""
Checkpoint Training Script
==========================
Trains the Experiment 3 model (Baseline + Multigraph) and saves a persisted
checkpoint for downstream inference/demo use.
"""

import os
import sys
import logging
import warnings
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph
from src.models.dataset import build_pyg_data, build_cv_folds
from src.detection.inference import train_inference_model

def train_and_save_checkpoint(dataset_path: str, output_path: str):
    logging.info("Starting checkpoint training pipeline...")
    
    # 1. Ingestion & Preprocessing
    ing_res = load_transactions(dataset_path)
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    
    # 2. Features
    baseline = compute_baseline(graph_res.graph, graph_res.edge_table)
    multi = compute_multigraph(graph_res.graph, graph_res.edge_table)
    
    # 3. PyG Data
    pyg = build_pyg_data(graph_res.graph, baseline, multigraph=multi)
    
    # 4. Folds (Fold 0 for training provenance)
    folds = build_cv_folds(pyg, seed=42)
    fold0 = folds[0]
    
    # 5. Train model (on-the-fly path, explicitly called here)
    logging.info("Training Experiment 3 model on Fold 0...")
    model, threshold = train_inference_model(pyg, fold0)
    
    # 6. Save checkpoint with metadata
    feature_names = list(baseline.features.columns) + list(multi.features.columns)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "version": "1.0",
            "architecture": "BaselineGraphSAGE",
            "input_dim": pyg.feature_dim,
            "hidden_dim": 64,
            "num_layers": 2,
            "feature_names": feature_names,
            "seed": 42,
            "provenance": "Experiment 3, Fold 0",
            "training_config": {
                "lr": 0.01,
                "weight_decay": 5e-4,
                "dropout": 0.5,
                "max_epochs": 200,
                "patience": 50
            },
            "threshold": threshold,
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    logging.info(f"Checkpoint saved successfully to: {output_path}")

if __name__ == "__main__":
    train_and_save_checkpoint("../dkdata.json", "artifacts/models/experiment3_graphsage.pt")
