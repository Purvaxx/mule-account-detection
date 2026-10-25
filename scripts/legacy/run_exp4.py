import sys
import logging
import warnings
import pandas as pd
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.temporal import compute_temporal
from src.features.multigraph import compute_multigraph
from src.models.dataset import build_pyg_data, build_cv_folds
from src.models.trainer import run_cross_validation

def run_experiment_4():
    print("Loading data...")
    ing_res = load_transactions("../dkdata.json")
    if not ing_res.is_valid:
        print("Ingestion failed")
        return
    
    print("Preprocessing...")
    pre_res = preprocess(ing_res)
    if not pre_res.is_valid:
        print("Preprocessing failed")
        return
        
    print("Building graph...")
    graph_res = build_graph(pre_res)
    if not graph_res.is_valid:
        print("Graph construction failed")
        return
        
    print("Computing features...")
    baseline = compute_baseline(graph_res.graph, graph_res.edge_table)
    temporal = compute_temporal(graph_res.edge_table, list(graph_res.graph.nodes()))
    multigraph = compute_multigraph(graph_res.graph, graph_res.edge_table)
    
    # ---------------------------------------------------------
    # Experiment 1: Baseline
    # ---------------------------------------------------------
    print("\n--- Running Experiment 1 (Baseline) ---")
    pyg_1 = build_pyg_data(graph_res.graph, baseline)
    folds_1 = build_cv_folds(pyg_1, seed=42)
    cv_res_1 = run_cross_validation(pyg_1, folds_1, max_epochs=200, patience=50, seed=42)
    
    # ---------------------------------------------------------
    # Experiment 3: Baseline + Multigraph
    # ---------------------------------------------------------
    print("\n--- Running Experiment 3 (Baseline + Multigraph) ---")
    pyg_3 = build_pyg_data(graph_res.graph, baseline, multigraph=multigraph)
    folds_3 = build_cv_folds(pyg_3, seed=42)
    cv_res_3 = run_cross_validation(pyg_3, folds_3, max_epochs=200, patience=50, seed=42)
    
    # ---------------------------------------------------------
    # Experiment 4: Baseline + Temporal + Multigraph
    # ---------------------------------------------------------
    print("\n--- Running Experiment 4 (Baseline + Temporal + Multigraph) ---")
    # Build PyG data with temporal AND multigraph features
    pyg_4 = build_pyg_data(graph_res.graph, baseline, temporal=temporal, multigraph=multigraph)
    
    print("\nVerifying constraints for Experiment 4:")
    print(f"Number of nodes: {pyg_4.num_nodes} (Expected: 3440)")
    print(f"Number of edges: {pyg_4.num_edges} (Expected: 16940)")
    print(f"Feature dim: {pyg_4.feature_dim} (Expected: 173)")
    
    # Check if any NaN remaining in feature tensor
    has_nan = torch.isnan(pyg_4.data.x).any().item()
    print(f"Any NaNs in tensor? {has_nan} (Expected: False. Handled by Section 28.1 and 28.2 preprocessing)")
    
    folds_4 = build_cv_folds(pyg_4, seed=42)
    cv_res_4 = run_cross_validation(pyg_4, folds_4, max_epochs=200, patience=50, seed=42)
    
    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("EXPERIMENT 4 EVALUATION REPORT")
    print("="*80)
    
    print(cv_res_4.summary())
    
    print("\n" + "="*80)
    print("COMPARISON: EXPERIMENT 4 vs EXPERIMENT 1 vs EXPERIMENT 3")
    print("="*80)
    
    print(f"Mean PR-AUC:")
    print(f"  Exp 1: {cv_res_1.mean_pr_auc:.4f}")
    print(f"  Exp 3: {cv_res_3.mean_pr_auc:.4f}")
    print(f"  Exp 4: {cv_res_4.mean_pr_auc:.4f}")
    print(f"  Delta Exp4-Exp1: {cv_res_4.mean_pr_auc - cv_res_1.mean_pr_auc:+.4f}")
    print(f"  Delta Exp4-Exp3: {cv_res_4.mean_pr_auc - cv_res_3.mean_pr_auc:+.4f}")

    print(f"\nMean ROC-AUC:")
    print(f"  Exp 1: {cv_res_1.mean_roc_auc:.4f}")
    print(f"  Exp 3: {cv_res_3.mean_roc_auc:.4f}")
    print(f"  Exp 4: {cv_res_4.mean_roc_auc:.4f}")
    print(f"  Delta Exp4-Exp1: {cv_res_4.mean_roc_auc - cv_res_1.mean_roc_auc:+.4f}")
    print(f"  Delta Exp4-Exp3: {cv_res_4.mean_roc_auc - cv_res_3.mean_roc_auc:+.4f}")

    print(f"\nMean F1:")
    print(f"  Exp 1: {cv_res_1.mean_f1:.4f}")
    print(f"  Exp 3: {cv_res_3.mean_f1:.4f}")
    print(f"  Exp 4: {cv_res_4.mean_f1:.4f}")
    print(f"  Delta Exp4-Exp1: {cv_res_4.mean_f1 - cv_res_1.mean_f1:+.4f}")
    print(f"  Delta Exp4-Exp3: {cv_res_4.mean_f1 - cv_res_3.mean_f1:+.4f}")

if __name__ == "__main__":
    run_experiment_4()
