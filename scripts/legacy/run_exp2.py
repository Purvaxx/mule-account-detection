import sys
import logging
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.temporal import compute_temporal
from src.models.dataset import build_pyg_data, build_cv_folds
from src.models.trainer import run_cross_validation

def run_experiment_2():
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
    
    # ---------------------------------------------------------
    # Experiment 1: Baseline
    # ---------------------------------------------------------
    print("\n--- Running Experiment 1 (Baseline) ---")
    pyg_1 = build_pyg_data(graph_res.graph, baseline)
    folds_1 = build_cv_folds(pyg_1, seed=42)
    cv_res_1 = run_cross_validation(pyg_1, folds_1, max_epochs=200, patience=50, seed=42)
    
    # ---------------------------------------------------------
    # Experiment 2: Baseline + Temporal
    # ---------------------------------------------------------
    print("\n--- Running Experiment 2 (Baseline + Temporal) ---")
    # Build PyG data with temporal features
    pyg_2 = build_pyg_data(graph_res.graph, baseline, temporal=temporal)
    
    print("\nVerifying constraints for Experiment 2:")
    print(f"Number of nodes: {pyg_2.num_nodes} (Expected: 3440)")
    print(f"Number of edges: {pyg_2.num_edges} (Expected: 16940)")
    print(f"Feature dim: {pyg_2.feature_dim} (Expected: 166)")
    
    # Check if any NaN remaining in feature tensor
    has_nan = torch.isnan(pyg_2.data.x).any().item()
    print(f"Any NaNs in tensor? {has_nan} (Expected: False. Handled by filling with 0.0 deterministically)")
    
    folds_2 = build_cv_folds(pyg_2, seed=42)
    cv_res_2 = run_cross_validation(pyg_2, folds_2, max_epochs=200, patience=50, seed=42)
    
    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("EXPERIMENT 2 EVALUATION REPORT")
    print("="*80)
    
    print(cv_res_2.summary())
    
    print("\n" + "="*80)
    print("COMPARISON: EXPERIMENT 2 vs EXPERIMENT 1")
    print("="*80)
    
    delta_pr_auc = cv_res_2.mean_pr_auc - cv_res_1.mean_pr_auc
    delta_roc_auc = cv_res_2.mean_roc_auc - cv_res_1.mean_roc_auc
    delta_f1 = cv_res_2.mean_f1 - cv_res_1.mean_f1
    
    print(f"Mean PR-AUC: Exp1={cv_res_1.mean_pr_auc:.4f} | Exp2={cv_res_2.mean_pr_auc:.4f} | Delta={delta_pr_auc:+.4f}")
    print(f"Mean ROC-AUC: Exp1={cv_res_1.mean_roc_auc:.4f} | Exp2={cv_res_2.mean_roc_auc:.4f} | Delta={delta_roc_auc:+.4f}")
    print(f"Mean F1:      Exp1={cv_res_1.mean_f1:.4f} | Exp2={cv_res_2.mean_f1:.4f} | Delta={delta_f1:+.4f}")
    
    if delta_pr_auc > 0.01:
        print("\nConclusion: Experiment 2 shows measurable improvement over Baseline.")
    elif delta_pr_auc < -0.01:
        print("\nConclusion: Experiment 2 performs worse than Baseline.")
    else:
        print("\nConclusion: Experiment 2 performance is essentially identical to Baseline. No significant improvement.")

if __name__ == "__main__":
    import torch
    run_experiment_2()
