"""
Experiment 5 — Baseline GNN + Temporal Graph Topology (TGA)
==============================================================

Evaluates whether causal, time-respecting directed-cycle information
provides additional predictive value beyond the 160-dimensional baseline.

Feature composition:
    160 baseline + 3 TGA = 163 total features

TGA features:
    1. temporal_reciprocal_count     (log1p preprocessed)
    2. temporal_cycle_count          (log1p preprocessed)
    3. earliest_cycle_completion_days (log1p for x >= 0; -1.0 sentinel preserved)

All other settings (architecture, optimizer, seeds, folds, etc.) are
identical to Experiment 1.
"""

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
from src.features.tda import compute_tga
from src.models.dataset import build_pyg_data, build_cv_folds
from src.models.trainer import run_cross_validation


def run_experiment_5():
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

    print("Computing baseline features...")
    baseline = compute_baseline(graph_res.graph, graph_res.edge_table)

    print("Computing TGA features...")
    tga = compute_tga(graph_res.graph, graph_res.edge_table)

    # ---------------------------------------------------------
    # Experiment 1: Baseline (for comparison)
    # ---------------------------------------------------------
    print("\n--- Running Experiment 1 (Baseline) ---")
    pyg_1 = build_pyg_data(graph_res.graph, baseline)
    folds_1 = build_cv_folds(pyg_1, seed=42)
    cv_res_1 = run_cross_validation(pyg_1, folds_1, max_epochs=200, patience=50, seed=42)

    # ---------------------------------------------------------
    # Experiment 5: Baseline + TGA
    # ---------------------------------------------------------
    print("\n--- Running Experiment 5 (Baseline + TGA) ---")
    pyg_5 = build_pyg_data(graph_res.graph, baseline, tga=tga)

    print("\nVerifying constraints for Experiment 5:")
    print(f"Number of nodes: {pyg_5.num_nodes} (Expected: 3440)")
    print(f"Number of edges: {pyg_5.num_edges} (Expected: 16940)")
    print(f"Feature dim: {pyg_5.feature_dim} (Expected: 163)")

    # Check if any NaN remaining in feature tensor
    has_nan = torch.isnan(pyg_5.data.x).any().item()
    has_inf = torch.isinf(pyg_5.data.x).any().item()
    print(f"Any NaNs in tensor? {has_nan} (Expected: False)")
    print(f"Any Infs in tensor? {has_inf} (Expected: False)")

    folds_5 = build_cv_folds(pyg_5, seed=42)
    cv_res_5 = run_cross_validation(pyg_5, folds_5, max_epochs=200, patience=50, seed=42)

    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 5 EVALUATION REPORT")
    print("Baseline GNN + Temporal Graph Topology (TGA)")
    print("160 baseline + 3 TGA = 163 features")
    print("=" * 80)

    print(cv_res_5.summary())

    # Experiment 1 comparison
    print("\n" + "=" * 80)
    print("COMPARISON: EXPERIMENT 5 vs EXPERIMENT 1")
    print("=" * 80)

    delta_pr_auc = cv_res_5.mean_pr_auc - cv_res_1.mean_pr_auc
    delta_roc_auc = cv_res_5.mean_roc_auc - cv_res_1.mean_roc_auc
    delta_f1 = cv_res_5.mean_f1 - cv_res_1.mean_f1
    delta_prec = cv_res_5.mean_precision - cv_res_1.mean_precision
    delta_recall = cv_res_5.mean_recall - cv_res_1.mean_recall

    print(f"Mean PR-AUC:   Exp1={cv_res_1.mean_pr_auc:.4f}±{cv_res_1.std_pr_auc:.4f} | "
          f"Exp5={cv_res_5.mean_pr_auc:.4f}±{cv_res_5.std_pr_auc:.4f} | "
          f"Delta={delta_pr_auc:+.4f}")
    print(f"Mean ROC-AUC:  Exp1={cv_res_1.mean_roc_auc:.4f}±{cv_res_1.std_roc_auc:.4f} | "
          f"Exp5={cv_res_5.mean_roc_auc:.4f}±{cv_res_5.std_roc_auc:.4f} | "
          f"Delta={delta_roc_auc:+.4f}")
    print(f"Mean F1:       Exp1={cv_res_1.mean_f1:.4f}±{cv_res_1.std_f1:.4f} | "
          f"Exp5={cv_res_5.mean_f1:.4f}±{cv_res_5.std_f1:.4f} | "
          f"Delta={delta_f1:+.4f}")
    print(f"Mean Precision:Exp1={cv_res_1.mean_precision:.4f}±{cv_res_1.std_precision:.4f} | "
          f"Exp5={cv_res_5.mean_precision:.4f}±{cv_res_5.std_precision:.4f} | "
          f"Delta={delta_prec:+.4f}")
    print(f"Mean Recall:   Exp1={cv_res_1.mean_recall:.4f}±{cv_res_1.std_recall:.4f} | "
          f"Exp5={cv_res_5.mean_recall:.4f}±{cv_res_5.std_recall:.4f} | "
          f"Delta={delta_recall:+.4f}")

    # Experiment 3 reference comparison
    print("\n" + "=" * 80)
    print("REFERENCE: EXPERIMENT 3 (Baseline + Directed-Multigraph)")
    print("=" * 80)
    print("Exp3 PR-AUC:  1.0000 ± 0.0000")
    print("Exp3 ROC-AUC: 1.0000 ± 0.0000")
    print("Exp3 F1:      1.0000 ± 0.0000")
    delta_pr_vs3 = cv_res_5.mean_pr_auc - 1.0
    delta_roc_vs3 = cv_res_5.mean_roc_auc - 1.0
    delta_f1_vs3 = cv_res_5.mean_f1 - 1.0
    print(f"\nExp5 vs Exp3:")
    print(f"  Delta PR-AUC:  {delta_pr_vs3:+.4f}")
    print(f"  Delta ROC-AUC: {delta_roc_vs3:+.4f}")
    print(f"  Delta F1:      {delta_f1_vs3:+.4f}")

    print("\n" + "=" * 80)
    print("SCIENTIFIC NOTES")
    print("=" * 80)
    print("1. TGA features are time-respecting directed-cycle counts, NOT")
    print("   persistent homology or algebraic topology.")
    print("2. With only 4 positive entities in 3,440 nodes, per-fold metrics")
    print("   have extreme variance and should not be over-interpreted.")
    print("3. TGA was INTENDED to capture causal, time-respecting money-flow")
    print("   loops that static multigraph features cannot distinguish from")
    print("   non-causal structural co-occurrence. That intent is stated as a")
    print("   design rationale, not as a measured result: under this protocol")
    print(f"   the PR-AUC delta vs Experiment 1 is {delta_pr_auc:+.4f} and the")
    print(f"   F1 delta is {delta_f1:+.4f}, so no uplift is demonstrated.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_5()
