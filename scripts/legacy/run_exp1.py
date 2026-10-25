import sys
import logging
import warnings
# Suppress the PyTorch JIT deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.ingestion import load_transactions
from src.data.preprocessing import preprocess
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.models.dataset import build_pyg_data, build_cv_folds
from src.models.trainer import run_cross_validation

def run_experiment_1():
    print("Running Experiment 1 - Baseline GNN...")
    # 1. Ingestion
    print("Loading data...")
    ing_res = load_transactions("../dkdata.json")
    if not ing_res.is_valid:
        print("Ingestion failed")
        return
    
    # 2. Preprocessing
    print("Preprocessing...")
    pre_res = preprocess(ing_res)
    if not pre_res.is_valid:
        print("Preprocessing failed")
        return
        
    # 3. Graph
    print("Building graph...")
    graph_res = build_graph(pre_res)
    if not graph_res.is_valid:
        print("Graph construction failed")
        return
        
    # 4. Features
    print("Computing baseline features...")
    baseline = compute_baseline(graph_res.graph, graph_res.edge_table)
    
    # 5. Dataset & Folds
    print("Building PyG data...")
    pyg = build_pyg_data(graph_res.graph, baseline)
    print("Building CV folds...")
    folds = build_cv_folds(pyg, seed=42)
    
    # Check pos_weight calculation and label leakage manually
    print("\nVerifying constraints:")
    for fold in folds:
        print(f"Fold {fold.fold_idx} pos_weight: {fold.pos_weight.item():.2f} (expected ~859)")
    
    forbidden = {"core", "payer_core", "beneficiary_core", "source_file", "investigation"}
    present_forbidden = forbidden & set(baseline.feature_names)
    print(f"Forbidden labels in feature names: {present_forbidden}")
    print(f"Parallel edges preserved? edge_index shape: {pyg.data.edge_index.shape} (should be (2, 16940))")
    print(f"Number of nodes: {pyg.num_nodes}, Features: {pyg.feature_dim}")
    
    # 6. CV
    print("\nRunning cross-validation...")
    cv_res = run_cross_validation(pyg, folds, max_epochs=200, patience=50, seed=42)
    
    print("\nFINAL RESULTS:\n")
    print(cv_res.summary())
    
if __name__ == "__main__":
    run_experiment_1()
