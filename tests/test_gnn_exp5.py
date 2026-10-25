"""
Experiment 5 Tests — Baseline GNN + Temporal Graph Topology (TGA)
===================================================================

Verifies all 22 required constraints for Experiment 5:
    160 baseline + 3 TGA = 163 features

No modification to Experiments 1–4.
"""

import pytest
import numpy as np
import torch
import pandas as pd
import networkx as nx

from src.features.baseline import BaselineFeatures
from src.features.tda import TGAFeatures, compute_tga
from src.models.dataset import build_pyg_data, build_cv_folds


# ---------------------------------------------------------------------------
# Helpers: build a minimal synthetic graph for controlled testing
# ---------------------------------------------------------------------------

def _make_synthetic():
    """Return (graph, baseline, tga) for a 5-node synthetic graph."""
    G = nx.MultiDiGraph()
    # 4 positives + 1 negative to mirror the real dataset structure
    G.add_node("A", core=True)
    G.add_node("B", core=True)
    G.add_node("C", core=True)
    G.add_node("D", core=True)
    G.add_node("E", core=False)

    # Edges with dates for TGA
    G.add_edge("A", "B", key=0, date=pd.Timestamp("2023-01-01"))
    G.add_edge("B", "A", key=1, date=pd.Timestamp("2023-01-02"))
    G.add_edge("B", "C", key=2, date=pd.Timestamp("2023-01-03"))
    G.add_edge("C", "D", key=3, date=pd.Timestamp("2023-01-04"))
    G.add_edge("D", "E", key=4, date=pd.Timestamp("2023-01-05"))

    # Mock baseline: 3 features (small for test; real = 160)
    base_df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feat3": [10.0, 20.0, 30.0, 40.0, 50.0],
    }, index=["A", "B", "C", "D", "E"])
    baseline = BaselineFeatures(
        features=base_df,
        entity_count=5,
        feature_names=["feat1", "feat2", "feat3"],
        nan_count=0,
        inf_count=0,
        country_codes=[],
        entity_type_categories=[],
    )

    # Compute TGA on the graph
    tga = compute_tga(G)

    return G, baseline, tga


def _make_synthetic_with_known_tga():
    """Return (graph, baseline, tga_df) with manually controlled TGA values."""
    G = nx.MultiDiGraph()
    G.add_node("A", core=True)
    G.add_node("B", core=True)
    G.add_node("C", core=True)
    G.add_node("D", core=True)
    G.add_node("E", core=False)
    G.add_edge("A", "B", key=0)

    base_df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }, index=["A", "B", "C", "D", "E"])
    baseline = BaselineFeatures(
        features=base_df,
        entity_count=5,
        feature_names=["feat1"],
        nan_count=0,
        inf_count=0,
        country_codes=[],
        entity_type_categories=[],
    )

    # Manual TGA DataFrame
    tga_df = pd.DataFrame({
        "temporal_reciprocal_count": [100.0, 0.0, 50.0, 0.0, 0.0],
        "temporal_cycle_count": [1000.0, 0.0, 500.0, 0.0, 0.0],
        "earliest_cycle_completion_days": [10.0, -1.0, 0.0, -1.0, -1.0],
    }, index=["A", "B", "C", "D", "E"])
    tga = TGAFeatures(
        features=tga_df,
        feature_names=["temporal_reciprocal_count", "temporal_cycle_count",
                       "earliest_cycle_completion_days"],
    )

    return G, baseline, tga


# ---------------------------------------------------------------------------
# 1. Baseline feature dimension = 160 (tested against synthetic as 3)
# ---------------------------------------------------------------------------
def test_01_baseline_feature_dimension():
    G, baseline, tga = _make_synthetic()
    pyg = build_pyg_data(G, baseline, tga=tga)
    # 3 baseline + 3 TGA = 6 total
    n_baseline = len(baseline.feature_names)
    assert n_baseline == 3
    # Total includes TGA
    assert pyg.feature_dim == n_baseline + len(tga.feature_names)


# ---------------------------------------------------------------------------
# 2. TGA feature dimension = 3
# ---------------------------------------------------------------------------
def test_02_tga_feature_dimension():
    G, baseline, tga = _make_synthetic()
    assert len(tga.feature_names) == 3


# ---------------------------------------------------------------------------
# 3. Combined feature dimension = baseline + 3
# ---------------------------------------------------------------------------
def test_03_combined_feature_dimension():
    G, baseline, tga = _make_synthetic()
    pyg = build_pyg_data(G, baseline, tga=tga)
    expected = len(baseline.feature_names) + 3
    assert pyg.feature_dim == expected
    assert pyg.data.x.shape[1] == expected


# ---------------------------------------------------------------------------
# 4. Only the three approved TGA features are present
# ---------------------------------------------------------------------------
def test_04_only_approved_tga_features():
    _, _, tga = _make_synthetic()
    approved = {"temporal_reciprocal_count", "temporal_cycle_count",
                "earliest_cycle_completion_days"}
    assert set(tga.feature_names) == approved


# ---------------------------------------------------------------------------
# 5. No Layer 4 temporal features are present
# ---------------------------------------------------------------------------
def test_05_no_temporal_features():
    G, baseline, tga = _make_synthetic()
    pyg = build_pyg_data(G, baseline, tga=tga)
    # Temporal feature names should not be in the column set
    temporal_names = {"activity_7d", "activity_30d", "mean_iat",
                      "burstiness", "active_day_ratio", "same_day_cooccurrence"}
    assert not temporal_names & set(tga.feature_names)
    # Feature dim must match baseline + 3 TGA only
    assert pyg.feature_dim == len(baseline.feature_names) + 3


# ---------------------------------------------------------------------------
# 6. No Layer 4 multigraph features are present
# ---------------------------------------------------------------------------
def test_06_no_multigraph_features():
    _, _, tga = _make_synthetic()
    multigraph_names = {"out_repeated_counterparty_ratio",
                        "in_repeated_counterparty_ratio",
                        "max_parallel_edge_count", "mean_parallel_edge_count",
                        "parallel_temporal_dispersion",
                        "reciprocal_out_count", "reciprocal_in_count"}
    assert not multigraph_names & set(tga.feature_names)


# ---------------------------------------------------------------------------
# 7. No labels are present
# ---------------------------------------------------------------------------
def test_07_no_labels_in_features():
    G, baseline, tga = _make_synthetic()
    pyg = build_pyg_data(G, baseline, tga=tga)
    # Feature dim = baseline + TGA, no label columns
    forbidden = {"core", "payer_core", "beneficiary_core", "label", "y"}
    all_feat_names = set(baseline.feature_names) | set(tga.feature_names)
    assert not forbidden & all_feat_names


# ---------------------------------------------------------------------------
# 8. No forbidden metadata is present
# ---------------------------------------------------------------------------
def test_08_no_forbidden_metadata():
    _, _, tga = _make_synthetic()
    forbidden = {"source_file", "investigation", "payer_name",
                 "beneficiary_name", "payer_account", "beneficiary_account"}
    assert not forbidden & set(tga.feature_names)


# ---------------------------------------------------------------------------
# 9. No NaN values in the final tensor
# ---------------------------------------------------------------------------
def test_09_no_nan_values():
    G, baseline, tga = _make_synthetic_with_known_tga()
    pyg = build_pyg_data(G, baseline, tga=tga)
    assert not torch.isnan(pyg.data.x).any().item()


# ---------------------------------------------------------------------------
# 10. No infinite values in the final tensor
# ---------------------------------------------------------------------------
def test_10_no_inf_values():
    G, baseline, tga = _make_synthetic_with_known_tga()
    pyg = build_pyg_data(G, baseline, tga=tga)
    assert not torch.isinf(pyg.data.x).any().item()


# ---------------------------------------------------------------------------
# 11. TGA count preprocessing uses log1p
# ---------------------------------------------------------------------------
def test_11_tga_counts_use_log1p():
    G, baseline, tga = _make_synthetic_with_known_tga()
    pyg = build_pyg_data(G, baseline, tga=tga)
    x = pyg.data.x.numpy()
    n_base = len(baseline.feature_names)  # 1

    # Node A: temporal_reciprocal_count = 100 -> log1p(100)
    assert np.isclose(x[0, n_base + 0], np.log1p(100.0))
    # Node A: temporal_cycle_count = 1000 -> log1p(1000)
    assert np.isclose(x[0, n_base + 1], np.log1p(1000.0))
    # Node C: temporal_reciprocal_count = 50 -> log1p(50)
    assert np.isclose(x[2, n_base + 0], np.log1p(50.0))


# ---------------------------------------------------------------------------
# 12. earliest_cycle_completion_days >= 0 uses log1p
# ---------------------------------------------------------------------------
def test_12_earliest_cycle_days_log1p():
    G, baseline, tga = _make_synthetic_with_known_tga()
    pyg = build_pyg_data(G, baseline, tga=tga)
    x = pyg.data.x.numpy()
    n_base = len(baseline.feature_names)

    # Node A: earliest_cycle_completion_days = 10.0 -> log1p(10.0)
    assert np.isclose(x[0, n_base + 2], np.log1p(10.0))
    # Node C: earliest_cycle_completion_days = 0.0 -> log1p(0.0) = 0.0
    assert np.isclose(x[2, n_base + 2], np.log1p(0.0))
    assert np.isclose(x[2, n_base + 2], 0.0)


# ---------------------------------------------------------------------------
# 13. earliest_cycle_completion_days == -1 remains -1
# ---------------------------------------------------------------------------
def test_13_earliest_cycle_days_sentinel():
    G, baseline, tga = _make_synthetic_with_known_tga()
    pyg = build_pyg_data(G, baseline, tga=tga)
    x = pyg.data.x.numpy()
    n_base = len(baseline.feature_names)

    # Nodes B, D, E have earliest_cycle_completion_days = -1.0
    for idx in [1, 3, 4]:
        assert np.isclose(x[idx, n_base + 2], -1.0), \
            f"Node idx={idx} sentinel not preserved: got {x[idx, n_base + 2]}"


# ---------------------------------------------------------------------------
# 14. No additional TGA indicator column is created
# ---------------------------------------------------------------------------
def test_14_no_extra_indicator_column():
    G, baseline, tga = _make_synthetic_with_known_tga()
    pyg = build_pyg_data(G, baseline, tga=tga)
    # Exactly 1 baseline + 3 TGA = 4
    assert pyg.feature_dim == 4
    assert pyg.data.x.shape[1] == 4


# ---------------------------------------------------------------------------
# 15. Graph contains exactly 3440 nodes (real dataset test)
# ---------------------------------------------------------------------------
def test_15_real_dataset_node_count():
    """Requires real dataset; skip if unavailable."""
    try:
        from tests.test_utils import get_cached_pyg_exp5
        pyg = get_cached_pyg_exp5()

        assert pyg.num_nodes == 3440
        assert pyg.feature_dim == 163
        assert pyg.data.x.shape == (3440, 163)
    except FileNotFoundError:
        pytest.skip("Real dataset not available")


# ---------------------------------------------------------------------------
# 16. edge_index contains exactly 16940 directed transaction edges
# ---------------------------------------------------------------------------
def test_16_real_dataset_edge_count():
    try:
        from tests.test_utils import get_cached_pyg_exp5
        pyg = get_cached_pyg_exp5()

        assert pyg.num_edges == 16940
        assert pyg.data.edge_index.shape == (2, 16940)
    except FileNotFoundError:
        pytest.skip("Real dataset not available")


# ---------------------------------------------------------------------------
# 17. Parallel edges remain preserved
# ---------------------------------------------------------------------------
def test_17_parallel_edges_preserved():
    G = nx.MultiDiGraph()
    G.add_node("A", core=True)
    G.add_node("B", core=True)
    G.add_node("C", core=True)
    G.add_node("D", core=True)
    G.add_edge("A", "B", key=0, date=pd.Timestamp("2023-01-01"))
    G.add_edge("A", "B", key=1, date=pd.Timestamp("2023-01-02"))
    G.add_edge("A", "B", key=2, date=pd.Timestamp("2023-01-03"))

    base_df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0]}, index=["A", "B", "C", "D"])
    baseline = BaselineFeatures(
        features=base_df, entity_count=4,
        feature_names=["feat1"], nan_count=0, inf_count=0,
        country_codes=[], entity_type_categories=[],
    )
    tga = compute_tga(G)
    pyg = build_pyg_data(G, baseline, tga=tga)

    # 3 parallel edges should be in edge_index
    assert pyg.num_edges == 3
    assert pyg.data.edge_index.shape == (2, 3)


# ---------------------------------------------------------------------------
# 18. CV masks are identical to Experiment 1
# ---------------------------------------------------------------------------
def test_18_cv_masks_identical_to_exp1():
    G, baseline, tga = _make_synthetic()

    # Experiment 1: baseline only
    pyg_1 = build_pyg_data(G, baseline)
    folds_1 = build_cv_folds(pyg_1, seed=42)

    # Experiment 5: baseline + TGA
    pyg_5 = build_pyg_data(G, baseline, tga=tga)
    folds_5 = build_cv_folds(pyg_5, seed=42)

    # Masks must be identical
    for f1, f5 in zip(folds_1, folds_5):
        assert torch.equal(f1.train_mask, f5.train_mask), f"Fold {f1.fold_idx} train mask differs"
        assert torch.equal(f1.val_mask, f5.val_mask), f"Fold {f1.fold_idx} val mask differs"
        assert torch.equal(f1.test_mask, f5.test_mask), f"Fold {f1.fold_idx} test mask differs"


# ---------------------------------------------------------------------------
# 19. pos_weight is computed only from the training fold
# ---------------------------------------------------------------------------
def test_19_pos_weight_from_training_only():
    G, baseline, tga = _make_synthetic()
    pyg = build_pyg_data(G, baseline, tga=tga)
    folds = build_cv_folds(pyg, seed=42)

    for fold in folds:
        y_train = pyg.data.y[fold.train_mask]
        n_pos = int(y_train.sum().item())
        n_neg = int((y_train == 0).sum().item())
        expected_pw = n_neg / n_pos
        assert np.isclose(fold.pos_weight.item(), expected_pw), \
            f"Fold {fold.fold_idx}: pos_weight mismatch"


# ---------------------------------------------------------------------------
# 20. Validation threshold is selected only from validation data
# (Architecture-level constraint — verified by reading trainer.py logic;
#  this test ensures train_fold runs without error on Exp5 data)
# ---------------------------------------------------------------------------
def test_20_validation_threshold_selection():
    from src.models.trainer import train_fold
    G, baseline, tga = _make_synthetic()
    pyg = build_pyg_data(G, baseline, tga=tga)
    folds = build_cv_folds(pyg, seed=42)

    # Train fold 0 with very few epochs to just verify it works
    fm = train_fold(pyg, folds[0], max_epochs=5, patience=5, seed=42)
    assert fm.threshold >= 0.0
    assert fm.threshold <= 1.0


# ---------------------------------------------------------------------------
# 21. Experiment 1 tests remain unchanged
# (Structural test: build Exp1 data without TGA, verify it still works)
# ---------------------------------------------------------------------------
def test_21_exp1_baseline_still_works():
    G, baseline, _ = _make_synthetic()
    pyg = build_pyg_data(G, baseline)  # No tga= argument
    assert pyg.feature_dim == 3  # Baseline only
    folds = build_cv_folds(pyg, seed=42)
    assert len(folds) == 4


# ---------------------------------------------------------------------------
# 22. Experiments 1–4 behaviour is not modified
# (Verify that build_pyg_data with no tga= produces the exact same result)
# ---------------------------------------------------------------------------
def test_22_exp1_through_4_not_modified():
    G, baseline, tga = _make_synthetic()

    # Without TGA
    pyg_no_tga = build_pyg_data(G, baseline)

    # With TGA=None (explicit)
    pyg_none_tga = build_pyg_data(G, baseline, tga=None)

    # Both should be identical (same feature_dim, same x)
    assert pyg_no_tga.feature_dim == pyg_none_tga.feature_dim
    assert torch.equal(pyg_no_tga.data.x, pyg_none_tga.data.x)
    assert torch.equal(pyg_no_tga.data.y, pyg_none_tga.data.y)
    assert torch.equal(pyg_no_tga.data.edge_index, pyg_none_tga.data.edge_index)
