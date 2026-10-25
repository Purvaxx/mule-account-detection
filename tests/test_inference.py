"""Tests for the inference and explainability modules."""

import os
import torch
import pandas as pd
import pytest

import pytest
import os
import pandas as pd
from tests.test_utils import get_cached_pyg_exp5, get_cached_real_dataset_graph, get_cached_baseline

@pytest.fixture(autouse=True)
def mock_prepare_detection_data(monkeypatch):
    """Mocks prepare_detection_data to return a cached DetectionState to speed up all inference tests."""
    from src.detection.inference import DetectionState

    def _mock_prepare(source_file):
        # Always return the cached Exp3/Exp5 state (they use the same dataset)
        # Note: Exp3 has 167 features, Exp5 has 163.
        # Let's just run prepare_detection_data ONCE and cache it ourselves.
        return get_cached_detection_state()

    monkeypatch.setattr("src.detection.inference.prepare_detection_data", _mock_prepare)


# The session-wide cache in tests/test_utils.py holds the same DetectionState
# (produced by the unpatched production function) and already deep-copies on
# every hand-out, so this module shares it rather than building a second copy of
# the same 8-second pipeline.
from tests.test_utils import get_cached_detection_state

from unittest.mock import patch

from src.detection.inference import run_inference
from src.detection.explain import generate_explanation, explain_suspicious_entities

def test_explain_formatting():
    """Test that the explanation module correctly formats the output."""
    data = {
        "entity_id": "TEST_ENTITY",
        "risk_score": 0.95,
        "predicted_class": 1,
        "in_degree": 10,
        "out_degree": 5,
        "total_incoming_usd": 10000.0,
        "total_outgoing_usd": 5000.0,
        "flow_imbalance": 0.3333,
        "in_repeated_ratio": 2.5,
        "out_repeated_ratio": 1.1,
        "max_parallel_edge": 6,
        "mean_parallel_edge": 1.5,
        "reciprocal_in": 1,
        "reciprocal_out": 0,
    }

    series = pd.Series(data)
    explanation = generate_explanation(series)

    # 1. Check sections exist
    assert "MODEL OUTPUT" in explanation
    assert "OBSERVED STRUCTURAL FEATURES" in explanation
    assert "HEURISTIC STRUCTURAL FLAGS" in explanation

    # 2. Check absence of prohibited terms
    prohibited = [
        "KEY EVIDENCE", "anomalous", "pass-through", "nesting", "fraud", "laundering", "malicious"
    ]
    for p in prohibited:
        assert p.lower() not in explanation.lower(), f"Prohibited term found: {p}"

    # 3. Check values and explicit thresholds
    assert "Model Risk Score: 0.9500" in explanation
    assert "Model Prediction: SUSPICIOUS" in explanation
    assert "Total Incoming: $10,000.00" in explanation
    assert "Incoming repetition ratio exceeds configured threshold (2.50 > 2.0)" in explanation
    assert "Maximum parallel-edge count exceeds configured threshold (6 > 5)" in explanation

    # 4. Check disclaimer presence
    assert "represent a direct explanation of the model's internal reasoning" in explanation

def test_explain_suspicious_entities():
    """Test generating explanations for multiple entities."""
    data = [{
        "entity_id": f"ENTITY_{i}",
        "risk_score": 0.9 if i == 0 else 0.1,
        "predicted_class": 1 if i == 0 else 0,
        "in_degree": 10,
        "out_degree": 5,
        "total_incoming_usd": 10000.0,
        "total_outgoing_usd": 5000.0,
        "flow_imbalance": 0.3333,
        "in_repeated_ratio": 2.5,
        "out_repeated_ratio": 1.1,
        "max_parallel_edge": 6,
        "mean_parallel_edge": 1.5,
        "reciprocal_in": 1,
        "reciprocal_out": 0,
    } for i in range(3)]

    df = pd.DataFrame(data)
    report = explain_suspicious_entities(df, top_k=2)

    assert "ENTITY_0" in report
    assert "SUSPICIOUS" in report

def test_checkpoint_rejection_and_loading(tmp_path):
    ckpt_path = tmp_path / "test_model.pt"

    # 1. Missing checkpoint argument
    with pytest.raises(ValueError, match="checkpoint path must be provided"):
        run_inference("../dkdata.json", None)

    # 2. Missing file
    with pytest.raises(FileNotFoundError):
        run_inference("../dkdata.json", str(ckpt_path))

    # 3. Create fake invalid checkpoint (wrong input_dim)
    torch.save({
        "metadata": {"input_dim": 160, "feature_names": []}
    }, ckpt_path)

    with pytest.raises(ValueError, match="Incompatible checkpoint: expected input_dim=167"):
        run_inference("../dkdata.json", str(ckpt_path))

def test_inference_does_not_train(tmp_path):
    ckpt_path = "artifacts/models/experiment3_graphsage.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Checkpoint not built yet")

    # Check that model has 167 features
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt["metadata"]["input_dim"] == 167
    assert len(ckpt["metadata"]["feature_names"]) == 167

    # Run inference and mock train_inference_model to ensure it is not called
    with patch("src.detection.inference.train_inference_model") as mock_train:
        df1 = run_inference("../dkdata.json", ckpt_path)
        mock_train.assert_not_called()

    # Determinism check across two runs
    df2 = run_inference("../dkdata.json", ckpt_path)
    pd.testing.assert_frame_equal(df1, df2)

def test_inference_pipeline_refactor():
    """Ensure prepare_detection_data and predict_from_prepared_data split works correctly."""
    from src.detection.inference import prepare_detection_data, predict_from_prepared_data, DetectionState
    ckpt_path = "artifacts/models/experiment3_graphsage.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Checkpoint not built yet")

    # 1. Prepare data
    state = prepare_detection_data("../dkdata.json")
    assert isinstance(state, DetectionState)

    # 3440 nodes, 16940 edges
    assert state.graph.number_of_nodes() == 3440
    assert state.graph.number_of_edges() == 16940

    # 167 features
    assert state.pyg.feature_dim == 167

    # 2. Predict from state
    df1 = predict_from_prepared_data(state, ckpt_path)

    # 3. Compare with wrapper
    df2 = run_inference("../dkdata.json", ckpt_path)
    pd.testing.assert_frame_equal(df1, df2)

def test_cache_mutation_isolation():
    state_a = get_cached_detection_state()
    state_b = get_cached_detection_state()

    # Mutate PyG graph
    original_nodes = state_a.pyg.num_nodes
    import torch
    state_a.pyg.data.x = torch.zeros((1, 167))

    # Mutate networkx graph
    state_a.graph.add_node("MUTATION_LEAK_NODE")

    # Mutate dataframes
    state_a.baseline.features.loc["MUTATION_LEAK_NODE"] = 0.0

    # Verify B is isolated
    assert state_b.pyg.num_nodes == original_nodes
    assert state_b.pyg.data.x.shape[0] > 1
    assert "MUTATION_LEAK_NODE" not in state_b.graph.nodes
    assert "MUTATION_LEAK_NODE" not in state_b.baseline.features.index
