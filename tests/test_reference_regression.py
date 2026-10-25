import pytest
import pandas as pd
import math
from src.detection.inference import run_inference
from src.ui.config import UIConfig

@pytest.mark.no_detection_cache
def test_reference_dataset_exact_regression():
    # Run the original path
    df = run_inference("../dkdata.json", UIConfig.CHECKPOINT_PATH)
    
    # Check shape
    assert len(df) == 3440
    
    # Check top entities
    df_sorted = df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)
    
    # Expected known top entities from Experiment 3 GraphSAGE checkpoint
    expected_top = [
        {'entity_id': 'HILUX SERVICES LP', 'risk_score': 0.56275397, 'predicted_class': 1},
        {'entity_id': 'LCM ALLIANCE LLP', 'risk_score': 0.55135107, 'predicted_class': 1},
        {'entity_id': 'METASTAR INVEST LLP', 'risk_score': 0.55129605, 'predicted_class': 1}
    ]
    
    for i, expected in enumerate(expected_top):
        actual = df_sorted.iloc[i]
        assert actual["entity_id"] == expected["entity_id"]
        assert actual["predicted_class"] == expected["predicted_class"]
        # Float tolerance
        assert math.isclose(actual["risk_score"], expected["risk_score"], abs_tol=1e-5)
    
    # Check if features are derived correctly (df output has the 14 columns)
    assert len(df.columns) >= 14
