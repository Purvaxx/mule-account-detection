import pytest
from pathlib import Path
import networkx as nx
import pandas as pd
import json

# The reference dataset is NOT vendored into the repository; it lives one level
# above the project directory (SIH/dkdata.json).  Resolve it from this file's
# location rather than the working directory, matching the convention already
# used by tests/test_ingestion.py, test_preprocessing.py and test_graph_builder.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]          # fraud-detection/
_DATASET_PATH = _REPO_ROOT.parent / "dkdata.json"         # SIH/dkdata.json

# Try to cache heavy objects globally to avoid recomputing them
_reference_graph = None
_reference_baseline = None
_reference_multigraph = None
_reference_temporal = None
_reference_tda = None
_reference_pyg = None

@pytest.fixture(scope="session")
def reference_dataset():
    """Provides the full reference dataset components efficiently."""
    if not _DATASET_PATH.exists():
        pytest.skip(f"Reference dataset not available at {_DATASET_PATH}")

    # Same ingestion -> preprocessing -> graph pipeline, taken from the shared
    # session cache in tests/test_utils.py so the reference graph is built once
    # for the whole suite.  The caller still gets its own copy.
    from tests.test_utils import get_cached_real_dataset_graph
    return get_cached_real_dataset_graph()

@pytest.fixture(scope="session")
def reference_features(reference_dataset):
    """Provides computed features for the reference dataset efficiently."""
    global _reference_baseline, _reference_multigraph, _reference_temporal, _reference_tda
    from src.features.baseline import compute_baseline
    from src.features.multigraph import compute_multigraph
    from src.features.temporal import compute_temporal
    from src.features.tda import compute_tga
    
    if _reference_baseline is None:
        _reference_baseline = compute_baseline(reference_dataset.graph, reference_dataset.edge_table)
    if _reference_multigraph is None:
        _reference_multigraph = compute_multigraph(reference_dataset.graph, reference_dataset.edge_table)
    if _reference_temporal is None:
        _reference_temporal = compute_temporal(reference_dataset.graph, reference_dataset.edge_table)
    if _reference_tda is None:
        _reference_tda = compute_tga(reference_dataset.graph)
        
    return {
        "baseline": _reference_baseline,
        "multigraph": _reference_multigraph,
        "temporal": _reference_temporal,
        "tda": _reference_tda
    }

@pytest.fixture(scope="session")
def small_fixture():
    """Provides a small deterministic mock dataframe for UI/API/Mapper tests."""
    df = pd.DataFrame({
        "id": ["t1", "t2"],
        "date": ["2020-01-01", "2020-01-02"],
        "payer_name": ["Alice", "Bob"],
        "payer_name_norm": ["alice", "bob"],
        "payer_account": ["A1", "B1"],
        "payer_type": ["PERSON", "PERSON"],
        "payer_jurisdiction": ["US", "US"],
        "payer_bank_country": ["US", "US"],
        "beneficiary_name": ["Charlie", "Dave"],
        "beneficiary_name_norm": ["charlie", "dave"],
        "beneficiary_account": ["C1", "D1"],
        "beneficiary_type": ["PERSON", "PERSON"],
        "beneficiary_jurisdiction": ["US", "US"],
        "beneficiary_bank_country": ["US", "US"],
        "amount_orig": [100.0, 200.0],
        "amount_orig_currency": ["USD", "USD"],
        "amount_usd": [100.0, 200.0],
        "amount_eur": [90.0, 180.0],
        "purpose": ["test", "test"],
        "source_file": ["test.csv", "test.csv"],
        "investigation": [False, False],
        "payer_core": [True, True],
        "beneficiary_core": [False, False]
    })
    return df


# ---------------------------------------------------------------------------
# Reference-dataset detection state reuse
# ---------------------------------------------------------------------------
# ``src.detection.inference.prepare_detection_data`` is a pure, deterministic
# function of the dataset file: ingestion -> preprocessing -> graph -> baseline
# and multigraph features -> PyG tensors.  Nothing in it is random and nothing
# is trained.  Roughly a dozen tests across the UI, inference and contract
# suites each called it on the same reference dataset, at ~8 s a call.
#
# The fixture below redirects those calls to a session cache that runs the real
# production function once and hands every caller its own deep copy, so tests
# stay mutation-isolated.  Tests that must exercise the uncached path — the
# Reference Dataset regression — opt out with ``@pytest.mark.no_detection_cache``.

@pytest.fixture(autouse=True)
def _reuse_reference_detection_state(request, monkeypatch):
    if request.node.get_closest_marker("no_detection_cache"):
        return

    from tests.test_utils import cached_prepare_detection_data
    import src.detection.inference as _detection_inference
    import src.ui.resolver as _ui_resolver

    monkeypatch.setattr(
        _detection_inference, "prepare_detection_data", cached_prepare_detection_data
    )
    # ``src.ui.resolver`` imported the symbol by value, so it needs its own
    # rebinding for the resolver's reference-dataset path to hit the cache.
    monkeypatch.setattr(
        _ui_resolver, "prepare_detection_data", cached_prepare_detection_data
    )


# The external-dataset equivalent.  ``prepare_workspace_detection_data`` rebuilds
# graph, features and tensors from a workspace's ``canonical.parquet``; the
# workspace-inference tests run it eleven times over byte-identical canonical
# data at ~6 s a call.  Cached on the content of that file, so two workspaces
# holding different data never share a state.

@pytest.fixture(autouse=True)
def _reuse_workspace_detection_state(request, monkeypatch):
    if request.node.get_closest_marker("no_detection_cache"):
        return

    from tests.test_utils import cached_prepare_workspace_detection_data
    import src.datasets.inference as _datasets_inference
    import src.ui.resolver as _ui_resolver

    monkeypatch.setattr(
        _datasets_inference,
        "prepare_workspace_detection_data",
        cached_prepare_workspace_detection_data,
    )
    monkeypatch.setattr(
        _ui_resolver,
        "prepare_workspace_detection_data",
        cached_prepare_workspace_detection_data,
    )
