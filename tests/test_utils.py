import functools
import copy
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared, session-lifetime caches for the reference dataset
# ---------------------------------------------------------------------------
# Every object below is produced by the *real* production function and handed
# out as a deep copy, so callers stay mutation-isolated from one another while
# the expensive computation happens once per session.  The pipeline is
# deterministic (no randomness, no training), so a cached value is
# indistinguishable from a freshly computed one.


def resolve_reference_dataset_path() -> str:
    """Locate ``dkdata.json`` the way the individual test modules do."""
    for candidate in ("../dkdata.json", "data/dkdata.json", "dkdata.json"):
        if os.path.exists(candidate):
            return candidate
    return "../dkdata.json"


@functools.lru_cache(maxsize=1)
def _internal_get_cached_real_dataset_graph():
    from src.data.ingestion import load_transactions
    from src.data.preprocessing import preprocess
    from src.graph.builder import build_graph

    path = Path("../dkdata.json")
    if not path.exists():
        path = Path("data/dkdata.json")
    if not path.exists():
        path = Path("dkdata.json")

    ing_res = load_transactions(str(path))
    pre_res = preprocess(ing_res)
    graph_res = build_graph(pre_res)
    return graph_res

def get_cached_real_dataset_graph():
    res = _internal_get_cached_real_dataset_graph()
    return copy.deepcopy(res)

@functools.lru_cache(maxsize=1)
def _internal_get_cached_baseline():
    from src.features.baseline import compute_baseline
    graph_res = _internal_get_cached_real_dataset_graph()
    return compute_baseline(graph_res.graph, graph_res.edge_table)

def get_cached_baseline():
    # Deep-copied whole rather than field-by-field, so the helper does not have
    # to track BaselineFeatures' field list to stay correct.
    return copy.deepcopy(_internal_get_cached_baseline())


@functools.lru_cache(maxsize=1)
def _internal_get_cached_multigraph():
    from src.features.multigraph import compute_multigraph
    graph_res = _internal_get_cached_real_dataset_graph()
    return compute_multigraph(graph_res.graph, graph_res.edge_table)


def get_cached_multigraph():
    return copy.deepcopy(_internal_get_cached_multigraph())


@functools.lru_cache(maxsize=1)
def _internal_get_cached_temporal():
    from src.features.temporal import compute_temporal
    graph_res = _internal_get_cached_real_dataset_graph()
    return compute_temporal(graph_res.edge_table, sorted(graph_res.graph.nodes()))


def get_cached_temporal():
    return copy.deepcopy(_internal_get_cached_temporal())

@functools.lru_cache(maxsize=1)
def _internal_get_cached_tga():
    from src.features.tda import compute_tga
    graph_res = _internal_get_cached_real_dataset_graph()
    return compute_tga(graph_res.graph, graph_res.edge_table)

def get_cached_tga():
    return copy.deepcopy(_internal_get_cached_tga())

@functools.lru_cache(maxsize=1)
def _internal_get_cached_pyg_exp5():
    from src.models.dataset import build_pyg_data
    graph_res = _internal_get_cached_real_dataset_graph()
    baseline = _internal_get_cached_baseline()
    tga = _internal_get_cached_tga()
    return build_pyg_data(graph_res.graph, baseline, tga=tga)

def get_cached_pyg_exp5():
    res = _internal_get_cached_pyg_exp5()
    return copy.deepcopy(res)


# ---------------------------------------------------------------------------
# Detection state (Layer 1 -> PyG, Experiment 3 / 167 features)
# ---------------------------------------------------------------------------
# ``prepare_detection_data`` is the single most repeated piece of work in the
# suite: the UI resolver, the inference wrapper and several contract tests each
# call it on the same reference dataset.  It is a pure function of the file, so
# it is computed once per (real) path and deep-copied on every hand-out.

from src.detection.inference import (
    prepare_detection_data as _PRODUCTION_PREPARE_DETECTION_DATA,
)

_DETECTION_STATE_CACHE: dict[str, object] = {}


def _original_prepare_detection_data(dataset_path):
    """Call the production function, never a patched stand-in.

    The reference is captured at import time — before any fixture installs the
    caching wrapper below — so that the cache can never recurse into itself.
    """
    return _PRODUCTION_PREPARE_DETECTION_DATA(dataset_path)


def cached_prepare_detection_data(dataset_path):
    """Drop-in replacement for ``prepare_detection_data`` backed by a cache."""
    key = os.path.realpath(str(dataset_path))
    if key not in _DETECTION_STATE_CACHE:
        _DETECTION_STATE_CACHE[key] = _original_prepare_detection_data(dataset_path)
    return copy.deepcopy(_DETECTION_STATE_CACHE[key])


def get_cached_detection_state(dataset_path=None):
    """A private ``DetectionState`` for the reference dataset."""
    if dataset_path is None:
        dataset_path = resolve_reference_dataset_path()
    return cached_prepare_detection_data(dataset_path)


# ---------------------------------------------------------------------------
# Workspace detection state (external-dataset path)
# ---------------------------------------------------------------------------
# ``prepare_workspace_detection_data`` is likewise pure: it reads a workspace's
# ``canonical.parquet`` and rebuilds graph, features and tensors from it.  The
# workspace-inference tests exercise many separate workspaces that hold
# byte-identical canonical data, so the cache is keyed on the content of that
# file (plus the identity source it is read under) rather than on the directory
# path.  Two workspaces with different canonical data never share an entry.

_WORKSPACE_STATE_CACHE: dict[tuple, object] = {}

from src.datasets.inference import (
    prepare_workspace_detection_data as _PRODUCTION_PREPARE_WORKSPACE_DATA,
)


def cached_prepare_workspace_detection_data(workspace_dir, identity_source=None):
    """Drop-in replacement for ``prepare_workspace_detection_data``."""
    import hashlib

    from src.data.preprocessing import IdentitySource

    if identity_source is None:
        identity_source = IdentitySource.NAME

    parquet_path = Path(workspace_dir) / "canonical.parquet"
    if not parquet_path.exists():
        # Let the production function raise its own error unchanged.
        return _PRODUCTION_PREPARE_WORKSPACE_DATA(
            workspace_dir, identity_source=identity_source
        )

    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    key = (digest, str(identity_source))
    if key not in _WORKSPACE_STATE_CACHE:
        _WORKSPACE_STATE_CACHE[key] = _PRODUCTION_PREPARE_WORKSPACE_DATA(
            workspace_dir, identity_source=identity_source
        )
    return copy.deepcopy(_WORKSPACE_STATE_CACHE[key])
