import pytest
import os
import json
import urllib.parse
from unittest.mock import patch, MagicMock
from http.server import HTTPServer
from socketserver import TCPServer
import threading
import requests
import time
import pandas as pd
import networkx as nx

from src.ui.app import APIHandler, resolver, adapter
from src.ui.context import AnalysisContext
from src.datasets.workspace_manager import WorkspaceManager

# Dummy context for testing
@pytest.fixture
def mock_contexts():
    # Setup mock contexts
    # Reference
    df_ref = pd.DataFrame({'entity_id': ['REF1', 'REF2'], 'predicted_class': [0, 1], 'risk_score': [0.1, 0.9], 'reciprocal_out': [0, 0], 'reciprocal_in': [0, 0], 'out_repeated_ratio': [0.0, 0.0], 'in_repeated_ratio': [0.0, 0.0], 'max_parallel_edge': [1, 1], 'flow_imbalance': [0.5, 0.5], 'in_degree': [1, 1], 'out_degree': [1, 1]})
    graph_ref = nx.MultiDiGraph()
    graph_ref.add_edge('REF1', 'REF2')
    ctx_ref = AnalysisContext(workspace_id=None, df=df_ref, graph=graph_ref)

    # Workspace 1
    df_ws1 = pd.DataFrame({'entity_id': ['WS1_1', 'WS1_2'], 'predicted_class': [1, 0], 'risk_score': [0.95, 0.05], 'reciprocal_out': [0, 0], 'reciprocal_in': [0, 0], 'out_repeated_ratio': [0.0, 0.0], 'in_repeated_ratio': [0.0, 0.0], 'max_parallel_edge': [1, 1], 'flow_imbalance': [0.5, 0.5], 'in_degree': [1, 1], 'out_degree': [1, 1]})
    graph_ws1 = nx.MultiDiGraph()
    graph_ws1.add_edge('WS1_1', 'WS1_2')
    ctx_ws1 = AnalysisContext(workspace_id='ws1', df=df_ws1, graph=graph_ws1)

    return {'default': ctx_ref, 'ws1': ctx_ws1}

@pytest.fixture
def server(mock_contexts):
    # Patch resolver
    with patch('src.ui.app.resolver.get_analysis_context') as mock_resolve:
        def side_effect(ws_id):
            if not ws_id or ws_id == 'default':
                return mock_contexts['default']
            elif ws_id == 'ws1':
                return mock_contexts['ws1']
            elif ws_id == 'missing':
                from src.ui.context import WorkspaceNotFoundError
                raise WorkspaceNotFoundError("Not found")
            elif ws_id == 'incomplete':
                from src.ui.context import WorkspaceNotReadyError
                raise WorkspaceNotReadyError("Not ready")
            elif ws_id == 'invalid!':
                from src.ui.context import InvalidWorkspaceError
                raise InvalidWorkspaceError("Invalid ID")
            elif ws_id == 'fault':
                raise ValueError("Unexpected fault")
            return mock_contexts['default']
        
        mock_resolve.side_effect = side_effect

        server = TCPServer(("127.0.0.1", 0), APIHandler)
        port = server.server_address[1]

        # poll_interval bounds how long shutdown() blocks; the default of
        # 0.5 s dominated teardown for every test in this module.
        thread = threading.Thread(target=server.serve_forever, args=(0.01,))
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        yield port

        server.shutdown()
        server.server_close()

def test_api_overview_no_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_entities"] == 2
    assert data["top_entities"][0]["entity_id"] == "REF2" # Ref2 has risk 0.9

def test_api_overview_with_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview?workspace_id=ws1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["top_entities"][0]["entity_id"] == "WS1_1"

def test_api_entity_no_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/entity/REF1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity_id"] == "REF1"

def test_api_entity_with_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/entity/WS1_1?workspace_id=ws1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity_id"] == "WS1_1"

def test_api_network_no_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/network/REF1")
    assert resp.status_code == 200
    data = resp.json()
    assert any(n["id"] == "REF2" for n in data["nodes"])

def test_api_network_with_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/network/WS1_1?workspace_id=ws1")
    assert resp.status_code == 200
    data = resp.json()
    assert any(n["id"] == "WS1_2" for n in data["nodes"])

def test_api_missing_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview?workspace_id=missing")
    assert resp.status_code == 404

def test_api_incomplete_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview?workspace_id=incomplete")
    assert resp.status_code == 409

def test_api_invalid_workspace(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview?workspace_id=invalid!")
    assert resp.status_code == 400

def test_api_valid_workspace_unknown_entity(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/entity/UNKNOWN?workspace_id=ws1")
    assert resp.status_code == 404

def test_api_reference_dataset_regression(server):
    # Missing ID defaults to reference
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview")
    assert resp.status_code == 200
    assert resp.json()["top_entities"][0]["entity_id"] == "REF2"

def test_api_no_cross_workspace_graph_leakage(server):
    # Query WS1 network, should NOT see REF2
    resp = requests.get(f"http://127.0.0.1:{server}/api/network/WS1_1?workspace_id=ws1")
    data = resp.json()
    for n in data["nodes"]:
        assert n["id"] != "REF2"
        assert n["id"] != "REF1"

def test_api_no_cross_workspace_dataframe_leakage(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/entity/REF1?workspace_id=ws1")
    # REF1 does not exist in WS1
    assert resp.status_code == 404

def test_api_no_ui_ranking_leakage(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview?workspace_id=ws1")
    data = resp.json()
    for e in data["top_entities"]:
        assert e["entity_id"].startswith("WS1")

def test_api_network_fallback_defaults(server):
    # Without max_neighbors in url
    resp = requests.get(f"http://127.0.0.1:{server}/api/network/REF1")
    assert resp.status_code == 200

def test_api_exact_string_matching(server):
    # REF1 vs ref1
    resp = requests.get(f"http://127.0.0.1:{server}/api/entity/ref1")
    assert resp.status_code == 404

def test_api_url_encoding_entity_id(server):
    # Space in entity ID
    resp = requests.get(f"http://127.0.0.1:{server}/api/entity/REF%201")
    assert resp.status_code == 404 # since REF 1 doesn't exist, but it parses correctly

def test_api_unexpected_fault(server):
    resp = requests.get(f"http://127.0.0.1:{server}/api/overview?workspace_id=fault")
    assert resp.status_code == 500
