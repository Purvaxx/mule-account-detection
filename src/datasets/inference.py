import os
import pandas as pd
from pathlib import Path
from src.datasets.workspace_manager import WorkspaceManager
from src.data.ingestion import IngestionResult
from src.data.preprocessing import preprocess, IdentitySource
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph
from src.models.dataset import build_pyg_data
from src.detection.inference import DetectionState, predict_from_prepared_data
import json
import logging
import torch
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def prepare_workspace_detection_data(
    workspace_dir: Path,
    identity_source: IdentitySource = IdentitySource.NAME,
) -> DetectionState:
    """Rebuild detection state from a workspace's canonical dataset.

    ``identity_source`` must match the value that was used during
    canonicalization, otherwise the graph is rebuilt under different entity
    semantics than the one the compatibility report was issued for.  Callers
    holding a workspace should pass
    ``ws.manifest.dataset_config.identity_source``.
    """
    parquet_path = workspace_dir / "canonical.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError("Canonical dataset missing. Run canonicalize first.")

    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError("Canonical dataset is empty.")

    # Using dummy IngestionResult
    ing_res = IngestionResult(
        transactions=df,
        record_count=len(df),
        missing_value_report={},
        validation_warnings=[],
        validation_errors=[]
    )

    pre_res = preprocess(ing_res, identity_source=identity_source)
    if not pre_res.is_valid:
        raise ValueError("Preprocessing failed: " + str(pre_res.validation_errors))

    graph_res = build_graph(pre_res)
    if not graph_res.is_valid:
        raise ValueError("Graph construction failed: " + str(graph_res.validation_errors))

    baseline = compute_baseline(graph_res.graph, graph_res.edge_table)
    multi = compute_multigraph(graph_res.graph, graph_res.edge_table)

    pyg = build_pyg_data(graph_res.graph, baseline, multigraph=multi)

    return DetectionState(
        graph=graph_res.graph,
        pyg=pyg,
        baseline=baseline,
        multi=multi
    )

def _write_trust_assessment(ws, df_preds, state, comp_report, model_version, assessed_at):
    """Persist ``trust.json`` for a completed run.

    The assessment is a read-only interpretation of the run.  A failure to
    produce it must never fail or alter the prediction itself, so it is
    contained: predictions are already written by the time this is called.
    """
    from src.datasets.trust import assess_prediction_trust, feature_block_coverage

    try:
        features = pd.concat(
            [state.baseline.features, state.multi.features], axis=1
        )
        coverage = feature_block_coverage(features)
        assessment = assess_prediction_trust(
            predictions=df_preds,
            coverage=coverage,
            labels_present=comp_report.get("labels_present", []),
            model_version=model_version,
            assessed_at=assessed_at,
            # Facts this run already measured and persisted.  Without these the
            # only degradation the analyst ever sees is the all-or-nothing
            # case; a partly-resolved geography column and an absent optional
            # signal stay buried in compatibility.json.
            geography_normalization=comp_report.get("geography_normalization", []),
            degraded_features=comp_report.get("degraded_features", []),
        )
        with open(ws.directory / "trust.json", "w", encoding="utf-8") as f:
            json.dump(assessment.to_dict(), f, indent=2)
        return assessment
    except Exception:
        # An unwritten trust report is visible as its absence; a lost prediction
        # would not be.  Contain the failure, but never silently: a trust
        # assessment that fails on every run would otherwise look exactly like
        # a codebase that never produced one.
        logger.exception(
            "Trust assessment failed for workspace %s; predictions were written "
            "and are unaffected, but no trust.json was produced.",
            getattr(ws.manifest, "workspace_id", "<unknown>"),
        )
        return None


def _write_risk_assessment(ws, df_preds, state, checkpoint_path):
    """Persist ``assessment.json``: the run in model-independent contract terms.

    Records which model (by checkpoint digest), which feature version, which
    task and score kind, which analysis window, and which threshold produced
    ``predictions.parquet`` (``src/detection/contract.py``).  It restates the
    run; it never alters a prediction.  Contained like the trust assessment: by
    the time this runs the predictions are written, and a failure here must
    not cost them - but it is logged, never silent.
    """
    from src.detection.engines import LegacyGraphSAGEEngine

    try:
        engine = LegacyGraphSAGEEngine(checkpoint_path)
        assessment = engine.assess(df_preds, state.graph)
        with open(ws.directory / "assessment.json", "w", encoding="utf-8") as f:
            json.dump(assessment.metadata(), f, indent=2)
        return assessment
    except Exception:
        logger.exception(
            "Risk assessment metadata failed for workspace %s; predictions were "
            "written and are unaffected, but no assessment.json was produced.",
            getattr(ws.manifest, "workspace_id", "<unknown>"),
        )
        return None


def load_risk_assessment_metadata(workspace_dir: Path):
    """Read a persisted ``assessment.json``, or ``None`` if the run predates it."""
    path = Path(workspace_dir) / "assessment.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trust_assessment(workspace_dir: Path):
    """Read a persisted trust assessment, or ``None`` if the run predates one.

    Only genuine absence returns ``None``.  A file that exists but cannot be
    read is an error and is raised: the caller must be able to tell "this run
    predates trust assessment" from "this run's assessment is unreadable",
    because the second silently removes a LOW-confidence warning from the
    dashboard while the scores stay on screen.
    """
    from src.datasets.trust import PredictionTrustAssessment

    path = Path(workspace_dir) / "trust.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return PredictionTrustAssessment.from_dict(json.load(f))


def predict_workspace(workspace_id: str, manager: WorkspaceManager, checkpoint_path: str = None) -> pd.DataFrame:
    ws = manager.get_workspace(workspace_id)
    
    # 1. verify READY
    comp_path = ws.directory / "compatibility.json"
    if not comp_path.exists():
        raise ValueError("Workspace compatibility not evaluated. Run canonicalize first.")
    with open(comp_path, "r", encoding="utf-8") as f:
        comp_report = json.load(f)
        
    if comp_report.get("status") != "READY":
        raise ValueError(f"Cannot run inference on workspace with status: {comp_report.get('status')}")

    # Set status to RUNNING
    ws.update(prediction_status="RUNNING")
    
    try:
        if checkpoint_path is None:
            checkpoint_path = "artifacts/models/experiment3_graphsage.pt"
            
        # 2. prepare_workspace_detection_data
        state = prepare_workspace_detection_data(
            ws.directory,
            identity_source=ws.manifest.dataset_config.identity_source,
        )
        
        # 3. predict_from_prepared_data
        df_preds = predict_from_prepared_data(state, checkpoint_path)

        # 4. save to workspace predictions.parquet
        df_preds.to_parquet(ws.directory / "predictions.parquet")

        # 5. assess what this run does and does not establish, and persist it
        #    alongside the predictions.  This never alters a prediction: it
        #    reads the same feature matrix the model consumed and the
        #    compatibility report, and records observed facts about the run.
        predicted_at = datetime.now(timezone.utc).isoformat()
        model_version = "Experiment 3 GraphSAGE checkpoint"
        _write_trust_assessment(
            ws, df_preds, state, comp_report, model_version, predicted_at
        )
        _write_risk_assessment(ws, df_preds, state, checkpoint_path)

        # 6. persist prediction state
        ws.update(
            prediction_status="COMPLETE",
            prediction_timestamp=predicted_at,
            model_version=model_version,
        )
        return df_preds
    except Exception as e:
        ws.update(prediction_status="FAILED")
        raise e
