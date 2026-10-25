import json
import logging
from src.datasets.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)

def get_workspaces_summary():
    manager = WorkspaceManager()
    workspaces = manager.list_workspaces()
    result = []
    for ws in workspaces:
        comp_path = ws.directory / "compatibility.json"
        comp_status = "UNKNOWN"
        config_assessment = None
        readiness = None
        readiness_reasons = []
        if comp_path.exists():
            # Guarded like trust.json below: the hub is a summary of every
            # workspace, and one unreadable compatibility report must not cost
            # the operator the listing of all the others.  The workspace's own
            # validate route is the authoritative surface and still raises.
            try:
                with open(comp_path, "r", encoding="utf-8") as f:
                    comp = json.load(f)
                comp_status = comp.get("status", "UNKNOWN")
                config_assessment = (comp.get("configuration_assessment") or {}).get("status")
                # Readiness answers a different question from status: status says
                # whether inference *may* run, readiness says what the model will
                # actually see if it does.  The hub shows both so a READY workspace
                # with an empty signal block is not silently indistinguishable
                # from a complete one.
                readiness = comp.get("readiness")
                readiness_reasons = comp.get("readiness_reasons") or []
            except Exception:
                logger.warning(
                    "Unreadable compatibility.json for workspace %s; hub will "
                    "show its compatibility as UNKNOWN.",
                    ws.manifest.workspace_id, exc_info=True,
                )

        # A completed run's trust assessment, so the hub can say what the
        # predictions rest on without the operator opening the dashboard.
        trust_path = ws.directory / "trust.json"
        trust_level = None
        trust_headline = None
        trust_flag_rate = None
        if trust_path.exists():
            try:
                with open(trust_path, "r", encoding="utf-8") as f:
                    trust = json.load(f)
                trust_level = trust.get("level")
                trust_headline = trust.get("headline")
                trust_flag_rate = trust.get("flag_rate")
            except Exception:
                # The hub is a summary; the dashboard banner is the
                # authoritative surface and reports this loudly. Still log it,
                # so an unreadable artifact is not indistinguishable from a
                # workspace that simply has no assessment.
                logger.warning(
                    "Unreadable trust.json for workspace %s; hub will show no "
                    "confidence for it.", ws.manifest.workspace_id, exc_info=True
                )


        prof_path = ws.directory / "profile.json"
        record_count = 0
        column_count = 0
        # Profiling reads a bounded sample (1000 rows by default) rather than
        # the whole upload, so ``row_estimate`` is the size of *what was
        # profiled*.  The hub used to print it under a bare "Records" heading,
        # which reported a 100,000-row workbook as 1,000 records and then
        # disagreed with the dashboard that had scored all of it.  Carry the
        # exactness through so the card can say which of the two it is.
        records_are_exact = True
        if prof_path.exists():
            try:
                with open(prof_path, "r") as f:
                    prof_data = json.load(f)
                    record_count = prof_data.get("row_estimate", 0)
                    column_count = len(prof_data.get("columns", []))
                    records_are_exact = bool(prof_data.get("is_exact_count", False))
            except Exception:
                pass
                
        result.append({
            "id": ws.manifest.workspace_id,
            "name": ws.manifest.display_name,
            "filename": ws.manifest.original_filename or "-",
            "created_at": ws.manifest.created_at,
            "status": ws.manifest.prediction_status or "NOT_RUN",
            "manifest_status": ws.manifest.status.value,
            "compatibility": comp_status,
            "records": record_count,
            "records_are_exact": records_are_exact,
            "columns": column_count,
            # Prediction artifacts are persisted, so this reflects whether a
            # previous run's results are still on disk after a restart.
            "prediction_status": ws.manifest.prediction_status or "NOT_RUN",
            "prediction_timestamp": ws.manifest.prediction_timestamp,
            "model_version": ws.manifest.model_version,
            "has_predictions": (ws.directory / "predictions.parquet").exists(),
            "can_predict": comp_status == "READY",
            # Explicit dataset semantics, so the hub can show what a workspace
            # was configured as without opening it.
            "identity_source": ws.manifest.dataset_config.identity_source.value,
            "currency_policy": ws.manifest.dataset_config.currency_policy.value,
            "geography_policy": ws.manifest.dataset_config.geography_policy.value,
            "config_assessment": config_assessment,
            "readiness": readiness,
            "readiness_reasons": readiness_reasons,
            "trust_level": trust_level,
            "trust_headline": trust_headline,
            "trust_flag_rate": trust_flag_rate,
        })
    # sort by created descending
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result
