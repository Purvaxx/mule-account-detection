import pandas as pd
from typing import Optional
from src.ui.context import AnalysisContext, WorkspaceNotFoundError, WorkspaceNotReadyError, InvalidWorkspaceError
from src.datasets.workspace_manager import WorkspaceManager, DEFAULT_WORKSPACE_DIR
from src.datasets.inference import prepare_workspace_detection_data
from src.detection.inference import prepare_detection_data, predict_from_prepared_data
from src.detection.cycles import analyze_cycles_safely
from src.ui.config import UIConfig

class WorkspaceAnalysisResolver:
    def __init__(self, base_dir=None):
        # Defaults to the single authoritative workspace store
        # (WorkspaceManager.DEFAULT_WORKSPACE_DIR).  This previously pointed at
        # "artifacts/workspaces", which meant the resolver read from a
        # different directory than the one the upload path wrote to, so no
        # workspace created through the UI was ever resolvable.
        self.base_dir = base_dir or DEFAULT_WORKSPACE_DIR
        self.cache = {}
        self.reference_context = None

    def invalidate(self, workspace_id: Optional[str] = None) -> None:
        """Drop cached analysis state.

        Called after a workspace is re-predicted so the dashboard rebuilds from
        the new ``predictions.parquet`` instead of serving the previous run.
        Passing ``None`` clears every cached workspace context; the Reference
        Dataset context is never affected, since it cannot be re-predicted.
        """
        if workspace_id is None:
            self.cache.clear()
        else:
            self.cache.pop(workspace_id, None)

    def get_analysis_context(self, workspace_id: Optional[str]) -> AnalysisContext:
        """Resolve a workspace id (or the Reference Dataset) to analysis state.

        Cycle evidence is computed here, once, from the *same* graph the
        predictions were produced from — never from a second graph rebuilt for
        the purpose, and never bypassing the workspace's identity strategy,
        since the graph already encodes it.  It is attached to the context, so
        it participates in exactly the caching lifecycle the context does: the
        Reference context computes it once per resolver, a workspace context
        once per cache entry, and ``invalidate`` drops it with the rest.

        Cycle analysis runs **after** predictions are in hand and cannot fail
        them: :func:`~src.detection.cycles.analyze_cycles_safely` returns an
        explicit unavailable state rather than raising.
        """
        if not workspace_id or workspace_id == 'default':
            if not self.reference_context:
                state = prepare_detection_data(UIConfig.DATASET_PATH)
                df = predict_from_prepared_data(state, UIConfig.CHECKPOINT_PATH)
                self.reference_context = AnalysisContext(
                    workspace_id=None,
                    df=df,
                    graph=state.graph,
                    cycles=analyze_cycles_safely(state.graph),
                )
            return self.reference_context

        if workspace_id in self.cache:
            return self.cache[workspace_id]

        manager = WorkspaceManager(base_dir=self.base_dir)
        try:
            ws = manager.get_workspace(workspace_id)
        except FileNotFoundError:
            raise WorkspaceNotFoundError(f"Workspace {workspace_id} not found")
        except Exception as e:
            raise InvalidWorkspaceError(f"Invalid workspace {workspace_id}: {e}")
            
        if ws.manifest.prediction_status != "COMPLETE":
            raise WorkspaceNotReadyError(f"Workspace predictions not complete. Status: {ws.manifest.prediction_status}")

        # Load graph from canonical state.  The identity source must match the
        # one inference ran under, or the displayed graph would not correspond
        # to the persisted predictions.
        state = prepare_workspace_detection_data(
            ws.directory,
            identity_source=ws.manifest.dataset_config.identity_source,
        )

        # Load predictions from artifact
        df = pd.read_parquet(ws.directory / "predictions.parquet")

        # Cycle evidence from the workspace's own graph, which was built under
        # the manifest's identity strategy above.  Entity ids are the graph's
        # nodes, so NAME and ACCOUNT_ID workspaces yield different entities and
        # therefore different cycle identities, with no special-casing here.
        ctx = AnalysisContext(
            workspace_id=workspace_id,
            df=df,
            graph=state.graph,
            cycles=analyze_cycles_safely(state.graph),
        )
        self.cache[workspace_id] = ctx
        return ctx
