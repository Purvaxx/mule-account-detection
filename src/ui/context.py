import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import Optional

from src.detection.cycles import CycleEvidenceState

class WorkspaceNotFoundError(Exception): pass
class WorkspaceNotReadyError(Exception): pass
class InvalidWorkspaceError(Exception): pass

@dataclass
class AnalysisContext:
    """The single carrier of resolved analysis state for one dataset.

    ``df`` and ``graph`` are the frozen-model side: predictions produced by the
    Experiment 3 checkpoint, and the authoritative ``MultiDiGraph`` they were
    produced from.  ``cycles`` is the Layer 10 side — post-hoc investigator
    evidence read off that *same* graph, downstream of and independent from
    inference.  Nothing in ``cycles`` participates in ``risk_score``,
    ``predicted_class``, the 167-feature representation, compatibility status,
    or the trust assessment.

    Attributes
    ----------
    workspace_id:
        ``None`` for the Reference Dataset; the workspace UUID otherwise.
    df:
        Per-entity predictions plus observed structural features.
    graph:
        The authoritative transaction graph the predictions were computed from.
    cycles:
        Cycle evidence for ``graph``, in one of three distinguishable states:

        * ``None``                        — never attempted.  This is the
          default so a lightweight or stub context stays valid without
          computing anything.
        * ``CycleAnalysis``               — attempted and completed.  An empty
          ``cycles`` tuple means the graph genuinely has none.
        * ``CycleAnalysisUnavailable``    — attempted and failed, carrying the
          reason.  Never collapses into "no cycles found".
    """

    workspace_id: Optional[str]
    df: pd.DataFrame
    graph: nx.MultiDiGraph
    cycles: Optional[CycleEvidenceState] = None
