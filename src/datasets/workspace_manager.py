import logging
import uuid
import shutil
from pathlib import Path
from typing import List
from datetime import datetime, timezone
from .workspace import Workspace, write_json_atomic
from .manifest import WorkspaceManifest

logger = logging.getLogger(__name__)

#: The authoritative runtime location for dataset workspaces.
#:
#: This is mutable runtime state, not a committed artifact: it is listed in
#: .gitignore as "Runtime dataset workspaces".  ``artifacts/`` is reserved for
#: version-controlled artifacts such as the frozen Experiment 3 checkpoint and
#: must not be used as a workspace store.  Every caller that does not have an
#: explicit reason to override this (tests use ``tmp_path``) should leave it at
#: the default so that a single store is used process-wide.
DEFAULT_WORKSPACE_DIR: str = ".workspaces"


class WorkspaceManager:
    def __init__(self, base_dir: str = DEFAULT_WORKSPACE_DIR):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _is_safe_uuid(self, workspace_id: str) -> bool:
        try:
            uuid_obj = uuid.UUID(workspace_id)
            return str(uuid_obj) == workspace_id
        except ValueError:
            return False

    def _get_path(self, workspace_id: str) -> Path:
        if not self._is_safe_uuid(workspace_id):
            raise ValueError(f"Invalid workspace ID format: {workspace_id}")
        
        path = (self.base_dir / workspace_id).resolve()
        
        # Guard against path traversal logic (e.g. if somehow an invalid uuid passed)
        if self.base_dir not in path.parents:
            raise ValueError("Path traversal attempt detected")
        return path

    def create_workspace(self, display_name: str) -> Workspace:
        workspace_id = str(uuid.uuid4())
        path = self._get_path(workspace_id)
        path.mkdir(parents=True, exist_ok=False)

        now = datetime.now(timezone.utc).isoformat()
        manifest = WorkspaceManifest(
            workspace_id=workspace_id,
            display_name=display_name,
            created_at=now,
            updated_at=now
        )

        ws = Workspace(path)
        ws._manifest = manifest
        ws.save()
        return ws

    def get_workspace(self, workspace_id: str) -> Workspace:
        path = self._get_path(workspace_id)
        if not path.exists():
            raise FileNotFoundError(f"Workspace {workspace_id} does not exist")
        ws = Workspace(path)
        ws.load()
        return ws

    def list_workspaces(self) -> List[Workspace]:
        """Every readable workspace on disk.

        One unreadable workspace must not cost the operator the other ones.
        The previous ``except (FileNotFoundError, ValueError)`` covered a
        missing or malformed-JSON manifest but not a *well-formed* manifest the
        current schema cannot construct — a key an older or newer version
        wrote raises ``TypeError`` out of the dataclass — so a single such
        directory made the whole Dataset Hub listing fail rather than showing
        the workspaces beside it. Addressing one workspace directly still
        raises: ``get_workspace`` is unchanged, so a caller that names a broken
        workspace is told what is wrong with it.
        """
        workspaces = []
        for entry in self.base_dir.iterdir():
            if entry.is_dir() and self._is_safe_uuid(entry.name):
                try:
                    ws = Workspace(entry)
                    ws.load()
                    workspaces.append(ws)
                except Exception:
                    # Skipped, not silently: a workspace missing from the hub
                    # for this reason should be diagnosable from the log.
                    logger.warning(
                        "Skipping unreadable workspace directory %s; it will "
                        "not appear in the workspace listing.",
                        entry.name, exc_info=True,
                    )
                    continue
        return workspaces

    def update_workspace(self, workspace_id: str, **kwargs) -> Workspace:
        ws = self.get_workspace(workspace_id)
        ws.update(**kwargs)
        return ws

    def delete_workspace(self, workspace_id: str):
        path = self._get_path(workspace_id)
        if path.exists():
            shutil.rmtree(path)

    def ingest_file(self, workspace_id: str, file_path: str) -> Workspace:
        from .parsers import safe_ingest
        from .manifest import WorkspaceStatus
        import os

        ws = self.get_workspace(workspace_id)
        original_name = os.path.basename(file_path)

        dest_path, sha256, size = safe_ingest(ws.directory, file_path)

        ws.update(
            source_file=dest_path.name,
            source_hash=sha256,
            file_size=size,
            original_filename=original_name,
            status=WorkspaceStatus.UPLOADED
        )
        return ws

    def profile_workspace(self, workspace_id: str, max_rows: int = 1000):
        from .parsers import load_sample
        from .profiler import profile_dataframe
        from .manifest import WorkspaceStatus

        ws = self.get_workspace(workspace_id)
        if not ws.manifest.source_file:
            raise ValueError("No file uploaded to workspace")

        ws.update(status=WorkspaceStatus.ANALYZING)
        
        file_path = ws.directory / ws.manifest.source_file
        
        # Load sample
        df = load_sample(file_path, max_rows=max_rows)
        profile = profile_dataframe(df, exact_count=False)
        
        write_json_atomic(ws.directory / "profile.json", profile.to_dict())

        ws.update(
            profiler_version="1.0",
            status=WorkspaceStatus.NEEDS_MAPPING
        )

    def get_profile(self, workspace_id: str) -> dict:
        import json
        ws = self.get_workspace(workspace_id)
        profile_path = ws.directory / "profile.json"
        if not profile_path.exists():
            return None
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def map_workspace(self, workspace_id: str):
        from .mapper import resolve_mappings

        ws = self.get_workspace(workspace_id)
        profile_dict = self.get_profile(workspace_id)
        if not profile_dict:
            raise ValueError("Workspace has not been profiled yet.")
            
        from .profiler import DatasetProfile
        profile = DatasetProfile.from_dict(profile_dict)
        
        manifest = resolve_mappings(profile)

        write_json_atomic(ws.directory / "mapping.json", manifest.to_dict())

        # We keep the workspace state as NEEDS_MAPPING, or advance it if everything was AUTO_ACCEPTED.
        # But per requirements, we are just generating the mapping.
        ws.update(mapping_version="1.0")

    def get_mapping(self, workspace_id: str) -> dict:
        import json
        ws = self.get_workspace(workspace_id)
        mapping_path = ws.directory / "mapping.json"
        if not mapping_path.exists():
            return None
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def apply_mapping_overrides(self, workspace_id: str, updates: List[dict]) -> dict:
        """Apply operator overrides of ``selected_concept`` / ``status``.

        Owned here rather than in the HTTP route because it mutates persisted
        workspace state: the write goes through the same atomic discipline as
        every other workspace artifact, instead of a plain ``open()`` that a
        crash could leave truncated.

        Returns the rewritten mapping.  Raises ``ValueError`` if the workspace
        has no mapping to override, or if an update names a column the mapping
        does not contain -- silently ignoring an unknown column would report
        success for an override that never happened.
        """
        ws = self.get_workspace(workspace_id)
        mapping = self.get_mapping(workspace_id)
        if not mapping:
            raise ValueError(
                f"Workspace {workspace_id} has no mapping to override. "
                "Request the mapping first."
            )

        by_column = {m["source_column"]: m for m in mapping.get("mappings", [])}

        unknown = [u.get("source_column") for u in updates
                   if u.get("source_column") not in by_column]
        if unknown:
            raise ValueError(
                "Unknown source column(s): " + ", ".join(repr(u) for u in unknown)
            )

        for upd in updates:
            target = by_column[upd["source_column"]]
            if "selected_concept" in upd:
                target["selected_concept"] = upd["selected_concept"]
            if "status" in upd:
                target["status"] = upd["status"]

        write_json_atomic(ws.directory / "mapping.json", mapping)
        return mapping
