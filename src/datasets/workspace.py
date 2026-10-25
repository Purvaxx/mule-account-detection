import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional
from .manifest import WorkspaceManifest


def write_json_atomic(path: Path, data: Any) -> None:
    """Write a workspace artifact so a crash cannot leave it half-written.

    Writes to ``<path>.tmp``, flushes, ``fsync``s, then ``replace``s the
    target, which is atomic on both POSIX and Windows.  Every persisted
    workspace artifact goes through here so the durability guarantee is stated
    once rather than re-derived per call site.
    """
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


class Workspace:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self._manifest: Optional[WorkspaceManifest] = None

    @property
    def manifest(self) -> WorkspaceManifest:
        if self._manifest is None:
            self.load()
        return self._manifest

    def load(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._manifest = WorkspaceManifest.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt manifest: {e}")

        self._reconcile_state()

    def _reconcile_state(self):
        from .manifest import WorkspaceStatus
        if self._manifest.status == WorkspaceStatus.ANALYZING:
            profile_path = self.directory / "profile.json"
            if profile_path.exists():
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        prof_data = json.load(f)
                    if "columns" in prof_data and "row_estimate" in prof_data:
                        self.update(status=WorkspaceStatus.NEEDS_MAPPING, profiler_version="1.0")
                except Exception:
                    pass

    def save(self):
        if self._manifest is None:
            raise ValueError("Manifest is empty")
        self._manifest.updated_at = datetime.now(timezone.utc).isoformat()
        write_json_atomic(self.manifest_path, self._manifest.to_dict())

    def update(self, **kwargs):
        """Update manifest attributes safely."""
        m = self.manifest
        for k, v in kwargs.items():
            if hasattr(m, k) and k not in ("workspace_id", "created_at"):
                setattr(m, k, v)
        self.save()
