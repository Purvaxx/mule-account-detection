import enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Identity strategy
# ---------------------------------------------------------------------------
# ``IdentitySource`` is defined once, in ``src.data.preprocessing``, because
# entity resolution is a Layer-2 (data) concern and the dependency direction is
# data -> graph -> features -> models -> detection -> application
# (docs/architecture.md §35).  The workspace layer sits above ``src.data`` and
# therefore imports it rather than declaring a second, parallel enum.
#
# It is re-exported here so that ``src.datasets.manifest.IdentitySource``
# remains a valid import path for the workspace/config layer.
from src.data.preprocessing import IdentitySource

__all__ = [
    "IdentitySource",
    "CurrencyPolicy",
    "GeographyPolicy",
    "WorkspaceConfig",
    "WorkspaceStatus",
    "WorkspaceManifest",
]


class CurrencyPolicy(str, enum.Enum):
    NATIVE_USD = "NATIVE_USD"
    EXTERNAL_FX_REQUIRED = "EXTERNAL_FX_REQUIRED"
    UNKNOWN = "UNKNOWN"


class GeographyPolicy(str, enum.Enum):
    PAYER_AND_BENEFICIARY_SEPARATE = "PAYER_AND_BENEFICIARY_SEPARATE"
    SINGLE_DOMESTIC_COUNTRY = "SINGLE_DOMESTIC_COUNTRY"
    UNSPECIFIED = "UNSPECIFIED"


@dataclass
class WorkspaceConfig:
    """Explicit, operator-supplied dataset semantics.

    These are properties of the dataset that the system cannot reliably infer
    and must not silently guess.  See docs/decisions.md.
    """

    identity_source: IdentitySource = IdentitySource.NAME
    currency_policy: CurrencyPolicy = CurrencyPolicy.UNKNOWN
    geography_policy: GeographyPolicy = GeographyPolicy.UNSPECIFIED
    single_country_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        data = dict(data or {})
        if isinstance(data.get("identity_source"), str):
            data["identity_source"] = IdentitySource(data["identity_source"])
        if isinstance(data.get("currency_policy"), str):
            data["currency_policy"] = CurrencyPolicy(data["currency_policy"])
        if isinstance(data.get("geography_policy"), str):
            data["geography_policy"] = GeographyPolicy(data["geography_policy"])
        return cls(**data)


class WorkspaceStatus(str, enum.Enum):
    EMPTY = "EMPTY"
    UPLOADED = "UPLOADED"
    ANALYZING = "ANALYZING"
    NEEDS_MAPPING = "NEEDS_MAPPING"
    READY = "READY"
    PARTIAL = "PARTIAL"
    INCOMPATIBLE = "INCOMPATIBLE"
    FAILED = "FAILED"


@dataclass
class WorkspaceManifest:
    workspace_id: str
    display_name: str
    created_at: str
    updated_at: str

    # Files
    source_file: Optional[str] = None
    original_filename: Optional[str] = None
    source_hash: Optional[str] = None
    file_size: Optional[int] = None

    # Metadata
    status: WorkspaceStatus = WorkspaceStatus.EMPTY
    profiler_version: Optional[str] = None
    mapping_version: Optional[str] = None

    # Stage 5
    prediction_status: str = "NOT_RUN"
    prediction_timestamp: Optional[str] = None
    model_version: Optional[str] = None

    schema_mapping: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    # Explicit external-dataset configuration.  Manifests written before this
    # field existed simply omit the key and fall back to the defaults.
    dataset_config: WorkspaceConfig = field(default_factory=WorkspaceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceManifest":
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = WorkspaceStatus(data["status"])
        if isinstance(data.get("dataset_config"), dict):
            data["dataset_config"] = WorkspaceConfig.from_dict(data["dataset_config"])
        return cls(**data)
