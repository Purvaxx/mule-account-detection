import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Set, Tuple

from src.datasets.mapper import CanonicalConcept, MappingManifest, MappingStatus
from src.datasets.parsers import load_full
from src.datasets.workspace_manager import WorkspaceManager
from src.data.ingestion import (
    ALWAYS_PRESENT_COLUMNS,
    REQUIRED_COLUMNS,
    IngestionResult,
    LABEL_COLUMNS,
    INFERENCE_DEGRADABLE_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    inference_required_columns,
    IDENTITY_REQUIRED_COLUMNS,
)

from src.data.preprocessing import preprocess
from src.datasets.derivation import (
    DerivationContext,
    instant_pair,
    plan_derivations,
)
from src.datasets.geography import normalize_country_series
from src.datasets.policy import (
    DimensionStatus,
    assess_configuration,
    collect_dataset_evidence,
    unresolved_concept_claims,
)
from src.graph.builder import build_graph
from src.features.baseline import compute_baseline
from src.features.multigraph import compute_multigraph
from src.features.temporal import compute_temporal
from src.features.tda import compute_tga

#: Union of every identity column across all identity strategies.
_ALL_IDENTITY_COLUMNS = frozenset().union(*IDENTITY_REQUIRED_COLUMNS.values())

# ---------------------------------------------------------------------------
# Readiness: the three outcomes an operator actually needs to distinguish
# ---------------------------------------------------------------------------
#: Every critical inference input is available and no optional signal is absent.
READINESS_FULL = "FULL"
#: Inference is mathematically valid, but the model is seeing less than it could:
#: an optional signal block is empty, a declared dimension needs review, or part
#: of the geography does not reach the frozen vocabulary.
READINESS_DEGRADED = "DEGRADED"
#: Something genuinely required is unavailable. Inference does not run.
READINESS_BLOCKED = "BLOCKED"



@dataclass
class DatasetCompatibilityReport:
    status: str
    identity_status: str
    currency_status: str
    geography_status: str
    canonical_fields: List[str]
    missing_required_fields: List[str]
    invalid_mappings: List[str]
    constructible_features: int
    unavailable_features: List[str]
    feature_dimension: int
    feature_order_valid: bool
    graph_requirements: List[str]
    datatype_errors: List[str]
    value_quality_warnings: List[str]
    distribution_warnings: List[str]
    model_version: str
    feature_schema_version: str
    canonical_schema_version: str
    validator_version: str
    field_statuses: List[Dict[str, Any]] = field(default_factory=list)
    #: Ground-truth label columns the dataset actually supplied.  Empty for an
    #: unlabelled dataset, which is a fully supported case.
    labels_present: List[str] = field(default_factory=list)
    #: Optional signal columns the dataset omitted (feature block degraded).
    degraded_features: List[str] = field(default_factory=list)
    #: Per-dimension verdict on the operator's declared configuration, checked
    #: against the actual upload.  See src/datasets/policy.py.
    configuration_assessment: Dict[str, Any] = field(default_factory=dict)
    #: What country resolution actually did to each geography column, if any.
    geography_normalization: List[Dict[str, Any]] = field(default_factory=list)
    #: FULL / DEGRADED / BLOCKED — what the model will actually see, as opposed
    #: to whether it may run at all (``status``).  See :func:`assess_readiness`.
    readiness: str = READINESS_BLOCKED
    #: One stated fact per line explaining the readiness classification.
    readiness_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def assess_readiness(
    report: "DatasetCompatibilityReport",
    assessment_dimensions: List[Any],
    country_block_empty: bool,
) -> Tuple[str, List[str]]:
    """Classify a validated dataset as ``FULL`` / ``DEGRADED`` / ``BLOCKED``.

    ``report.status`` answers *may this run*; readiness answers *what will the
    model actually see if it does*.  The two are deliberately separate:
    ``READY`` on a dataset with no entity types and no geography is a true
    statement about the contract and a misleading one about the analysis, so the
    report carries both and the UI shows both.

    This function **classifies; it does not re-derive**.  It is given the
    finished report precisely so that every reason it returns is a fact the
    report already states, and so the two can never drift apart.
    """
    reasons: List[str] = []

    if report.status != "READY":
        for name in report.missing_required_fields:
            reasons.append(f"Required inference input '{name}' is not available.")
        reasons.extend(report.graph_requirements)
        reasons.extend(report.datatype_errors)
        if not report.feature_order_valid and not report.graph_requirements:
            reasons.append(
                "The generated feature matrix does not match the checkpoint's "
                "167-feature contract."
            )
        for dim in assessment_dimensions:
            if dim.status == DimensionStatus.INCOMPATIBLE:
                reasons.append(f"{dim.dimension.capitalize()}: {dim.reason}")
        if not reasons:
            reasons.append(
                "The dataset did not pass compatibility validation; see the "
                "quality warnings for the specific finding."
            )
        return READINESS_BLOCKED, reasons

    if report.degraded_features:
        reasons.append(
            "Optional signal columns are absent ("
            + ", ".join(report.degraded_features) + "); "
            "their feature block is zero/unknown for every entity."
        )
    if country_block_empty:
        reasons.append(
            "No value reaches the frozen country vocabulary, so all 151 "
            "bank_country features are zero."
        )
    for geography in report.geography_normalization:
        if geography.get("unresolved_values"):
            reasons.append(
                f"{geography['column']}: {geography['resolved_ratio']:.0%} of "
                f"values resolve into the frozen country vocabulary; the rest "
                f"set no country feature."
            )
    for dim in assessment_dimensions:
        if dim.status == DimensionStatus.NEEDS_REVIEW:
            reasons.append(f"{dim.dimension.capitalize()}: {dim.reason}")

    if reasons:
        return READINESS_DEGRADED, reasons
    return READINESS_FULL, [
        "Every critical inference input is available and every optional signal "
        "block is populated."
    ]


def load_expected_features() -> List[str]:
    ckpt_path = Path("artifacts/models/experiment3_graphsage.pt")
    if not ckpt_path.exists():
        return []
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    return ckpt.get("metadata", {}).get("feature_names", [])

def canonicalize_and_validate(workspace_id: str, manager: WorkspaceManager) -> DatasetCompatibilityReport:
    ws = manager.get_workspace(workspace_id)
    manifest = ws.manifest

    mapping_path = ws.directory / "mapping.json"
    if not mapping_path.exists():
        raise ValueError("Workspace lacks mapping.json")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    mapping_manifest = MappingManifest.from_dict(mapping_data)

    concept_to_sources = {}
    invalid_mappings = []

    for m in mapping_manifest.mappings:
        if m.selected_concept and m.status in (MappingStatus.AUTO_ACCEPTED, MappingStatus.NEEDS_CONFIRMATION):
            if m.selected_concept not in concept_to_sources:
                concept_to_sources[m.selected_concept] = []
            concept_to_sources[m.selected_concept].append(m.source_column)
        elif m.status == MappingStatus.CONFLICT:
            invalid_mappings.append(f"Conflict on {m.source_column}")

    dataset_config = manifest.dataset_config

    # src.datasets.parsers owns how a workspace file becomes a DataFrame --
    # the extension dispatch, the case handling and the error wrapping all live
    # there, beside the bounded read profiling uses.
    raw_path = ws.directory / manifest.source_file
    df = load_full(raw_path)

    # ------------------------------------------------------------------
    # Configuration assessment
    # ------------------------------------------------------------------
    # Whether the operator's declared identity / currency / geography can be
    # honoured is decided in one place (src/datasets/policy.py) against the
    # dataset that was actually uploaded, not against the declaration alone.
    # The config API route and the configuration UI consult the same function,
    # so the three surfaces cannot disagree.
    config_evidence = collect_dataset_evidence(df, concept_to_sources, is_exact=True)
    # A concept that was claimed but left unresolved is not the same as one the
    # dataset never mentions; the currency check depends on the difference.
    unresolved_claims = unresolved_concept_claims(mapping_data)
    assessment = assess_configuration(
        dataset_config, concept_to_sources, config_evidence, unresolved_claims
    )

    identity_status = assessment.identity.status
    currency_status = assessment.currency.status
    geography_status = assessment.geography.status

    canonical_df = pd.DataFrame()
    canonical_fields = []

    concept_to_col = {
        CanonicalConcept.TRANSACTION_ID.value: "id",
        CanonicalConcept.TIMESTAMP.value: "date",
        CanonicalConcept.SENDER_NAME.value: "payer_name",
        CanonicalConcept.RECEIVER_NAME.value: "beneficiary_name",
        CanonicalConcept.SENDER_ACCOUNT.value: "payer_account",
        CanonicalConcept.RECEIVER_ACCOUNT.value: "beneficiary_account",
        CanonicalConcept.TRANSACTION_AMOUNT.value: "amount_orig",
        CanonicalConcept.ORIGINAL_CURRENCY.value: "amount_orig_currency",
        CanonicalConcept.USD_AMOUNT.value: "amount_usd",
        CanonicalConcept.EUR_AMOUNT.value: "amount_eur",
        CanonicalConcept.SENDER_TYPE.value: "payer_type",
        CanonicalConcept.RECEIVER_TYPE.value: "beneficiary_type",
        CanonicalConcept.SENDER_JURISDICTION.value: "payer_jurisdiction",
        CanonicalConcept.RECEIVER_JURISDICTION.value: "beneficiary_jurisdiction",
        CanonicalConcept.SENDER_BANK_COUNTRY.value: "payer_bank_country",
        CanonicalConcept.RECEIVER_BANK_COUNTRY.value: "beneficiary_bank_country",
        CanonicalConcept.PURPOSE.value: "purpose",
        CanonicalConcept.SOURCE_FILE.value: "source_file",
        CanonicalConcept.INVESTIGATION.value: "investigation",
        CanonicalConcept.SENDER_CORE.value: "payer_core",
        CanonicalConcept.RECEIVER_CORE.value: "beneficiary_core"
    }

    for concept, src_cols in concept_to_sources.items():
        if concept in concept_to_col:
            col_name = concept_to_col[concept]
            if len(src_cols) == 1:
                canonical_df[col_name] = df[src_cols[0]]
            elif instant_pair(src_cols) is not None:
                # A calendar column and a clock column are two halves of one
                # instant, not two spellings of it.  Pasting them together as
                # text produces "2022-10-07 00:00:00 10:35:19", which parses as
                # nothing.  src/datasets/derivation.py composes them properly.
                continue
            else:
                series = df[src_cols[0]].astype(str)
                for c in src_cols[1:]:
                    series = series + " " + df[c].astype(str)
                canonical_df[col_name] = series
            if col_name not in canonical_fields:
                canonical_fields.append(col_name)


    field_statuses = []
    mapped_canonical = {}
    confidence_map = {}
    conflict_map = {}

    for m in mapping_manifest.mappings:
        if m.status == MappingStatus.CONFLICT:
            for c in m.candidates:
                c_col = concept_to_col.get(c.canonical_concept)
                if c_col and c.score >= 0.3:
                    if c_col not in conflict_map:
                        conflict_map[c_col] = []
                    if m.source_column not in conflict_map[c_col]:
                        conflict_map[c_col].append(m.source_column)

        if m.selected_concept and m.status in (MappingStatus.AUTO_ACCEPTED, MappingStatus.NEEDS_CONFIRMATION):
            c_col = concept_to_col.get(m.selected_concept)
            if c_col:
                if c_col not in mapped_canonical:
                    mapped_canonical[c_col] = []
                mapped_canonical[c_col].append(m.source_column)
                if c_col not in confidence_map:
                    confidence_map[c_col] = "HIGH"
                if m.status == MappingStatus.NEEDS_CONFIRMATION:
                    confidence_map[c_col] = "MEDIUM"

    # ------------------------------------------------------------------
    # Deterministic derivation
    # ------------------------------------------------------------------
    # Which canonical columns can be *computed* from what the mapping already
    # established, and under which stated rule, is owned by
    # src/datasets/derivation.py.  This layer applies the plan and reports it;
    # it does not decide what is derivable.
    derivations = plan_derivations(DerivationContext(
        frame=canonical_df,
        raw=df,
        mapped_sources=mapped_canonical,
        config=dataset_config,
        upload_filename=ws.manifest.original_filename,
        currency_supported=(
            assessment.currency.status != DimensionStatus.INCOMPATIBLE),
    ))
    derived_by_column: Dict[str, Any] = {}
    for derivation in derivations:
        canonical_df[derivation.canonical_name] = derivation.values
        if derivation.canonical_name not in canonical_fields:
            canonical_fields.append(derivation.canonical_name)
        derived_by_column[derivation.canonical_name] = derivation

    for col in REQUIRED_COLUMNS:
        if col in derived_by_column:
            # A computed field's rule describes it better than the mapping does,
            # even when source columns were also mapped to it: `date` composed
            # from a calendar and a clock column is a composition, not a
            # concatenation of two rival claimants.
            field_statuses.append(derived_by_column[col].to_field_status())
        elif col in mapped_canonical:
            src_cols = mapped_canonical[col]
            if len(src_cols) > 1:
                state = "DERIVABLE"
                rule = f"Concatenated {len(src_cols)} source columns"
            else:
                state = "AVAILABLE" if confidence_map[col] == "HIGH" else "NEEDS_REVIEW"
                rule = None
            field_statuses.append({
                "canonical_name": col,
                "state": state,
                "source_columns": src_cols,
                "confidence": confidence_map[col],
                "rule": rule
            })
        elif col in conflict_map:
            field_statuses.append({
                "canonical_name": col,
                "state": "NEEDS_REVIEW",
                "source_columns": conflict_map[col],
                "confidence": "LOW",
                "rule": "Multiple conflicting source columns"
            })
        elif col in LABEL_COLUMNS:
            field_statuses.append({
                "canonical_name": col,
                "state": "NOT_SUPPLIED",
                "source_columns": [],
                "confidence": "N/A",
                "rule": "Ground-truth label; not required for inference"
            })
        elif col in INFERENCE_DEGRADABLE_COLUMNS:
            field_statuses.append({
                "canonical_name": col,
                "state": "NOT_SUPPLIED",
                "source_columns": [],
                "confidence": "N/A",
                "rule": "Optional signal; feature block degrades to zero/unknown"
            })
        elif col in OPTIONAL_METADATA_COLUMNS:
            field_statuses.append({
                "canonical_name": col,
                "state": "NOT_SUPPLIED",
                "source_columns": [],
                "confidence": "N/A",
                "rule": "Optional metadata; not consumed by the 167 features"
            })
        elif col not in inference_required_columns(dataset_config.identity_source):
            # e.g. payer_name_norm when the workspace resolves on ACCOUNT_ID.
            field_statuses.append({
                "canonical_name": col,
                "state": "NOT_SUPPLIED",
                "source_columns": [],
                "confidence": "N/A",
                "rule": "Not required under the selected identity strategy"
            })
        else:
            field_statuses.append({
                "canonical_name": col,
                "state": "MISSING",
                "source_columns": [],
                "confidence": "N/A",
                "rule": "No supported source field found"
            })


    # ------------------------------------------------------------------
    # Geography normalization
    # ------------------------------------------------------------------
    # The frozen model matches bank country by exact string against a 151-entry
    # vocabulary, so a dataset whose values are country NAMES ("United Kingdom")
    # or alpha-3 codes ("GBR") contributes nothing.  Resolve those into the
    # vocabulary under stated, semantically safe rules (src/datasets/geography.py)
    # and record exactly what was changed.
    #
    # Only applied when the operator declared a geography policy: under
    # UNSPECIFIED there is no geography claim to normalize.  Values no rule
    # resolves are left untouched, never bucketed and never invented, so their
    # existing all-zero behaviour is unchanged.  The frozen vocabulary,
    # src/features/baseline.py and the checkpoint are not involved.
    geography_reports: List[Dict[str, Any]] = []
    if dataset_config.geography_policy.value != "UNSPECIFIED":
        for geo_col in ("payer_bank_country", "beneficiary_bank_country"):
            if geo_col in canonical_df.columns:
                normalized, geo_report = normalize_country_series(
                    canonical_df[geo_col], column=geo_col
                )
                canonical_df[geo_col] = normalized
                geography_reports.append(geo_report.to_dict())

    # ------------------------------------------------------------------
    # Inference contract (not the training contract)
    # ------------------------------------------------------------------
    # READY is gated on what inference genuinely needs, NOT on the Reference
    # Dataset's labelled-training schema.  Ground-truth labels
    # (payer_core / beneficiary_core / investigation) and analyst metadata are
    # preserved when supplied but never required: the frozen model consumes
    # neither.  See src/data/ingestion.py for the full partition.
    required_for_inference = inference_required_columns(dataset_config.identity_source)
    missing_required = sorted(c for c in required_for_inference if c not in canonical_fields)

    # Signal-bearing but structurally optional fields degrade to a zeroed
    # country block / entity_type_unknown rather than failing.  Report the
    # degradation honestly instead of silently scoring on empty blocks.
    degraded = sorted(c for c in INFERENCE_DEGRADABLE_COLUMNS if c not in canonical_fields)

    # Which ground-truth labels the dataset actually supplied.  Never
    # fabricated, never required.
    labels_present = sorted(c for c in LABEL_COLUMNS if c in canonical_fields)

    # ------------------------------------------------------------------
    # Materialise genuinely-absent non-inference columns as explicit NA
    # ------------------------------------------------------------------
    # Layer 2 (preprocess) still validates against the full canonical schema.
    # For an unlabelled dataset those columns do not exist, so we create them
    # holding NA -- the honest representation of "this dataset has no such
    # data".  No value is invented: nothing is defaulted to 0, "US", False or
    # any other substantive value.
    for col in REQUIRED_COLUMNS:
        if col not in canonical_df.columns:
            canonical_df[col] = pd.NA

    graph_reqs = []
    dq_warnings = []
    datatype_errors = []

    # Coerce dtypes for every column that exists, but only treat an
    # all-NA result as a datatype ERROR when the dataset actually supplied the
    # column.  A column materialised as NA above was never supplied, so "it is
    # all NA" is the honest recorded state, not a parse failure.
    def _coerce(col: str, kind: str) -> None:
        if col not in canonical_df.columns:
            return
        if kind == "datetime":
            canonical_df[col] = pd.to_datetime(canonical_df[col], errors="coerce")
            what = "datetime"
        else:
            canonical_df[col] = pd.to_numeric(canonical_df[col], errors="coerce")
            what = "numeric"
        if col in canonical_fields and canonical_df[col].isna().all() and not df.empty:
            datatype_errors.append(f"{col} could not be parsed as {what}")

    _coerce("date", "datetime")
    _coerce("amount_usd", "numeric")
    _coerce("amount_orig", "numeric")

    # The ingestion layer requires original df, we just pass the mapped df.
    ing_res = IngestionResult(
        transactions=canonical_df,
        record_count=len(canonical_df),
        missing_value_report={},
        validation_warnings=[],
        validation_errors=[]
    )

    # Run through the actual existing pipelines
    pre_res = None
    try:
        pre_res = preprocess(ing_res, identity_source=dataset_config.identity_source)
        if not pre_res.is_valid:
            graph_reqs.extend(pre_res.validation_errors)
    except Exception as e:
        graph_reqs.append(f"Preprocessing failed: {str(e)}")

    graph_res = None
    if pre_res and pre_res.is_valid:
        try:
            graph_res = build_graph(pre_res)
            if not graph_res.is_valid:
                graph_reqs.extend(graph_res.validation_errors)
        except Exception as e:
            graph_reqs.append(f"Graph construction failed: {str(e)}")

    unavailable_features = []
    constructible = 0
    feature_dimension = 0
    feature_order_valid = False
    #: Observed, not assumed: every bank_country_* feature is zero across the
    #: whole entity population.  Read by assess_readiness below.
    country_block_empty = False

    expected_features = load_expected_features()
    if not expected_features:
        unavailable_features.append("Cannot load checkpoint feature schema")
        expected_features = []

    if graph_res and graph_res.is_valid:
        if graph_res.edge_table.empty:
            dq_warnings.append("Dataset produces 0 valid edges (all sender/receiver pairs have nulls)")
        else:
            try:
                b = compute_baseline(graph_res.graph, graph_res.edge_table)
                m = compute_multigraph(graph_res.graph, graph_res.edge_table)
                # Exp3 checkpoint only uses baseline and multigraph

                generated_features = list(b.feature_names) + list(m.feature_names)
                feature_dimension = len(generated_features)

                # A dataset can MAP bank country and still contribute no
                # geographic signal, if none of its codes are in the frozen
                # Experiment 3 vocabulary (e.g. country names like "UK"/"Spain"
                # rather than ISO codes "GB"/"ES").  That is a supported,
                # documented degradation -- but it must not be silent.
                country_cols = [c for c in b.features.columns
                                if c.startswith("bank_country_")]
                mapped_country = any(
                    c in canonical_fields
                    for c in ("payer_bank_country", "beneficiary_bank_country")
                )
                country_block_empty = bool(
                    country_cols and not b.features[country_cols].values.any()
                )
                if mapped_country and country_block_empty:
                    dq_warnings.append(
                        "Bank country was mapped, but none of its values could be "
                        "resolved into the frozen Experiment 3 country vocabulary; "
                        "all 151 bank_country_* features are zero. Geographic "
                        "signal is not reaching the model."
                    )

                if generated_features == expected_features:
                    feature_order_valid = True
                    constructible = feature_dimension
                else:
                    feature_order_valid = False
                    missing_feats = set(expected_features) - set(generated_features)
                    extra_feats = set(generated_features) - set(expected_features)
                    unavailable_features.extend(list(missing_feats))
                    if extra_feats:
                        unavailable_features.append(f"Produces {len(extra_feats)} extra unexpected features")

                # Check for NaNs and Infs in the matrices (dataframes)
                all_feats = [b.features, m.features]
                for f_df in all_feats:
                    # some feature modules might return df with NaN if it's not handled
                    if f_df.isna().any().any():
                        dq_warnings.append("Features contain NaN values")
                    if np.isinf(f_df.values).any():
                        dq_warnings.append("Features contain infinite values")

            except Exception as e:
                graph_reqs.append(f"Feature computation failed: {str(e)}")

    if canonical_df.empty:
        dq_warnings.append("Dataset is empty")

    # Tag every field with which half of the contract it belongs to, so the UI
    # can distinguish "missing and blocking" from "absent but irrelevant".
    for fs in field_statuses:
        name = fs["canonical_name"]
        if name in required_for_inference:
            fs["requirement"] = "INFERENCE_REQUIRED"
        elif name in LABEL_COLUMNS:
            fs["requirement"] = "LABEL_ONLY"
        elif name in INFERENCE_DEGRADABLE_COLUMNS:
            fs["requirement"] = "OPTIONAL_SIGNAL"
        elif name in _ALL_IDENTITY_COLUMNS:
            fs["requirement"] = "IDENTITY_ALTERNATE"
        else:
            fs["requirement"] = "OPTIONAL_METADATA"

    if degraded:
        dq_warnings.append(
            "Dataset supplies no " + ", ".join(degraded) + ". "
            "The corresponding feature block is zero/unknown for every entity; "
            "the model sees reduced signal."
        )
    if not labels_present:
        dq_warnings.append(
            "Dataset supplies no ground-truth labels. Model predictions cannot "
            "be scored against truth for this workspace."
        )

    # Surface every configuration dimension that is not cleanly satisfied, in
    # the operator's own terms, rather than only the derived status code.
    for dim in assessment.dimensions:
        if dim.status != DimensionStatus.READY:
            dq_warnings.append(
                f"{dim.dimension.capitalize()} ({dim.selection}): {dim.reason}"
            )

    for geo_report in geography_reports:
        if geo_report["normalized_examples"]:
            examples = ", ".join(
                f"{raw!r}->{code}"
                for raw, code in list(geo_report["normalized_examples"].items())[:5]
            )
            dq_warnings.append(
                f"{geo_report['column']}: resolved "
                f"{geo_report['resolved_ratio']:.0%} of values into the frozen "
                f"country vocabulary (e.g. {examples})."
            )
        recognized = geo_report.get("recognized_out_of_vocabulary") or {}
        if recognized:
            named = ", ".join(
                f"{raw!r} ({code})" for raw, code in list(recognized.items())[:5]
            )
            dq_warnings.append(
                f"{geo_report['column']}: {len(recognized)} value(s) name a real "
                f"country the Experiment 3 checkpoint has no feature column for "
                f"({named}). They are left exactly as supplied and set no "
                f"country feature. This is a limit of the frozen vocabulary, "
                f"not unreadable data."
            )

    # A configuration the system cannot honour at all must block inference:
    # CurrencyPolicy.EXTERNAL_FX_REQUIRED has no FX provider, a NATIVE_USD
    # declaration the dataset's own currency column contradicts, and an identity
    # or geography source whose required concepts were never mapped all produce
    # confident but meaningless results.  policy.assess_configuration decides
    # which of those applies; this layer only enforces the verdict.
    #
    # NEEDS_REVIEW is deliberately NOT a blocker.  CurrencyPolicy.UNKNOWN and
    # GeographyPolicy.UNSPECIFIED are the defaults for any workspace whose
    # configuration was never set, so treating NEEDS_REVIEW as a downgrade
    # would demote every existing workspace and break the already-promised
    # READY path.  It is reported for the operator, not enforced.
    config_blocking = assessment.blocking

    if dq_warnings and any("empty" in w or "0 valid edges" in w for w in dq_warnings):
        status = "FAILED"
    elif (
        missing_required
        or graph_reqs
        or not feature_order_valid
        or datatype_errors
        or config_blocking
    ):
        status = "INCOMPATIBLE"
    else:
        status = "READY"

    report = DatasetCompatibilityReport(
        status=status,
        identity_status=identity_status,
        currency_status=currency_status,
        geography_status=geography_status,
        canonical_fields=canonical_fields,
        missing_required_fields=missing_required,
        invalid_mappings=invalid_mappings,
        constructible_features=constructible,
        unavailable_features=unavailable_features,
        feature_dimension=feature_dimension,
        feature_order_valid=feature_order_valid,
        graph_requirements=graph_reqs,
        datatype_errors=datatype_errors,
        value_quality_warnings=dq_warnings,
        distribution_warnings=["Dataset characteristics have not been statistically tested against training baseline."] if status == "READY" else [],
        model_version="exp3_checkpoint",
        feature_schema_version="1.0",
        canonical_schema_version="1.0",
        validator_version="1.0",
        field_statuses=field_statuses,
        labels_present=labels_present,
        degraded_features=degraded,
        configuration_assessment=assessment.to_dict(),
        geography_normalization=geography_reports,
    )

    # Whether inference MAY run (`status`) and what the model will actually SEE
    # if it does (`readiness`) are separate questions.  A dataset can satisfy
    # the 167-feature contract exactly while supplying no entity types and no
    # resolvable geography, and reporting only READY would hide that.
    # Classified from the finished report so the two cannot state different
    # facts about the same run.
    report.readiness, report.readiness_reasons = assess_readiness(
        report, assessment.dimensions, country_block_empty
    )

    with open(ws.directory / "compatibility.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    if status in ("READY", "PARTIAL", "INCOMPATIBLE"):
        try:
            canonical_df.to_parquet(ws.directory / "canonical.parquet")
        except ImportError:
            pass

    from src.datasets.manifest import WorkspaceStatus
    try:
        new_status = WorkspaceStatus(status)
        ws.update(status=new_status)
    except ValueError:
        pass

    return report
