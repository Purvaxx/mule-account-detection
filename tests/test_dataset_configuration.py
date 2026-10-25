"""
Explicit dataset configuration: geography resolution, policy assessment, trust
=============================================================================

Covers the behaviour added when identity / currency / geography became an
explicit, validated, operator-facing configuration:

1. ``src/datasets/geography.py`` — resolving external notations into the frozen
   Experiment 3 country vocabulary under named rules, and refusing to invent
   anything it cannot resolve.
2. ``src/datasets/policy.py`` — assessing a declared configuration against the
   dataset that was actually uploaded.
3. ``src/datasets/fx.py`` — the declared-but-unimplemented FX seam.
4. ``src/datasets/trust.py`` — what a completed prediction run does and does not
   establish.
5. The ``/api/workspaces/<id>/config`` and ``/trust`` routes, and manifest
   persistence of the choices.

The frozen contract is asserted, not assumed: the checkpoint, the 167-feature
vocabulary and the Reference Dataset path must be untouched by all of the above.
"""

import json
import threading
import time
import socketserver
import urllib.error
import urllib.request

import pandas as pd
import pytest

from src.datasets.fx import fx_availability, get_fx_provider
from src.datasets.geography import (
    FROZEN_COUNTRY_CODES,
    ResolutionRule,
    normalize_country_series,
    resolve_country,
    selectable_country_codes,
)
from src.datasets.manifest import (
    CurrencyPolicy,
    GeographyPolicy,
    IdentitySource,
    WorkspaceConfig,
)
from src.datasets.policy import (
    DimensionStatus,
    assess_configuration,
    collect_dataset_evidence,
    configuration_options,
    mapped_concepts,
)
from src.datasets.trust import (
    SignalLevel,
    TrustLevel,
    assess_prediction_trust,
    feature_block_coverage,
)
from src.datasets.workspace_manager import WorkspaceManager


# ---------------------------------------------------------------------------
# 1. Geography resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,code,rule", [
    ("GB", "GB", ResolutionRule.EXACT),
    ("33", "33", ResolutionRule.EXACT),
    (".D", ".D", ResolutionRule.EXACT),
    ("gb", "GB", ResolutionRule.CASE_NORMALIZED),
    ("  ES  ", "ES", ResolutionRule.CASE_NORMALIZED),
    ("GBR", "GB", ResolutionRule.ISO_ALPHA3),
    ("deu", "DE", ResolutionRule.ISO_ALPHA3),
    ("United Kingdom", "GB", ResolutionRule.ISO_NAME),
    ("UK", "GB", ResolutionRule.ISO_NAME),
    ("Spain", "ES", ResolutionRule.ISO_NAME),
    ("the netherlands", "NL", ResolutionRule.ISO_NAME),
])
def test_country_resolution_rules(raw, code, rule):
    """Each notation resolves under the rule it is actually resolved by."""
    res = resolve_country(raw)
    assert res.code == code
    assert res.rule == rule


@pytest.mark.parametrize("raw", [
    "Morocco",      # real country, but MA is not in the frozen vocabulary
    "United States",  # likewise US
    "USA",
    "Japan",
    "Atlantis",
    "",
    "   ",
    None,
])
def test_unresolvable_values_are_never_invented(raw):
    """A value no rule resolves stays unresolved. No bucket, no nearest guess."""
    res = resolve_country(raw)
    assert res.code is None
    assert res.rule == ResolutionRule.UNRESOLVED
    assert not res.resolved


def test_resolution_never_leaves_the_frozen_vocabulary():
    """Every code the module can emit has a column in the checkpoint."""
    frozen = set(FROZEN_COUNTRY_CODES)
    for entry in selectable_country_codes():
        assert entry["code"] in frozen
    for raw in ("United Kingdom", "GBR", "es", "Tanzania", "Türkiye"):
        res = resolve_country(raw)
        assert res.code in frozen


def test_normalize_series_reports_and_preserves():
    """Resolvable values are rewritten; unresolvable ones are left untouched."""
    series = pd.Series(["UK", "Spain", "Morocco", "UK", None, "Germany"])
    normalized, report = normalize_country_series(series, column="payer_bank_country")

    assert list(normalized.dropna()) == ["GB", "ES", "Morocco", "GB", "DE"]
    # Absence stays absence; normalization has nothing to say about a null.
    assert list(normalized.isna()) == list(series.isna())
    assert report.column == "payer_bank_country"
    assert report.total_values == 5
    assert report.resolved_values == 4
    assert report.unresolved_values == ["Morocco"]
    assert report.normalized_examples == {"UK": "GB", "Spain": "ES", "Germany": "DE"}
    assert report.rules_applied == {ResolutionRule.ISO_NAME: 3}


def test_normalization_does_not_touch_reference_dataset_codes():
    """The Reference Dataset's own codes pass through byte-identical."""
    reference_like = pd.Series(["EE", "LV", "33", ".D", "/M", "90", "G8"])
    normalized, report = normalize_country_series(reference_like)
    assert list(normalized) == list(reference_like)
    assert report.normalized_examples == {}
    assert report.unresolved_values == []


# ---------------------------------------------------------------------------
# 2. FX seam
# ---------------------------------------------------------------------------

def test_no_fx_provider_is_configured():
    """The seam exists; no implementation does, and it says so."""
    assert get_fx_provider() is None
    availability = fx_availability()
    assert availability["available"] is False
    assert availability["provider"] is None
    assert "fabricated" in availability["reason"]


# ---------------------------------------------------------------------------
# 3. Configuration assessment
# ---------------------------------------------------------------------------

def _evidence(frame, concept_sources):
    return collect_dataset_evidence(frame, concept_sources, is_exact=True)


def test_identity_requires_the_concepts_its_strategy_uses():
    """Declaring ACCOUNT_ID without account concepts is blocking, not a warning."""
    config = WorkspaceConfig(identity_source=IdentitySource.ACCOUNT_ID)
    a = assess_configuration(config, {"SENDER_NAME": ["a"], "RECEIVER_NAME": ["b"]})
    assert a.identity.status == DimensionStatus.INCOMPATIBLE
    assert a.identity.blocking
    assert a.blocking


def test_identity_flags_thin_coverage_without_blocking():
    """A mapped but mostly-empty identity column is reviewable, not fatal."""
    frame = pd.DataFrame({
        "from": ["A", None, None, None, None],
        "to": ["B", "C", "D", "E", "F"],
    })
    sources = {"SENDER_ACCOUNT": ["from"], "RECEIVER_ACCOUNT": ["to"]}
    config = WorkspaceConfig(identity_source=IdentitySource.ACCOUNT_ID)
    a = assess_configuration(config, sources, _evidence(frame, sources))
    assert a.identity.status == DimensionStatus.NEEDS_REVIEW
    assert not a.identity.blocking
    assert "dropped during preprocessing" in a.identity.reason


def test_native_usd_is_contradicted_by_the_datasets_own_currency_column():
    """A declaration the upload disproves must block, however confidently made."""
    frame = pd.DataFrame({
        "amt": [1.0, 2.0, 3.0, 4.0],
        "ccy": ["USD", "EUR", "GBP", "USD"],
    })
    sources = {"TRANSACTION_AMOUNT": ["amt"], "ORIGINAL_CURRENCY": ["ccy"]}
    config = WorkspaceConfig(currency_policy=CurrencyPolicy.NATIVE_USD)
    a = assess_configuration(config, sources, _evidence(frame, sources))

    assert a.currency.status == DimensionStatus.INCOMPATIBLE
    assert a.currency.evidence["distinct_currencies"] == 3
    assert a.currency.evidence["usd_share"] == 0.5
    assert "will not be fabricated" in a.currency.reason


def test_a_supplied_usd_column_is_not_contradicted_by_the_native_currency():
    """SAML-D's shape: its own amount_usd column, plus a native-currency column.

    ``ORIGINAL_CURRENCY`` describes ``amount_orig``, which the model does not
    consume. When ``USD_AMOUNT`` is mapped, ``amount_usd`` is projected directly
    and never copied from the native amount, so the currency column cannot
    contradict the declaration.
    """
    frame = pd.DataFrame({
        "Amount": [1459.15, 6019.64],
        "amount_usd": [1853.12, 7644.94],
        "Payment_currency": ["UK pounds", "Euro"],
    })
    sources = {
        "TRANSACTION_AMOUNT": ["Amount"],
        "USD_AMOUNT": ["amount_usd"],
        "ORIGINAL_CURRENCY": ["Payment_currency"],
    }
    config = WorkspaceConfig(currency_policy=CurrencyPolicy.NATIVE_USD)
    a = assess_configuration(config, sources, _evidence(frame, sources))

    assert a.currency.status == DimensionStatus.READY
    assert a.currency.evidence["usd_amount_supplied"] is True
    assert "supplies its own USD amount column" in a.currency.reason


def test_native_usd_is_accepted_when_the_dataset_agrees():
    frame = pd.DataFrame({"amt": [1.0, 2.0], "ccy": ["USD", "usd"]})
    sources = {"TRANSACTION_AMOUNT": ["amt"], "ORIGINAL_CURRENCY": ["ccy"]}
    config = WorkspaceConfig(currency_policy=CurrencyPolicy.NATIVE_USD)
    a = assess_configuration(config, sources, _evidence(frame, sources))
    assert a.currency.status == DimensionStatus.READY


def test_external_fx_required_is_blocked_by_the_missing_provider():
    config = WorkspaceConfig(currency_policy=CurrencyPolicy.EXTERNAL_FX_REQUIRED)
    a = assess_configuration(config, {})
    assert a.currency.status == DimensionStatus.INCOMPATIBLE
    assert a.currency.evidence["fx"]["available"] is False


def test_unknown_currency_is_reviewable_not_blocking():
    """The default for an unconfigured workspace must not demote it."""
    a = assess_configuration(WorkspaceConfig(), {})
    assert a.currency.status == DimensionStatus.NEEDS_REVIEW
    assert not a.currency.blocking


def test_unspecified_geography_is_reviewable_not_blocking():
    a = assess_configuration(WorkspaceConfig(), {})
    assert a.geography.status == DimensionStatus.NEEDS_REVIEW
    assert not a.geography.blocking
    assert "Nothing is invented" in a.geography.reason


def test_separate_geography_requires_both_sides_mapped():
    """Sender geography is never copied from the receiver, so both are required."""
    config = WorkspaceConfig(
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE
    )
    a = assess_configuration(config, {"SENDER_BANK_COUNTRY": ["src"]})
    assert a.geography.status == DimensionStatus.INCOMPATIBLE
    assert a.geography.evidence["missing_concepts"] == ["RECEIVER_BANK_COUNTRY"]
    assert "never copied onto the other" in a.geography.reason


def test_separate_geography_reports_partial_resolvability():
    frame = pd.DataFrame({
        "sbc": ["United Kingdom", "Morocco"],
        "rbc": ["Spain", "Spain"],
    })
    sources = {"SENDER_BANK_COUNTRY": ["sbc"], "RECEIVER_BANK_COUNTRY": ["rbc"]}
    config = WorkspaceConfig(
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE
    )
    a = assess_configuration(config, sources, _evidence(frame, sources))
    assert a.geography.status == DimensionStatus.NEEDS_REVIEW
    assert a.geography.evidence["unresolved_examples"] == ["Morocco"]
    assert 0.0 < a.geography.evidence["resolvable_share"] < 1.0


def test_separate_geography_is_ready_when_everything_resolves():
    frame = pd.DataFrame({"sbc": ["United Kingdom"], "rbc": ["ESP"]})
    sources = {"SENDER_BANK_COUNTRY": ["sbc"], "RECEIVER_BANK_COUNTRY": ["rbc"]}
    config = WorkspaceConfig(
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE
    )
    a = assess_configuration(config, sources, _evidence(frame, sources))
    assert a.geography.status == DimensionStatus.READY


def test_single_domestic_country_must_resolve_into_the_vocabulary():
    unresolvable = WorkspaceConfig(
        geography_policy=GeographyPolicy.SINGLE_DOMESTIC_COUNTRY,
        single_country_code="Atlantis",
    )
    a = assess_configuration(unresolvable, {})
    assert a.geography.status == DimensionStatus.NEEDS_REVIEW
    assert "sets no feature column" in a.geography.reason

    resolvable = WorkspaceConfig(
        geography_policy=GeographyPolicy.SINGLE_DOMESTIC_COUNTRY,
        single_country_code="Latvia",
    )
    b = assess_configuration(resolvable, {})
    assert b.geography.status == DimensionStatus.READY
    assert b.geography.evidence["resolution"]["code"] == "LV"


def test_assessment_reports_when_it_had_no_evidence():
    """An assessment made without the file says so instead of implying certainty."""
    a = assess_configuration(WorkspaceConfig(), {})
    assert a.evidence_available is False
    assert a.sampled_rows == 0


def test_configuration_options_are_derived_not_hardcoded():
    """The UI's vocabularies come from the frozen contract."""
    options = configuration_options()
    assert {o["value"] for o in options["identity_source"]} == {
        m.value for m in IdentitySource
    }
    assert {o["value"] for o in options["currency_policy"]} == {
        m.value for m in CurrencyPolicy
    }
    assert {o["value"] for o in options["geography_policy"]} == {
        m.value for m in GeographyPolicy
    }
    assert options["country_codes"]
    assert all(c["code"] in set(FROZEN_COUNTRY_CODES) for c in options["country_codes"])
    assert options["fx"]["available"] is False


def test_mapped_concepts_mirrors_the_canonicalizer_admission_rule():
    """Only projected mappings count, so assessment and canonicalization agree."""
    mapping = {"mappings": [
        {"source_column": "a", "selected_concept": "TIMESTAMP", "status": "AUTO_ACCEPTED"},
        {"source_column": "b", "selected_concept": "USD_AMOUNT", "status": "NEEDS_CONFIRMATION"},
        {"source_column": "c", "selected_concept": "SENDER_NAME", "status": "CONFLICT"},
        {"source_column": "d", "selected_concept": None, "status": "UNMAPPED"},
    ]}
    assert mapped_concepts(mapping) == {"TIMESTAMP": ["a"], "USD_AMOUNT": ["b"]}


# ---------------------------------------------------------------------------
# 4. Prediction trust
# ---------------------------------------------------------------------------

def _features(country_active: bool, type_known: bool, n=4):
    data = {
        "log_in_degree": [1.0] * n,
        "entity_type_Company": [1.0 if type_known else 0.0] * n,
        "entity_type_unknown": [0.0 if type_known else 1.0] * n,
        "bank_country_GB": [1.0 if country_active else 0.0] * n,
        "bank_country_ES": [0.0] * n,
    }
    return pd.DataFrame(data)


def test_feature_block_coverage_detects_an_empty_geography_block():
    cov = feature_block_coverage(_features(country_active=False, type_known=True))
    assert "geography" in cov["zero_blocks"]
    assert cov["blocks"]["geography"]["exercised"] == 0
    assert cov["feature_dimension"] == 5


def test_feature_block_coverage_detects_all_unknown_entity_types():
    """A one-hot that is entirely 'unknown' carries no signal, though non-zero."""
    cov = feature_block_coverage(_features(country_active=True, type_known=False))
    assert "entity_type" in cov["zero_blocks"]
    assert cov["blocks"]["entity_type"]["all_unknown"] is True
    assert "geography" not in cov["zero_blocks"]


def test_feature_block_coverage_clean_run_has_no_zero_blocks():
    cov = feature_block_coverage(_features(country_active=True, type_known=True))
    assert cov["zero_blocks"] == []


def _preds(flagged, total):
    return pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(total)],
        "risk_score": [0.9] * flagged + [0.1] * (total - flagged),
        "predicted_class": [1] * flagged + [0] * (total - flagged),
    })


def test_a_majority_flag_rate_drops_confidence_to_low():
    """The SAML-D symptom: 70% flagged is reported as unusable, not as fraud."""
    t = assess_prediction_trust(_preds(70, 100), labels_present=["investigation"])
    assert t.level == TrustLevel.LOW
    assert t.flag_rate == 0.7
    flag_signal = next(s for s in t.signals if s.key == "flag_rate")
    assert flag_signal.level == SignalLevel.CRITICAL
    assert "not that the dataset is mostly fraudulent" in flag_signal.why_it_matters
    assert "should not be read as an alert list" in t.headline


def test_an_empty_feature_block_limits_confidence():
    cov = feature_block_coverage(_features(country_active=False, type_known=True))
    t = assess_prediction_trust(_preds(2, 100), cov, labels_present=["investigation"])
    assert t.level == TrustLevel.LIMITED
    assert t.zero_blocks == ["geography"]
    assert any(s.key == "zero_block_geography" for s in t.signals)


def test_missing_ground_truth_is_reported_as_unmeasurable():
    t = assess_prediction_trust(_preds(2, 100), labels_present=[])
    assert t.labels_available is False
    labels = next(s for s in t.signals if s.key == "labels")
    assert labels.level == SignalLevel.CAUTION
    assert "unknown, not" in labels.why_it_matters


def test_a_clean_run_is_unverified_never_high():
    """Nothing available at inference time can establish that the model transfers."""
    cov = feature_block_coverage(_features(country_active=True, type_known=True))
    t = assess_prediction_trust(_preds(2, 100), cov, labels_present=["investigation"])
    assert t.level == TrustLevel.UNVERIFIED
    assert not hasattr(TrustLevel, "HIGH")
    assert "not evidence of fraud" in t.headline


def test_the_distribution_caveat_is_always_present():
    for labels in ([], ["investigation"]):
        t = assess_prediction_trust(_preds(1, 100), labels_present=labels)
        assert any(s.key == "distribution_untested" for s in t.signals)


def test_every_signal_states_its_rule():
    """§46: an explanation without its rule is not an explanation."""
    t = assess_prediction_trust(_preds(60, 100), labels_present=[])
    for signal in t.signals:
        assert signal.rule
        assert signal.observed
        assert signal.why_it_matters


def _geo_report(column, total, resolved, unresolved=()):
    return {
        "column": column,
        "total_values": total,
        "resolved_values": resolved,
        "resolved_ratio": (resolved / total) if total else 0.0,
        "distinct_raw": 10,
        "normalized_examples": {},
        "unresolved_values": list(unresolved),
        "rules_applied": {},
    }


def test_largely_unresolved_geography_limits_confidence():
    """A column that resolves in part exercises the geography block, so the
    all-zero check cannot see it -- but half its rows still carry no country."""
    cov = feature_block_coverage(_features(country_active=True, type_known=True))
    t = assess_prediction_trust(
        _preds(2, 100), cov, labels_present=["investigation"],
        geography_normalization=[
            _geo_report("payer_bank_country", 1000, 550, ["Morocco", "Japan"])
        ],
    )
    signal = next(s for s in t.signals
                  if s.key == "geography_unresolved_payer_bank_country")
    assert signal.level == SignalLevel.CAUTION
    assert "450 of 1,000" in signal.observed
    assert "Morocco" in signal.observed
    assert t.level == TrustLevel.LIMITED


def test_a_small_unresolved_residue_is_stated_without_downgrading():
    """SAML-D's real shape: 96.9% resolved still leaves thousands of rows with
    no country. A hard cutoff would hide that entirely for being a small share."""
    cov = feature_block_coverage(_features(country_active=True, type_known=True))
    t = assess_prediction_trust(
        _preds(2, 100), cov, labels_present=["investigation"],
        geography_normalization=[
            _geo_report("beneficiary_bank_country", 100000, 96923, ["Japan"])
        ],
    )
    signal = next(s for s in t.signals
                  if s.key == "geography_unresolved_beneficiary_bank_country")
    assert signal.level == SignalLevel.INFO
    assert "3,077 of 100,000" in signal.observed
    # Stated, but not a reason to downgrade the run.
    assert t.level == TrustLevel.UNVERIFIED


def test_fully_resolved_geography_raises_no_signal():
    cov = feature_block_coverage(_features(country_active=True, type_known=True))
    t = assess_prediction_trust(
        _preds(2, 100), cov, labels_present=["investigation"],
        geography_normalization=[_geo_report("payer_bank_country", 1000, 1000)],
    )
    assert not any(s.key.startswith("geography_unresolved") for s in t.signals)
    assert t.level == TrustLevel.UNVERIFIED


def test_absent_optional_signals_are_reported_beside_the_scores():
    cov = feature_block_coverage(_features(country_active=True, type_known=True))
    t = assess_prediction_trust(
        _preds(2, 100), cov, labels_present=["investigation"],
        degraded_features=["payer_type", "beneficiary_type"],
    )
    signal = next(s for s in t.signals if s.key == "degraded_features")
    assert signal.level == SignalLevel.CAUTION
    assert "payer_type" in signal.observed
    assert "exactly the trained width" in signal.why_it_matters
    assert t.level == TrustLevel.LIMITED


def test_new_signals_never_alter_a_prediction():
    """Widening the assessment must not touch what the model produced."""
    preds = _preds(2, 100)
    before = preds.copy(deep=True)
    assess_prediction_trust(
        preds,
        geography_normalization=[_geo_report("payer_bank_country", 10, 1)],
        degraded_features=["payer_type"],
    )
    pd.testing.assert_frame_equal(preds, before)


def test_trust_assessment_survives_a_json_round_trip():
    t = assess_prediction_trust(_preds(60, 100), labels_present=[])
    from src.datasets.trust import PredictionTrustAssessment
    restored = PredictionTrustAssessment.from_dict(json.loads(json.dumps(t.to_dict())))
    assert restored.level == t.level
    assert [s.key for s in restored.signals] == [s.key for s in t.signals]


# ---------------------------------------------------------------------------
# 5. Workspace integration: persistence, canonicalization, API
# ---------------------------------------------------------------------------

def _frame(n=40):
    return pd.DataFrame({
        "txn_time": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "from_acct": [f"A{i % 6}" for i in range(n)],
        "to_acct": [f"A{(i + 1) % 6}" for i in range(n)],
        "usd_amount": [100.0 + i for i in range(n)],
        "sender_bank_location": ["United Kingdom", "Spain", "Morocco", "Germany"] * (n // 4),
        "receiver_bank_location": ["Netherlands", "Italy", "Nigeria", "France"] * (n // 4),
    })


_WANTED = {
    "txn_time": "TIMESTAMP",
    "from_acct": "SENDER_ACCOUNT",
    "to_acct": "RECEIVER_ACCOUNT",
    "usd_amount": "USD_AMOUNT",
    "sender_bank_location": "SENDER_BANK_COUNTRY",
    "receiver_bank_location": "RECEIVER_BANK_COUNTRY",
}


@pytest.fixture
def geo_workspace(tmp_path):
    """A profiled, mapped, account-keyed workspace whose geography is names."""
    manager = WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))
    csv = tmp_path / "geo.csv"
    _frame().to_csv(csv, index=False)

    ws = manager.create_workspace("geo")
    wid = ws.manifest.workspace_id
    manager.ingest_file(wid, str(csv))
    manager.profile_workspace(wid)
    manager.map_workspace(wid)

    ws = manager.get_workspace(wid)
    path = ws.directory / "mapping.json"
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    for entry in mapping["mappings"]:
        if entry["source_column"] in _WANTED:
            entry["selected_concept"] = _WANTED[entry["source_column"]]
            entry["status"] = "AUTO_ACCEPTED"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    return manager, wid


def _configure(manager, wid, **kwargs):
    ws = manager.get_workspace(wid)
    for key, value in kwargs.items():
        setattr(ws.manifest.dataset_config, key, value)
    ws.update()


def test_configuration_persists_in_the_manifest_across_reload(geo_workspace, tmp_path):
    manager, wid = geo_workspace
    _configure(
        manager, wid,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.SINGLE_DOMESTIC_COUNTRY,
        single_country_code="LV",
    )

    # A completely fresh manager, reading manifest.json off disk.
    reopened = WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))
    config = reopened.get_workspace(wid).manifest.dataset_config
    assert config.identity_source is IdentitySource.ACCOUNT_ID
    assert config.currency_policy is CurrencyPolicy.NATIVE_USD
    assert config.geography_policy is GeographyPolicy.SINGLE_DOMESTIC_COUNTRY
    assert config.single_country_code == "LV"

    raw = json.loads((reopened.get_workspace(wid).manifest_path).read_text(encoding="utf-8"))
    assert raw["dataset_config"]["single_country_code"] == "LV"


def test_canonicalization_resolves_geography_and_reports_what_it_did(geo_workspace):
    from src.datasets.canonicalizer import canonicalize_and_validate

    manager, wid = geo_workspace
    _configure(
        manager, wid,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE,
    )
    report = canonicalize_and_validate(wid, manager)

    assert report.status == "READY"
    assert report.feature_dimension == 167
    assert report.feature_order_valid is True

    columns = {g["column"] for g in report.geography_normalization}
    assert columns == {"payer_bank_country", "beneficiary_bank_country"}

    payer = next(g for g in report.geography_normalization
                 if g["column"] == "payer_bank_country")
    assert payer["normalized_examples"]["United Kingdom"] == "GB"
    assert payer["unresolved_values"] == ["Morocco"]

    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet"
    )
    stored = set(canonical["payer_bank_country"].dropna())
    assert {"GB", "ES", "DE"} <= stored
    # Unresolvable values are preserved verbatim, not blanked or bucketed.
    assert "Morocco" in stored


def test_unspecified_geography_leaves_values_untouched(geo_workspace):
    """Normalization only runs where a geography policy was actually declared."""
    from src.datasets.canonicalizer import canonicalize_and_validate

    manager, wid = geo_workspace
    _configure(
        manager, wid,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.UNSPECIFIED,
    )
    report = canonicalize_and_validate(wid, manager)

    assert report.geography_normalization == []
    canonical = pd.read_parquet(
        manager.get_workspace(wid).directory / "canonical.parquet"
    )
    assert "United Kingdom" in set(canonical["payer_bank_country"].dropna())


def test_compatibility_report_carries_the_configuration_assessment(geo_workspace):
    from src.datasets.canonicalizer import canonicalize_and_validate

    manager, wid = geo_workspace
    _configure(
        manager, wid,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE,
    )
    report = canonicalize_and_validate(wid, manager)

    assessment = report.configuration_assessment
    assert set(assessment) >= {"status", "identity", "currency", "geography"}
    assert assessment["identity"]["status"] == report.identity_status
    assert assessment["currency"]["status"] == report.currency_status
    assert assessment["geography"]["status"] == report.geography_status
    assert assessment["evidence_available"] is True

    # Persisted, so the hub and the config panel can read the same verdict.
    persisted = json.loads(
        (manager.get_workspace(wid).directory / "compatibility.json")
        .read_text(encoding="utf-8")
    )
    assert persisted["configuration_assessment"]["status"] == assessment["status"]


def test_a_blocking_configuration_prevents_ready(geo_workspace):
    """Declaring separate geography without a receiver country cannot reach READY."""
    from src.datasets.canonicalizer import canonicalize_and_validate

    manager, wid = geo_workspace
    ws = manager.get_workspace(wid)
    path = ws.directory / "mapping.json"
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    for entry in mapping["mappings"]:
        if entry["source_column"] == "receiver_bank_location":
            entry["selected_concept"] = None
            entry["status"] = "UNMAPPED"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    _configure(
        manager, wid,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE,
    )
    report = canonicalize_and_validate(wid, manager)
    assert report.status == "INCOMPATIBLE"
    assert report.geography_status == "INCOMPATIBLE"


def test_prediction_writes_a_trust_assessment(geo_workspace):
    from src.datasets.canonicalizer import canonicalize_and_validate
    from src.datasets.inference import load_trust_assessment, predict_workspace

    manager, wid = geo_workspace
    _configure(
        manager, wid,
        identity_source=IdentitySource.ACCOUNT_ID,
        currency_policy=CurrencyPolicy.NATIVE_USD,
        geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE,
    )
    assert canonicalize_and_validate(wid, manager).status == "READY"

    preds = predict_workspace(wid, manager)
    ws = manager.get_workspace(wid)

    trust = load_trust_assessment(ws.directory)
    assert trust is not None
    assert trust.entity_count == len(preds)
    assert trust.feature_dimension == 167
    assert trust.labels_available is False
    assert trust.level in {TrustLevel.LOW, TrustLevel.LIMITED, TrustLevel.UNVERIFIED}

    # Trust is an interpretation, never a modification: predictions carry the
    # frozen model's own output and no trust column.
    assert "risk_score" in preds.columns
    assert not any(c.startswith("trust") for c in preds.columns)

    # Persisted, so it survives a restart exactly as predictions do.
    assert load_trust_assessment(ws.directory).level == trust.level


# ---------------------------------------------------------------------------
# 6. Configuration and trust API
# ---------------------------------------------------------------------------

@pytest.fixture
def api(geo_workspace, monkeypatch):
    """The real HTTP handler, bound to an isolated workspace store."""
    import src.ui.app as ui_app
    import src.ui.workspaces as ui_workspaces
    from src.ui.app import APIHandler

    manager, wid = geo_workspace
    monkeypatch.setattr(ui_app, "WorkspaceManager", lambda *a, **k: manager)
    monkeypatch.setattr(ui_workspaces, "WorkspaceManager", lambda *a, **k: manager)

    server = socketserver.TCPServer(("127.0.0.1", 0), APIHandler)
    port = server.server_address[1]
    # poll_interval bounds how long shutdown() blocks (default 0.5 s).
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        yield f"http://127.0.0.1:{port}", manager, wid
    finally:
        server.shutdown()
        server.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, json.loads(r.read())


def _post(base, path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


def test_config_get_returns_state_options_and_verdict(api):
    base, _manager, wid = api
    status, view = _get(base, f"/api/workspaces/{wid}/config")

    assert status == 200
    assert view["config"]["identity_source"] == IdentitySource.NAME.value
    assert set(view["options"]) >= {
        "identity_source", "currency_policy", "geography_policy", "country_codes", "fx"
    }
    # The verdict rests on the real upload, bounded like profiling.
    assert view["assessment"]["evidence_available"] is True
    assert view["assessment"]["sampled_rows"] > 0
    assert "SENDER_ACCOUNT" in view["mapped_concepts"]


def test_config_post_persists_and_returns_the_same_view_model(api):
    base, manager, wid = api
    status, view = _post(base, f"/api/workspaces/{wid}/config", {
        "identity_source": "ACCOUNT_ID",
        "currency_policy": "NATIVE_USD",
        "geography_policy": "PAYER_AND_BENEFICIARY_SEPARATE",
    })

    assert status == 200
    assert view["status"] == "success"
    assert view["assessment"]["identity"]["status"] == DimensionStatus.READY
    assert view["assessment"]["currency"]["status"] == DimensionStatus.READY

    config = manager.get_workspace(wid).manifest.dataset_config
    assert config.identity_source is IdentitySource.ACCOUNT_ID
    assert config.geography_policy is GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE

    # GET and POST must agree, or the UI would render two different truths.
    _status, fetched = _get(base, f"/api/workspaces/{wid}/config")
    assert fetched["assessment"] == view["assessment"]


def test_config_post_rejects_an_unknown_policy_value(api):
    base, manager, wid = api
    before = manager.get_workspace(wid).manifest.dataset_config.identity_source

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base, f"/api/workspaces/{wid}/config", {"identity_source": "EMAIL"})
    assert exc.value.code == 400

    # Rejected wholesale: nothing was half-applied.
    after = manager.get_workspace(wid).manifest.dataset_config.identity_source
    assert after is before


def test_config_post_rejects_a_partial_payload_atomically(api):
    """A valid field alongside an invalid one must not be persisted."""
    base, manager, wid = api
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base, f"/api/workspaces/{wid}/config", {
            "identity_source": "ACCOUNT_ID",
            "currency_policy": "BITCOIN",
        })
    assert exc.value.code == 400

    config = manager.get_workspace(wid).manifest.dataset_config
    assert config.identity_source is IdentitySource.NAME
    assert config.currency_policy is CurrencyPolicy.UNKNOWN


def test_trust_endpoint_is_404_before_a_run_and_present_after(api):
    from src.datasets.canonicalizer import canonicalize_and_validate
    from src.datasets.inference import predict_workspace

    base, manager, wid = api

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base, f"/api/workspaces/{wid}/trust")
    assert exc.value.code == 404

    _post(base, f"/api/workspaces/{wid}/config", {
        "identity_source": "ACCOUNT_ID",
        "currency_policy": "NATIVE_USD",
        "geography_policy": "PAYER_AND_BENEFICIARY_SEPARATE",
    })
    assert canonicalize_and_validate(wid, manager).status == "READY"
    predict_workspace(wid, manager)

    status, trust = _get(base, f"/api/workspaces/{wid}/trust")
    assert status == 200
    assert trust["level"] in {"LOW", "LIMITED", "UNVERIFIED"}
    assert trust["signals"]


def test_workspace_summary_exposes_configuration_and_confidence(api):
    from src.datasets.canonicalizer import canonicalize_and_validate
    from src.datasets.inference import predict_workspace

    base, manager, wid = api
    _post(base, f"/api/workspaces/{wid}/config", {
        "identity_source": "ACCOUNT_ID",
        "currency_policy": "NATIVE_USD",
        "geography_policy": "PAYER_AND_BENEFICIARY_SEPARATE",
    })
    canonicalize_and_validate(wid, manager)
    predict_workspace(wid, manager)

    _status, summary = _get(base, "/api/workspaces")
    row = next(r for r in summary if r["id"] == wid)
    assert row["identity_source"] == "ACCOUNT_ID"
    assert row["currency_policy"] == "NATIVE_USD"
    assert row["geography_policy"] == "PAYER_AND_BENEFICIARY_SEPARATE"
    assert row["config_assessment"] in {"READY", "NEEDS_REVIEW"}
    assert row["trust_level"] in {"LOW", "LIMITED", "UNVERIFIED"}
    assert 0.0 <= row["trust_flag_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Regressions found in final quality review
# ---------------------------------------------------------------------------

def test_an_unmeasurable_run_is_never_reported_as_unverified():
    """No predicted_class means nothing was observed — not that nothing is wrong.

    UNVERIFIED is the most reassuring level this module can emit. It must be
    reachable only by a run whose alert rate was actually measured, otherwise a
    schema drift in predictions.parquet silently argues in favour of the scores.
    """
    from src.datasets.trust import assess_prediction_trust, TrustLevel, SignalLevel

    preds = pd.DataFrame({"entity_id": ["a", "b"], "risk_score": [0.1, 0.9]})
    assessment = assess_prediction_trust(predictions=preds)

    assert assessment.level == TrustLevel.LOW
    flag_signal = next(s for s in assessment.signals if s.key == "flag_rate")
    assert flag_signal.level == SignalLevel.CRITICAL
    assert flag_signal.value == "not measurable"
    # The headline must not quote an alert rate that was never measured.
    assert "0%" not in assessment.headline
    assert "never measured" in assessment.headline


def test_a_measurable_clean_run_is_still_reachable():
    """The guard above must not make UNVERIFIED unreachable for a real run."""
    from src.datasets.trust import assess_prediction_trust, TrustLevel

    preds = pd.DataFrame({
        "entity_id": [str(i) for i in range(100)],
        "predicted_class": [1] * 5 + [0] * 95,
    })
    assessment = assess_prediction_trust(predictions=preds, labels_present=["investigation"])
    assert assessment.level == TrustLevel.UNVERIFIED
    assert assessment.flag_rate == 0.05


def test_an_unreadable_trust_artifact_is_raised_not_reported_as_absent(tmp_path):
    """A corrupt trust.json must not look like 'this run predates trust'.

    Both currently hide the confidence banner, so if they are indistinguishable
    a LOW-confidence run loses its warning while its scores stay on screen.
    """
    from src.datasets.inference import load_trust_assessment

    assert load_trust_assessment(tmp_path) is None          # genuinely absent

    (tmp_path / "trust.json").write_text('{"level": "LOW", ', encoding="utf-8")
    with pytest.raises(ValueError):
        load_trust_assessment(tmp_path)


def test_geography_share_is_measured_against_the_whole_sample():
    """The evidence's value counts are bounded; the verdict must say so.

    With more distinct values than ``_EVIDENCE_VALUE_LIMIT``, every inspected
    value can resolve while much of the column was never looked at. Claiming
    'every sampled value resolves' there asserts more than was observed.
    """
    from src.datasets.policy import (
        assess_configuration,
        collect_dataset_evidence,
        DimensionStatus,
        _EVIDENCE_VALUE_LIMIT,
    )
    from src.datasets.manifest import WorkspaceConfig, GeographyPolicy
    from src.datasets.geography import selectable_country_codes

    resolvable = [c["code"] for c in selectable_country_codes()]
    assert len(resolvable) > _EVIDENCE_VALUE_LIMIT, "fixture needs a longer tail"

    # Every value resolves, but there are more distinct values than the
    # evidence collector retains counts for.
    frame = pd.DataFrame({
        "s_country": resolvable,
        "r_country": resolvable,
    })
    sources = {
        "SENDER_BANK_COUNTRY": ["s_country"],
        "RECEIVER_BANK_COUNTRY": ["r_country"],
    }
    evidence = collect_dataset_evidence(frame, sources)
    config = WorkspaceConfig(geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)
    geo = assess_configuration(config, sources, evidence).geography

    assert geo.status == DimensionStatus.NEEDS_REVIEW
    assert geo.evidence["fully_inspected"] is False
    assert "every sampled value" not in geo.reason
    assert "inspected" in geo.reason
    # The share is measured against the sampled rows, not against the subset
    # that happened to be inspected.
    assert geo.evidence["resolvable_share"] < 1.0


def test_geography_still_reports_ready_when_it_saw_everything():
    """A fully-inspected, fully-resolvable column must remain READY."""
    from src.datasets.policy import (
        assess_configuration,
        collect_dataset_evidence,
        DimensionStatus,
    )
    from src.datasets.manifest import WorkspaceConfig, GeographyPolicy

    frame = pd.DataFrame({"s": ["GB", "ES", "DE"] * 10, "r": ["FR", "IT", "NL"] * 10})
    sources = {"SENDER_BANK_COUNTRY": ["s"], "RECEIVER_BANK_COUNTRY": ["r"]}
    evidence = collect_dataset_evidence(frame, sources)
    config = WorkspaceConfig(geography_policy=GeographyPolicy.PAYER_AND_BENEFICIARY_SEPARATE)
    geo = assess_configuration(config, sources, evidence).geography

    assert geo.status == DimensionStatus.READY
    assert geo.evidence["fully_inspected"] is True
    assert geo.evidence["resolvable_share"] == 1.0


def test_revalidating_drops_the_cached_analysis_context(api):
    """Re-canonicalization rewrites canonical.parquet; the cache must not survive.

    The resolver returns a cached context before it checks anything else, so a
    stale entry is served indefinitely — including for a workspace that
    re-validation has since turned INCOMPATIBLE.
    """
    import src.ui.app as ui_app

    base, _manager, wid = api
    ui_app.resolver.cache[wid] = "stale-context"

    _post(base, f"/api/workspaces/{wid}/validate", {})
    assert wid not in ui_app.resolver.cache


def test_saving_configuration_drops_the_cached_analysis_context(api):
    """identity_source decides how the graph is rebuilt, so the cache is stale."""
    import src.ui.app as ui_app

    base, _manager, wid = api
    ui_app.resolver.cache[wid] = "stale-context"

    _post(base, f"/api/workspaces/{wid}/config", {"identity_source": "ACCOUNT_ID"})
    assert wid not in ui_app.resolver.cache
