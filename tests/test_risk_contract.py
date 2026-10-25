"""
Tests for the model-independent risk contract and the engines behind it
(``src/detection/contract.py``, ``src/detection/engines.py``).

Grouped by the property being protected:

* contract rules       - an assessment that breaks a rule is refused, not repaired
* legacy wrapping      - the frozen checkpoint's numbers pass through unchanged
* candidate engine     - a benchmarked model scores an analysis window, and only it
* investigation links  - cycle evidence attaches to a score without moving it
* persistence          - every workspace run records its assessment metadata
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.detection.contract import (
    SCORE_CALIBRATED,
    SCORE_MODEL_PROBABILITY,
    SCORE_RANK_AGGREGATE,
    TASK_FORECAST,
    TASK_RETROSPECTIVE,
    AnalysisWindow,
    ContractViolation,
    ModelIdentity,
    RiskAssessment,
    build_assessment,
    entity_evidence,
)
from src.detection.engines import (
    LEGACY_EVIDENCE_COLUMNS,
    CandidateRiskEngine,
    LegacyGraphSAGEEngine,
    TransactionWindow,
    feature_version_for,
)

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "artifacts" / "models" / "experiment3_graphsage.pt"
REFERENCE = REPO.parent / "dkdata.json"


def _identity(**overrides) -> ModelIdentity:
    base = dict(
        model_id="unit",
        model_version="v1",
        family="logistic_regression",
        feature_version="aml_txn_v1@1",
        task=TASK_RETROSPECTIVE,
        score_kind=SCORE_MODEL_PROBABILITY,
    )
    base.update(overrides)
    return ModelIdentity(**base)


def _assessment(**overrides) -> RiskAssessment:
    kwargs = dict(model=_identity(), window=AnalysisWindow())
    kwargs.update(overrides)
    return build_assessment(["a", "b", "c"], [0.2, 0.9, 0.5], **kwargs)


# ---------------------------------------------------------------------------
# Contract rules
# ---------------------------------------------------------------------------


class TestContractRules:
    def test_rank_follows_descending_score(self):
        a = _assessment()
        assert a.ranked()["entity_id"].tolist() == ["b", "c", "a"]
        assert a.ranked()["rank"].tolist() == [1, 2, 3]

    def test_ties_are_broken_deterministically_by_entity_id(self):
        a = build_assessment(["z", "y", "x"], [0.5, 0.5, 0.5], model=_identity(), window=AnalysisWindow())
        assert a.ranked()["entity_id"].tolist() == ["x", "y", "z"]

    def test_decision_needs_a_threshold_and_its_basis(self):
        assert _assessment().scores["decision"].isna().all()
        with pytest.raises(ContractViolation, match="threshold_basis"):
            _assessment(threshold=0.5)
        a = _assessment(threshold=0.5, threshold_basis="alert budget on validation")
        assert a.scores.set_index("entity_id")["decision"].to_dict() == {"a": 0, "b": 1, "c": 1}

    def test_ground_truth_is_null_where_unknown_never_zero(self):
        a = _assessment()
        assert a.scores["ground_truth"].isna().all()
        b = _assessment(ground_truth=[1, None, 0])
        assert b.scores["ground_truth"].isna().sum() == 1

    @pytest.mark.parametrize(
        "ids, scores, match",
        [
            (["a", "a"], [0.1, 0.2], "not unique"),
            (["a", None], [0.1, 0.2], "nulls"),
            (["a", "b"], [0.1, float("nan")], "non-finite"),
            (["a", "b"], [0.1, 1.5], r"\[0, 1\]"),
        ],
    )
    def test_malformed_scores_are_refused(self, ids, scores, match):
        with pytest.raises(ContractViolation, match=match):
            build_assessment(ids, scores, model=_identity(), window=AnalysisWindow())

    def test_a_rank_aggregate_is_not_held_to_zero_one(self):
        a = build_assessment(
            ["a", "b"], [3.0, -1.0], model=_identity(score_kind=SCORE_RANK_AGGREGATE), window=AnalysisWindow()
        )
        assert a.ranked()["entity_id"].tolist() == ["a", "b"]

    def test_an_inconsistent_rank_is_refused(self):
        good = _assessment()
        bad = good.scores.copy()
        bad["rank"] = [1, 2, 3]  # a (0.2) ranked first
        with pytest.raises(ContractViolation, match="rank is not consistent"):
            RiskAssessment(model=good.model, window=good.window, scores=bad)

    def test_evidence_may_not_carry_labels_or_unscored_entities(self):
        with pytest.raises(ContractViolation, match="label or investigation"):
            _assessment(evidence=pd.DataFrame({"payer_core": [1, 0, 0]}, index=["a", "b", "c"]))
        with pytest.raises(ContractViolation, match="not scored"):
            _assessment(evidence=pd.DataFrame({"x": [1.0]}, index=["zz"]))

    def test_identity_rejects_unknown_task_and_score_kind(self):
        with pytest.raises(ContractViolation, match="task"):
            _identity(task="detection")
        with pytest.raises(ContractViolation, match="score_kind"):
            _identity(score_kind="risk")
        with pytest.raises(ContractViolation, match="model_version"):
            _identity(model_version=" ")
        assert _identity(task=TASK_FORECAST, score_kind=SCORE_CALIBRATED).task == TASK_FORECAST

    def test_window_bounds_are_ordered_and_observed_separately(self):
        with pytest.raises(ContractViolation, match="not before"):
            AnalysisWindow(start="2022-09-05", end="2022-09-05")
        w = AnalysisWindow.observed(["2022-09-02", "2022-09-04", None], start="2022-09-01", end="2022-09-05")
        assert str(w.observed_start) == "2022-09-02 00:00:00"
        assert w.to_dict()["end"] == "2022-09-05 00:00:00"

    def test_metadata_is_small_complete_and_serialisable(self):
        a = _assessment(
            threshold=0.5,
            threshold_basis="unit",
            evidence=pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=["a", "b", "c"]),
        )
        meta = json.loads(json.dumps(a.metadata()))
        assert meta["model"]["task"] == TASK_RETROSPECTIVE
        assert meta["n_entities"] == 3 and meta["n_decision_positive"] == 2
        assert meta["evidence_columns"] == ["f"]
        assert meta["contract_version"] == "1.0"
        refs = a.evidence_for("b")
        assert [(r.kind, r.ref, r.value) for r in refs] == [("feature", "f", 2.0)]
        with pytest.raises(KeyError):
            a.evidence_for("nope")


# ---------------------------------------------------------------------------
# Investigation links
# ---------------------------------------------------------------------------


class TestInvestigationLinks:
    def _graph(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        g.add_nodes_from(["A", "B", "C", "D"])
        for u, v in (("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")):
            g.add_edge(u, v, amount_usd=100.0, date=pd.Timestamp("2022-01-01"))
        return g

    def test_cycle_evidence_attaches_without_moving_the_score(self):
        from src.detection.cycles import analyze_cycles

        a = build_assessment(["A", "B", "C", "D"], [0.9, 0.8, 0.7, 0.1], model=_identity(), window=AnalysisWindow())
        before = a.scores.copy()
        analysis = analyze_cycles(self._graph())
        refs = entity_evidence(a, "A", analysis)
        kinds = [r.kind for r in refs]
        assert kinds.count("cycle") == 1
        assert {r.ref for r in refs if r.kind == "counterparty"} == {"B", "C"}
        assert [r for r in refs if r.kind == "cycle"][0].value == 3
        assert entity_evidence(a, "D", analysis) == []
        pd.testing.assert_frame_equal(a.scores, before)

    def test_unavailable_analysis_is_reported_not_read_as_no_cycles(self):
        from src.detection.cycles import CycleAnalysisUnavailable

        a = build_assessment(["A"], [0.5], model=_identity(), window=AnalysisWindow())
        refs = entity_evidence(a, "A", CycleAnalysisUnavailable(reason="ANALYSIS_FAILED", error_type="MemoryError", detail="x"))
        assert [(r.kind, r.ref) for r in refs] == [("cycle_analysis", "unavailable")]
        assert entity_evidence(a, "A", None) == []


# ---------------------------------------------------------------------------
# Candidate engine
# ---------------------------------------------------------------------------


FAMILIES = ("aml_txn_v1", "aml_temporal_v1")
BASE = pd.Timestamp("2022-03-01")


def _ledger(extra: list[tuple] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    names = [f"N{i:02d}" for i in range(30)] + ["R0", "R1", "R2"]
    rows = []
    for d in range(10):
        for _ in range(40):
            s, t = rng.choice(len(names), size=2, replace=False)
            rows.append((names[s], names[t], float(rng.integers(10, 500)), BASE + pd.Timedelta(days=d + rng.random()), 0))
        for i in range(3):
            rows.append((f"R{i}", f"R{(i + 1) % 3}", 9000.0, BASE + pd.Timedelta(days=d + 0.5), 1))
    rows.extend(extra)
    return pd.DataFrame(rows, columns=["src", "dst", "amount", "ts", "flag"])


def _dataset(frame):
    from src.experiments import DatasetSpec
    from src.experiments.sources import SourceSpec, build_transaction_dataset

    return build_transaction_dataset(
        frame,
        SourceSpec(source="src", target="dst", label="flag", amount="amount", timestamp="ts"),
        DatasetSpec(name="contract_unit", version="1", provenance="synthetic", label_definition="flagged", synthetic=True),
    )


def _fitted(family: str, dataset, window_end: pd.Timestamp):
    """A candidate fitted on the features of everything before ``window_end``."""
    from src.experiments.candidate import FitContext, build_candidate
    from src.experiments.features import FeatureRequest, build_feature_matrix
    from src.experiments.splits import SplitFold

    edges = dataset.edges_before(window_end)
    x = build_feature_matrix(
        FeatureRequest(entity_ids=list(dataset.entity_ids), edge_table=edges, graph=None, as_of=window_end),
        FAMILIES,
    ).values
    idx = np.arange(dataset.n_entities)
    params = {"n_estimators": 20, "max_depth": 2} if family == "gradient_boosting" else {}
    candidate = build_candidate(family, params)
    ctx = FitContext(
        dataset=dataset,
        features=x,
        fold=SplitFold(0, idx, idx[:0], idx[:0]),
        train_idx=idx,
        val_idx=idx[:0],
        y_train=dataset.labels,
        y_val=dataset.labels[:0],
    )
    candidate.fit(ctx)
    return candidate, ctx


def _engine(candidate, family: str) -> CandidateRiskEngine:
    return CandidateRiskEngine(
        candidate,
        FAMILIES,
        _identity(family=family, feature_version=feature_version_for(FAMILIES),
                  score_kind=SCORE_RANK_AGGREGATE if family == "structural_heuristic" else SCORE_MODEL_PROBABILITY),
    )


class TestCandidateEngine:
    @pytest.mark.parametrize("family", ["logistic_regression", "gradient_boosting", "structural_heuristic"])
    def test_score_features_equals_predict_scores(self, family):
        ds = _dataset(_ledger())
        candidate, ctx = _fitted(family, ds, BASE + pd.Timedelta(days=5))
        idx = np.arange(ds.n_entities)
        np.testing.assert_allclose(candidate.score_features(ctx.features), candidate.predict_scores(ctx, idx))

    def test_window_scores_satisfy_the_contract(self):
        ds = _dataset(_ledger())
        candidate, _ = _fitted("logistic_regression", ds, BASE + pd.Timedelta(days=5))
        window = AnalysisWindow(start=BASE + pd.Timedelta(days=5), end=BASE + pd.Timedelta(days=10))
        a = _engine(candidate, "logistic_regression").score(
            TransactionWindow(list(ds.entity_ids), ds.edge_table, window)
        )
        assert a.n_entities > 0
        assert a.model.feature_version == feature_version_for(FAMILIES)
        assert a.window.observed_start >= window.start and a.window.observed_end < window.end
        assert a.evidence_columns and all(c.split("::")[0] in FAMILIES for c in a.evidence_columns)
        # the ring launders all through the window, so it tops the ranking
        assert set(a.top(3)["entity_id"]) == {"R0", "R1", "R2"}

    def test_nothing_after_the_window_changes_a_score(self):
        end = BASE + pd.Timedelta(days=8)
        window = AnalysisWindow(start=BASE + pd.Timedelta(days=3), end=end)
        clean = _dataset(_ledger())
        later = [("N01", "N02", 99999.0, end + pd.Timedelta(hours=h), 1) for h in (0, 5, 30)]
        noisy = _dataset(_ledger(later))
        candidate, _ = _fitted("logistic_regression", clean, BASE + pd.Timedelta(days=3))
        a = _engine(candidate, "logistic_regression").score(TransactionWindow(list(clean.entity_ids), clean.edge_table, window))
        b = _engine(candidate, "logistic_regression").score(TransactionWindow(list(noisy.entity_ids), noisy.edge_table, window))
        pd.testing.assert_frame_equal(a.ranked(), b.ranked())

    def test_entities_without_window_activity_are_not_scored(self):
        early = [("OLD0", "OLD1", 10.0, BASE - pd.Timedelta(days=3), 0)]
        ds = _dataset(_ledger(early))
        candidate, _ = _fitted("logistic_regression", ds, BASE + pd.Timedelta(days=5))
        window = AnalysisWindow(start=BASE, end=BASE + pd.Timedelta(days=10))
        a = _engine(candidate, "logistic_regression").score(TransactionWindow(list(ds.entity_ids), ds.edge_table, window))
        assert "OLD0" not in set(a.scores["entity_id"])
        assert any("not scored" in n for n in a.notes)

    def test_engine_refuses_a_mismatched_feature_version_or_unfitted_model(self):
        from src.experiments.candidate import build_candidate

        ds = _dataset(_ledger())
        candidate, _ = _fitted("logistic_regression", ds, BASE + pd.Timedelta(days=5))
        with pytest.raises(ContractViolation, match="feature_version"):
            CandidateRiskEngine(candidate, FAMILIES, _identity(feature_version="aml_txn_v1@1"))
        with pytest.raises(ValueError, match="fitted"):
            CandidateRiskEngine(build_candidate("logistic_regression"), FAMILIES, _identity())

    def test_a_graph_candidate_does_not_pretend_to_score_a_matrix(self):
        from src.experiments.candidate import Candidate

        class GraphOnly(Candidate):
            def fit(self, ctx):
                return self

            def predict_scores(self, ctx, index):
                return np.zeros(len(index))

        with pytest.raises(NotImplementedError, match="does not score a feature matrix"):
            GraphOnly().score_features(pd.DataFrame({"x": [1.0]}))


# ---------------------------------------------------------------------------
# Legacy wrapping - the frozen checkpoint, unchanged
# ---------------------------------------------------------------------------


needs_reference = pytest.mark.skipif(
    not (REFERENCE.exists() and CHECKPOINT.exists()), reason="reference dataset or checkpoint absent"
)


@needs_reference
class TestLegacyWrapping:
    def test_pinned_reference_scores_pass_through_unchanged(self):
        from src.detection.inference import predict_from_prepared_data, prepare_detection_data

        state = prepare_detection_data(str(REFERENCE))
        frame = predict_from_prepared_data(state, str(CHECKPOINT))
        engine = LegacyGraphSAGEEngine(CHECKPOINT)
        a = engine.score(state)

        top = a.top(3)
        expected = [
            ("HILUX SERVICES LP", 0.56275397),
            ("LCM ALLIANCE LLP", 0.55135107),
            ("METASTAR INVEST LLP", 0.55129605),
        ]
        for (entity, score), (_, row) in zip(expected, top.iterrows()):
            assert row["entity_id"] == entity
            assert math.isclose(row["risk_score"], score, abs_tol=1e-5)
            assert row["decision"] == 1

        merged = a.to_frame().set_index("entity_id")
        shipped = frame.set_index("entity_id")
        assert len(merged) == len(shipped) == 3440
        np.testing.assert_array_equal(
            merged.loc[shipped.index, "risk_score"].to_numpy(), shipped["risk_score"].to_numpy()
        )
        assert (merged.loc[shipped.index, "decision"].astype(int) == shipped["predicted_class"]).all()
        for column in LEGACY_EVIDENCE_COLUMNS:
            np.testing.assert_array_equal(merged.loc[shipped.index, column].to_numpy(), shipped[column].to_numpy())

    def test_identity_is_the_checkpoint_and_says_what_it_can_support(self):
        from src.experiments.candidate import checkpoint_identity

        engine = LegacyGraphSAGEEngine(CHECKPOINT)
        ident = engine.identity
        assert ident.model_version.endswith(checkpoint_identity(CHECKPOINT)["sha256"][:12])
        assert ident.task == TASK_RETROSPECTIVE
        assert ident.score_kind == SCORE_MODEL_PROBABILITY
        assert "regression reference only" in ident.evaluation_regime
        assert ident.feature_version == "legacy_exp3_v1(167)"

    def test_a_frame_that_disagrees_with_the_checkpoint_is_refused(self):
        from src.detection.inference import predict_from_prepared_data, prepare_detection_data

        state = prepare_detection_data(str(REFERENCE))
        frame = predict_from_prepared_data(state, str(CHECKPOINT)).copy()
        frame.loc[frame.index[-1], "predicted_class"] = 1 - frame.loc[frame.index[-1], "predicted_class"]
        with pytest.raises(ContractViolation, match="predicted_class disagrees"):
            LegacyGraphSAGEEngine(CHECKPOINT).assess(frame, state.graph)


# ---------------------------------------------------------------------------
# Persistence on the workspace prediction route
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint absent")
class TestWorkspacePersistence:
    def test_every_run_records_its_assessment(self, tmp_path):
        from src.datasets.inference import load_risk_assessment_metadata, predict_workspace
        from src.datasets.workspace_manager import WorkspaceManager
        from tests.test_workspace_inference import create_ready_workspace

        manager = WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))
        ws = create_ready_workspace(manager, None)
        df = predict_workspace(ws.manifest.workspace_id, manager)
        meta = load_risk_assessment_metadata(ws.directory)
        assert meta is not None
        assert meta["model"]["model_id"] == "experiment3_graphsage"
        assert meta["model"]["task"] == TASK_RETROSPECTIVE
        assert meta["n_entities"] == len(df)
        assert meta["n_decision_positive"] == int(df["predicted_class"].sum())
        assert meta["evidence_columns"] == list(LEGACY_EVIDENCE_COLUMNS)
        assert meta["window"]["observed_start"] is not None

    def test_a_failed_assessment_never_costs_the_predictions(self, tmp_path, monkeypatch):
        from src.datasets.inference import load_risk_assessment_metadata, predict_workspace
        from src.datasets.workspace_manager import WorkspaceManager
        from tests.test_workspace_inference import create_ready_workspace

        def boom(self, *args, **kwargs):
            raise RuntimeError("assessment failure")

        monkeypatch.setattr(LegacyGraphSAGEEngine, "assess", boom)
        manager = WorkspaceManager(base_dir=str(tmp_path / ".workspaces"))
        ws = create_ready_workspace(manager, None)
        df = predict_workspace(ws.manifest.workspace_id, manager)
        assert not df.empty
        assert (ws.directory / "predictions.parquet").exists()
        assert load_risk_assessment_metadata(ws.directory) is None
        assert manager.get_workspace(ws.manifest.workspace_id).manifest.prediction_status == "COMPLETE"
