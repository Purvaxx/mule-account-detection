"""
Tests for the Layer 6 experiment framework (``src.experiments``).

Grouped by the property being protected:

* metric honesty      - undefined metrics stay ``None``
* imbalance tolerance - arbitrary positive counts, including zero
* leakage control     - partition disjointness, label columns, threshold source
* split semantics     - each strategy's declared regime is the regime it gives
* contract stability  - a candidate can be swapped without touching the runner
* provenance          - the artifact round-trips and records what was run

The fixtures are small synthetic graphs.  Anything they demonstrate is a
statement about the framework, never about detection performance on real data.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.experiments import (
    CalibrationSpec,
    DatasetSpec,
    EvaluationSpec,
    ExperimentDataset,
    ExperimentSpec,
    FeatureSpec,
    ModelSpec,
    SplitSpec,
    ThresholdSpec,
    available_candidates,
    available_feature_sets,
    available_split_strategies,
    build_candidate,
    build_split_plan,
    load_result,
    run_experiment,
    save_result,
)
from src.experiments.candidate import Candidate, FitContext, register_candidate
from src.experiments.features import FeatureRequest, build_feature_matrix
from src.experiments.leakage import LeakageError, audit_features, audit_split_plan
from src.experiments.metrics import (
    aggregate_fold_metrics,
    binary_metrics,
    calibration_metrics,
    ranking_metrics,
    threshold_metrics,
)
from src.experiments.thresholds import select_threshold

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(
    n_entities: int = 120,
    n_positive: int = 8,
    n_random_edges: int = 600,
    seed: int = 3,
    synthetic: bool = True,
) -> ExperimentDataset:
    """A small transaction network with a densely-connected positive ring."""
    rng = np.random.default_rng(seed)
    entities = [f"E{i:04d}" for i in range(n_entities)]
    base = pd.Timestamp("2021-01-01")
    rows = []
    for i in range(n_random_edges):
        s, t = rng.choice(n_entities, size=2, replace=False)
        rows.append(
            {
                "source_entity": entities[s],
                "target_entity": entities[t],
                "edge_key": f"tx{i}",
                "amount_usd": float(abs(rng.normal(800, 200))),
                "date": base + pd.Timedelta(days=int(rng.integers(0, 360))),
                "payer_bank_country": "EE",
                "beneficiary_bank_country": "LV",
            }
        )
    if n_positive:
        ring = entities[:n_positive]
        for j in range(n_positive * 12):
            rows.append(
                {
                    "source_entity": ring[j % n_positive],
                    "target_entity": ring[(j + 1) % n_positive],
                    "edge_key": f"ring{j}",
                    "amount_usd": 9000.0,
                    "date": base + pd.Timedelta(days=int(rng.integers(0, 360))),
                    "payer_bank_country": "EE",
                    "beneficiary_bank_country": "EE",
                }
            )
    edge_table = pd.DataFrame(rows)

    graph = nx.MultiDiGraph()
    for entity in entities:
        graph.add_node(entity, entity_type="Company")
    for row in edge_table.itertuples(index=False):
        graph.add_edge(
            row.source_entity,
            row.target_entity,
            key=row.edge_key,
            amount_usd=row.amount_usd,
            date=row.date,
        )

    labels = np.zeros(n_entities, dtype=np.int8)
    labels[:n_positive] = 1

    return ExperimentDataset(
        spec=DatasetSpec(
            name="synthetic_ring",
            version="test-1",
            provenance=f"synthetic:test_experiment_framework@{seed}",
            label_definition="member of the injected circular-transfer ring",
            synthetic=synthetic,
        ),
        entity_ids=entities,
        labels=labels,
        edge_table=edge_table,
        graph=graph,
    )


def _spec(dataset: ExperimentDataset, **overrides) -> ExperimentSpec:
    defaults = dict(
        name="unit",
        dataset=dataset.spec,
        features=FeatureSpec(sets=("aml_core_v1",), scope="global"),
        split=SplitSpec(strategy="stratified_kfold", params={"n_folds": 4}, seed=1),
        model=ModelSpec(family="logistic_regression"),
        threshold=ThresholdSpec(strategy="alert_budget", params={"rate": 0.1}),
        calibration=CalibrationSpec(method="none"),
        evaluation=EvaluationSpec(k_values=(5, 10), alert_rates=(0.05, 0.1)),
        seed=13,
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


@pytest.fixture(scope="module")
def dataset() -> ExperimentDataset:
    return _make_dataset()


# ---------------------------------------------------------------------------
# Metric honesty
# ---------------------------------------------------------------------------

class TestMetricHonesty:
    def test_single_class_partition_reports_none_not_zero(self):
        """The legacy trainer returned 0.0 here, which is indistinguishable
        from a model that ranked every positive last."""
        m = ranking_metrics([0, 0, 0, 0], [0.9, 0.2, 0.5, 0.1])
        assert m["average_precision"] is None
        assert m["roc_auc"] is None
        assert "no positives in partition" in m["undefined"]["average_precision"]

    def test_no_negatives_also_undefined(self):
        m = ranking_metrics([1, 1, 1], [0.3, 0.6, 0.9])
        assert m["average_precision"] is None
        assert m["roc_auc"] is None

    def test_defined_when_both_classes_present(self):
        m = ranking_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], k_values=(2,))
        assert m["average_precision"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["precision_at_k"]["2"] == pytest.approx(1.0)
        assert m["recall_at_k"]["2"] == pytest.approx(1.0)

    def test_ap_lift_is_relative_to_prevalence(self):
        m = ranking_metrics([0] * 99 + [1], [0.0] * 99 + [1.0])
        assert m["prevalence"] == pytest.approx(0.01)
        assert m["average_precision"] == pytest.approx(1.0)
        assert m["ap_lift"] == pytest.approx(100.0)

    def test_precision_at_k_ties_are_deterministic(self):
        y = [0, 1, 0, 1, 0, 1]
        s = [0.5] * 6
        first = ranking_metrics(y, s, k_values=(3,))["precision_at_k"]["3"]
        second = ranking_metrics(y, s, k_values=(3,))["precision_at_k"]["3"]
        assert first == second

    def test_k_outside_range_is_none_not_clamped(self):
        m = ranking_metrics([0, 1], [0.1, 0.9], k_values=(50,))
        assert m["precision_at_k"]["50"] is None

    def test_threshold_metrics_with_no_alerts(self):
        m = threshold_metrics([0, 1, 0], [0.1, 0.2, 0.05], threshold=0.9)
        assert m["alerts"] == 0
        assert m["precision"] is None
        assert m["f1"] is None
        assert m["false_positive_burden"] is None

    def test_false_positive_burden(self):
        m = threshold_metrics([1, 0, 0, 0], [0.9, 0.8, 0.7, 0.1], threshold=0.5)
        assert m["confusion"] == [[1, 2], [0, 1]]
        assert m["false_positive_burden"] == pytest.approx(2.0)
        assert m["alert_rate"] == pytest.approx(0.75)

    def test_no_threshold_means_all_threshold_metrics_none(self):
        m = threshold_metrics([0, 1], [0.2, 0.8], threshold=None)
        assert m["threshold"] is None
        assert all(m[k] is None for k in ("precision", "recall", "f1", "confusion"))

    def test_calibration_refused_for_non_probability_scores(self):
        m = calibration_metrics([0, 1], [-3.0, 4.5])
        assert m["brier"] is None
        assert "not probabilities" in m["undefined"]["all"]

    def test_calibration_computed_for_probabilities(self):
        y = [0] * 50 + [1] * 50
        p = [0.1] * 50 + [0.9] * 50
        m = calibration_metrics(y, p, n_bins=10)
        assert m["brier"] == pytest.approx(0.01)
        assert m["expected_calibration_error"] == pytest.approx(0.1)
        assert sum(b["count"] for b in m["bins"]) == 100

    def test_aggregate_skips_undefined_folds_and_counts_them(self):
        folds = [
            binary_metrics([0, 1], [0.1, 0.9]),
            binary_metrics([0, 0], [0.1, 0.2]),
        ]
        agg = aggregate_fold_metrics(folds)
        assert agg["average_precision"]["n_defined"] == 1
        assert agg["average_precision"]["n_folds"] == 2
        assert agg["average_precision"]["mean"] == pytest.approx(1.0)

    def test_aggregate_all_undefined_is_none_not_nan(self):
        folds = [binary_metrics([0, 0], [0.1, 0.2]) for _ in range(3)]
        agg = aggregate_fold_metrics(folds)
        assert agg["average_precision"]["mean"] is None
        assert agg["average_precision"]["n_defined"] == 0


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

class TestThresholdSelection:
    def test_max_f1_requires_validation_positives(self):
        d = select_threshold(ThresholdSpec(strategy="max_f1"), [0, 0, 0], [0.1, 0.5, 0.9])
        assert d.value is None
        assert "no positives" in d.reason
        assert d.selected_on == "validation"

    def test_max_f1_finds_a_separating_threshold(self):
        d = select_threshold(
            ThresholdSpec(strategy="max_f1"), [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        )
        assert d.value is not None
        assert 0.2 < d.value <= 0.8
        assert d.objective_value == pytest.approx(1.0)

    def test_alert_budget_works_without_any_positives(self):
        """An alert budget is a resourcing decision, not a label-dependent one."""
        scores = np.linspace(0, 1, 100)
        d = select_threshold(
            ThresholdSpec(strategy="alert_budget", params={"rate": 0.1}),
            np.zeros(100, dtype=int),
            scores,
        )
        assert d.value is not None
        assert np.mean(scores >= d.value) == pytest.approx(0.1, abs=0.02)

    def test_min_precision_reports_when_unachievable(self):
        d = select_threshold(
            ThresholdSpec(strategy="min_precision", params={"target": 0.99}),
            [0] * 99 + [1],
            list(np.linspace(1.0, 0.0, 100)),
        )
        assert d.value is None
        assert "precision >= 0.99" in d.reason

    def test_fixed_requires_an_explicit_value(self):
        d = select_threshold(ThresholdSpec(strategy="fixed"), [0, 1], [0.1, 0.9])
        assert d.value is None
        assert "requires an explicit" in d.reason

    def test_no_implicit_half_default_anywhere(self):
        """0.5 is never assumed; an unselectable threshold stays ``None``."""
        d = select_threshold(ThresholdSpec(strategy="max_f1"), [], [])
        assert d.value is None

    def test_threshold_selection_has_no_test_parameter(self):
        import inspect

        params = set(inspect.signature(select_threshold).parameters)
        assert params == {"spec", "y_val", "scores_val"}


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

class TestSplits:
    def test_all_strategies_registered(self):
        assert {
            "stratified_kfold",
            "time_forward_holdout",
            "time_forward_rolling",
            "component_grouped_kfold",
            "legacy_exp3_transductive",
        } <= set(available_split_strategies())

    @pytest.mark.parametrize(
        "strategy,params",
        [
            ("stratified_kfold", {"n_folds": 4}),
            ("legacy_exp3_transductive", {"n_folds": 4}),
            ("time_forward_holdout", {}),
            ("time_forward_rolling", {"n_folds": 2}),
            ("component_grouped_kfold", {"n_folds": 3}),
        ],
    )
    def test_partitions_are_always_disjoint(self, dataset, strategy, params):
        plan = build_split_plan(dataset, SplitSpec(strategy=strategy, params=params, seed=5))
        for fold in plan.folds:
            assert not set(fold.train_idx) & set(fold.val_idx)
            assert not set(fold.train_idx) & set(fold.test_idx)
            assert not set(fold.val_idx) & set(fold.test_idx)

    def test_split_is_reproducible_from_its_seed(self, dataset):
        spec = SplitSpec(strategy="stratified_kfold", params={"n_folds": 4}, seed=99)
        a = build_split_plan(dataset, spec)
        b = build_split_plan(dataset, spec)
        for fa, fb in zip(a.folds, b.folds):
            assert np.array_equal(fa.train_idx, fb.train_idx)
            assert np.array_equal(fa.test_idx, fb.test_idx)

    def test_different_seed_gives_a_different_split(self, dataset):
        a = build_split_plan(dataset, SplitSpec(strategy="stratified_kfold", seed=1))
        b = build_split_plan(dataset, SplitSpec(strategy="stratified_kfold", seed=2))
        assert not np.array_equal(a.folds[0].test_idx, b.folds[0].test_idx)

    def test_time_split_never_trains_on_the_future(self, dataset):
        plan = build_split_plan(
            dataset,
            SplitSpec(
                strategy="time_forward_holdout",
                params={"time_reference": "first"},
                seed=1,
            ),
        )
        assert plan.time_aware is True
        assert plan.regime() == "time-forward"
        times = dataset.entity_times(how="first")
        values = times.to_numpy()
        fold = plan.folds[0]
        assert pd.Series(values[fold.train_idx]).max() <= pd.Series(values[fold.test_idx]).min()

    def test_rolling_windows_expand(self, dataset):
        plan = build_split_plan(
            dataset, SplitSpec(strategy="time_forward_rolling", params={"n_folds": 3}, seed=1)
        )
        sizes = [len(f.train_idx) for f in plan.folds]
        assert sizes == sorted(sizes)
        assert all(f.train_window_end is not None for f in plan.folds)

    def test_time_split_refuses_a_dataset_without_dates(self):
        rows = pd.DataFrame(
            {
                "source_entity": ["A", "B"],
                "target_entity": ["B", "C"],
                "edge_key": ["1", "2"],
                "amount_usd": [1.0, 2.0],
            }
        )
        ds = ExperimentDataset(
            spec=DatasetSpec("t", "1", "synthetic:test", "n/a", synthetic=True),
            entity_ids=["A", "B", "C"],
            labels=np.array([1, 0, 0], dtype=np.int8),
            edge_table=rows,
        )
        with pytest.raises(ValueError, match="no usable"):
            build_split_plan(ds, SplitSpec(strategy="time_forward_holdout"))

    def test_component_grouping_keeps_components_intact(self):
        rows = []
        for component in range(6):
            base = component * 5
            for i in range(4):
                rows.append(
                    {
                        "source_entity": f"E{base + i}",
                        "target_entity": f"E{base + i + 1}",
                        "edge_key": f"{component}-{i}",
                        "amount_usd": 10.0,
                        "date": pd.Timestamp("2021-01-01"),
                    }
                )
        edge_table = pd.DataFrame(rows)
        entities = sorted({*edge_table["source_entity"], *edge_table["target_entity"]})
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(entities)
        labels = np.zeros(len(entities), dtype=np.int8)
        labels[0] = 1
        ds = ExperimentDataset(
            spec=DatasetSpec("c", "1", "synthetic:test", "first entity", synthetic=True),
            entity_ids=entities,
            labels=labels,
            edge_table=edge_table,
            graph=graph,
        )
        plan = build_split_plan(
            ds, SplitSpec(strategy="component_grouped_kfold", params={"n_folds": 3}, seed=4)
        )
        assert plan.graph_leakage_controlled is True
        for fold in plan.folds:
            train_ids = set(ds.ids_for(fold.train_idx))
            test_ids = set(ds.ids_for(fold.test_idx))
            crossing = [
                (u, v)
                for u, v in zip(edge_table["source_entity"], edge_table["target_entity"])
                if (u in train_ids and v in test_ids) or (u in test_ids and v in train_ids)
            ]
            assert crossing == []

    def test_legacy_strategy_declares_itself_transductive(self, dataset):
        plan = build_split_plan(
            dataset, SplitSpec(strategy="legacy_exp3_transductive", params={"n_folds": 4})
        )
        assert plan.graph_leakage_controlled is False
        assert plan.time_aware is False
        assert any("transductive" in n for n in plan.notes)


# ---------------------------------------------------------------------------
# Class-imbalance tolerance
# ---------------------------------------------------------------------------

class TestImbalanceTolerance:
    @pytest.mark.parametrize("n_positive", [1, 2, 4, 7, 30])
    def test_arbitrary_positive_counts_run(self, n_positive):
        """The legacy fold builder asserted exactly four positives."""
        ds = _make_dataset(n_entities=80, n_positive=n_positive, n_random_edges=300, seed=n_positive)
        result = run_experiment(_spec(ds, name=f"pos{n_positive}"), ds)
        assert len(result.folds) == 4

    def test_zero_positive_test_partition_is_reported_not_faked(self):
        ds = _make_dataset(n_entities=60, n_positive=2, n_random_edges=200, seed=9)
        result = run_experiment(
            _spec(ds, split=SplitSpec(strategy="stratified_kfold", params={"n_folds": 4}, seed=1)),
            ds,
        )
        empty = [f for f in result.folds if f.counts["n_test_pos"] == 0]
        assert empty, "expected at least one positive-free test fold with 2 positives over 4 folds"
        for fold in empty:
            assert fold.test_metrics["average_precision"] is None
        assert result.aggregate["average_precision"]["n_defined"] < len(result.folds)

    def test_dataset_with_no_positives_at_all_does_not_crash(self):
        ds = _make_dataset(n_entities=40, n_positive=0, n_random_edges=120, seed=2)
        result = run_experiment(
            _spec(ds, name="nopos", model=ModelSpec(family="structural_heuristic")), ds
        )
        assert result.aggregate["average_precision"]["n_defined"] == 0
        assert any("no positives" in n for n in result.split_summary["notes"])

    def test_single_class_training_partition_yields_constant_score(self):
        ds = _make_dataset(n_entities=40, n_positive=0, n_random_edges=120, seed=2)
        result = run_experiment(_spec(ds, name="nopos_lr"), ds)
        assert result.folds[0].fit_report["status"] == "not_fitted"


# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------

class TestLeakageControl:
    def test_label_column_in_features_is_a_hard_error(self, dataset):
        plan = build_split_plan(dataset, SplitSpec(strategy="stratified_kfold"))
        features = pd.DataFrame(
            {"aml_core_v1::core": np.zeros(dataset.n_entities)},
            index=pd.Index(dataset.entity_ids),
        )
        findings = audit_features(features, dataset, plan.folds[0], scope="global")
        errors = [f for f in findings if f.severity == "error"]
        assert errors and errors[0].check == "label_columns"

    def test_runner_aborts_when_a_label_column_reaches_the_matrix(self, dataset, monkeypatch):
        import src.experiments.runner as runner_module
        from src.experiments.features import FeatureMatrix

        def poisoned(dataset_, spec_, fold_):
            values = pd.DataFrame(
                {"x::investigation": np.zeros(dataset_.n_entities)},
                index=pd.Index(dataset_.entity_ids),
            )
            return FeatureMatrix(values=values, scope="global")

        monkeypatch.setattr(runner_module, "_compute_features", poisoned)
        with pytest.raises(LeakageError, match="label_columns"):
            run_experiment(_spec(dataset, name="poisoned"), dataset)

    def test_perfect_separator_is_flagged_as_a_warning(self, dataset):
        plan = build_split_plan(dataset, SplitSpec(strategy="stratified_kfold", seed=1))
        fold = plan.folds[0]
        leaked = pd.Series(dataset.labels.astype(float) * 100.0, index=pd.Index(dataset.entity_ids))
        features = pd.DataFrame({"aml_core_v1::leaky": leaked})
        findings = audit_features(features, dataset, fold, scope="global")
        assert any(f.check == "constant_or_perfect_features" for f in findings)

    def test_global_feature_scope_is_declared_in_the_artifact(self, dataset):
        result = run_experiment(_spec(dataset, name="scope_global"), dataset)
        scope_findings = [f for f in result.leakage if f["check"] == "feature_scope"]
        assert scope_findings
        assert "whole population" in scope_findings[0]["message"]

    def test_train_window_scope_hides_future_transactions(self, dataset):
        """Features computed under a training window must differ from features
        computed over the whole history - otherwise the window did nothing."""
        plan = build_split_plan(
            dataset, SplitSpec(strategy="time_forward_holdout", params={}, seed=1)
        )
        fold = plan.folds[0]
        windowed = build_feature_matrix(
            FeatureRequest(
                entity_ids=list(dataset.entity_ids),
                edge_table=dataset.edges_before(fold.train_window_end),
                graph=dataset.subgraph_from_edges(dataset.edges_before(fold.train_window_end)),
                as_of=fold.train_window_end,
            ),
            ("aml_core_v1",),
        )
        full = build_feature_matrix(
            FeatureRequest(
                entity_ids=list(dataset.entity_ids),
                edge_table=dataset.edge_table,
                graph=dataset.graph,
            ),
            ("aml_core_v1",),
        )
        assert len(dataset.edges_before(fold.train_window_end)) < len(dataset.edge_table)
        assert not windowed.values["aml_core_v1::txn_total"].equals(
            full.values["aml_core_v1::txn_total"]
        )

    def test_time_aware_run_declares_its_window(self, dataset):
        result = run_experiment(
            _spec(
                dataset,
                name="windowed",
                features=FeatureSpec(sets=("aml_core_v1",), scope="train_window"),
                split=SplitSpec(strategy="time_forward_holdout", seed=1),
            ),
            dataset,
        )
        scope = [f for f in result.leakage if f["check"] == "feature_scope"]
        assert scope and "transactions before" in scope[0]["message"]

    def test_non_temporal_split_on_temporal_data_warns(self, dataset):
        plan = build_split_plan(dataset, SplitSpec(strategy="stratified_kfold"))
        findings = audit_split_plan(plan, dataset)
        assert any(
            f.check == "temporal_ordering" and f.severity == "warning" for f in findings
        )

    def test_threshold_comes_from_validation_only(self, dataset):
        """The recorded threshold must equal the one selectable from validation
        scores alone; nothing about the test partition may move it."""
        result = run_experiment(
            _spec(dataset, name="thr", threshold=ThresholdSpec(strategy="max_f1")), dataset
        )
        for fold in result.folds:
            assert fold.threshold["selected_on"] == "validation"
            assert fold.threshold["n_val_positives"] == fold.counts["n_val_pos"]

    def test_fit_context_exposes_no_test_labels(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(FitContext)}
        assert "y_test" not in names
        assert "test_idx" not in names


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_refused_below_the_positive_floor(self, dataset):
        result = run_experiment(
            _spec(
                dataset,
                name="cal_small",
                calibration=CalibrationSpec(method="isotonic", min_val_positives=1000),
            ),
            dataset,
        )
        fold = result.folds[0]
        assert fold.calibration["method"] == "none"
        assert any("below the configured minimum" in n for n in fold.notes)

    def test_calibration_fits_when_positives_suffice(self):
        ds = _make_dataset(n_entities=400, n_positive=120, n_random_edges=1500, seed=6)
        result = run_experiment(
            _spec(
                ds,
                name="cal_ok",
                calibration=CalibrationSpec(method="platt", min_val_positives=5),
                split=SplitSpec(strategy="stratified_kfold", params={"n_folds": 3}, seed=1),
            ),
            ds,
        )
        fold = result.folds[0]
        assert fold.calibration["method"] == "platt"
        assert fold.calibration["fitted"] is True
        assert fold.test_metrics["calibration"]["brier"] is not None

    def test_no_calibration_metrics_for_non_probability_scores(self, dataset):
        result = run_experiment(
            _spec(dataset, name="heur_cal", model=ModelSpec(family="structural_heuristic")),
            dataset,
        )
        assert result.folds[0].test_metrics["calibration"] is None


# ---------------------------------------------------------------------------
# Candidate contract
# ---------------------------------------------------------------------------

class TestCandidateContract:
    def test_shipped_families_are_registered(self):
        families = set(available_candidates())
        assert {"structural_heuristic", "logistic_regression", "gradient_boosting"} <= families

    @pytest.mark.parametrize(
        "family", ["structural_heuristic", "logistic_regression", "gradient_boosting"]
    )
    def test_every_baseline_runs_on_the_same_spec(self, dataset, family):
        params = {"n_estimators": 40} if family == "gradient_boosting" else {}
        result = run_experiment(
            _spec(dataset, name=family, model=ModelSpec(family=family, params=params)), dataset
        )
        assert len(result.folds) == 4
        assert result.spec.model.family == family

    def test_a_new_candidate_needs_no_runner_change(self, dataset):
        """The architectural success criterion: adding a model family touches
        only the model file."""

        @register_candidate("unit_test_constant")
        class _Constant(Candidate):
            emits_probabilities = True

            def fit(self, ctx: FitContext) -> "_Constant":
                self._value = float(ctx.y_train.mean()) if len(ctx.y_train) else 0.0
                self.fit_report = {"status": "fitted", "value": self._value}
                self._fitted = True
                return self

            def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
                return np.full(len(index), self._value, dtype=np.float64)

        try:
            result = run_experiment(
                _spec(dataset, name="ctor", model=ModelSpec(family="unit_test_constant")), dataset
            )
            assert result.folds[0].fit_report["status"] == "fitted"
        finally:
            from src.experiments.candidate import _REGISTRY

            _REGISTRY.pop("unit_test_constant", None)

    def test_heuristic_never_reads_labels(self, dataset):
        result = run_experiment(
            _spec(dataset, name="heur", model=ModelSpec(family="structural_heuristic")), dataset
        )
        assert result.folds[0].fit_report["labels_used"] is False

    def test_preprocessing_is_fitted_on_train_only(self, dataset):
        """The imputer's learned medians must come from training rows.  If the
        full matrix were used, adding an extreme test-only row would move them."""
        plan = build_split_plan(dataset, SplitSpec(strategy="stratified_kfold", seed=1))
        fold = plan.folds[0]
        matrix = build_feature_matrix(
            FeatureRequest(
                entity_ids=list(dataset.entity_ids),
                edge_table=dataset.edge_table,
                graph=dataset.graph,
            ),
            ("aml_core_v1",),
        )
        ctx = FitContext(
            dataset=dataset,
            features=matrix.values,
            fold=fold,
            train_idx=fold.train_idx,
            val_idx=fold.val_idx,
            y_train=dataset.labels[fold.train_idx],
            y_val=dataset.labels[fold.val_idx],
        )
        model = build_candidate("logistic_regression")
        model.fit(ctx)
        train_only = model._imputer.statistics_.copy()

        poisoned = matrix.values.copy()
        column = "aml_core_v1::amount_max"
        poisoned.iloc[fold.test_idx, poisoned.columns.get_loc(column)] = 1e12
        ctx2 = FitContext(
            dataset=dataset,
            features=poisoned,
            fold=fold,
            train_idx=fold.train_idx,
            val_idx=fold.val_idx,
            y_train=dataset.labels[fold.train_idx],
            y_val=dataset.labels[fold.val_idx],
        )
        model2 = build_candidate("logistic_regression")
        model2.fit(ctx2)
        assert np.allclose(train_only, model2._imputer.statistics_, equal_nan=True)


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

class TestFeatureRegistry:
    def test_legacy_and_new_families_are_both_available(self):
        names = set(available_feature_sets())
        assert {"legacy_exp3_v1", "baseline_v1", "multigraph_v1", "aml_core_v1"} <= names

    def test_legacy_exp3_reproduces_the_167_column_representation(self, dataset):
        matrix = build_feature_matrix(
            FeatureRequest(
                entity_ids=list(dataset.entity_ids),
                edge_table=dataset.edge_table,
                graph=dataset.graph,
            ),
            ("legacy_exp3_v1",),
        )
        assert matrix.values.shape[1] == 167

    def test_legacy_adapter_matches_the_legacy_tensor_builder(self, dataset):
        """Guards against the adapter drifting from ``build_pyg_data``, which is
        the only thing that makes a legacy-vs-new comparison meaningful."""
        from src.features.baseline import compute_baseline
        from src.features.multigraph import compute_multigraph
        from src.models.dataset import build_pyg_data

        baseline = compute_baseline(dataset.graph, dataset.edge_table)
        multigraph = compute_multigraph(dataset.graph, dataset.edge_table)
        pyg = build_pyg_data(dataset.graph, baseline, multigraph=multigraph)

        matrix = build_feature_matrix(
            FeatureRequest(
                entity_ids=sorted(dataset.graph.nodes()),
                edge_table=dataset.edge_table,
                graph=dataset.graph,
            ),
            ("legacy_exp3_v1",),
        )
        legacy = pyg.data.x.numpy()
        new = matrix.values.to_numpy(dtype=np.float64)
        assert legacy.shape == new.shape
        assert np.allclose(legacy, new, atol=1e-5)

    def test_aml_core_leaves_undefined_statistics_as_nan(self):
        edge_table = pd.DataFrame(
            {
                "source_entity": ["A"],
                "target_entity": ["B"],
                "edge_key": ["1"],
                "amount_usd": [100.0],
                "date": [pd.Timestamp("2021-01-01")],
            }
        )
        matrix = build_feature_matrix(
            FeatureRequest(entity_ids=["A", "B", "C"], edge_table=edge_table, graph=None),
            ("aml_core_v1",),
        )
        values = matrix.values
        # One transaction each: inter-arrival time is undefined, not zero.
        assert np.isnan(values.loc["A", "aml_core_v1::mean_iat_days"])
        # C never transacted: counts are a real zero, ratios are undefined.
        assert values.loc["C", "aml_core_v1::txn_total"] == 0.0
        assert np.isnan(values.loc["C", "aml_core_v1::flow_imbalance"])

    def test_aml_core_column_names_are_prefixed_by_family(self, dataset):
        matrix = build_feature_matrix(
            FeatureRequest(
                entity_ids=list(dataset.entity_ids),
                edge_table=dataset.edge_table,
                graph=dataset.graph,
            ),
            ("aml_core_v1",),
        )
        assert all(str(c).startswith("aml_core_v1::") for c in matrix.values.columns)
        assert matrix.sets[0]["version"] == "1.0"

    def test_feature_sets_compose_in_the_declared_order(self, dataset):
        matrix = build_feature_matrix(
            FeatureRequest(
                entity_ids=list(dataset.entity_ids),
                edge_table=dataset.edge_table,
                graph=dataset.graph,
            ),
            ("multigraph_v1", "aml_core_v1"),
        )
        columns = list(matrix.values.columns)
        assert columns[0].startswith("multigraph_v1::")
        assert columns[-1].startswith("aml_core_v1::")


# ---------------------------------------------------------------------------
# Provenance and reproducibility
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_spec_round_trips_through_json(self, dataset):
        spec = _spec(dataset, name="rt")
        restored = ExperimentSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert restored.to_dict() == spec.to_dict()

    def test_artifact_records_every_required_dimension(self, dataset, tmp_path):
        result = run_experiment(_spec(dataset, name="prov"), dataset, output_dir=tmp_path)
        files = list(tmp_path.glob("prov__*.json"))
        assert len(files) == 1
        artifact = load_result(files[0])

        assert artifact["dataset"]["spec"]["version"] == "test-1"
        assert artifact["dataset"]["spec"]["provenance"].startswith("synthetic:")
        assert artifact["dataset"]["spec"]["label_definition"]
        assert artifact["features"]["sets"][0]["name"] == "aml_core_v1"
        assert artifact["features"]["scope"] == "global"
        assert artifact["split"]["strategy"] == "stratified_kfold"
        assert artifact["split"]["seed"] == 1
        assert artifact["split"]["regime"]
        assert artifact["experiment"]["model"]["family"] == "logistic_regression"
        assert artifact["experiment"]["model"]["imbalance"] == "class_weight"
        assert artifact["experiment"]["seed"] == 13
        assert artifact["folds"][0]["threshold"]["selected_on"] == "validation"
        assert "calibration" in artifact["folds"][0]
        assert artifact["folds"][0]["runtime_seconds"]["fit"] >= 0
        assert artifact["environment"]["packages"]["numpy"]
        assert artifact["determinism"]["not_controlled"]
        assert artifact["started_at"] and artifact["finished_at"]
        assert artifact["schema_version"]

    def test_synthetic_flag_is_carried_into_the_artifact(self, dataset, tmp_path):
        result = run_experiment(_spec(dataset, name="synth_flag"), dataset)
        assert result.spec.dataset.synthetic is True
        assert any("SYNTHETIC DATASET" in n for n in result.notes)
        assert "(SYNTHETIC)" in result.headline()

    def test_same_seed_gives_the_same_metrics(self, dataset):
        a = run_experiment(_spec(dataset, name="seeded"), dataset)
        b = run_experiment(_spec(dataset, name="seeded"), dataset)
        for fa, fb in zip(a.folds, b.folds):
            assert fa.test_metrics["average_precision"] == pytest.approx(
                fb.test_metrics["average_precision"], rel=1e-9
            )

    def test_checkpoint_identity_is_recorded_when_persisted(self, dataset, tmp_path):
        result = run_experiment(
            _spec(
                dataset,
                name="ckpt",
                model=ModelSpec(family="gradient_boosting", params={"n_estimators": 30}),
            ),
            dataset,
            checkpoint_dir=tmp_path,
        )
        checkpoint = result.folds[0].checkpoint
        assert checkpoint is not None
        assert len(checkpoint["sha256"]) == 64
        assert checkpoint["bytes"] > 0

    def test_headline_makes_no_superiority_claim(self, dataset):
        text = run_experiment(_spec(dataset, name="head"), dataset).headline()
        assert "best" not in text.lower()
        assert "state of the art" not in text.lower()


# ---------------------------------------------------------------------------
# Legacy GraphSAGE candidate
# ---------------------------------------------------------------------------

@pytest.mark.ml
class TestLegacyGraphSAGECandidate:
    def test_it_runs_under_the_new_contract(self, dataset):
        torch = pytest.importorskip("torch")
        result = run_experiment(
            _spec(
                dataset,
                name="sage",
                features=FeatureSpec(sets=("legacy_exp3_v1",), scope="global"),
                split=SplitSpec(strategy="legacy_exp3_transductive", params={"n_folds": 4}, seed=1),
                model=ModelSpec(
                    family="legacy_graphsage",
                    params={"max_epochs": 8, "patience": 4},
                    imbalance="pos_weight",
                ),
            ),
            dataset,
        )
        report = result.folds[0].fit_report
        assert report["status"] == "fitted"
        assert report["input_dim"] == 167
        assert "GraphSAGE" in report["architecture"]

    def test_it_is_not_bound_to_167_features(self, dataset):
        """The whole point of the new feature layer: the architecture accepts
        whatever representation the experiment configured."""
        pytest.importorskip("torch")
        result = run_experiment(
            _spec(
                dataset,
                name="sage_aml",
                features=FeatureSpec(sets=("aml_core_v1",), scope="global"),
                split=SplitSpec(strategy="stratified_kfold", params={"n_folds": 3}, seed=1),
                model=ModelSpec(
                    family="legacy_graphsage", params={"max_epochs": 5, "patience": 3}
                ),
            ),
            dataset,
        )
        assert result.folds[0].fit_report["input_dim"] == 29

    def test_frozen_checkpoint_is_untouched(self):
        """This candidate trains a new model; it must never write to the frozen
        Experiment 3 inference checkpoint."""
        import inspect

        import src.experiments.legacy_graphsage as module

        source = inspect.getsource(module)
        assert "experiment3_graphsage.pt" not in source.replace(
            "``artifacts/models/experiment3_graphsage.pt``", ""
        )


# ---------------------------------------------------------------------------
# Scaling behaviour
# ---------------------------------------------------------------------------

class TestFeatureReuseAndGraphAvoidance:
    """Properties that make the framework affordable on a benchmark-scale dataset.

    They are behavioural, not performance assertions: reusing a matrix that
    should not be reused, or building a graph nothing asked for, are both
    correctness problems before they are cost problems.
    """

    def test_global_scope_computes_the_matrix_once(self, dataset, monkeypatch):
        """Every fold sees identical features under the global regime."""
        from src.experiments import runner as runner_module

        calls: list[int] = []
        original = runner_module._compute_features

        def counting(ds, spec, fold):
            calls.append(fold.fold_idx)
            return original(ds, spec, fold)

        monkeypatch.setattr(runner_module, "_compute_features", counting)
        result = run_experiment(_spec(dataset), dataset)
        assert len(result.folds) == 4
        assert len(calls) == 1, f"expected one feature computation, got {len(calls)}"

    def test_train_window_scope_recomputes_per_fold(self, dataset, monkeypatch):
        """A fold must never inherit a matrix built past its own boundary."""
        from src.experiments import runner as runner_module

        calls: list[int] = []
        original = runner_module._compute_features

        def counting(ds, spec, fold):
            calls.append(fold.fold_idx)
            return original(ds, spec, fold)

        monkeypatch.setattr(runner_module, "_compute_features", counting)
        spec = _spec(
            dataset,
            features=FeatureSpec(sets=("aml_core_v1",), scope="train_window"),
            split=SplitSpec(
                strategy="time_forward_rolling",
                params={"n_folds": 2},
                seed=1,
            ),
        )
        result = run_experiment(spec, dataset)
        assert len(calls) == len(result.folds)

    def test_no_graph_is_built_when_no_family_asks_for_one(self, dataset, monkeypatch):
        """aml_* families read the edge table; a MultiDiGraph must not be constructed."""
        from src.experiments import runner as runner_module

        built: list[int] = []

        def refuse(self, edges):  # noqa: ANN001 - patched method
            built.append(1)
            raise AssertionError("subgraph_from_edges must not be called here")

        monkeypatch.setattr(ExperimentDataset, "subgraph_from_edges", refuse)
        spec = _spec(
            dataset,
            features=FeatureSpec(
                sets=("aml_txn_v1", "aml_flow_v1", "aml_network_v1"),
                scope="train_window",
            ),
            split=SplitSpec(
                strategy="time_forward_rolling", params={"n_folds": 2}, seed=1
            ),
        )
        run_experiment(spec, dataset)
        assert built == []

    def test_a_graph_is_still_built_for_a_legacy_family(self, dataset, monkeypatch):
        """The avoidance must be conditional, not a removal."""
        from src.experiments.runner import _needs_graph

        legacy = _spec(dataset, features=FeatureSpec(sets=("baseline_v1",), scope="global"))
        modern = _spec(dataset, features=FeatureSpec(sets=("aml_txn_v1",), scope="global"))
        assert _needs_graph(legacy) is True
        assert _needs_graph(modern) is False


class TestResumableBatch:
    """``run_experiment.py --skip-existing`` must skip only a matching result."""

    def _write(self, directory, name, version):
        import json

        path = directory / f"{name}__20260101T000000.json"
        path.write_text(
            json.dumps(
                {
                    "experiment": {"name": name},
                    "dataset": {"spec": {"version": version}},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_matching_name_and_dataset_version_is_found(self, tmp_path):
        from run_experiment import _existing_artifact

        written = self._write(tmp_path, "unit", "v1")
        assert _existing_artifact(tmp_path, "unit", "v1") == written

    def test_a_different_dataset_version_is_not_a_result_for_this_run(self, tmp_path):
        """Skipping on the name alone would silently keep a stale number."""
        from run_experiment import _existing_artifact

        self._write(tmp_path, "unit", "v1")
        assert _existing_artifact(tmp_path, "unit", "v2") is None

    def test_missing_directory_and_unreadable_file_are_not_errors(self, tmp_path):
        from run_experiment import _existing_artifact

        assert _existing_artifact(tmp_path / "absent", "unit", "v1") is None
        (tmp_path / "unit__bad.json").write_text("{not json", encoding="utf-8")
        assert _existing_artifact(tmp_path, "unit", "v1") is None


class TestTimeForwardScopeInteraction:
    """A time-forward split at global feature scope is scientifically void.

    ``time_forward_holdout`` assigns entities to partitions by their own
    activity time.  ``aml_temporal_v1::recency_days`` at ``global`` scope is
    measured from the end of the *whole* file, so it is a monotone transform of
    that same assignment variable: on IBM HI-Small the train and test ranges are
    disjoint (train min 8.900 days, test max 8.410).  The feature becomes a
    partition indicator and the model extrapolates outside its training support.

    The combination is legitimate to *run* as a diagnostic - it is how the
    split effect was separated from the feature-scope effect in
    ``docs/temporal_collapse.md`` - so the framework does not forbid it.  What
    it must do is declare it, so that nobody quotes the number as a
    time-forward result.
    """

    def test_global_scope_under_a_time_aware_split_is_declared(self, dataset):
        spec = _spec(
            dataset,
            features=FeatureSpec(sets=("aml_temporal_v1",), scope="global"),
            split=SplitSpec(
                strategy="time_forward_rolling", params={"n_folds": 2}, seed=1
            ),
        )
        result = run_experiment(spec, dataset)
        blob = json.dumps(result.to_dict())
        # the transductive-feature regime is recorded ...
        assert "transductive feature regime" in blob
        # ... and the split still declares itself time-aware, so the two
        # statements sit side by side in the artifact.
        assert result.to_dict()["split"]["time_aware"] is True
        assert result.to_dict()["features"]["scope"] == "global"

    def test_recency_is_the_only_as_of_dependent_column_pair(self, dataset):
        """Pins which columns move with the reference 'now'.

        If a future family adds another ``as_of``-dependent column without
        noticing, this fails - and that column would carry the same
        partition-indicator risk under a time-aware split.
        """
        import numpy as np
        import pandas as pd

        from src.experiments.features import FeatureRequest, build_feature_matrix

        edges = dataset.edge_table
        entities = list(dataset.entity_ids)
        early = build_feature_matrix(
            FeatureRequest(entity_ids=entities, edge_table=edges, graph=None),
            ["aml_temporal_v1"],
        ).values
        later = build_feature_matrix(
            FeatureRequest(
                entity_ids=entities,
                edge_table=edges,
                graph=None,
                as_of=pd.Timestamp("2030-01-01"),
            ),
            ["aml_temporal_v1"],
        ).values
        moved = {
            str(c).split("::")[-1]
            for c in early.columns
            if not np.allclose(
                early[c].to_numpy(), later[c].to_numpy(), equal_nan=True
            )
        }
        assert moved == {"recency_days", "tenure_days"}
