"""
Tests for cross-period detection transfer (``src/experiments/detection_transfer.py``).

Grouped by the property being protected:

* plan validity      - overlapping, empty or unsound window designs are refused
* window labels      - the label is flagged involvement *inside* the window
* feature window     - a snapshot summarises that window and nothing else
* no leakage         - later-window labels and rows cannot touch the fit
* artifact           - the result declares its regime, strata and diagnostics
* contract           - the frozen model reaches the later window through
                       ``CandidateRiskEngine``

The ledgers are tiny and hand-built. Nothing here is a statement about
detection performance.
"""

from __future__ import annotations

import json

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
    build_detection_transfer_plan,
    load_result,
    run_experiment,
    save_result,
)
from src.experiments.candidate import Candidate, FitContext
from src.experiments.detection_transfer import (
    DETECTION_TRANSFER_STRATEGY,
    resolve_detection_window,
    run_detection_transfer_experiment,
)
from src.experiments.calendar import ledger_arrays
from src.experiments.snapshots import clear_snapshot_cache, window_snapshot
from src.experiments.sources import SourceSpec, build_transaction_dataset

pytestmark = pytest.mark.fast

BASE = pd.Timestamp("2022-05-02")  # a Monday
FAMILIES = ("aml_txn_v1", "aml_temporal_v1")


def day(d: float) -> pd.Timestamp:
    return BASE + pd.Timedelta(days=d)


WINDOWS = [["W1", str(day(0)), str(day(2))], ["W2", str(day(2)), str(day(4))],
           ["W3", str(day(4)), str(day(6))]]


def _ledger(extra: list[tuple] = ()) -> pd.DataFrame:
    """Six days of background traffic; laundering rings in every window.

    ``R0..R3`` launder in all three windows. ``T0/T1`` only in W1, ``V0/V1``
    only in W2, ``U0/U1`` only in W3. ``LATE0/LATE1`` transact only in W3, so
    they are new entities there.
    """
    rng = np.random.default_rng(5)
    names = [f"N{i:02d}" for i in range(36)] + ["R0", "R1", "R2", "R3", "T0", "T1", "V0", "V1", "U0", "U1"]
    rows: list[tuple] = []
    for d in range(6):
        for _ in range(70):
            s, t = rng.choice(len(names), size=2, replace=False)
            rows.append((names[s], names[t], float(rng.integers(20, 600)), day(d + rng.random()), 0))
        for i in range(4):
            rows.append((f"R{i}", f"R{(i + 1) % 4}", 8000.0, day(d + 0.4), 1))
    rows.append(("T0", "T1", 7000.0, day(0.6), 1))
    rows.append(("V0", "V1", 7000.0, day(2.6), 1))
    rows.append(("U0", "U1", 7000.0, day(4.6), 1))
    for d in (4.2, 5.2):
        rows.append(("LATE0", "LATE1", 300.0, day(d), 0))
    rows.extend(extra)
    return pd.DataFrame(rows, columns=["src", "dst", "amount", "ts", "flag"])


def _dataset(frame: pd.DataFrame, version: str = "dt-1") -> ExperimentDataset:
    return build_transaction_dataset(
        frame,
        SourceSpec(source="src", target="dst", label="flag", amount="amount", timestamp="ts"),
        DatasetSpec(
            name="synthetic_detection",
            version=version,
            provenance="synthetic:test_detection_transfer",
            label_definition="flagged involvement inside the detection window",
            synthetic=True,
        ),
    )


def _split(**params) -> SplitSpec:
    base = {"windows": WINDOWS, "partition": {"fit": 0.6, "val": 0.2, "in_period": 0.2},
            "min_positives": 1}
    base.update(params)
    return SplitSpec(strategy=DETECTION_TRANSFER_STRATEGY, params=base, seed=3)


def _spec(dataset: ExperimentDataset, **overrides) -> ExperimentSpec:
    defaults = dict(
        name="r1_unit",
        dataset=dataset.spec,
        features=FeatureSpec(sets=FAMILIES, scope="detection_window"),
        split=_split(),
        model=ModelSpec(family="logistic_regression"),
        threshold=ThresholdSpec(strategy="alert_budget", params={"rate": 0.1}),
        calibration=CalibrationSpec(method="none"),
        evaluation=EvaluationSpec(k_values=(10,), alert_rates=(0.01, 0.1)),
        seed=11,
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _ledger()


@pytest.fixture(scope="module")
def dataset(frame) -> ExperimentDataset:
    return _dataset(frame)


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_snapshot_cache()
    yield
    clear_snapshot_cache()


def _labels(plan, dataset, window_name) -> dict[str, int]:
    window = next(w for w in plan.windows if w.name == window_name)
    return {dataset.entity_ids[i]: int(y) for i, y in zip(window.eligible_idx, window.labels)}


class _Recorder(Candidate):
    """Snapshot-safe candidate that records exactly what it was shown."""

    calendar_compatible = True
    emits_probabilities = True
    seen: list[dict] = []

    def fit(self, ctx: FitContext) -> "_Recorder":
        _Recorder.seen.append(
            {
                "y_train": ctx.y_train.copy(),
                "y_val": ctx.y_val.copy(),
                "fit_features": ctx.features.take(ctx.train_idx, axis=0).copy(),
                "edges_span": (
                    pd.to_datetime(ctx.train_edges["date"]).min(),
                    pd.to_datetime(ctx.train_edges["date"]).max(),
                ),
            }
        )
        self._fitted = True
        return self

    def _column(self, features):
        return [c for c in features.columns if c.endswith("::txn_total")][0]

    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        values = ctx.features[self._column(ctx.features)].to_numpy(dtype=np.float64)[index]
        return 1.0 / (1.0 + np.exp(-values / 10.0))

    def score_features(self, features: pd.DataFrame) -> np.ndarray:
        values = features[self._column(features)].to_numpy(dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-values / 10.0))


# ---------------------------------------------------------------------------
# Plan validity
# ---------------------------------------------------------------------------


class TestPlanValidity:
    def test_folds_are_consecutive_and_calendar_separated(self, dataset):
        plan = build_detection_transfer_plan(dataset, _split())
        assert plan.folds == [(0, 1), (1, 2)]
        assert plan.regime() == "detection-transfer"
        for a, b in plan.folds:
            assert plan.windows[a].end <= plan.windows[b].start

    def test_overlapping_windows_are_refused(self, dataset):
        overlapping = [["A", str(day(0)), str(day(3))], ["B", str(day(2)), str(day(4))]]
        with pytest.raises(ValueError, match="overlap in calendar time"):
            build_detection_transfer_plan(dataset, _split(windows=overlapping))

    def test_a_single_window_cannot_transfer(self, dataset):
        with pytest.raises(ValueError, match="at least two windows"):
            build_detection_transfer_plan(dataset, _split(windows=WINDOWS[:1]))

    def test_max_window_end_excludes_a_tail(self, dataset):
        with pytest.raises(ValueError, match="max_window_end"):
            build_detection_transfer_plan(dataset, _split(max_window_end=str(day(5))))
        build_detection_transfer_plan(dataset, _split(max_window_end=str(day(6))))

    def test_partition_must_be_complete_and_normalised(self, dataset):
        with pytest.raises(ValueError, match="partition must name"):
            build_detection_transfer_plan(dataset, _split(partition={"fit": 0.8, "val": 0.2}))
        with pytest.raises(ValueError, match="sum to 1"):
            build_detection_transfer_plan(
                dataset, _split(partition={"fit": 0.8, "val": 0.2, "in_period": 0.2})
            )

    def test_too_few_positives_is_refused(self, dataset):
        with pytest.raises(ValueError, match="fewer than the required"):
            build_detection_transfer_plan(dataset, _split(min_positives=10_000))

    def test_unknown_parameter_is_refused(self, dataset):
        with pytest.raises(ValueError, match="unknown"):
            build_detection_transfer_plan(dataset, _split(horizon_days=2))

    def test_every_window_positive_is_scoreable(self, dataset):
        """A flagged transaction inside the window makes both endpoints active."""
        plan = build_detection_transfer_plan(dataset, _split())
        for window in plan.windows:
            assert window.n_positive_not_eligible == 0
            assert window.n_positive > 0

    def test_unequal_window_lengths_are_recorded(self, dataset):
        uneven = [["A", str(day(0)), str(day(2))], ["B", str(day(2)), str(day(5))]]
        plan = build_detection_transfer_plan(dataset, _split(windows=uneven))
        assert any("not equal length" in n for n in plan.notes)

    def test_plan_declares_the_task_and_the_overlap_policy(self, dataset):
        summary = build_detection_transfer_plan(dataset, _split()).to_dict()
        assert summary["task"] == "retrospective_detection"
        assert summary["entity_disjoint"] is False
        assert summary["unit"] == "(entity, detection window)"
        assert any("no survivorship selection" in n for n in summary["notes"])


# ---------------------------------------------------------------------------
# Window labels
# ---------------------------------------------------------------------------


class TestWindowLabels:
    def test_label_is_involvement_inside_the_window_only(self, dataset):
        plan = build_detection_transfer_plan(dataset, _split())
        w1, w2, w3 = (_labels(plan, dataset, n) for n in ("W1", "W2", "W3"))
        assert w1["T0"] == w1["T1"] == 1
        assert w2.get("T0", 0) == 0 and w3.get("T0", 0) == 0
        assert w2["V0"] == w2["V1"] == 1
        assert w1.get("V0", 0) == 0 and w3.get("V0", 0) == 0
        assert w3["U0"] == w3["U1"] == 1
        for w in (w1, w2, w3):
            assert w["R0"] == 1  # the ring launders in every window

    def test_window_label_differs_from_the_whole_file_label(self, dataset):
        plan = build_detection_transfer_plan(dataset, _split())
        whole = dict(zip(dataset.entity_ids, dataset.labels.tolist()))
        assert whole["T0"] == 1
        assert _labels(plan, dataset, "W3").get("T0", 0) == 0

    def test_window_start_is_inclusive_and_end_exclusive(self, frame):
        extra = [("N00", "N01", 5000.0, day(2), 1), ("N02", "N03", 5000.0, day(4), 1)]
        ds = _dataset(pd.concat([frame, pd.DataFrame(extra, columns=frame.columns)], ignore_index=True), "dt-edge")
        plan = build_detection_transfer_plan(ds, _split())
        w2 = _labels(plan, ds, "W2")
        assert w2["N00"] == 1 and w2["N01"] == 1   # exactly at W2's start
        assert w2["N02"] == 0 and w2["N03"] == 0   # exactly at W2's end -> W3
        assert _labels(plan, ds, "W3")["N02"] == 1

    def test_resolve_window_rejects_an_inverted_span(self, dataset):
        with pytest.raises(ValueError, match="not before"):
            resolve_detection_window(ledger_arrays(dataset), "bad", day(3), day(1))


# ---------------------------------------------------------------------------
# Feature window
# ---------------------------------------------------------------------------


class TestFeatureWindow:
    def test_snapshot_ignores_rows_outside_the_window(self, frame):
        base = _dataset(frame, "dt-a")
        perturbed = frame.copy()
        outside = (perturbed["ts"] < day(2)) | (perturbed["ts"] >= day(4))
        perturbed.loc[outside, "amount"] *= 17.0
        extra = pd.DataFrame(
            [("N05", "N06", 1.0, day(0.1), 0), ("N05", "N06", 1.0, day(5.9), 0)] * 12,
            columns=frame.columns,
        )
        other = _dataset(pd.concat([perturbed, extra], ignore_index=True), "dt-b")
        a, *_ = window_snapshot(base, FAMILIES, start=day(2), end=day(4), as_of=day(4))
        b, *_ = window_snapshot(other, FAMILIES, start=day(2), end=day(4), as_of=day(4))
        pd.testing.assert_frame_equal(a, b)

    def test_recency_is_measured_from_the_window_end(self, dataset):
        frame, *_ = window_snapshot(dataset, FAMILIES, start=day(2), end=day(4), as_of=day(4))
        recency = frame["aml_temporal_v1::recency_days"].dropna()
        assert recency.min() >= 0.0
        assert recency.max() <= 2.0 + 1e-9  # cannot exceed the window length


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_later_window_labels_and_rows_cannot_touch_the_fit(self, frame):
        base = _dataset(frame, "dt-a")
        changed = frame.copy()
        later = changed["ts"] >= day(4)
        changed.loc[later & (changed["flag"] == 0), "flag"] = 1
        changed = pd.concat(
            [changed, pd.DataFrame([("N07", "N08", 40.0, day(5.5), 1)] * 9, columns=frame.columns)],
            ignore_index=True,
        )
        other = _dataset(changed, "dt-c")

        _Recorder.seen.clear()
        run_detection_transfer_experiment(_spec(base), base, candidate_factory=_Recorder)
        first = list(_Recorder.seen)
        _Recorder.seen.clear()
        run_detection_transfer_experiment(_spec(other), other, candidate_factory=_Recorder)
        second = list(_Recorder.seen)

        # fold 0 trains on W1 [d0, d2): nothing at or after d4 can reach it
        np.testing.assert_array_equal(first[0]["y_train"], second[0]["y_train"])
        pd.testing.assert_frame_equal(first[0]["fit_features"], second[0]["fit_features"])
        assert first[0]["edges_span"][1] < day(2)

    def test_scores_do_not_move_when_rows_after_the_test_window_change(self, frame):
        two = [["W1", str(day(0)), str(day(2))], ["W2", str(day(2)), str(day(4))]]
        base = _dataset(frame, "dt-a")
        noisy = pd.concat(
            [frame, pd.DataFrame([("N01", "N02", 9999.0, day(d), 1) for d in (4.1, 5.1, 5.7)],
                                 columns=frame.columns)],
            ignore_index=True,
        )
        other = _dataset(noisy, "dt-d")
        spec = _spec(base, split=_split(windows=two))
        a = run_detection_transfer_experiment(spec, base, candidate_factory=_Recorder)
        b = run_detection_transfer_experiment(_spec(other, split=_split(windows=two)), other,
                                              candidate_factory=_Recorder)
        assert a.folds[0].test_metrics["average_precision"] == pytest.approx(
            b.folds[0].test_metrics["average_precision"]
        )

    def test_threshold_comes_from_the_training_window_validation_partition(self, dataset):
        result = run_experiment(_spec(dataset), dataset)
        fold = result.folds[0].to_dict()
        assert fold["threshold"]["selected_on"] == "validation"
        assert fold["threshold"]["n_val_positives"] == fold["counts"]["n_val_pos"]
        assert "validation partition" in fold["diagnostics"]["risk_assessment"]["threshold_basis"]

    def test_a_fitted_calibrator_would_not_travel_with_the_frozen_model(self, dataset):
        spec = _spec(dataset, calibration=CalibrationSpec(method="platt", min_val_positives=1))
        with pytest.raises(ValueError, match="calibration.method='none'"):
            run_experiment(spec, dataset)

    def test_partitions_are_disjoint_and_cover_the_training_window(self, dataset):
        fold = run_experiment(_spec(dataset), dataset).folds[0].counts
        assert fold["n_fit"] + fold["n_val"] + fold["n_in_period"] == fold["n_train_window"]
        assert fold["n_test_seen"] + fold["n_test_new"] == fold["n_test"]

    def test_the_run_is_deterministic(self, dataset):
        a = run_experiment(_spec(dataset), dataset).to_dict()
        b = run_experiment(_spec(dataset), dataset).to_dict()
        for x, y in zip(a["folds"], b["folds"]):
            assert x["counts"] == y["counts"]
            assert x["test_metrics"]["average_precision"] == y["test_metrics"]["average_precision"]

    @pytest.mark.parametrize("scope", ["global", "train_window", "origin_window"])
    def test_other_scopes_are_refused(self, dataset, scope):
        with pytest.raises(ValueError, match="detection_window"):
            run_experiment(_spec(dataset, features=FeatureSpec(sets=FAMILIES, scope=scope)), dataset)

    def test_other_splits_refuse_the_detection_scope(self, dataset):
        spec = _spec(dataset, split=SplitSpec(strategy="stratified_kfold", params={"n_folds": 3}, seed=1))
        with pytest.raises(ValueError, match="detection_window"):
            run_experiment(spec, dataset)

    def test_graph_requiring_families_are_refused(self, dataset):
        spec = _spec(dataset, features=FeatureSpec(sets=("baseline_v1",), scope="detection_window"))
        with pytest.raises(ValueError, match="graph"):
            run_experiment(spec, dataset)

    def test_a_candidate_that_cannot_be_frozen_is_refused(self, dataset):
        class Unsafe(_Recorder):
            calendar_compatible = False

        with pytest.raises(ValueError, match="frozen"):
            run_detection_transfer_experiment(_spec(dataset), dataset, candidate_factory=Unsafe)


# ---------------------------------------------------------------------------
# Artifact and contract
# ---------------------------------------------------------------------------


class TestArtifactAndContract:
    def test_artifact_declares_regime_windows_strata_and_diagnostics(self, dataset, tmp_path):
        result = run_experiment(_spec(dataset), dataset)
        blob = result.to_dict()
        assert blob["split"]["regime"] == "detection-transfer"
        assert blob["split"]["strategy"] == DETECTION_TRANSFER_STRATEGY
        assert blob["features"]["scope"] == "detection_window"
        assert [w["name"] for w in blob["split"]["windows"]] == ["W1", "W2", "W3"]
        assert len(blob["folds"]) == 2

        fold = blob["folds"][0]
        sub = fold["subgroup_test_metrics"]
        assert set(sub) == {"in_period_reference", "seen_entities", "new_entities", "first_time_positives"}
        assert sub["in_period_reference"]["n"] == fold["counts"]["n_in_period"]
        diag = fold["diagnostics"]
        assert set(diag) >= {"windows", "drift_psi", "retention_vs_in_period", "risk_assessment"}
        assert diag["drift_psi"]["n_defined"] > 0
        assert diag["windows"]["train"]["name"] == "W1" and diag["windows"]["test"]["name"] == "W2"
        json.dumps(blob, default=str)
        assert load_result(save_result(result, tmp_path))["split"]["regime"] == "detection-transfer"

    def test_the_frozen_model_reaches_the_later_window_through_the_contract(self, dataset):
        fold = run_experiment(_spec(dataset), dataset).folds[0].to_dict()
        assessment = fold["diagnostics"]["risk_assessment"]
        assert assessment["model"]["task"] == "retrospective_detection"
        assert assessment["model"]["score_kind"] in ("model_probability", "rank_aggregate")
        assert assessment["model"]["evaluation_reference"] == "docs/r1_detection_transfer.md"
        assert "W1" in assessment["model"]["model_version"]
        assert assessment["n_entities"] == fold["counts"]["n_test"]
        assert assessment["window"]["start"] == str(day(2))
        assert assessment["window"]["end"] == str(day(4))

    @pytest.mark.parametrize(
        "family", ["structural_heuristic", "logistic_regression", "gradient_boosting"]
    )
    def test_every_baseline_runs_under_the_protocol(self, dataset, family):
        params = {"n_estimators": 15, "max_depth": 2} if family == "gradient_boosting" else {}
        spec = _spec(dataset, model=ModelSpec(family=family, params=params))
        result = run_experiment(spec, dataset)
        assert result.aggregate["average_precision"]["n_defined"] == 2

    def test_in_period_reference_and_retention_are_both_reported(self, dataset):
        fold = run_experiment(_spec(dataset), dataset).folds[0].to_dict()
        cross = fold["test_metrics"]["average_precision"]
        in_period = fold["subgroup_test_metrics"]["in_period_reference"]["metrics"]["average_precision"]
        retention = fold["diagnostics"]["retention_vs_in_period"]["average_precision"]
        if cross is not None and in_period not in (None, 0):
            assert retention == pytest.approx(cross / in_period)


class TestSnapshotSharing:
    def test_the_cache_serves_identical_blocks_and_is_dataset_scoped(self, frame, dataset):
        a, _, reused_first, _ = window_snapshot(dataset, FAMILIES, start=day(0), end=day(2), as_of=day(2))
        assert reused_first == []
        b, _, reused_second, _ = window_snapshot(dataset, FAMILIES, start=day(0), end=day(2), as_of=day(2))
        assert reused_second == list(FAMILIES)
        pd.testing.assert_frame_equal(a, b)
        clear_snapshot_cache()
        fresh, *_ = window_snapshot(dataset, FAMILIES, start=day(0), end=day(2), as_of=day(2))
        pd.testing.assert_frame_equal(a, fresh)

        scaled = frame.copy()
        scaled["amount"] *= 4.0
        other = _dataset(scaled, "dt-1")  # same name and version on purpose
        c, *_ = window_snapshot(other, FAMILIES, start=day(0), end=day(2), as_of=day(2))
        assert not c["aml_txn_v1::amount_total"].equals(a["aml_txn_v1::amount_total"])
