"""
Tests for the calendar-forward evaluation path (``src.experiments.calendar``).

Grouped by the property being protected:

* plan validity         - origins, windows and their ordering are enforced
* future-label rule     - the label is flagged involvement inside [T, T+H)
* eligibility           - only entities with feature-window history are scored
* information boundary  - no feature sees a transaction at or after its origin
* no future leakage     - nothing after the train window changes the fit
* runner contract       - scope, candidate and artifact declarations

The ledgers are tiny and hand-built.  Nothing here is a statement about
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
    build_calendar_plan,
    load_result,
    run_experiment,
    save_result,
)
from src.experiments.calendar import (
    CALENDAR_STRATEGY,
    clear_snapshot_cache,
    run_calendar_experiment,
    snapshot_features,
)
from src.experiments.candidate import Candidate, FitContext
from src.experiments.sources import SourceSpec, build_transaction_dataset

pytestmark = pytest.mark.fast

BASE = pd.Timestamp("2022-03-01")  # a Tuesday


def day(d: float) -> pd.Timestamp:
    return BASE + pd.Timedelta(days=d)


ORIGINS = [str(day(4)), str(day(6)), str(day(8))]
FAMILIES = ("aml_txn_v1", "aml_temporal_v1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(seed: int = 0) -> pd.DataFrame:
    """Twelve days of background traffic plus laundering that starts over time.

    ``R0..R3`` launder every day from day 1 (prior involvement at every origin).
    ``L0..L3`` transact normally from day 0 and first launder in the test window
    [day 8, day 10) - the new-involvement case.  ``V0``/``V1`` launder only in the
    validation window, ``T0``/``T1`` only in the train window.
    """
    rng = np.random.default_rng(seed)
    normal = [f"N{i:02d}" for i in range(40)]
    special = ["R0", "R1", "R2", "R3", "L0", "L1", "L2", "L3", "V0", "V1", "T0", "T1"]
    everyone = normal + special
    rows = []
    for d in range(12):
        for _ in range(60):
            s, t = rng.choice(len(everyone), size=2, replace=False)
            rows.append((everyone[s], everyone[t], float(rng.integers(50, 900)), day(d + rng.random()), 0))

    def launder(a: str, b: str, when: pd.Timestamp) -> None:
        rows.append((a, b, 9500.0, when, 1))

    for d in range(1, 12):
        launder(f"R{d % 4}", f"R{(d + 1) % 4}", day(d + 0.5))
    for i in range(4):
        launder(f"L{i}", f"L{(i + 1) % 4}", day(8.25 + 0.1 * i))
    launder("V0", "V1", day(6.5))
    launder("T0", "T1", day(4.5))
    return pd.DataFrame(rows, columns=["src", "dst", "amount", "ts", "flag"])


def _dataset(frame: pd.DataFrame, version: str = "cal-1") -> ExperimentDataset:
    return build_transaction_dataset(
        frame,
        SourceSpec(source="src", target="dst", label="flag", amount="amount", timestamp="ts"),
        DatasetSpec(
            name="synthetic_calendar",
            version=version,
            provenance="synthetic:test_calendar_forward",
            label_definition="entity is on at least one flagged transaction",
            synthetic=True,
        ),
    )


def _split(**params) -> SplitSpec:
    base = {"origins": ORIGINS, "horizon_days": 2}
    base.update(params)
    return SplitSpec(strategy=CALENDAR_STRATEGY, params=base, seed=1)


def _spec(dataset: ExperimentDataset, **overrides) -> ExperimentSpec:
    defaults = dict(
        name="calendar_unit",
        dataset=dataset.spec,
        features=FeatureSpec(sets=FAMILIES, scope="origin_window"),
        split=_split(),
        model=ModelSpec(family="logistic_regression"),
        threshold=ThresholdSpec(strategy="alert_budget", params={"rate": 0.1}),
        calibration=CalibrationSpec(method="none"),
        evaluation=EvaluationSpec(k_values=(5,), alert_rates=(0.1,)),
        seed=7,
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


def _labels_by_entity(plan, dataset, role) -> dict[str, int]:
    o = plan.origins[role]
    return {dataset.entity_ids[i]: int(y) for i, y in zip(o.eligible_idx, o.labels)}


class _Recorder(Candidate):
    """Label-free, snapshot-safe candidate that records what it was shown."""

    calendar_compatible = True
    emits_probabilities = False
    seen: list[dict] = []

    def fit(self, ctx: FitContext) -> "_Recorder":
        _Recorder.seen.append(
            {
                "y_train": ctx.y_train.copy(),
                "y_val": ctx.y_val.copy(),
                "train_features": ctx.features.iloc[ctx.train_idx].copy(),
                "train_edges_max": pd.to_datetime(ctx.train_edges["date"]).max(),
            }
        )
        self._fitted = True
        return self

    def predict_scores(self, ctx: FitContext, index: np.ndarray) -> np.ndarray:
        column = [c for c in ctx.features.columns if c.endswith("::txn_total")][0]
        return ctx.features[column].to_numpy(dtype=np.float64)[index]


# ---------------------------------------------------------------------------
# Plan validity
# ---------------------------------------------------------------------------


class TestPlanValidity:
    def test_origins_must_be_strictly_increasing(self, dataset):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_calendar_plan(dataset, _split(origins=[ORIGINS[1], ORIGINS[0], ORIGINS[2]]))

    def test_exactly_three_origins(self, dataset):
        with pytest.raises(ValueError, match="exactly 3 origins"):
            build_calendar_plan(dataset, _split(origins=ORIGINS[:2]))

    def test_a_label_window_must_close_before_the_next_origin(self, dataset):
        # train window [day 4, day 7) overlaps the validation origin at day 6
        with pytest.raises(ValueError, match="not over before the next origin"):
            build_calendar_plan(dataset, _split(horizon_days=3))

    def test_contiguous_windows_are_allowed(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        assert plan.origins["train"].label_end == plan.origins["val"].origin
        assert plan.origins["val"].label_end == plan.origins["test"].origin

    def test_test_window_may_not_pass_max_label_end(self, dataset):
        with pytest.raises(ValueError, match="max_label_end"):
            build_calendar_plan(dataset, _split(max_label_end=str(day(9))))
        build_calendar_plan(dataset, _split(max_label_end=str(day(10))))

    def test_unknown_parameter_is_refused(self, dataset):
        with pytest.raises(ValueError, match="unknown"):
            build_calendar_plan(dataset, _split(time_reference="last"))

    def test_nonpositive_horizon_and_lookback_are_refused(self, dataset):
        with pytest.raises(ValueError, match="horizon_days"):
            build_calendar_plan(dataset, _split(horizon_days=0))
        with pytest.raises(ValueError, match="lookback_days"):
            build_calendar_plan(dataset, _split(lookback_days=-1))

    def test_a_dataset_without_transaction_flags_is_refused(self, dataset):
        stripped = ExperimentDataset(
            spec=dataset.spec,
            entity_ids=dataset.entity_ids,
            labels=dataset.labels,
            edge_table=dataset.edge_table,
        )
        with pytest.raises(ValueError, match="transaction-level"):
            build_calendar_plan(stripped, _split())

    def test_an_origin_after_the_data_is_refused(self, dataset):
        late = [str(day(20)), str(day(22)), str(day(24))]
        with pytest.raises(ValueError, match="after the last observed transaction"):
            build_calendar_plan(dataset, _split(origins=late))

    def test_plan_declares_its_regime(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        summary = plan.to_dict()
        assert summary["regime"] == "calendar-forward"
        assert summary["time_aware"] is True
        assert summary["entity_disjoint"] is False
        assert summary["unit"] == "(entity, origin)"


# ---------------------------------------------------------------------------
# Future-label construction
# ---------------------------------------------------------------------------


class TestFutureLabels:
    def test_label_is_involvement_inside_the_window_only(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        train = _labels_by_entity(plan, dataset, "train")
        val = _labels_by_entity(plan, dataset, "val")
        test = _labels_by_entity(plan, dataset, "test")
        # T0/T1 launder at day 4.5: train window [4, 6) only
        assert train["T0"] == train["T1"] == 1
        assert val["T0"] == test["T0"] == 0
        # V0/V1 launder at day 6.5: validation window [6, 8) only
        assert val["V0"] == val["V1"] == 1
        assert train["V0"] == test["V0"] == 0
        # L* first launder at day 8.25+: test window [8, 10) only
        for name in ("L0", "L1", "L2", "L3"):
            assert test[name] == 1
            assert train[name] == 0 and val[name] == 0

    def test_window_label_differs_from_the_whole_file_label(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        whole = dict(zip(dataset.entity_ids, dataset.labels.tolist()))
        test = _labels_by_entity(plan, dataset, "test")
        # T0 is positive over the whole file, but not in the test window
        assert whole["T0"] == 1 and test["T0"] == 0

    def test_window_start_is_inclusive_and_end_exclusive(self, frame):
        extra = pd.DataFrame(
            [
                ("N00", "N01", 5000.0, day(8), 1),   # exactly at the test origin
                ("N02", "N03", 5000.0, day(10), 1),  # exactly at the test window end
            ],
            columns=frame.columns,
        )
        ds = _dataset(pd.concat([frame, extra], ignore_index=True), version="cal-edges")
        test = _labels_by_entity(build_calendar_plan(ds, _split()), ds, "test")
        assert test["N00"] == 1 and test["N01"] == 1
        assert test["N02"] == 0 and test["N03"] == 0

    def test_prior_involvement_is_recorded_but_is_not_the_label(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        o = plan.origins["test"]
        prior = {dataset.entity_ids[i]: bool(p) for i, p in zip(o.eligible_idx, o.prior_involvement)}
        assert prior["T0"] is True  # flagged before the test origin ...
        assert _labels_by_entity(plan, dataset, "test")["T0"] == 0  # ... label is 0
        assert prior["L0"] is False  # first involvement falls in the window

    def test_counts_are_consistent(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        for role in ("train", "val", "test"):
            d = plan.origins[role].to_dict()
            assert d["n_positive"] == (
                d["n_positive_with_prior_involvement"] + d["n_positive_new_involvement"]
            )
            assert d["n_window_positive"] == d["n_positive"] + d["n_cold_start_positive"]


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_entity_without_history_before_the_origin_is_not_scored(self, frame):
        extra = pd.DataFrame(
            [
                ("X0", "N05", 300.0, day(5), 0),     # first appears after the train origin
                ("Y0", "Y1", 7000.0, day(8.5), 1),   # first ever activity is laundering in test
            ],
            columns=frame.columns,
        )
        ds = _dataset(pd.concat([frame, extra], ignore_index=True), version="cal-new")
        plan = build_calendar_plan(ds, _split())
        ids = lambda role: {ds.entity_ids[i] for i in plan.origins[role].eligible_idx}  # noqa: E731
        assert "X0" not in ids("train")
        assert "X0" in ids("val") and "X0" in ids("test")
        assert "Y0" not in ids("test") and "Y1" not in ids("test")
        assert plan.origins["test"].n_cold_start_positive >= 2
        assert any("no feature-window history" in n for n in plan.notes)

    def test_lookback_restricts_eligibility_to_recent_history(self, frame):
        extra = pd.DataFrame([("OLD0", "OLD1", 100.0, day(0.5), 0)], columns=frame.columns)
        ds = _dataset(pd.concat([frame, extra], ignore_index=True), version="cal-old")
        expanding = build_calendar_plan(ds, _split())
        lookback = build_calendar_plan(ds, _split(lookback_days=3))
        i = ds.index_of("OLD0")
        assert i in expanding.origins["test"].eligible_idx
        assert i not in lookback.origins["test"].eligible_idx
        assert lookback.origins["test"].feature_start == day(5)


# ---------------------------------------------------------------------------
# Information boundary
# ---------------------------------------------------------------------------


class TestInformationBoundary:
    def test_snapshot_ignores_everything_at_or_after_its_origin(self, frame):
        plan_frame = frame.copy()
        base = _dataset(plan_frame, version="cal-a")
        future = frame["ts"] >= day(8)
        perturbed = frame.copy()
        perturbed.loc[future, "amount"] *= 10.0
        perturbed = pd.concat(
            [perturbed, pd.DataFrame([("N01", "N02", 1.0, day(8), 0)] * 25, columns=frame.columns)],
            ignore_index=True,
        )
        other = _dataset(perturbed, version="cal-b")
        for role in ("train", "val", "test"):
            a = snapshot_features(base, FAMILIES, build_calendar_plan(base, _split()).origins[role])
            b = snapshot_features(other, FAMILIES, build_calendar_plan(other, _split()).origins[role])
            pd.testing.assert_frame_equal(a, b)

    def test_recency_and_tenure_are_measured_from_each_origin(self, dataset):
        plan = build_calendar_plan(dataset, _split())
        for role in ("train", "test"):
            o = plan.origins[role]
            snap = snapshot_features(dataset, FAMILIES, o)
            dates = pd.to_datetime(dataset.edge_table["date"])
            before = dataset.edge_table[dates < o.origin]
            last = pd.concat(
                [
                    before[["source_entity", "date"]].rename(columns={"source_entity": "e"}),
                    before[["target_entity", "date"]].rename(columns={"target_entity": "e"}),
                ]
            ).assign(e=lambda f: f["e"].astype(str)).groupby("e")["date"].max()
            expected = (o.origin - last["N00"]) / pd.Timedelta(days=1)
            # the event view resolves time to the second; the ledger has
            # sub-second timestamps, so agreement is to one second
            assert snap.loc["N00", "aml_temporal_v1::recency_days"] == pytest.approx(
                expected, abs=1.0 / 86400
            )
            assert snap["aml_temporal_v1::recency_days"].min() >= 0.0

    def test_audit_checks_the_rows_actually_used(self, dataset):
        result = run_experiment(_spec(dataset), dataset)
        boundary = [f for f in result.leakage if f["check"] == "calendar_information_boundary"]
        assert len(boundary) == 3
        assert all(f["severity"] == "info" for f in boundary)
        ordering = [f for f in result.leakage if f["check"] == "calendar_label_ordering"]
        assert len(ordering) == 2 and all(f["severity"] == "info" for f in ordering)


# ---------------------------------------------------------------------------
# No future leakage into the fit
# ---------------------------------------------------------------------------


class TestNoFutureLeakage:
    def test_the_fit_does_not_move_when_later_labels_and_rows_change(self, frame):
        base = _dataset(frame, version="cal-a")
        changed = frame.copy()
        # Re-flag rows inside the validation and test label windows and add
        # post-train traffic; nothing the train snapshot or train labels depend on.
        later = changed["ts"] >= day(6)
        changed.loc[later & (changed["flag"] == 0), "flag"] = (
            np.arange(int((later & (changed["flag"] == 0)).sum())) % 7 == 0
        ).astype(int)
        changed = pd.concat(
            [changed, pd.DataFrame([("N07", "N08", 50.0, day(7), 1)] * 5, columns=frame.columns)],
            ignore_index=True,
        )
        other = _dataset(changed, version="cal-c")

        _Recorder.seen.clear()
        spec = _spec(base, threshold=ThresholdSpec(strategy="alert_budget", params={"rate": 0.1}))
        run_calendar_experiment(spec, base, candidate_factory=_Recorder)
        run_calendar_experiment(_spec(other), other, candidate_factory=_Recorder)
        first, second = _Recorder.seen
        np.testing.assert_array_equal(first["y_train"], second["y_train"])
        pd.testing.assert_frame_equal(first["train_features"], second["train_features"])
        assert first["train_edges_max"] < day(4) and second["train_edges_max"] < day(4)
        # ... while the later labels really did change
        assert not np.array_equal(first["y_val"], second["y_val"])

    def test_fit_context_exposes_train_labels_from_the_train_window(self, dataset):
        _Recorder.seen.clear()
        run_calendar_experiment(_spec(dataset), dataset, candidate_factory=_Recorder)
        plan = build_calendar_plan(dataset, _split())
        np.testing.assert_array_equal(_Recorder.seen[0]["y_train"], plan.origins["train"].labels)
        np.testing.assert_array_equal(_Recorder.seen[0]["y_val"], plan.origins["val"].labels)

    def test_transaction_flags_never_reach_the_edge_table(self, dataset):
        assert dataset.transaction_labels is not None
        assert len(dataset.transaction_labels) == len(dataset.edge_table)
        assert "flag" not in dataset.edge_table.columns
        assert int(dataset.transaction_labels.sum()) == 11 + 4 + 1 + 1

    def test_mismatched_transaction_flags_are_refused(self, dataset):
        with pytest.raises(ValueError, match="transaction_labels"):
            ExperimentDataset(
                spec=dataset.spec,
                entity_ids=dataset.entity_ids,
                labels=dataset.labels,
                edge_table=dataset.edge_table,
                transaction_labels=np.zeros(3, dtype=bool),
            )


# ---------------------------------------------------------------------------
# Runner contract
# ---------------------------------------------------------------------------


class TestRunnerContract:
    @pytest.mark.parametrize("scope", ["global", "train_window"])
    def test_calendar_split_refuses_other_scopes(self, dataset, scope):
        with pytest.raises(ValueError, match="origin_window"):
            run_experiment(_spec(dataset, features=FeatureSpec(sets=FAMILIES, scope=scope)), dataset)

    def test_other_splits_refuse_origin_window(self, dataset):
        spec = _spec(
            dataset,
            split=SplitSpec(strategy="stratified_kfold", params={"n_folds": 3}, seed=1),
        )
        with pytest.raises(ValueError, match="origin_window"):
            run_experiment(spec, dataset)

    def test_graph_requiring_families_are_refused(self, dataset):
        spec = _spec(dataset, features=FeatureSpec(sets=("baseline_v1",), scope="origin_window"))
        with pytest.raises(ValueError, match="graph"):
            run_experiment(spec, dataset)

    def test_candidates_must_opt_in(self, dataset):
        class Unsafe(_Recorder):
            calendar_compatible = False

        with pytest.raises(ValueError, match="calendar_compatible"):
            run_calendar_experiment(_spec(dataset), dataset, candidate_factory=Unsafe)

    def test_legacy_graphsage_is_not_calendar_compatible(self):
        from src.experiments.candidate import available_candidates, build_candidate

        if "legacy_graphsage" not in available_candidates():
            pytest.skip("torch backend unavailable")
        assert build_candidate("legacy_graphsage").calendar_compatible is False
        for family in ("structural_heuristic", "logistic_regression", "gradient_boosting"):
            assert build_candidate(family).calendar_compatible is True

    @pytest.mark.parametrize(
        "family", ["structural_heuristic", "logistic_regression", "gradient_boosting"]
    )
    def test_every_baseline_runs(self, dataset, family):
        params = {"n_estimators": 20, "max_depth": 2} if family == "gradient_boosting" else {}
        result = run_experiment(_spec(dataset, model=ModelSpec(family=family, params=params)), dataset)
        assert result.aggregate["average_precision"]["n_defined"] == 1

    def test_artifact_declares_the_regime_and_the_windows(self, dataset, tmp_path):
        result = run_experiment(_spec(dataset), dataset)
        blob = result.to_dict()
        assert blob["split"]["regime"] == "calendar-forward"
        assert blob["split"]["strategy"] == CALENDAR_STRATEGY
        assert blob["features"]["scope"] == "origin_window"
        assert blob["features"]["as_of"] == {
            "train": str(day(4)), "val": str(day(6)), "test": str(day(8))
        }
        fold = blob["folds"][0]
        assert fold["counts"]["n_test_pos"] == blob["split"]["origins"]["test"]["n_positive"]
        sub = fold["subgroup_test_metrics"]["no_prior_involvement"]
        assert sub["n_positive"] == 4  # L0..L3
        assert fold["threshold"]["selected_on"] == "validation"
        path = save_result(result, tmp_path)
        assert load_result(path)["split"]["regime"] == "calendar-forward"
        json.dumps(blob, default=str)

    def test_threshold_and_calibration_use_validation_only(self, dataset):
        spec = _spec(dataset, calibration=CalibrationSpec(method="platt", min_val_positives=1))
        blob = run_experiment(spec, dataset).to_dict()
        fold = blob["folds"][0]
        assert fold["threshold"]["selected_on"] == "validation"
        assert fold["threshold"]["n_val_positives"] == fold["counts"]["n_val_pos"]
        assert fold["calibration"]["fitted"] is True
        assert fold["calibration"]["diagnostics"]["n_fit"] == fold["counts"]["n_val"]


class TestDesignAudit:
    """``calendar_design_audit.py`` - label-free evidence for a design."""

    def test_psi_is_zero_for_identical_and_large_for_shifted(self):
        from calendar_design_audit import psi

        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
        assert psi(x, x + 3.0) > 0.25
        assert psi(np.full(10, np.nan), x) is None

    def test_design_report_uses_the_benchmark_rules(self, dataset):
        from calendar_design_audit import design_report

        params = {"origins": ORIGINS, "horizon_days": 2, "max_label_end": str(day(10))}
        report = design_report(dataset, params, FAMILIES, [3.0])
        assert set(report) == {"expanding", "lookback_3d"}
        plan = build_calendar_plan(dataset, _split(**params))
        assert report["expanding"]["origins"]["test"] == plan.origins["test"].to_dict()
        assert report["lookback_3d"]["origins"]["test"]["feature_window"][0] == str(day(5))
        assert report["expanding"]["psi"]["n_defined"] > 0
        assert 0.0 <= report["expanding"]["persistence_reference_ap"] <= 1.0

    def test_origin_scan_counts_labels_beyond_the_supported_period(self, dataset):
        from calendar_design_audit import origin_scan
        from src.experiments.calendar import ledger_arrays

        rows = origin_scan(ledger_arrays(dataset), [2.0], day(9))
        by_origin = {r["origin"]: r for r in rows}
        # window [day 8, day 10) straddles day 9: the R-ring flag at day 9.5
        # falls after max_label_end
        assert by_origin[str(day(8))]["n_positive_only_after_max_label_end"] >= 1
        assert by_origin[str(day(4))]["n_positive_only_after_max_label_end"] == 0

    def test_daily_table_counts_every_row(self, dataset):
        from calendar_design_audit import daily_table
        from src.experiments.calendar import ledger_arrays

        table = daily_table(dataset, ledger_arrays(dataset))
        assert sum(r["transactions"] for r in table) == len(dataset.edge_table)
        assert sum(r["flagged"] for r in table) == int(dataset.transaction_labels.sum())


class TestSnapshotReuse:
    def test_reused_blocks_equal_a_fresh_computation(self, dataset):
        spec = _spec(dataset)
        first = run_experiment(spec, dataset).to_dict()
        assert first["features"]["snapshot_reuse"]["test"] == []
        plan = build_calendar_plan(dataset, _split())
        cached = snapshot_features(dataset, FAMILIES, plan.origins["test"])
        second = run_experiment(
            _spec(dataset, features=FeatureSpec(sets=("aml_txn_v1",), scope="origin_window")),
            dataset,
        ).to_dict()
        assert second["features"]["snapshot_reuse"]["test"] == ["aml_txn_v1"]
        clear_snapshot_cache()
        fresh = snapshot_features(dataset, FAMILIES, plan.origins["test"])
        pd.testing.assert_frame_equal(cached, fresh)

    def test_a_different_dataset_never_reads_the_cache(self, frame, dataset):
        plan = build_calendar_plan(dataset, _split())
        snapshot_features(dataset, FAMILIES, plan.origins["test"])
        scaled = frame.copy()
        scaled["amount"] *= 3.0
        other = _dataset(scaled, version="cal-1")  # same name and version on purpose
        a = snapshot_features(other, FAMILIES, build_calendar_plan(other, _split()).origins["test"])
        assert not a["aml_txn_v1::amount_total"].equals(
            snapshot_features(dataset, FAMILIES, plan.origins["test"])["aml_txn_v1::amount_total"]
        )
