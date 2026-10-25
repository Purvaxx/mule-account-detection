"""
Tests for ``aml_temporal_v2`` - window-relative timing (R6).

Grouped by the property being protected:

* registry        - a separate family, not a silent redefinition of v1
* definitions     - hand-computed values on a tiny ledger
* invariance      - the property v2 exists for: a monotone time warp of the
                    window leaves every activity-clock feature unchanged, while
                    v1's calendar-day features move
* independence    - no ``as_of``, no labels, nothing outside the window
* conventions     - bounds, and undefined stays NaN

Design and rationale: ``docs/r6_temporal_v2.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.features import FeatureRequest, build_feature_matrix, get_feature_set

pytestmark = pytest.mark.fast

BASE = pd.Timestamp("2022-01-03")  # a Monday
V2 = "aml_temporal_v2"
V1 = "aml_temporal_v1"


def _edges(hours: list[float], pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_entity": [p[0] for p in pairs],
            "target_entity": [p[1] for p in pairs],
            "date": [BASE + pd.Timedelta(hours=h) for h in hours],
            "amount": [10.0 * (i + 1) for i in range(len(hours))],
        }
    )


def _matrix(edges: pd.DataFrame, entities: list[str], sets, as_of=None) -> pd.DataFrame:
    request = FeatureRequest(
        entity_ids=entities, edge_table=edges, graph=None, as_of=as_of
    )
    return build_feature_matrix(request, sets).values


#: Five transactions, so tau is 0, 0.2, 0.4, 0.6, 0.8 in time order.
HOURS = [0.0, 3.0, 30.0, 40.0, 47.0]
PAIRS = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A"), ("A", "B")]
ENTITIES = ["A", "B", "C"]


class TestRegistry:
    def test_v2_is_a_separate_registered_family(self):
        fs = get_feature_set(V2)
        assert (fs.version, fs.family) == ("2.0", "temporal")
        assert fs.time_scopable and not fs.requires_graph
        assert get_feature_set(V1).version == "1.0"  # v1 untouched

    def test_v2_replaces_the_position_columns_and_drops_weekend_share(self):
        columns = {
            str(c).split("::")[-1]
            for c in _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2]).columns
        }
        assert len(columns) == 15
        # redefined quantities are renamed, so a v1 and a v2 number can never be
        # read as the same thing
        for gone in ("recency_days", "tenure_days", "span_days", "daily_concentration",
                     "max_daily_txn", "active_days", "second_half_share",
                     "max_gap_days", "txn_per_active_day", "txn_per_span_day"):
            assert gone not in columns
        assert "weekend_share" not in columns
        for kept in ("mean_iat_hours", "std_iat_hours", "min_iat_hours",
                     "burstiness", "night_share"):
            assert kept in columns


class TestDefinitions:
    def test_hand_computed_activity_clock_features(self):
        m = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2])
        # A acts on transactions 1, 2, 4, 5 -> tau 0.0, 0.2, 0.6, 0.8
        assert m.loc["A", f"{V2}::tenure_activity"] == pytest.approx(1.0)
        assert m.loc["A", f"{V2}::recency_activity"] == pytest.approx(0.2)
        assert m.loc["A", f"{V2}::span_activity"] == pytest.approx(0.8)
        assert m.loc["A", f"{V2}::active_bin_fraction"] == pytest.approx(0.4)
        assert m.loc["A", f"{V2}::activity_concentration"] == pytest.approx(0.25)
        assert m.loc["A", f"{V2}::window_half_share"] == pytest.approx(0.5)
        assert m.loc["A", f"{V2}::max_gap_activity"] == pytest.approx(0.4)
        # C acts on transactions 2, 3, 4 -> tau 0.2, 0.4, 0.6
        assert m.loc["C", f"{V2}::recency_activity"] == pytest.approx(0.4)
        assert m.loc["C", f"{V2}::tenure_activity"] == pytest.approx(0.8)
        assert m.loc["C", f"{V2}::span_activity"] == pytest.approx(0.4)

    def test_wall_clock_velocity_is_still_wall_clock(self):
        m = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2])
        # A's wall-clock gaps: 3h, 37h, 7h -> mean 15.667h
        assert m.loc["A", f"{V2}::mean_iat_hours"] == pytest.approx(47.0 / 3.0)


class TestDesignedInvariance:
    """The property v2 exists for.

    A *monotone time warp* keeps every transaction's position in the window's
    activity ordering and changes only where the transactions sit on the wall
    clock - exactly the difference between a Wed+Thu window (even volume) and a
    Fri+Sat one (3.1:1).  Every activity-clock feature must be unchanged.
    """

    WARPED = [0.0, 1.0, 2.0, 3.0, 4.0]  # all five squeezed into the first hours

    def test_activity_clock_features_are_invariant_to_the_warp(self):
        a = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2])
        b = _matrix(_edges(self.WARPED, PAIRS), ENTITIES, [V2])
        clock = [
            "active_bin_fraction", "span_activity", "txn_per_active_bin",
            "txn_per_span_activity", "max_gap_activity", "max_bin_txn",
            "activity_concentration", "window_half_share", "recency_activity",
            "tenure_activity",
        ]
        for column in clock:
            pd.testing.assert_series_equal(
                a[f"{V2}::{column}"], b[f"{V2}::{column}"], check_names=False
            )

    def test_v1_calendar_features_are_not_invariant_to_the_warp(self):
        """The contrast that motivates v2 - v1 moves where v2 does not."""
        a = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V1])
        b = _matrix(_edges(self.WARPED, PAIRS), ENTITIES, [V1])
        moved = {
            str(c).split("::")[-1]
            for c in a.columns
            if not np.allclose(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True)
        }
        assert {"active_days", "span_days", "daily_concentration"} <= moved

    def test_wall_clock_features_do_move_as_declared(self):
        a = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2])
        b = _matrix(_edges(self.WARPED, PAIRS), ENTITIES, [V2])
        assert a.loc["A", f"{V2}::mean_iat_hours"] != pytest.approx(
            b.loc["A", f"{V2}::mean_iat_hours"]
        )


class TestIndependence:
    def test_v2_does_not_depend_on_as_of(self):
        edges = _edges(HOURS, PAIRS)
        none = _matrix(edges, ENTITIES, [V2], as_of=None)
        end = _matrix(edges, ENTITIES, [V2], as_of=BASE + pd.Timedelta(days=2))
        far = _matrix(edges, ENTITIES, [V2], as_of=pd.Timestamp("2030-01-01"))
        pd.testing.assert_frame_equal(none, end)
        pd.testing.assert_frame_equal(none, far)

    def test_v1_does_depend_on_as_of(self):
        """Stated for contrast: this is the dependence v2 removes."""
        edges = _edges(HOURS, PAIRS)
        end = _matrix(edges, ENTITIES, [V1], as_of=BASE + pd.Timedelta(days=2))
        far = _matrix(edges, ENTITIES, [V1], as_of=pd.Timestamp("2030-01-01"))
        assert not np.allclose(
            end[f"{V1}::recency_days"].to_numpy(), far[f"{V1}::recency_days"].to_numpy()
        )

    def test_rows_outside_the_window_cannot_reach_a_v2_feature(self):
        """The window-locality R1 relies on, exercised through the snapshot path."""
        from src.experiments import DatasetSpec
        from src.experiments.snapshots import clear_snapshot_cache, window_snapshot
        from src.experiments.sources import SourceSpec, build_transaction_dataset

        inside = pd.DataFrame(
            {
                "src": ["A", "A", "B", "C"],
                "dst": ["B", "C", "C", "A"],
                "amount": [10.0, 20.0, 30.0, 40.0],
                "ts": [BASE + pd.Timedelta(hours=h) for h in (1, 5, 20, 30)],
                "flag": [0, 0, 0, 0],
            }
        )
        outside = pd.DataFrame(
            {
                "src": ["A", "B"],
                "dst": ["C", "A"],
                "amount": [9999.0, 8888.0],
                "ts": [BASE - pd.Timedelta(hours=5), BASE + pd.Timedelta(days=3)],
                "flag": [1, 1],
            }
        )

        def dataset(frame, version):
            return build_transaction_dataset(
                frame,
                SourceSpec(source="src", target="dst", label="flag", amount="amount", timestamp="ts"),
                DatasetSpec(name="v2_window", version=version, provenance="synthetic",
                            label_definition="flagged", synthetic=True),
            )

        start, end = BASE, BASE + pd.Timedelta(days=2)
        clear_snapshot_cache()
        a, *_ = window_snapshot(dataset(inside, "a"), [V2], start=start, end=end, as_of=end)
        b, *_ = window_snapshot(
            dataset(pd.concat([inside, outside], ignore_index=True), "b"),
            [V2], start=start, end=end, as_of=end,
        )
        pd.testing.assert_frame_equal(a, b)

    def test_v2_reads_no_labels(self):
        """Flipping every transaction label leaves the features identical."""
        from src.experiments import DatasetSpec
        from src.experiments.sources import SourceSpec, build_transaction_dataset

        frame = pd.DataFrame(
            {
                "src": ["A", "A", "B"],
                "dst": ["B", "C", "C"],
                "amount": [10.0, 20.0, 30.0],
                "ts": [BASE + pd.Timedelta(hours=h) for h in (1, 4, 9)],
                "flag": [0, 0, 0],
            }
        )
        flipped = frame.assign(flag=[1, 1, 1])
        out = []
        for f, v in ((frame, "clean"), (flipped, "flagged")):
            ds = build_transaction_dataset(
                f,
                SourceSpec(source="src", target="dst", label="flag", amount="amount", timestamp="ts"),
                DatasetSpec(name="v2_labels", version=v, provenance="synthetic",
                            label_definition="flagged", synthetic=True),
            )
            out.append(_matrix(ds.edge_table, list(ds.entity_ids), [V2]))
        pd.testing.assert_frame_equal(out[0], out[1])


class TestConventions:
    def test_shares_and_fractions_stay_within_bounds(self):
        m = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2])
        for column in ("recency_activity", "tenure_activity", "span_activity",
                       "window_half_share", "activity_concentration",
                       "active_bin_fraction", "max_gap_activity"):
            values = m[f"{V2}::{column}"].dropna().to_numpy()
            assert values.min() >= 0.0 and values.max() <= 1.0, column

    def test_undefined_statistics_stay_nan(self):
        edges = _edges([0.0], [("A", "B")])
        m = _matrix(edges, ["A", "B", "Z"], [V2])
        # a single transaction defines no inter-arrival time and no gap
        assert np.isnan(m.loc["A", f"{V2}::mean_iat_hours"])
        assert np.isnan(m.loc["A", f"{V2}::max_gap_activity"])
        assert m.loc["A", f"{V2}::span_activity"] == pytest.approx(0.0)
        # an entity with no transaction in the window has nothing defined
        assert m.loc["Z"].isna().all()

    def test_undated_rows_do_not_enter_the_clock(self):
        edges = _edges(HOURS, PAIRS)
        edges.loc[len(edges)] = {
            "source_entity": "A", "target_entity": "B", "date": pd.NaT, "amount": 1.0
        }
        with_undated = _matrix(edges, ENTITIES, [V2])
        without = _matrix(_edges(HOURS, PAIRS), ENTITIES, [V2])
        for column in ("recency_activity", "tenure_activity", "span_activity",
                       "activity_concentration"):
            pd.testing.assert_series_equal(
                with_undated[f"{V2}::{column}"], without[f"{V2}::{column}"],
                check_names=False,
            )
