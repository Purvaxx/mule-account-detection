"""
Calendar-Forward Design Audit
=============================

Label-free evidence for the choice of origins, horizon and feature history of a
``calendar_forward`` benchmark.  The design is read from an existing calendar
experiment config, so the audit and the benchmark cannot drift apart:

    python calendar_design_audit.py configs/experiments/ibm_cal_boosting_graph.json

Writes ``outputs/audits/calendar_design_<dataset>.json``:

``daily``        per calendar day: transactions, flagged transactions, flagged
                 rate, active entities, and the payment-format mix of the
                 flagged rows when the dataset carries a payment format
``origin_scan``  every midnight origin x each ``--horizons`` value: eligible
                 entities, window positives, how many were already involved
                 before the origin, how many are cold-start, and how many draw
                 their label only from after ``max_label_end``
``designs``      the configured origins under expanding history and under each
                 ``--lookbacks`` value: population, positives, and the feature
                 distribution shift (PSI) between the train and test snapshots
``persistence``  average precision, at the test origin, of "already involved
                 before the origin" - a label-using reference, never a candidate

No model is fitted and no model score is read, so nothing here can steer a
design toward a better benchmark number.  PSI uses ten quantile bins fitted on
the train snapshot's eligible rows, undefined values excluded, bin shares
floored at 1e-6.  The origin resolution and the snapshots are the calendar
module's own (``resolve_origin``, ``snapshot_features``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.calendar import (
    CALENDAR_STRATEGY,
    ROLES,
    LedgerArrays,
    build_calendar_plan,
    clear_snapshot_cache,
    ledger_arrays,
    resolve_origin,
    snapshot_features,
)
from src.experiments.dataset import ExperimentDataset
from src.experiments.spec import SplitSpec

DEFAULT_OUTPUT_DIR = Path("outputs/audits")
_FORMAT_COLUMNS = ("Payment Format", "Payment_type", "payment_type")


#: The project's one PSI definition lives in ``src/experiments/drift.py`` so that
#: a drift number from this audit and one from a cross-period experiment mean the
#: same thing.  Re-exported here because this script's callers import it.
from src.experiments.drift import psi  # noqa: E402


def daily_table(dataset: ExperimentDataset, ledger: LedgerArrays) -> list[dict[str, Any]]:
    days = pd.DatetimeIndex(ledger.dates).floor("D")
    frame = pd.DataFrame({"day": days, "flag": ledger.flags})
    out = frame.groupby("day").agg(transactions=("flag", "size"), flagged=("flag", "sum"))
    out["flagged_rate"] = out["flagged"] / out["transactions"]
    both = pd.DataFrame(
        {"day": np.concatenate([days, days]), "e": np.concatenate([ledger.src, ledger.dst])}
    )
    out["active_entities"] = both.groupby("day")["e"].nunique()
    column = next((c for c in _FORMAT_COLUMNS if c in dataset.edge_table.columns), None)
    if column is not None:
        flagged = pd.DataFrame(
            {
                "day": days[ledger.flags],
                "fmt": dataset.edge_table[column].astype(str).to_numpy()[ledger.flags],
            }
        )
        mix = flagged.groupby(["day", "fmt"]).size().unstack(fill_value=0)
        shares = mix.div(mix.sum(axis=1), axis=0)
        out["flagged_format_share"] = [
            {k: round(float(v), 4) for k, v in shares.loc[d].items() if v > 0}
            if d in shares.index
            else {}
            for d in out.index
        ]
    records = out.reset_index()
    records["day"] = records["day"].astype(str)
    return records.to_dict(orient="records")


def origin_scan(
    ledger: LedgerArrays,
    horizons: Sequence[float],
    max_label_end: Optional[pd.Timestamp],
) -> list[dict[str, Any]]:
    """Every midnight origin after the first day, for each horizon."""
    dated = ~np.isnat(ledger.dates)
    origin = pd.Timestamp(ledger.dates[dated].min()).floor("D") + pd.Timedelta(days=1)
    last = ledger.last_observed
    rows: list[dict[str, Any]] = []
    while origin <= last:
        for h in horizons:
            horizon = pd.Timedelta(days=float(h))
            full = resolve_origin(ledger, "scan", origin, horizon)
            summary = full.to_dict()
            row = {
                "origin": str(origin),
                "horizon_days": float(h),
                "label_window_end": str(full.label_end),
                **{
                    k: summary[k]
                    for k in (
                        "n_eligible", "n_positive", "prevalence",
                        "n_positive_with_prior_involvement", "n_positive_new_involvement",
                        "n_cold_start_positive",
                    )
                },
            }
            if max_label_end is not None:
                if origin >= max_label_end:
                    beyond = full.n_positive
                elif full.label_end > max_label_end:
                    before = resolve_origin(ledger, "scan", origin, max_label_end - origin)
                    beyond = int((full.labels.astype(bool) & ~before.labels.astype(bool)).sum())
                else:
                    beyond = 0
                row["n_positive_only_after_max_label_end"] = beyond
            rows.append(row)
        origin += pd.Timedelta(days=1)
    return rows


def design_report(
    dataset: ExperimentDataset,
    split_params: dict[str, Any],
    sets: Sequence[str],
    lookbacks: Sequence[float],
) -> dict[str, Any]:
    """Population, positives and train-vs-test feature shift for each design."""
    from sklearn.metrics import average_precision_score

    out: dict[str, Any] = {}
    for lookback in [None, *lookbacks]:
        name = "expanding" if lookback is None else f"lookback_{lookback:g}d"
        params = dict(split_params, lookback_days=lookback)
        plan = build_calendar_plan(dataset, SplitSpec(strategy=CALENDAR_STRATEGY, params=params))
        train, test = plan.origins["train"], plan.origins["test"]
        a = snapshot_features(dataset, sets, train).take(train.eligible_idx)
        b = snapshot_features(dataset, sets, test).take(test.eligible_idx)
        clear_snapshot_cache()  # bound memory: each design has its own windows

        rows = []
        for column in a.columns:
            x = a[column].to_numpy(dtype=np.float64)
            y = b[column].to_numpy(dtype=np.float64)
            value = psi(x, y)
            if value is None:
                continue
            rows.append(
                {
                    "feature": str(column),
                    "psi": round(value, 4),
                    "train_median": float(np.nanmedian(x)),
                    "test_median": float(np.nanmedian(y)),
                }
            )
        table = pd.DataFrame(rows)
        by_family = (
            table.assign(family=table["feature"].str.split("::").str[0])
            .groupby("family")["psi"]
            .median()
            .round(4)
            .to_dict()
            if len(table)
            else {}
        )
        labels = test.labels
        persistence = (
            float(average_precision_score(labels, test.prior_involvement.astype(float)))
            if 0 < labels.sum() < len(labels)
            else None
        )
        out[name] = {
            "lookback_days": lookback,
            "origins": {r: plan.origins[r].to_dict() for r in ROLES},
            "psi": {
                "n_defined": int(len(table)),
                "n_above_0_25": int((table["psi"] > 0.25).sum()) if len(table) else 0,
                "n_above_0_10": int((table["psi"] > 0.10).sum()) if len(table) else 0,
                "median": float(table["psi"].median()) if len(table) else None,
                "median_by_family": by_family,
                "top": table.sort_values("psi", ascending=False).head(12).to_dict(orient="records")
                if len(table)
                else [],
            },
            "persistence_reference_ap": persistence,
            "notes": list(plan.notes),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="a calendar_forward experiment config")
    parser.add_argument("--lookbacks", type=float, nargs="*", default=[3.0, 4.0])
    parser.add_argument("--horizons", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path, default=None, help="override the dataset file path")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    from src.experiments.sources import load_source

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if payload["split"]["strategy"] != CALENDAR_STRATEGY:
        parser.error(f"{args.config} is not a {CALENDAR_STRATEGY} config")
    kwargs = {"path": args.dataset} if args.dataset is not None else {}
    dataset = load_source(payload["dataset"]["name"], **kwargs)
    split_params = dict(payload["split"]["params"])
    max_end = split_params.get("max_label_end")
    ledger = ledger_arrays(dataset)

    report = {
        "tool": "calendar_design_audit.py",
        "config": str(args.config),
        "dataset": dataset.spec.to_dict(),
        "split_params": split_params,
        "feature_sets": list(payload["features"]["sets"]),
        "daily": daily_table(dataset, ledger),
        "origin_scan": origin_scan(
            ledger, args.horizons, None if max_end is None else pd.Timestamp(max_end)
        ),
        "designs": design_report(dataset, split_params, payload["features"]["sets"], args.lookbacks),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"calendar_design_{dataset.spec.name}.json"
    path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    for name, design in report["designs"].items():
        te = design["origins"]["test"]
        print(
            f"{name:14s} test eligible {te['n_eligible']:,}  positives {te['n_positive']:,}  "
            f"cold-start excluded {te['n_cold_start_positive']:,}  "
            f"PSI>0.25 {design['psi']['n_above_0_25']}/{design['psi']['n_defined']}  "
            f"median PSI {design['psi']['median']:.4f}"
        )
    print(f"written to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
