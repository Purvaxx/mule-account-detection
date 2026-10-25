"""
Experiment Comparison CLI
=========================

Reads experiment artifacts and renders one comparison table per
(dataset, evaluation regime, feature scope) group.

    python compare_experiments.py outputs/experiments/samld_*.json
    python compare_experiments.py outputs/experiments/*.json --report docs/model_selection_table.md

Why it groups
-------------
A transductive number and a time-forward number answer different questions, and
a run whose features saw the whole population is not comparable with one whose
features stopped at a training boundary.  Putting them in one column would be
the single most misleading thing this project could do with its own results, so
the grouping is enforced here rather than left to whoever writes the table.

What it refuses to do
---------------------
It does not name a winner.  It prints the mean, the dispersion, the fold-level
values, the positive count each fold's metric was computed from, and how many
folds could define the metric at all.  A metric that no fold could define is
printed as ``undefined``, never as zero.

It also does not compare across dataset versions.  Two artifacts carrying
different ``dataset.version`` strings were produced from different files and are
placed in different groups.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

PRIMARY = [
    ("average_precision", "AP"),
    ("ap_lift", "lift"),
    ("roc_auc", "ROC"),
    ("precision_at_rate.0.01", "P@1%"),
    ("recall_at_rate.0.01", "R@1%"),
    ("at_threshold.false_positive_burden", "FP/TP"),
    ("at_threshold.alert_rate", "alert rate"),
]


def _get(aggregate: dict[str, Any], key: str) -> Optional[dict[str, Any]]:
    return aggregate.get(key)


def _cell(aggregate: dict[str, Any], key: str) -> str:
    block = _get(aggregate, key)
    if block is None or block.get("mean") is None or block.get("n_defined", 0) == 0:
        return "undefined"
    mean = block["mean"]
    std = block.get("std")
    # One precision per cell, chosen from the magnitude of the mean, so a
    # value and its dispersion are never printed at different scales.
    digits = 4 if abs(mean) < 100 else 1
    text = f"{mean:.{digits}f}"
    if std is not None and block.get("n_folds", 1) > 1:
        text += f" ±{std:.{digits}f}"
    if block.get("n_defined") != block.get("n_folds"):
        text += f" ({block['n_defined']}/{block['n_folds']} folds)"
    return text


def _fold_values(artifact: dict[str, Any], key: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for fold in artifact["folds"]:
        node: Any = fold["test_metrics"]
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        out.append(node if isinstance(node, (int, float)) else None)
    return out


def _group_key(artifact: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        artifact["dataset"]["spec"]["name"],
        artifact["dataset"]["spec"]["version"],
        artifact["split"]["regime"],
        artifact["features"].get("scope", "n/a"),
    )


def _render_group(key: tuple[str, str, str, str], artifacts: list[dict[str, Any]]) -> list[str]:
    name, version, regime, scope = key
    synthetic = artifacts[0]["dataset"]["spec"]["synthetic"]
    lines = [
        f"### `{name}` — {regime} regime, `{scope}` feature scope",
        "",
        f"Dataset version `{version}`. "
        + (
            "**SYNTHETIC — not evidence about real-world performance.**"
            if synthetic
            else "Real, observed data."
        ),
        "",
    ]
    header = "| experiment | model | features | seed | folds | test positives/fold | " + " | ".join(
        label for _, label in PRIMARY
    ) + " |"
    lines.append(header)
    lines.append("| --- | --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in PRIMARY) + " |")

    for artifact in sorted(artifacts, key=lambda a: a["experiment"]["name"]):
        experiment = artifact["experiment"]
        aggregate = artifact["aggregate"]
        positives = aggregate.get("n_positive", {})
        pos_text = (
            f"{positives['mean']:.0f}" if positives.get("mean") is not None else "—"
        )
        features = ", ".join(
            block["name"] for block in artifact["features"].get("sets", [])
        ) or "none"
        lines.append(
            f"| `{experiment['name']}` "
            f"| {experiment['model']['family']} "
            f"| {features} "
            f"| {experiment['seed']} "
            f"| {len(artifact['folds'])} "
            f"| {pos_text} "
            + "".join(f"| {_cell(aggregate, key)} " for key, _ in PRIMARY)
            + "|"
        )

    lines.append("")
    lines.append("Fold-level average precision:")
    lines.append("")
    for artifact in sorted(artifacts, key=lambda a: a["experiment"]["name"]):
        values = _fold_values(artifact, "average_precision")
        rendered = ", ".join("undefined" if v is None else f"{v:.4f}" for v in values)
        lines.append(f"* `{artifact['experiment']['name']}`: {rendered}")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, default=None, help="also write Markdown here")
    args = parser.parse_args(argv)

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in args.artifacts:
        with path.open("r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        groups[_group_key(artifact)].append(artifact)

    lines = [
        "<!-- GENERATED by compare_experiments.py. Do not edit by hand. -->",
        "",
        "Every table below is one dataset version under one evaluation regime "
        "and one feature scope. Rows within a table are comparable; rows in "
        "different tables are not.",
        "",
    ]
    for key in sorted(groups):
        lines.extend(_render_group(key, groups[key]))

    text = "\n".join(lines)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten to {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
