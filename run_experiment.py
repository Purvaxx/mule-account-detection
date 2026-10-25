"""
Experiment CLI
==============

Runs one or more experiment specifications and writes an artifact for each.

    python run_experiment.py configs/experiments/heuristic_floor.json
    python run_experiment.py configs/experiments/*.json --out outputs/experiments
    python run_experiment.py configs/experiments/samld_*.json
    python run_experiment.py --list

Which dataset a config runs on is the ``dataset.name`` field of the config
itself, resolved through ``src/experiments/sources.py``.  A dataset is loaded
once per invocation however many configs name it, because loading SAML-D is a
minute and the point of a batch is to amortise that.

Every experiment-specific constant lives in the JSON specification; this script
holds none.  Adding a model family means adding a candidate module and a config
file, not editing this file.

The console output is a short honest summary per run.  It does not rank the
runs and does not name a winner: comparing artifacts is a separate, deliberate
step that has to account for which folds could define which metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.experiments import ExperimentSpec, run_experiment
from src.experiments.calendar import CALENDAR_STRATEGY
from src.experiments.candidate import available_candidates
from src.experiments.features import available_feature_sets
from src.experiments.sources import available_sources, load_source
from src.experiments.splits import available_split_strategies
from src.experiments.thresholds import available_threshold_strategies

DEFAULT_OUTPUT_DIR = Path("outputs/experiments")


def _existing_artifact(output_dir: Path, name: str, dataset_version: str) -> Path | None:
    """An artifact in ``output_dir`` for this spec name and dataset version.

    Matching on the *dataset version* as well as the name is what makes this
    safe: a result produced from a different file is not a result for this run,
    and skipping on the name alone would silently keep a stale number.
    """
    if not output_dir.exists():
        return None
    for path in sorted(output_dir.glob(f"{name}__*.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("experiment", {}).get("name") == name
            and payload.get("dataset", {}).get("spec", {}).get("version") == dataset_version
        ):
            return path
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="experiment spec JSON file(s)")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"artifact output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="persist each fold's model here and record its digest",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "override the dataset every config names "
            f"(one of: {', '.join(available_sources())})"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="override the dataset file path passed to the loader",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip a config whose artifact for this dataset version is already "
            "in --out; makes a long batch resumable after an interruption"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the registered candidates, splits, feature sets and thresholds",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _print_registry() -> None:
    print("dataset sources       :", ", ".join(available_sources()))
    print("candidate families    :", ", ".join(available_candidates()))
    print("split strategies      :", ", ".join(available_split_strategies()))
    print("calendar strategy     :", CALENDAR_STRATEGY, "(feature scope 'origin_window')")
    print("feature sets          :", ", ".join(available_feature_sets()))
    print("threshold strategies  :", ", ".join(available_threshold_strategies()))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.list:
        _print_registry()
        return 0

    if not args.configs:
        _build_parser().print_help()
        return 2

    loaded: dict[str, object] = {}

    def dataset_for(source_name: str):
        """Load ``source_name`` once and reuse it for the rest of the batch."""
        if source_name not in loaded:
            kwargs = {}
            if args.dataset is not None:
                kwargs["path"] = args.dataset
            dataset = load_source(source_name, **kwargs)
            print(f"dataset: {dataset.spec.name} v{dataset.spec.version}")
            print(
                f"         {dataset.n_entities:,} entities, "
                f"{dataset.n_positive:,} positive ({dataset.prevalence:.4%}), "
                f"{0 if dataset.edge_table is None else len(dataset.edge_table):,} "
                f"transactions"
                + ("  [SYNTHETIC]" if dataset.spec.synthetic else "")
            )
            loaded[source_name] = dataset
        return loaded[source_name]

    failures = 0
    for config_path in args.configs:
        with config_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        source_name = args.source or payload["dataset"]["name"]
        try:
            dataset = dataset_for(source_name)
        except (FileNotFoundError, KeyError) as exc:
            failures += 1
            print(f"{payload.get('name', config_path.name)}: SKIPPED ({exc})", file=sys.stderr)
            continue

        # The dataset block in a config names the intent; the loader supplies the
        # verified identity.  Trusting the file's own version string would let a
        # stale config mislabel an artifact.
        payload["dataset"] = dataset.spec.to_dict()
        spec = ExperimentSpec.from_dict(payload)

        print()
        print("=" * 72)
        if args.skip_existing:
            existing = _existing_artifact(args.out, spec.name, dataset.spec.version)
            if existing is not None:
                print(f"{spec.name}: skipped, already in {existing.name}")
                continue
        try:
            result = run_experiment(
                spec,
                dataset,
                output_dir=args.out,
                checkpoint_dir=args.checkpoints,
            )
        except Exception as exc:  # noqa: BLE001 - a failed run must not stop the batch
            failures += 1
            print(f"{spec.name}: FAILED ({type(exc).__name__}: {exc})")
            if args.verbose:
                raise
            continue

        print(result.headline())
        for note in result.notes:
            print(f"  note   : {note}")

    print()
    print("=" * 72)
    print(f"artifacts written to {args.out.resolve()}")
    if failures:
        print(f"{failures} experiment(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
