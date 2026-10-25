"""
Tests for the dataset audit layer (``src.experiments.audit``) and the external
benchmark loaders (``src.experiments.sources``).

Grouped by the property being protected:

* loader honesty     - the label rule is applied as written, alias columns are
                       refused, amounts are never renamed into a currency they
                       are not in
* audit arithmetic   - the blocks report what is in the data, and report
                       ``None`` with a reason where a quantity is undefined
* verdict logic      - each classification is reachable and is reached for the
                       stated reason

Every fixture is a hand-built table whose answers are known by construction.
Nothing here makes a claim about a real dataset; the real datasets are audited
by ``audit_datasets.py`` and the findings live in ``docs/datasets.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.audit import (
    BLOCKED,
    DOMAIN_AML,
    DOMAIN_CARD_FRAUD,
    MIN_POSITIVES_FOR_CV,
    MODEL_SELECTION_READY,
    REGRESSION_ONLY,
    STRUCTURAL_VALIDATION_ONLY,
    audit_dataset,
    save_audit,
)
from src.experiments.dataset import ExperimentDataset
from src.experiments.sources import (
    SourceSpec,
    amount_column,
    build_transaction_dataset,
    entity_codes,
)
from src.experiments.spec import DatasetSpec

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _spec(name: str = "fixture", synthetic: bool = True) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        version="v0",
        provenance="test fixture",
        label_definition=(
            "positive = the account appears on at least one transaction flagged "
            "by the publisher"
        ),
        synthetic=synthetic,
    )


def _raw_frame(
    n_normal: int = 400,
    n_flagged_rings: int = 15,
    ring_size: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    """Normal star traffic plus flagged rings, in the source file's own schema.

    Normal accounts pay a hub; ring accounts pay each other round a loop.  The
    rings are labelled.  The point of the shape is that the two populations are
    distinguishable but not identical - it is a fixture, not a benchmark.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2023-01-01")
    rows: list[dict] = []
    for i in range(n_normal):
        for _ in range(int(rng.integers(2, 6))):
            rows.append(
                {
                    "sender": f"N{i:05d}",
                    "receiver": f"HUB{i % 20:02d}",
                    "amt": float(rng.uniform(10, 900)),
                    "when": base + pd.Timedelta(days=int(rng.integers(0, 300))),
                    "flag": 0,
                    "mechanism": "normal_star",
                }
            )
    for r in range(n_flagged_rings):
        members = [f"R{r:03d}_{j}" for j in range(ring_size)]
        for j, member in enumerate(members):
            rows.append(
                {
                    "sender": member,
                    "receiver": members[(j + 1) % ring_size],
                    "amt": float(rng.uniform(4000, 6000)),
                    "when": base + pd.Timedelta(days=int(rng.integers(0, 300))),
                    "flag": 1,
                    "mechanism": "ring",
                }
            )
    return pd.DataFrame(rows)


def _source(**overrides) -> SourceSpec:
    params = dict(
        source="sender",
        target="receiver",
        label="flag",
        amount="amt",
        amount_is_usd=False,
        timestamp="when",
        attribute_columns=(),
    )
    params.update(overrides)
    return SourceSpec(**params)


@pytest.fixture
def benchmark() -> ExperimentDataset:
    return build_transaction_dataset(_raw_frame(), _source(), _spec())


# ---------------------------------------------------------------------------
# Loader honesty
# ---------------------------------------------------------------------------

class TestLoaderHonesty:
    def test_entity_label_is_the_declared_rule(self, benchmark):
        """A positive is exactly an endpoint of a flagged transaction."""
        frame = _raw_frame()
        flagged = frame[frame["flag"] == 1]
        expected = set(flagged["sender"]) | set(flagged["receiver"])
        actual = {
            benchmark.entity_ids[i]
            for i in np.flatnonzero(benchmark.labels == 1)
        }
        assert actual == expected
        assert benchmark.n_positive == len(expected)

    def test_label_column_never_reaches_the_edge_table(self, benchmark):
        assert "flag" not in benchmark.edge_table.columns

    def test_alias_column_in_attributes_is_refused(self):
        """Carrying the label under another name must be an error, not a warning."""
        with pytest.raises(ValueError, match="must not include the label column"):
            build_transaction_dataset(
                _raw_frame(), _source(attribute_columns=("flag",)), _spec()
            )

    def test_native_amount_is_not_published_as_usd(self, benchmark):
        """A multi-currency source keeps 'amount'; only a USD source gets 'amount_usd'."""
        assert "amount" in benchmark.edge_table.columns
        assert "amount_usd" not in benchmark.edge_table.columns
        assert amount_column(benchmark.edge_table) == "amount"

        usd = build_transaction_dataset(
            _raw_frame(), _source(amount_is_usd=True), _spec()
        )
        assert "amount_usd" in usd.edge_table.columns
        assert amount_column(usd.edge_table) == "amount_usd"

    def test_entity_codes_are_the_positional_index(self, benchmark):
        """The categorical codes must *be* the framework's entity addresses."""
        src, dst = entity_codes(benchmark.edge_table, benchmark.entity_ids)
        ids = np.array(benchmark.entity_ids, dtype=object)
        assert list(ids[src[:50]]) == list(
            benchmark.edge_table["source_entity"].astype(str)[:50]
        )
        assert list(ids[dst[:50]]) == list(
            benchmark.edge_table["target_entity"].astype(str)[:50]
        )

    def test_entity_codes_fall_back_for_a_plain_string_edge_table(self, benchmark):
        """Layer 3 edge tables carry strings, not categories, and must still work."""
        plain = benchmark.edge_table.copy()
        plain["source_entity"] = plain["source_entity"].astype(str)
        plain["target_entity"] = plain["target_entity"].astype(str)
        src, dst = entity_codes(plain, benchmark.entity_ids)
        expected_src, expected_dst = entity_codes(
            benchmark.edge_table, benchmark.entity_ids
        )
        assert np.array_equal(src, expected_src)
        assert np.array_equal(dst, expected_dst)

    def test_entity_codes_reject_an_unknown_entity(self, benchmark):
        plain = benchmark.edge_table.copy()
        plain["source_entity"] = plain["source_entity"].astype(str)
        plain["target_entity"] = plain["target_entity"].astype(str)
        plain.loc[plain.index[0], "source_entity"] = "NOT_IN_POPULATION"
        with pytest.raises(ValueError, match="not in entity_ids"):
            entity_codes(plain, benchmark.entity_ids)


# ---------------------------------------------------------------------------
# Audit arithmetic
# ---------------------------------------------------------------------------

class TestAuditBlocks:
    def test_data_block_counts_match_the_dataset(self, benchmark):
        audit = audit_dataset(benchmark, domain=DOMAIN_AML)
        assert audit.data["n_entities"] == benchmark.n_entities
        assert audit.data["n_transactions"] == len(benchmark.edge_table)
        assert audit.data["n_positive"] == benchmark.n_positive
        assert audit.data["n_negative"] == benchmark.n_negative
        assert audit.data["n_positive"] + audit.data["n_negative"] == audit.data["n_entities"]

    def test_duplicate_structure_is_reported(self, benchmark):
        audit = audit_dataset(benchmark, domain=DOMAIN_AML)
        duplicates = audit.data["duplicate_structure"]
        assert duplicates["n_self_loops"] == 0
        assert duplicates["n_distinct_ordered_pairs"] <= len(benchmark.edge_table)
        assert duplicates["mean_transactions_per_pair"] >= 1.0

    def test_timestamp_range_reflects_the_edges(self, benchmark):
        audit = audit_dataset(benchmark, domain=DOMAIN_AML)
        dates = pd.to_datetime(benchmark.edge_table["date"])
        assert audit.data["timestamp_range"]["min"] == str(dates.min())
        assert audit.data["timestamp_range"]["max"] == str(dates.max())

    def test_undated_dataset_reports_no_temporal_evaluation(self):
        frame = _raw_frame().drop(columns=["when"])
        dataset = build_transaction_dataset(
            frame, _source(timestamp=None), _spec()
        )
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.data["timestamp_range"] is None
        assert audit.labels["supports_time_forward_evaluation"] is False
        assert audit.labels["forward_split_positives"] is None

    def test_undefined_quantity_reports_a_reason_not_a_zero(self):
        """A constant or absent quantity is ``None`` plus a reason, never 0.0."""
        frame = _raw_frame().drop(columns=["amt"])
        dataset = build_transaction_dataset(frame, _source(amount=None), _spec())
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        volume = audit.structural_triviality["single_quantity"]["incoming_volume"]
        assert volume["average_precision"] is None
        assert "undefined" in volume["reason"]

    def test_self_loops_do_not_count_as_counterparties(self):
        """An account paying itself must not inflate its fan-in or fan-out."""
        frame = _raw_frame(n_normal=60, n_flagged_rings=12)
        loops = pd.DataFrame(
            [
                {
                    "sender": f"N{i:05d}",
                    "receiver": f"N{i:05d}",
                    "amt": 1.0,
                    "when": pd.Timestamp("2023-06-01"),
                    "flag": 0,
                    "mechanism": "self",
                }
                for i in range(10)
            ]
        )
        dataset = build_transaction_dataset(
            pd.concat([frame, loops], ignore_index=True), _source(), _spec()
        )
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.data["duplicate_structure"]["n_self_loops"] == 10
        # A self-loop is still a transaction, so the count block sees it, but
        # no counterparty-shaped quantity may.
        quantities = audit.structural_triviality["single_quantity"]
        assert quantities["fan_out"]["average_precision"] is not None
        assert quantities["counterparty_count"]["average_precision"] is not None

    def test_alias_columns_are_recorded_as_a_leak_risk(self, benchmark):
        audit = audit_dataset(
            benchmark, domain=DOMAIN_AML, label_alias_columns=["mechanism"]
        )
        assert audit.labels["label_alias_columns_present"] == ["mechanism"]
        assert "mechanism" in audit.labels["leakage_note"]
        assert any("mechanism" in limitation for limitation in audit.limitations)

    def test_synthetic_flag_is_carried_into_the_limitations(self, benchmark):
        audit = audit_dataset(benchmark, domain=DOMAIN_AML)
        assert any(limitation.startswith("SYNTHETIC") for limitation in audit.limitations)

    def test_probe_reports_counts_beside_its_metric(self, benchmark):
        probe = audit_dataset(benchmark, domain=DOMAIN_AML).structural_triviality[
            "combined_probe"
        ]
        assert probe["n_test_positive"] > 0
        assert probe["n_train_positive"] > 0
        assert 0.0 <= probe["test_average_precision"] <= 1.0

    def test_audit_is_serialisable(self, benchmark, tmp_path):
        import json

        audit = audit_dataset(benchmark, domain=DOMAIN_AML)
        path = save_audit(audit, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["verdict"] == audit.verdict
        assert payload["domain"] == DOMAIN_AML
        assert payload["data"]["n_entities"] == benchmark.n_entities


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class TestVerdicts:
    def test_all_positive_population_is_blocked(self):
        frame = _raw_frame()
        frame["flag"] = 1
        dataset = build_transaction_dataset(frame, _source(), _spec())
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.verdict == BLOCKED
        assert "no negative class" in audit.reason

    def test_no_positive_population_is_blocked(self):
        frame = _raw_frame()
        frame["flag"] = 0
        dataset = build_transaction_dataset(frame, _source(), _spec())
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.verdict == BLOCKED
        assert audit.structural_triviality["verdict"] == "undefined"

    def test_too_few_positives_is_regression_only(self):
        dataset = build_transaction_dataset(
            _raw_frame(n_flagged_rings=2), _source(), _spec()
        )
        assert dataset.n_positive < MIN_POSITIVES_FOR_CV
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.verdict == REGRESSION_ONLY
        assert audit.labels["supports_cross_validation"] is False

    def test_wrong_domain_is_never_model_selection_ready(self, benchmark):
        """Statistical quality cannot make a card-fraud label an AML label."""
        audit = audit_dataset(benchmark, domain=DOMAIN_CARD_FRAUD)
        assert audit.verdict == STRUCTURAL_VALIDATION_ONLY
        assert "not money laundering" in audit.reason
        assert any("DOMAIN MISMATCH" in limitation for limitation in audit.limitations)

    def test_structurally_trivial_labels_are_structural_validation_only(self):
        """A label written by the same process as the structure cannot rank models.

        Here the flagged accounts are given an activity level no normal account
        reaches, which is the failure mode the audit exists to catch.
        """
        rng = np.random.default_rng(11)
        base = pd.Timestamp("2023-01-01")
        rows: list[dict] = []
        for i in range(600):
            rows.append(
                {
                    "sender": f"N{i:05d}",
                    "receiver": f"HUB{i % 10:02d}",
                    "amt": 100.0,
                    "when": base + pd.Timedelta(days=int(rng.integers(0, 100))),
                    "flag": 0,
                    "mechanism": "normal",
                }
            )
        for i in range(60):
            for j in range(50):
                rows.append(
                    {
                        "sender": f"BAD{i:03d}",
                        "receiver": f"BADT{i:03d}_{j}",
                        "amt": 100.0,
                        "when": base + pd.Timedelta(days=int(rng.integers(0, 100))),
                        "flag": 1,
                        "mechanism": "fan_out",
                    }
                )
        dataset = build_transaction_dataset(pd.DataFrame(rows), _source(), _spec())
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.structural_triviality["verdict"] == "trivially_recoverable"
        assert audit.verdict == STRUCTURAL_VALIDATION_ONLY
        assert "function of structure" in audit.reason

    def test_non_trivial_labels_can_reach_model_selection_ready(self):
        """A label independent of structure must not be called trivial.

        Assigning the flag at random over a homogeneous population is the
        control case: nothing about the graph predicts it, so the probe should
        stay near the prevalence floor and the verdict should be READY.
        """
        rng = np.random.default_rng(5)
        base = pd.Timestamp("2023-01-01")
        senders = [f"A{i:04d}" for i in range(500)]
        receivers = [f"B{i:04d}" for i in range(500)]
        rows = []
        for _ in range(6000):
            rows.append(
                {
                    "sender": senders[int(rng.integers(0, 500))],
                    "receiver": receivers[int(rng.integers(0, 500))],
                    "amt": float(rng.uniform(10, 1000)),
                    "when": base + pd.Timedelta(days=int(rng.integers(0, 200))),
                    "flag": int(rng.random() < 0.02),
                    "mechanism": "random",
                }
            )
        dataset = build_transaction_dataset(pd.DataFrame(rows), _source(), _spec())
        audit = audit_dataset(dataset, domain=DOMAIN_AML)
        assert audit.structural_triviality["verdict"] == "non_trivial"
        assert audit.verdict == MODEL_SELECTION_READY

    def test_split_feasibility_agrees_with_the_label_block(self, benchmark):
        audit = audit_dataset(benchmark, domain=DOMAIN_AML)
        feasibility = audit.split_feasibility
        assert feasibility["stratified_kfold"]["usable"] == audit.labels[
            "supports_cross_validation"
        ]
        assert feasibility["time_forward_holdout"]["usable"] == audit.labels[
            "supports_time_forward_evaluation"
        ]


# ---------------------------------------------------------------------------
# IBM HI-Small loader and acquisition provenance
# ---------------------------------------------------------------------------

class TestIbmHiSmallLoader:
    """The HI-Small ledger has two schema traps; both are pinned here.

    The fixture reproduces the published header exactly, including the two
    columns that are *both* named ``Account``.
    """

    HEADER = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
        "Receiving Currency,Amount Paid,Payment Currency,Payment Format,"
        "Is Laundering"
    )

    def _ledger(self, tmp_path, rows):
        path = tmp_path / "HI-Small_Trans.csv"
        path.write_text("\n".join([self.HEADER, *rows]) + "\n", encoding="utf-8")
        return path

    def _default_rows(self):
        return [
            # bank 010 account AAA pays bank 020 account BBB, flagged
            "2022/09/01 00:20,010,AAA,020,BBB,10.00,US Dollar,10.00,US Dollar,ACH,1",
            # a self-loop: same bank, same account
            "2022/09/01 01:00,010,AAA,010,AAA,50.00,US Dollar,50.00,US Dollar,Reinvestment,0",
            # the SAME account string at a DIFFERENT bank - a distinct entity
            "2022/09/02 02:00,099,AAA,020,CCC,20.00,Euro,18.00,Euro,Cheque,0",
            "2022/09/03 03:00,020,BBB,020,CCC,30.00,US Dollar,30.00,US Dollar,Wire,0",
        ]

    def test_duplicate_account_column_is_read_as_payer_and_payee(self, tmp_path):
        """Both endpoints must be read; pandas renames the second to Account.1."""
        from src.experiments.sources import load_ibm_aml_hi_small

        dataset = load_ibm_aml_hi_small(self._ledger(tmp_path, self._default_rows()))
        edges = dataset.edge_table
        assert list(edges["source_entity"].astype(str))[:1] == ["010:AAA"]
        assert list(edges["target_entity"].astype(str))[:1] == ["020:BBB"]

    def test_entity_is_the_bank_account_pair(self, tmp_path):
        """The same account string at two banks is two entities, not one."""
        from src.experiments.sources import load_ibm_aml_hi_small

        dataset = load_ibm_aml_hi_small(self._ledger(tmp_path, self._default_rows()))
        assert "010:AAA" in dataset.entity_ids
        assert "099:AAA" in dataset.entity_ids
        positive = {
            dataset.entity_ids[i] for i in np.flatnonzero(dataset.labels == 1)
        }
        assert positive == {"010:AAA", "020:BBB"}

    def test_label_column_and_typology_never_reach_the_edge_table(self, tmp_path):
        from src.experiments.sources import load_ibm_aml_hi_small

        dataset = load_ibm_aml_hi_small(self._ledger(tmp_path, self._default_rows()))
        assert "Is Laundering" not in dataset.edge_table.columns
        assert not any(
            "launder" in str(c).lower() for c in dataset.edge_table.columns
        )

    def test_amount_is_native_not_usd(self, tmp_path):
        """15 currencies and no FX rate: the column must not claim to be USD."""
        from src.experiments.sources import load_ibm_aml_hi_small

        dataset = load_ibm_aml_hi_small(self._ledger(tmp_path, self._default_rows()))
        assert "amount" in dataset.edge_table.columns
        assert "amount_usd" not in dataset.edge_table.columns

    def test_self_loops_are_kept_as_transactions(self, tmp_path):
        """11.6 % of the real ledger is self-paying; dropping it would lose volume."""
        from src.experiments.sources import load_ibm_aml_hi_small

        dataset = load_ibm_aml_hi_small(self._ledger(tmp_path, self._default_rows()))
        edges = dataset.edge_table
        same = edges["source_entity"].astype(str) == edges["target_entity"].astype(str)
        assert int(same.sum()) == 1

    def test_spec_declares_synthetic_and_the_derived_label(self, tmp_path):
        from src.experiments.sources import load_ibm_aml_hi_small

        dataset = load_ibm_aml_hi_small(self._ledger(tmp_path, self._default_rows()))
        assert dataset.spec.synthetic is True
        assert "at least one transaction" in dataset.spec.label_definition
        assert any("AMLworld" in note for note in dataset.spec.notes)

    def test_a_file_with_the_wrong_schema_is_refused(self, tmp_path):
        """Silently mapping the wrong columns would be worse than failing."""
        from src.experiments.sources import load_ibm_aml_hi_small

        path = tmp_path / "HI-Small_Trans.csv"
        path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected HI-Small schema"):
            load_ibm_aml_hi_small(path)


class TestAcquisitionManifest:
    """Provenance for a dataset the repository deliberately does not vendor."""

    def test_registry_states_licence_and_source_for_every_entry(self):
        from acquire_dataset import REGISTRY

        assert REGISTRY, "registry must not be empty"
        for name, dataset in REGISTRY.items():
            assert dataset.licence and dataset.licence != "Unknown", name
            assert dataset.landing_page.startswith("https://"), name
            assert dataset.citation, name

    def test_manifest_records_what_is_on_disk(self, tmp_path):
        import hashlib

        from acquire_dataset import REGISTRY, acquire, save_manifest

        dataset = REGISTRY["ibm_aml_hi_small"]
        payload = b"Timestamp,From Bank,Account\n2022/09/01 00:20,010,AAA\n"
        (tmp_path / dataset.filename).write_bytes(payload)

        manifest = acquire(dataset, tmp_path, verify_only=True)
        assert manifest.bytes == len(payload)
        assert manifest.sha256 == hashlib.sha256(payload).hexdigest()
        assert manifest.header_line.startswith("Timestamp,From Bank,Account")
        assert manifest.licence == dataset.licence

        path = save_manifest(manifest, tmp_path / "manifests")
        assert path.exists()

    def test_a_differing_digest_is_an_error_not_an_overwrite(self, tmp_path):
        """Two different files must not silently share one provenance record."""
        from acquire_dataset import REGISTRY, acquire, save_manifest

        dataset = REGISTRY["ibm_aml_hi_small"]
        target = tmp_path / dataset.filename
        manifests = tmp_path / "manifests"

        target.write_bytes(b"first\n")
        save_manifest(acquire(dataset, tmp_path, verify_only=True), manifests)

        target.write_bytes(b"second, a different file entirely\n")
        with pytest.raises(RuntimeError, match="different files"):
            save_manifest(acquire(dataset, tmp_path, verify_only=True), manifests)

    def test_verify_only_refuses_to_invent_a_missing_file(self, tmp_path):
        from acquire_dataset import REGISTRY, acquire

        with pytest.raises(FileNotFoundError):
            acquire(REGISTRY["ibm_aml_hi_small"], tmp_path, verify_only=True)
