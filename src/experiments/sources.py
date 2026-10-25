"""
Layer 6 - External Benchmark Dataset Sources
=============================================

Adapters that turn a published transaction file into an
:class:`~src.experiments.dataset.ExperimentDataset`.

``reference.py`` loads the project's own Reference Dataset through Layers 1-3.
This module is the other door: public AML/fraud benchmarks that are read
directly, without the frozen vocabularies or the workspace ingestion path,
because those exist to serve the frozen checkpoint and would silently reshape a
benchmark to fit it.

Design
------
One generic builder, :func:`build_transaction_dataset`, plus a thin wrapper per
benchmark that supplies its column mapping and its label rule.  A new benchmark
is a function of about twenty lines; it never requires touching the framework.

Three properties matter and are enforced here rather than assumed:

**Entity identity is declared, not guessed.**  Each loader states what a node
is (an account number, a normalised name).  A benchmark whose accounts are not
stable identifiers is unusable as a node-classification dataset and the audit
says so.

**Entity labels are derived, and the derivation is written into the spec.**
SAML-D and IBM AML label *transactions*.  A node-classification experiment
needs a node label, so the loaders derive one - "the account took part in at
least one transaction the publisher flagged" - and record that sentence in
``DatasetSpec.label_definition``.  The derivation is a modelling decision with
consequences (it makes a counterparty of a launderer positive too), and the
audit measures those consequences instead of hiding them.

**Amounts are never converted.**  ``src/datasets/fx.py`` declares the FX seam
and implements nothing; that policy holds here.  A benchmark denominated in
several currencies keeps its native ``amount`` column, the spec records the
currency mix, and no feature pretends the values are comparable.  Loaders emit
``amount_usd`` only when the source is genuinely single-currency USD.

Memory
------
``source_entity`` and ``target_entity`` are ``pandas.Categorical`` whose
categories are exactly ``entity_ids`` in order.  The codes are therefore the
framework's positional entity index, which makes vectorised feature
computation a lookup rather than a join, and keeps a 9.5-million-row edge table
inside a few hundred megabytes.  :func:`entity_codes` is the accessor.

``graph`` is left ``None`` by :func:`build_transaction_dataset`.  A
``networkx.MultiDiGraph`` over 9.5 million edges costs tens of gigabytes, and
none of the ``aml_*`` families needs one - they read the edge table and a
deduplicated SciPy adjacency.  Families that do require a graph declare
``requires_graph``, and the runner builds one only when such a family is
requested, from exactly the edges that fold is allowed to see.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.dataset import ExperimentDataset
from src.experiments.spec import DatasetSpec

logger = logging.getLogger(__name__)

__all__ = [
    "SourceSpec",
    "build_transaction_dataset",
    "entity_codes",
    "amount_column",
    "load_samld",
    "load_ibm_aml_sample",
    "load_ibm_aml_hi_small",
    "load_financial_fraud",
    "available_sources",
    "load_source",
    "DEFAULT_SEARCH_PATHS",
]


#: Directories searched for a benchmark file when the caller gives a bare name.
#: The benchmarks are not vendored - they are large public downloads - so the
#: loaders resolve them relative to the repository, then to its parent, then to
#: the user's download area, and report which path they used.
DEFAULT_SEARCH_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "data" / "raw",
    Path(__file__).resolve().parents[2].parent,
    Path(__file__).resolve().parents[2].parent.parent / "Laundering Data",
)


def _resolve(path: Path | str, candidates: Sequence[str]) -> Path:
    """Return an existing file for ``path``, or search the default locations."""
    p = Path(path)
    if p.exists():
        return p
    for directory in DEFAULT_SEARCH_PATHS:
        for name in candidates:
            candidate = directory / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"{path} not found, and none of {list(candidates)} exists under "
        f"{[str(d) for d in DEFAULT_SEARCH_PATHS]}"
    )


def _content_version(path: Path) -> str:
    """Change-detecting version string derived from size and mtime.

    Not a content hash: hashing a gigabyte on every run is not worth it.  It is
    enough to notice that two artifacts were produced from different files, and
    the spec notes say exactly that.
    """
    stat = path.stat()
    digest = hashlib.sha256(f"{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()
    return f"size{stat.st_size}-{digest[:8]}"


# ---------------------------------------------------------------------------
# Generic builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """How to read one benchmark file into the framework's edge schema.

    Attributes
    ----------
    source, target:
        Column names holding the two entity identifiers.
    amount:
        Column holding the transaction amount, or ``None``.
    amount_is_usd:
        ``True`` only when every row is denominated in USD.  Controls whether
        the amount is emitted as ``amount_usd`` (comparable) or ``amount``
        (native, not comparable across rows).
    timestamp:
        Column holding a parseable datetime, or ``None``.
    date, time:
        Alternative to ``timestamp``: separate calendar and clock columns to be
        composed.  A benchmark that ships only a date has day resolution and the
        spec records that.
    label:
        Column holding the publisher's transaction-level flag.
    attribute_columns:
        Extra transaction columns carried into the edge table.  Anything that
        aliases the label must not appear here.
    """

    source: str
    target: str
    label: str
    amount: Optional[str] = None
    amount_is_usd: bool = False
    timestamp: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    attribute_columns: tuple[str, ...] = ()


def _read_any(
    path: Path,
    usecols: Optional[Sequence[str]],
    nrows: Optional[int],
    dtype: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Read a csv / zipped csv / xlsx into a DataFrame.

    ``dtype`` exists for identifier columns.  Left to infer, pandas reads a
    zero-padded code such as a bank number as an integer and drops the padding,
    which silently merges two distinct identifiers - a loader must pin those
    columns to ``str``.
    """
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"{path} contains no .csv member")
            with archive.open(members[0]) as handle:
                return pd.read_csv(handle, usecols=usecols, nrows=nrows, dtype=dtype)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, usecols=usecols, nrows=nrows, dtype=dtype)
    if suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path, nrows=nrows)
        if usecols is not None:
            frame = frame[list(usecols)]
        return frame
    raise ValueError(f"Unsupported benchmark file type: {path.suffix}")


def _compose_time(frame: pd.DataFrame, source: SourceSpec) -> pd.Series:
    """Build the edge ``date`` column, or an all-NaT column when undated."""
    if source.timestamp is not None:
        return pd.to_datetime(frame[source.timestamp], errors="coerce", format="mixed")
    if source.date is not None and source.time is not None:
        combined = (
            frame[source.date].astype(str).str.strip()
            + " "
            + frame[source.time].astype(str).str.strip()
        )
        return pd.to_datetime(combined, errors="coerce", format="mixed")
    if source.date is not None:
        return pd.to_datetime(frame[source.date], errors="coerce", format="mixed")
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def build_transaction_dataset(
    frame: pd.DataFrame,
    source: SourceSpec,
    spec: DatasetSpec,
) -> ExperimentDataset:
    """Turn a raw transaction frame into an ``ExperimentDataset``.

    The entity label is ``1`` when the entity appears as sender or receiver on
    at least one row whose ``source.label`` column is truthy.  That rule is the
    caller's to justify; this function only applies it consistently and leaves
    the justification in ``spec.label_definition``.

    The label column itself is *not* carried into the edge table.  Neither is
    any column the caller did not list in ``attribute_columns`` - the default is
    to drop, so a mechanism-naming column such as SAML-D's ``Laundering_type``
    cannot reach a feature by accident.
    """
    src_raw = frame[source.source].astype(str)
    dst_raw = frame[source.target].astype(str)

    entity_ids = sorted(set(src_raw.unique()) | set(dst_raw.unique()))
    categories = pd.Index(entity_ids)

    edges = pd.DataFrame(
        {
            "source_entity": pd.Categorical(src_raw, categories=categories),
            "target_entity": pd.Categorical(dst_raw, categories=categories),
        }
    )
    edges["date"] = _compose_time(frame, source).to_numpy()
    if source.amount is not None:
        amounts = pd.to_numeric(frame[source.amount], errors="coerce").to_numpy(
            dtype=np.float64
        )
        edges["amount_usd" if source.amount_is_usd else "amount"] = amounts
    for column in source.attribute_columns:
        if column == source.label:
            raise ValueError(
                f"attribute_columns must not include the label column "
                f"{column!r}; it would put the answer in the feature matrix."
            )
        values = frame[column]
        # Low-cardinality text (payment type, currency, bank location) is stored
        # as a category. Under pandas 3 a text column arrives as the ``str``
        # dtype rather than ``object``, so the test is "is it numeric", not
        # "is it object" - getting that wrong costs gigabytes on a 9.5-million
        # row edge table.
        edges[column] = (
            values.to_numpy()
            if pd.api.types.is_numeric_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype)
            else values.astype("category")
        )
    edges["edge_key"] = np.arange(len(frame), dtype=np.int64)

    flagged = frame[source.label].to_numpy()
    flagged = (
        flagged.astype(bool)
        if flagged.dtype != object
        else pd.Series(flagged).astype(str).str.lower().isin(("1", "true", "yes")).to_numpy()
    )
    labels = np.zeros(len(entity_ids), dtype=np.int8)
    if flagged.any():
        labels[edges["source_entity"].cat.codes.to_numpy()[flagged]] = 1
        labels[edges["target_entity"].cat.codes.to_numpy()[flagged]] = 1

    logger.info(
        "ExperimentDataset '%s' v%s: %d entities, %d positive (%.4f%%), %d transactions",
        spec.name,
        spec.version,
        len(entity_ids),
        int(labels.sum()),
        100.0 * float(labels.mean()) if entity_ids else 0.0,
        len(edges),
    )
    return ExperimentDataset(
        spec=spec,
        entity_ids=entity_ids,
        labels=labels,
        edge_table=edges,
        graph=None,
        attributes=None,
        # Kept beside the edge table, never inside it: a column there would be
        # visible to every feature family.
        transaction_labels=flagged,
    )


def entity_codes(dataset_or_edges: Any, entity_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Positional source and target indices for an edge table.

    Uses the categorical codes directly when the edge table was built by
    :func:`build_transaction_dataset` (the categories are ``entity_ids`` in
    order, so a code *is* the framework's entity index).  Falls back to a hash
    join for edge tables produced by Layer 3, which carry plain strings.
    """
    edges = getattr(dataset_or_edges, "edge_table", dataset_or_edges)
    if edges is None or len(edges) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty

    src = edges["source_entity"]
    dst = edges["target_entity"]
    index = pd.Index(entity_ids)
    if isinstance(src.dtype, pd.CategoricalDtype) and src.cat.categories.equals(index):
        return src.cat.codes.to_numpy(), dst.cat.codes.to_numpy()

    lookup = pd.Series(np.arange(len(entity_ids), dtype=np.int64), index=index)
    src_codes = lookup.reindex(src.astype(str)).to_numpy()
    dst_codes = lookup.reindex(dst.astype(str)).to_numpy()
    if np.isnan(src_codes).any() or np.isnan(dst_codes).any():
        raise ValueError(
            "edge table references entities that are not in entity_ids; the "
            "dataset and its edges disagree about the population."
        )
    return src_codes.astype(np.int64), dst_codes.astype(np.int64)


def amount_column(edges: Optional[pd.DataFrame]) -> Optional[str]:
    """Name of the usable amount column: ``amount_usd``, else ``amount``, else None.

    Kept in one place so no feature family invents a currency conversion by
    reading a native-currency column under a USD name.
    """
    if edges is None:
        return None
    for name in ("amount_usd", "amount"):
        if name in edges.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# SAML-D
# ---------------------------------------------------------------------------

#: SAML-D column that names the generating mechanism of every row, normal rows
#: included.  ``Is_laundering == 1`` exactly when it does not start with
#: ``Normal``, verified over all 9,504,852 rows: it is an alias of the label and
#: must never enter a feature matrix or an edge attribute.
SAMLD_LABEL_ALIAS = "Laundering_type"


def load_samld(
    path: Path | str = "SAML-D.csv.zip",
    *,
    name: str = "samld",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    nrows: Optional[int] = None,
) -> ExperimentDataset:
    """Load SAML-D as an entity-labelled dataset.

    SAML-D (Oztas et al., 2023) is a *synthetic* AML benchmark: its
    transactions were generated from declared typologies, not observed.  The
    returned spec carries ``synthetic=True`` and every artifact inherits it.

    Parameters
    ----------
    date_from, date_to:
        Optional inclusive calendar bounds.  A window is a legitimate way to
        make a run cheaper, but it changes the dataset, so it is written into
        the version string and the provenance rather than left implicit.
    nrows:
        Read only the first ``nrows`` rows.  For smoke tests only - the file is
        ordered by time, so a prefix is a time window with an undeclared end.

    Notes
    -----
    ``Laundering_type`` is dropped, not carried.  ``Amount`` is native
    (13 currencies, 95.7 % GBP) and is emitted as ``amount``, never
    ``amount_usd``.
    """
    resolved = _resolve(path, ("SAML-D.csv.zip", "SAML-D.csv"))
    columns = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
    ]
    frame = _read_any(resolved, columns, nrows)

    window_note = ""
    if date_from or date_to:
        days = pd.to_datetime(frame["Date"], errors="coerce")
        mask = pd.Series(True, index=frame.index)
        if date_from:
            mask &= days >= pd.Timestamp(date_from)
        if date_to:
            mask &= days <= pd.Timestamp(date_to)
        frame = frame.loc[mask].reset_index(drop=True)
        window_note = f"@{date_from or 'start'}..{date_to or 'end'}"

    source = SourceSpec(
        source="Sender_account",
        target="Receiver_account",
        label="Is_laundering",
        amount="Amount",
        amount_is_usd=False,
        date="Date",
        time="Time",
        attribute_columns=(
            "Payment_type",
            "Payment_currency",
            "Received_currency",
            "Sender_bank_location",
            "Receiver_bank_location",
        ),
    )
    spec = DatasetSpec(
        name=name,
        version=_content_version(resolved) + window_note + (f"-n{nrows}" if nrows else ""),
        provenance=f"file:{resolved}{window_note}",
        label_definition=(
            "positive = the account is the sender or the receiver of at least "
            "one transaction whose published Is_laundering flag is 1. SAML-D "
            "labels transactions; this entity label is derived, and it makes a "
            "launderer's counterparty positive as well"
        ),
        synthetic=True,
        notes=(
            "SAML-D is generated from declared typologies. Results on it are "
            "evidence about a simulator, not about real-world detection.",
            "Amounts are native across 13 currencies (95.7% GBP) and are not "
            "FX-converted; the amount column is 'amount', not 'amount_usd'.",
            "The published Laundering_type column is an exact alias of the "
            "label and is not loaded.",
            "Version is derived from file size and mtime, not a content hash.",
        ),
    )
    return build_transaction_dataset(frame, source, spec)


# ---------------------------------------------------------------------------
# IBM AML
# ---------------------------------------------------------------------------


def load_ibm_aml_sample(
    path: Path | str = "IBM_AML_sample.xlsx",
    *,
    name: str = "ibm_aml_sample",
) -> ExperimentDataset:
    """Load the IBM AML extract present in this tree.

    The file in this repository's data area is the *laundering-pattern* extract
    of the IBM synthetic AML data, not its transaction ledger: every row carries
    ``is_suspicious == 1``.  The loader does not hide that - it produces a
    dataset with a 100 % positive rate, and the audit classifies it accordingly.
    """
    resolved = _resolve(path, ("IBM_AML_sample.xlsx", "IBM data.txt"))
    frame = _read_any(resolved, None, None)
    source = SourceSpec(
        source="payer_account",
        target="beneficiary_account",
        label="is_suspicious",
        amount="amount_orig",
        amount_is_usd=False,
        timestamp="transaction_datetime",
        attribute_columns=("payment_type", "currency_orig", "currency_dest"),
    )
    spec = DatasetSpec(
        name=name,
        version=_content_version(resolved),
        provenance=f"file:{resolved}",
        label_definition=(
            "positive = the account appears on at least one row flagged "
            "is_suspicious. Every row in this extract is flagged, so every "
            "entity is positive"
        ),
        synthetic=True,
        notes=(
            "This is the laundering-pattern extract of IBM's synthetic AML "
            "data. It contains no negative class and cannot train or evaluate "
            "a classifier.",
            "Amounts span 15 currencies and are not FX-converted.",
        ),
    )
    return build_transaction_dataset(frame, source, spec)


#: IBM's ledger names the payer and payee account columns *identically*
#: ("Account"), so pandas disambiguates the second as "Account.1". Both the
#: original and the de-duplicated name are recorded here so a future reader
#: does not think the second one was invented.
IBM_PAYER_ACCOUNT = "Account"
IBM_BENEFICIARY_ACCOUNT = "Account.1"


def load_ibm_aml_hi_small(
    path: Path | str = "HI-Small_Trans.csv",
    *,
    name: str = "ibm_aml_hi_small",
    nrows: Optional[int] = None,
) -> ExperimentDataset:
    """Load IBM's HI-Small transaction ledger as an entity-labelled dataset.

    This is the *ledger*, with both classes - not the ``HI-Small_Patterns.txt``
    typology listing, which contains flagged transactions only and cannot train
    a classifier. Acquire it with ``python acquire_dataset.py ibm_aml_hi_small``;
    the provenance manifest is ``data/manifests/ibm_aml_hi_small.json``.

    IBM's AMLworld data is **synthetic**, generated by a virtual-world
    simulator. ``DatasetSpec.synthetic`` is ``True`` and every artifact inherits
    it.

    Entity identity
    ---------------
    Bank codes are zero-padded text (``010``, ``00952``). They are read as
    ``str``: under integer coercion the file's 30,528 distinct bank codes
    collapse to 30,470, merging 58 pairs such as ``012`` with ``0012``.

    An entity is the ``(bank, account)`` pair, joined as ``"<bank>:<account>"``,
    **not** the account string alone. Measured on the file: 515,080 distinct
    account strings but 515,088 distinct pairs - eight account strings appear
    at two different banks. Eight collisions out of half a million would change
    almost nothing, and the pair is still the right key: an account number is
    issued by a bank and is only unique within it, so merging two banks'
    accounts because they share a hex string would be an entity-resolution
    error, not a simplification.

    Labels
    ------
    IBM labels **transactions** (``Is Laundering``). A node-classification
    experiment needs a node label, so this loader derives one - the account is
    the payer or the payee of at least one flagged transaction - and writes that
    sentence into ``DatasetSpec.label_definition``. The derivation makes a
    launderer's counterparty positive too, and the audit measures the
    consequences rather than hiding them.

    Amounts
    -------
    ``Amount Paid`` is denominated in ``Payment Currency`` and ``Amount
    Received`` in ``Receiving Currency``; they differ on 72,158 rows, which is
    currency conversion, not an error. 15 currencies with US Dollar at 37 %, so
    the amount is emitted as native ``amount``, never ``amount_usd``. No FX rate
    is invented (``src/datasets/fx.py``).

    Self-loops
    ----------
    591,212 rows (11.6 %) have the same payer and payee - the ``Reinvestment``
    payment format. They are kept in the edge table because they are real
    transactions and real volume; the structural feature families already
    exclude them from counterparty-shaped columns.
    """
    resolved = _resolve(path, ("HI-Small_Trans.csv",))
    # Bank codes are zero-padded and must stay text. Measured on the published
    # file: 30,528 distinct bank codes as strings collapse to 30,470 under
    # integer coercion - 58 collisions, including '012' with '0012'. Letting
    # pandas infer would merge banks that the publisher kept apart.
    frame = _read_any(
        resolved,
        None,
        nrows,
        dtype={
            "From Bank": str,
            "To Bank": str,
            IBM_PAYER_ACCOUNT: str,
            IBM_BENEFICIARY_ACCOUNT: str,
        },
    )

    missing = {
        c
        for c in (
            "Timestamp", "From Bank", IBM_PAYER_ACCOUNT, "To Bank",
            IBM_BENEFICIARY_ACCOUNT, "Amount Paid", "Payment Currency",
            "Receiving Currency", "Payment Format", "Is Laundering",
        )
        if c not in frame.columns
    } 
    if missing:
        raise ValueError(
            f"{resolved} does not have the expected HI-Small schema; missing "
            f"{sorted(missing)}. Found: {list(frame.columns)}"
        )

    # Composite entity keys, built before the generic builder sees the frame.
    frame = frame.assign(
        payer_entity=frame["From Bank"].astype(str) + ":" + frame[IBM_PAYER_ACCOUNT].astype(str),
        beneficiary_entity=(
            frame["To Bank"].astype(str) + ":" + frame[IBM_BENEFICIARY_ACCOUNT].astype(str)
        ),
    )

    source = SourceSpec(
        source="payer_entity",
        target="beneficiary_entity",
        label="Is Laundering",
        amount="Amount Paid",
        amount_is_usd=False,
        timestamp="Timestamp",
        attribute_columns=(
            "Payment Format",
            "Payment Currency",
            "Receiving Currency",
        ),
    )
    spec = DatasetSpec(
        name=name,
        version=_content_version(resolved) + (f"-n{nrows}" if nrows else ""),
        provenance=f"file:{resolved}",
        label_definition=(
            "positive = the (bank, account) entity is the payer or the payee of "
            "at least one transaction whose published 'Is Laundering' flag is 1. "
            "IBM labels transactions; this entity label is derived, and it makes "
            "a launderer's counterparty positive as well"
        ),
        synthetic=True,
        notes=(
            "SYNTHETIC: IBM AMLworld virtual-world simulator (Altman et al., "
            "NeurIPS 2023 Datasets and Benchmarks). Not evidence about "
            "real-world detection.",
            "Entity is the (bank, account) pair; account strings are not "
            "globally unique (8 collisions across banks).",
            "Amounts are native across 15 currencies (37% USD) and are not "
            "FX-converted; the amount column is 'amount', not 'amount_usd'.",
            "The typology names live in the separate HI-Small_Patterns.txt and "
            "are deliberately not loaded: their group headers state the "
            "generating mechanism and its degree/hop parameter.",
            "Version is derived from file size and mtime, not a content hash; "
            "the content hash is in data/manifests/ibm_aml_hi_small.json.",
        ),
    )
    return build_transaction_dataset(frame, source, spec)


# ---------------------------------------------------------------------------
# Kaggle financial fraud
# ---------------------------------------------------------------------------


def load_financial_fraud(
    path: Path | str = "financial_fraud_detection_dataset.csv",
    *,
    name: str = "financial_fraud_kaggle",
    nrows: Optional[int] = None,
) -> ExperimentDataset:
    """Load the Kaggle ``financial_fraud_detection_dataset`` extract.

    This is card-fraud data, not laundering data: its single ``fraud_type`` is
    ``card_not_present``.  It also ships pre-computed anomaly scores
    (``velocity_score``, ``spending_deviation_score``, ``geo_anomaly_score``)
    which were produced alongside the label; none of them is loaded.
    """
    resolved = _resolve(
        path, ("financial_fraud_detection_dataset.csv",)
    )
    columns = [
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "transaction_type",
        "payment_channel",
        "is_fraud",
    ]
    frame = _read_any(resolved, columns, nrows)
    source = SourceSpec(
        source="sender_account",
        target="receiver_account",
        label="is_fraud",
        amount="amount",
        amount_is_usd=False,
        timestamp="timestamp",
        attribute_columns=("transaction_type", "payment_channel"),
    )
    spec = DatasetSpec(
        name=name,
        version=_content_version(resolved) + (f"-n{nrows}" if nrows else ""),
        provenance=f"file:{resolved}",
        label_definition=(
            "positive = the account appears on at least one transaction whose "
            "is_fraud flag is true. The published flag marks card-not-present "
            "fraud, not money laundering"
        ),
        synthetic=True,
        notes=(
            "Card-fraud data with a single fraud_type. Not an AML benchmark.",
            "The published velocity / spending-deviation / geo-anomaly score "
            "columns were generated with the label and are not loaded.",
            "Currency is unstated; the amount column is 'amount'.",
        ),
    )
    return build_transaction_dataset(frame, source, spec)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SOURCES = {
    "samld": load_samld,
    "ibm_aml_sample": load_ibm_aml_sample,
    "ibm_aml_hi_small": load_ibm_aml_hi_small,
    "financial_fraud_kaggle": load_financial_fraud,
}


def available_sources() -> list[str]:
    """Names accepted by :func:`load_source`, plus ``reference_dkdata``."""
    return sorted(list(_SOURCES) + ["reference_dkdata"])


def load_source(name: str, **kwargs: Any) -> ExperimentDataset:
    """Load a benchmark by registry name.

    ``reference_dkdata`` is routed to :mod:`src.experiments.reference` so that
    every dataset in the programme is reachable through one call.
    """
    if name == "reference_dkdata":
        from src.experiments.reference import load_reference_dataset

        return load_reference_dataset(**kwargs)
    if name not in _SOURCES:
        raise KeyError(f"Unknown source {name!r}. Available: {available_sources()}")
    return _SOURCES[name](**kwargs)
