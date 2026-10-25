# Adaptive Dataset Workspaces Architecture

This document describes the external-dataset subsystem: how an arbitrary uploaded
spreadsheet is isolated, profiled, semantically mapped, canonicalized, validated
against the frozen 167-feature model contract, and — eventually — scored.

Each stage below is labelled **CURRENT**, **IN PROGRESS**, or **PLANNED**.

> ## Subsystem status: end-to-end path delivered
>
> The full external-dataset path now runs:
>
> ```text
> upload -> profile -> semantic mapping -> canonicalization -> compatibility
>        -> READY -> Run Analysis -> predict_workspace() -> persisted
>        predictions -> workspace dashboard
> ```
>
> Verified on a real external dataset (SAML-D, 100,000 transactions, 65,168
> account entities) through the actual HTTP API, including restart persistence.
>
> Reaching `READY` is not automatic and is not forced. Of the three real
> datasets in [Verified external dataset results](#verified-external-dataset-results),
> two are `BLOCKED` on a stated, genuine limitation.
>
> `READY` is gated on the **inference contract**, not the labelled-training
> schema: an unlabelled dataset with no `payer_core` / `beneficiary_core` /
> `investigation` reaches `READY` and runs inference. See
> [The inference contract](#the-inference-contract).

---

## Workspace

A **Workspace** is an isolated namespace representing a single dataset analysis
session. It guarantees that processing, mapping, or validating one dataset never
bleeds state or references into another. The workspace abstraction is the
boundary that untrusted dataset columns must cross before they are canonicalized
and permitted to enter the validated ML pipeline.

### Workspace ID — **CURRENT**

Each workspace is identified by a strict UUID4 (`workspace_id`).

**Safety:** filesystem paths are never derived from a user-supplied display name
(e.g. `Bank Transactions August`). The user's string is stored solely as a
manifest property. Path traversal (`../../etc`) and OS-illegal-character
injection are structurally impossible.

Two independent guards enforce this in `WorkspaceManager`:

* `_is_safe_uuid` requires `str(uuid.UUID(workspace_id)) == workspace_id`, so
  only canonical UUID4 strings are accepted;
* `_get_path` resolves the path and asserts the base directory is among its
  parents.

### Manifest — **CURRENT**

Workspace state is persisted in `manifest.json` as a typed dataclass —
`WorkspaceManifest` — enforcing safe nulls and enum statuses rather than a loose
dictionary. It stores:

* **Identity and dates** — `workspace_id`, `display_name`, `created_at`, `updated_at`
* **File references** — `source_file`, `original_filename`, `source_hash`, `file_size`
* **Operational status** — `status` (typed `WorkspaceStatus` enum)
* **Stage versions** — `profiler_version`, `mapping_version`
* **Prediction state** — `prediction_status`, `prediction_timestamp`, `model_version`
* **Mapping and settings** — `schema_mapping`, `settings`
* **Dataset configuration** — `dataset_config` (`identity_source`,
  `currency_policy`, `geography_policy`, `single_country_code`), the
  operator's explicit declarations. Manifests written before this field
  existed simply omit the key and fall back to the defaults.

### Persistence and durability — **CURRENT**

All state writes are **atomic**, through one helper —
`workspace.write_json_atomic` — which writes to `<name>.tmp`, calls `flush()`
then `os.fsync()`, and only then `Path.replace()`s the target. `manifest.json`,
`profile.json` and `mapping.json` all go through it, including the operator
mapping overrides applied by `WorkspaceManager.apply_mapping_overrides`. A
crash mid-write therefore cannot leave a partially written or corrupt
artifact.

The durability guarantee is stated once rather than re-derived per call site;
previously `mapping.json` was written with a plain `open()` from two places,
one of them the HTTP route.

`Workspace.update()` refuses to mutate `workspace_id` or `created_at`, and
touches `updated_at` on every save.

### Recovery — **CURRENT**

`Workspace.load()` calls `_reconcile_state()` on every load. A workspace found
stranded in `ANALYZING` — the state it would be left in if the process died
mid-profiling — is checked for a complete `profile.json` (one containing both
`columns` and `row_estimate`). If the profile is intact, the workspace is
advanced to `NEEDS_MAPPING` and stamped with `profiler_version`. If not, it is
left as-is for re-profiling.

A corrupt (unparseable) manifest raises `ValueError`; a missing one raises
`FileNotFoundError`. `list_workspaces()` skips both rather than failing the whole
listing.

### Isolation — **CURRENT**

* **Filesystem** — every directory fetch is verified to be a direct child of the
  base directory.
* **Reference leakage** — updating or renaming workspace A modifies strictly
  `<base>/A/manifest.json`. Loading workspace B constructs an entirely new
  `Workspace` object parsed freshly from disk, guaranteeing zero shared state.

### Lifecycle — **CURRENT**

```text
EMPTY ──▶ UPLOADED ──▶ ANALYZING ──▶ NEEDS_MAPPING ──▶ READY
                            │                             │
                            └──────────────▶ FAILED ◀─────┴──▶ INCOMPATIBLE
                                                              PARTIAL
```

| Status | Meaning |
|---|---|
| `EMPTY` | Workspace instantiated; UUID directory and base manifest created |
| `UPLOADED` | A dataset file has been written into the workspace |
| `ANALYZING` | The profiling engine is operating |
| `NEEDS_MAPPING` | Profiling complete; semantic mapping requires review |
| `READY` | Mapping complete and strictly compatible with the frozen model |
| `PARTIAL` | Partially satisfiable — declared, not currently produced |
| `INCOMPATIBLE` | Rejected: missing fields, bad types, or feature-contract mismatch |
| `FAILED` | Empty dataset, zero valid edges, or an unrecoverable error |

### Workspace artifacts

| File | Stage | Status |
|---|---|---|
| `manifest.json` | all | **CURRENT** |
| `raw_data.<ext>` | 2 — ingestion | **CURRENT** |
| `profile.json` | 2 — profiling | **CURRENT** |
| `mapping.json` | 3 — semantic mapping | **CURRENT** |
| `canonical.parquet` | 4 — canonicalization | **CURRENT** |
| `compatibility.json` | 4 — validation | **CURRENT** |
| `predictions.parquet` | 5 — inference | **CURRENT** |
| `trust.json` | 5 — inference | **CURRENT** |

---

## Stage 2 — Safe ingestion & schema profiling — **CURRENT**

### Raw source preservation

Uploaded `.csv`, `.xlsx`, or `.xls` files are copied into the workspace as
`raw_data.<ext>`. The original is preserved **completely intact** — no silent
conversions or truncations occur at the file level. The file is SHA-256 hashed and
its metadata logged to `manifest.json`.

Guards in `parsers.safe_ingest`:

* extension must be in `ALLOWED_EXTENSIONS` = `{.csv, .xlsx, .xls}`;
* **multi-sheet Excel workbooks are rejected** with an explicit message rather
  than silently taking sheet 0;
* copying and hashing are chunked at 8 KB, so large files never load fully into
  memory;
* empty uploads are rejected and the partial destination file removed.

### Dataset profiling

`WorkspaceManager.profile_workspace` runs a **bounded** profiler.

* **Large-file strategy** — only a representative sample (default 1,000 rows) is
  read, to prevent server OOM. The profiler reports `is_exact_count=False` when
  bounded.
* **Captured per column** — `null_ratio`, `uniqueness_ratio`, inferred type
  (`string`, `float`, `integer`, `datetime`, `boolean`), min/max numeric
  statistics, and a 5-item sample array.
* **Normalization** — source column names are normalized for analytics parsing
  (`Account Number` → `account_number`) while `original_name` is preserved
  exactly.

### Profiling is not semantic mapping

`profile.json` documents **structural metadata and type hints only** — e.g. "this
column contains 95% unique floats". It makes no decision linking columns to
Experiment 3 semantic features. All semantic decisions are deferred to Stage 3.

---

## Stage 3 — Semantic column mapping — **CURRENT**

### Objective

Determine which arbitrary source columns map to the canonical concepts the
pipeline requires. This bridges unknown tabular data to the strict 167-dimension
model contract.

### Canonical concepts

`CanonicalConcept` (21 members) was derived from
`src/data/ingestion.py::REQUIRED_COLUMNS`: transaction IDs, timestamps, amounts
and currencies, topological anchors (sender/receiver names and accounts), and
metadata (types, jurisdictions, bank countries, purpose, source file,
investigation flag, and the `SENDER_CORE` / `RECEIVER_CORE` labels).

### Confidence-scoring engine

Rather than boolean matching, `mapper.py` computes a continuous score from
multiple evidence vectors drawn from the `DatasetProfile`:

1. **Lexical matching** — normalized aliasing (`A/C Number` → `account_number`)
   cross-referenced against `SPECIFIC_ALIASES` and `GENERIC_ALIASES`.
2. **Directional composition** — a *role* token composed with a *concept-family*
   token; see [Directional lexicon](#directional-lexicon--current) below.
3. **Temporal composition** — any token naming a point in time
   (`date`, `time`, `datetime`, `timestamp`, `day`) supports `TIMESTAMP` as
   strongly as an exact alias, so `event_time`, `transaction_time` and
   `settlement_date` are reached without enumerating them. Time is not
   directional, so there is no role dimension here.
4. **Type compatibility** — mapping a `string` column to `TRANSACTION_AMOUNT`
   incurs a heavy penalty; a *numeric* column under a party word is that
   party's identifier rather than its name.
5. **Cardinality alignment** — identifiers such as `TRANSACTION_ID` are expected
   to exceed 90% uniqueness; jurisdictions, bank countries and entity types to
   fall below 5%, and to be penalised above 50%.
6. **Pattern matching** — regex hints discovered during Stage 2. An
   `identifier-like` column is weak evidence *against* a counterparty name.
7. **Cross-column relationships** — two columns claiming one concept are ranked
   against each other, and in one case recognised as *complementary* rather than
   rival; see [Complementary claimants](#complementary-claimants--current).

Type and cardinality are **corroborating** evidence only: a candidate with no
lexical support at all is discarded, because shape describes what a column
looks like and never what it means. They are decisive only *between* readings
that lexical evidence has already put on the table — `payer` holding strings of
labels is a name, `payer` holding integers is an account.

### Directional lexicon — **CURRENT**

Direction is a property of a column name independent of what the column is
about, so it is modelled as its own dimension rather than enumerated per
phrase:

| Dimension | Examples |
|---|---|
| Role — sender side | `sender`, `payer`, `originator`, `remitter`, `debtor`, `ordering`, `from`, `source`, `origin`, `initiator`, `outgoing` |
| Role — receiver side | `receiver`, `beneficiary`, `recipient`, `payee`, `creditor`, `destination`, `to`, `target`, `incoming` |
| Family | `name`, `account`, `type`, `bank_country`, `jurisdiction`, `core` |

A role token composed with a family token scores as strongly as an exact alias.
Adding `payee` to the role lexicon therefore teaches every receiver-side
concept at once.

Three further compositions extend the same idea:

* **A role token beside an id token names that party.** `originator_id` and
  `beneficiary_id` identify the originator and the beneficiary. Previously the
  bare `id` matched `TRANSACTION_ID` by substring, so an entire schema dialect's
  two identity columns were read as the transaction's own identifier and the
  dataset had no counterparties at all. A column naming a party is also evidence
  *against* `TRANSACTION_ID`, which identifies the transaction.
* **A column that is nothing but a role token names the party, attribute
  unstated.** `payer` / `payee` / `sender` / `from` propose *both* identity
  readings — name and account — and the values decide between them. A qualified
  column such as `amount_dest` contains a role token but is not a party column,
  so this applies only when every meaningful token is a role token.
* **A bank token with no place token names the institution, not the party.**
  IBM AML's `payer_bank_account` holds 78,394 distinct bank identifiers.
  Reading it as the payer's own account would replace the entity population with
  banks, so no family is claimed at all.

Two rules keep this safe:

* **A contradicting role is evidence against.** A column naming one side scores
  *negatively* for the other side's concepts, so `Receiver_bank_location` can
  never be proposed as a sender concept.
* **`bank_country` outranks `jurisdiction`.** Both contain a place token, but
  only the bank reading feeds the model's 151-column country block, so a bank
  token present alongside a place token resolves to bank country.

`type` and `core` are not claimed without an explicit side: `payment_type`
describes the payment, not a party.

Before this existed, `SENDER_BANK_COUNTRY`, `RECEIVER_BANK_COUNTRY`,
`SENDER_TYPE` and `RECEIVER_TYPE` had no alias entry at all and were
unreachable by the automatic mapper — the four concepts behind 155 of the 167
features could only be set by operator override, while `SENDER_JURISDICTION`,
which the model does not consume, won the geography columns.

### Ambiguity and conflicts

* **Generic aliases** — a column named `account_number` matches `SENDER_ACCOUNT`
  and `RECEIVER_ACCOUNT` symmetrically. Both candidates are registered and the
  column is flagged `NEEDS_CONFIRMATION` rather than guessed.
* **A tie *within* one column is resolved the same way as a tie between two: not
  at all.** Candidates were sorted by score with a stable sort, so an exact tie
  fell through to `CanonicalConcept` declaration order — a bare `country` column
  silently became `SENDER_JURISDICTION` and a bare `account_number` silently
  became `SENDER_ACCOUNT`, choosing a side of the transaction on the strength of
  an enum's line ordering. Such a column now carries all its candidates with no
  selection, for review.
* **Conflicts** — claimants for one concept are ranked by score. A clear winner
  keeps the concept; the columns it beat drop to `NEEDS_CONFIRMATION` with
  their selection cleared, so they stay visible for review without being
  projected into the same canonical field. **A tie is still a `CONFLICT`** and
  neither column is auto-selected: indistinguishable evidence is a genuine
  ambiguity, and arbitration must not manufacture a winner from one.
* **Unknowns** — irrelevant data (e.g. `device_id`) fails to reach threshold and
  defaults to `UNMAPPED`, preserved for UI context without polluting graph
  tensors.

### Complementary claimants — **CURRENT**

Two columns claiming `TIMESTAMP` are normally rivals. A **calendar** column and
a **clock** column are not: each names half an instant and together they name a
whole one. When exactly one claimant names a date part and exactly one names a
time part, both keep the concept at `NEEDS_CONFIRMATION` and canonicalization
composes them (see [Deterministic derivation](#stage-35--deterministic-derivation--current)).

This is what SAML-D as published requires: it ships `Date` and `Time` and no
combined column, so under plain arbitration the two tied, annihilated each other
as a `CONFLICT`, and the dataset reached validation with no `date` at all.

The date/time reading is `mapper.timestamp_part` — the same token vocabulary
that proposed the mapping — so no layer holds a private list of column names.

### Demotions — **CURRENT**

Two readings are weaker than another without being wrong, and both previously
produced a destructive tie. Each halves the lexical score rather than
disqualifying the column, so it still ranks and still appears for review:

| Demotion | Why |
|---|---|
| A **derived form** (`norm`, `normalized`, `clean`, `std`, `canonical`, `key`) | `payer_name_norm` is somebody else's normalization of `payer_name`, not a rival fact. As equal claimants both were discarded and the dataset lost its identity entirely — the failure the Reference Dataset's own raw workbook hits. The plain column wins, and the pipeline re-derives `payer_name_norm` under the entity-resolution rule the checkpoint was trained with, rather than adopting a foreign one. |
| A **receiving-side** amount or currency (`amount_dest`, `currency_dest`, `Received_currency`) | The canonical schema has `amount_orig` / `amount_orig_currency` — what the payer *sent* — and no counterpart for what the beneficiary received. A receiving-side column is a different quantity, not another spelling. |

### Mapping manifest and explainability

Decisions are recorded in `mapping.json` via the typed `MappingManifest`. Each
`ColumnMapping` retains **every** candidate with its `score` and its `evidence`
components, so the UI can explain *why* a column scored as it did rather than
only *what* it resolved to.

Mapping is an **intent representation**. Stage 3 does not build graphs, encode
features, or run inference.

### Human review

`POST /api/workspaces/<id>/mapping` accepts operator overrides of
`selected_concept` and `status`, single or batched, and rewrites `mapping.json`.

---

## Stage 3.5 — Deterministic derivation — **CURRENT**

`src/datasets/derivation.py` owns one question:

> Which canonical columns can be **computed** for this dataset, and under what
> stated rule?

Before it existed each rule lived inline in `canonicalize_and_validate` — four
as arms of an `elif` ladder interleaved with the reporting of field statuses,
two more as blocks after the loop. There was no single place to read what the
system can derive and no seam at which a rule could be tested without building
a workspace on disk.

| Rule | Fires when | Produces |
|---|---|---|
| `_derive_instant` | `TIMESTAMP` is mapped to exactly one calendar column and one clock column | `date`, composed properly rather than pasted as text |
| `_derive_payer_name_norm` / `_derive_beneficiary_name_norm` | the corresponding name is mapped | lowercase, alphanumeric-only entity key |
| `_derive_source_file` | no column is mapped to `SOURCE_FILE` | the upload's own filename |
| `_derive_row_id` | no column is mapped to `TRANSACTION_ID` | `tx_<i>` |
| `_derive_amount_usd` | `NATIVE_USD` is declared **and the dataset does not contradict it** | a copy of `amount_orig` |
| `_derive_single_country` | `SINGLE_DOMESTIC_COUNTRY` is declared with a code | both bank-country columns |

Two invariants hold for every rule:

* **A derivation is evidence-bearing, not gap-filling.** Each fires only when
  its inputs are actually present, and records the source columns it read.
  Nothing is defaulted to `0`, `"US"`, `False` or any other substantive value,
  no FX rate is invented, and no value is copied from one side of a transaction
  to the other.
* **Ambiguity is not a derivation.** A rule fires only where exactly one reading
  is possible. Where the evidence supports more than one answer, no rule fires
  and the field stays missing for review.

Every rule has one shape — it returns the derivations it supports, possibly none
— so the planning loop has no special case. Planning is **pure**: the
canonicalizer applies the plan and reports it, which is what makes the
compatibility report list exactly the rules that ran.

### Composing an instant

`compose_instant` handles the three shapes a spreadsheet gives a clock column:
clock-shaped text (`"10:35:19"`, `"10:00"`), a `datetime.time`, and Excel's own
representation of a time on a placeholder date. A row whose **time** is
unreadable keeps its date at midnight — the calendar fact was supplied and
dropping the whole instant would silently remove the row from the graph. A row
whose **date** is unreadable stays `NaT`; there is no instant to place it at.

Verified against the real SAML-D upload: composing its `Date` and `Time`
columns reproduces its own `transaction_datetime` column exactly, row for row.

### `amount_usd` is not derived over a contradiction

`_derive_amount_usd` refuses when the currency assessment blocks. The
compatibility gate already stops such a workspace from reaching inference, but
without this the canonical artifact on disk would hold euros in a column named
`amount_usd` — an invented FX rate of exactly 1.0. Refused at the point of
derivation as well as at the gate.

---

## Stage 4 — Canonicalization & compatibility validation — **CURRENT**

Implemented in `src/datasets/canonicalizer.py`.

### Canonicalization is driven by the mapping manifest

`canonicalize_and_validate` reads `mapping.json` and projects source columns into
canonical names via a fixed `concept_to_col` table. Only mappings with status
`AUTO_ACCEPTED` or `NEEDS_CONFIRMATION` are projected; `CONFLICT` mappings are
recorded as invalid. Multiple source columns mapped to one concept are
concatenated with a space.

### Derived fields

Which fields are computed rather than mapped is decided by
[`src/datasets/derivation.py`](#stage-35--deterministic-derivation--current).
This layer applies the plan and records each derivation in `field_statuses`
with the rule that produced it — a derivation's rule describes the field better
than the mapping does, so it takes precedence over the generic
"concatenated *N* source columns" description.

### Per-field compatibility states

| State | Meaning |
|---|---|
| `AVAILABLE` | Directly mapped with high confidence |
| `DERIVABLE` | Computed from other fields by a stated rule |
| `NEEDS_REVIEW` | Mapped with medium confidence, or conflicting sources |
| `MISSING` | No supported source field found |

### Validation runs the real pipeline

Compatibility is not approximated. The canonical frame is pushed through the
**actual** `preprocess` → `build_graph` → `compute_baseline` → `compute_multigraph`
code, and the result compared against the checkpoint.

### The fixed 167-feature contract

This is the gate everything else exists to satisfy.

```text
  5   structural scalars
  4   entity_type_{Company,Person,Invalid,unknown}
151   bank_country_*        (frozen vocabulary)
───
160   baseline
  7   directed-multigraph
───
167   total — exact order required
```

The validator computes
`generated = list(baseline.feature_names) + list(multigraph.feature_names)` and
sets `feature_order_valid` only on **exact list equality** with the checkpoint's
`metadata["feature_names"]` — including order. Missing and extra features are
enumerated in `unavailable_features`.

Because the vocabulary is **frozen** rather than derived from the uploaded data
(see `docs/decisions.md`), an external dataset always produces exactly 160
baseline columns in the trained order. Out-of-vocabulary country codes are
ignored — they set no column and contribute no signal; there is no "other"
bucket, because the checkpoint has no such column. Unrecognised entity types map
to `entity_type_unknown`.

### Readiness — **CURRENT**

`status` and `readiness` answer two different questions and the report carries
both:

| Field | Question |
|---|---|
| `status` | **May this run?** `READY` / `INCOMPATIBLE` / `FAILED`. Gates inference. |
| `readiness` | **What will the model actually see if it does?** `FULL` / `DEGRADED` / `BLOCKED`. |

| Readiness | Meaning |
|---|---|
| `FULL` | Every critical inference input is available and every optional signal block is populated. |
| `DEGRADED` | Inference is mathematically valid, but the model sees less: an optional signal block is empty, part of the geography does not reach the frozen vocabulary, or a declared dimension is `NEEDS_REVIEW`. |
| `BLOCKED` | Genuinely required information is unavailable. Inference does not run. |

`readiness_reasons` states one fact per line. A `READY` dataset that supplies no
entity type and no resolvable geography is `DEGRADED`, and saying only `READY`
about it is a true statement about the contract and a misleading one about the
analysis.

`assess_readiness` is given the **finished report** and classifies from it, so
every reason it returns is a fact the report already states and the two cannot
drift apart.

### Compatibility report

`DatasetCompatibilityReport` is persisted to `compatibility.json` with
`status` ∈ {`READY`, `INCOMPATIBLE`, `FAILED`} plus `identity_status`,
`currency_status`, `geography_status`, `configuration_assessment` (the full
per-dimension verdict with reasons and evidence), `geography_normalization`,
`readiness`, `readiness_reasons`, per-field statuses, `feature_dimension`,
`feature_order_valid`,
`datatype_errors`, `value_quality_warnings`, and version stamps
(`model_version`, `feature_schema_version`, `canonical_schema_version`,
`validator_version`).

Status resolution:

* `FAILED` — dataset empty, or produces 0 valid edges;
* `INCOMPATIBLE` — missing required fields, graph requirements unmet,
  `feature_order_valid` false, or datatype errors;
* `READY` — everything satisfied.

A `READY` report always carries the standing warning *"Dataset characteristics
have not been statistically tested against training baseline."*

---

## Dataset configuration — **CURRENT**

`WorkspaceConfig` (`src/datasets/manifest.py`) carries operator decisions that
the system cannot reliably infer.

### Identity strategy

| `IdentitySource` | Meaning |
|---|---|
| `NAME` | Resolve entities by normalized counterparty name. **Default**, and the Reference Dataset setting. |
| `ACCOUNT_ID` | Resolve entities by account identifier. Intended for external datasets with stable identifiers and no usable names. |

**Implemented and verified** at the preprocessing and graph layers.
`preprocess(ingestion_result, identity_source=...)` switches both the identifier
columns and the missing-identifier guard, and names the active source in its
error messages. `tests/test_identity_mode.py` confirms that under `NAME` two
transactions from one sender collapse to a single node with out-degree 2, while
under `ACCOUNT_ID` they become two nodes each with out-degree 1 — and that the
feature matrix remains exactly 160 baseline columns either way.

The canonicalizer reads `identity_source` from workspace config, passes it to
`preprocess`, and marks `identity_status = INCOMPATIBLE` when the mapping does
not supply the concepts that source requires (names for `NAME`, accounts for
`ACCOUNT_ID`).

`prepare_workspace_detection_data` takes `identity_source` and both callers —
`predict_workspace` and `WorkspaceAnalysisResolver` — pass
`ws.manifest.dataset_config.identity_source`, so the graph inference scores and
the graph the dashboard displays are both built under the identity semantics the
compatibility report was issued for.

### Currency policy

| `CurrencyPolicy` | Handling |
|---|---|
| `NATIVE_USD` | Amounts are already USD; `amount_orig` is copied to `amount_usd` as a recorded `DERIVABLE` field. Requires a mapped amount concept, and is **contradicted** if the dataset's own currency column disagrees — or if it states a currency the mapping has not resolved. |
| `EXTERNAL_FX_REQUIRED` | `INCOMPATIBLE` — no FX provider is configured. See [The FX seam](#the-fx-seam). |
| `UNKNOWN` | `NEEDS_REVIEW`. Default. The assessment reports what the file's own currency column says, without declaring anything on the operator's behalf. |

#### An unresolved currency claim is not an absent one — **CURRENT**

The contradiction check consults the *mapped* `ORIGINAL_CURRENCY` concept. When
two columns tie for it and both drop to `CONFLICT`, the concept is unmapped, the
check finds nothing to look at and passes, and the reason line asserts *"The
dataset states no currency of its own"* about a file that states two. A
fifteen-currency book reached `READY` with euros and yen written into
`amount_usd`.

`policy.unresolved_concept_claims` closes this: a concept claimed by a column at
or above the `NEEDS_CONFIRMATION` bar, without being selected for it, is
reported as an unresolved claim. `NATIVE_USD` over an unresolved currency claim
is `INCOMPATIBLE` until the operator confirms which column states the currency.
Both the canonicalizer and the config route pass it, so the two surfaces cannot
disagree.

A second, independent guard sits at the derivation layer: `_derive_amount_usd`
refuses to fire when the currency assessment blocks, so a fabricated USD column
is never written even to the diagnostic `canonical.parquet` of a workspace that
will never reach inference.

#### Evidence is reported; the declaration stays the operator's — **CURRENT**

Under `UNKNOWN`, when the dataset's own currency column reads USD throughout,
the assessment says so — `supports_native_usd: true` in the evidence, and a
reason stating that declaring *Already USD* would be supported by the file
itself. When the column is mixed it says that instead, with the distinct count
and the USD share.

It still does not *set* the policy. The declaration rule holds that no evidence ever
makes a declaration on the operator's behalf, and a currency column is a claim
about `amount_orig` in a file only sampled in part — not a guarantee about every
row of it.

### Geography policy

| `GeographyPolicy` | Handling |
|---|---|
| `PAYER_AND_BENEFICIARY_SEPARATE` | Distinct payer/beneficiary bank countries from the mapping. `INCOMPATIBLE` unless **both** sides are mapped — one side's geography is never copied onto the other. Values are resolved into the frozen vocabulary; see [Geography normalization](#geography-normalization). |
| `SINGLE_DOMESTIC_COUNTRY` | Both columns filled from `single_country_code`; `NEEDS_REVIEW` if that code is unset or does not resolve into the frozen vocabulary |
| `UNSPECIFIED` | Permitted; yields an all-zero country block, reported as `NEEDS_REVIEW`. Default. No geography normalization runs, because no geography was claimed. |

---

## Configuration assessment — **CURRENT**

Recording a decision is not the same as being able to honour it.
`src/datasets/policy.py` owns one question:

> Does the uploaded dataset actually support what the operator declared?

It answers per dimension with a **status**, a **stated reason**, and the
**evidence** the answer rests on. Three surfaces consult the same function, so
they cannot disagree:

| Consumer | Uses it to |
|---|---|
| `canonicalize_and_validate` | decide `READY` / `INCOMPATIBLE` |
| `GET`/`POST /api/workspaces/<id>/config` | validate before persisting, and report |
| the Dataset Configuration UI | render status, reason and evidence |

### Evidence comes from the file, not the declaration

`collect_dataset_evidence` summarises the mapped columns of the real upload:
non-null coverage, distinct count, and bounded value counts. The canonicalizer
passes the full frame (`is_exact=True`); the config route passes the same
bounded sample profiling uses (`is_exact=False`). When no evidence can be read
the assessment reports `evidence_available: false` rather than assuming the
declaration holds.

### What evidence can establish

| Dimension | Blocking (`INCOMPATIBLE`) | Reviewable (`NEEDS_REVIEW`) |
|---|---|---|
| Identity | required concepts for the chosen strategy are not mapped | a mapped identity column is under 95% populated in the sample (those rows are dropped in preprocessing) |
| Currency | `NATIVE_USD` while the dataset's own currency column reports non-USD rows; `EXTERNAL_FX_REQUIRED` with no provider; `NATIVE_USD` with no amount mapped | `UNKNOWN` |
| Geography | `PAYER_AND_BENEFICIARY_SEPARATE` with either side unmapped | values that do not resolve into the frozen vocabulary; `single_country_code` unset or unresolvable; `UNSPECIFIED` |

`NEEDS_REVIEW` is deliberately **not** a blocker. `CurrencyPolicy.UNKNOWN` and
`GeographyPolicy.UNSPECIFIED` are the defaults for any workspace whose
configuration was never set, so treating `NEEDS_REVIEW` as a downgrade would
demote every existing workspace.

The full assessment is persisted in `compatibility.json` under
`configuration_assessment`, and `identity_status` / `currency_status` /
`geography_status` are taken directly from it.

### Configuration surface — **CURRENT**

`GET /api/workspaces/<id>/config` returns the current declaration, the
assessment, the selectable options (derived from the frozen contract, not
hard-coded), the mapped concepts, and the evidence. `POST` accepts
`identity_source`, `currency_policy`, `geography_policy` and
`single_country_code`, rejects an unknown enum value with **HTTP 400**, applies
a payload **atomically** (a valid field alongside an invalid one is not
persisted), and returns the same view model `GET` returns.

The **Dataset Configuration** panel on the Schema Mapping screen renders that
view model: one card per dimension, each showing the selectable options with
what each requires, the saved verdict and its reason, and the evidence —
observed currencies, unresolved geography values, missing concepts. Choices are
local until **Confirm Configuration** posts them, so a displayed verdict always
describes a saved configuration. The country picker offers only frozen-vocabulary
countries, so an operator cannot select a value that would set no feature.

---

## Geography normalization — **CURRENT**

`src/datasets/geography.py` resolves an external notation into the frozen
151-entry country vocabulary under four named rules:

| Rule | Example |
|---|---|
| `EXACT` | `"GB"` → `GB`; `"33"` → `33`; `".D"` → `.D` |
| `CASE_NORMALIZED` | `"gb"`, `"  ES  "` → `GB`, `ES` |
| `ISO_ALPHA3` | `"GBR"` → `GB` |
| `ISO_NAME` | `"United Kingdom"`, `"UK"`, `"Spain"` → `GB`, `GB`, `ES` |

**Nothing is invented.** A value no rule resolves — `"Morocco"`, `"United
States"`, `"Japan"`, `"Atlantis"` — is `UNRESOLVED` and left in the data
**exactly as supplied**: not blanked, not bucketed, not substituted, and never
derived from the other side of the transaction. Its all-zero feature behaviour
is unchanged and the raw value remains visible for review.

Resolution never leaves the frozen vocabulary. One ISO 3166-1 table is
maintained and **split by membership of that vocabulary**, so which half a
country belongs in is answered by the checkpoint rather than by a maintainer's
judgement. Non-ISO members of the vocabulary (`JH`, `OS`, `TA`, `TX`, `G8`,
`R0`, `T1`) are given no name aliases, because what they denote has never been
established.

### Recognised, but the checkpoint has no column for it — **CURRENT**

`"United States"`, `"USA"` and `"US"` all denote a real country, and the frozen
151-entry vocabulary contains no `US`. Reported as plain `UNRESOLVED`, they are
indistinguishable from `"Atlantis"` or a corrupted cell — and those are not the
same fact:

* *"We know exactly what this says and the model has no column for it"* is a
  limit of the checkpoint. Nothing in the data can fix it.
* *"This is not a country we recognise"* is a problem with the data, which an
  operator can actually act on.

A value matching the out-of-vocabulary half of the ISO table is therefore
**recognised** — `CountryResolution.recognized_as` carries the code it denotes,
and `recognized_but_unsupported` is true — and is otherwise treated exactly as
before: `code` stays `None`, `rule` stays `UNRESOLVED`, `resolved` stays
`False`, the value stays in the data byte-identical, and it sets no feature
column.

**This is not a mapping into the vocabulary.** There is no nearest-neighbour, no
regional bucket and no substitution; inventing one would place a transaction in
a country it is not in. The report's `recognized_out_of_vocabulary` is a labelled
**subset of** `unresolved_values`, never a replacement for it, and the
configuration assessment states the same distinction in its reason text.

Normalization runs in the canonicalizer, on external-dataset values only, and
only when a geography policy was declared. A `GeographyNormalizationReport` per
column — what was resolved, under which rules, and what was not — is persisted
in `compatibility.json` under `geography_normalization`.

**The frozen contract is untouched.** `src/features/baseline.py`, the 151-entry
vocabulary, the feature ordering and the checkpoint are not modified. The
Reference Dataset does not pass through the workspace canonicalization path;
its own codes resolve only under `EXACT` and pass through byte-identical.

---

## The FX seam

`src/datasets/fx.py` declares the shape of a future FX layer and **implements no
conversion**. `get_fx_provider()` returns `None`; `fx_availability()` supplies
the operator-facing reason shown in the configuration panel, and the currency
assessment consults it rather than hard-coding the refusal.

`FxQuote` makes provenance structurally mandatory — `source`, `as_of`,
`retrieved_at` — because a converted amount that cannot say where its rate came
from is not evidence. The `FxRateProvider` protocol requires implementations to
be historical and to **raise rather than interpolate**, so a missing quote
reports the dataset incompatible instead of silently converting at a nearby rate.

Until a provider exists, `EXTERNAL_FX_REQUIRED` stays blocking and `amount_usd`
is never fabricated for a multi-currency dataset.


## Stage 5 — Workspace inference — **CURRENT**

`src/datasets/inference.py::predict_workspace`:

1. Requires a persisted `compatibility.json` with `status == "READY"`; refuses
   otherwise.
2. Sets `prediction_status = RUNNING`.
3. Rebuilds detection state from `canonical.parquet` — preprocess, graph,
   baseline + multigraph features, PyG tensor.
4. Scores with the frozen checkpoint via `predict_from_prepared_data`.
5. Writes `predictions.parquet`.
6. Assesses and writes `trust.json` — see
   [Prediction trust](#prediction-trust--current). This reads the same feature
   matrix the model consumed; it never alters a prediction.
7. Stamps `prediction_status = COMPLETE`, `prediction_timestamp`, and
   `model_version`; sets `FAILED` on any exception.

The UI consumes this through `WorkspaceAnalysisResolver`, which requires
`prediction_status == "COMPLETE"` and raises `WorkspaceNotReadyError` otherwise.

### Prediction route

`POST /api/workspaces/<workspace_id>/predict` triggers a run. It delegates
entirely to `predict_workspace()` — no inference logic is duplicated in the
route. A non-`READY` workspace is refused with **HTTP 409**; an unknown
workspace with **404**. On success the resolver's cached context for that
workspace is invalidated so the dashboard shows the new run.

Inference is **never** triggered implicitly. Startup does not re-run it;
`predictions.parquet` is read back from disk as-is.

---

## Prediction trust — **CURRENT**

The frozen checkpoint was trained on the Reference Dataset alone. Scoring an
arbitrary uploaded dataset produces a number for every entity regardless of
whether that dataset resembles anything the model has seen, and the dashboard
would otherwise render it with the same visual authority.

`src/datasets/trust.py` turns **observable facts about the run** into signals
with stated rules, persisted to `trust.json` next to `predictions.parquet`.

| Signal | Measured from |
|---|---|
| `flag_rate` | share of entities scoring above the checkpoint's 0.45 threshold |
| `zero_block_geography` | all 151 `bank_country_*` features zero across every entity |
| `zero_block_entity_type` | every entity in `entity_type_unknown` (non-zero, but no signal) |
| `geography_unresolved_<column>` | per column, how many raw geography values resolved into no country |
| `degraded_features` | optional signal columns the dataset never supplied |
| `labels` | whether the compatibility report lists any ground-truth column |
| `distribution_untested` | always present; names the standing limitation |

The last two read facts the compatibility report already recorded during this
run, passed through by `predict_workspace`. They exist because the zero-block
checks are all-or-nothing:

* A geography column that resolves only **partly** exercises the block, so no
  zero-block signal fires — yet every unresolved row carries an all-zero
  country vector while appearing, in the source data, to have a country. Any
  unresolved value is reported so the fact is never hidden; below 95% resolved
  it is raised from `INFO` to `CAUTION`. A hard cutoff instead would make
  thousands of countryless rows invisible for being a small enough share.
* `degraded_features` names *which* optional columns were absent, so
  `ENTITY TYPE BLOCK EMPTY` is accompanied by the reason it is empty.

Each signal carries **observed / why it matters / rule**, the same shape as the
analyst-facing heuristic flags, so the alert-rate threshold reads visibly as an
operational heuristic and never as a statistical test.

**Nothing is compared against a training-distribution statistic.** The
checkpoint records neither class prevalence nor feature moments, so such a
comparison would have to be invented.

### Confidence levels

| Level | Meaning |
|---|---|
| `LOW` | an observed property makes the scores unsafe to read as an alert list (currently: ≥50% flagged) |
| `LIMITED` | an observed property limits what the scores mean (an empty feature block, no ground truth) |
| `UNVERIFIED` | nothing observed argues against the scores; the dataset is still untested against the training distribution |

There is deliberately **no `HIGH`**: nothing available at inference time can
establish that a frozen model transfers to an unseen dataset.

### Trust never touches a prediction

The assessment runs *after* `predictions.parquet` is written and never
re-scores, re-thresholds, reorders or suppresses anything. `risk_score` and
`predicted_class` are exactly what the frozen model produced, and
`predictions.parquet` gains no trust column. A failure to write `trust.json`
cannot fail a prediction; its absence is itself visible.

`GET /api/workspaces/<id>/trust` returns the assessment, **404** when a run
predates it or none exists. The Dataset Hub shows a confidence pill per
workspace; the workspace dashboard shows a banner with the expandable signal
list; the entity panel adds a `LOW` / `LIMITED` caveat beside the risk score
itself, so an analyst who scrolled past the banner still sees it.

---

## The inference contract

`READY` is gated on what inference genuinely needs — **not** on the Reference
Dataset's labelled-training schema. `src/data/ingestion.py` partitions the
canonical schema into disjoint requirement classes:

| Class | Columns | Required for READY? |
|---|---|---|
| `INFERENCE_REQUIRED_BASE` | `date`, `amount_usd` | **Yes** |
| `IDENTITY_REQUIRED_COLUMNS["NAME"]` | `payer_name_norm`, `beneficiary_name_norm` | Yes, under `NAME` |
| `IDENTITY_REQUIRED_COLUMNS["ACCOUNT_ID"]` | `payer_account`, `beneficiary_account` | Yes, under `ACCOUNT_ID` |
| `INFERENCE_DEGRADABLE_COLUMNS` | `payer_type`, `beneficiary_type`, `payer_bank_country`, `beneficiary_bank_country` | No — degrade |
| `LABEL_COLUMNS` | `payer_core`, `beneficiary_core`, `investigation` | **No — labels** |
| `OPTIONAL_METADATA_COLUMNS` | the remaining 10 | No |

`inference_required_columns(identity_source)` returns the base set plus the
identity columns for the selected strategy.

**Labels never gate READY.** `payer_core` / `beneficiary_core` are the model's
prediction *target* and `investigation` is analyst metadata; the 167 features
consume none of them. An unlabelled dataset is a fully supported case.

**Degradation is reported, not silent.** When an optional signal column is
absent the 167-vector is still exactly the right width and order, with
`entity_type_unknown = 1` and a zeroed 151-column country block. The report
records this in `degraded_features` and raises a quality warning. A dataset that
*maps* bank country using values outside the frozen vocabulary (e.g. country
names `UK` / `Spain` rather than ISO `GB` / `ES`) is warned about separately,
since its geographic block is also entirely zero.

**Absence is never filled in.** Genuinely-absent non-inference columns are
materialised as explicit `NA` so Layer 2 can continue validating against the
full canonical schema. `NA` is the honest representation of "no data": nothing
is defaulted to `0`, `"US"`, `False`, or any other substantive value, and no FX
conversion is invented. Datatype errors are raised only for columns the dataset
actually supplied.

**Ground truth is preserved when present.** Supplied labels are carried into
`canonical.parquet` verbatim and listed in `labels_present`. They are kept
strictly separate from model output: `predictions.parquet` contains
`risk_score` and `predicted_class` plus observed structural features, and never
a label column. Neither ever overwrites the other.

---

## Verified external dataset results

Each dataset below was pushed through the actual workspace path — upload,
profile, **automatic mapping with no operator override**, canonicalization,
compatibility validation. Pinned by `tests/test_real_datasets.py`, which skips
when a file is not present (`.workspaces/` is git-ignored and the Reference
Dataset's source workbook lives above the repository).

The tests read each file to a **row bound** (20,000) rather than in full: a
bound truncates, it does not alter. No value is changed, no column added or
removed, and the schema under test is the file's own.

**Two of the three are BLOCKED**, for reasons the repository can state
precisely. Nothing was adjusted to make them pass.

> **Correction to a previous claim.** This document previously reported SAML-D
> as *"uploaded raw, unmodified and un-enriched"* and stated that *"the file
> itself supplies an `amount_usd` column"*. The file in the runtime workspace is
> an **enriched** copy: this project's own canonical columns (`amount_usd`,
> `amount_eur`, `payer_core`, `beneficiary_core`, `payer_type`, `purpose`,
> `investigation`, `source_file`, `transaction_datetime`, `payer_jurisdiction`)
> had been added to it. Published SAML-D has twelve columns and none of those.
> The result below is for SAML-D **as published**, recovered by projecting onto
> the columns the dataset actually ships.

### SAML-D as published — BLOCKED, on a real blocker

| Property | Determined from the file |
|---|---|
| Identity | `ACCOUNT_ID` — `Sender_account` / `Receiver_account`, no names. `identity_status = READY` |
| Amount | `Amount`, mapped automatically |
| Currency | `Payment_currency` — several currencies (`UK pounds`, `Dirham`, ...). **No USD amount** |
| Time | `Date` **+** `Time`, composed into one instant |
| Geography | `Sender_bank_location` / `Receiver_bank_location`, country names |
| Optional signals | No entity type — `payer_type` / `beneficiary_type` degraded |
| Labels | `Laundering_type` → `investigation` (preserved, not required) |

Result: **`INCOMPATIBLE` / `BLOCKED`**, `missing_required_fields == ["amount_usd"]`.

Everything else is satisfied: the graph builds, `graph_requirements` is empty,
and the feature matrix is **exactly 167 wide in the checkpoint's order**. What
is missing is a USD amount, and producing one would need dated, attributable FX
rates the repository does not have and will not invent.

The whole schema is mapped **automatically**, including the split instant that
previously annihilated the timestamp concept:

| Source column | Proposed concept | Note |
|---|---|---|
| `Sender_account` / `Receiver_account` | `SENDER_ACCOUNT` / `RECEIVER_ACCOUNT` | |
| `Amount` | `TRANSACTION_AMOUNT` | |
| `Payment_currency` | `ORIGINAL_CURRENCY` | |
| `Received_currency` | *(none)* | received, not sent — demoted, then beaten |
| `Sender_bank_location` / `Receiver_bank_location` | `SENDER_BANK_COUNTRY` / `RECEIVER_BANK_COUNTRY` | |
| `Date` **and** `Time` | both `TIMESTAMP` | complementary, composed |
| `Payment_type` | `PURPOSE` | |
| `Laundering_type` | `INVESTIGATION` | |

Geography resolves at **99.3%** (`payer_bank_country`) and **95.6%**
(`beneficiary_bank_country`) into the frozen vocabulary under `ISO_NAME`
(`UK` → `GB`, `Spain` → `ES`, `Germany` → `DE`, ...). The remainder — `USA`,
`Japan`, `Morocco`, `Nigeria`, `Albania`, `India`, `Mexico`, `Pakistan` — are
real countries the checkpoint has no column for, and are now reported as such
(`recognized_out_of_vocabulary`) rather than filed alongside unreadable data.

**Historical note.** The enriched copy, which supplies its own `amount_usd`,
does reach `READY` and was scored end to end: 65,168 account entities over
100,000 transactions, predictions persisted and reloaded after restart. That run
exercised the pipeline, but roughly 70% of its entities scored above the
checkpoint's threshold — an alert rate no usable system produces — so it was
reported at confidence `LOW`. It is not evidence of model validity. See
[Prediction trust](#prediction-trust--current).

### IBM AML — BLOCKED, on the same real blocker

| Property | Determined from the file |
|---|---|
| Identity | `ACCOUNT_ID` — legitimate: no names, stable accounts. `identity_status = READY` |
| Currency | **15 distinct currencies; ~40% US Dollar** |
| Geography | None. `payer_bank_account` holds 78,394 distinct **bank** identifiers |
| Optional signals | Neither entity type nor bank country — all four degraded |
| Labels | `is_suspicious` (all `1` in this sample) |

Result: **`INCOMPATIBLE` / `BLOCKED`**, `missing_required_fields == ["amount_usd"]`.
Declaring `NATIVE_USD` over it is separately refused with
`currency_status = INCOMPATIBLE`, and `amount_usd` stays null in
`canonical.parquet`.

Two mapping behaviours this dataset pins:

* `payer_bank_account` / `beneficiary_bank_account` are claimed by **no**
  concept. A bank token with no place token names the institution; reading it
  as the payer's own account would replace the entity population with banks,
  and reading it as a country would produce the same nonsense one level down.
* `currency_orig` wins `ORIGINAL_CURRENCY` over `currency_dest`, which is the
  *received* currency and a different quantity. Without that the two tied,
  both dropped to `CONFLICT`, and the `NATIVE_USD` contradiction check had
  nothing to look at.

Its geography stays `UNSPECIFIED` rather than duplicating an ambiguous
bank-account identifier into both sender and receiver country fields.

### Azerbaijani Laundromat (OCCRP) raw workbook — READY / DEGRADED

The Reference Dataset's own source workbook, run through the **external** path.
Included as a third real dataset with a genuinely different shape: named
counterparties rather than account ids, an explicit USD amount column, and
numeric bank-country codes.

| Property | Determined from the file |
|---|---|
| Identity | `NAME` — `payer_name` / `beneficiary_name` |
| Currency | `NATIVE_USD` — the file supplies `amount_usd`, mapped to `USD_AMOUNT` |
| Time | `date` |
| Geography | `payer_bank_country` / `beneficiary_bank_country`, **100% `EXACT`** — its codes are already frozen-vocabulary members and pass through untouched |
| Optional signals | All four supplied; `degraded_features == []` |
| Labels | `payer_core`, `beneficiary_core`, `investigation` — preserved, not required |

Result: **`READY`**, 167 features, order valid, no missing required field.
`readiness = DEGRADED` only because the geography verdict states that the
bounded evidence sample did not inspect every distinct value.

This is **not independent evidence of model validity** — the checkpoint was
trained on this data. It is evidence that the external path can read a real
schema, and it pins one specific failure: the raw workbook carries both
`payer_name` and `payer_name_norm`, which tied for `SENDER_NAME`, annihilated
each other as a `CONFLICT`, and left the dataset with no counterparty identity
at all. The plain column now wins and the pipeline re-derives `payer_name_norm`
under its own entity-resolution rule.

---

## Known defects

* `WorkspaceStatus.PARTIAL` is honoured as a blocker but is never produced by
  any current code path. Removing an enum member is a public-interface change,
  so it is left for an explicit decision.
* `POST /api/workspaces/upload` reports an ingestion refusal — a multi-sheet
  workbook, an empty file — as **HTTP 500**, though both are client errors that
  the extension check beside them reports as 400. The status is pinned by
  `tests/test_api_workspaces.py`; changing it is a public-interface change and
  is left for an explicit decision.
* `WorkspaceManifest.from_dict` splats the persisted dict into the dataclass, so
  a manifest carrying a key the current schema does not declare raises
  `TypeError` rather than being ignored. Failing loudly is deliberate — silently
  discarding persisted state would be worse — but it makes one bad manifest fail
  the whole hub listing.
* `normalize_country_series` matches on `str(value)`, so a numeric bank-country
  column that pandas read as `float64` (a code plus any blank cell) stringifies
  as `"33.0"` and resolves to nothing. Only reachable for a dataset using the
  Reference Dataset's own numeric codes, which no external dataset does.

Resolved:

* Workspace routes were dispatched by a `path.startswith(...)` /
  `path.endswith(...)` ladder, so `/api/workspaces/<a>/<b>/config` matched the
  config branch and read `<a>` as the workspace id. Routes are now matched on
  the whole, exactly-sized path segment list from one table, and the workspace
  id is captured by the pattern rather than re-extracted as `split('/')[3]` in
  seven places.
* The exception → HTTP status map was written out per route, so the routes that
  did not repeat it were silently inconsistent: an unknown workspace on
  `/mapping` returned 500. It is now stated once, with per-route additions only
  where a status genuinely differs (`/predict` reports a non-`READY` workspace
  as 409).
* `mapping.json` was rewritten by the HTTP route with a plain `open()`,
  bypassing the atomic-write discipline every other workspace artifact follows.
  Overrides now go through `WorkspaceManager.apply_mapping_overrides`, which
  also rejects an update naming a column the mapping does not contain instead
  of silently reporting success for an override that never happened.
* The canonicalizer re-implemented the workspace file read inline
  (`pd.read_csv` / `pd.read_excel`) rather than using `parsers`, so the
  extension dispatch existed twice and the canonicalizer's copy was
  case-sensitive — a `.CSV` upload took the Excel branch. It now calls
  `parsers.load_full`, the strict full-read counterpart to `load_sample`.
* Re-validating a workspace, or saving its configuration, now drops the cached
  analysis context. The resolver returns its cache before any freshness check,
  so a stale context previously survived a re-canonicalization indefinitely.
* A prediction frame with no `predicted_class` is reported as `LOW` with an
  explicit "alert rate not measurable" signal, instead of falling through to
  `UNVERIFIED` with a measured-looking 0%.
* An unreadable `trust.json` now raises instead of reading as "this run predates
  trust assessment" — the two are indistinguishable to the UI, and one of them
  silently removes a LOW-confidence warning.
* The geography verdict measures resolvability against the whole sample and
  states when only the most frequent values were inspected, rather than
  claiming "every sampled value resolves" from a bounded value-count.
* The one-shot `patch_*.py` and `extract_html.py` scratch scripts were removed
  from the repository root.
* `src/ui/app.py` now imports `parsers.ALLOWED_EXTENSIONS` rather than
  repeating the upload extension allowlist.
* The module docstring in `src/features/baseline.py` now describes the frozen
  vocabularies and the out-of-vocabulary policy it actually implements.
* Template bindings on nullable state are optional-chained; see
  `docs/UI_ARCHITECTURE.md`.

---

## Related documentation

* `docs/decisions.md` — decision records for workspace isolation, the fixed
  feature vocabulary, identity strategy, and compatibility gating
* `docs/architecture.md` — the full layered architecture and the external-dataset
  pipeline's place in it
* `docs/UI_ARCHITECTURE.md` — Dataset Hub, mapping review, and compatibility
  presentation
* `README.md` — current limitations, summarized
