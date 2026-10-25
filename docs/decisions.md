# Architectural Decision Records

This file records significant architectural decisions for the Graph-Based Mule
Account & Circular Transaction Detection system.

Each record states its context, decision, alternatives and consequences. Decisions are never rewritten when the
architecture later changes; a new decision is recorded that supersedes the
previous one.

**Status vocabulary:** `Accepted`, `Superseded`, `Proposed`.

These records were reconstructed on 2026-08-30 from the current source tree,
current tests, and the committed history. They document decisions that are
already reflected in the repository. Where a decision is only partially
implemented, that is stated explicitly rather than presented as complete.

---

# Decision: Freeze the Experiment 3 GraphSAGE Checkpoint as the Inference Model

Date: 2026-08-30 (recorded); decision enacted in `scripts/legacy/train_checkpoint.py` / `artifacts/models/experiment3_graphsage.pt`

Status: Accepted

Context:

The ablation study (Experiments 1–4, `docs/architecture.md` §28.3) produced four
GraphSAGE configurations over the reference dataset. Experiment 3
(Baseline + Directed-Multigraph, 167 features) was selected as the shipping
configuration. The UI and all demonstration paths must produce identical,
reproducible scores rather than retraining on every run.

Problem:

On-the-fly training makes inference non-deterministic, slow, and impossible to
regression-test. It also risks silently changing published scores whenever
feature code is touched.

Options considered:

1. Retrain the model on every inference run.
2. Persist a trained checkpoint and load it for inference.
3. Persist the checkpoint and additionally persist the derived feature matrix.

Decision:

Persist a single frozen checkpoint at
`artifacts/models/experiment3_graphsage.pt` containing `model_state_dict` and a
`metadata` block, and make it the only supported inference path.

The checkpoint metadata is the authoritative model contract:

```text
version            "1.0"
architecture       "BaselineGraphSAGE"
input_dim          167
hidden_dim         64
num_layers         2
feature_names      167 ordered feature names
seed               42
provenance         "Experiment 3, Fold 0"
training_config    {lr, weight_decay, dropout, max_epochs, patience}
threshold          0.45
```

Reasoning:

Reproducibility and regression-testability outweigh the flexibility of
retraining. Freezing the checkpoint turns "did the pipeline change?" into a
cheap, exact assertion.

Advantages:

* Deterministic, byte-stable predictions.
* Inference runs in seconds rather than minutes.
* Feature-engineering regressions are caught immediately and loudly.
* The UI can be redesigned with no risk to model behaviour.

Disadvantages:

* Any genuine feature change requires an explicit retrain and a new decision
  record.
* The checkpoint is a binary artifact that must be kept in the repository.

Consequences:

`src/detection/inference.py::predict_from_prepared_data` enforces the contract
before loading weights, and refuses to run on any mismatch:

* rejects the checkpoint unless `metadata["input_dim"] == 167`;
* rejects the data unless `pyg.feature_dim == metadata["input_dim"]`;
* rejects the data unless
  `list(baseline.features.columns) + list(multi.features.columns)` is **exactly
  equal**, including order, to `metadata["feature_names"]`.

The following must not be changed without a superseding decision and a retrain:
model architecture, checkpoint, feature ordering, graph semantics, inference
mathematics.

`tests/test_reference_regression.py` pins the frozen behaviour to exact values
(3440 entities; top entities `HILUX SERVICES LP` 0.56275397, `LCM ALLIANCE LLP`
0.55135107, `METASTAR INVEST LLP` 0.55129605, all `predicted_class == 1`).
Verified passing on 2026-08-30.

Alternatives rejected:

Retraining per run — rejected as non-reproducible and untestable.

---

# Decision: Replace the Dataset-Derived Country Vocabulary with a Fixed Experiment 3 Training Vocabulary

Date: 2026-08-30 (recorded); decision enacted in commit `bd4a089` "Freeze Experiment 3 feature vocabulary"

Status: Accepted

Context:

`bank_country_*` was originally built as a multi-hot indicator over **every raw
country code observed in the dataset being processed**. Entity type was
similarly one-hot encoded over observed categories. On the reference dataset
this yields 151 country columns and 4 entity-type columns, and therefore the 160
baseline features the checkpoint was trained on.

Problem:

A dataset-derived vocabulary is a function of the input data. Any dataset other
than the reference dataset produces a different number of columns, in a
different order. Because the frozen checkpoint requires exactly 167 features in
exactly one order (see the preceding decision), a dataset-derived vocabulary
makes the model structurally unusable on any other dataset — and, worse, could
silently produce a same-width but differently-ordered matrix whose predictions
are meaningless.

Options considered:

1. Keep the dynamic vocabulary and reject any dataset that does not reproduce it.
2. Freeze the vocabulary to the exact Experiment 3 training categories.
3. Add a learned embedding or hashing scheme for unseen countries.

Decision:

Freeze the vocabulary as module-level constants in `src/features/baseline.py`:

* `_KNOWN_ENTITY_TYPES = ("Company", "Person", "Invalid")`, plus an explicit
  `entity_type_unknown` column — 4 columns total.
* `_KNOWN_COUNTRY_CODES` — 151 literal codes as observed during Experiment 3
  training, including non-ISO variants (`.A`, `.D`, `.K`, `/M`, `00`–`99`,
  `G8`, `JH`, `R0`, `T1`, `TA`, `TX`).

Feature order is fixed as: 5 structural scalars → 4 entity-type columns → 151
country columns = **160 baseline features**. Appending the 7 directed-multigraph
features from `src/features/multigraph.py` gives the **167** the checkpoint
expects.

Reasoning:

The vocabulary is part of the trained model's contract, not a property of the
input data. It belongs with the model. Freezing it is the only option that makes
the same weight matrix meaningful across datasets.

Advantages:

* Feature width and order are now constant and dataset-independent.
* The reference dataset still produces a bit-identical matrix, so the frozen
  checkpoint remains valid and the regression test still passes.
* External datasets become *evaluable* against the contract rather than
  structurally impossible.

Disadvantages:

* Country codes not in the training vocabulary carry no signal.
* Entity types outside the three known categories are collapsed together.
* The constants are opaque literals whose provenance lives only in this record.

Consequences:

Two explicit out-of-vocabulary policies are now in force:

* **Unseen country codes are ignored.** They set no column. An entity whose
  countries are all unseen gets an all-zero country block. They are *not*
  bucketed into an "other" column, because no such column exists in the
  checkpoint.
* **Unseen entity types map to `entity_type_unknown`.** This column does exist
  in the checkpoint and absorbs them.

`tests/test_feature_contract.py` asserts both policies and asserts
`baseline.feature_names == checkpoint metadata["feature_names"][:160]`.

Known documentation defect: the module docstring of `src/features/baseline.py`
(the "Baseline feature set" list and the "Country encoding note") still describes
the superseded dynamic behaviour — "every raw bank-country code observed for this
entity", "Raw codes are preserved exactly". The **code** is correct; the
**docstring** is stale. Correcting it requires a source edit and is deferred to
the project owner.

Alternatives rejected:

Hashing / embedding unseen countries — rejected because it would change the
input dimension or the learned semantics, both of which would invalidate the
frozen checkpoint.

---

# Decision: Isolate Each External Dataset in a UUID-Addressed Workspace

Date: 2026-08-30 (recorded); decision enacted across commits `894ce5e` … `5bd35ff`

Status: Accepted

Context:

The system must analyse user-supplied datasets alongside the reference dataset
without the two contaminating each other, and must survive process restarts
mid-analysis.

Problem:

Deriving filesystem paths from user-supplied display names invites path
traversal and illegal-character injection. Holding dataset state in memory loses
all progress on restart. Sharing state between datasets risks cross-dataset
leakage of mappings or predictions.

Options considered:

1. A single global working directory reused per dataset.
2. Directories named after the user's display name.
3. UUID-addressed directories with a typed on-disk manifest.

Decision:

Each dataset analysis session is a **Workspace**: a directory named by a
strict UUID4 under a base directory, holding a typed `manifest.json` plus that
workspace's own artifacts. The user's display name is stored *only* as a
manifest property and never touches the filesystem.

Reasoning:

A UUID is unforgeable, path-safe, and collision-free. Persisting a typed
manifest makes state recoverable and inspectable.

Advantages:

* Path traversal is structurally impossible: `WorkspaceManager._is_safe_uuid`
  requires `str(uuid.UUID(id)) == id`, and `_get_path` additionally asserts the
  resolved path is a direct child of the base directory.
* Loading workspace B constructs a fresh object parsed from disk, so no state
  leaks from workspace A.
* Manifest and profile writes are atomic (write to `.tmp`, `flush`, `fsync`,
  then `replace`), so a crash mid-write cannot corrupt them.
* `Workspace._reconcile_state` recovers a workspace stranded in `ANALYZING` by
  detecting a complete `profile.json` and advancing it to `NEEDS_MAPPING`.

Disadvantages:

* Workspace IDs are not human-readable.
* Artifacts are duplicated per workspace.

Consequences:

Workspace status is a typed enum with the lifecycle `EMPTY` → `UPLOADED` →
`ANALYZING` → `NEEDS_MAPPING` → `READY` | `PARTIAL` | `INCOMPATIBLE` | `FAILED`.
Ingestion accepts `.csv`, `.xlsx`, `.xls` only, rejects multi-sheet workbooks and
empty files, preserves the upload byte-for-byte as `raw_data.<ext>`, and records
a SHA-256 hash. Profiling is bounded to a row sample (default 1000) to prevent
OOM.

Alternatives rejected:

Display-name-derived directories — rejected on security grounds.

---

# Decision: Separate Structural Profiling from Semantic Mapping, and Semantic Mapping from Canonicalization

Date: 2026-08-30 (recorded); decision enacted across commits `894ce5e` … `58dc74b`

Status: Accepted

Context:

An arbitrary uploaded spreadsheet has unknown column names and unknown
semantics. Its columns must reach the canonical schema
(`src/data/ingestion.py::REQUIRED_COLUMNS`) before any graph or tensor is built.

Problem:

Guessing column meaning inline during ingestion would bury an unreviewable,
unexplainable decision inside the data path, and would let a wrong guess
propagate silently into graph topology and model input.

Options considered:

1. Infer column meaning inline during ingestion.
2. Require the user to hand-write a mapping file.
3. Three explicit stages — profile structurally, then propose semantics with
   confidence, then canonicalize from the reviewed manifest.

Decision:

Adopt three separate stages with separate on-disk artifacts:

* **Profiling** (`profiler.py` → `profile.json`) records *structure only*:
  dtype, `null_ratio`, `uniqueness_ratio`, min/max, samples, normalized and
  original column names. It makes no semantic claim.
* **Semantic mapping** (`mapper.py` → `mapping.json`) proposes a
  `CanonicalConcept` per source column with a continuous confidence score built
  from lexical aliasing, type compatibility, cardinality alignment, and pattern
  hints. Each candidate retains its evidence components.
* **Canonicalization** (`canonicalizer.py` → `canonical.parquet`,
  `compatibility.json`) projects source columns into canonical names strictly
  according to the reviewed manifest.

Reasoning:

Each stage has one job and one reviewable artifact. Ambiguity is surfaced as
status rather than resolved by guessing.

Advantages:

* Mapping decisions are explainable: the UI can show *why* a column scored as it
  did, from the retained evidence components.
* Ambiguity is explicit — a column matching both `SENDER_ACCOUNT` and
  `RECEIVER_ACCOUNT` becomes `NEEDS_CONFIRMATION`; two columns claiming one
  concept become `CONFLICT`; irrelevant columns become `UNMAPPED` rather than
  polluting the tensor.
* Canonicalization is a pure projection of a reviewed manifest, so it can be
  re-run deterministically.

Disadvantages:

* Three artifacts and three status vocabularies to keep coherent.
* Requires human review for anything not auto-accepted.

Consequences:

`MappingStatus` is `AUTO_ACCEPTED` | `NEEDS_CONFIRMATION` | `UNMAPPED` |
`CONFLICT`. Only `AUTO_ACCEPTED` and `NEEDS_CONFIRMATION` mappings are projected;
`CONFLICT` blocks auto-acceptance. Some canonical fields are *derived* rather
than mapped, each recorded with an explicit rule in `field_statuses`:
`payer_name_norm` / `beneficiary_name_norm` (lowercase + alphanumeric strip),
`source_file` (from upload metadata), and `id` (deterministic row identifier).

Alternatives rejected:

Inline inference during ingestion — rejected as unexplainable and unreviewable.

---

# Decision: Make Entity Identity an Explicit Choice Between NAME and ACCOUNT_ID

Date: 2026-08-30 (recorded); decision enacted in commit `f9928f7` "Add explicit external dataset configuration"

Status: Accepted (partially implemented — see Consequences)

Context:

The reference dataset resolves entities by normalized counterparty name
(`payer_name_norm` / `beneficiary_name_norm`). Many external transaction
datasets carry stable account identifiers and no usable names, or names too
inconsistent to resolve on.

Problem:

Entity identity determines graph node identity, and therefore degree, flow,
reciprocity, and every structural feature the model consumes. Silently
defaulting to name resolution on an account-keyed dataset produces a graph whose
topology is wrong in a way no downstream validation would catch.

Options considered:

1. Always resolve on name.
2. Auto-detect the better identifier from data characteristics.
3. Make identity an explicit, recorded configuration choice.

Decision:

Introduce `IdentitySource` with exactly two members, `NAME` and `ACCOUNT_ID`,
and thread it through entity resolution as an explicit parameter.

`NAME` remains the default and the reference-dataset setting, preserving
existing behaviour and the frozen regression.

Reasoning:

Identity strategy is a property of the dataset that the operator knows and the
system cannot reliably infer. Auto-detection would hide a decision that changes
the meaning of every downstream number.

Advantages:

* `preprocess()` keeps `IdentitySource.NAME` as its default, so the reference
  path and the frozen checkpoint are entirely unaffected.
* The guard that rejects missing identifiers now checks *the columns actually
  being used* — `payer_account` / `beneficiary_account` under `ACCOUNT_ID` — and
  names the active identity source in its error message.
* Verified by `tests/test_identity_mode.py`: under `NAME`, two transactions from
  one sender collapse to a single node with out-degree 2; under `ACCOUNT_ID`
  they become two distinct nodes each with out-degree 1, and the resulting
  feature matrix still has exactly 160 baseline columns.

Disadvantages:

* Requires an operator decision per external dataset.
* Two enums named `IdentitySource` now exist — one in
  `src/data/preprocessing.py` (consumed by the pipeline) and one in
  `src/datasets/manifest.py` (persisted in workspace config). They are
  structurally identical and interoperate only because both are `str` enums.
  This duplication is a known defect.

Consequences:

Implemented and verified:

* `IdentitySource` in `src/data/preprocessing.py`; `preprocess()` accepts it and
  switches the identifier columns and the missing-identifier guard accordingly.
* `build_graph` consumes the resulting entity IDs unchanged.
* `canonicalizer.canonicalize_and_validate` reads
  `manifest.dataset_config.identity_source` and passes it to `preprocess()`, and
  validates that the mapping supplies the concepts that identity source needs.

Not yet implemented:

* `src/datasets/inference.py::prepare_workspace_detection_data` calls
  `preprocess(ing_res)` **without** `identity_source`, so workspace inference
  always resolves on `NAME` regardless of configuration. Any `ACCOUNT_ID`
  workspace that reached inference would be scored on a graph built under the
  wrong identity semantics.
* No frontend surface exposes the identity choice; only the
  `POST /api/workspaces/<id>/config` endpoint sets it.

This decision is recorded as `Accepted` because the strategy is settled and the
core resolution path is implemented and tested. The gaps above are implementation
debt against an accepted decision, not an unsettled question.

---

# Decision: Gate External Datasets Behind an Explicit Compatibility Report

Date: 2026-08-30 (recorded); decision enacted in commits `58dc74b`, `b197de4`, `f9928f7`

Status: Accepted (superseded in part — the required-field rule was replaced by
*Separate the Inference Contract from the Training/Label Contract*, below)

Context:

The frozen checkpoint is only meaningful on a 167-feature matrix in exactly the
trained order. An external dataset may be missing fields, carry unparseable
types, or produce a degenerate graph.

Problem:

Running the model on a structurally invalid dataset would produce
confident-looking but meaningless risk scores.

Options considered:

1. Attempt inference and let it fail at tensor construction.
2. Check only that column count matches.
3. Produce an explicit, versioned compatibility report and refuse anything not
   `READY`.

Decision:

`canonicalize_and_validate` builds the canonical frame, runs it through the
*real* preprocessing, graph, and feature code, and emits a
`DatasetCompatibilityReport` to `compatibility.json` with an overall `status` of
`READY` | `INCOMPATIBLE` | `FAILED`, alongside per-field statuses, feature
dimension, `feature_order_valid`, datatype errors, and quality warnings.
`predict_workspace` refuses to run unless the persisted report says `READY`.

Reasoning:

Validation must exercise the same code inference will use, not a parallel
approximation of it.

Advantages:

* Compatibility is decided by actually constructing the graph and features.
* `feature_order_valid` is an exact list equality against the checkpoint's
  `feature_names`, so ordering drift cannot pass.
* Report carries `model_version`, `feature_schema_version`,
  `canonical_schema_version`, `validator_version` for auditability.
* Per-field `state` (`AVAILABLE` / `DERIVABLE` / `NEEDS_REVIEW` / `MISSING`)
  with source columns and rules drives explainable UI.

Disadvantages:

* Validation is as expensive as inference, since it does the same work.
* The required-field set is inherited from the reference dataset's schema and is
  stricter than the model mathematically needs.

Consequences:

The compatibility mechanism itself stands. Two defects that blocked it, and the
required-field rule it originally used, have since been resolved:

* The `TypeError` from constructing `DatasetCompatibilityReport` without
  `identity_status` / `currency_status` / `geography_status` is fixed; all three
  are populated, and an `INCOMPATIBLE` value in any of them now blocks `READY`.
* The `src.datasets` import failure (deleted `WorkspaceManifest` /
  `WorkspaceStatus`) is repaired.
* `missing_required` was computed against `ingestion.ALWAYS_PRESENT_COLUMNS`,
  which includes the ground-truth labels. That rule is **superseded** by the
  inference-contract decision below; it is no longer used for `READY`.

External-dataset `READY` inference is now delivered end to end and verified on a
real external dataset.

Alternatives rejected:

Column-count-only checking — rejected because a same-width, differently-ordered
matrix silently produces meaningless scores.

---

# Decision: Isolate the Presentation Layer from the ML Pipeline

Date: 2026-08-30 (recorded); decision enacted in commits `13c21e3` and earlier UI work

Status: Accepted

Context:

The dashboard is judge-facing and expected to be redesigned freely, while the
model behaviour must remain frozen and reproducible.

Problem:

A UI that can reach into the ML pipeline can accidentally retrain it, reorder
features, or blur the distinction between model output and heuristics.

Options considered:

1. Let UI code call feature and model code directly.
2. Route all UI access through a read-only adapter over a resolved context.

Decision:

`src/ui/` consumes the pipeline strictly through `WorkspaceAnalysisResolver`,
which returns an immutable `AnalysisContext` (`workspace_id`, predictions
DataFrame, graph). `DetectionAdapter` converts that context into JSON view
models. Everything outside `src/ui/` and `run_ui.py` is frozen from the UI's
perspective.

Reasoning:

A single narrow boundary makes "the UI cannot change model behaviour" checkable
rather than aspirational.

Advantages:

* Theme, layout, and components can be replaced without touching ML code.
* Normal UI execution never triggers training — it loads the checkpoint only.
* Preserves the required separation of *model output*, *observed structural
  features*, and *heuristic structural flags*.

Disadvantages:

* New UI data requires an adapter change as well as a template change.

Consequences:

`AnalysisContext` unifies the reference dataset and workspace datasets behind one
shape, so the same adapter and the same frontend serve both. Reference context is
resolved once and cached; workspace contexts are cached per ID and require
`prediction_status == "COMPLETE"`.

Known defect: `WorkspaceAnalysisResolver` defaults its base directory to
`artifacts/workspaces`, while `WorkspaceManager` (and every call site in
`src/ui/app.py`) defaults to `.workspaces`. Workspaces created through the UI are
written to `.workspaces/` and are not visible to the resolver.

---

# Decision: Anchor All Regression Testing on the Reference Dataset

Date: 2026-08-30 (recorded); decision enacted in commit `5bd35ff` and `tests/conftest.py`

Status: Accepted

Context:

The system has many layers whose outputs feed the frozen checkpoint. Any of them
could drift.

Problem:

Unit tests over synthetic fixtures cannot detect a change that alters real
feature values or graph topology at scale.

Options considered:

1. Synthetic fixtures only.
2. The full reference dataset as the primary anchor, with cheap fixtures for
   API- and UI-shaped tests.

Decision:

Treat the reference dataset (`../dkdata.json`, 3440 entities, 16,940
transactions, 4 labelled core entities) as the primary regression anchor, pinned
by exact expected outputs. Provide session-scoped fixtures so its expensive
construction happens once per run, plus a `small_fixture` for tests that do not
need scale.

Reasoning:

Exact-value regression on real data is the only check that catches semantic drift
through the whole stack.

Advantages:

* `test_reference_dataset_exact_regression` pins entity count and exact risk
  scores to 1e-5, catching any change in ingestion, resolution, graph, features,
  ordering, or model loading.
* Session-scoped fixtures plus module-level caches avoid recomputation.
* Markers (`fast`, `integration`, `ml`, `reference`) allow selective runs.

Disadvantages:

* The full suite is expensive (~14 minutes).
* The anchor dataset lives **outside** the repository at `../dkdata.json`.

Consequences:

The reference dataset is not vendored; `data/raw/` and `data/processed/` are
empty. `UIConfig.DATASET_PATH`, `test_reference_regression.py`, and all
`scripts/legacy/` scripts use `../dkdata.json`.

Known defect: `tests/conftest.py` fixtures `reference_dataset` /
`reference_features` load `data/dkdata.json`, which does not exist. Tests
depending on those fixtures cannot pass as written; the passing regression test
does not use them.

---

# Decision: Restrict External Dataset Ingestion to CSV / XLSX / XLS

Date: 2026-08-30 (recorded); decision enacted in commit `894ce5e`, hardened in `db05365`

Status: Accepted

Context:

Uploaded files are untrusted and may be large, malformed, or macro-bearing.

Problem:

Unrestricted parsing is a memory and security risk, and ambiguous multi-sheet
workbooks have no single obvious table.

Options considered:

1. Accept any format pandas can read.
2. Restrict to a small allowlist with bounded, guarded parsing.

Decision:

Allow exactly `.csv`, `.xlsx`, `.xls`. Reject anything else at both the HTTP
upload handler and `parsers.safe_ingest`.

Reasoning:

A narrow allowlist with bounded reads is the defensible default for untrusted
input.

Advantages:

* Multi-sheet workbooks are rejected with an explicit message rather than
  silently taking sheet 0.
* Empty uploads are rejected and the partial file removed.
* Hashing and copying are chunked (8 KB), so large files never load fully into
  memory.
* Profiling reads at most `max_rows` (default 1000) and reports
  `is_exact_count=False` when bounded.

Disadvantages:

* JSON, Parquet, and fixed-width sources are not accepted, even though the
  reference dataset itself is JSON and loads by a separate path.

Consequences:

`ALLOWED_EXTENSIONS` in `src/datasets/parsers.py` is the single authority;
`src/ui/app.py` duplicates the same list at the upload boundary. `xlrd>=2.0.1` is
required for `.xls` and is the sole entry in `requirements.txt`.

---

# Decision: Model External Currency and Geography as Explicit Policies

Date: 2026-08-30 (recorded); decision enacted in commit `f9928f7`

Status: Proposed

Context:

The 167 features consume `amount_usd` and raw bank-country codes. External
datasets may report a single native currency, or may carry no geography at all.

Problem:

Silently treating an arbitrary amount column as USD, or leaving geography
undefined, changes model inputs without recording that it happened.

Options considered:

1. Assume amounts are USD and geography is absent.
2. Record explicit policies alongside identity source in workspace config.

Decision:

Introduce `CurrencyPolicy` (`NATIVE_USD`, `EXTERNAL_FX_REQUIRED`, `UNKNOWN`) and
`GeographyPolicy` (`PAYER_AND_BENEFICIARY_SEPARATE`, `SINGLE_DOMESTIC_COUNTRY`,
`UNSPECIFIED`) with an optional `single_country_code`, carried on
`WorkspaceConfig` next to `identity_source`.

Reasoning:

These are operator knowledge, not inferable properties, and belong in recorded
configuration for the same reason identity source does.

Advantages:

* `NATIVE_USD` makes the amount reinterpretation explicit and records it as a
  `DERIVABLE` field with the rule "NATIVE_USD policy copies amount_orig to
  amount_usd".
* `EXTERNAL_FX_REQUIRED` is honestly reported as `INCOMPATIBLE`, since no FX
  conversion exists.
* `SINGLE_DOMESTIC_COUNTRY` fills both bank-country columns from one code, which
  only carries signal if that code is in the frozen 151-code vocabulary.
* `UNSPECIFIED` geography is permitted and yields an all-zero country block.

Disadvantages:

* Adds operator burden.
* A single domestic country outside the frozen vocabulary silently contributes
  nothing.

Consequences:

Status is **`Proposed`**, not `Accepted`, because the mechanism does not
currently execute. `WorkspaceConfig` exists and its logic is written into
`canonicalize_and_validate`, but:

* `WorkspaceConfig` is not a field of any manifest — `WorkspaceManifest` no
  longer exists in `src/datasets/manifest.py`, so `manifest.dataset_config` has
  no definition to resolve against;
* `WorkspaceConfig.from_dict` is a mis-copied `WorkspaceManifest.from_dict`: it
  is annotated `-> "WorkspaceManifest"`, references the undefined names
  `WorkspaceStatus` and `WorkspaceManifest`, and recurses on a `dataset_config`
  key that `WorkspaceConfig` does not have;
* the `POST /api/workspaces/<id>/config` endpoint writes to
  `ws.manifest.dataset_config`, which cannot resolve;
* no frontend surface exposes any of the three policies.

This decision should be promoted to `Accepted` only once the mechanism is
restored and exercised by a test.

---

# Decision: Separate the Inference Contract from the Training/Label Contract

Date: 2026-08-30

Status: Accepted

Context:

`ALWAYS_PRESENT_COLUMNS` was used as the required-field rule for external
dataset compatibility. It describes the schema of the Reference Dataset, which
is a *labelled training* dataset, and includes `payer_core`,
`beneficiary_core` (the prediction target) and `investigation` (analyst
metadata).

Problem:

A genuine unlabelled external dataset has no ground truth, so it could never
reach `READY` — even though the frozen model consumes none of those columns.
The rule also demanded `amount_eur`, `purpose`, `source_file`, both
jurisdictions and both raw name columns, none of which reach the 167 features.
The compatibility gate was therefore blocking datasets for reasons unrelated to
whether inference would work.

Options considered:

1. Leave as-is; external datasets remain training-shaped only.
2. Exempt labels only when the workspace is inference-only.
3. Split the schema into explicit requirement classes and gate `READY` on the
   inference class alone.

Decision:

Option 3. `src/data/ingestion.py` now partitions the canonical schema into four
disjoint classes, leaving `REQUIRED_COLUMNS` and `ALWAYS_PRESENT_COLUMNS`
untouched for training and evaluation:

```text
INFERENCE_REQUIRED_BASE        date, amount_usd
IDENTITY_REQUIRED_COLUMNS      NAME       -> payer_name_norm, beneficiary_name_norm
                               ACCOUNT_ID -> payer_account,   beneficiary_account
INFERENCE_DEGRADABLE_COLUMNS   payer_type, beneficiary_type,
                               payer_bank_country, beneficiary_bank_country
LABEL_COLUMNS                  payer_core, beneficiary_core, investigation
OPTIONAL_METADATA_COLUMNS      the remaining 10
```

`inference_required_columns(identity_source)` returns the base set plus the
identity columns for the selected strategy. `READY` is gated on exactly that.

Reasoning:

The requirement was established empirically, not by inspection: all 167
features are computable from a graph and edge table carrying only identity,
`date` and `amount_usd`. Entity type and bank country are *degradable* — when
absent the vector is still exactly 167 wide in exactly the trained order, with
`entity_type_unknown = 1` and a zeroed 151-column country block.

Advantages:

* An unlabelled, account-keyed dataset reaches `READY` and runs inference.
* The frozen model contract is untouched: still exactly 167 features, still
  validated by exact ordered list equality.
* Ground truth is preserved verbatim when supplied, and reported in
  `labels_present`, without ever being required.
* Degradation is reported rather than silent: absent optional signal, absent
  labels, and mapped-but-out-of-vocabulary country codes each raise a warning.
* Training and evaluation keep the stricter contract they need.

Disadvantages:

* Two contracts now coexist; a reader must know which applies.
* A dataset can reach `READY` while contributing little signal (all-zero
  country block, all-unknown entity type). This is surfaced as a warning, not
  a block, because it is a signal-quality judgement rather than a structural
  defect.

Consequences:

Canonicalization materialises genuinely-absent non-inference columns as
explicit `NA` so Layer 2 continues to validate against the full canonical
schema unchanged. `NA` is the honest representation of absence: no value is
invented — nothing is defaulted to `0`, `"US"`, `False` or any other
substantive value. Datatype errors are raised only for columns the dataset
actually supplied.

`DatasetCompatibilityReport` gained `labels_present` and `degraded_features`,
and every entry in `field_statuses` carries a `requirement` tag
(`INFERENCE_REQUIRED` / `LABEL_ONLY` / `OPTIONAL_SIGNAL` /
`OPTIONAL_METADATA` / `IDENTITY_ALTERNATE`).

Verified by `tests/test_inference_contract.py` (21 tests) and unchanged
Reference Dataset regression.

Alternatives rejected:

Deleting the label columns from the schema — rejected: they are genuinely
required for training and evaluation, and the Reference Dataset supplies them.

---

# Decision: Expose Workspace Inference Through an Explicit Prediction Route

Date: 2026-08-30

Status: Accepted

Context:

`predict_workspace()` was fully implemented but reachable only from tests. A
workspace could reach `READY` and never become viewable, because the resolver
requires `prediction_status == "COMPLETE"`.

Problem:

The chain stopped one link short of the already-promised dashboard.

Decision:

Add `POST /api/workspaces/<id>/predict`, delegating entirely to the existing
`predict_workspace()`. No inference logic is duplicated in the route: the READY
gate, status transitions and artifact persistence all remain owned by
`src/datasets/inference.py`.

Reasoning:

The missing piece was transport, not behaviour.

Advantages:

* Non-`READY` workspaces are refused with HTTP 409, driven by the existing
  `ValueError` from `predict_workspace`.
* A successful run invalidates the resolver's cached context for that
  workspace, so the dashboard reflects the new run rather than a stale one.
* Inference is never triggered implicitly: it runs only on this explicit call.

Consequences:

The Dataset Hub distinguishes *eligible to analyse* from *results exist*:
`can_predict` (compatibility `READY`) and `has_predictions` /
`prediction_status` are separate fields. The UI shows `NOT RUN` / `ANALYZING` /
`COMPLETE` / `FAILED`, offers **Run Analysis** only when `can_predict`, and
offers **Open Dashboard** only once predictions exist. Startup never re-runs
inference; persisted `predictions.parquet` is read back as-is.

---

# Decision: Assess the Declared Dataset Configuration Against the Actual Upload

Date: 2026-08-30

Status: Accepted

Context:

`WorkspaceConfig` records three decisions the system must not guess: identity
source, currency policy, geography policy. Recording a decision is not the same
as being able to honour it. Before this change the check for "can this dataset
support what was declared?" was three inline `if`-ladders inside
`canonicalize_and_validate`, the `POST /config` route performed no validation at
all, and no frontend surface existed — so the declarations were effectively
unreachable and unverified.

Problem:

Three consumers needed the same answer and none of them shared one. The
canonicalizer re-derived it, the route skipped it, and the UI could not ask.
Worse, the answer was derived purely from *which concepts were mapped*, never
from what the mapped columns actually contain — so a dataset could be declared
`NATIVE_USD` while its own currency column reported fifteen currencies.

Decision:

Introduce `src/datasets/policy.py` as the single owner of the question. It
returns a per-dimension status, a stated reason and the evidence the answer
rests on, and it is consulted by the canonicalizer, the `GET`/`POST`
`/api/workspaces/<id>/config` routes, and the configuration UI.

Evidence comes from a bounded sample of the **real uploaded file** — the same
bound profiling uses — never from the declaration. Where no evidence could be
read the assessment reports `evidence_available: false` instead of assuming the
declaration holds.

Reasoning:

The interface is one function; the implementation is the accumulated knowledge
of what each policy requires, derives, and can be contradicted by. Deleting it
would scatter that knowledge back across three call sites.

Advantages:

* A `NATIVE_USD` declaration the dataset's own currency column disproves is now
  `INCOMPATIBLE`, with the observed currency mix as evidence.
* `PAYER_AND_BENEFICIARY_SEPARATE` without both bank-country concepts mapped is
  now blocking. Previously it silently passed and produced an empty geographic
  block, because one side's geography is never copied from the other.
* The `/config` route validates enum values, rejects a payload wholesale rather
  than half-applying it, and returns the same view model `GET` returns.
* The UI derives its policy vocabularies and country list from the frozen
  contract instead of hard-coding them.

Consequences:

`NEEDS_REVIEW` remains deliberately non-blocking. `CurrencyPolicy.UNKNOWN` and
`GeographyPolicy.UNSPECIFIED` are the defaults for any workspace whose
configuration was never set, so treating `NEEDS_REVIEW` as a downgrade would
demote every existing workspace and break the already-promised `READY` path.

Alternatives rejected:

Inferring the configuration from the data — rejected: it is precisely the class
of decision the project requires to be declared, not guessed. Evidence may
*contradict* a declaration; it never *makes* one.

---

# Decision: Resolve External Country Notations Into the Frozen Vocabulary

Date: 2026-08-30

Status: Accepted

Context:

The Experiment 3 checkpoint encodes bank country against a frozen 151-entry
vocabulary matched by **exact string equality**, with unrecognised values
ignored (no "other" bucket). SAML-D expresses bank location as country *names*
("UK", "Spain"), so all 151 `bank_country_*` features were zero for every
entity and the dataset contributed no geographic signal — reported only as a
warning.

Problem:

`"United Kingdom"`, `"GBR"`, `"gb"` and `"GB"` denote the same country, but only
the last one reached the model.

Decision:

Introduce `src/datasets/geography.py`, which resolves a raw value into the
frozen vocabulary under four named rules — `EXACT`, `CASE_NORMALIZED`,
`ISO_ALPHA3`, `ISO_NAME` — and reports every value it could not resolve.
Resolution runs in the canonicalizer, on external-dataset values only, and only
when a geography policy was actually declared.

The frozen vocabulary, `src/features/baseline.py`, the feature ordering and the
checkpoint are **not** touched. The Reference Dataset does not pass through the
workspace canonicalization path and is therefore unaffected; its own numeric and
symbolic codes (`"33"`, `".D"`, `"/M"`) resolve only under `EXACT` and pass
through byte-identical.

Reasoning:

This is normalization of *notation*, upstream of a contract that is unchanged.
It is the one place in the geography path where a semantically safe improvement
was available without touching anything frozen.

Advantages:

* The alias table is restricted to codes that are both genuine ISO 3166-1 and
  present in the frozen vocabulary: resolving to a code the checkpoint has no
  column for would produce the same empty block while appearing to succeed.
* Non-ISO members of the vocabulary (`JH`, `OS`, `TA`, `TX`, `G8`, `R0`, `T1`)
  are given no name aliases, because the repository has never established what
  they mean.
* The country picker offers only resolvable countries, so an operator cannot
  select a value that sets nothing.

Consequences:

Country names outside the vocabulary — `"Morocco"`, `"United States"`,
`"Japan"` — remain unresolved and are left in the data **exactly as supplied**:
not blanked, not bucketed, not substituted. Their existing all-zero behaviour is
unchanged and the raw value stays visible for review. A
`GeographyNormalizationReport` per column records what was changed, what was
not, and under which rule.

Alternatives rejected:

Adding an "other" country bucket — rejected: the checkpoint has no such column.

Extending the frozen vocabulary — rejected: it invalidates the checkpoint
(the frozen model contract).

Deriving a missing side's geography from the side that is present — rejected:
that would invent sender or receiver geography the dataset never stated.

---

# Decision: Report Prediction Trust as a Persisted Artifact of Each Run

Date: 2026-08-30

Status: Accepted

Context:

The frozen checkpoint was trained on the Reference Dataset alone. On SAML-D it
scores roughly 70% of entities above its 0.45 threshold — a rate no usable
alerting system produces. The dashboard rendered those scores identically to
Reference Dataset scores: `SUSPICIOUS` in red, no caveat. Compatibility warnings
were written once to `compatibility.json` and never surfaced again.

Problem:

Out-of-distribution model output was presented with the same visual authority as
in-distribution model output.

Decision:

Introduce `src/datasets/trust.py`, which turns observable facts about a
completed run into signals with stated rules, persisted to `trust.json`
alongside `predictions.parquet` and surfaced in the Dataset Hub, a dashboard
banner, and the entity panel.

Measured per run: the share of entities flagged; which feature blocks were
identically zero or entirely `unknown`, i.e. how much of the 167-dimensional
input the dataset exercised; and whether ground truth exists at all.

Reasoning:

Every input is a fact about the run itself. Nothing is compared against a
training-distribution statistic, because the checkpoint records neither class
prevalence nor feature moments and one will not be invented (research-integrity rule).

Advantages:

* Signals share the shape of the existing analyst-facing heuristic flags —
  observed / why it matters / rule — so the alert-rate threshold is visibly an
  operational heuristic, never presented as a statistical test.
* An all-`unknown` entity-type one-hot is detected as carrying no signal even
  though it is not all-zero.
* The confidence ladder has no `HIGH`: its ceiling is `UNVERIFIED`, because
  nothing available at inference time can establish that a frozen model
  transfers to an unseen dataset.

Consequences:

The assessment never re-scores, re-thresholds, reorders or suppresses a
prediction: `risk_score` and `predicted_class` remain exactly what the frozen
model produced, and `predictions.parquet` gains no trust column. A failure to
write `trust.json` cannot fail a prediction — the assessment runs after
predictions are already persisted, and its absence is itself visible.

---

# Decision: Declare the FX Seam Without Implementing It

Date: 2026-08-30

Status: Accepted

Context:

IBM AML is denominated in 15 currencies with only 42% USD, so
`CurrencyPolicy.EXTERNAL_FX_REQUIRED` is the honest declaration — and it is
reported `INCOMPATIBLE`, because converting those amounts honestly needs dated,
attributable historical rates the repository does not have.

Problem:

"No FX support" was a hard-coded branch rather than a described capability, so
nothing said what a future FX layer would have to satisfy.

Decision:

Add `src/datasets/fx.py` containing an `FxRateProvider` protocol, an `FxQuote`
carrying mandatory provenance (`source`, `as_of`, `retrieved_at`), and
`get_fx_provider()`, which returns `None`. **No conversion is implemented.**
The currency assessment consults `get_fx_provider()` rather than hard-coding the
refusal, so introducing a real provider is a change in one module.

Reasoning:

Fixing the shape now costs one protocol and one nullable lookup, and makes the
refusal explain itself. Building conversion machinery with no rate source would
be dead code that invites someone to fill it with invented rates.

Advantages:

* `FxQuote` makes provenance structurally mandatory, not optional decoration:
  a converted amount that cannot say where its rate came from is not evidence.
* The protocol requires implementations to raise rather than interpolate, so a
  missing quote reports the dataset incompatible instead of silently converting
  at a nearby rate.

Consequences:

`fx_availability()` supplies the operator-facing reason shown in the
configuration panel. Until a provider exists, `EXTERNAL_FX_REQUIRED` stays
blocking, and `amount_usd` is never fabricated for a multi-currency dataset.

---

# Decision: Model Transaction Direction as Its Own Mapping Dimension

Date: 2026-08-30

Status: Accepted

Context:

`src/datasets/mapper.py` proposed a canonical concept for each source column
from two flat tables of whole phrases: `SPECIFIC_ALIASES` (e.g.
`"sender_account"`) and `GENERIC_ALIASES` (e.g. `"location"`, matching both
jurisdiction concepts symmetrically).

Problem:

Run against the real SAML-D upload, the mapper produced:

```text
Sender_bank_location     CONFLICT   top=SENDER_JURISDICTION 0.50
Receiver_bank_location   CONFLICT   top=SENDER_JURISDICTION 0.50
payer_type               UNMAPPED   top=SENDER_JURISDICTION 0.25
Date / Time / transaction_datetime   all CONFLICT
```

Three distinct failures:

1. **Direction was lost.** It was encoded only inside whole-phrase entries. A
   column using an unlisted noun (`originator`, `payee`, `debtor`) or an
   unlisted suffix (`bank_location`) fell through to the direction-blind
   generic alias, so a column literally named `Receiver_bank_location` scored
   highest as a *sender* concept, tied with the sender column, and both were
   discarded as `CONFLICT`.
2. **Four concepts were unreachable.** `SENDER_BANK_COUNTRY`,
   `RECEIVER_BANK_COUNTRY`, `SENDER_TYPE` and `RECEIVER_TYPE` had no entry in
   either table, so they could never clear the 0.2 retention floor. Those four
   feed 155 of the 167 features. Meanwhile `SENDER_JURISDICTION`, which the
   model does not consume, had aliases and won the geography columns.
3. **Conflict arbitration ignored score.** The *first* claimant of a concept
   was registered and every later one rejected regardless of score, and both
   were discarded — so a `Date` + `Time` pair alongside a real datetime column
   annihilated the timestamp concept entirely.

Options considered:

1. Add the missing phrases to `SPECIFIC_ALIASES`. Rejected: it is one entry per
   concept per dialect, it does not fix directionality for the *next* unlisted
   noun, and it leaves conflict arbitration broken.
2. Infer direction from the data (e.g. compare column value overlap). Rejected:
   this is semantic inference inside the data path, which the declaration rule
   forbids.
3. Model role and concept-family as two composable dimensions.

Decision:

Option 3. A role lexicon (sender-side / receiver-side tokens) composes with a
concept-family lexicon (`name`, `account`, `type`, `bank_country`,
`jurisdiction`, `core`). A matching role scores as strongly as an exact alias;
**a contradicting role scores negatively**, so a column naming one side is never
proposed as the other. Conflict arbitration ranks claimants by score, keeping a
clear winner and demoting the columns it beat to `NEEDS_CONFIRMATION` with
their selection cleared.

Reasoning:

Direction is a property of a column name that is independent of what the column
is about, so enumerating the cross product was the wrong shape. One lexicon
entry now generalises across every concept on that side.

Advantages:

* `payee`, `debtor`, `creditor`, `remitter` and `originator` are understood for
  every concept at once.
* `bank_country` outranks `jurisdiction` when a bank token is present, so the
  geography that actually feeds the model is preferred over the one that does
  not.
* Verified on the real SAML-D upload: bank location, entity type, both
  jurisdictions and the currency column now resolve, each on the correct side.
* Type and cardinality were demoted to corroborating evidence — a candidate
  with no lexical support is discarded, so a low-cardinality string column such
  as `source_file` is no longer offered as an entity type for merely repeating.

Disadvantages:

* Two lexicons must be kept coherent; a token added to the wrong one changes
  many concepts rather than one.
* A high-scoring column can now take a concept from a lower-scoring one, where
  previously both were rejected. This is reported, not silent: the beaten
  column keeps its candidate list and is marked `NEEDS_CONFIRMATION`.

Consequences:

A **tie remains a `CONFLICT`** and neither column is auto-selected.
Indistinguishable evidence is a genuine ambiguity for the operator to resolve,
and score-ranked arbitration must not manufacture a winner from one. A `Date` +
`Time` pair with no full-datetime column therefore still conflicts, which is
the honest outcome.

Nothing downstream of the mapping manifest changed: canonicalization,
the frozen 167-feature vocabulary, feature ordering and the checkpoint are
untouched, and human review remains the gate on anything below
`AUTO_ACCEPTED`.

Alternatives rejected:

Dataset-specific branches for known schemas — rejected outright: the goal is
broad AML/fraud schema support, not recognition of particular files.

---

# Decision: Make the Theme Module the Only Source of the Visual Language

Date: 2026-08-30

Status: Accepted

Context:

`docs/UI_ARCHITECTURE.md` named `src/ui/theme.py` the "single source of truth
for color palettes, graph appearance, and visual hierarchy", and told a future
maintainer to modify it to restyle the console. It was served at
`GET /api/theme`.

Problem:

`app.js` fetched `/api/theme` into `this.theme` and never read the result.
Editing `theme.py` changed nothing on screen. The palette that actually
rendered lived in two other places — an inline `tailwind.config` block in
`pages/index.html` and a hardcoded `premiumColors` object literal in `app.js` —
plus raw hex in a `<style>` element and roughly forty default-Tailwind classes
(`bg-slate-700`, `bg-rose-900`, `bg-emerald-800`, …) scattered through the
templates. The three sets disagreed. Applying a design system consistently was
not possible while the module that supposedly owned it had no consumers.

Decision:

Make the seam real, with exactly two adapters:

* the **document adapter** — `theme.css_variables()` renders the tokens as
  `:root` custom properties, injected into the page at compose time by
  `serve_index`. The Tailwind config resolves its palette from those variables.
* the **graph adapter** — `/api/theme` serves `THEME`, and `app.js` builds its
  Vis-Network options from `THEME["graph"]`, falling back to the injected
  custom properties rather than to a literal.

Solid colours are emitted twice — `--aml-<name>` and `--aml-<name>-rgb` — so
Tailwind opacity modifiers (`bg-system-surface/50`) still compose.

Reasoning:

One adapter would be a hypothetical seam; two are what make it real. Injecting
at compose time rather than fetching means the palette is correct on first
paint.

Advantages:

* A redesign is a change to one module, as the documentation already claimed.
* The palette became assertable. `tests/test_ui.py` now fails if any template
  or `app.js` hardcodes a colour, and checks that every risk state has a token.
* Red is now reserved: `RISK["suspicious"]` is the critical token, and the
  primary and secondary accents are asserted not to equal it.

Disadvantages:

* The page will not render its palette if opened as a bare file rather than
  served, because the injection happens server-side. The UI is only ever served.

Consequences:

`theme.py` carries presentation only — it names the colour a risk state is
drawn in, and never decides which state applies. The confidence ladder's
`UNVERIFIED` ceiling is asserted not to be drawn as success, so no colour can
imply the `HIGH` level that deliberately does not exist.

---

# Decision: Own Deterministic Derivation as a Module, and Report Readiness Separately from Status

Date: 2026-08-30

Status: Accepted

Context:

The external-dataset path was verified against two files whose schemas the
mapper had already been tuned for. Running it against genuinely unseen dialects
— and against the *published* form of datasets the repository had previously
tested in an enriched form — exposed four separate failures, all of the same
kind: the system had one place where a decision was *made* and a different place
where it was *reported*, and the two disagreed.

Problem:

1. **Derivation had no owner.** `payer_name_norm`, `source_file`, `id`,
   `amount_usd` and the single-domestic-country fill were each computed inline
   in `canonicalize_and_validate`, four as arms of an `elif` ladder interleaved
   with the reporting of field statuses and two as blocks after the loop. There
   was no list of what the system can derive, no seam at which a rule could be
   tested without building a workspace on disk, and adding a rule meant adding
   another arm to a 500-line function.

2. **A split instant could not be composed.** SAML-D as published ships `Date`
   and `Time` and no combined column. Both claimed `TIMESTAMP`, tied, and were
   discarded as a `CONFLICT`, so the dataset reached validation with no `date`
   at all. The generic multi-source fallback would have pasted them as text —
   `"2022-10-07 00:00:00 10:35:19"` — which parses as nothing.

3. **`READY` said two different things.** A dataset supplying no entity type and
   no resolvable geography satisfies the 167-feature contract exactly and is
   correctly `READY`, while the model sees an `entity_type_unknown` column and
   151 zeros. The report stated the first prominently and the second only as one
   line among the quality warnings.

4. **An ambiguous currency column was read as an absent one.** The `NATIVE_USD`
   contradiction check consults the mapped `ORIGINAL_CURRENCY` concept. When two
   columns tie for it and both drop to `CONFLICT`, the concept is unmapped, the
   check finds nothing to look at and passes, and the reason line asserts *"The
   dataset states no currency of its own"* about a file that states two. A
   fifteen-currency book reached `READY` with euros and yen written into
   `amount_usd`.

Options considered:

1. Add a `DATE_PART` / `TIME_PART` pair to `CanonicalConcept` for (2). Rejected:
   it grows the public concept enum for one dataset shape, and a lone `date`
   column would then have to choose between two concepts that mean the same
   thing.
2. Special-case the pair inside the canonicalizer's projection loop. Rejected:
   that is the ladder the module already suffers from, and it would put a
   date/time reading in a second place, able to drift from the mapper's.
3. Replace `status` with a three-valued readiness for (3). Rejected: `status`
   is a public interface consumed by the inference gate, the routes and the hub,
   and pinned by tests. The two questions are genuinely different and both are
   worth answering.
4. Have the mapper resolve the currency tie for (4). Rejected: it manufactures a
   winner from indistinguishable evidence, which is the thing conflict
   arbitration exists to refuse.

Decision:

* **`src/datasets/derivation.py` owns which canonical columns can be computed
  for a given dataset and under what rule.** Every rule has one shape — it
  returns the derivations it supports, possibly none — and each records the
  source columns it read. Planning is pure; the canonicalizer applies the plan
  and reports it, so the compatibility report lists exactly the rules that ran.

* **A calendar column and a clock column are complementary, not rival.** When
  exactly one `TIMESTAMP` claimant names a date part and exactly one names a
  time part, arbitration keeps both and `compose_instant` builds the instant
  properly. The date/time reading is `mapper.timestamp_part`, the same token
  vocabulary that proposed the mapping, so there is one definition of it.

* **The report carries `readiness` ∈ {`FULL`, `DEGRADED`, `BLOCKED`} beside
  `status`,** with one stated reason per line. `status` answers *may this run*;
  readiness answers *what will the model actually see if it does*. Readiness is
  classified from the finished report, so it cannot state a fact the report does
  not.

* **An unresolved concept claim is not an absent one.**
  `policy.unresolved_concept_claims` reports concepts a column claimed without
  being selected for, and a `NATIVE_USD` declaration over an unresolved currency
  claim is `INCOMPATIBLE` until the operator resolves the mapping. Separately,
  `_derive_amount_usd` refuses to fire when the currency assessment blocks, so a
  fabricated USD column is not written even to the diagnostic artifact of a
  workspace that will never reach inference.

Reasoning:

Every one of the four is the same defect: a fact was established in one place
and reported from another. Giving derivation an owner, deriving the date/time
reading from the mapper's own vocabulary, classifying readiness from the
finished report, and making an unresolved claim visible to the assessment all
close the same gap in different places.

The frozen contract is untouched by all of it. Derivation runs strictly upstream
of `preprocess`; readiness is a label on a report; the currency change only ever
*removes* a column that should not have existed.

Advantages:

* SAML-D as published, which previously had no `date`, now maps completely with
  no operator help — verified in `tests/test_real_datasets.py`.
* The Reference Dataset's own raw workbook, which previously lost its
  counterparty identity to a `payer_name` / `payer_name_norm` tie, now reaches
  the full 167-feature contract through the external path.
* A multi-currency book can no longer pass a `NATIVE_USD` declaration by having
  an ambiguous currency column, and cannot leave euros on disk labelled as
  dollars.
* Derivation rules are testable directly: `compose_instant` and `instant_pair`
  have unit tests that build no workspace.

Disadvantages:

* `canonicalize_and_validate` is shorter but still long, and still owns
  projection, field statuses, dtype coercion, pipeline execution, feature
  validation and persistence. Only the derivation concern was extracted; the
  rest is left for an explicit decision.
* `readiness` is a fifth thing the compatibility report says about a dataset,
  alongside `status` and three per-dimension statuses. It earns its place
  because it is the one an analyst acts on, but the report is not getting
  smaller.

Consequences:

Adding a derivation rule now means adding one function to `_RULES` in
`derivation.py`, and it appears in the compatibility report automatically. A
rule that cannot state its inputs, or that would fire on ambiguous evidence, has
nowhere to live — which is the intended constraint.

Alternatives rejected:

Auto-declaring `NATIVE_USD` when a dataset's currency column reads USD
throughout. The evidence is now *reported* in the assessment's reason and
evidence (`supports_native_usd`), but the declaration remains the operator's:
The declaration rule holds that no evidence ever makes a declaration on the
operator's behalf, and a currency column is a claim about `amount_orig`, not a
guarantee about every row of a file only sampled in part.

---

# Decision: Evaluate Models Through a Model-Agnostic Experiment Layer, and Demote Experiment 3 to a Benchmark

Date: 2026-09-06

Status: Accepted

Context:

Experiments 1-5 (`docs/architecture.md` §28.3) were run under one hard-coded
protocol living in `src/models/dataset.py` and `src/models/trainer.py`:

* `build_cv_folds` asserts `len(pos_indices) == 4`. The fold builder cannot run
  on any dataset that does not have exactly four positives.
* `run_cross_validation` asserts exactly four folds.
* The split is transductive by construction — one graph, one whole-population
  feature matrix, per-fold boolean masks — and nothing records that fact.
* `evaluate` returns `0.0` for PR-AUC and ROC-AUC when a partition holds a
  single class. Averaged over folds this is indistinguishable from a model that
  ranked every positive last.
* The threshold is always the F1-maximising one. F1 weights a false positive and
  a false negative equally, which no AML operation does.
* No calibration stage exists, so a raw sigmoid output and a probability are the
  same object.

The consequence is that the ablation could not answer the question it was built
to answer. With one positive per test fold, average precision can only take the
value `1/rank` of that positive, so every configuration returned either
`0.7534` or `1.0000`. The project audit concluded that the frozen GraphSAGE is
not demonstrably the best detector, and this protocol is the reason the question
stayed open.

Problem:

Benchmarking a replacement — gradient boosting, an edge-aware GNN, a temporal
GNN, a hybrid — required rewriting the evaluation code each time, against a
different informal standard, with results that could not be placed in one table.
Meanwhile the frozen 167-feature checkpoint is a hard compatibility obligation
for the UI, the workspace prediction route and the regression test, so it cannot
simply be removed.

Options considered:

1. **Generalise `src/models/` in place.** Rejected. `build_pyg_data`,
   `build_cv_folds`, `train_fold` and `run_cross_validation` are consumed by
   `src/detection/inference.py`, which enforces the frozen contract, and by five
   `run_exp*.py` scripts and four test modules. Making them model-agnostic means
   changing the signatures the frozen inference path depends on, to serve models
   that do not exist yet.

2. **Adopt an MLOps framework (MLflow, W&B, Hydra).** Rejected for now. The
   defects are scientific — a broken split, a dishonest metric, a threshold
   chosen by the wrong objective — and none of them is fixed by an experiment
   tracker. Adding a service dependency would raise the appearance of rigour
   without supplying any. The artifact format is plain JSON; adopting a tracker
   later is reading those files.

3. **A new layer above the existing one, with adapters.** Chosen.

4. **Delete GraphSAGE and start over.** Rejected. It is the shipping inference
   model and the only regression anchor the project has.

Decision:

* **`src/experiments/` is the layer in which models are compared.** It depends
  on Layers 1-5 through adapters and is not depended upon by them. Layers 1-5,
  `src/detection/inference.py`, the frozen checkpoint and the frozen
  vocabularies are unchanged by this decision.

* **A candidate implements four things**: `fit`, `predict_scores`, `config` and
  `emits_probabilities`. Adding a model family touches one file and no part of
  the runner, the metrics, the splits or the artifact format.

* **A split strategy declares its evaluation regime** — entity-disjoint,
  time-aware, graph-leakage-controlled — and the regime is written into the
  artifact. Five are registered, including `legacy_exp3_transductive`, which
  reproduces the historical protocol's *structure* for any positive count and is
  explicitly labelled transductive.

* **Feature computation has a declared scope.** `"global"` sees the whole
  population, which is what Experiments 1-5 did and is legitimate research
  framing once stated. `"train_window"` restricts feature computation to the
  fold's training window.

* **An undefined metric is `null` with a reason.** Aggregates report `n_defined`
  beside every mean. No metric is manufactured.

* **Score, calibrated probability and decision are three separate objects.**
  Calibrators and thresholds are fitted on validation; `select_threshold` has no
  parameter through which test data could reach it. There is no default
  threshold of `0.5` anywhere.

* **Experiment 3 GraphSAGE is a legacy benchmark and a regression reference, not
  an architectural invariant.** It is exposed as the `legacy_graphsage`
  candidate, which trains a new model from the same architecture and accepts any
  feature representation. The frozen checkpoint is never written to and no §28.3
  number is restated.

* **The 167-column vector is the `legacy_exp3_v1` comparison feature set**, one
  registered family among several, verified against `build_pyg_data` by test so
  the adapter cannot drift. It is not required of new candidates.

Consequences:

* A future contributor can implement one model family and get a comparable artifact
  without touching evaluation code. That is the criterion this decision was
  designed against.

* Running the framework over the Reference Dataset established a limit that
  changes how every future result must be read:

  * The four labelled entities are ranks 1-4 of 3,440 on raw transaction count,
    fan-in, fan-out, incoming volume, counterparty count and degree centrality.
    Largest negative: 554 transactions. Smallest positive: 1,777.
  * Ranking on that one column yields average precision `1.0000`.
  * The unsupervised heuristic, which reads no labels, matches logistic
    regression and gradient boosting on test average precision.
  * All four positives fall in the earliest half of first-activity time, so
    every forward-looking test partition holds zero positives.

  The Reference Dataset is therefore a regression signal and not a
  model-selection benchmark. This extends the `reciprocal_out_count` separation
  already recorded in §28.3: the separation is not specific to the multigraph
  features, it is present in plain degree. It is a property of the data, not a
  criticism of any model, and it does not change any historical result.

* Selecting a champion requires a dataset with enough positives to separate
  candidates. Acquiring or constructing one is the blocking dependency for model
  selection, and it is not addressed by this decision.

* `xgboost` and `lightgbm` remain optional. `gradient_boosting` resolves its
  backend at fit time and records which one ran, so a comparison across backends
  is visible as such rather than silent.

---

# Decision: Classify Every Dataset Before It Is Used for Model Comparison

Date: 2026-09-06

Status: Accepted

Context:

`src/experiments/` made model comparison possible. It did not make it
meaningful. The first question a comparison needs answered is whether the data
underneath it can distinguish one model from another, and until now the project
had answered that only for the Reference Dataset, and only as a side effect of
running experiments on it.

Four labelled transaction datasets are reachable from this tree: the Reference
Dataset (`dkdata.json`), SAML-D, an IBM AML extract, and a Kaggle financial
fraud file. Nothing in the repository read any of the last three, and nothing
recorded what any of them could support.

The specific hazard is not "a dataset is small". It is that a dataset can look
ideal - nine and a half million transactions, eight thousand positives, eleven
months of even temporal coverage, no missing values - and still be unable to
rank models, because its labels were written by the same process that wrote its
structure. That failure is invisible in a per-model score. Every model scores
well, and the ranking between them is noise.

Problem:

Without a classification step, the project would have produced a SAML-D model
ranking, seen average precision above 0.99, and concluded that its feature work
and its model choice were validated. Both conclusions would have been false, and
the artifacts would have looked exactly like real evidence.

Options considered:

1. **Judge suitability informally, per dataset, when the question comes up.**
   Rejected. It is the status quo, it is not reproducible, and it puts the
   judgement in a conversation rather than in an artifact.

2. **Use positive count as the only criterion.** Rejected. SAML-D has 7,902
   positive entities and fails anyway. A count threshold would have passed it.

3. **Test only single-column separability.** Rejected. On SAML-D the best single
   simple quantity reaches average precision 0.1831, which looks entirely
   healthy. The label is trivial as a *combination*, not as a column.

4. **A classification stage with a combined structural probe.** Chosen.

Decision:

* **`src/experiments/audit.py` classifies a dataset before it is used.** Three
  measured blocks - `data`, `labels`, `structural_triviality` - plus a split
  feasibility block and a verdict with a stated reason.

* **The decisive test is a combined probe.** An untuned
  `HistGradientBoostingClassifier` is fitted on a stratified half of the
  population using only fifteen simple quantities an analyst could compute
  without a model: transaction count, in- and out-degree, fan-in, fan-out,
  counterparty count, incoming, outgoing and total volume, degree centrality,
  component size, first and last activity, active span and velocity. It is
  scored on the other half. It is a probe, not a candidate; its result is
  reported as a property of the dataset.

* **Four verdicts, with a hard threshold.** Probe average precision >= 0.99 is
  `trivially_recoverable` and the dataset becomes `STRUCTURAL_VALIDATION_ONLY`.
  Fewer than 40 positives is `REGRESSION_ONLY`. A single-class population is
  `BLOCKED`. Otherwise `MODEL_SELECTION_READY`.

* **Label domain is declared by the loader, not measured.** Whether a
  publisher's positive class means "money laundering" cannot be recovered from
  the file. `audit_datasets.py`'s loader table states it; a dataset whose domain
  is not money laundering can never be `MODEL_SELECTION_READY` for this project
  however well it scores.

* **A quantity that a dataset cannot support is `None` with a reason.** The
  probe drops all-NaN and constant columns and records which, rather than
  imputing an absent quantity into existence.

Consequences:

* **No dataset was `MODEL_SELECTION_READY` at the time of this record**
  (superseded later the same day by the acquisition of `ibm_aml_hi_small`, which
  is). Measured and recorded in `outputs/audits/`:

  | dataset | entities | positives | probe AP | verdict |
  | --- | ---: | ---: | ---: | --- |
  | `reference_dkdata` | 3,440 | 4 | undefined (too few) | `REGRESSION_ONLY` |
  | `samld` | 855,460 | 7,902 | **0.9971** | `STRUCTURAL_VALIDATION_ONLY` |
  | `ibm_aml_sample` | 3,183 | 3,183 | undefined (single class) | `BLOCKED` |
  | `financial_fraud_kaggle` | 899,988 | 295,613 | 0.4226 | `STRUCTURAL_VALIDATION_ONLY` (domain) |

* **SAML-D's defect is specific, and the audit reports it as data.** The
  `class_free_bands` block lists discrete values that *no negative entity
  takes*: `counterparty_count` in {5, 6, 8} holds 619 positives and zero
  negatives, `fan_out` in {5, 6, 7, 8} holds 610. The generator builds normal
  traffic as degree-1 stars - median counterparty count 1, 90th percentile 1 -
  and laundering typologies as multi-hop meshes. Mean neighbour counterparty
  count is 5.5 for positives and 44.5 for negatives: inverted, and barely
  overlapping. Neither the Reference Dataset nor the Kaggle fraud file has a
  single class-free band, so the diagnostic is discriminating rather than
  universally alarmed. This is a property of the simulator, not of money
  laundering.

* **The blocking dependency for model selection is data, and it is now
  documented rather than suspected.** `docs/datasets.md` section 7 lists each gap
  with a concrete remedy. Building a stronger model does not address any of them.

* **The Reference Dataset's role is unchanged**, and this audit independently
  reproduces the finding that put it there.

* Adding a dataset is two edits: a `SourceSpec` and `DatasetSpec` in
  `src/experiments/sources.py`, and a row in `audit_datasets.py`'s loader table.

---

# Decision: Represent AML Evidence as Six Separable, Graph-Object-Free Feature Families

Date: 2026-09-06

Status: Accepted

Context:

The feature registry held the legacy 167-column representation, its three
component blocks, and `aml_core_v1` - one 29-column composite covering
transaction, flow and network evidence together. Two problems followed.

First, `aml_core_v1` cannot answer an ablation question. Asking which
information source contributes requires switching sources off independently,
and a composite has one switch.

Second, it does not scale. It builds a per-entity cross product of inflows and
outflows to find the shortest turnaround, and it needs a `networkx` graph for
clustering and centrality. On the Reference Dataset's 16,940 transactions that
is invisible. On SAML-D's 9,504,852 it is not: a `MultiDiGraph` over that edge
set costs tens of gigabytes, and the cross product is quadratic in an entity's
transaction count.

Problem:

The project needed AML features that (a) can be ablated by information source,
(b) run on a nine-million-row edge table, and (c) do not silently change the
meaning of the columns existing artifacts already recorded.

Decision:

* **Six new families in `src/experiments/aml_features.py`**, registered
  separately: `aml_txn_v1` (transaction), `aml_temporal_v1` (temporal),
  `aml_flow_v1` (flow), `aml_network_v1` (network), `aml_cycle_v1` (structure),
  `aml_context_v1` (context). 68 columns in total, and no short column name
  appears in two families: a column present twice inflates its own importance
  in every model that reads the matrix.

* **`aml_core_v1` is not modified.** A family's name pins its semantics;
  changing what it computes would invalidate every artifact that recorded it.
  The new families sit beside it.

* **No family requires a `networkx` object.** Structural columns are computed
  from a deduplicated SciPy CSR adjacency. On SAML-D, 9,504,852 transactions
  deduplicate to 887,497 ordered pairs, which makes the sparse square an
  8.0-million non-zero product and puts directed 3- and 4-cycle counts within a
  few seconds. The runner builds a `networkx` graph only for families that
  declare `requires_graph`, which now means only the legacy ones.

* **`FeatureRequest` carries a `cache`.** The flattened event view and the sparse
  adjacency are built once and shared by every family computing against that
  request. The cache is keyed by the request, so a fold with a different time
  window gets a fresh one and cannot inherit a summary of transactions it is not
  allowed to see.

* **Temporal joins are vectorised, not looped.** "The most recent inflow at this
  entity at or before this outflow" is answered for the whole population with
  one `searchsorted` over a globally monotone entity-by-time key. A per-entity
  Python loop over 855,460 entities would have dominated the computation.

* **Temporal semantics are documented per column.** Only `recency_days` and
  `tenure_days` depend on a reference "now" (`FeatureRequest.as_of`, falling
  back to the latest observed timestamp). Every other column is a summary of the
  transactions present in the request, so a training-window restriction is
  honoured by construction. `tests/test_aml_features.py` pins that only those two
  columns move when `as_of` moves.

* **Undefined stays undefined.** A statistic with no meaning for an entity is
  `NaN`, never 0 and never a sentinel. Imputation is the candidate's
  train-fitted business.

* **Numerically stable dispersion.** Standard deviations are computed in two
  passes from an already-known mean. The one-pass `E[X^2] - E[X]^2` identity is
  the obvious single-`bincount` version and is wrong at this scale: SAML-D
  amounts reach 1.26e7, a busy entity's sum of squares reaches ~1e17, and the
  subtraction destroys most of the significant digits exactly for entities whose
  amounts barely vary - the structuring signature the feature exists to catch.

Consequences:

* The full 68-column matrix over 855,460 entities and 9,504,852 transactions
  computes in about 110 seconds on one CPU. Under the global feature regime the
  runner now computes it once and reuses it for every fold, because it is
  identical across folds by definition; under `train_window` it does not, because
  reusing it would import transactions from beyond another fold's boundary.

* Ablation cells A-E are configuration, not code.

* `aml_cycle_v1` is the one family whose cost is governed by graph density
  rather than row count: the sparse square is the term that grows. On a denser
  benchmark it, and not the others, would need a budget. This is stated in the
  module rather than discovered later.

* The heuristic floor resolves its signals by short column name, so a spec that
  uses the new families declares its signal list explicitly. The default list,
  which matches `aml_core_v1`, is unchanged.

---

# Decision: Do Not Build an Advanced Graph Model in This Phase

Date: 2026-09-06

Status: **Superseded** by "Defer Advanced Graph Learning Pending a 2-Hop Test"
(same date). Its NO-GO rested on there being no dataset able to rank models;
`ibm_aml_hi_small` was acquired later the same day and can. The record is kept
because its reasoning was correct given what was then available.

Context:

The project owner authorised replacing GraphSAGE, the 167-feature
representation, the graph construction, the temporal representation, the
heterogeneous representation, the training objective and the inference
architecture, on controlled evidence. The natural expectation was that this
phase would end by choosing an edge-aware or temporal graph architecture.

It ends by not choosing one, and the reason is data.

Evidence:

All results below are in `outputs/experiments/`, reproducible with
`python run_experiment.py configs/experiments/samld_*.json` and rendered by
`compare_experiments.py`. SAML-D is synthetic and classified
`STRUCTURAL_VALIDATION_ONLY`; these numbers describe a simulator.

* Central comparison, `stratified_kfold` 4 folds, 1,976 test positives per fold:
  label-free structural floor `0.1768 +/- 0.0013`; gradient boosting without
  graph features `0.9157 +/- 0.0050`; L2 logistic regression on all six families
  `0.9987 +/- 0.0008`; gradient boosting on all six `0.9997 +/- 0.0002`.
  Repeated at seed 1337: floor `0.1769`, boosting `0.9999`.
* Ablation, same model and split, families varied: transaction only `0.8454`;
  plus temporal `0.8603`; plus flow `0.8863`; plus network and cycle `0.9997`;
  all six `0.9997`. **D equals E to four decimal places.**
* Time-forward, `train_window` feature scope: floor `0.1504`, boosting
  `0.8152`. The regime moves the number far more than the model does.
* Control on `financial_fraud_kaggle`, identical code path on a non-trivial
  label: floor `0.3871`, boosting `0.4225`, against prevalence `0.3285`. The
  stack does not manufacture high scores.

Decision:

**No edge-aware, temporal or heterogeneous GNN is built in the next phase.**
Three reasons, in order of force:

1. **Nothing can measure it.** A new architecture would be compared against
   gradient boosting at `0.9997` and logistic regression at `0.9987`. No
   architectural difference survives that ceiling.

2. **SAML-D's graph would not reward message passing even if its labels were
   sound.** 90 % of accounts have exactly one counterparty; the graph has zero
   directed 3-cycles; 15,592 weakly connected components with a median size of
   51. There is almost no neighbourhood to aggregate, and the one hop that
   exists is already a single tabular column (`neighbour_counterparty_mean`,
   which alone separates the classes better than any degree measure).

3. **The blocker is data, and a model does not move it.** The remedies are in
   `docs/datasets.md` section 7.

Persistent homology is deferred by the same argument and recorded in
`docs/architecture.md` section 45.10: topology has to be shown to add
information beyond graph statistics, cycles, communities, temporal patterns and
supervised learning, and no available dataset leaves headroom in which to show
it. When it is justified, it should be applied to suspicious subgraphs rather
than the whole transaction universe.

Consequences:

* **A NO-GO is a result.** It is recorded here so future work does not
  re-derive it, and so nobody reads "GraphSAGE is weak" as licence to reach for
  a different architecture instead of a better dataset.

* **Revisit on evidence, not on schedule.** The condition is a dataset whose
  labels are not a function of its structure and whose graph has real multi-hop
  structure. The framework is ready: adding a model family is one file behind
  the `Candidate` contract, and the metrics, splits, thresholds, calibration and
  leakage audits do not change.

* **One result must not be misread.** Supervised models beat the label-free
  floor by a very wide margin on SAML-D. That is not evidence that supervision
  works; it is evidence that a signed rank sum cannot express a *band*.
  SAML-D's positives sit in a band of counterparty counts with ordinary
  high-degree hubs ranked above them, so the floor puts benign hubs first -
  precision@100 `0.2975`, precision@1000 `0.0582`, then precision@5000
  recovering to `0.2645`. A second, non-monotone label-free baseline would make
  future comparisons sharper and has not been built.

* **The frozen Experiment 3 contract is untouched by this decision.** It remains
  byte-stable for the UI, the workspace prediction route and
  `tests/test_reference_regression.py`, and it remains a legacy benchmark rather
  than an architectural invariant.

---

# Decision: Acquire the IBM AMLworld Ledger and Adopt It as the Model-Selection Benchmark

Date: 2026-09-06

Status: Accepted

Supersedes: "Classify Every Dataset Before It Is Used for Model Comparison"
retains its method and its verdicts; only its *conclusion that no dataset was
usable* is superseded.

Context:

The previous phase established that none of four reachable datasets could rank
models, and named acquiring a full IBM AML transaction ledger as the single
highest-value next action. The file present in this tree was `IBM data.txt` -
16,467 laundering attempts over 187,337 lines, every row flagged. That is IBM's
`HI-Small_Patterns.txt`, the typology listing, not the ledger. It has no
negative class, so it was `BLOCKED`.

Problem:

Without a ledger the project had two options, both bad: rank models on a dataset
whose labels a probe recovers at average precision 0.9971 (SAML-D), or build a
synthetic generator and risk measuring its own assumptions.

Options considered:

1. **Build a controlled synthetic benchmark.** Deferred, not rejected. It was
   justified while nothing could rank models. It stopped being the next step the
   moment a published benchmark could.
2. **Use SAML-D anyway, with caveats.** Rejected. Caveats do not repair a
   ceiling: three of four candidates land within 0.09 of each other at the top.
3. **Reconstruct a ledger from the patterns file.** Rejected outright. The
   patterns file is the positive class; fabricating a negative class around it
   would be inventing the data the experiment is supposed to measure.
4. **Acquire the published ledger.** Chosen.

Decision:

* **`HI-Small_Trans.csv` is acquired, not vendored.** 475,664,283 bytes, SHA-256
  `b19d39f5...`, Community Data License Agreement - Sharing 1.0, published by
  Erik Altman (IBM) on Kaggle, from Altman et al., *Realistic Synthetic
  Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023
  Datasets and Benchmarks. It downloaded without credentials in this
  environment.

* **`acquire_dataset.py` is the reproducible mechanism**, and
  `data/manifests/*.json` the committed provenance record: source URL,
  acquisition timestamp, byte count, SHA-256, licence, citation and header line.
  `--verify-only` hashes a manually placed file and writes the same manifest, so
  a credential-less or offline environment reaches the same state by hand. A
  manifest whose digest disagrees with the file on disk is an **error**, never a
  silent overwrite: two different files must not share one provenance record.

* **`.gitignore` keeps raw benchmarks out and manifests in.** Nothing changed
  about repository policy; it is now written down.

* **`ibm_aml_hi_small` is the model-selection benchmark.** Verdict
  `MODEL_SELECTION_READY` on measured evidence: 515,088 entities, 5,078,345
  transactions, 6,357 positive entities (1.234 %), structural probe average
  precision **0.2098** against a 0.0123 prevalence floor, and **no class-free
  bands**. Compare SAML-D at 0.9971 with three class-free bands.

* **Two schema traps are handled in the loader and pinned by tests.** Both would
  silently corrupt the graph rather than fail:
  - The payer and payee account columns are **both named `Account`**; pandas
    renames the second `Account.1`. Reading "the Account column" yields the
    payer twice and produces a graph of self-loops.
  - Bank codes are **zero-padded text**. Under inferred dtypes the file's 30,528
    distinct bank codes collapse to 30,470 - 58 collisions, `012` merged with
    `0012`. Identifier columns are pinned to `str`. On this file the collisions
    happen not to produce colliding `(bank, account)` pairs, so the entity
    population is 515,088 either way; that is luck, not a reason to rely on
    inference.

* **The entity is the `(bank, account)` pair.** 515,080 distinct account strings
  but 515,088 distinct pairs: eight account strings appear at two banks. An
  account number is issued by a bank and is unique only within it.

* **The entity label is derived and declared.** IBM labels transactions (5,177
  of 5,078,345, 0.102 %). The rule - payer or payee of at least one flagged
  transaction - is written into `DatasetSpec.label_definition`, and it makes a
  launderer's counterparty positive.

* **The typology file is never a feature source.** Its group headers state the
  mechanism *and its parameter* (`FAN-IN: Max 13-degree Fan-In`,
  `CYCLE: Max 12 hops`), which makes it a description of the label-generating
  process. It is useful for auditing that process and for nothing else.

Consequences:

* **Model selection is unblocked.** Results: `docs/model_selection.md`. The
  candidates now separate: label-free floor `0.0390`, logistic regression
  `0.0900`, gradient boosting without graph features `0.2717`, with them
  `0.2797`. On SAML-D the same four were `0.1768 / 0.9987 / 0.9157 / 0.9997`.

* **Time-forward is possible but only on *last* activity.** Half of all entities
  first transact within 28 minutes of the file's start, so a split on *first*
  activity puts the 60th-percentile boundary at 03:55 on day 1 and leaves a
  four-hour training window. On last activity the same quantile gives a ~9-day
  window and 3,940 positives after the cut.

* **A controlled synthetic benchmark is no longer the next step.** Revisit it
  only if IBM's limitations - 17 days, derived entity label, a single simulator
  - bind on a specific question.

* **The data gap that remains is *real* data.** Every dataset that can rank
  models in this project is synthetic. `reference_dkdata` is still the only
  observed one and still has four positives.

* **The previous phase was not wrong.** Its conclusion was correct about the
  datasets then reachable. Acquiring a fifth changed the answer, which is what
  the audit layer exists to detect.

---

# Decision: Defer Advanced Graph Learning Pending a 2-Hop Test

Date: 2026-09-06

Status: **Superseded** by "Classify the Temporal Collapse as Mixed, and Correct
the Time-Forward Split Before Any Graph Work" (same date). Its numbers were
produced with `aml_context_v1` silently dead, and its "next experiment" is now
second in line behind a corrected split. Kept unedited as the record of what was
believed at the time.

Supersedes: "Do Not Build an Advanced Graph Model in This Phase" (same date). Its
NO-GO rested on there being no dataset able to rank models. That premise no
longer holds.

Context:

With `ibm_aml_hi_small` usable, the graph question could finally be asked
against evidence rather than against a ceiling.

Evidence:

All from `outputs/experiments/ibm_*.json`, `stratified_kfold` 4 folds, ~1,589
test positives per fold, prevalence 0.0123. IBM AMLworld is synthetic.

* Central comparison: label-free floor `0.0390 ± 0.0011`; logistic regression
  `0.0900 ± 0.0068`; gradient boosting without graph features
  `0.2717 ± 0.0091`; with them `0.2797 ± 0.0103`. Seed 1337 reproduces the floor
  at `0.0392` and the champion at `0.2804`.
* Ablation: transaction only `0.0589`; plus temporal **`0.2222`**; plus flow
  `0.1410`; plus network and cycle `0.1193`; all six `0.2797`.
* Time-forward, `train_window` scope: floor `0.0389`, champion `0.0620`
  (ROC `0.5985`).

Decision:

**DEFERRED.** Not GO, and not the NO-GO that SAML-D forced.

The reasoning is one row of the criteria table. Network and cycle features add
**+0.060** AP over transaction-only features, so relational information is
informative. But added *last*, on top of transaction, temporal, flow and context,
they are worth **+0.0080** AP - smaller than the fold dispersion of either run
(±0.0091 and ±0.0103). Almost everything the graph knows is already implied by
behaviour.

A GNN's specific claim is **multi-hop** structure. The tabular network family
reaches exactly one hop. Whether a second hop adds anything is untested and is
cheap to test.

**The next experiment is a 2-hop feature family, not a model.** Add
`aml_network_v2` carrying neighbour-of-neighbour aggregates using the sparse
adjacency already present - one more `A @ A` product, no new dependency - and run
one ablation cell against `ibm_boosting_graph`.

* Gain clearly above ±0.01 -> **GO**, candidate an **edge-aware GNN
  (GINE-style)**, because this dataset's edges carry amount, timestamp, currency
  pair and payment format, and payment format alone separates classes (ACH
  0.746 % positive rate against credit card 0.016 %). A node-only formulation
  would discard the informative part.
* No gain -> **NO-GO** for message passing here, established for one matrix
  product instead of a model build.

Rejected candidates, with reasons: a **temporal graph model** (17-day window, and
temporal *features* are already the largest single contributor at +0.163); a
**heterogeneous GNN** (one node type - banks are an attribute of the entity key,
not a typed node with behaviour); **re-running the legacy GraphSAGE** (node-only,
discards edge attributes).

Consequences:

* **No GNN is built in this phase**, and the reason is measured rather than
  assumed.

* **The most valuable open problem is not architecture.** Transductive `0.2797`
  falls to `0.0620` under the operational regime - a 78 % drop, ROC `0.9125` to
  `0.5985`. The strongest structural quantities are `last_activity` (AP 0.1404)
  and `active_span_days` (0.1321): temporal *position* inside a 17-day window,
  which truncation removes. Explaining that gap is worth more than any
  architecture change, and it is a warning about every transductive number this
  project reports.

* **The frozen Experiment 3 contract is untouched.** It remains byte-stable for
  the UI, the workspace prediction route and `tests/test_reference_regression.py`,
  and it remains a legacy benchmark.

---

# Decision: Classify the Temporal Collapse as Mixed, and Correct the Time-Forward Split Before Any Graph Work

Date: 2026-09-06

Status: Accepted. Its prescribed next experiment was run on 2026-09-11, and its
graph deferral is superseded by "Establish the Calendar-Forward Benchmark and
Record NO-GO for Advanced Graph Learning on IBM HI-Small". The classification
of the collapse stands.

Context:

The IBM HI-Small benchmark produced average precision `0.3827` under
`stratified_kfold` with `global` features and `0.1110` under
`time_forward_holdout` with `train_window` features - a 71 % drop, with ROC
falling from `0.9215` to `0.6190`. (The figures that prompted the investigation
were `0.2797` and `0.0620`; they were computed with six feature columns silently
dead - see the defect below.) That gap was larger than any effect the model
comparison had found, and no advanced-model decision could be defended until it
was explained.

Problem:

The two runs differed in **two** things, not one: the split *and* the feature
scope. The published comparison could not attribute the drop to either. A
project that reported `0.2797` as its benchmark and `0.0620` as "the operational
number" without understanding the difference would be reporting two numbers it
could not interpret.

Investigation:

Recorded in full in `docs/temporal_collapse.md`. Every claim below is a measured
result in `outputs/experiments/ibm_*.json`.

* **Decomposition.** Adding the missing diagnostic cell (time-forward split at
  `global` scope, `ibm_tf_global_boosting_graph`) splits the drop:
  `0.3827 -> 0.2453` from the split (50.6 %) and `0.2453 -> 0.1110` from the
  feature scope (49.4 %) - **almost exactly equal**. The fourth cell of the 2x2 is structurally impossible: a
  non-temporal split defines no training-window boundary.

* **Mechanism.** The collapse is concentrated in `aml_temporal_v1`. Measured on
  the same test entities, `recency_days` scores AP `0.3033` at `global` scope
  and `0.0351` at `train_window`; `span_days` `0.2835` against `0.0370`. Best
  single feature per family collapses by `0.2500` (temporal), `0.0231` (flow),
  `0.0168` (cycle), `0.0040` (network) and `-0.0039` (transaction).

* **Why the middle cell is void.** `recency_days` at `global` scope is measured
  from the end of the file, and the split assigns entities by last activity, so
  the feature is a monotone transform of the assignment variable. Its ranges are
  disjoint: train `[8.900, 17.679]`, test `[0.000, 8.410]`. The model
  extrapolates outside its training support, which is why a feature worth AP
  `0.3033` inside the test partition yields a model scoring `0.2453`.

* **The split manufactures its own shift.** Placing entities by *last* activity
  selects test entities that were still transacting at the end - the long-lived,
  high-volume ones. Train median transaction count **3**, test median **37**.
  **48 of 54** features exceed PSI `0.25`. Prevalence: train `0.78 %`, test
  `2.64 %`. This is a survivorship split, not a forecasting split.

* **The generator drifts.** Days 1-10 carry 99.98 % of transactions at a
  `0.03-0.21 %` positive rate. Days 11-18 hold 1,108 transactions at a
  **57-73 %** positive rate and are **100 % ACH**, where the body has five
  payment formats among positives. **26.3 %** of test positives draw their label
  from that tail.

* **Not a coverage problem.** 99.6 % of test entities and 99.3 % of test
  positives have pre-boundary history; only 0.3 % of post-boundary entities are
  new; 93.1 % of transactions are visible before the boundary. The benchmark is
  99.7 % transductive in entity terms.

* **The ablation ordering inverts.** Under the standard split: A `0.0589`,
  B(+temporal) `0.2222`, C(+flow) `0.1410`, D(+network) `0.1193`, E `0.2797`.
  Under time-forward: A `0.0363`, B **`0.0295` (ROC `0.4783`, worse than
  chance)**, C `0.0553`, D `0.0618`, E `0.1110`. Temporal is the strongest
  family under random folds and actively harmful under shift. **But the graph
  families are not the survivors**: comparing the two full models,
  `boosting_behaviour` (no graph) reaches `0.3734` / `0.1120` against
  `boosting_graph` `0.3827` / `0.1110`, so network + cycle are worth `+0.0093`
  under random folds and `-0.0010` under shift. Cells A-D carry no context, which
  is why network looked like the survivor there.

* **Supervision survives, but narrowly.** Under time-forward: floor `0.0389`,
  logistic regression `0.0525`, boosting without graph `0.1120`, boosting with
  graph `0.1110` - 2.9x the floor, against 9.8x under random folds.

Decision:

* **The collapse is classified MIXED**, with contributions in this order:
  split methodology (50.6 % of the drop, and the origin of the distribution
  shift), feature scope (49.4 %, legitimate and correctly costly), distribution
  shift (largely induced by the split), label/generator drift (the days 11-18
  tail), and a genuine but smaller temporal generalization failure.
  **Entity-coverage is ruled out on evidence.**

* **Both numbers are kept and reported with their regimes.** `0.3827` is a valid
  transductive result. `0.1110` is a valid result for a survivorship split over
  a drifting label process. Neither is "the" performance of the system, and the
  71 % drop overstates pure temporal failure because half of it is the split.

* **`ibm_tf_global_boosting_graph` (`0.2453`) is a diagnostic, never a result.** It is
  retained so the decomposition is reproducible and is documented as void for
  quotation.

* **The next experiment is a corrected time-forward split**, not a model: a
  calendar cutoff applied to every entity alike, with the label defined from
  post-boundary flagged transactions, so train and test differ in *time* rather
  than in *who is in them*. It must also decide how to treat the days 11-18
  tail, which at 57-73 % positive would otherwise dominate any post-boundary
  label.

* **The 2-hop experiment is deferred behind that, and this investigation
  weakened its motivation.** Added last, network + cycle are worth `+0.0093`
  under random folds and `-0.0010` under shift. The earlier reading that network
  was the surviving family was an artifact of ablation cells A-D carrying no
  context. Measuring 2-hop value on the current split would also inherit the
  survivorship defect.

* **No GNN, no TDA and no 2-hop features were added.**

Consequences:

* **A defect was found and fixed during the audit.** `aml_context_v1` matched
  source columns by exact name and knew none of IBM's spellings
  (`Payment Format`, `Payment Currency`, `Receiving Currency`). It produced six
  all-NaN columns **with no error**, so every IBM run before this date used 62
  effective features rather than 68 and discarded `Payment Format` - a strong
  label correlate (ACH `0.746 %` positive rate against Credit Card `0.016 %`).
  The spellings are registered, the family now records what it resolved and
  warns when it resolves nothing, and every affected cell was re-run. Pre-fix
  artifacts are kept under `outputs/experiments/_prefix_context/`.

* **A general lesson about the feature registry.** A family that matches source
  columns by name fails silently when it matches nothing. That failure mode is
  now instrumented for `aml_context_v1`; any future family that resolves columns
  by name should do the same.

* **A general lesson about split design.** A split that assigns entities by a
  behavioural quantity will differ between partitions in every feature
  correlated with that quantity. When the same quantity is also computable as a
  feature, the configuration is void. `docs/architecture.md` §45.4 now says so.

* **The benchmark remains `MODEL_SELECTION_READY`.** Labels are not structurally
  trivial (probe AP `0.2098`), there are 6,357 positives, and both regimes are
  declared. What changed is that its time-forward configuration is now known to
  be a survivorship split over a drifting label process and must be described
  that way.

---

# Decision: Establish the Calendar-Forward Benchmark and Record NO-GO for Advanced Graph Learning on IBM HI-Small

Date: 2026-09-11

Status: Accepted

Supersedes: the graph deferral in "Classify the Temporal Collapse as Mixed, and
Correct the Time-Forward Split Before Any Graph Work" (2026-09-06), whose
prescribed experiment this is.

Context:

The survivorship time-forward split placed each entity by its last activity and
so compared two populations rather than two periods (train median 3
transactions, test median 37; 48 of 54 features past PSI 0.25). The previous
decision required a calendar cutoff applied to every entity alike before any
2-hop or GNN work, and a decision about the days 11-18 tail of IBM HI-Small,
which is 57-73 % flagged and 100 % ACH because normal traffic has stopped.

Problem:

A calendar cutoff is not an entity partition. The same account must be scored
before and after the cutoff, with a label drawn from a time window, and the
framework held only whole-file entity labels, which include the future. Every
existing split strategy is validated as entity-disjoint.

Options considered:

1. One cutoff, random entity split, label to the end of the file.
2. One cutoff, random entity split, bounded label window.
3. Rolling origin with a fixed feature lookback (3 or 4 days).
4. Rolling origin with expanding history, three contiguous 2-day windows
   ending before the tail.
5. Registering the calendar split inside `splits.py` beside the entity
   partitions.

Decision:

* **Adopt option 4 as the calendar-forward regime.** Rows are (entity, origin)
  pairs. Origins 2022-09-05 (train), 2022-09-07 (validation), 2022-09-09
  (test); features from every transaction before the origin; label = payer or
  payee of at least one flagged transaction in `[origin, origin + 2 days)`;
  eligible = at least one transaction before the origin; `max_label_end`
  2022-09-11. Configs `configs/experiments/ibm_cal_*.json`.
* **Implement it as its own evaluation path**, `src/experiments/calendar.py`,
  routed by `runner.run_experiment`, with the feature scope `origin_window`.
  `ExperimentDataset.transaction_labels` carries the publisher's flags beside
  the edge table. Candidates opt in with `calendar_compatible`; the legacy
  GraphSAGE, which reads whole-file labels, is refused.
  `calendar_design_audit.py` makes the design evidence an artifact.
* **Advanced graph learning (edge-aware / GINE-style message passing) is NO-GO
  on IBM HI-Small.** The 2-hop family is not run in this phase and is off the
  critical path; it remains available as a cheap falsification test under the
  calendar regime with at least four model seeds.

Evidence (`docs/calendar_forward_benchmark.md`, `outputs/experiments/ibm_cal_*.json`,
`outputs/audits/calendar_design_ibm_aml_hi_small.json`; IBM AMLworld is
synthetic):

* Design, measured before any model ran and without labels: expanding history
  shifts 14 of 66 features past PSI 0.25 (median 0.010); a 4-day lookback 37
  (median 0.340) and a 3-day lookback 21 (0.149), each also losing 27-29 % of
  test positives. Prevalence 0.279 % / 0.286 % / 0.267 % across the roles.
  97.9-98.4 % of window positives have history.
* Test origin: 514,022 entities, 1,374 positive. Floor AP 0.0046 (1.7x, ROC
  0.652); logistic regression 0.0162 (6.1x, 0.642); boosting without graph
  0.0133 ± 0.0035 over four model seeds; with graph 0.0109 ± 0.0032. ROC
  0.69-0.70. Brier skill against the base rate +0.0013 to +0.0018.
* First involvement (82 % of positives): every candidate at 1.4-2.0x lift.
  A label-using persistence reference (already involved before the origin)
  reaches AP 0.0118.
* Graph increment, paired over four seeds: -0.0024 ± 0.0027 AP, negative in
  all four; ROC -0.0040 ± 0.0049. Against +0.0093 under the standard regime and
  -0.0010 under survivorship.

Reasoning:

Options 1 and 2 fit the model on labels from the very window being scored,
which do not exist at scoring time; option 1 also draws on the tail (at
2022-09-10 with a three-day horizon, 36 % of eligible positives are positive
only through it). Option 3 induces more shift than it removes, because the
day-1 burn-in falls inside the train lookback only. Option 5 would have meant
relaxing the entity-disjointness check that protects every other strategy.
Under option 4 the graph families are the one relational signal a GNN would
build on, and they contribute nothing in the only regime that asks a
forecasting question cleanly.

Advantages:

* Train and test differ in time, not in who is in them; the information
  boundary is audited on the rows actually used.
* The tail is excluded by a declared, checked parameter.
* The first-involvement subgroup is reported in every artifact.

Disadvantages:

* Ten days admit one rolling-origin design; robustness to the choice of
  origins is untested.
* The windows fall on different weekdays; the temporal family keeps a
  history-length shift (four days of history at the train origin, eight at
  the test origin).
* At 0.27 % prevalence average precision is set by a handful of top-ranked
  positives; one configuration spans 0.0084-0.0167 across seeds.
* Its target (2-day window involvement) differs from the standard and
  survivorship benchmarks' (whole-file involvement).

Consequences:

* **Three IBM regimes are reported, never merged**: standard (retrospective
  detection, features overlap the label period), survivorship time-forward
  (a population comparison, kept as a diagnostic), calendar-forward
  (prospective forecasting).
* A model comparison under calendar-forward needs at least four model seeds;
  single-seed AP differences of a few thousandths are noise.
* No GNN is built. The frozen Experiment 3 contract is untouched.
* **For the project owner:** the standard headline (0.3827) is a detection
  result over an observed period; forecasting beyond it reaches 1.4-2.0x lift
  for new involvement. Which of the two tasks the system is meant to solve is
  an owner decision, and the evidence for them diverges by more than an order
  of magnitude.

Alternatives rejected:

Options 1, 2, 3 and 5 above, for the reasons given. Building the 2-hop family
or a GINE model on the survivorship split, which would inherit its population
shift.

---

# Decision: Adopt an ML + Investigation-Intelligence Hybrid With a Model-Independent Risk Contract

Date: 2026-09-11

Status: Accepted (subject to project-owner review)

Context:

Three IBM HI-Small regimes are measured: retrospective detection within one
period (boosting AP 0.3827, 9.8x the label-free floor), survivorship
time-forward (a population comparison), and calendar-forward forecasting (AP
0.0133 ± 0.0035 at 0.267 % prevalence; 1.4-2.0x lift on first involvement;
Brier skill +0.001 to +0.002). Graph features are worth +0.0093 AP
retrospectively and −0.0024 ± 0.0027 when forecasting. The shipped inference
path is still the frozen GraphSAGE, whose only evaluation is a 4-positive
dataset, and its output frame - GraphSAGE's sigmoid plus eleven legacy columns
- is what the UI reads.

Problem:

The original objective names GNNs, TDA, cycles and rings; the evidence supports
a tabular ranker for retrospective detection and nothing that forecasts well.
Without a stated target, every downstream component is coupled to whichever
model happens to be shipped, and a retrospective score can be presented as a
forecast.

Options considered:

1. ML-first detection system.
2. Forecasting-first system.
3. ML + investigation-intelligence hybrid.
4. Investigation analytics without ML.

Decision:

* **Option 3.** ML ranks entities by suspicious behaviour inside a declared
  analysis window (retrospective detection) under an alert budget. Graph and
  temporal analytics - cycles, flows, counterparties, paths - are
  post-prediction investigation evidence and never change a score. Forecasting
  is a research track. No GNN; TDA gated.
* **The prediction contract of `docs/architecture.md` §45.9 is delivered** as
  `src/detection/contract.py`: entity, score, rank, decision with threshold
  basis, model identity and version, feature version, task, score kind,
  evaluation reference, analysis window, evidence. Enforced at construction.
* **Two engines implement it** (`src/detection/engines.py`):
  `LegacyGraphSAGEEngine` restates the frozen checkpoint's output unchanged,
  identified by checkpoint digest; `CandidateRiskEngine` scores the `aml_*`
  families of any benchmarked candidate over an analysis window, through the
  new `Candidate.score_features` seam.
* **Every workspace run persists `assessment.json`**, contained so it can never
  cost a prediction.

Reasoning:

Option 1 omits the half that explains alerts and addresses circular flows.
Option 2 is contradicted by the calendar-forward benchmark. Option 4 has no
measured ranking. Option 3 is the only one whose ranking claim is measured and
whose investigation half serves the objective without a new model. The contract
is the smallest change that makes the model replaceable and makes the
retrospective/forecasting distinction impossible to lose.

Advantages:

* The model can be replaced without touching persistence, investigation or UI.
* The task and score kind travel with every score.
* The frozen model contract is untouched: its numbers pass through
  the legacy engine bit-identically (tested against the pinned Reference values).

Disadvantages:

* The UI still reads legacy column names; until it migrates, the contract has
  one consumer (persistence).
* The two engines take different inputs.
* The within-period evidence for retrospective detection does not yet show
  transfer to a later period.

Consequences:

* Next steps, in order: cross-period retrospective detection (roadmap R1), a
  persisted and versioned production engine (R4), the UI adapter reading the
  contract.
* No model was trained or changed in this phase.

Alternatives rejected:

Options 1, 2 and 4 above. Replacing the frozen checkpoint in the UI now, which
would ship a model whose cross-period behaviour is unmeasured.

---

# Decision: Classify Cross-Period Detection Transfer as LIMITED, and Require Per-Window Thresholds

Date: 2026-09-12

Status: Accepted

Context:

`docs/system_architecture_decision.md` chose retrospective detection over an
analysis window as the ML objective, and recorded as roadmap R1 the missing
evidence: the IBM standard benchmark (AP 0.3827) trains and scores inside one
period, so nothing showed that a fitted model still works on a *later* period.
Forecasting had already been measured and found weak
(`docs/calendar_forward_benchmark.md`).

Problem:

Cross-period detection is a third regime, distinct from both. Its semantics -
features and labels from the *same* completed window, transfer measured across
windows - had no implementation, and the framework's existing temporal machinery
was built for forecasting.

Evidence (`docs/r1_detection_transfer.md`, `outputs/experiments/ibm_r1_*.json`;
pre-registered before any model was fitted; IBM AMLworld is synthetic):

Four equal 2-day detection windows tile 2022-09-03 .. 09-11, excluding the Sep 1
burn-in and the Sep 11-18 tail; three chronological folds; 1,397-1,497 positive
entities per test window; boosting at three seeds.

* Fold W1->W2: AP `0.3145 ± 0.0050`, 20.5x the label-free floor, AP retention
  vs the same-period reference **0.975**, FP/TP 2.07 at its frozen threshold.
* Fold W2->W3: AP `0.3425 ± 0.0034`, 17.6x floor, retention **0.984**,
  FP/TP 3.15. Zero features past PSI 0.25.
* Fold W3->W4: AP `0.0396 ± 0.0080`, 3.2x floor, retention **0.111**, and the
  frozen threshold admits 12 % of the population at FP/TP 73.8. 16 of 48
  features past PSI 0.25, `aml_temporal_v1` family median PSI 0.272.
* Transfer is not entity memorisation: new entities score as well as seen ones
  (fold 1: 0.349 vs 0.346), and first-time-flagged populations reach AP
  0.26-0.29 in the two transferring folds.
* Graph families remain a wash: paired AP −0.0031 / +0.0141 / −0.0052 across
  folds (mean +0.0019, positive in 1 of 3).
* Ruled out as the cause of the fold-3 collapse: thinner evidence per positive
  (flagged transactions per positive entity are median 1, mean 1.31-1.38 in
  every window). Consistent with it: W4 is Fri+Sat where W3 is Wed+Thu, carries
  354,447 entities against 235,433 at 1.6x lower prevalence, and 12.4 % of its
  positives continue into the excluded tail against 2.7-5.8 % earlier.

Decision:

* **Cross-period detection transfer on IBM HI-Small is LIMITED.** The capability
  is real and substantial between comparable windows, and it collapses when
  window composition differs materially. Report it as such - never as "the model
  generalises across time", and never with the forecasting numbers.
* **A frozen decision threshold is not portable across windows.** Production
  re-derives the alert budget on each analysis window. Carrying fold 2's
  threshold moved the alert rate from ~1 % to 12 %.
* **PSI on `aml_temporal_v1` between the fitted and the scored window is a
  gating signal**, promoted from "valuable later" to a precondition for applying
  a frozen model to a new window. It separated the folds that transferred from
  the one that did not.
* **The graph NO-GO stands.** R1 produced no consistent paired AP gain.
* **Implementation:** `src/experiments/detection_transfer.py` (new strategy
  `detection_transfer`, new feature scope `detection_window`),
  `src/experiments/snapshots.py` and `src/experiments/drift.py` extracted so the
  forecasting and detection regimes share one snapshot cache and one PSI
  definition. The frozen model reaches the later window through
  `CandidateRiskEngine`, so R1 runs on the production contract.

Reasoning:

The three regimes now measured answer three questions, and the spread between
them is the project's most important result: within-period detection 0.3827,
cross-period detection 0.31-0.34 between comparable windows but 0.04 across a
dissimilar one, and forecasting 0.0133. A product claim can rest on the second
only with the comparability condition and per-window thresholds attached.

Advantages:

* The product's central claim now has temporal evidence, with its failure mode
  located and measured rather than assumed.
* Detection and forecasting semantics are separate strategies, scopes and
  artifacts; neither can be reported as the other.

Disadvantages:

* Eight usable days admit one window design and three non-independent folds.
* Same-window semantics mean features summarise the transactions the label is
  derived from; R1 measures transfer of that recognition, not independent
  evidence.
* Synthetic data throughout.

Consequences:

* Next experiment is R6: window-relative temporal features (`aml_temporal_v2`),
  which R1 localised as the failure mode. Success is fold-2 retention >= 0.5
  with folds 0-1 >= 0.9.
* No GNN, no TDA, no transaction-level model was built.

Alternatives rejected:

Using the Sep 11-18 tail as a later period (99.9 % positive, 100 % ACH - a
simulator artifact); assigning entities to periods by their own last activity
(the survivorship design of `docs/temporal_collapse.md`); a single train/test
boundary, which would have hidden that two folds transfer and one does not.

# Decision: Retain `aml_temporal_v1` - Window-Relative Temporal Features Do Not Fix Cross-Period Transfer

Date: 2026-09-12. Status: accepted. Supersedes nothing; closes roadmap R6.

Context:

R1 classified cross-period detection transfer as **LIMITED** and localised the
W3->W4 collapse to temporal features that encode calendar position and
day-volume distribution: `daily_concentration` (PSI 2.49), `recency_days`
(1.52), `span_days` (1.02), `second_half_share` (0.96). R6 was recorded as the
next experiment with a pre-registered success condition.

Problem:

Whether that localisation is a *cause* or a *symptom*. If calendar encoding
causes the collapse, re-expressing those quantities on the window's own activity
clock should restore transfer. If window composition causes it, the
re-expression will change the numbers without rescuing the fold.

Evidence (`docs/r6_temporal_v2.md`, `docs/r6_temporal_v2_table.md`,
`outputs/experiments/ibm_r6_*.json`; pre-registered before the run; IBM AMLworld
is synthetic):

A single-variable ablation of R1. `aml_temporal_v1` -> `aml_temporal_v2` (15
columns against 16: ten redefined on the activity clock `tau`, five wall-clock
velocity columns unchanged, `weekend_share` dropped). Dataset SHA-256
`b19d39f5...16c5b040`, folds, window semantics, labels, eligibility, model,
hyperparameters, preprocessing, seeds 42/1337/7, 1 % alert budget, metrics and
leakage controls all byte-identical in config; only `features.sets` differs.

* W1->W2: AP `0.3145 ± 0.0050` -> `0.2762 ± 0.0087`, retention **0.975 ->
  0.881**.
* W2->W3: AP `0.3425 ± 0.0034` -> `0.3307 ± 0.0042`, retention 0.984 -> 0.977.
* W3->W4: AP `0.0396 ± 0.0080` -> `0.0633 ± 0.0004`, retention **0.111 ->
  0.192**, lift 10.1 -> 16.1, seed sd/mean **0.202 -> 0.007**.
* The extreme position drift *was* compressed: the largest temporal PSI falls
  2.4934 -> 0.9122, and `daily_concentration` / `recency_days` leave the top
  five. The typical drift did not move: `aml_temporal_*` family median PSI
  0.2722 -> 0.2717, still 16 of 48 features past 0.25, overall median 0.1175 ->
  0.1622.
* The operating point is not stabilised: at the frozen threshold W3->W4 still
  admits ~13 % of the population (46,541 alerts against 44,282) at FP/TP 72.7
  against 75.2.
* Leakage: 0 findings above `info` in all three runs; 14 dedicated tests prove
  the designed invariance, independence from `as_of` and from labels, and
  window-locality.

Decision:

* **`aml_temporal_v1` remains the production temporal family.** The
  pre-registered FAIL condition is met on its own terms: W3->W4 retention stays
  below 0.30 **and** the relevant drift is not materially reduced at family
  level. v2 also *cost* 0.094 retention and 12 % relative AP on W1->W2 - the
  comparable-window case production faces most of the time. A representation
  that helps the fold you cannot detect and hurts the fold you can is not a
  production improvement.
* **`aml_temporal_v2` stays in the repository as a registered, tested family
  that is measured and not adopted.** It is referenced only by the three R6
  configs and is not part of the production feature contract.
* **Temporal-feature experimentation stops here.** No R7/R8 temporal
  transformation is proposed. Per the pre-registered rule, window comparability
  - population composition - is the stronger explanation and is treated as a
  data requirement, not a feature-engineering target.

Reasoning:

The activity clock normalises *when* things happened; it cannot normalise *who
is present*. After the change the largest remaining shift among the model's
features is `amount_top1_share` (1.1735), an `aml_txn_v1` column v2 deliberately
does not touch, followed by activity-clock columns that still differ
(`span_activity` 0.91, `active_bin_fraction` 0.89). W4 carries 354,447 eligible
entities against W3's 235,433 at 1.6x lower prevalence. Calendar encoding was
the symptom R1 could see, not the cause.

The one durable finding is diagnostic rather than performance: v2's W3->W4 seed
sd/mean of 0.007 against v1's 0.202 shows the failure there is *systematic*, not
seed noise. That strengthens the R1 reading instead of weakening it.

Advantages:

* The R1 localisation is now tested rather than assumed, at the cost of one
  controlled ablation.
* The production feature contract is unchanged, so no migration is owed.
* A registered, invariance-tested v2 remains available if a dataset with
  genuinely comparable windows ever justifies it.

Disadvantages:

* Three non-independent folds on one synthetic simulator; the negative result is
  about IBM HI-Small, not about window-relative features in general.
* v2's W3->W4 gain (+0.0236 AP, +60 % relative) is real and is being declined
  because of its W1->W2 cost - a defensible trade, not a free one.

Consequences:

* Production safeguards, already indicated by R1 and unchanged by R6: model and
  feature versioning carried by `RiskAssessment`, **per-window alert-budget
  thresholds**, and a **PSI gate** refusing or flagging a frozen model applied
  to a window whose temporal-family drift exceeds the level at which R1
  measured transfer failure.
* Next action is that gate in the risk-engine path, not another feature
  experiment.
* `docs/system_architecture_decision.md` §5 feature contract is unchanged.

Alternatives rejected:

Quantile/rank-normalising features against the scored population, which would
drive the PSI comparison to zero by construction and make the drift evidence
circular; transforming `aml_txn_v1`'s `amount_top1_share` in the same run, which
would have broken the single-variable ablation; running several v2 variants and
reporting the best, which the pre-registration forbids; dropping W3->W4 as
"not comparable", which would have removed the only fold the experiment was
about.
