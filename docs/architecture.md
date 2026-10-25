# Graph-Based Mule Account & Circular Transaction Detection

## System Architecture Specification

**Document Status:** Living Architecture Document
**Architecture Version:** 0.1
**Project Phase:** Research / Architecture / Initial Implementation
**Primary Objective:** Detect complex financial-crime patterns using graph-based machine learning, temporal analysis, topological analysis, and explainable risk scoring.

---

# 1. System Purpose

The system is designed to identify suspicious financial behaviour that may be difficult to detect using traditional transaction-level rules.

The primary detection targets are:

* Mule accounts
* Circular transaction networks
* Money-laundering rings
* Suspicious account communities
* Shell-company-like transaction structures
* Synthetic-identity-related behaviour
* Coordinated transaction activity
* Structurally anomalous transaction networks
* Temporally anomalous transaction behaviour

The system is intended to operate at multiple levels:

1. **Transaction level**
2. **Account/entity level**
3. **Subgraph/network level**

The architecture combines conventional financial features with graph structure, temporal behaviour, learned representations, and topological information.

---

# 2. Architectural Goals

The architecture must satisfy the following goals.

## 2.1 Detection capability

The system should detect suspicious behaviour that arises from:

* Individual anomalous transactions
* Repeated transactions
* Abnormal transaction sequences
* Suspicious account relationships
* Circular flows
* Dense suspicious communities
* Multi-hop relationships
* Coordinated entities
* Temporal bursts
* Complex network topology

---

## 2.2 Modularity

Major components must be independently replaceable.

For example:

```text
Data Layer
    ↓
Graph Layer
    ↓
Feature Layer
    ↓
Model Layer
    ↓
Detection Layer
```

Replacing the GNN should not require rewriting the data ingestion system.

Replacing the TDA implementation should not require rewriting graph construction.

Replacing the dashboard should not affect the ML pipeline.

---

## 2.3 Research extensibility

The architecture must support experimentation with multiple approaches.

Potential alternatives include:

* Different graph representations
* Different GNN architectures
* Different temporal modelling approaches
* Different TDA techniques
* Different feature-fusion strategies
* Different anomaly-detection strategies
* Different explainability methods

The architecture must therefore avoid unnecessarily coupling the system to one technique.

---

## 2.4 Explainability

The system should not merely output:

```text
Fraud = TRUE
```

It should provide evidence explaining the classification.

Example:

```text
Entity: A1024

Risk Score: 0.94

Evidence:
- High outgoing transaction velocity
- Strong flow imbalance
- Multiple suspicious counterparties
- Participation in a circular transaction pattern
- Abnormal temporal behaviour
- Elevated graph-model risk
- Topological anomaly detected
```

---

## 2.5 Reproducibility

Experiments must be reproducible.

The system should preserve:

* Dataset version
* Feature configuration
* Model configuration
* Random seeds
* Dataset split
* Hyperparameters
* Training configuration
* Evaluation metrics

---

# 3. High-Level System Architecture

```text
                         ┌─────────────────────────┐
                         │     RAW DATA SOURCES    │
                         │─────────────────────────│
                         │ Transactions             │
                         │ Entities (by name_norm) │
                         │ Jurisdiction / country  │
                         │ [Future: Devices, KYC]  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    DATA INGESTION        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   DATA PREPROCESSING     │
                         │─────────────────────────│
                         │ Validation               │
                         │ Cleaning                 │
                         │ Normalization            │
                         │ Entity resolution        │
                         │ Temporal normalization  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   GRAPH CONSTRUCTION     │
                         │─────────────────────────│
                         │ Nodes                   │
                         │ Directed edges          │
                         │ Edge attributes         │
                         │ Node attributes         │
                         │ Temporal information    │
                         └────────────┬────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                  │
                   ▼                  ▼                  ▼
          ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
          │ GRAPH FEATURES │ │ TEMPORAL       │ │ TGA FEATURES   │
          │                │ │ FEATURES       │ │                │
          │ Degree         │ │ Velocity       │ │ Time-Respecting│
          │ Centrality     │ │ Burst activity │ │ Directed Cycles│
          │ Motifs         │ │ Sequences      │ │ Graph Topology │
          │ Cycles         │ │ Time gaps      │ │                │
          └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                  │                  │                  │
                  └──────────────────┼──────────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │    GRAPH ML MODEL       │
                         │─────────────────────────│
                         │ GNN                     │
                         │ Temporal modelling      │
                         │ Heterogeneous modelling │
                         │ Representation learning │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   RISK / ANOMALY ENGINE │
                         └────────────┬────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                  │
                   ▼                  ▼                  ▼
          ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
          │ ENTITY RISK    │ │ TRANSACTION    │ │ SUBGRAPH       │
          │                │ │ RISK           │ │ RISK           │
          └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                  │                  │                  │
                  └──────────────────┼──────────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │   EXPLAINABILITY        │
                         │─────────────────────────│
                         │ Feature evidence        │
                         │ Graph evidence          │
                         │ Temporal evidence       │
                         │ Topological evidence    │
                         │ Suspicious subgraphs    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  INVESTIGATION / UI     │
                         │─────────────────────────│
                         │ Risk dashboard           │
                         │ Graph visualization      │
                         │ Alerts                   │
                         │ Evidence                 │
                         └─────────────────────────┘
```

---

# 4. Architectural Layers

The system is divided into the following logical layers:

```text
Layer 1   Data Ingestion
Layer 2   Data Validation & Preprocessing
Layer 3   Entity Resolution
Layer 4   Graph Construction
Layer 5   Feature Engineering
Layer 6   Temporal Analysis
Layer 7   Topological Analysis
Layer 8   Graph Representation Learning
Layer 9   Detection & Risk Scoring
Layer 10  Subgraph / Cycle Analysis
Layer 11  Explainability
Layer 12  Evaluation
Layer 13  Visualization / Application
```

The layers should communicate through explicit interfaces rather than sharing uncontrolled global state.

## 4.1 External Dataset Admission Layers (Layer 0)

Layers 1–13 above assume data already conforming to the canonical schema — which is true of the Reference Dataset. External datasets must first be *admitted*: an arbitrary uploaded spreadsheet has unknown columns and unknown semantics, and nothing may enter Layer 1 until those are resolved and validated.

This admission stage sits **before** Layer 1 and is implemented in `src/datasets/`:

```text
Layer 0a  Workspace isolation & persistence
Layer 0b  Safe ingestion            (.csv / .xlsx / .xls, hashed, byte-preserved)
Layer 0c  Structural profiling      (types and statistics only — no semantics)
Layer 0d  Semantic mapping          (confidence-scored, evidence-retaining, reviewable)
Layer 0e  Deterministic derivation  (what can be computed, and under what rule)
Layer 0f  Canonicalization          (projection driven by the mapping manifest)
Layer 0g  Compatibility validation  (against the frozen 167-feature contract)
```

The controlling boundary is:

```text
ingestion → profiling → semantic mapping → canonicalization → compatibility
         → fixed feature encoding → graph / PyG → frozen Experiment 3 model
         → workspace predictions → UI
```

Two invariants define this boundary:

* **Semantics are never inferred inside the data path.** Profiling makes no semantic claim; mapping proposes with confidence and evidence; canonicalization only projects what the reviewed manifest already decided.
* **Nothing reaches the model without passing compatibility.** `predict_workspace` refuses any workspace whose persisted report is not `READY`.
* **A derivation is evidence-bearing, not gap-filling.** Layer 0e
  (`src/datasets/derivation.py`) states which canonical columns can be
  *computed* and under which rule. Each fires only where exactly one reading is
  possible and records the source columns it read; nothing is defaulted to a
  substantive value, no FX rate is invented, and no value crosses from one side
  of a transaction to the other. Ambiguity produces no derivation — the field
  stays missing for review.

Layers 0a–0g are implemented and execute end to end; see §32.2 and `docs/workspaces.md`.

---

# 5. Layer 1 — Data Ingestion

## Responsibility

Convert raw input data into a standardized internal representation.

Possible input sources include:

* CSV
* JSON
* Database exports
* Synthetic transaction generators
* Public financial datasets

The exact source depends on the selected dataset.

---

## Current dataset fields

The canonical raw dataset is `dkdata.json`. It contains the following fields:

```text
id
date
payer_name / payer_name_norm
payer_account
payer_type
payer_jurisdiction
payer_bank_country
payer_core
beneficiary_name / beneficiary_name_norm
beneficiary_account
beneficiary_type
beneficiary_jurisdiction
beneficiary_bank_country
beneficiary_core
amount_orig / amount_orig_currency
amount_usd / amount_eur
purpose
source_file
investigation
```

Additional fields may exist in future datasets. The ingestion layer must not assume that every future dataset contains every field.

---

## Output

A validated raw transaction representation.

Conceptually:

```text
Raw Source
    ↓
Parser
    ↓
Schema validation
    ↓
Internal transaction representation
```

---

# 6. Layer 2 — Data Validation and Preprocessing

## Responsibilities

* Missing-value handling
* Duplicate detection
* Data type conversion
* Timestamp normalization
* Amount normalization
* Invalid-record detection
* Data consistency checks
* Feature preparation

---

## Requirements

Raw data must remain immutable.

Use:

```text
data/raw/
data/interim/
data/processed/
```

for different processing stages.

The preprocessing pipeline must be reproducible.

---

# 7. Layer 3 — Entity Resolution

Financial datasets may contain multiple representations of the same underlying entity.

For the current dataset, the validated entity resolution policy is **Name-first**:
* `Entity_ID` = `name_norm`

This strategy is required because:
* `name_norm` exists for 100% of observations.
* 294 normalized names map to multiple accounts.
* 40 accounts map to multiple normalized names.
* 35 cross-role account collisions would artificially fragment transaction chains under an account-first strategy.
* 0 normalized names have conflicting core labels.

Account identifiers are NOT node identifiers. Instead, `payer_account` and `beneficiary_account` are preserved as transaction (edge) metadata showing which banking vehicle was used for a specific transaction.

*Future/Extended Architecture:* If additional datasets are introduced, potential future identifiers could include Device ID, IP, Customer ID, or Phone/email-derived identifiers. The current implementation must not invent these unless supported by data.

---

# 8. Layer 4 — Graph Construction

## Primary representation

The primary graph is a **directed financial transaction graph**.

The graph may be heterogeneous depending on the dataset.

---

## Node types

**Current Dataset Architecture:**
```text
Entity
```
*Future/Extended Architecture:* Additional nodes such as `Device`, `IP`, `Merchant`, and `Customer` are NOT supported by the current dataset and must remain documented only as future possibilities.

---

## Potential edge types

**Current Dataset Architecture:**
```text
Entity ──TRANSACTS_WITH──> Entity
```

*Future/Extended Architecture:* Complex heterogeneous identity graphs with edges like `owns`, `pays`, or `uses` require additional data and are not part of the current implementation.

---

# 9. Directedness

Transaction direction is semantically meaningful.

The following are distinct:

```text
A → B
```

and

```text
B → A
```

The graph layer must preserve direction unless a downstream algorithm explicitly requires an undirected representation.

Any conversion from directed to undirected must be documented.

---

# 10. Edge Attributes

Transaction edges (`TRANSACTS_WITH`) contain:

```text
amount_usd
amount_orig
currency
date
purpose
payer_account_used
beneficiary_account_used
```

Derived attributes may include:

```text
transaction_frequency
time_since_previous_transaction
amount_statistics
velocity
```

Derived features should remain distinguishable from original transaction attributes.

---

# 11. Multiple Transactions (Directed-Multigraph)

If:

```text
A → B
```

occurs multiple times, the system must not silently collapse all transactions into a single edge unless an explicit aggregation is defined. The architecture must preserve support for multiple directed transactions between the same pair of entities.

Possible representations include:

### Multi-edge representation

```text
A ──T1──> B
A ──T2──> B
A ──T3──> B
```

### Aggregated temporal representation

```text
A ──> B

count = 3
total_amount = ...
timestamps = [...]
```

The representation must be selected according to the downstream model requirements.

---

# 12. Layer 5 — Feature Engineering

Based on the verified preprocessed dataset and graph construction audit, the feature engineering boundaries are finalized into distinct experimental sets. This ensures that the baseline model is clearly separated from novel research upgrades.

## 12.1 Baseline Features
Features that must exist in the initial non-novel model. These use standard structural and categorical properties:
* **Entity Type:** Categorical.
* **Bank Country:** Multi-valued representation. (Some entities genuinely operate across multiple countries; this must be represented natively without normalization loss).
* **Log In-Degree:** `log(in-degree)`
* **Log Out-Degree:** `log(out-degree)`
* **Log Total Incoming USD:** `log(total incoming USD)`
* **Log Total Outgoing USD:** `log(total outgoing USD)`
* **Flow Imbalance:** Net flow direction metric.

*Important:* The baseline model must NOT include advanced temporal dynamics, parallel-edge ratios, or TDA features.

### 12.1.1 Fixed Feature Vocabulary (CURRENT)

The categorical vocabularies above are **frozen constants**, not derived from whatever dataset is being processed. This is a change from the original dynamic behaviour and is recorded in `docs/decisions.md` → *Replace the Dataset-Derived Country Vocabulary with a Fixed Experiment 3 Training Vocabulary*.

**Motivation.** A dataset-derived vocabulary makes feature width and feature order a function of the input data. The frozen Experiment 3 checkpoint requires exactly 167 features in exactly one order, so a derived vocabulary makes the model structurally unusable on any dataset other than the Reference Dataset — and, worse, could produce a same-width but differently-ordered matrix whose predictions are silently meaningless. The vocabulary is part of the trained model's contract, not a property of the input data, so it belongs with the model.

Defined in `src/features/baseline.py`:

* `_KNOWN_ENTITY_TYPES = ("Company", "Person", "Invalid")`, plus an explicit `entity_type_unknown` column — **4 columns**.
* `_KNOWN_COUNTRY_CODES` — **151 literal codes** as observed during Experiment 3 training, including non-ISO encodings (`.A`, `.D`, `.K`, `/M`, `00`–`99`, `G8`, `JH`, `R0`, `T1`, `TA`, `TX`). See `docs/research.md` §2 for why these were never normalized.

Fixed feature order:

```text
  5   structural scalars   log_in_degree, log_out_degree,
                           log_total_incoming_usd, log_total_outgoing_usd,
                           flow_imbalance
  4   entity type          entity_type_{Company,Person,Invalid,unknown}
151   bank country         bank_country_*
───
160   baseline features
```

**Out-of-vocabulary policy** (both explicit and tested in `tests/test_feature_contract.py`):

* **Unseen country codes are ignored.** They set no column; an entity whose countries are all unseen receives an all-zero country block. They are *not* bucketed into an "other" column, because the checkpoint has no such column.
* **Unseen entity types map to `entity_type_unknown`**, which does exist in the checkpoint.

*Known defect:* the module docstring of `src/features/baseline.py` still describes the superseded dynamic behaviour. The code is correct; the docstring is stale.

## 12.2 Temporal Upgrade
Features representing our temporal behavioural enhancement.
* **7-Day Transaction Activity**
* **30-Day Transaction Activity**
* **Mean Inter-Transaction Time**
* **Burstiness:** Coefficient of variation of inter-transaction times.
* **Active-Day Ratio**
* **Same-Day Inflow/Outflow Co-occurrence**

*Important Constraint:* Because the dataset dates have day-level precision, "same-day inflow/outflow co-occurrence" must NOT be described as causal pass-through. Do not claim that an inflow occurred before an outflow when both occur on the exact same day.

## 12.3 Directed-Multigraph Upgrade
Features and representations enabled specifically by preserving directed parallel transactions instead of collapsing them to a simple graph.
* **Repeated-Counterparty Ratio**
* **Parallel-Edge Transaction Count**
* **Parallel-Edge Temporal Dispersion**
* **Reciprocal Transaction Behaviour**

*Important Constraint:* These must NOT be included in the baseline feature set.

## 12.4 TDA (Topological Data Analysis) Inputs
* TDA features remain a strictly separate downstream research component.
* Do not prematurely convert TDA outputs into ordinary baseline features.

---

# 13. Layer 6 — Temporal Analysis & Leakage Prevention

Temporal modelling is essential because suspicious behaviour often occurs through transaction sequences. However, distinguishing between static retrospective analysis and future-aware predictive tasks is critical to avoid temporal leakage.

## 13.1 Retrospective / Static Analysis
* Valid for transductive anomaly detection or forensic investigation of a frozen historical dataset.
* Global graph statistics (computed over the entire timeline, including future events relative to early transactions) are acceptable in this specific context.

## 13.2 Causal Temporal Prediction
* Valid for predictive models forecasting future suspicious behavior.
* Features must avoid future information. If predicting behaviour at time `t`, future transactions (where `t_tx > t`) must not influence the prediction.
* Global graph statistics must NOT be used for causal temporal prediction.

---

# 15. Layer 7 — Topological Data Analysis

TDA is intended to capture structural properties that conventional graph statistics may miss.

Conceptual pipeline:

```text
Graph
  ↓
Filtration
  ↓
Topological representation
  ↓
Persistence computation
  ↓
Persistence features
  ↓
Fusion with ML pipeline
```

Potential representations:

* Persistence diagrams
* Persistence landscapes
* Persistence images
* Persistence statistics
* Time-respecting directed-cycle features
* Other appropriate vectorizations

The final method must be selected based on:

* Relevance
* Computational cost
* Interpretability
* Predictive contribution
* Dataset structure

**Status:** TDA is a research component and is **not part of the shipped inference path**. The final filtration and construction strategy is NOT yet finalized. Available filtration variables from the current dataset are `amount_usd` (edge weight) and `date` (day-level). Intra-day filtration is not available. TDA will operate downstream on the Entity–Entity directed graph. Cross-identity linkage features for TDA require additional data not present in the current dataset.

The one TDA-family construction that has been evaluated is the three
time-respecting directed-cycle features (`src/features/tda.py`), measured in
**Experiment 5** (§28.3). It produced no uplift over the baseline. That result
is a property of these three features under this protocol on this dataset, and
does not resolve the broader filtration question above — persistence-based
representations remain unimplemented and unevaluated.

---

# 16. Layer 8 — Graph Representation Learning

The main machine-learning component is a graph-based representation-learning system.

Potential model families:

```text
GraphSAGE
GAT
Heterogeneous GNN
Temporal GNN
Other validated architectures
```

The architecture should remain configurable.

---

# 17. Prediction Targets

> **Current product target (2026-09-11):** entity risk ranking over a declared
> analysis window - retrospective detection - with graph and temporal analytics
> as post-prediction investigation evidence. The node-classification framing
> below is the Reference Dataset's; see §47 and
> `docs/system_architecture_decision.md`.

## Primary task — Node classification (Directly supervised)

```text
Entity → core / non-core
```

The `core` boolean label (`payer_core` / `beneficiary_core`) is the confirmed ground truth for the current dataset. It is 100% consistent across all observations for a given `name_norm`. This is the primary supervised ML task.

## Secondary tasks

**Edge classification (Weakly supervised)**
```text
Transaction → suspicious / legitimate
```
An edge can be weakly labelled as suspicious if it connects two `core` nodes, but there is no direct per-transaction ground truth label in the dataset.

**Subgraph / cycle analysis (Unsupervised / derived)**
```text
Transaction subgraph → suspicious structure
```
No per-subgraph ground truth exists. Cycle and community detection operate as unsupervised structural analysis, not as a supervised classification task.

**Anomaly detection (Unsupervised / derived)**
Can be used as a baseline, but supervised node classification is the stronger primary approach given available labels.

---

*Note: `source_file`, `investigation`, and raw name/account fields must NOT be used as ML features. They are identifiers or data-provenance metadata, not behavioural signals.*

---

# 18. Model Pipeline

Conceptually:

```text
Graph
  +
Node features
  +
Edge features
  +
Temporal information
        ↓
Graph encoder
        ↓
Node / edge / subgraph embeddings
        ↓
Prediction head
        ↓
Risk probability
```

For heterogeneous graphs:

```text
Node type
+
Edge type
+
Node attributes
+
Edge attributes
        ↓
Heterogeneous message passing
```

---

# 19. Class Imbalance

Financial fraud datasets are expected to be imbalanced.

The model-training layer must support:

* Class-weighted loss
* Focal loss
* Sampling strategies
* Negative sampling where appropriate

The chosen approach must be evaluated.

Accuracy must not be treated as the sole performance metric.

---

# 20. Layer 9 — Detection and Risk Scoring

> **Decided 2026-09-11 (§47):** no score fusion. One risk engine produces the
> ranking; graph, temporal and cycle evidence is attached *after* scoring and
> never changes a score or rank. The fusion options below were not needed: on
> IBM HI-Small graph features add +0.0093 AP retrospectively and nothing when
> forecasting.

The detection layer combines evidence from multiple sources.

Conceptually:

```text
GNN output
    +
Graph structural evidence
    +
Temporal evidence
    +
TDA evidence
    +
Explicit suspicious-pattern evidence
        ↓
Risk scoring
```

The exact fusion mechanism is not finalized.

Potential approaches:

* Feature-level fusion
* Embedding-level fusion
* Decision-level fusion
* Learned fusion
* Attention-based fusion

---

# 21. Risk Score

The system should produce a normalized score, for example:

```text
0.00 ───────────────────────────── 1.00
Low                              High
```

The mathematical relationship between model probability and risk score must be explicitly defined.

Do not automatically equate:

```text
model_probability = risk_score
```

unless experimentally justified.

---

# 22. Layer 10 — Cycle and Subgraph Detection

Circular transaction detection is a primary project objective.

Examples:

```text
A → B → C → A
```

```text
A → B → C → D → A
```

The system should investigate:

* Cycle length
* Cycle frequency
* Amount consistency
* Time intervals
* Repeated cycles
* Overlapping cycles
* Number of participants
* Flow direction
* Cycle density

## 22.1 Two different cycle measurements — do not compare them

The repository now contains **two** things that count "cycles". They measure
different quantities, on different graphs, for different purposes. Their numbers
differ by orders of magnitude, and that is arithmetic, not disagreement. Neither
is a correction of the other, and they must never be presented side by side as
competing counts.

| | Experiment 5 / TGA (`src/features/tda.py`) | Layer 10 (`src/detection/cycles.py`) |
|---|---|---|
| Counts | **Transaction combinations** | **Entity-level directed cycles** |
| Unit | One ordered choice of transaction per leg | One structure in the graph |
| Graph | 1-hop ego-network per node, parallel edges enumerated | Whole graph, parallel edges collapsed to unique `(source, target)` pairs |
| Parallel edges | Multiplied out (cross-product) | Collapsed, then reattached as per-leg evidence |
| Rotation | Each rotation counted separately per starting edge | Normalised to one canonical identity |
| Temporal rule | Time-respecting selection required to count | Reported as a three-valued property of the structure |
| Output | Three per-node scalars | Structured evidence objects |
| Role | Model feature candidate — **evaluated and rejected** (Finding 3.4) | Post-hoc investigator evidence — never a model input |

The same three entities carrying 50 transactions per leg are:

```text
TGA        up to 125,000 time-respecting transaction combinations
Layer 10   1 cycle, 3 legs, 50 transactions each
```

On the Reference Dataset the gap is real and measured: TGA's
`temporal_cycle_count` reaches 3.1 × 10⁷ for a single node (`scripts/legacy/audit_tga.txt`),
while Layer 10 finds **5,589** entity-level cycles in the entire graph at
`max_cycle_length = 6`.

Both are correct answers to their own question:

* TGA asks *"how many ways could value have moved around a loop through this
  node, in time order?"* — a magnitude, suited to being a feature.
* Layer 10 asks *"which loops exist, between whom, and what happened on each
  leg?"* — a structure, suited to being shown to an investigator.

Experiment 5 measured TGA's predictive contribution and found none
(`docs/research.md` Finding 3.4). That is a result about *those three scalar
features*; it says nothing about the value of cycles as investigator evidence,
which is a separate and still-unevaluated question (`docs/research.md` →
*Not yet established*).

Layer 10 is implemented in `src/detection/cycles.py`, tested in
`tests/test_cycles.py`, and reaches both analysis paths through the shared
`AnalysisContext` (`tests/test_cycles_context.py`). It reads the authoritative
`MultiDiGraph` and contributes nothing to the 167-feature representation, the
frozen checkpoint, `risk_score`, `predicted_class`, compatibility status, or the
trust assessment:

```text
Graph
  ├──> frozen 167-feature representation ──> Experiment 3 GraphSAGE
  │                                              └──> risk_score / predicted_class
  └──> cycle / structural analysis        ──> investigator evidence
```

Bounds are configuration (`CycleAnalysisConfig`), not hidden constants, and are
echoed into every result's metadata: `max_cycle_length = 6`,
`min_cycle_length = 2`, `max_cycles = 10000`. The retention bound is a hard
safety cap for pathological graphs, set above the Reference Dataset's complete
enumeration (5,589 cycles, ~2 s) so the normal result is complete rather than a
truncated sample. A run that does hit it reports `truncated = true` with a
reason; nothing is discarded silently.

### Availability — **CURRENT**

Cycle evidence is computed by `WorkspaceAnalysisResolver` and carried on
`AnalysisContext.cycles`, for the Reference Dataset and for every external
workspace alike, off the graph that path already built. It is cached exactly as
the context is: computed once per resolved context, dropped by `invalidate`.

Because it runs *after* predictions and reads only the graph, an enumeration
failure is a value (`CycleAnalysisUnavailable`), not an exception: it cannot
change `risk_score`, `predicted_class`, `READY` / `DEGRADED` / `BLOCKED`, or the
trust assessment. "Analysis failed" and "no cycles found" are represented
separately and never collapse into one another. See `docs/UI_ARCHITECTURE.md` →
*Cycle evidence on the context*.

It is served read-only by two routes that read the cached context and enumerate
nothing of their own (`docs/UI_ARCHITECTURE.md` → *`GET /api/cycles`*):
`GET /api/cycles` is a paged index of summary rows plus aggregates over the
whole selection, and `GET /api/cycles/<cycle_id>` returns one cycle's complete
evidence. The split exists because the full evidence for the Reference Dataset
is 19.3 MB, against 50 KB for a default index page — a client fetches detail
only for the cycle it is looking at. What an index row omits is named in the
response, and the page in force is always reported, so neither reduction is
silent.

Both report `COMPLETE` with a possibly-empty result, or `UNAVAILABLE` with a
`null` payload — the two can never be confused, so a failed analysis is never
served as "this dataset has no cycles".

Cycle results are **not persisted**: they are derived from the graph on demand
and live only as long as the context that carries them. There is no UI surface
yet.

---

# 23. Suspicious Subgraph Detection

The system should identify connected suspicious structures.

Example:

```text
             Entity A
             /       \
            v         v
       Entity B    Entity C
            |         |
            v         v
       Entity D <----+
```

The final system should allow suspicious structures to be extracted for investigation.

---

# 24. Layer 11 — Explainability

The explainability layer converts model output into investigator-readable evidence.

Possible explanation sources:

### Feature evidence

```text
High transaction velocity
High outgoing volume
Unusual counterparty diversity
```

### Graph evidence

```text
High centrality
Suspicious neighbourhood
Participation in dense community
```

### Temporal evidence

```text
Rapid fund movement
Burst activity
Repeated short-interval transfers
```

### Topological evidence

```text
Abnormal persistence structure
Suspicious topological signature
```

### Pattern evidence

```text
Participation in circular transaction path
```

---

# 25. Explanation Output

Example:

```text
Entity: A1024

Risk Score: 0.94

Primary evidence:
1. High outgoing transaction velocity
2. Rapid inflow → outflow behaviour
3. Participation in a 5-node transaction cycle
4. Strong connection to previously suspicious entities
5. Abnormal temporal activity
6. Elevated graph-model risk
```

The explanation must distinguish:

```text
Observed evidence
vs
Model-derived evidence
vs
Heuristic/rule-based evidence
```

---

# 26. Layer 12 — Evaluation

The evaluation system must support:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrix
* False-positive rate
* False-negative rate

For heavily imbalanced fraud datasets, PR-AUC, precision and recall are particularly important.

---

# 27. GNN Experimental Design (Baseline Framework)

The following strictly defines the baseline evaluation for the GNN architecture based on the verified dataset properties.

## 27.1 Primary Task
**Transductive node classification** of canonical Entity nodes:
```text
Entity → core / non-core
```

## 27.2 Baseline Inputs
The exact inputs to the FIRST baseline GNN experiment:
* **Node Features:** `3,440 × 160` baseline node feature matrix only.
* **Graph Topology:** Full directed multigraph topology in PyTorch Geometric (PyG).
* **PyG Semantics:** `edge_index` preserves duplicate/parallel edges as separate columns. Message aggregation may combine their contributions during propagation. This does NOT mean the underlying graph representation has collapsed the edges.
* **STRICT EXCLUSIONS:** The baseline MUST NOT use:
  * Temporal upgrade features
  * Multigraph upgrade features
  * TDA features
  * Target labels as input (no label propagation)
  * Edge attributes in the baseline
  * `source_file`, `investigation`, raw names, or account identifiers

## 27.3 Baseline Model Architecture
* **Model:** Two-layer GraphSAGE
* **Hidden dimension:** 64
* **Activation:** ReLU
* **Dropout:** 0.5
* **Output:** Final binary logit output

## 27.4 Loss & Class Imbalance
* **Loss function:** `BCEWithLogitsLoss`.
* **Imbalance handling:** A dynamically computed `pos_weight` derived from the training set mask is strictly required:
  ```python
  pos_weight = negative_training_count / positive_training_count
  ```

## 27.5 Evaluation Protocol
**Primary Evaluation: Stratified 4-fold cross-validation.**
Because there are exactly 4 positive entities:
* Each fold contains exactly 1 positive entity in the test set.
* Each fold contains exactly 1 positive entity in the validation set.
* The remaining 2 positive entities are used for training.
* The graph topology remains fixed across folds; only the supervision masks change.

**Secondary Reproducibility Split:**
An optional fixed stratified 50/25/25 split is defined for reproducibility and demo purposes, but is NOT the sole scientific evaluation.

**Threshold Selection:**
The classification threshold must be selected using ONLY the validation set. The selected threshold is then applied unchanged to the test set. Never tune the classification threshold on the test set.

**Reported Metrics:**
* PR-AUC (Primary Metric)
* ROC-AUC
* Precision
* Recall
* F1-score
* Confusion matrix

Report fold-by-fold results and mean ± standard deviation.

> [!WARNING]
> **Statistical Limitation:** With only four positive entities, all evaluation metrics have high variance. Results should not be interpreted as statistically strong evidence from this dataset alone.

---

# 28. Ablation Architecture

To systematically measure the contribution of each engineered feature group, experiments follow a strict controlled-addition protocol:

```text
Experiment 1:
    Baseline GNN

Experiment 2:
    Baseline + Temporal Upgrade

Experiment 3:
    Baseline + Directed-Multigraph Upgrade

Experiment 4:
    Baseline + Temporal + Directed-Multigraph

Experiment 5:
    Final System + targeted TDA
```

## 28.1 Experiment 2 Preprocessing Policy

To ensure numerical stability during neural network optimization without distorting the underlying temporal semantics or leaking test-set statistics, Experiment 2 applies a strict, deterministic preprocessing policy to the six temporal features before appending them to the baseline feature matrix. This ensures the raw temporal magnitudes (which can reach into the hundreds) do not overwhelm the properly scaled baseline features.

**Handling Insufficient-History NaNs:**
Entities with fewer than two transactions have undefined `mean_iat` and `burstiness`. To explicitly distinguish undefined temporal statistics from genuine zero-valued statistics (e.g., instantaneous transactions), these `NaN` values are deterministically filled with `-1.0`. Filling with `0.0` is mathematically incorrect as it conflates "no history" with "perfectly regular" or "instantaneous" history.

**Exact Feature Transformations:**
The following transformations are applied deterministically and identically across all folds:

* **`activity_7d`**: Transformed using `log1p(x)`. Raw values are unscaled transaction counts. Logarithmic compression bounds the scale to be commensurate with the baseline degree metrics.
* **`activity_30d`**: Transformed using `log1p(x)`. Like 7-day activity, raw counts can reach into the hundreds, requiring logarithmic compression to prevent gradient instability.
* **`mean_iat`**: For valid statistics (`x >= 0`), transformed using `log1p(x)`. Undefined values (`NaN`) are explicitly represented as `-1.0`. Inter-transaction times exhibit a heavy-tailed distribution (spanning from 0 to hundreds of days), making logarithmic scaling appropriate to match the baseline domain while preserving the distinct `-1.0` flag for missing data.
* **`burstiness`**: Left unscaled. Undefined values (`NaN`) are explicitly represented as `-1.0`. For entities with defined burstiness, the coefficient of variation naturally bounds itself into a low-variance numeric range (typically between 0 and 7), requiring no further compression.
* **`active_day_ratio`**: Left unscaled. This metric is natively bounded between `0.0` and `1.0`.
* **`same_day_cooccurrence`**: Left unscaled. This metric evaluates to a deterministic value strictly bounded between `0.0` and `1.0`.

This policy preserves the day-level temporal semantics defined in Layer 4 while ensuring the GNN inputs remain within a safe optimization envelope comparable to Experiment 1.

## 28.2 Experiment 3 Preprocessing Policy

To ensure numerical stability during neural network optimization without modifying the underlying multigraph semantics or leaking test-set statistics, Experiment 3 applies a strict, deterministic numerical conditioning preprocessing policy to the seven directed-multigraph features before they are appended to the baseline feature matrix.

**The Seven Multigraph Features:**
1. `out_repeated_counterparty_ratio`
2. `in_repeated_counterparty_ratio`
3. `max_parallel_edge_count`
4. `mean_parallel_edge_count`
5. `parallel_temporal_dispersion`
6. `reciprocal_out_count`
7. `reciprocal_in_count`

**Exact Feature Transformations:**
The following transformations are applied to all seven features deterministically and identically across all folds:
* **`log1p(x)` Transformation**: Apply `log1p(x)` to each of the seven features before model tensor construction. This bounds their scale (which naturally extends into the hundreds) to be commensurate with the baseline metrics.
* **NaN Encoding**: While undefined statistics are not expected due to node connectivity requirements, if any unexpected `NaN` occurs, it must be encoded deterministically as `-1.0`.

**Architectural Rules for Experiment 3:**
* The underlying Layer 4 feature definitions must remain completely unchanged. This preprocessing constitutes numerical conditioning only and does not alter semantic meaning.
* No dataset-level mean/std standardization is permitted.
* No train/test-derived scaling statistics are permitted.
* No additional feature transformations are permitted.
* No temporal features, TDA, or edge attributes are included.

**Controlled Inclusion of Redundant and Temporal Dependencies:**
* **Redundancy**: In the current implementation, `reciprocal_out_count` and `reciprocal_in_count` are mathematically identical. However, both remain intentionally included in the input tensor to strictly test the 7-feature specification. Neither should be removed or redesigned.
* **Temporal Leakage**: `parallel_temporal_dispersion` natively utilizes transaction dates to compute standard deviation. In the context of this specific experiment (static transductive node classification), this is utilized retrospectively/statically. True causal temporal prediction in production would require proper `as_of` temporal filtration.

**Expected Experiment 3 Input Tensor:**
* Baseline features (160) + Multigraph features (7) = **167 total features**.

## 28.3 Experimental Results (Experiments 1–5)

A strictly controlled ablation study was conducted across five experiments. All five experiments utilized identical GraphSAGE architectures, model hyperparameters, dataset partitioning, and the same fixed 4-fold cross-validation (CV) protocol. Extensive audits confirmed that no target-label leakage was present in any configuration. 

### Quantitative Results

**Experiment 1: Baseline GNN**
*   **Features:** 160
*   **PR-AUC:** `0.7534 ± 0.4271`
*   **ROC-AUC:** `0.9790 ± 0.0363`
*   **F1 Score:** `0.7500 ± 0.4330`

**Experiment 2: Baseline + Temporal**
*   **Features:** 166
*   **PR-AUC:** `0.7550 ± 0.4244`
*   **ROC-AUC:** `0.9857 ± 0.0247`
*   **F1 Score:** `0.6000 ± 0.4243`

**Experiment 3: Baseline + Directed-Multigraph**
*   **Features:** 167
*   **PR-AUC:** `1.0000 ± 0.0000`
*   **ROC-AUC:** `1.0000 ± 0.0000`
*   **F1 Score:** `1.0000 ± 0.0000`

**Experiment 4: Baseline + Temporal + Directed-Multigraph**
*   **Features:** 173
*   **PR-AUC:** `0.7534 ± 0.4271`
*   **ROC-AUC:** `0.9790 ± 0.0363`
*   **F1 Score:** `0.6250 ± 0.4146`

**Experiment 5: Baseline + Temporal Graph Topology (TGA)**
*   **Features:** 163 (160 baseline + 3 TGA)
*   **PR-AUC:** `0.7534 ± 0.4272`
*   **ROC-AUC:** `0.9788 ± 0.0368`
*   **F1 Score:** `0.6250 ± 0.4146`
*   **Precision:** `0.5833 ± 0.4330`
*   **Recall:** `0.7500 ± 0.4330`

Source: `scripts/legacy/run_exp5.py`; captured output in `outputs/experiments/exp5_report.txt`.
Experiment 5 runs Experiment 1 in the same process as its comparison arm, and
that arm reproduced the Experiment 1 row above exactly (`PR-AUC 0.7534 ± 0.4271`,
`ROC-AUC 0.9790 ± 0.0363`, `F1 0.7500 ± 0.4330`). The full experiment was
executed twice and produced bit-identical fold and mean metrics both times.

### Analysis and Scientific Limitations

*   **Multigraph Signal (Experiment 3):** The perfect separation in Experiment 3 is driven primarily by the `reciprocal_out_count` feature. On this dataset, the four core entities exhibit a minimum `reciprocal_out_count` of `8`, while the maximum among all 3,436 non-core entities is strictly `3`. This represents perfect separation **on this particular dataset**, and must NOT be presented as evidence of generalizable, real-world AML performance. 
*   **Statistical Instability:** The dataset contains only 4 positive entities. Because of this extreme sparsity, fold-level metrics and threshold-based scores (like F1) are extremely unstable. Perfect metric values (`1.0`) do not guarantee generalization, and the system should not be claimed to solve money laundering.
*   **Temporal and Combined Performance (Experiments 2 & 4):** Under this fixed experimental protocol, Experiment 2 demonstrated no meaningful temporal uplift compared to the Baseline. Furthermore, Experiment 4 did not preserve the perfect performance achieved in Experiment 3. This degradation in Experiment 4 is an observed result under the fixed protocol. It is not causally proven exactly how this degradation occurs, nor does it establish that temporal features are inherently useless.
*   **TGA Contribution (Experiment 5) — a negative result:** Adding the three time-respecting directed-cycle features to the 160-feature baseline produced **no measured uplift**. Against Experiment 1, PR-AUC moved by `-0.0000` and ROC-AUC by `-0.0003`, while F1 fell by `-0.1250` and precision by `-0.1667` at unchanged recall. Against Experiment 3 the configuration is worse on every reported metric (`PR-AUC -0.2466`, `ROC-AUC -0.0212`, `F1 -0.3750`). Per the project's research-integrity rule, this is reported rather than adjusted: on this dataset, under this protocol, the TGA features do not contribute predictive value beyond the baseline. It is consistent with the distributional overlap independently recorded in `docs/research.md` Finding 3.1. It does **not** establish that topological information is useless in general — only that these three features, on this dataset, under this protocol, did not help. Experiment 3 remains the shipping configuration; nothing about the frozen 167-feature contract changes.

---

# 29. Research Innovation Layer

The project includes research-derived enhancements.

Each enhancement must follow:

```text
Published Method
       ↓
Identified Limitation
       ↓
Proposed Modification
       ↓
Implementation
       ↓
Experiment
       ↓
Ablation
       ↓
Measured Improvement
```

The implementation must clearly distinguish:

* Existing research
* Our implementation
* Our modification
* Experimental result

---

# 30. Experimental Architecture

Experiments should be separated from production inference.

```text
configs/
    ↓
Experiment runner
    ↓
Training
    ↓
Evaluation
    ↓
outputs/
```

Each experiment should record:

```text
Dataset
Features
Model
Hyperparameters
Seed
Split
Metrics
Runtime
```

---

# 31. Training Pipeline

Conceptual flow:

```text
Dataset
   ↓
Preprocessing
   ↓
Graph generation
   ↓
Train/validation/test split
   ↓
Feature generation
   ↓
Model initialization
   ↓
Training
   ↓
Validation
   ↓
Checkpoint selection
   ↓
Test evaluation
   ↓
Metrics
```

The test set must not influence model selection.

---

# 32. Inference Pipeline

Inference must use the same feature definitions as training. This is enforced in code, not by convention.

## 32.1 Reference Dataset inference (CURRENT — verified)

Implemented in `src/detection/inference.py`.

```text
Reference Dataset (../dkdata.json)
        ↓  load_transactions
Ingestion & schema validation
        ↓  preprocess (IdentitySource.NAME)
Entity resolution
        ↓  build_graph
Authoritative MultiDiGraph
        ↓  compute_baseline (160) + compute_multigraph (7)
Fixed 167-feature encoding
        ↓  build_pyg_data
PyG tensor
        ↓  frozen Experiment 3 checkpoint
Risk score + predicted class per entity
        ↓
Explanation / UI
```

**Contract enforcement.** `predict_from_prepared_data` refuses to run unless all three hold:

1. `metadata["input_dim"] == 167`;
2. `pyg.feature_dim == metadata["input_dim"]`;
3. `list(baseline.features.columns) + list(multi.features.columns)` is **exactly equal, including order**, to `metadata["feature_names"]`.

The model is loaded, never trained: `checkpoint_path=None` raises rather than falling back to on-the-fly training. Classification uses the checkpoint's own `threshold` (0.45).

Pinned by `tests/test_reference_regression.py` to exact values (3,440 entities; `HILUX SERVICES LP` 0.56275397, `LCM ALLIANCE LLP` 0.55135107, `METASTAR INVEST LLP` 0.55129605).

## 32.2 External dataset inference (CURRENT — delivered)

Implemented across `src/datasets/`. The intended pipeline inserts five stages ahead of the shared encoding step:

```text
ingestion            .csv / .xlsx / .xls upload, SHA-256, byte-preserved
      ↓
profiling            structural metadata only — types, null/uniqueness ratios
      ↓
semantic mapping     confidence-scored source column → CanonicalConcept,
                     with retained evidence and human review
      ↓
derivation           canonical columns computed under stated rules
                     (name_norm, composed instant, row id, declared USD/country)
      ↓
canonicalization     projection driven by the reviewed mapping manifest
      ↓
compatibility        DatasetCompatibilityReport → READY | INCOMPATIBLE | FAILED
                     plus readiness → FULL | DEGRADED | BLOCKED
      ↓
fixed feature encoding    160 baseline + 7 multigraph = 167, frozen vocabulary
      ↓
graph / PyG
      ↓
frozen Experiment 3 model
      ↓
workspace predictions     predictions.parquet
      ↓
UI                        via AnalysisContext
```

`READY` is gated on the **inference contract**, not the labelled-training schema. Ground-truth labels (`payer_core`, `beneficiary_core`, `investigation`) are never required, because the 167 features consume none of them; they are preserved when supplied. Entity type and bank country degrade to `entity_type_unknown` / a zeroed country block rather than blocking, and the degradation is reported. See `docs/workspaces.md` → *The inference contract*.

Inference is triggered explicitly by `POST /api/workspaces/<id>/predict` (or the **Run Analysis** action), never implicitly. Compatibility `READY` means *eligible to analyse*; it does not mean predictions exist. Predictions persist to `predictions.parquet` and are read back after restart without re-running the model.

`status` answers *may this run*; `readiness` answers *what will the model actually see if it does*. A dataset supplying no entity type and no resolvable geography satisfies the 167-feature contract exactly and is correctly `READY`, while the model sees `entity_type_unknown` and 151 zeros — so the report carries both and the UI shows both.

Three real external datasets are run through this path in `tests/test_real_datasets.py`, each as far as it gets. One reaches `READY`, at `readiness = DEGRADED`: the Azerbaijani Laundromat (OCCRP) raw workbook, which is not independent evidence of model validity because the frozen checkpoint was trained on that data. A dataset whose amounts span multiple currencies still cannot reach `READY`: no FX conversion exists, and `EXTERNAL_FX_REQUIRED` is honestly reported as `INCOMPATIBLE` rather than guessing rates. The other two — SAML-D as published and IBM AML — are `BLOCKED` on exactly that limitation; see `docs/workspaces.md` → *Verified external dataset results*.

## 32.3 Shared invariant

Both paths converge on the same encoding and the same frozen checkpoint. Whatever the data source, the model sees exactly 167 features in exactly one order or it does not run at all.

---

# 33. Data and Model Versioning

The project should distinguish:

```text
Dataset version
Feature version
Model version
Experiment version
```

Changing the feature definition should invalidate direct comparisons with incompatible previous experiments unless explicitly documented.

---

# 34. Software Architecture

The expected source structure is:

```text
src/
│
├── data/
│   ├── ingestion.py
│   ├── preprocessing.py
│   └── validation.py
│
├── graph/
│   ├── schema.py
│   ├── builder.py
│   ├── features.py
│   └── cycles.py
│
├── temporal/
│   ├── features.py
│   └── modelling.py
│
├── tda/
│   ├── filtration.py
│   ├── persistence.py
│   └── features.py
│
├── models/
│   ├── baselines/
│   ├── gnn/
│   └── training/
│
├── detection/
│   ├── risk_scoring.py
│   ├── subgraph_detection.py
│   └── explainability.py
│
├── evaluation/
│   ├── metrics.py
│   ├── experiments.py
│   └── ablation.py
│
└── utils/
```

This structure is not mandatory.

It should evolve with the implementation.

## 34.1 Actual Source Structure (CURRENT)

The structure above is the original expectation. The implementation evolved differently: feature engineering consolidated into a single `features/` package rather than separate `temporal/` and `tda/` packages, and two packages not anticipated above — `datasets/` and `ui/` — were added. `evaluation/` and `validation.py` were never created. Experiment orchestration now lives in `src/experiments/` (§45), reached through the root-level `run_experiment.py`, `audit_datasets.py` and `compare_experiments.py` CLIs; the older `run_exp*.py` and `audit_*.py` scripts remain as the historical Experiment 1–5 record.

Directories are not created merely because documentation mentions them.

```text
src/
│
├── data/
│   ├── ingestion.py          Canonical schema, REQUIRED_COLUMNS, loading
│   └── preprocessing.py      Validation, entity resolution, IdentitySource
│
├── datasets/                 External dataset workspaces  (see docs/workspaces.md)
│   ├── manifest.py           WorkspaceManifest/Status, WorkspaceConfig + policies
│   ├── workspace.py          Atomic persistence, crash recovery
│   ├── workspace_manager.py  UUID isolation, ingestion, profiling, mapping
│   ├── parsers.py            Allowlisted, bounded file parsing
│   ├── profiler.py           Structural profiling only
│   ├── mapper.py             CanonicalConcept, confidence scoring, evidence
│   ├── derivation.py         What can be computed, and under what rule
│   ├── geography.py          Country resolution into the frozen vocabulary
│   ├── policy.py             Declared configuration vs. the actual upload
│   ├── fx.py                 FX seam — declared, deliberately unimplemented
│   ├── trust.py              Per-run prediction confidence assessment
│   ├── canonicalizer.py      Canonicalization + compatibility validation
│   └── inference.py          Workspace-scoped prediction
│
├── features/
│   ├── baseline.py           160 features + frozen vocabulary (§12.1.1)
│   ├── multigraph.py         7 directed-multigraph features
│   ├── temporal.py           Temporal features (research; not in the checkpoint)
│   └── tda.py                Topological features (research; not in the checkpoint)
│
├── graph/
│   └── builder.py            Authoritative MultiDiGraph construction
│
├── models/
│   ├── baseline_gnn.py       BaselineGraphSAGE
│   ├── dataset.py            PyG tensor construction, CV folds
│   └── trainer.py            Training loop, threshold selection
│
├── detection/
│   ├── contract.py           Model-independent risk contract (RiskAssessment)  (§47)
│   ├── engines.py            Risk engines: CandidateRiskEngine, LegacyGraphSAGEEngine
│   ├── inference.py          Frozen-checkpoint inference + contract enforcement
│   ├── cycles.py             Layer 10 cycle evidence (post-prediction)
│   └── explain.py            Explanation generation
│
├── experiments/              Layer 6 — model-agnostic experiments  (§45)
│   ├── spec.py               Serializable ExperimentSpec
│   ├── dataset.py            ExperimentDataset: entities, labels, edges, graph
│   ├── reference.py          Reference Dataset loader (via Layers 1–3)
│   ├── sources.py            External benchmark loaders (SAML-D, IBM AML, …)
│   ├── audit.py              Dataset scientific audit and usability verdict
│   ├── features.py           Versioned feature registry + legacy adapters
│   ├── aml_features.py       Scalable AML families (txn/temporal/flow/network/cycle/context)
│   ├── splits.py             Named split strategies, each declaring its regime
│   ├── calendar.py           Calendar-forward evaluation: (entity, origin) samples, window labels
│   ├── candidate.py          Candidate contract + registry
│   ├── baselines.py          Heuristic floor, logistic regression, boosting
│   ├── legacy_graphsage.py   Experiment 3 architecture behind the same contract
│   ├── metrics.py            Rare-positive metrics; undefined stays None
│   ├── thresholds.py         Validation-only threshold selection
│   ├── calibration.py        Validation-only score-to-probability mapping
│   ├── leakage.py            Automatic audits recorded in every artifact
│   ├── seeding.py            Central seeds + honest determinism report
│   ├── runner.py             Orchestration
│   └── artifacts.py          Result containers and JSON persistence
│
└── ui/                       Presentation layer  (see docs/UI_ARCHITECTURE.md)
    ├── app.py  adapter.py  resolver.py  context.py  workspaces.py
    ├── config.py  theme.py  help_content.py
    ├── components/           app.js + HTML fragments
    └── pages/index.html
```

---

# 35. Module Dependency Direction

Prefer the following dependency direction:

```text
data
 ↓
graph
 ↓
features
 ↓
models
 ↓
detection
 ↓
application
```

Evaluation should be able to consume outputs from the model without becoming tightly coupled to model internals.

Visualization should consume structured outputs rather than directly manipulating model internals.

Avoid circular module dependencies.

---

# 36. Interface Boundaries

Important interfaces should be explicit.

Examples:

```text
DataLoader
GraphBuilder
FeatureExtractor
TemporalFeatureExtractor
TDAFeatureExtractor
GraphModel
Trainer
Evaluator
RiskScorer
Explainer
SubgraphDetector
```

The exact class/interface design may change during implementation.

---

# 37. Configuration Architecture

Configuration should control:

```text
Dataset
Graph schema
Feature selection
Model architecture
Hyperparameters
Training parameters
Evaluation settings
Random seed
Output locations
```

Prefer configuration over modifying source code for every experiment.

---

# 38. Visualization Architecture

The final application may contain:

```text
Dashboard
│
├── Risk Overview
├── Account Search
├── Transaction Search
├── Suspicious Networks
├── Circular Transaction View
├── Timeline
├── Risk Explanation
└── Model Metrics
```

The visualization layer should consume structured backend outputs.

---

# 39. Security Architecture

Sensitive information must not appear in:

* Logs
* Public repositories
* Screenshots
* Demo output
* Hard-coded configuration
* Source code

Secrets must be supplied through environment variables or secure configuration.

---

# 40. Computational Constraints

The initial development environment is expected to be CPU-first.

The system should avoid unnecessary assumptions about GPU availability.

Potentially expensive components include:

* Large graph construction
* Dense graph operations
* GNN training
* TDA computation
* Cycle enumeration
* Large subgraph extraction

Where necessary, use:

* Sparse structures
* Sampling
* Batching
* Time-windowing
* Efficient graph libraries

Optimizations should be evidence-driven.

---

# 41. Architecture Evolution

This document is a **living specification**.

The architecture is allowed to change.

When a major change is proposed:

```text
Current architecture
        ↓
Proposed architecture
        ↓
Impact analysis
        ↓
Trade-off analysis
        ↓
Decision
        ↓
Migration
        ↓
Testing
        ↓
Documentation update
```

The system should prefer incremental migration over unnecessary full rewrites.

---

# 42. Architectural Decision Boundaries

The following decisions are currently **not permanently finalized**:

* Dataset
* Exact graph schema
* Exact node types
* Exact edge types
* Homogeneous vs heterogeneous representation
* Exact temporal architecture
* Exact GNN architecture
* Exact TDA method
* Feature-fusion mechanism
* Risk-score formula
* Explainability mechanism
* Deployment stack

These decisions must be validated through research and experimentation.

---

# 43. Current Recommended Development Order

The initial implementation should proceed approximately in this order:

```text
1. Dataset selection
        ↓
2. Data schema
        ↓
3. Data ingestion
        ↓
4. Data validation / preprocessing
        ↓
5. Entity resolution
        ↓
6. Graph schema
        ↓
7. Graph construction
        ↓
8. Graph validation
        ↓
9. Baseline features
        ↓
10. Traditional ML baseline
        ↓
11. Temporal features
        ↓
12. GNN baseline
        ↓
13. Cycle/subgraph detection
        ↓
14. TDA pipeline
        ↓
15. Proposed research enhancements
        ↓
16. Feature/model fusion
        ↓
17. Explainability
        ↓
18. Ablation experiments
        ↓
19. Final evaluation
        ↓
20. Dashboard / demonstration
```

This order may change if experimental findings justify it.

---

# 44. Definition of an Architecturally Complete System

The system is architecturally mature when:

* Data ingestion is reproducible.
* The graph schema is explicitly defined.
* Graph construction is validated.
* Features are reproducible.
* Temporal leakage is controlled.
* Baselines exist.
* The GNN pipeline is reproducible.
* TDA integration is validated.
* Research enhancements are experimentally evaluated.
* Risk scoring is defined.
* Suspicious subgraphs can be extracted.
* Predictions have explanations.
* Ablation experiments exist.
* Results are reproducible.
* Documentation matches implementation.
* The complete pipeline can run end-to-end.

---

# 45. Layer 6 — Model-Agnostic Experiment Framework (CURRENT)

`src/experiments/` is the layer in which model candidates are trained, evaluated
and compared. It sits above Layers 1–5 and depends on them only through
adapters; Layers 1–5 do not import it.

## 45.1 Why it exists

Experiments 1–5 (§28.3) shared one hard-coded protocol: a single transductive
4-fold split that asserted exactly four positives, whole-graph feature
computation, an F1-maximising threshold, and metrics that returned `0.0` where
they were undefined. Under that protocol every configuration produced either
`PR-AUC 0.7534` or `1.0000`, because with one positive per test fold average
precision can only take the value `1/rank`. The protocol could not distinguish
between models, and it could not express any question other than the one it was
written for.

Layer 6 replaces the protocol, not the models. Experiment 3 is reachable through
it as one candidate among several.

## 45.2 Contract

```text
ExperimentDataset
   -> SplitPlan            (named strategy, declared regime)
   -> FeatureMatrix        (versioned families, declared scope)
   -> Candidate.fit        (train partition only)
   -> validation scoring
   -> calibration          (fitted on validation)
   -> threshold selection  (chosen on validation)
   -> test scoring         (once, everything already frozen)
   -> ExperimentResult     (JSON artifact)
```

A candidate implements four things — `fit`, `predict_scores`, `config` and
`emits_probabilities`. Logistic regression, gradient boosting, the legacy
GraphSAGE, and a future edge-aware or temporal or heterogeneous GNN all satisfy
the same interface. Adding a model family touches one file.

## 45.3 Modules

| module | responsibility |
| --- | --- |
| `spec.py` | Serializable `ExperimentSpec`. The only home for experiment constants. |
| `dataset.py` | `ExperimentDataset`: entities, labels, edges, graph. Labels never travel with the structure. |
| `reference.py` | Builds an `ExperimentDataset` from the Reference Dataset via Layers 1–3. |
| `sources.py` | Loaders for external benchmarks (SAML-D, IBM AML, Kaggle fraud). One generic builder plus a ~20-line wrapper per benchmark. |
| `audit.py` | Dataset scientific audit: data facts, label quality, structural triviality, split feasibility, usability verdict. |
| `features.py` | Versioned feature registry. Legacy adapters plus `aml_core_v1`. |
| `aml_features.py` | Six scalable AML families, separately registered so an ablation can turn one off. |
| `splits.py` | Five split strategies, each declaring its evaluation regime. |
| `calendar.py` | The `calendar_forward` regime: fixed calendar origins, labels from a post-origin window, features from before each origin. Not an entity partition, so it has its own loop; `runner.run_experiment` routes to it. |
| `candidate.py` | The `Candidate` contract, `FitContext`, and the registry. |
| `baselines.py` | Structural heuristic, logistic regression, gradient boosting. |
| `legacy_graphsage.py` | The Experiment 3 architecture behind the same contract. |
| `metrics.py` | Rare-positive metrics. Undefined stays `None`. |
| `thresholds.py` | Validation-only threshold selection. |
| `calibration.py` | Validation-only score-to-probability mapping, with a small-data guard. |
| `leakage.py` | Automatic audits, recorded in every artifact. |
| `seeding.py` | Central seeds and an honest determinism report. |
| `runner.py` | Orchestration. |
| `artifacts.py` | Result containers and JSON persistence. |

Entry points:

| CLI | purpose |
| --- | --- |
| `run_experiment.py` | Run one or more specs from `configs/experiments/`. The spec's `dataset.name` selects the loader; a dataset is loaded once per invocation however many specs name it. |
| `audit_datasets.py` | Audit every reachable dataset. Writes `outputs/audits/*.json` and `docs/dataset_audit_table.md`. |
| `compare_experiments.py` | Group artifacts by (dataset version, regime, feature scope) and render a comparison table. It never names a winner. |

## 45.3.1 Feature families

`legacy_exp3_v1`, `baseline_v1`, `multigraph_v1`, `temporal_v1` and `tga_v1` are
the historical representation, retained for comparison and unchanged.
`aml_core_v1` is the first purpose-built AML block and is pinned by the
artifacts that recorded it. The six families below are the current AML
representation; they are separately registered because Part IX's question —
*which information source contributes?* — can only be answered if the sources
can be switched off independently.

| family | declared family | what it summarises |
| --- | --- | --- |
| `aml_txn_v1` | transaction | directional counts, sums, means, dispersion, largest-transaction share, round-amount share |
| `aml_temporal_v1` | temporal | active days, span, velocity, inter-arrival mean/dispersion, burstiness, daily concentration, night and weekend share, dormant-to-active shift, recency, tenure |
| `aml_flow_v1` | flow | fan-in, fan-out, gather-scatter signature, imbalance, pass-through ratio, inflow-to-outflow turnaround latency at two bands |
| `aml_network_v1` | network | counterparty count (the undirected degree) and diversity, repeated-pair concentration, reciprocity, component size and density, one hop of neighbour aggregation |
| `aml_cycle_v1` | structure | two-paths through the entity, directed triangles, closed length-4 walks, temporally ordered layering events with amount match |
| `aml_context_v1` | context | payment-channel, currency and jurisdiction mix, populated only from columns the dataset carries |

68 columns in total, and **no short column name appears in two families**. A
column present twice inflates its own importance in every model that reads the
matrix, so transaction count lives only in `aml_txn_v1`, reciprocity only in
`aml_network_v1`, and the flow family carries `flow_imbalance` rather than the
source and sink shares that are affine transforms of it.

Four properties make them usable at benchmark scale:

* **No `networkx` object.** Every family reads the edge table and a
  deduplicated SciPy adjacency. A `MultiDiGraph` over 9.5 million edges costs
  tens of gigabytes; the deduplicated graph of the same data has 887 thousand
  edges. The runner builds a `networkx` graph only when a requested family
  declares `requires_graph`, which now means only the legacy families.
* **One shared intermediate per request.** `FeatureRequest.cache` holds the
  flattened event view and the sparse adjacency, so six families cost one
  flattening rather than six. The cache is keyed by the request, so a fold with
  a different time window cannot inherit a summary of transactions it may not
  see.
* **Vectorised temporal joins.** "The most recent inflow at *this* entity at or
  before this outflow" is answered for all 855 thousand entities with one
  `searchsorted` over a globally monotone `entity × time` key, not a per-entity
  loop. The key is built in `int64`, so second-level resolution cannot be lost
  silently on a larger dataset.
* **Two-pass dispersion.** Standard deviations sum squared deviations from an
  already-computed mean rather than using `E[X²] - E[X]²`. At SAML-D's amount
  scale the one-pass identity subtracts two numbers near 10¹⁷ and loses most of
  its significant digits — precisely for the entities whose amounts barely vary,
  which is the structuring signature the feature exists to catch.

Temporal semantics are documented per column in `src/experiments/aml_features.py`.
Only `recency_days` and `tenure_days` depend on a reference "now"; every other
column is a summary of the transactions present in the request and therefore
honours a training window exactly.

## 45.4 Evaluation regimes

A split strategy declares three properties, and the artifact reports the regime
they imply. This is the layer's central discipline: two runs under different
regimes are answers to different questions and must never share a comparison
table.

| strategy | entity-disjoint | time-aware | graph-leakage controlled |
| --- | --- | --- | --- |
| `stratified_kfold` | yes | no | no |
| `time_forward_holdout` | yes | yes | no |
| `time_forward_rolling` | yes | yes | no |
| `component_grouped_kfold` | yes | no | yes |
| `legacy_exp3_transductive` | yes | no | no |
| `calendar_forward` (`calendar.py`) | **no** — rows are (entity, origin) samples, disjoint in *time* | yes | no |

### A split strategy and a feature can be the same quantity

`time_forward_holdout` assigns entities to partitions by their own activity
time. `aml_temporal_v1::recency_days` at `global` scope is measured from the end
of the whole file, so it is a monotone transform of that assignment variable.
Measured on IBM HI-Small: train range `[8.900, 17.679]` days, test range
`[0.000, 8.410]` — **disjoint**. The feature becomes a partition indicator, the
model extrapolates outside its training support, and the run is scientifically
void as a time-forward result.

The framework does not forbid the combination, because it is exactly how the
split effect was separated from the feature-scope effect in
`docs/temporal_collapse.md`. It declares it: the artifact records
`features.scope = "global"` beside `split.time_aware = true`, and
`tests/test_experiment_framework.py::TestTimeForwardScopeInteraction` pins that
both statements are present. **A time-aware split at `global` feature scope is a
diagnostic, never a reportable result.**

The more general rule: a split that assigns entities by a behavioural quantity
will differ between partitions in every feature correlated with that quantity.
On IBM HI-Small that produced a train median of 3 transactions against a test
median of 37, and 48 of 54 features past PSI 0.25 — a distribution shift
manufactured by the split rather than observed in the data.

`FeatureSpec.scope` is recorded separately. `"global"` computes features over
the whole population — the regime Experiments 1–5 used, legitimate for
transductive research framing and declared as such. `"train_window"` restricts
feature computation to transactions inside the fold's training window, which is
what a deployment actually faces. `"origin_window"` gives every scored snapshot
the transactions before *its own* origin; only `calendar_forward` defines
origins, it accepts no other scope, and every other split refuses this one.

### Calendar-forward: hold out time, keep the population

`time_forward_holdout` holds out *entities*, chosen by when they were last
active. `calendar_forward` holds out *time*. The unit of prediction is the
pair (entity, origin):

```text
origin T            a fixed calendar instant, the same for every entity
features            transactions dated < T (optionally >= T - lookback); recency from T
label               payer or payee of >= 1 publisher-flagged transaction dated in [T, T + H)
eligible            >= 1 dated transaction in the feature window; others excluded and counted
roles               train origin -> fit;  validation origin -> calibrate, threshold;
                    test origin -> scored once
ordering            T_train + H <= T_val  and  T_val + H <= T_test   (enforced)
```

The same account recurs at all three origins with a different snapshot and a
different label. Whole-file entity labels are never read: they include the
future. The window label comes from `ExperimentDataset.transaction_labels`,
which the loader keeps *beside* the edge table so that no feature family can
see it. Candidates must declare `calendar_compatible` (they read only
`ctx.features`, `ctx.y_train`, `ctx.y_val`); the legacy GraphSAGE does not, and
is refused rather than allowed to read whole-file labels. Families that need a
`networkx` graph or do not honour a time window are refused too.

The audit checks the rows actually handed to each snapshot's feature families
(`calendar_information_boundary`), the label-window ordering
(`calendar_label_ordering`) and records the expected entity overlap. Each
artifact also reports test metrics on the entities with no flagged transaction
before the test origin - the first-involvement subgroup.

Snapshot feature blocks are cached per (window, family) for the life of one
dataset object, so a batch of calendar configs computes each window once.

The IBM design, the choice of origins and horizon, and the results are in
`docs/calendar_forward_benchmark.md`.

## 45.5 Metric policy

Primary: average precision, AP lift over prevalence, precision@K, recall@K,
alert volume, alert rate, and false positives per true positive. Secondary:
ROC-AUC and threshold-dependent precision / recall / F1 / confusion matrix.
Calibration metrics (Brier, expected calibration error, reliability bins) are
computed only when the scores are on a probability scale.

**An undefined metric is `null`, with a stated reason.** A fold whose test
partition holds no positives contributes nothing to the mean rather than
dragging it toward zero, and the artifact records `n_defined` alongside every
aggregate.

## 45.6 Relationship to the frozen checkpoint

Layer 6 never writes to `artifacts/models/experiment3_graphsage.pt` and never
restates a §28.3 result. `legacy_graphsage` trains a *new* model from the same
architecture; its numbers are a benchmark of the architecture under the new
protocol, not a reproduction of the historical run. The generic legacy split is
structurally equivalent to `src/models/dataset.build_cv_folds` but not
bit-identical to it, so its fold assignment and its numbers differ.

`src/models/`, `src/features/`, `src/detection/inference.py` and the frozen
contract of §41.1 are unchanged.

## 45.7 What the Reference Dataset can establish

Measured 2026-09-06 through this layer:

* The four labelled entities occupy ranks 1–4 of 3,440 on raw transaction count,
  fan-in, fan-out, incoming volume, counterparty count and degree centrality.
  Largest negative: 554 transactions. Smallest positive: 1,777.
* Ranking by that single column yields average precision `1.0000`.
* The unsupervised `structural_heuristic`, which reads no labels, matches
  logistic regression and gradient boosting on test average precision under the
  4-fold protocol.
* All four positives fall in the earliest half of first-activity time, so every
  forward-looking test partition contains zero positives.

The dataset therefore functions as a **regression signal**, confirming the
pipeline still behaves, and cannot function as a **model-selection benchmark**.
Separating candidates requires a dataset with enough positives to do so. This is
a property of the data, not a defect in any model.

## 45.8 Dataset suitability is a stage, not an opinion

`src/experiments/audit.py` classifies a dataset before it is used for a
comparison, and the classification is an artifact (`outputs/audits/*.json`), not
a paragraph. Its three measured blocks are `data` (sizes, time coverage,
missingness, duplicate structure, field inventory), `labels` (granularity,
cross-validation feasibility, temporal feasibility, alias columns) and
`structural_triviality`.

The third block is the one that matters. It scores each of fifteen simple
quantities as a ranking of the population, and then fits an **untuned**
`HistGradientBoostingClassifier` on all fifteen together, on a stratified half,
scored on the other half. A label can be trivial as a *combination* while no
single column looks suspicious, and that is not a hypothetical: on SAML-D the
best single quantity reaches average precision 0.1831 and the combination
reaches **0.9971**.

Four verdicts — `MODEL_SELECTION_READY`, `STRUCTURAL_VALIDATION_ONLY`,
`REGRESSION_ONLY`, `BLOCKED` — each with a stated reason. The label's *domain*
is declared by the loader rather than measured, because whether a publisher's
positive class means "money laundering" is a fact about that publisher's
definition and no statistic on the file can recover it.

Findings and their interpretation: `docs/datasets.md`. Current state, measured
2026-09-06: the model-selection benchmark is **`ibm_aml_hi_small`** (IBM
AMLworld HI-Small ledger, structural probe AP 0.2098, no class-free bands).
It is acquired with `acquire_dataset.py` and is not committed; provenance is
`data/manifests/ibm_aml_hi_small.json`. Every other reachable dataset remains
regression-only, structural-validation-only or blocked.

## 45.9 The prediction contract (DECLARED — not yet the shipped interface)

> **Delivered 2026-09-11** as `src/detection/contract.py` (`RiskAssessment`) and
> `src/detection/engines.py`; see §47. Every workspace run now persists its
> assessment metadata (`assessment.json`). **Still open:** the UI adapter reads
> the legacy column names described below. The text below is the record of the
> seam as declared.

The application must not need to know which model family produced a score.
The contract every future backend is expected to satisfy:

```text
entity_id            stable identifier within the scored population
risk_score           float; ordering is meaningful, magnitude may not be
prediction           the decision at the run's frozen threshold
model_version        which model, at which version, produced the score
feature_version      which feature families, at which versions, fed it
evidence             per-entity observed quantities behind the score
confidence           how far the run is from the conditions the model was fitted under
ground_truth         the label where one exists; absent is absent, never 0
```

**Current state, stated honestly.** `src/detection/inference.py` returns a frame
carrying `entity_id`, `risk_score`, `predicted_class` and eleven evidence
columns. Those evidence columns are the legacy 167-feature blocks by name, so
the frame is *shaped* model-agnostically but *populated* model-specifically.
`model_version` and `feature_version` are carried by the workspace manifest and
the trust assessment (`src/datasets/trust.py`) rather than by the prediction
frame itself, and `confidence` exists only as that separate trust artifact.

This section declares the seam. It does not describe delivered behaviour, and
the frozen inference contract of §41.1 is unchanged by it. Closing the gap is a
change to the inference path and to the UI adapter, and it requires its own
decision record.

## 45.10 Topological data analysis — deferred, with a condition

Persistent homology appears in the project's problem statement. It is **not**
implemented, and this is deliberate.

The precondition is that topology must be shown to add information beyond graph
statistics, cycles, communities, temporal patterns and supervised learning. That
question cannot currently be asked: it needs a dataset on which those five
already leave measurable headroom, and §45.8 records that no available dataset
does. On SAML-D a linear model on tabular AML features reaches average precision
0.9983, so there is nothing for topology to add there that would mean anything.

When a suitable dataset exists, TDA should be applied **selectively to
suspicious subgraphs** rather than over the whole transaction universe: the
`tga_v1` family already computes time-respecting cycle counts on 1-hop ego
networks, which is the natural place to attach a persistence computation.
Recorded as a future experiment, not as current work.

---

# 47. Target System Architecture (CURRENT DECISION)

Decided 2026-09-11. Reasoning, evidence, feature evaluation, industrial gaps and
roadmap: `docs/system_architecture_decision.md`. Decision record:
`docs/decisions.md`.

**The system is an ML + investigation-intelligence hybrid.** ML ranks entities
by suspicious behaviour inside a declared analysis window (retrospective
detection); graph and temporal analytics explain and extend each alert.

```text
DATA INGESTION → SCHEMA / SEMANTIC MAPPING → DATA QUALITY + TRUST     src/data, src/datasets
      ↓
BEHAVIOUR ENGINE        aml_* families over the window (src/experiments/aml_features.py)
      ↓                 legacy 167-vector for the frozen checkpoint only
ML RISK ENGINE          src/detection/engines.py
      ↓                   CandidateRiskEngine | LegacyGraphSAGEEngine
RISK ASSESSMENT         src/detection/contract.py - the only interface downstream
      ↓
INVESTIGATION GRAPH → CYCLE / FLOW / TEMPORAL ANALYSIS → EXPLANATION   src/detection/cycles.py,
      ↓                                                                contract.entity_evidence
ANALYST UI              src/ui (migration to the contract pending)
```

## 47.1 Boundaries

1. **Downstream of the risk engine, only the contract.** A `RiskAssessment`
   carries entity, score, rank, decision (with threshold basis), model identity
   and version, feature version, task, score kind, evaluation reference,
   analysis window and evidence. Nothing reads a model-specific column.
2. **Investigation analytics never change a score.** They consume the
   assessment and the graph; `entity_evidence` attaches cycle and counterparty
   references to an already-scored entity.
3. **Every score carries its task.** `retrospective_detection` or
   `forward_forecasting`; a retrospective number cannot be displayed as a
   forecast.
4. **Engines share an output, not an input.** `CandidateRiskEngine` scores the
   `aml_*` families from a `TransactionWindow` (only rows inside the window;
   recency measured from its end). `LegacyGraphSAGEEngine` wraps the frozen
   canonical path (§41.1) and restates its output without changing a number.
   One input representation for both is follow-on work; the frozen checkpoint
   cannot consume the `aml_*` representation.

## 47.2 State

| component | state |
| --- | --- |
| contract, engines, `Candidate.score_features` | delivered, tested (`tests/test_risk_contract.py`) |
| per-run `assessment.json` on the workspace route | delivered |
| UI adapter reading the contract | **not yet** - reads legacy columns |
| persisted, versioned `CandidateRiskEngine` (model registry) | **not yet** - roadmap R4 |
| cross-period retrospective benchmark | **not yet** - roadmap R1, decides the production model's claim |
| path / component / community investigation analytics | **not yet** - roadmap R2, R8 |
| GNN / TDA | **not built**: NO-GO / gated (`docs/model_selection.md` §I.8) |

---

# 46. Final Architectural Principle

The system should follow:

```text
RAW DATA
   ↓
TRUSTWORTHY REPRESENTATION
   ↓
GRAPH + TEMPORAL STRUCTURE
   ↓
STRUCTURAL / TOPOLOGICAL FEATURES
   ↓
GRAPH REPRESENTATION LEARNING
   ↓
MULTI-SIGNAL RISK ANALYSIS
   ↓
SUSPICIOUS SUBGRAPH DETECTION
   ↓
EXPLAINABLE RESULT
```

The purpose of the architecture is not to maximize the number of AI techniques used.

The purpose is to detect complex financial-crime patterns more effectively than simpler approaches, while producing evidence that can be understood, evaluated and demonstrated.

Every architectural component should therefore answer three questions:

1. **What problem does this component solve?**
2. **Why is this component necessary compared with simpler alternatives?**
3. **Can its contribution be measured experimentally?**

If a component cannot satisfactorily answer these questions, its inclusion should be reconsidered.
