# System Architecture Decision: What the AML System Optimises For

Date: 2026-09-11. Status: **Accepted** (recorded in `docs/decisions.md`; the
project owner may supersede it).

Every quantitative statement below cites a measured project artifact. Statements
marked **[recommendation]** are architectural judgement, not measurement. IBM
AMLworld and SAML-D are **synthetic**; nothing here is evidence about real-world
detection rates.

---

## 0. The decision in one paragraph

The system is an **ML + investigation-intelligence hybrid** (option 3). Machine
learning does one job: **rank entities by suspicious behaviour observed inside
a declared analysis window**, under an alert budget - retrospective detection,
the task the project's benchmark actually supports (AP 0.3827, 9.8x the
label-free floor). Graph and temporal analytics do a different job: **explain
and extend each alert for an investigator** - cycles, flows, counterparties,
paths, evolution over time - as post-prediction evidence that never changes a
score. Forecasting future involvement is **not** a product claim (1.4-2.0x lift
on first involvement), and no GNN is built (graph increment -0.0024 ± 0.0027
under forecasting, +0.0093 only when features see the label period). The seam
between the two halves is the model-independent risk contract
(`src/detection/contract.py`), delivered in this phase.

---

> **Update 2026-09-12 (R1).** The central missing piece of §3 - whether
> retrospective detection transfers to a *later* period - has been measured:
> **LIMITED TEMPORAL TRANSFER**. Between comparable 2-day windows a frozen model
> keeps 97-98 % of its same-period average precision (0.3145 and 0.3425, 17-20x
> the label-free floor); across a window whose composition differs it keeps 11 %.
> Two production rules follow: re-derive the alert threshold per window, and gate
> on temporal-feature drift. Full record: `docs/r1_detection_transfer.md`.

## 1. Audit of the current system

### 1.1 What exists

| layer | module | state |
| --- | --- | --- |
| external dataset admission | `src/datasets/` (profiler, mapper, canonicalizer, policy, trust, workspaces) | delivered, tested; IBM and SAML-D are `BLOCKED` there on missing USD amounts (no FX) |
| ingestion, preprocessing, entity resolution | `src/data/` | delivered (Reference Dataset, workspaces) |
| graph construction | `src/graph/builder.py` | delivered: directed multigraph, parallel edges kept |
| legacy features | `src/features/` (167-vector) | frozen for the checkpoint |
| AML feature families | `src/experiments/aml_features.py` (6 families, 68 columns) | delivered, benchmark-scale, leakage-audited |
| experiment framework | `src/experiments/` (splits, calendar, leakage, metrics, thresholds, calibration, runner, artifacts, audit) | delivered; three evaluation regimes |
| production inference | `src/detection/inference.py` | frozen Experiment 3 GraphSAGE; Reference Dataset + workspaces |
| risk contract + engines | `src/detection/contract.py`, `src/detection/engines.py` | **new in this phase** |
| cycle evidence | `src/detection/cycles.py` | delivered as post-hoc evidence, served at `/api/cycles`; no UI; never evaluated as a detection capability |
| explanation | adapter flags (`src/ui/adapter.py`), `src/detection/explain.py` | rule-based flags on legacy columns |
| analyst UI | `src/ui/` | delivered for the frozen model; reads legacy column names |

### 1.2 Proven capabilities (measured)

* **Data admission is auditable.** Profiling, mapping with retained evidence,
  compatibility gating and per-run trust assessment are tested end to end
  (`docs/workspaces.md`).
* **Datasets are classified before use.** IBM HI-Small is
  `MODEL_SELECTION_READY` (probe AP 0.2098, no class-free bands); SAML-D is
  structurally trivial (probe AP 0.9971); the Reference Dataset has 4 positives
  (`docs/datasets.md`).
* **Retrospective detection works on IBM.** Boosting on the six families:
  AP 0.3827 ± 0.0057, 31x prevalence, ROC 0.922, against a label-free floor of
  0.0390 (`docs/model_selection.md` §I.1).
* **Forward forecasting is weak on IBM.** Calendar-forward: boosting AP
  0.0133 ± 0.0035 at 0.267 % prevalence; first involvement 1.4-2.0x lift; Brier
  skill +0.001 to +0.002 (`docs/calendar_forward_benchmark.md`).
* **Graph features carry little beyond behaviour.** +0.0093 AP (standard),
  −0.0010 (survivorship), −0.0024 ± 0.0027 (calendar-forward).
* **The evaluation stack does not manufacture scores.** Leakage audits in every
  artifact; the Kaggle control lands near prevalence (AP 0.42 at 0.33).
* **Cycles can be enumerated and described** with per-leg transaction
  evidence, temporal precision and amount completeness (`src/detection/cycles.py`).

### 1.3 Historical assumptions (believed, not established)

* **GraphSAGE + 167 features as the production model.** Its only evaluation is
  the Reference Dataset, where transaction count alone gives AP 1.0; retrained
  under the new protocol it scored below the label-free floor there.
* **"Directed-multigraph structure carries the signal."** True on the Reference
  Dataset (one feature, 4 positives) and on SAML-D (labels written from the
  graph). On the one dataset that can rank models it is marginal.
* **Node classification of `core` as *the* task.** A property of the Reference
  Dataset's labels, not of the AML problem.
* **Model probability = risk.** `docs/architecture.md` §21 already warns
  against it; the contract now enforces the distinction (`score_kind`).
* **The survivorship time-forward split as "the operational number"** - a
  population comparison, not a forecast (`docs/temporal_collapse.md`).

### 1.4 Unvalidated ambitions

GNN superiority (any kind), temporal and heterogeneous GNNs, TDA, forecasting as
a product, decision-level fusion (`docs/architecture.md` §20), calibrated risk
probabilities, suspicious-subgraph detection *quality*, community/ring
discovery, transaction-level scoring, and every real-world performance claim.

---

## 2. The product target

| criterion | A retrospective detection | B forward forecasting | C investigation intelligence | D hybrid (A + C) |
| --- | --- | --- | --- | --- |
| empirical evidence | **strong**: 0.3827 AP, 9.8x floor | **weak**: 1.4-2.0x on first involvement, near-zero Brier skill | cycles implemented; **no quality measurement** | A's evidence + C's capability |
| original objective (mule accounts, circular flows, rings, explainability) | detects mule-like entities | not stated | covers cycles, flows, rings, explanations | **covers both halves** |
| industrial AML workflow | alert generation over a review period - the standard transaction-monitoring shape **[recommendation]** | used for early warning, rarely the primary alert source **[recommendation]** | case investigation after an alert | alert → investigate, the actual workflow |
| available data | IBM supports it (entity labels over a period) | IBM supports measuring it, and it measures weak | IBM's graph supports it; typology ground truth exists in `HI-Small_Patterns.txt` (not yet acquired) | yes |
| SIH demonstration value | a ranked list | a weak number | visual, explainable evidence | ranked list **and** why |
| scientific defensibility | defensible if labelled retrospective | not defensible as a product claim | defensible if evidence is not presented as a verdict | defensible with the claim/demonstration split in §11 |

**Chosen target: D.** A alone leaves every alert unexplained and ignores the
circular-flow half of the objective. C alone has no measured ranking behind it.
B is kept as a research track (it is the harder, more valuable question) but
its current evidence cannot carry a product claim.

---

## 3. The role of ML

**Optimise:** entity risk ranking over a declared analysis window, evaluated
with average precision, precision@K / recall@K, alert volume and false positives
per true positive at an alert budget.

**Do not optimise (now):**

| candidate role | decision | evidence |
| --- | --- | --- |
| entity risk ranking (retrospective) | **yes - the ML objective** | 0.3827 AP standard |
| alert prioritisation at a budget | **yes, as the operating point of the above** | threshold chosen on validation at a 1 % alert budget in every run |
| future forecasting | research track only | 1.4-2.0x lift on first involvement |
| calibrated probability | no; scores are ranks with a declared `score_kind` | Brier skill +0.001 to +0.002 under forecasting |
| transaction risk ranking | not yet - unmeasured, and IBM publishes transaction labels (roadmap R3) | none |
| anomaly detection | not as the detector; as a second label-free floor (R7) | the rank-sum floor reaches only 3.2x on IBM |
| investigation prioritisation (which alert first) | follows from the ranking; no separate model | - |

**What the benchmark can prove:** that on IBM AMLworld, a model trained on
labelled entities ranks *other entities of the same period* at ~31x prevalence
from their observed behaviour; that it cannot forecast first involvement two
days ahead; that graph features add little in either regime.

**What it cannot prove:** that the model transfers to a *new* period (the
standard benchmark is random folds within one period - roadmap R1 measures
this); anything about real data; anything about cycle or subgraph detection
quality; that any particular alert is laundering.

---

## 4. The role of graph analytics

A graph model does not have to be the predictive model, and on this evidence it
should not be.

| capability | as ML feature | as investigation analytics | state |
| --- | --- | --- | --- |
| counterparty structure, reciprocity, repeat pairs | yes (`aml_network_v1`) | yes: who, how often, both directions | features delivered; ego-network in UI |
| fan-in / fan-out / pass-through / turnaround | yes (`aml_flow_v1`) | yes: the mule signature, per entity | features delivered; not surfaced as evidence yet |
| circular transactions | yes (`aml_cycle_v1`: 2-paths, triangles, 4-walks, ordered layering) | **yes: the cycle itself, with per-leg transactions** | both delivered; cycles have no UI |
| connected components | yes (component size, density) | yes: the extent of a suspicious cluster | features only |
| community / ring discovery | not justified (no measured headroom) | yes, bounded to the neighbourhood of alerts | not built |
| transaction-chain / path reconstruction | no | yes: follow money from an alert | not built |
| evidence generation, explainability | - | yes | contract carries evidence references (this phase) |

**Decision: both, with investigation as the primary role.** Graph *features*
stay in the feature contract because they are cheap, monotone-safe, and each has
a direct investigative reading - not because they move AP (+0.0093 standard,
−0.0024 forecasting). Graph *analytics* are the investigator's view of an alert,
computed after scoring, bounded to the alert's neighbourhood, and presented as
facts, never as verdicts. Their value is currently a capability, not a
measurement; roadmap R2 measures it.

---

## 5. The feature system

| family | predictive: standard | predictive: forecasting | stability under time | investigative value | production contract |
| --- | --- | --- | --- | --- | --- |
| `aml_txn_v1` | 0.0589 alone | best single cell (0.0157) | stable (PSI family median 0.004) | high: volumes, amounts, round amounts | **keep** |
| `aml_temporal_v1` | **+0.1633** over txn - largest | inert (0.0152); inverts under survivorship (ROC 0.478) | **unstable** (family PSI 0.275; `tenure_days` 7.40, `recency_days` 2.31, `span_days` 1.22) | medium: velocity, burstiness, dormancy | **keep as-is**: `recency_days` / `tenure_days` are only meaningful with `as_of` = window end, which the engine enforces. The window-relative redesign was built and measured as R6 (`aml_temporal_v2`) and **rejected** - it did not reduce family-level drift and cost retention on the folds that transfer (`docs/r6_temporal_v2.md`). Instability is handled by the PSI gate, not by re-expressing the family |
| `aml_flow_v1` | +0.0821 over txn | raises ROC 0.683 → 0.691, lowers AP | stable (0.009) | **high**: fan-in/out, pass-through, turnaround = mule typology | **keep** |
| `aml_network_v1` + `aml_cycle_v1` | +0.0604 over txn; +0.0093 added last | raises ROC to 0.700 alone; −0.0024 added last | stable (0.012 / 0.014) | **high**: counterparties, reciprocity, cycles, layering | **keep** (features); no GNN |
| `aml_context_v1` | restoring it moved boosting without graph from 0.2717 to 0.3734 (pre-/post-fix artifacts) | - | stable (0.001) | medium: payment channel and currency mix | **keep**; its strength is partly a property of IBM's format mix (ACH 0.746 % positive vs credit card 0.016 %) |

* **Redundancy:** no short column name appears in two `aml_*` families (a
  design rule, tested). The only known exact duplicate (`reciprocal_in_count` =
  `reciprocal_out_count`) is in the legacy 167-vector and stays for the frozen
  checkpoint only.
* **No family is removed.** The ones with small AP contribution (network,
  cycle) are the ones with the most direct investigative meaning.
* **Production feature version:** the six `aml_*_v1` families, identified in
  every assessment as `name@version+...` (`engines.feature_version_for`). The
  167-vector remains the frozen checkpoint's input and nothing else's.

---

## 6. Final architecture

```text
 DATA INGESTION              src/data/, src/datasets/parsers.py              delivered
       ↓
 SCHEMA / SEMANTIC MAPPING   src/datasets/profiler, mapper, canonicalizer    delivered
       ↓
 DATA QUALITY + TRUST        src/datasets/policy, compatibility, trust       delivered
       ↓
 BEHAVIOUR ENGINE            src/experiments/aml_features.py (6 families)    delivered
   (window-scoped features)  legacy src/features/ for the frozen checkpoint  frozen
       ↓
 ML RISK ENGINE              src/detection/engines.py                        NEW: contract seam
   CandidateRiskEngine  - a benchmarked model over an analysis window
   LegacyGraphSAGEEngine - the frozen checkpoint, restated
       ↓
 RISK ASSESSMENT (contract)  src/detection/contract.py                       NEW
   scores + rank + decision + model/feature version + task + window + evidence
       ↓
 INVESTIGATION GRAPH         the window's MultiDiGraph (src/graph/)          delivered
       ↓
 TEMPORAL / FLOW / CYCLE     src/detection/cycles.py                         delivered (cycles)
   ANALYSIS                  paths, components, communities                  roadmap R2/R8
       ↓
 EXPLANATION                 contract.entity_evidence: feature + cycle refs  NEW (seam)
                             adapter flags                                   legacy
       ↓
 ANALYST UI                  src/ui/                                         reads legacy columns;
                                                                             migration to the contract pending
```

Boundaries, in order of importance:

1. **Nothing downstream of the risk engine reads a model-specific column.** It
   reads a `RiskAssessment`. (Delivered for persistence; the UI adapter still
   reads legacy names - the migration is the next implementation step.)
2. **Investigation analytics never change a score.** They take the assessment
   and the graph and return evidence (`entity_evidence`; `cycles.py` already
   states and tests this).
3. **Every score carries its task.** A retrospective model cannot be shown as a
   forecaster; a model with no evaluation says so (`evaluation_reference=None`).
4. **The frozen checkpoint is one engine among others.** The frozen model contract is
   untouched; the engine restates its output without changing a number.

---

## 7. The production ML contract

Implemented in `src/detection/contract.py`:

```text
Input  (CandidateRiskEngine)
    TransactionWindow
        entity_ids          the population
        edge_table          normalized transactions: source_entity, target_entity,
                            date, amount | amount_usd, context columns
        window              AnalysisWindow(start, end)  - declared bounds
        attributes          optional per-entity attributes
Input  (LegacyGraphSAGEEngine)
    DetectionState from the frozen canonical path (unchanged)

Output RiskAssessment
    model   ModelIdentity   model_id, model_version, family, feature_version,
                            task (retrospective_detection | forward_forecasting),
                            score_kind (calibrated_probability | model_probability |
                                        rank_aggregate),
                            evaluation_reference, evaluation_regime
    window  AnalysisWindow  declared start/end + observed start/end
    scores  entity_id, risk_score, rank (1 = highest, ties by entity_id),
            decision (only with a threshold and its basis), ground_truth (null ≠ 0)
    evidence  observed quantities per entity, by name; no label columns allowed
    threshold, threshold_basis, produced_at, notes, contract_version
    metadata()   everything but the rows - persisted per run as assessment.json
    entity_evidence(assessment, entity, cycle_analysis)
                 feature evidence + cycle and counterparty references
```

Every rule is enforced at construction and tested (`tests/test_risk_contract.py`).
Workspace runs now persist `assessment.json` beside `predictions.parquet`,
identifying the model by checkpoint digest rather than the free-text
`"Experiment 3 GraphSAGE checkpoint"`.

Not yet delivered, in order: the UI adapter reading the contract; a persisted,
versioned `CandidateRiskEngine` (model registry, §8); one input representation
for both engines.

---

## 8. Industrial requirements

| requirement | today | class |
| --- | --- | --- |
| reproducible training | experiment runner: seeded, config-driven, artifacts | **must-have now: extend to a persisted production model** (R1) |
| dataset provenance | manifests with SHA-256 (`data/manifests/`) | have |
| model registry / versioning | frozen checkpoint only; experiment models not persisted by default | **must-have now**: a file-based registry (model file + identity + training artifact reference). A registry *service* is unjustified |
| feature versioning | family name + version in every artifact and assessment | have |
| experiment tracking | JSON artifacts + `compare_experiments.py` | have; a tracking server is **unjustified** |
| deterministic pipelines | seeds + honest determinism report | have |
| train / validation / test separation | enforced by the runner | have |
| temporal evaluation | calendar-forward regime | have |
| calibration | Platt on validation; carries almost no information under forecasting | have; **do not present scores as probabilities** |
| threshold management | validation-only; alert budget | have; the contract now requires a threshold basis |
| alert-budget evaluation, P@K, R@K | every artifact | have |
| analyst explanations | legacy rule flags; evidence references (new) | **must-have now: evidence from the contract in the UI** |
| audit trail | artifacts, trust.json, assessment.json | have; per-alert analyst actions are later |
| data-quality monitoring | profiling, compatibility, readiness, trust | have |
| drift monitoring | one PSI definition in `src/experiments/drift.py`, used by the design audit and by every cross-period fold | **must-have now** (promoted by R1): PSI on `aml_temporal_v1` between the fitted window and the scored window separated the folds that transferred (retention 0.98) from the one that did not (0.11). Applying a frozen model to a new window without it is unsafe |
| threshold portability | thresholds are chosen on validation, per run | **must-have now** (promoted by R1): re-derive the alert budget on each analysis window. Carrying a threshold across windows moved the alert rate from ~1 % to 12 % and FP/TP from 3 to 74 |
| OOD detection | trust ceiling `UNVERIFIED` by design | valuable later |
| model health | none | valuable later |
| inference latency | six families on 4.2 M transactions ~65 s; full IBM snapshot ~92 s; peak ~3.3 GB | adequate for batch review windows |
| scalability | 5-10 M transactions in memory on 16 GB | adequate; distributed compute **unjustified** |
| failure handling | cycle enumeration and trust/assessment writes contained, never costing a prediction | have |
| **unjustified complexity** | GNN serving, feature store, Kubernetes, streaming/real-time scoring, online learning, experiment-tracking service, distributed training | - |

---

## 9. Research roadmap

Prioritised. Advanced methods appear only behind their prerequisite evidence.

| # | experiment | hypothesis | motivation | data | success | failure | cost | production impact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ~~**R1**~~ **DONE 2026-09-12** | **Cross-period retrospective detection** | *answered:* **LIMITED TEMPORAL TRANSFER.** AP retention 0.975 / 0.984 between comparable 2-day windows (AP 0.3145 / 0.3425, 17-20x the floor), and 0.111 across a dissimilar one; transfer is not entity memorisation; a frozen threshold is not portable (alert rate 1 % -> 12 %); PSI on `aml_temporal_v1` separates the folds that transfer from the one that does not | — | IBM, four 2-day windows over 09-03..09-11, three chronological folds | — | — | ran in ~25 min | **delivered:** the product claim now has cross-period evidence with a stated comparability condition; production must re-derive thresholds per window and gate on temporal drift. Record: `docs/r1_detection_transfer.md` |
| ~~**R6**~~ **DONE 2026-09-12** | **Window-relative temporal features** (`aml_temporal_v2`) | *answered:* **NO - hypothesis rejected.** Re-expressing temporal position on the window's own activity clock compressed the extreme drift (largest temporal PSI 2.4934 -> 0.9122) but left the typical drift untouched (family median 0.2722 -> 0.2717, still 16/48 past 0.25). W3->W4 retention 0.111 -> **0.192**, far short of 0.50, while W1->W2 retention *fell* 0.975 -> **0.881**. Window composition, not calendar encoding, is the cause | — | IBM, the same R1 protocol, single-variable ablation | — | — | ran in ~12 min | **delivered:** `aml_temporal_v1` retained; the feature contract is unchanged; window comparability is a data requirement and the PSI gate is confirmed as the safeguard. `aml_temporal_v2` is registered and tested but not adopted. Record: `docs/r6_temporal_v2.md` |
| R2 | Investigation-evidence yield | graph analytics around top-K alerts surface flagged transactions more often for true than false alerts | the graph's value is claimed, not measured | IBM transaction labels (+ typology file) | evidence (cycle / path / fan pattern containing a flagged transaction) found for ≥ 2x as many true as false alerts at K = 1 % | no difference: graph analytics are presentation, not signal | low | shapes the investigation layer |
| R3 | Transaction-level scoring | IBM's transaction labels support a transaction ranker whose alerts are more precise than entity alerts | the entity label is derived and counterparty-contagious | IBM, `Is Laundering` per row | transaction AP ≥ 10x prevalence over ≥ 4 seeds | ≤ 3x | medium | adds a transaction-risk output to the contract |
| R4 | Persisted tabular engine + registry | the benchmarked boosting model reproduces its artifact metrics after save/load through `CandidateRiskEngine` | reproducible training, model versioning | IBM | identical scores after reload; AP within 1e-9 | any drift | low | the production model path |
| R5 | Tuned boosting / ranking objective | a ranking objective or tuned depth/rate improves AP beyond seed spread | fixed configuration everywhere so far | IBM (R1 regime) | +0.02 AP over 4 seeds | within seed spread | low-medium | model choice only |
| ~~R6~~ | ~~Window-relative temporal features~~ | **closed 2026-09-12 - see the R6 row above.** Measured and rejected; `aml_temporal_v1` retained | — | — | — | — | — | none |
| R7 | Non-monotone label-free floor | an isolation-forest floor is a stricter baseline than the rank sum | recorded gap | IBM | runs; reported beside every result | - | low | baseline only |
| R8 | Ring / community discovery around alerts | bounded community detection groups true alerts more than false ones | the "ring" half of the objective | IBM + typology file | typology-instance recall ≥ 0.5 on alert neighbourhoods | < 0.2 | medium | investigation layer |
| R9 | Edge-aware GNN (GINE-style) | message passing over edge attributes beats tabular transaction features | **gated**: only if R3 shows edge attributes carry value tabular features miss **and** a graph increment exceeds seed spread in R1 | IBM | +0.02 AP over 4 seeds in R1/R3 | anything less | high | would add an engine; contract unchanged |
| R10 | Temporal / heterogeneous GNN | - | **NO-GO**: 10 days of data, one node type | - | - | - | - | - |
| R11 | TDA on alert subgraphs | persistent homology separates typologies that cycle/community methods miss | **gated** behind R2 and R8 showing headroom | IBM typology file | measurable gain over R8 | - | high | - |

---

## 10. Dataset strategy

| purpose | IBM HI-Small sufficient? |
| --- | --- |
| engineering validation (scale, correctness) | **yes** (5.08 M transactions; SAML-D adds 9.5 M) |
| model comparison | **yes for retrospective ranking on this simulator**; forecasting comparisons need ≥ 4 seeds (AP spans 0.0084-0.0167 across seeds) |
| SIH demonstration | **yes**, labelled synthetic |
| research claims | only claims *about this simulator* |
| real-world generalisation | **no** - no dataset in reach supports it |

**A controlled benchmark is not needed now.** The next questions (R1-R3) are
answerable on IBM, and R2/R8 need typology ground truth that IBM already
publishes in `HI-Small_Patterns.txt` - acquire it (with a manifest) as
evaluation ground truth for investigation analytics, never as a feature. The
LI-Small variant of the same generator adds a second prevalence regime cheaply.

**Build a controlled benchmark only if** R2/R8 need ground truth IBM lacks, or a
GNN claim (R9) needs a dataset whose labels depend on multi-hop structure by
construction. It should then test exactly:

1. planted typologies (cycle, fan-in, fan-out, gather-scatter, layering chain)
   with known member sets - recall of investigation analytics;
2. a controlled dial on how much the label depends on multi-hop structure versus
   1-hop behaviour - *when* message passing should help;
3. temporal stationarity (no burn-in, no tail) - cross-period transfer without
   generator artefacts;
4. label definitions with and without counterparty contagion.

The real gap remains **real data**; no synthetic benchmark closes it.

---

## 11. SIH demonstration strategy

| what the ML benchmark proves | what the analytical system demonstrates |
| --- | --- |
| On IBM AMLworld (synthetic), boosting ranks laundering-involved entities at AP 0.3827 - 31x prevalence, 9.8x a label-free floor - within one period | For a flagged entity: its transactions, counterparties, fan-in/out, the cycles it participates in with per-leg amounts and timing, and how its activity evolved |
| Forecasting two days ahead is weak (1.4-2.0x on first involvement) | That these views exist and are traceable to raw transactions |
| Graph features add little to ranking | That circular flows and layering patterns can be *found and shown* - not that finding them improves accuracy (R2 unmeasured) |
| Nothing about real banks | Nothing about real banks |

Demonstrate: rank → select an alert → evidence (behaviour features, flow
pattern, cycles, counterparties) → timeline, with the model identity, task and
evaluation regime visible on screen (they are in every assessment). Say
"ranked highly because…", never "is laundering". Label every IBM view
synthetic. Do not show 0.3827 as a forecast, or the frozen GraphSAGE as a
validated detector.

---

## 12. Decision

**OPTION 3 - ML + investigation-intelligence hybrid.**

* Option 1 (ML-first) is the ranking half of this decision without the half
  that explains an alert or addresses circular flows.
* Option 2 (forecasting-first) is contradicted by the calendar-forward
  benchmark.
* Option 3 is the only option whose ranking claim is measured *and* whose
  investigation half serves the original objective without a new model.

What would change it: R1 failing (the ranking claim narrows to within-period);
R2 showing graph evidence is no more frequent for true than false alerts (graph
analytics become presentation only); a real dataset.

---

## 13. Implemented in this phase

The highest-value alignment is the boundary, not a model:

* `src/detection/contract.py` - the risk contract, enforced.
* `src/detection/engines.py` - `LegacyGraphSAGEEngine` (frozen numbers restated,
  identity by checkpoint digest) and `CandidateRiskEngine` (benchmarked models
  over an analysis window; nothing after the window can move a score).
* `Candidate.score_features` - the production scoring seam, implemented for
  the logistic, boosting and label-free candidates.
* `src/datasets/inference.py` - every workspace run persists `assessment.json`,
  contained so it can never cost a prediction.
* `tests/test_risk_contract.py`.

Deliberately not done: UI migration to the contract, a model registry, the IBM
production model, any GNN, TDA, or hyperparameter search.
