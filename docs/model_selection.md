# Model Selection

What was measured, on which data, under which protocol, and what the evidence
supports.

Reproduce:

```bash
python audit_datasets.py
python run_experiment.py configs/experiments/samld_*.json
python compare_experiments.py outputs/experiments/samld_*.json --report docs/model_selection_table.md
```

* generated results table: `docs/model_selection_table.md`
* machine-readable artifacts: `outputs/experiments/*.json`
* dataset audit and verdicts: `docs/datasets.md`

Run date: 2026-09-06. Environment: Python 3.11.8, Windows, CPU only, 16 GB.
Gradient-boosting backend: scikit-learn `HistGradientBoostingClassifier`
(neither `xgboost` nor `lightgbm` is installed; the backend that ran is recorded
in each artifact's fit report).

> **Read this first.** The document is in two halves. Sections 0-9 are SAML-D
> and the Reference Dataset, **neither of which can rank models** — they are
> kept because they establish the framework's correctness and the reasoning that
> led here. The section headed *IBM AMLworld HI-Small* is the model-selection
> benchmark and is where the architecture decision is actually made.

---

## 0. Read this before any number below

Every model result in this document was produced on **SAML-D**, which is:

* **synthetic** — generated from declared typologies, not observed; and
* classified **`STRUCTURAL_VALIDATION_ONLY`**, because an untuned probe on
  fifteen simple structural quantities alone reaches average precision 0.9971 on
  its entity labels.

So these runs **do not rank models**. What they do establish is different and
still worth having:

1. the whole stack is correct and affordable at 855,460 entities and 9,504,852
   transactions;
2. what a label-free structural floor achieves;
3. which information sources a supervised model actually uses on this generator;
4. that the SAML-D result is a property of the data, reproducibly, in an
   artifact rather than in an assertion.

No number here is evidence about real-world detection performance.

---

## 1. Protocol

| element | choice | why |
| --- | --- | --- |
| unit of prediction | entity (account) | the project's task is mule-account detection |
| label | account is sender or receiver of >= 1 transaction flagged `Is_laundering` | SAML-D labels transactions; the derivation is recorded in `DatasetSpec.label_definition` |
| main split | `stratified_kfold`, 4 folds, entity-disjoint | first-look comparability; explicitly **not** time-aware |
| operational split | `time_forward_holdout`, entities placed by **last** activity, features scoped to `train_window` | the deployment question: can behaviour up to a boundary predict involvement realised after it |
| threshold | `alert_budget` at 1 % of the scored population, chosen on **validation** | an AML team's constraint is alerts per day, not a probability |
| calibration | Platt, fitted on validation, minimum 20 validation positives | supervised candidates only; the heuristic emits a rank aggregate, not a probability |
| test partition | scored once, after threshold and calibrator are frozen | enforced by the runner, not by convention |
| imbalance | balanced class weights | recorded in the artifact so runs using different strategies are not silently compared |

Placing entities by *last* activity in the time-forward split is a deliberate
choice with a visible consequence, stated rather than hidden: positive accounts
are more active and so more likely to have a recent last transaction, which
concentrates positives in the later partition. Placing them by *first* activity
instead would put entities with **no** pre-boundary history into the test
partition, leaving a training-window model nothing to score. Both readings are
legitimate; the artifact records which was used and the partition counts it
produced.

Undefined metrics stay `null` with a reason. A fold whose test partition holds
no positives contributes nothing to a mean rather than dragging it toward zero,
and `n_defined` sits beside every aggregate.

---

## 2. The Reference Dataset — regression, not comparison

Four positives, 3,440 entities, `stratified_kfold` with 4 folds, so **one
positive per test partition**. Under those conditions average precision can only
take the value `1 / rank` of that single entity, and it is not an estimate of
anything.

| experiment | model | features | test AP | fold values |
| --- | --- | --- | --- | --- |
| `heuristic_floor` | structural heuristic (label-free) | `aml_core_v1` | **1.0000 ± 0.0000** | 1.0000, 1.0000, 1.0000, 1.0000 |
| `logistic_aml_core` | logistic regression | `aml_core_v1` | **1.0000 ± 0.0000** | 1.0000, 1.0000, 1.0000, 1.0000 |
| `boosting_aml_core` | gradient boosting | `aml_core_v1` | **1.0000 ± 0.0000** | 1.0000, 1.0000, 1.0000, 1.0000 |
| `legacy_graphsage_exp3` | Experiment 3 GraphSAGE | `legacy_exp3_v1` (167) | 0.7535 ± 0.4269 | 0.0141, 1.0000, 1.0000, 1.0000 |

Two things follow, and only two:

1. **Three candidates tie at a perfect score, one of which reads no labels at
   all.** A protocol on which an unsupervised rank aggregate is indistinguishable
   from gradient boosting is not measuring detection skill.
2. **The legacy GraphSAGE, retrained under this protocol, scores below the
   label-free floor**, and its dispersion (±0.4269 across four folds) is larger
   than the quantity being estimated. One fold placed its single positive at
   rank 71. This is a benchmark of the architecture under the new protocol, not
   a reproduction of the historical §28.3 run, and the honest reading is that
   the dataset cannot separate these models rather than that GraphSAGE is bad.

`time_forward_boosting` returns **undefined in every fold**: each test partition
holds zero positives. The runner reports `null` with the reason rather than
substituting `0.0`, and its alert rate of 1.0 is the correct behaviour of a model
trained on a single-class partition — it emits a constant prior, so an alert
budget threshold admits everything. That is the finding, not a failure.

---

## 3. SAML-D — the central comparison

855,460 entities, 7,902 positive (0.924 %), `stratified_kfold` 4 folds,
1,976 test positives per fold, `global` feature scope. Seeds 42 and 1337.

| candidate | features | test AP | ROC | P@1 % | R@1 % | FP per TP |
| --- | --- | --- | --- | --- | --- | --- |
| `samld_heuristic_floor` — **label-free** | all six | **0.1768 ± 0.0013** | 0.9458 | 0.0462 | 0.0500 | 20.87 |
| `samld_boosting_behaviour` — no graph | txn + temporal + flow + context | 0.9157 ± 0.0050 | 0.9990 | 0.8456 | 0.9156 | 0.186 |
| `samld_logistic_aml` | all six | 0.9987 ± 0.0008 | 1.0000 | 0.9225 | 0.9989 | 0.087 |
| `samld_boosting_graph` | all six | **0.9997 ± 0.0002** | 0.9998 | 0.9232 | 0.9996 | 0.081 |

Seed repeat, so that the fold spread can be read against a run-to-run spread:

| candidate | seed 42 | seed 1337 |
| --- | --- | --- |
| `samld_heuristic_floor` | 0.1768 ± 0.0013 | 0.1769 ± 0.0036 |
| `samld_boosting_graph` | 0.9997 ± 0.0002 | 0.9999 ± 0.0002 |

Fold-level values are in `docs/model_selection_table.md`; no fold of any
supervised run falls below 0.9995, and no fold of the floor rises above 0.183.
The dispersion is far smaller than the gaps, so nothing here rests on a lucky
fold — and none of it ranks models, because the dataset saturates.

### 3.1 What the floor's failure actually shows

The label-free floor reaches AP 0.1768 at a prevalence of 0.0092 — a 19.1×
lift, and a genuinely poor detector. Its ROC-AUC is 0.9458, which is the giveaway:
the *ordering* is good, the *top of the list* is not. Averaged over the four
folds: precision@100 = 0.2975, precision@500 = 0.0880, precision@1000 = 0.0582,
then precision@5000 recovers to 0.2645 at recall 0.6695. Precision is not
monotone in K and is not required to be — a block of roughly a thousand
negatives sits above the positives, and the positives are then dense between
ranks 1,000 and 5,000, where the floor recovers two thirds of the cases.

The reason is structural and worth recording, because it is a limitation of the
*form* of the floor rather than of the data. The heuristic is a **signed rank
sum**: it can only express "more of this is more suspicious". SAML-D's positives
occupy a *band* — counterparty counts of roughly 5 to 25, with the highest-degree
accounts being ordinary hubs. A rank aggregate cannot represent a band, so it
puts the busiest benign hubs at the top. That is why ranking by
`counterparty_count` gives AP 0.182 while a tree on the same single column
reaches far more.

**Consequence for future datasets:** a supervised model beating this floor is
weaker evidence than it looks whenever the true signal is non-monotone. The floor
is still the right baseline — it reads no labels — but a second, non-monotone
label-free baseline would make the comparison sharper. Recorded as future work,
not done here.

### 3.2 Calibration

`samld_boosting_graph`, Platt-scaled on validation only: Brier 2.51e-5,
expected calibration error 2.58e-4. Those are the numbers of a model that has
learned a deterministic rule, not of a well-calibrated risk estimate, and they
should be read as another symptom of the dataset rather than as a virtue.

---

## 4. SAML-D — time-forward, the operational regime

`time_forward_holdout`, entities placed by last activity, features restricted to
`train_window`. One fold: train 513,276 (1,046 positive), validation 171,092
(1,660 positive), test 171,092 (5,196 positive, prevalence 3.04 %).

| candidate | test AP | lift | ROC | P@1 % | R@1 % |
| --- | --- | --- | --- | --- | --- |
| `samld_tf_heuristic_floor` | 0.1504 | 4.95× | 0.9132 | — | — |
| `samld_tf_boosting_graph` | **0.8152** | 26.8× | 0.9723 | 1.0000 | 0.3293 |

Two things are worth taking from this, and they are the most transferable
results in this document:

1. **The regime matters more than the model.** The same model that reaches
   0.9997 transductively reaches 0.8152 when its features may only see
   transactions before the boundary. Anyone quoting the first number as
   operational performance would be overstating it by an enormous margin. This
   is why `compare_experiments.py` refuses to put them in one table.
2. **At a 1 % alert budget the model finds every alert correct and one third of
   the cases** (precision 1.0000, recall 0.3293). That is a usable operating
   point *on this generator*, and it is the shape of result an AML team actually
   cares about — but it is still a result about a simulator.

Brier 0.0796 and expected calibration error 0.0796 under this regime: far worse
than the transductive run, and honest.

---

## 5. Ablation — which information source contributes?

Same model, same split, same seed; only the feature families change.

| cell | families | test AP | Δ vs A |
| --- | --- | --- | --- |
| A | `aml_txn_v1` | 0.8454 ± 0.0128 | — |
| B | A + `aml_temporal_v1` | 0.8603 ± 0.0101 | +0.015 |
| C | A + `aml_flow_v1` | 0.8863 ± 0.0075 | +0.041 |
| D | A + `aml_network_v1` + `aml_cycle_v1` | **0.9997 ± 0.0003** | **+0.154** |
| E | all six | 0.9997 ± 0.0002 | +0.154 |

**D equals E.** Once network and cycle structure are present, temporal, flow and
context features add nothing measurable — to four decimal places. Transaction
volume alone already reaches 0.8454; the entire remaining gap is closed by the
counterparty structure, and that structure is the generator's signature.

Read as an AML result this would say "the graph is where the signal is". Read
correctly it says the label was written from the graph. The ablation is
informative about the *dataset*, not about money laundering.

---

## 6. Control — is the evaluation stack capable of a non-degenerate result?

The obvious objection to §3 is that something in the protocol produces near-1.0
scores. `financial_fraud_kaggle` answers it: same features, same candidates, same
split, same code path, on the one available dataset whose labels are not
structurally trivial.

| candidate | test AP | prevalence | lift | ROC |
| --- | --- | --- | --- | --- |
| `ffk_heuristic_floor` | 0.3871 ± 0.0006 | 0.3285 | 1.18× | 0.5823 |
| `ffk_boosting_graph` | 0.4225 ± 0.0016 | 0.3285 | 1.29× | 0.6151 |

Barely above prevalence, and correctly so: 5,000,000 transactions produce
4,999,985 distinct ordered pairs, so the account graph is a random matching with
nothing for a network feature to find. The stack does not manufacture high
scores. It reports what is there.

This is a control, not an AML result — the label is card-not-present fraud.

---

## 7. Advanced graph learning: **NO-GO**

Against the project owner's stated criteria.

| GO condition | met? | evidence |
| --- | --- | --- |
| labelled data adequate | **no** | no dataset is `MODEL_SELECTION_READY` (`docs/datasets.md`) |
| leakage controlled | yes | partition, label-column, perfect-separator, temporal-ordering and scope audits run in every fold and are recorded in every artifact |
| supervised beats the label-free floor | yes, but | 0.1768 → 0.9997 on SAML-D — on labels a probe recovers from structure at 0.9971 |
| graph information contributes beyond transaction/temporal | yes, but | ablation D − C = +0.113 — and it contributes *because the labels were built from the graph* |
| the remaining problem plausibly needs relational representation | **no** | see below |

| NO-GO condition | met? | evidence |
| --- | --- | --- |
| labels trivial | **yes** | probe AP 0.9971 on SAML-D; single-column AP 1.0000 on the Reference Dataset |
| datasets too small | **yes** | the one real dataset has 4 positives |
| graph features already capture essentially all useful information | **yes** | ablation D = E to four decimals |
| GNN improvement cannot be demonstrated | **yes** | there is no dataset on which to demonstrate it: a candidate cannot beat 0.9997 by enough to measure |
| evaluation remains ambiguous | no | the regimes are explicit and enforced |

**Decision: do not build an edge-aware, temporal or heterogeneous GNN in the
next phase.** Three reasons, in order of force:

1. **There is nothing to measure it against.** A new architecture would be
   evaluated on a dataset where gradient boosting already scores 0.9997 and
   *L2 logistic regression* scores 0.9987. No architectural difference survives
   that ceiling.

2. **SAML-D's graph would not reward message passing even if the labels were
   sound.** 90 % of accounts have exactly one counterparty; the graph has
   **zero directed 3-cycles**; there are 15,592 weakly connected components with
   a median size of 51. A message-passing layer has almost no neighbourhood to
   aggregate over, and the one hop that exists is already captured by a single
   tabular column — `neighbour_counterparty_mean`.

3. **The blocker is data, and building a model does not move it.** This is a
   NO-GO on evidence, not a judgement that graph learning is wrong for AML. If a
   dataset arrives whose labels are not a function of its structure and whose
   graph has real multi-hop structure, this decision should be revisited on that
   evidence — the framework is ready for it, and adding a model family is one
   file.

Do not read this as "GraphSAGE is weak, therefore use a Transformer GNN". The
correct next step is not an architecture at all.

---

## 8. Architecture state after this phase

| component | status |
| --- | --- |
| Experiment 3 GraphSAGE checkpoint + 167-feature contract | **legacy, frozen.** Byte-stable for the UI, the workspace prediction route and `tests/test_reference_regression.py`. Not an architectural invariant. |
| `legacy_exp3_v1`, `baseline_v1`, `multigraph_v1`, `temporal_v1`, `tga_v1` | **legacy comparison feature sets.** Runnable, unchanged. |
| `aml_core_v1` | **stable.** Pinned by the artifacts that recorded it; not modified. |
| `aml_txn_v1`, `aml_temporal_v1`, `aml_flow_v1`, `aml_network_v1`, `aml_cycle_v1`, `aml_context_v1` | **stable, current.** 68 columns, no name shared between families. |
| `src/experiments/` Layer 6 (splits, metrics, thresholds, calibration, leakage, runner, artifacts) | **stable.** |
| `src/experiments/audit.py` + `audit_datasets.py` | **stable, and a required stage** before any comparison. |
| `src/experiments/sources.py` | **stable.** Adding a benchmark is a `SourceSpec` plus a `DatasetSpec`. |
| edge-aware / temporal / heterogeneous GNN | **not built.** NO-GO on evidence; see §7. |
| persistent homology / TDA | **not built.** Deferred with a stated precondition — `docs/architecture.md` §45.10. |
| prediction/evidence contract for the UI | **declared, not delivered** — `docs/architecture.md` §45.9. |

---

## 9. What this phase did not establish

Listed so that absent evidence stays visible.

* **No real-world detection performance.** Every model number above is from
  synthetic data.
* **No model ranking.** The candidates are not ordered by any of this. On
  SAML-D they tie at the ceiling; on the Reference Dataset three of them tie at
  1.0000 including one that reads no labels.
* **No transaction-level result.** SAML-D labels transactions and this project
  classifies entities. The entity derivation is what the triviality finding
  applies to; whether *transaction*-level SAML-D is also trivial is untested and
  is a different task from the one the project is built for.
* **No hyperparameter search.** Every candidate ran at one fixed configuration.
  Tuning inside the fold, on validation, is supported by the runner and was not
  used — on a saturated dataset it would only be measuring noise.
* **No second non-monotone label-free baseline** (see §3.1).

---

# IBM AMLworld HI-Small — the model-selection benchmark

Everything above this line is SAML-D and the Reference Dataset, neither of which
can rank models. This section is the one that can.

**Still synthetic.** IBM AMLworld is a virtual-world simulator. These numbers
rank models *on that simulator*. They are not evidence about real-world
detection rates, prevalence or alert burden.

> **A defect was fixed on 2026-09-06 after these runs were first produced.**
> `aml_context_v1` matched none of IBM's column spellings and silently produced
> six all-NaN columns, so early IBM runs used 62 effective features rather than
> 68. Every affected cell was re-run; the numbers below are post-fix. See
> `docs/temporal_collapse.md` §6, and `outputs/experiments/_prefix_context/` for
> the superseded artifacts.

Dataset: 515,088 entities, 5,078,345 transactions, 6,357 positive entities
(1.234 %), 17 days. Structural probe AP **0.2098**, no class-free bands — see
`docs/datasets.md` §4. Provenance: `data/manifests/ibm_aml_hi_small.json`.

## I.1 Central comparison

`stratified_kfold`, 4 folds, ~1,589 test positives per fold, `global` feature
scope, seed 42. Prevalence 0.0123.

| candidate | features | test AP | lift | ROC | vs floor |
| --- | --- | --- | ---: | ---: | ---: |
| `ibm_heuristic_floor` — **label-free** | all six | **0.0390 ± 0.0011** | 3.2× | 0.7527 | — |
| `ibm_logistic_aml` | all six | 0.1710 | 13.9× | 0.8709 | ×4.4 |
| `ibm_boosting_behaviour` — no graph | txn + temporal + flow + context | 0.3734 | 30.3× | 0.9009 | ×9.6 |
| `ibm_boosting_graph` | all six | **0.3827** | 31.0× | 0.9215 | ×9.8 |

This is what a benchmark that can separate models looks like. Compare SAML-D,
where the same four candidates landed at 0.1768 / 0.9987 / 0.9157 / 0.9997 —
three of them within 0.09 of each other at the ceiling.

Three things are now measurable that were not before:

1. **Model family matters.** Gradient boosting reaches 3.0× the average
   precision of logistic regression on identical features (0.3734 vs 0.1710).
   On SAML-D these two were indistinguishable.
2. **The label-free floor is genuinely weak** — 3.2× lift, ROC 0.7527. Beating
   it is informative rather than automatic.
3. **There is headroom.** The best candidate reaches AP 0.28. Nothing saturates,
   so a better model has somewhere to go.

## I.2 Ablation — which information source contributes?

Same model, split and seed; only the feature families change.

| cell | families | test AP | Δ vs A |
| --- | --- | --- | ---: |
| A | `aml_txn_v1` | 0.0589 ± 0.0034 | — |
| B | A + `aml_temporal_v1` | **0.2222 ± 0.0084** | **+0.1633** |
| C | A + `aml_flow_v1` | 0.1410 ± 0.0069 | +0.0821 |
| D | A + `aml_network_v1` + `aml_cycle_v1` | 0.1193 ± 0.0101 | +0.0604 |
| E | all six | **0.3827** | +0.3238 |

Read carefully, because this is the ablation that drives the architecture
decision:

* **Temporal behaviour is the dominant single source** (+0.163 over transaction
  features alone). Timing, velocity, burstiness and dormancy carry more than
  twice what network structure does.
* **Network and cycle structure do contribute** (+0.060 over A on their own).
* **The sources are complementary.** E exceeds the best single addition B by
  +0.1605, so no one family subsumes the others. On SAML-D, by contrast, D and E
  were identical to four decimal places — network structure there *was* the
  label.
* **Network's marginal value, added last, is small but consistent.** Going from
  `ibm_boosting_behaviour` (no graph) to `ibm_boosting_graph` (all six) is
  **+0.0093 AP** and **+0.0206 ROC**. Both runs share the same four folds
  (split seed 42), and the paired per-fold gain is positive in every one:
  +0.0042, +0.0145, +0.0127, +0.0059. It exceeds the fold spread of either run
  (±0.0076 and ±0.0057).

So under random folds the graph families add a real, small increment. That is
not the whole story: under the time-forward regime the same comparison is
**−0.0010** (`docs/temporal_collapse.md` §9). The increment exists where
temporal position is available and disappears where it is not.

> Pre-fix, this passage read "+0.0080 AP, smaller than either run's dispersion".
> Those figures were computed with `aml_context_v1` silently dead; they are
> superseded.

## I.3 Time-forward, and the temporal collapse

`time_forward_holdout`, entities placed by **last** activity, features
restricted to `train_window`. Train 309,003 (2,417 positive) / validation
102,984 (1,217) / test 103,101 (2,723). Test prevalence 2.64 %.

| candidate | test AP | lift | ROC |
| --- | ---: | ---: | ---: |
| `ibm_tf_heuristic_floor` | 0.0389 | 1.47× | 0.5519 |
| `ibm_tf_logistic_aml` | 0.0525 | 1.99× | 0.5957 |
| `ibm_tf_boosting_behaviour` | **0.1120** | 4.24× | 0.5920 |
| `ibm_tf_boosting_graph` | 0.1110 | 4.20× | 0.6190 |

**The transductive 0.3827 becomes 0.1110 — a 71 % drop**, with ROC falling from
0.9215 to 0.6190.

That collapse was investigated in full; the record is
**`docs/temporal_collapse.md`**. The short version:

* It decomposes almost exactly evenly. Holding scope at `global` and changing
  only the split costs 0.3827 → 0.2453 (50.6 %); holding the split and scoping
  the features costs 0.2453 → 0.1110 (49.4 %).
* The split assigns entities by *last activity*, so it **manufactures its own
  distribution shift**: train median 3 transactions against test median 37, 48
  of 54 features past PSI 0.25, prevalence 0.78 % against 2.64 %. It is a
  survivorship split, not a forecasting split.
* The generator drifts. Days 1–10 hold 99.98 % of transactions at a 0.03–0.21 %
  positive rate; days 11–18 hold 1,108 transactions at **57–73 % positive and
  100 % ACH**. 26.3 % of test positives draw their label from that tail.
* It is **not** a cold-start problem: 99.6 % of test entities have pre-boundary
  history and the benchmark is 99.7 % transductive in entity terms.
* Classification: **MIXED** — split methodology, feature scope, distribution
  shift and label drift, with a genuine but smaller temporal generalization
  failure. Entity coverage ruled out on evidence.

**Both numbers stand, with their regimes attached.** 0.3827 is a valid
transductive result. 0.1110 is a valid result for a survivorship split over a
drifting label process. Neither is "the" performance of the system.

## I.4 Time-forward ablation — the ordering inverts

| cell | families | standard | time-forward | TF ROC |
| --- | --- | ---: | ---: | ---: |
| A | `aml_txn_v1` | 0.0589 | 0.0363 | 0.5334 |
| B | A + temporal | **0.2222** | **0.0295** | **0.4783** |
| C | A + flow | 0.1410 | 0.0553 | 0.6074 |
| D | A + network + cycle | 0.1193 | 0.0618 | 0.6228 |
| E | all six | **0.3827** | **0.1110** | 0.6190 |

Temporal features are the strongest addition under random folds (+0.1633 over A)
and **actively harmful** under shift — B ranks worse than chance (ROC 0.4783).

Marginal value of the graph families, comparing the two full models:

| | standard | time-forward |
| --- | ---: | ---: |
| `boosting_behaviour` (no graph) | 0.3734 | **0.1120** |
| `boosting_graph` (all six) | 0.3827 | **0.1110** |
| network + cycle contribute | **+0.0093** | **−0.0010** |

## I.5 Advanced graph learning: still **DEFERRED**

> **Superseded 2026-09-11 by §I.7: NO-GO on this benchmark.** Kept as the record
> of the reasoning at the time.

Unchanged in verdict, weakened in motivation. Network and cycle features are
worth +0.0093 under random folds — small, but positive in all four paired
folds — and −0.0010 under temporal shift, where the increment disappears. An earlier reading that network was "the family
that survives temporal shift" was an artifact of ablation cells A–D carrying no
`aml_context_v1`; it does not survive the fix described in
`docs/temporal_collapse.md` §6.

**The next experiment is not the 2-hop family and not a model.** It is a
corrected time-forward split: a calendar cutoff applied to every entity alike,
with the label defined from post-boundary flagged transactions, so train and test
differ in *time* rather than in *who is in them*. Measuring anything — including
2-hop structure — on the current split inherits its survivorship defect.

## I.6 What this benchmark still does not establish

* **Nothing about real-world performance.** IBM AMLworld is a simulator. Its
  1.23 % entity prevalence, its 17-day horizon and its typology mix are the
  simulator's choices.
* **No hyperparameter search.** Every candidate ran at one fixed configuration.
  The comparison is between families under equal treatment, not between tuned
  systems.
* **The entity label is derived** from transaction flags, so a launderer's
  counterparty is labelled positive. A transaction-level formulation — which is
  the granularity IBM actually publishes — is untested and is a different task
  from the one this project is built for.
* **One generator.** Agreement between IBM and SAML-D would mean more than
  either alone; they disagree sharply about how much structure encodes the
  label (probe AP 0.2098 against 0.9971), which is itself worth knowing.
* **Why the time-forward regime is so much harder** is unexplained. It is the
  largest unexplained effect in this document.

## I.7 Calendar-forward - the corrected time-forward regime

Run 2026-09-11. Full record: **`docs/calendar_forward_benchmark.md`**; generated
table: `docs/calendar_forward_table.md`; artifacts `outputs/experiments/ibm_cal_*.json`.

Every entity with history is scored at fixed calendar origins - train
2022-09-05, validation 2022-09-07, test 2022-09-09 - from the transactions
before each origin. The label is involvement in a flagged transaction in the two
days after the origin; the last window ends 2022-09-11, before the days 11-18
tail. Train and test differ in time, not in who is in them. Test origin:
514,022 entities, 1,374 positive (0.267 %).

| candidate | AP | lift | ROC | first involvement: AP (lift) |
| --- | ---: | ---: | ---: | ---: |
| `ibm_cal_heuristic_floor` - **label-free** | 0.0046 | 1.7x | 0.652 | 0.0031 (1.4x) |
| `ibm_cal_logistic_aml` | 0.0162 | 6.1x | 0.642 | 0.0036 (1.7x) |
| `ibm_cal_boosting_behaviour`, four seeds | 0.0133 ± 0.0035 | 5.0x | 0.697 | 0.0040 (1.8x) |
| `ibm_cal_boosting_graph`, four seeds | 0.0109 ± 0.0032 | 4.1x | 0.693 | 0.0039 (1.8x) |

The three IBM regimes answer three questions and are not merged:

| regime | target | features see the label period? | prevalence | boosting + graph |
| --- | --- | --- | ---: | --- |
| standard (§I.1) | whole-file involvement | yes | 1.234 % | 0.3827 AP, 31.0x, ROC 0.922 |
| survivorship time-forward (§I.3) | whole-file involvement | partly | 2.641 % | 0.1110 AP, 4.2x, ROC 0.619 |
| calendar-forward | 2-day window involvement | no | 0.267 % | 0.0109 ± 0.0032 AP, 4.1x, ROC 0.693 |

What it establishes:

* **The regime is sound.** 14 of 66 features past PSI 0.25 between the train
  and test snapshots (48 of 54 under survivorship); prevalence stable across
  roles; the information boundary audited on the rows used. The design was
  chosen on label-free measurements before any model ran.
* **Forecasting is weak.** 2.5-6.2x lift overall, 1.4-2.0x for the 82 % of
  positives with no earlier involvement, Brier skill +0.001 to +0.002. The
  standard benchmark's strength is retrospective: its features include the
  flagged transactions themselves. This partly answers §I.6's open question of
  why the time-forward regime is so much harder.
* **No family improves the top of the ranking over transaction features
  alone**, and the survivorship regime's temporal inversion (ROC 0.478) does not
  recur (0.676): it was the split.
* **Graph features add nothing.** Paired over four model seeds the increment is
  -0.0024 ± 0.0027 AP, negative in all four.

## I.8 Advanced graph learning: **NO-GO** on IBM HI-Small

The deferral in §I.5 waited on a corrected split. With it, the graph families'
marginal value is +0.0093 under the standard regime (where features see the
label period), -0.0010 under survivorship and -0.0024 ± 0.0027 under
calendar-forward. No edge-aware / GINE-style model is built. The 2-hop family is
off the critical path; if run, it runs under calendar-forward with at least four
model seeds. Recorded in `docs/decisions.md`.

## I.9 What this benchmark means for the product

Decided 2026-09-11 (`docs/system_architecture_decision.md`). The IBM results
define the ML engine's job: **retrospective entity risk ranking over an
analysis window** - the task measured at AP 0.3827 (standard regime). They do
not support forecasting as a product claim (§I.7), and they do not support a
GNN (§I.8). Graph structure moves to the investigation layer, where its value
is still to be measured (roadmap R2).

The standard regime's random folds keep training and scoring inside one
period. Whether a model fitted on one window ranks a *later* window as well is
unmeasured, and it is the next experiment (roadmap R1) because it decides what
the production model may claim.

## I.10 Cross-period detection transfer (R1)

Measured 2026-09-12; full record `docs/r1_detection_transfer.md`, generated table
`docs/r1_detection_transfer_table.md`, artifacts
`outputs/experiments/ibm_r1_*.json`.

The regime in §I.1 trains and scores inside one period. R1 asks whether a model
fitted on one completed 2-day detection window still ranks a *later* window,
applied unchanged - features and labels always from the window being scored.

| fold | train -> test | prevalence | floor AP | logistic AP | boosting AP (3 seeds) | AP retention vs same-period |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | W1 -> W2 | 0.621 % | 0.0153 | 0.0518 | **0.3145 ± 0.0050** | 0.975 |
| 1 | W2 -> W3 | 0.636 % | 0.0194 | 0.1239 | **0.3425 ± 0.0034** | 0.984 |
| 2 | W3 -> W4 | 0.394 % | 0.0125 | 0.0191 | **0.0396 ± 0.0080** | 0.111 |

* **Boosting beats logistic regression in every fold** (2.8x-6.1x), which the
  saturated SAML-D comparison could never show and the forecasting regime showed
  only weakly.
* **Transfer is near-complete between comparable windows and collapses across a
  dissimilar one**, tracking PSI on `aml_temporal_v1` (0.000 / 0.113 / 0.272).
* **The four regimes now measured**: within-period detection 0.3827;
  cross-period detection 0.31-0.34 (comparable windows) / 0.04 (dissimilar);
  survivorship time-forward 0.1110; forecasting 0.0133. They are four questions,
  not four attempts at one.
* **Graph families stay a wash** (paired AP −0.0031 / +0.0141 / −0.0052), so
  §I.8's NO-GO is unchanged.

Decision: **LIMITED TEMPORAL TRANSFER** (`docs/decisions.md`).
