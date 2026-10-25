# The Temporal Generalization Collapse

Why IBM HI-Small scores average precision **0.3827** under the standard
benchmark and **0.1110** under the time-forward one, and what that means.

This is an investigation record, not a model-expansion record. No GNN, no TDA,
no 2-hop features and no hyperparameter search were added.

Evidence: `outputs/experiments/ibm_*.json`, regenerated 2026-09-06 after the
`aml_context_v1` fix in §6. Pre-fix artifacts are retained under
`outputs/experiments/_prefix_context/`.

---

## 1. Reproduction, and what actually differs

Both artifacts were compared field by field. They agree on **dataset version,
label definition, feature families, feature count, model family,
hyperparameters, seed, threshold strategy and calibration**. They differ in
**two** things, not one:

| | standard | time-forward |
| --- | --- | --- |
| split | `stratified_kfold`, 4 folds | `time_forward_holdout` |
| **feature scope** | **`global`** | **`train_window`** |

The headline comparison therefore changed the split *and* the information the
features were allowed to see, at the same time. Any single-cause explanation of
the drop was unsupported before this investigation, because the experiment could
not distinguish the two.

The missing cell is `stratified_kfold` + `train_window`, which is **structurally
impossible**: a non-temporal split defines no training-window boundary, and the
runner raises rather than inventing one. The 2×2 is a 2×2 minus one cell by
construction, and the decomposition below reads around that.

## 2. The decomposition

Running the time-forward split at `global` scope isolates the two effects.

| cell | split | scope | AP | ROC | |
| --- | --- | --- | ---: | ---: | --- |
| standard | stratified | global | **0.3827** | 0.9215 | the benchmark |
| `ibm_tf_global_boosting_graph` | time-forward | global | **0.2453** | 0.7662 | new, diagnostic only |
| `ibm_tf_boosting_graph` | time-forward | train_window | **0.1110** | 0.6190 | the operational result |

```text
total drop             0.3827 -> 0.1110  = -0.2717   (-71 %)
  split effect         0.3827 -> 0.2453  = -0.1374    50.6 %
  feature-scope effect 0.2453 -> 0.1110  = -0.1343    49.4 %
```

**The two effects are almost exactly equal.** Neither the split nor the feature
scope dominates; the collapse needs both to be explained.

> These are the numbers **after** the `aml_context_v1` fix in §6. The figures
> that originally prompted this investigation - 0.2797 and 0.0620, a 78 % drop
> attributed 65/35 to split and scope - were computed with six feature columns
> silently dead. The qualitative findings survived the fix; the split/scope
> attribution did not, and moved from 65/35 to 51/49. Pre-fix artifacts are in
> `outputs/experiments/_prefix_context/`.

Both effects are real. Neither is a bug. The middle cell is **not** a legitimate
result to quote — see §4.

## 3. Feature-scope audit

All six families read `FeatureRequest.edge_table`, which the runner truncates to
the fold's training window under `train_window` scope. So **no feature can see a
transaction after the boundary when scoped**, and all six declare
`time_scopable=True`. Two columns additionally depend on a reference "now"
(`FeatureRequest.as_of`): `recency_days` and `tenure_days`.

The audit question that matters is not "does it use future rows" — under scoping
none of them do — but **"does its meaning change between regimes, and can it
encode position inside the 17-day file?"**

Measured, on the *same* test entities, average precision of each feature used
alone (test prevalence 0.0264):

| feature | whole population, global | TF test, global | TF test, train_window | collapse |
| --- | ---: | ---: | ---: | ---: |
| `aml_temporal_v1::recency_days` | 0.1404 | **0.3033** | 0.0351 | **0.2682** |
| `aml_temporal_v1::span_days` | 0.1321 | **0.2835** | 0.0370 | **0.2465** |
| `aml_temporal_v1::active_days` | 0.0630 | 0.1267 | 0.0438 | 0.0828 |
| `aml_temporal_v1::weekend_share` | 0.0230 | 0.1106 | 0.0533 | 0.0573 |
| `aml_temporal_v1::max_gap_days` | 0.0292 | 0.0848 | 0.0463 | 0.0385 |
| `aml_context_v1::n_currencies` | 0.0561 | 0.0814 | 0.0440 | 0.0374 |
| `aml_flow_v1::fan_product_norm` | 0.0368 | 0.0785 | 0.0554 | 0.0231 |
| `aml_cycle_v1::two_path_through` | 0.0404 | 0.0673 | 0.0505 | 0.0169 |
| `aml_network_v1::repeat_pair_mean` | — | 0.0529 | 0.0489 | 0.0040 |
| `aml_txn_v1::txn_total` | 0.0255 | 0.0367 | 0.0406 | −0.0039 |

Best single feature per family, TF test partition:

| family | global | train_window | best collapse |
| --- | ---: | ---: | ---: |
| `aml_temporal_v1` | 0.3033 | 0.0533 | **0.2499** |
| `aml_context_v1` | 0.0814 | 0.0440 | 0.0374 |
| `aml_flow_v1` | 0.0785 | 0.0554 | 0.0231 |
| `aml_cycle_v1` | 0.0673 | 0.0505 | 0.0169 |
| `aml_network_v1` | 0.0529 | 0.0489 | 0.0040 |
| `aml_txn_v1` | 0.0367 | 0.0406 | **−0.0039** |

**The collapse is concentrated almost entirely in `aml_temporal_v1`.** Network
and transaction features barely move; transaction features are marginally
*better* when scoped.

## 4. Why `recency_days` is the mechanism, and why the middle cell is not quotable

`recency_days` = reference-now − the entity's last activity. Under `global`
scope the reference is the end of the file. The time-forward split assigns
entities to partitions **by last activity**. So under global scope
`recency_days` is a monotone transform of the very variable that defines the
split:

| partition | min | median | max |
| --- | ---: | ---: | ---: |
| train | 8.900 | 10.661 | 17.679 |
| test | 0.000 | 8.003 | 8.410 |

Test max (8.410) < train min (8.900): **the ranges are disjoint.** The feature
is a partition indicator.

This produces the apparent paradox in §3: within the test partition
`recency_days` alone ranks at AP 0.3033 — higher than the entire time-forward
model, and higher than every other feature — yet the model that has it scores
only 0.2453. A model trained on `recency ∈ [8.9, 17.7]` and applied to
`recency ∈ [0, 8.4]` is extrapolating outside its training support. The
information is there and the model cannot use it.

**Consequence:** `ibm_tf_global_boosting_graph` (0.2453) is a diagnostic, not a
result. It measures a configuration in which a feature encodes the split. It is
recorded so the decomposition can be read, and it must not be quoted as a
time-forward score. `0.1110` is the operational number.

Under `train_window` scope the reference-now is the boundary and every entity is
truncated at the same instant, so `recency_days` measures "how long before the
boundary did this entity go quiet" — legitimate, and much weaker (0.0351).

## 5. Distribution shift

Population Stability Index between train and test, on the `train_window`
features the model actually consumes (PSI > 0.25 is conventionally a major
shift):

**48 of 54 defined features exceed PSI 0.25; 53 of 54 exceed 0.10.**

| feature | PSI | train median | test median |
| --- | ---: | ---: | ---: |
| `aml_temporal_v1::max_gap_days` | 3.63 | 2.72 | 1.29 |
| `aml_txn_v1::txn_total` | 3.61 | 3.00 | **37.00** |
| `aml_temporal_v1::active_days` | 3.29 | 2.00 | 9.00 |
| `aml_temporal_v1::daily_concentration` | 3.28 | 0.67 | 0.20 |
| `aml_network_v1::counterparty_diversity` | 2.90 | 0.20 | 0.09 |
| `aml_temporal_v1::mean_iat_hours` | 2.68 | 11.01 | 5.31 |
| `aml_network_v1::repeat_pair_mean` | 2.45 | 2.00 | 10.00 |

Train entities have a median of **3** transactions; test entities **37**. This
is not incidental — it is **manufactured by the split design**. Placing entities
by *last* activity means test entities are by construction the ones that were
still transacting at the end, which are the long-lived, high-volume ones. The
split selects on a variable correlated with almost every feature.

Prevalence shifts with it: train **0.78 %**, validation 1.18 %, test **2.64 %**.

## 6. A defect found during the audit: `aml_context_v1` was silently dead

`aml_context_v1` matches source columns by exact name. IBM spells them
`Payment Format`, `Payment Currency`, `Receiving Currency`; the family knew only
`Payment_type`, `Payment_currency`, `Received_currency` and similar. **It
matched nothing, and produced six all-NaN columns with no error.**

Every IBM run before 2026-09-06 therefore used **62 effective features, not
68**, and discarded `Payment Format` — one of this dataset's strongest label
correlates (ACH 0.746 % positive rate against Credit Card 0.016 %).

Fixed: the IBM spellings are registered, the family now records which source
columns it resolved in `FeatureRequest.cache["aml_context_v1_resolved"]`, and it
logs a warning when it resolves none. Pinned by
`tests/test_aml_features.py::TestContextFamily::test_ibm_column_spellings_resolve`
and `::test_unresolved_context_is_recorded_not_merely_silent`.

Every affected cell was re-run. Pre-fix artifacts are kept under
`outputs/experiments/_prefix_context/` rather than deleted.

## 7. Label drift — the generator changes behaviour

Transaction volume and positive rate by calendar day, from the source file:

| day | transactions | positives | positive rate |
| --- | ---: | ---: | ---: |
| 2022-09-01 | 1,114,921 | 322 | 0.029 % |
| 2022-09-02 | 754,449 | 408 | 0.054 % |
| 2022-09-03 | 207,382 | 391 | 0.189 % |
| 2022-09-05 | 482,650 | 471 | 0.098 % |
| 2022-09-09 | 654,467 | 514 | 0.079 % |
| 2022-09-10 | 208,325 | 442 | 0.212 % |
| **2022-09-11** | **396** | **232** | **58.6 %** |
| **2022-09-12** | **281** | **170** | **60.5 %** |
| **2022-09-13** | **184** | **106** | **57.6 %** |
| **2022-09-14** | 121 | 70 | 57.9 % |
| **2022-09-15** | 46 | 28 | 60.9 % |
| **2022-09-16** | 46 | 26 | 56.5 % |
| **2022-09-17** | 23 | 15 | 65.2 % |
| **2022-09-18** | 11 | 8 | 72.7 % |

Two findings:

1. **The dataset is ~10 days, not 17.** 99.98 % of transactions fall in days
   1–10. The remaining eight days hold 1,108 transactions in total.
2. **The tail is a different generative mechanism.** Days 11–18 run at a
   **57–73 % positive rate** — roughly 500× the body of the file — and are
   **100 % ACH**, where the body has five payment formats among its positives.
   This is the simulator emitting the completion of long laundering chains after
   normal traffic has stopped.

**26.3 % of the test partition's positives (715 of 2,723)** draw their label at
least partly from that tail. The label-generating process is materially
different in the period the time-forward split evaluates on.

## 8. Entity overlap — the benchmark is transductive

Across the training boundary (2022-09-09 18:43):

| | |
| --- | ---: |
| entities active before the boundary | 514,469 |
| entities active after the boundary | 206,085 |
| ...of which also seen before (**transductive**) | 205,466 (**99.7 %**) |
| ...of which new after (**inductive**) | 619 (**0.3 %**) |
| post-boundary transactions with both endpoints previously seen | 348,304 / 349,241 (99.7 %) |
| test entities with pre-boundary history | 102,731 / 103,101 (99.6 %) |
| test **positives** with pre-boundary history | 2,703 / 2,723 (99.3 %) |

**The collapse is not a cold-start problem.** 99.6 % of test entities are
scoreable from pre-boundary history, and 93.1 % of all transactions are visible
before the boundary. Only 6.9 % of the edge table is withheld, and that removes
71 % of the model's average precision.

The benchmark is **transductive in entity terms and temporal in behaviour
terms**: the same accounts, later behaviour.

## 9. Time-forward ablation — the family ordering inverts

Same model, same split, same seed; only the feature families change.

| cell | families | standard AP | time-forward AP | time-forward ROC |
| --- | --- | ---: | ---: | ---: |
| A | `aml_txn_v1` | 0.0589 | 0.0363 | 0.5334 |
| B | A + `aml_temporal_v1` | **0.2222** | **0.0295** | **0.4783** |
| C | A + `aml_flow_v1` | 0.1410 | 0.0553 | 0.6074 |
| D | A + `aml_network_v1` + `aml_cycle_v1` | 0.1193 | **0.0618** | **0.6228** |
| E | all six (incl. context) | **0.3827** | **0.1110** | 0.6190 |

Cells A–D carry no `aml_context_v1` by design, so they were unaffected by the
§6 fix and were not re-run; E does carry it and was.

**Within the A–D ladder the ordering completely reverses.**

* Under the standard split, temporal is the strongest addition (+0.1633 over A)
  and network the weakest (+0.0604).
* Under time-forward, **temporal is actively harmful**: B scores *below* A
  (0.0295 vs 0.0363) with **ROC 0.4783 — worse than chance**. Network becomes
  the strongest addition (+0.0255 over A) and reaches ROC 0.6228, the best of any
  A–D cell.

Adding the temporal family under shift produces a ranking worse than random.
That is the sharpest available evidence that the standard headline is carried by
temporal-position information which does not merely fail to transfer — it
inverts.

**But network does not survive as a *marginal* contributor.** Comparing the two
full models, which differ only in whether the graph families are present:

| | standard | time-forward |
| --- | ---: | ---: |
| `boosting_behaviour` (no graph, with context) | 0.3734 | **0.1120** |
| `boosting_graph` (all six) | 0.3827 | **0.1110** |
| marginal value of network + cycle | **+0.0093** | **−0.0010** |

Under time-forward the graph families are worth **nothing** on top of
transaction, temporal, flow and context — very slightly negative. The time-forward regime is a single holdout, so there
is no fold dispersion to compare against; −0.0010 is simply negligible. D looked like "the surviving family" only because cells A–D lack
context; once context is present it supplies what network was standing in for.

This corrects a conclusion drawn before the §6 fix. Network structure is *not*
the family that survives temporal shift; **context and flow are**, and the graph
adds nothing measurable in either regime.

## 10. Baseline comparison under the operational regime

| candidate | AP | lift | ROC | vs floor |
| --- | ---: | ---: | ---: | --- |
| label-free floor | 0.0389 | 1.47× | 0.5519 | — |
| logistic regression | 0.0525 | 1.99× | 0.5957 | ×1.35 |
| boosting, no graph | **0.1120** | **4.24×** | 0.5920 | ×2.88 |
| boosting + graph | 0.1110 | 4.20× | 0.6190 | ×2.85 |

Read with prevalence (0.0264), not in isolation. **Supervised learning does
survive the shift**, at roughly 2.9× the label-free floor, but the margin is far
narrower than under random folds.

For contrast, under the standard split the same four are 0.0390 / 0.1710 /
0.3734 / 0.3827 — boosting beats the floor by 9.8× there against 2.9× here.

(Before the §6 context fix, logistic scored 0.0346 under this regime — *below*
the floor. That is no longer true: with its context columns restored it reaches
0.0525. The pre-fix reading would have supported a much stronger claim about
supervised failure than the evidence now supports.)

---

## 11. Root-cause classification

The collapse is **MIXED**, with four contributing causes and one explicitly
ruled out. In order of measured contribution:

### (a) SPLIT METHODOLOGY — the largest single contributor

**Evidence.** Holding scope constant at `global`, changing only the split costs
0.3827 → 0.2453 (50.6 % of the total drop). The split places entities by *last
activity*, which by construction makes the partitions differ in every
activity-correlated feature: train median 3 transactions against test median 37,
48 of 54 features with PSI > 0.25, and prevalence 0.78 % against 2.64 %.

This is a **survivorship split**, not a forecasting split. It asks "given
accounts that went quiet early, score accounts that were still active at the
end", which is a population comparison rather than a temporal prediction.

### (b) DISTRIBUTION SHIFT — real, and largely induced by (a)

**Evidence.** The PSI table in §5. The shift is not incidental drift; it follows
mechanically from selecting the partition on an activity-correlated variable.

### (c) LABEL / GENERATOR DRIFT — real and substantial

**Evidence.** §7. Days 11–18 run at 57–73 % positive against 0.03–0.21 % in the
body, and are 100 % ACH. 26.3 % of test positives draw their label from that
tail. The mechanism generating late positives is not the mechanism generating
early ones.

### (d) FEATURE SCOPE — real, correct, and correctly costly

**Evidence.** Holding the split constant, `global` → `train_window` costs
0.2453 → 0.1110 (49.4 %). This is **not leakage in the sense of a bug**: the
framework declared both regimes and the runner enforced the window. It is the
legitimate price of removing whole-file temporal position. The
`recency_days`/`span_days` measurements in §3–4 identify exactly which
information is removed.

**One genuine methodological fault** sits here: under `global` scope the
time-forward split's own selection variable is reproducible as a feature
(`recency_days`, train/test ranges disjoint). That combination is
scientifically void and is now documented as such.

### (e) GENUINE TEMPORAL GENERALIZATION FAILURE — present, and smaller than it looks

**Evidence.** With everything scoped correctly, boosting + graph reaches
0.1110 at 4.20× prevalence against a floor of 0.0389 at 1.47× — 2.85× the
floor. The model does generalize, but far more weakly than under random folds
(9.8× the floor there). But a large part of what looks like "failure" is (a)–(c):
the model is being asked to transfer between two populations that the split
itself made different, with a label process that changed.

### RULED OUT: ENTITY-COVERAGE ISSUE

**Evidence.** §8. 99.6 % of test entities and 99.3 % of test positives have
pre-boundary history; only 0.3 % of post-boundary entities are new; 93.1 % of
transactions are visible before the boundary. The collapse is not caused by
unscoreable cold-start entities.

---

## 12. Scientific interpretation

**Both numbers are real and they answer different questions.** 0.3827 is a valid
transductive result: same population, whole-file features, random folds. 0.1110
is a valid result for a specific and unusual operational question: score
early-quiet accounts, evaluate on late-active ones, using only pre-boundary
history, against a label process that shifts.

**Neither is "the" performance of the system,** and the 71 % drop overstates the
case for pure temporal failure, because half of it comes from a split that
manufactures its own distribution shift.

**The most transferable finding is the ablation inversion (§9).** Temporal
features carry the standard benchmark (+0.1633 over transaction-only) and invert
under shift (ROC 0.4783, worse than chance). Feature families that look
strongest under random folds are not the ones that survive time — and the graph
families, which look like the survivors in the A–D ladder, contribute **−0.0010**
once context is present. Nothing in this dataset makes the case that relational
structure is what carries temporal generalization.

**The benchmark remains usable, with its regimes correctly labelled.** It is not
disqualified: labels are not structurally trivial (probe AP 0.2098), there are
6,357 positives, and the framework reports both regimes separately. What has
changed is that the *time-forward* configuration is now known to be a
survivorship split over a drifting label process, and must be described that way
rather than as "the operational number".

---

## 13. Recommended next experiment

> **Done, 2026-09-11.** The calendar-cutoff split described below was built
> (`src/experiments/calendar.py`) and run; design, results and the resulting
> graph decision are in `docs/calendar_forward_benchmark.md`. This section is
> kept as the record of what was prescribed.

**Build a time-forward split that does not select on activity, and re-measure.**

The current `time_forward_holdout` places *entities* by their own last activity.
That is what manufactures the population difference. The scientifically cleaner
alternative is a **calendar cutoff applied to every entity alike**:

* score **all** entities as of a fixed boundary using only pre-boundary
  transactions;
* define the label from post-boundary flagged transactions only;
* keep the population fixed, so train and test differ in *time*, not in *who is
  in them*.

That isolates genuine temporal generalization from survivorship, and it is a
split strategy plus a label rule — no new features, no new model. It also
requires deciding what to do with the days 11–18 tail, which at 57–73 % positive
would otherwise dominate any post-boundary label.

**This should be run before the 2-hop experiment**, because the 2-hop question
is "does multi-hop structure add value", and §9 has just shown that the answer
depends on which regime you ask it in: added last, network + cycle are worth
+0.0093 under random folds and −0.0010 under shift. A 2-hop result measured on
the current time-forward split would inherit that split's survivorship defect.

### Status of the previously proposed 2-hop experiment

**Still deferred, and this investigation weakened rather than strengthened its
motivation.** Added last, network + cycle are worth **+0.0093** under the
standard split — small but positive in all four paired folds — and **−0.0010**
under time-forward, a single holdout with no dispersion estimate. The increment
exists where whole-file temporal position is available and vanishes where it is
not. The earlier reading that network was "the
surviving family" was an artifact of ablation cells A–D carrying no context; it
does not survive the §6 fix.

The 2-hop experiment remains the right *test* of the graph question, because
1-hop tabular features may simply be the wrong probe for multi-hop structure.
But it should be run only after the split is corrected, and its prior should now
be that the answer is "no measurable gain".

Order: **(1)** corrected time-forward split; **(2)** 2-hop family measured under
both regimes; **(3)** only then a GO/NO-GO on edge-aware message passing.

No GNN, no TDA and no 2-hop features were added in this phase.
