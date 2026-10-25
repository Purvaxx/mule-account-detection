# R1: Cross-Period Detection Transfer (IBM HI-Small)

Does a detection model trained on one completed analysis period still rank
entities usefully in a **later** completed period, applied unchanged?

**This is not forecasting.** `docs/calendar_forward_benchmark.md` measured
forecasting (features before an origin, labels after it) and found it weak. R1
asks a different question, with different semantics, and its numbers are not
comparable to that benchmark's.

```text
task            retrospective detection transfer
feature window  the detection window [A, B)
label window    the same detection window [A, B)
score point     B  ("given this completed period, rank entities to investigate")
transfer test   fit on [A1, B1), freeze, apply unchanged to [A2, B2) with B1 <= A2
```

Status: **pre-registered 2026-09-12, before any model was fitted.** Sections 1-8
were written first; §9 onward record what came back. IBM AMLworld is
**synthetic**: any result here is transfer *within this simulator*, never
evidence about real-world temporal generalisation.

---

## 1. Hypothesis

A model fitted on the behaviour of one completed 2-day detection window retains
useful ranking performance when applied, unchanged, to the next 2-day window -
that is, it has learned suspicious *behaviour* rather than properties of one
temporal slice.

## 2. Feasibility audit (data only, ran first)

`outputs/audits/r1_window_audit_ibm_aml_hi_small.json`. No model and no score,
so no window choice here can be tuned toward a benchmark number.

| window | span | weekdays | transactions | flagged txns | eligible entities | positive entities | prevalence |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 2022-09-03 .. 09-05 | Sat, Sun | 414,812 | 798 | 162,388 | 1,217 | 0.749 % |
| W2 | 2022-09-05 .. 09-07 | Mon, Tue | 964,739 | 1,002 | 234,947 | 1,459 | 0.621 % |
| W3 | 2022-09-07 .. 09-09 | Wed, Thu | 965,524 | 1,036 | 235,433 | 1,497 | 0.636 % |
| W4 | 2022-09-09 .. 09-11 | Fri, Sat | 862,792 | 956 | 354,447 | 1,397 | 0.394 % |

Findings that decide the design:

1. **Enough positives.** 1,217-1,497 positive entities per window, from
   798-1,036 flagged transactions.
2. **No unscoreable positives.** A flagged transaction inside a window makes
   both its endpoints active in that window, so every window positive is
   eligible: `positives_not_eligible = 0` in all four windows. Cold start
   cannot bias this design.
3. **The task is not persistence.** First-time positives - positive in the test
   window and in no earlier window - are 1,287 / 1,261 / 1,158 of the test
   positives in folds 1-3.
4. **Entity overlap varies and is reported as strata.** Seen share of the test
   window: 61.4 % (W1->W2), 92.3 % (W2->W3), 62.7 % (W3->W4).
5. **Prevalence shifts about 2x across windows** (0.394 % - 0.749 %). Real, and
   reported beside every number.
6. **Flagged payment-format mix is stable** (ACH 85.6-90.9 %), so these windows
   are not different generative regimes the way the tail is.
7. **Excluded, with reasons:**
   * **Sep 11-18 tail** - 1,108 transactions, 715 of 716 active entities
     positive (99.9 %), 100 % ACH. Normal traffic has stopped and the simulator
     is completing chains. Never a train or test window.
   * **Sep 1 burn-in** - 1,114,921 transactions (2.3x a normal weekday) and
     437,073 active entities on day one.
   * **Sep 2** - excluded so that four *equal* 2-day windows tile
     Sep 3 - Sep 11 exactly, anchored on the tail boundary.
8. **An alternative 3-day design exists** (Sep 2-11; 3 windows, 2 folds;
   1,782 / 2,079 / 2,134 positives) and is measured in the same audit. The 2-day
   design is chosen **because it yields three chronological folds rather than
   two**, decided before any model ran. The 3-day design is not run, so this is
   not a search over designs.

**Verdict: R1 is feasible.** Three folds, ~1,400 test positives each, no
cold-start bias, no tail contamination.

## 3. Temporal design

```text
Fold 1:  train W1 [09-03, 09-05)  ->  test W2 [09-05, 09-07)
Fold 2:  train W2 [09-05, 09-07)  ->  test W3 [09-07, 09-09)
Fold 3:  train W3 [09-07, 09-09)  ->  test W4 [09-09, 09-11)
```

Windows are equal length, non-overlapping and contiguous:
`train.end == test.start`, so no calendar instant is in both. Train/test
assignment is by **calendar time only** - never by an entity's own activity,
which is what made the earlier survivorship split misleading
(`docs/temporal_collapse.md`).

Inside the training window only, entities are split **stratified by label** into
fit 60 % / validation 20 % / in-period reference 20 % (seeded). Validation
supplies the threshold. The in-period reference partition is
the same-period yardstick against which cross-period degradation is measured:
it is scored by the same frozen model, on the same window it was fitted on, and
it never took part in fitting.

## 4. Labels

`entity_label = 1` in window `[A, B)` if and only if the entity is payer or
payee of at least one transaction whose published `Is Laundering` flag is 1 and
whose timestamp lies in `[A, B)`; otherwise 0. Derived from
`ExperimentDataset.transaction_labels`, which sits beside the edge table and is
invisible to feature families. No typology file, no labels from outside the
window, no label-derived features.

## 5. Features

`feature_window = detection_window`: every feature is computed from the
transactions dated inside `[A, B)`, with `as_of = B` for the two
reference-dependent columns (`recency_days`, `tenure_days`). No history before
`A` is used by either period - the cleanest window-local formulation, applied
identically to train and test.

Families: the six `aml_*` families (`aml_txn_v1`, `aml_temporal_v1`,
`aml_flow_v1`, `aml_context_v1`, `aml_network_v1`, `aml_cycle_v1`) - the
representation the IBM benchmarks already use.

## 6. Candidates

| role | candidate |
| --- | --- |
| A - label-free floor | `structural_heuristic` (reads no labels) |
| B - simple model | `logistic_regression`, same features and protocol |
| C - main candidate | `gradient_boosting` without the graph families |
| D - graph variant | `gradient_boosting` with `aml_network_v1` + `aml_cycle_v1` |

The legacy GraphSAGE checkpoint is **not** used: it is a regression reference on
a 4-positive dataset, it consumes a different feature contract, and it cannot be
fitted per window under this protocol. No GNN is built for R1.

Seeds 42, 1337 and 7 for the boosting candidates, whose internal early-stopping
split is seed-dependent. The floor reads no labels; logistic regression is
deterministic given its data.

## 7. Leakage controls

| control | how it is enforced |
| --- | --- |
| no calendar overlap | plan validation: windows strictly increasing, non-overlapping, `train.end <= test.start` |
| features never cross the window | the rows actually handed to the families are re-checked - min date >= A, max date < B, no undated rows - and a violation is a hard error |
| test labels touched once | labels are read only to compute metrics, after the frozen model has scored |
| threshold and calibration | fitted on the training window's validation partition only |
| no test-period tuning | one fixed hyperparameter set, seeds declared in advance, no feature selection |
| no survivorship | partitioning is by calendar time, never by an entity's last activity |
| model frozen | the fitted candidate is wrapped in a `CandidateRiskEngine` and applied unchanged |

## 8. Metrics and pre-registered decision criteria

Per fold: average precision, ROC-AUC, precision@K and recall@K, and alerts /
true positives / false positives per true positive at alert budgets
**0.1 %, 0.5 %, 1 %, 2 %, 5 %**, plus lift over prevalence and calibration where
the score is probability-shaped. Strata: overall, seen entities, new entities,
first-time positives. Drift: PSI per feature between the fit partition and the
test window (10 quantile bins fitted on train, undefined excluded) - the same
methodology as `calendar_design_audit.py`.

Degradation is reported as **retention** = cross-period value / in-period value,
for AP, lift, precision@1 % and recall@1 %.

The thresholds below are **engineering decision criteria**, fixed now, not
scientific constants:

* **Gate (validity).** Leakage tests pass; every test window has at least 500
  positive entities; no window overlaps the tail.
* **STRONG TEMPORAL TRANSFER** - in **3 of 3** folds: cross-period AP >= 3x the
  label-free floor on the same window, **and** AP retention >= 0.5, **and**
  precision@1 % >= 5x prevalence; with seed standard deviation <= 25 % of the
  mean cross-period AP.
* **LIMITED TEMPORAL TRANSFER** - in **at least 2 of 3** folds: cross-period AP
  >= 2x the floor **and** AP retention >= 0.2, but STRONG not met.
* **NO EVIDENCE OF TEMPORAL TRANSFER** - in at least 2 of 3 folds: cross-period
  AP < 1.5x the floor, or lift at the 1 % budget < 2x.
* **INCONCLUSIVE** - the gate fails, or the folds contradict each other with no
  stated cause.

No fold may be discarded. No date may be changed after seeing a result.

### Amendment, 2026-09-12 (during implementation, before any result existed)

**No calibrator is fitted.** The frozen model reaches the later window through
`CandidateRiskEngine`, which scores the model itself; a calibrator fitted on the
training window's validation partition would not travel with it, and applying
one afterwards would break the contract rule that `decision` equals
`risk_score >= threshold`. R1 therefore sets `calibration.method = "none"`,
records `score_kind = model_probability` (not `calibrated_probability`), and
reports the Brier score and expected calibration error **of the raw model
probability**. The runner refuses a configuration that asks for a fitted
calibrator rather than silently dropping it. This changes no decision criterion:
thresholds are alert budgets, which are rank-based, and the previous phase
measured Platt scaling as adding no information (Brier skill +0.001 to +0.002).

---

## 9. Results

Ran 2026-09-12. Artifacts: `outputs/experiments/ibm_r1_*.json` (8 runs: floor,
logistic regression, and boosting with and without graph families at seeds 42,
1337 and 7). Every run reported **zero** leakage findings above `info`.

Dataset: `ibm_aml_hi_small`, SHA-256
`b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`
(`data/manifests/ibm_aml_hi_small.json`), 5,078,345 transactions, 515,088
entities, 2022-09-01 .. 2022-09-18. **Synthetic.**

### 9.1 Cross-period transfer, per fold

Boosting without graph features, mean of three seeds; floor and logistic are
deterministic single runs. "In-period" is the same frozen model on held-out
entities of the window it was fitted on.

| fold | train -> test | test entities | positives | prevalence | floor AP | logistic AP | **boosting AP** | boosting ROC | in-period AP | **AP retention** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | W1 -> W2 | 234,947 | 1,459 | 0.621 % | 0.0153 | 0.0518 | **0.3145 ± 0.0050** | 0.913 | 0.3251 | **0.975** |
| 1 | W2 -> W3 | 235,433 | 1,497 | 0.636 % | 0.0194 | 0.1239 | **0.3425 ± 0.0034** | 0.922 | 0.3482 | **0.984** |
| 2 | W3 -> W4 | 354,447 | 1,397 | 0.394 % | 0.0125 | 0.0191 | **0.0396 ± 0.0080** | 0.750 | 0.3549 | **0.111** |

Against the label-free floor: **20.5x, 17.6x, 3.2x**. Boosting beats logistic
regression in every fold, by 2.8x to 6.1x.

### 9.2 Operating points (boosting without graph, seed 42)

| fold | P@0.1 % | P@1 % | R@1 % | P@5 % | alerts at the frozen threshold | TP | FP | FP per TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **0.987** | 0.2285 | 0.368 | 0.0741 | 1,467 (0.6 %) | 478 | 989 | **2.07** |
| 1 | **0.987** | 0.2442 | 0.384 | 0.0877 | 2,397 (1.0 %) | 578 | 1,819 | **3.15** |
| 2 | 0.127 | 0.0759 | 0.193 | 0.0234 | **42,415 (12.0 %)** | 567 | 41,848 | **73.8** |

The floor reaches P@1 % of 0.031 / 0.046 / 0.024 for comparison.

**The operating point transfers worse than the ranking.** In fold 2 ROC falls
only 0.92 -> 0.75 while the frozen threshold admits 12 % of the population
instead of ~1 %, and false positives per true positive go from 2-3 to 74. The
Brier score of the raw model probability moves the same way: 0.068-0.072 in
folds 0-1, 0.377 in fold 2.

### 9.3 Entity strata (boosting without graph, seed 42; seeds agree)

| fold | seen entities AP | new entities AP | first-time-positive population AP |
| --- | ---: | ---: | ---: |
| 0 | 0.3432 (n=144,331) | 0.2913 (n=90,616) | 0.2640 |
| 1 | 0.3459 (n=217,267) | 0.3492 (n=18,166) | 0.2859 |
| 2 | 0.0255 (n=222,224) | 0.0582 (n=132,223) | 0.0207 |

**Transfer is not entity memorisation.** New entities - absent from the training
window entirely - score about as well as seen ones (fold 1: 0.349 vs 0.346), and
the population whose positives were never flagged in an earlier window still
reaches AP 0.26-0.29 in folds 0-1.

### 9.4 Drift (PSI, fit partition vs test window)

| fold | features PSI > 0.25 | median PSI | `aml_temporal_v1` family median | largest shifts |
| --- | ---: | ---: | ---: | --- |
| 0 | 6 of 48 | 0.099 | 0.113 | `rapid_turnover_share` 0.38, `inflow_outflow_ratio` 0.36, `txn_in` 0.31 |
| 1 | **0 of 48** | **0.000** | 0.000 | nothing above 0.0002 |
| 2 | **16 of 48** | 0.118 | **0.272** | `daily_concentration` 2.49, `recency_days` 1.52, `amount_top1_share` 1.17, `span_days` 1.02, `second_half_share` 0.96 |

With the graph families present, fold 0 also shows network drift
(`component_edge_density` 2.02, `component_size` 1.99) - and transfers fine
anyway.

**Drift and retention line up.** Zero drift (fold 1) gives retention 0.98;
moderate drift (fold 0) still gives 0.975; heavy *temporal-shape* drift (fold 2)
gives 0.11.

### 9.5 Graph features

Paired per fold, mean over three seeds (with graph minus without):

| fold | without graph | with graph | delta AP | delta ROC |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.3145 | 0.3114 | −0.0031 | +0.024 |
| 1 | 0.3425 | 0.3566 | +0.0141 | +0.030 |
| 2 | 0.0396 | 0.0344 | −0.0052 | −0.026 |

AP is a wash (mean +0.0019, positive in 1 of 3 folds). ROC improves by
0.02-0.03 in the two folds that transfer and falls 0.026 in the one that does
not. **This does not reverse the graph NO-GO** (`docs/model_selection.md` §I.8):
the criterion was a consistent paired AP gain exceeding the seed spread, and
this is not one. The ROC pattern is recorded as an observation, not a
justification for message passing.

### 9.6 Why fold 2 collapses

Measured, in order of evidential strength:

1. **Window composition differs.** W3 is Wed+Thu (482,751 + 482,773
   transactions); W4 is Fri+Sat (654,467 + 208,325). W4 therefore holds 354,447
   eligible entities against 235,433, and its prevalence is 0.394 % against
   0.636 %. The features that drift hardest are exactly the window-shape ones
   (`daily_concentration`, `span_days`, `second_half_share`, `recency_days`).
2. **Ruled out: thinner evidence per positive.** Flagged transactions per
   positive entity are the same in every window - median 1, mean 1.31 / 1.37 /
   1.38 / 1.37 - so W4's positives are not harder because their footprint is
   smaller. Median flagged amount is also stable (7,789 / 7,885 / 8,632 /
   8,871).
3. **Evidence consistent with generator wind-down adjacency.** 12.4 % of W4's
   positives (173 of 1,397) also transact in the excluded Sep 11-18 tail,
   against 2.7 %, 2.9 % and 5.8 % for W1-W3. W4 is the window whose laundering
   continues past the body, and the tail is where the simulator stops normal
   traffic. This is an association, not an established mechanism.
4. **Not prevalence alone.** Lift accounts for prevalence, and it falls from
   ~51-54x to 8-12x; AP retention 0.11 is not explained by the 1.6x prevalence
   change.
5. **Not leakage.** Every fold's boundary audit passed, the in-period reference
   reaches 0.355 in fold 2 (the model learns W3 as well as ever), and only
   transfer fails.

### 9.7 Verdict against the pre-registered criteria

Gate: leakage checks pass; every test window has 1,397-1,497 positives
(>= 500); no window overlaps the tail. **Gate passed.**

| criterion | fold 0 | fold 1 | fold 2 |
| --- | --- | --- | --- |
| AP >= 3x floor | yes (20.5x) | yes (17.6x) | yes (3.2x) |
| AP retention >= 0.5 | yes (0.975) | yes (0.984) | **no (0.111)** |
| P@1 % >= 5x prevalence | yes (36.6x) | yes (38.9x) | yes (19.2x) |
| seed sd <= 25 % of mean | yes (1.6 %) | yes (1.0 %) | yes (20.2 %) |

STRONG requires all three folds and fails on fold 2's retention. LIMITED
requires at least two folds at AP >= 2x floor and retention >= 0.2: folds 0 and
1 satisfy it comfortably.

## **Decision: LIMITED TEMPORAL TRANSFER**

A detection model fitted on one completed 2-day window transfers **almost intact
to a comparable next window** (AP retention 0.975 and 0.984, 17-20x the
label-free floor, FP/TP of 2-3 at its frozen threshold) and **collapses when the
next window's composition differs materially** (retention 0.11, FP/TP 74). The
capability is real and is not entity memorisation; its portability is
conditional on the two periods being comparable, and the frozen *threshold* is
markedly less portable than the *ranking*.

## 10. Limitations

* **Synthetic.** IBM AMLworld. This is transfer inside a simulator.
* **Same-window semantics, by design.** Features summarise the transactions the
  label is derived from, which is what retrospective detection means here - and
  it is why P@0.1 % reaches 0.99 in folds 0-1. R1 measures whether recognising a
  completed laundering footprint transfers across periods; it does not establish
  that the footprint is evidence independent of the label.
* **Three folds, eight days.** One design fits; W1 is weekend-only and W4
  borders the excluded tail. No fold is independent of the others' windows
  (consecutive windows share entities).
* **Two-day windows only.** Window length was not varied; the 3-day alternative
  was measured for feasibility but not run.
* **One hyperparameter configuration**, no tuning, as pre-registered.

## 11. Implications for the architecture

The ML + investigation-intelligence hybrid stands, with two changes to what may
be claimed and one new production requirement:

1. **The retrospective detection claim now has cross-period evidence** - in 2 of
   3 folds, with a measured failure mode in the third. It is no longer only a
   within-period random-fold number.
2. **A frozen threshold is not portable.** Production must re-derive the alert
   budget per analysis window; carrying a threshold across windows moved the
   alert rate from 1 % to 12 %.
3. **Temporal-shape drift is a gating signal, not a nice-to-have.** PSI on
   `aml_temporal_v1` between the fitted window and the scored window separated
   the two folds that transferred from the one that did not. Drift monitoring
   moves from "valuable later" to a precondition for applying a frozen model to
   a new window.

## 12. Next experiment (single highest value)

**R6, window-relative temporal features (`aml_temporal_v2`).** R1 localised the
failure to window-shape columns: `daily_concentration` (PSI 2.49), `recency_days`
(1.52), `span_days` (1.02), `second_half_share` (0.96). Express them relative to
the window (fractions of window length, activity concentration normalised by the
window's own volume profile) and re-run R1 unchanged.

* **Hypothesis:** the fold-2 collapse is caused by window-composition encoding,
  not by a change in laundering behaviour.
* **Success:** fold 2 AP retention >= 0.5 while folds 0-1 stay >= 0.9.
* **Failure:** fold 2 retention stays < 0.3, which would point to the tail
  adjacency of W4 instead and make window comparability a data requirement
  rather than a feature-design problem.
* **Cost:** low - one new feature family version, same protocol, same eight
  configs.
* **Production impact:** yes, it would change the feature contract version.

> **Outcome, added 2026-09-12 (this section is left as it was pre-registered).**
> R6 ran and **failed its own success condition**: W3->W4 retention reached only
> **0.192** against the 0.50 target, and W1->W2 *fell* from 0.975 to **0.881**.
> The transformation compressed the extreme drift this section pointed at
> (largest temporal PSI 2.4934 -> 0.9122) but not the typical drift (family
> median 0.2722 -> 0.2717, still 16/48 past 0.25), so the stated failure branch
> applies: **window comparability is a data requirement, not a feature-design
> problem.** `aml_temporal_v1` is retained and the feature contract is
> unchanged. Record: `docs/r6_temporal_v2.md`.
