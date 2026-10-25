# R6: Window-Relative Temporal Features - a decision gate

Can window-relative temporal features reduce the cross-period transfer failure
R1 found, without sacrificing the transfer that already works?

This is a **controlled ablation and a decision gate**, not a research programme.
Exactly one thing changes from R1: the temporal feature family.

```text
R1   aml_temporal_v1   ->   R6   aml_temporal_v2
```

Everything else is held fixed: the IBM HI-Small dataset and its SHA-256
(`b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`,
475,664,283 bytes, `data/manifests/ibm_aml_hi_small.json`; both the R1 and the R6
artifacts carry the same run-time fingerprint `size475664283-c9a6aae1`), the
three folds W1->W2 / W2->W3 / W3->W4, detection-window semantics, the entity
label and eligibility rules, calendar separation, the boosting model family
without graph features, its hyperparameters, preprocessing, seeds 42 / 1337 / 7,
the 1 % alert-budget threshold chosen on the training window's validation
partition, the metric set, the leakage controls and the production
`RiskAssessment` contract.

Status: **pre-registered 2026-09-12, before `aml_temporal_v2` was run.**
Sections 1-6 were written first; §7 onward record what came back.

Substitution note: `docs/ML_STRATEGY.md` does not exist in this repository. The
strategy document is `docs/system_architecture_decision.md`, and R1's record is
`docs/r1_detection_transfer.md`.

---

## 1. What R1 established

* W1->W2 AP `0.3145 ± 0.0050`, retention **0.975**.
* W2->W3 AP `0.3425 ± 0.0034`, retention **0.984**; **zero** features past
  PSI 0.25.
* W3->W4 AP `0.0396 ± 0.0080`, retention **0.111**; **16 of 48** features past
  PSI 0.25, `aml_temporal_v1` family median PSI **0.272**.
* The largest drifting quantities were `daily_concentration` (PSI 2.49),
  `recency_days` (1.52), `span_days` (1.02) and `second_half_share` (0.96).
* Verdict: **LIMITED TEMPORAL TRANSFER**.

## 2. Hypothesis

The W3->W4 collapse is driven by temporal features that encode **where the
window sits in the calendar and how its volume is distributed across days**,
rather than by a change in laundering behaviour. Expressing temporal position
and dispersion relative to the window's own activity should reduce that drift
and improve W3->W4 retention without harming W1->W2 or W2->W3.

W3 is Wed+Thu (482,751 + 482,773 transactions); W4 is Fri+Sat (654,467 +
208,325). Any feature that counts calendar days, or measures position against a
wall clock, inherits that asymmetry.

## 3. The design principle

> **Temporal position and dispersion are measured on the window's own activity
> clock. Only quantities whose behavioural meaning is intrinsically wall-clock
> stay in wall-clock units.**

The activity clock of a detection window is

```text
tau(t) = (number of the window's transactions dated strictly before t)
         / (number of the window's transactions)          in [0, 1)
```

computed from the transactions **inside the window only**. Under tau, "half way
through the window" means half of the window's activity has happened, not
12 hours of wall clock - so a window whose volume is concentrated on one of its
two days is measured on the same footing as a window whose volume is even.

Two consequences worth stating:

* `aml_temporal_v2` does **not** use `FeatureRequest.as_of` at all. The end of
  the window is `tau = 1` by construction, so the reference-"now" dependence
  that v1 carries in `recency_days` / `tenure_days` disappears.
* The clock is a property of the window's whole transaction set, so a v2 feature
  depends on other entities' activity in the same window - as
  `aml_network_v1`'s component features already do. It uses no labels, no rows
  outside the window, and nothing about any other window.

## 4. Exact v1 -> v2 changes

15 columns, against v1's 16. Unchanged columns keep their names; every redefined
column is renamed, so a v1 and a v2 number can never be read as the same
quantity.

| v2 column | v1 counterpart | v1 definition | v2 definition | why it changes | expected invariance |
| --- | --- | --- | --- | --- | --- |
| `recency_activity` | `recency_days` | `as_of` − last event, in days | `1 − tau(last event)`: share of window activity after the entity's last transaction | v1 is bounded by window length and by where the window's volume sits; it was the second-largest drifter (PSI 1.52) | invariant to window length, to volume distribution across days, and to `as_of` |
| `tenure_activity` | `tenure_days` | `as_of` − first event, in days | `1 − tau(first event)` | same | same |
| `span_activity` | `span_days` | last − first, in days | `tau(last) − tau(first)` | v1 measures a wall-clock span whose meaning depends on the window's volume profile (PSI 1.02) | a span of "30 % of the window's activity" means the same in both windows |
| `activity_concentration` | `daily_concentration` | max transactions in one **calendar day** / total | max transactions in one **activity decile** / total | the largest drifter (PSI 2.49): it counts calendar days, so Fri+Sat (3.1:1 volume) and Wed+Thu (1:1) cannot agree, and it also depends on where midnight falls inside the window | invariant to midnight alignment and to the day-volume ratio |
| `max_bin_txn` | `max_daily_txn` | max transactions in one calendar day | max transactions in one activity decile | same reason; it is the numerator of the above | same |
| `active_bin_fraction` | `active_days` | distinct calendar days with activity | distinct activity deciles with activity / 10 | a count of calendar days is a window-length and alignment artefact | a fraction in [0, 1] whatever the window's length or day count |
| `txn_per_active_bin` | `txn_per_active_day` | count / active days | count / active deciles | follows its denominator | same |
| `txn_per_span_activity` | `txn_per_span_day` | count / span in days | count / span in activity | follows its denominator | same |
| `max_gap_activity` | `max_gap_days` | largest wall-clock gap between consecutive events | largest gap in activity units | dormancy relative to how much happened, not to how many hours passed | invariant to volume profile |
| `window_half_share` | `second_half_share` | share of events after the midpoint of the **entity's own span** | share of events with `tau >= 0.5` - the window's activity midpoint | v1 is degenerate for a single-transaction entity (always 1.0) and its meaning shifts with the entity's own span; the share of such entities differs between windows (PSI 0.96) | well defined for every entity; the same reference point for all of them |
| *(dropped)* | `weekend_share` | share of events on Sat/Sun | **removed** | in a 2-day detection window every entity shares the same calendar days, so this has near-zero *within-window* variance and its value is determined by the window's placement (1.0 in W1, 0.0 in W2 and W3, mixed in W4). It encodes *which window this is*, not the entity's behaviour | removing a pure window-composition encoder |
| `mean_iat_hours` | `mean_iat_hours` | unchanged | unchanged | velocity is intrinsically wall-clock: "one transaction every 20 minutes" is an AML signal in hours, not in activity share | none claimed |
| `std_iat_hours` | `std_iat_hours` | unchanged | unchanged | same | none claimed |
| `min_iat_hours` | `min_iat_hours` | unchanged | unchanged | same | none claimed |
| `burstiness` | `burstiness` | Goh-Barabasi `(sd − mean)/(sd + mean)` over wall-clock gaps | unchanged | already scale-free; a ratio of dispersion to mean is invariant to a constant rescale of time | already invariant |
| `night_share` | `night_share` | share of events in hours 0-6 of the timestamp | unchanged | within-day timing, independent of where the window sits, as long as the window covers whole days | already invariant |

Deliberately **not** done, and why:

* **No quantile / rank normalisation of features against the scored
  population.** It would force every feature's distribution to be identical in
  every window, which would drive the PSI comparison in §9 to zero *by
  construction* and make the drift evidence circular. The activity clock changes
  what is measured; a rank transform would only hide the measurement.
* **No change to `aml_txn_v1`, `aml_flow_v1`, `aml_context_v1`** - even though
  `amount_top1_share` (PSI 1.17) also drifts. R6 is a single-variable ablation.
* **No new model, no retuning, no new windows, no relabelling.**

## 5. Protocol

Identical to R1 (`docs/r1_detection_transfer.md` §3-§8): four 2-day detection
windows tiling 2022-09-03 .. 09-11, three chronological folds, features and
labels both from the window being scored, fit 60 % / validation 20 % / in-period
reference 20 % inside the training window, `alert_budget` 1 % chosen on
validation, no calibrator, seeds 42 / 1337 / 7, boosting without graph features.

Feature sets: `aml_txn_v1`, **`aml_temporal_v2`**, `aml_flow_v1`,
`aml_context_v1` - the R1 set with one family swapped.

## 6. Decision gate (pre-registered)

Targets are **decision criteria, not optimisation targets**, and are taken from
the task that commissioned R6:

* **PASS - adopt `aml_temporal_v2`** if all of:
  1. W3->W4 retention **>= 0.50**;
  2. W1->W2 retention **>= 0.90** and W2->W3 retention **>= 0.90**;
  3. operational alert behaviour no worse (alert volume at the frozen threshold
     and FP/TP not materially degraded);
  4. drift decreases in the temporal family for W3->W4;
  5. no leakage detected.
* **FAIL - retain `aml_temporal_v1`** if W3->W4 retention stays **< 0.30** and
  the transformation does not materially reduce the relevant drift. Then stop
  temporal-feature experimentation and conclude that window comparability /
  simulator distribution shift is the stronger explanation.
* **MIXED / INCONCLUSIVE** for anything between, stated honestly.

Commitments, per the task's anti-overfitting rule: v2 was designed from feature
definitions and the invariance principle in §3, **one variant only**, documented
before it ran. No fold or seed may be dropped, and R6 test performance may not
be used to redesign v2.

---

## 7. Results

Ran 2026-09-12. Artifacts `outputs/experiments/ibm_r6_boosting_behaviour*.json`
(seeds 42 / 1337 / 7), compared against R1's
`ibm_r1_boosting_behaviour*.json`; the two side by side in
`docs/r6_temporal_v2_table.md`, generated by `compare_experiments.py`.
50 feature columns against R1's 51
(`weekend_share` dropped). **Zero** leakage findings above `info` in all three
runs; the window-boundary and calendar-separation checks are present for every
window and fold.

### 7.1 Fold results

| fold | v1 AP | v2 AP | Δ AP | v1 retention | v2 retention | v1 ROC | v2 ROC | Δ ROC | v1 in-period | v2 in-period |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1->W2 | 0.3145 ± 0.0050 | 0.2762 ± 0.0087 | **−0.0383** | **0.975** | **0.881** | 0.9132 | 0.9089 | −0.0044 | 0.3251 | 0.3159 |
| W2->W3 | 0.3425 ± 0.0034 | 0.3307 ± 0.0042 | −0.0118 | 0.984 | 0.977 | 0.9216 | 0.9203 | −0.0013 | 0.3482 | 0.3390 |
| W3->W4 | 0.0396 ± 0.0080 | 0.0633 ± 0.0004 | **+0.0236** | **0.111** | **0.192** | 0.7496 | 0.7629 | +0.0132 | 0.3549 | 0.3298 |

Lift over prevalence: 50.7 -> 44.5, 53.9 -> 52.0, **10.1 -> 16.1**.

v2 moves the problem fold in the intended direction (+60 % relative AP) and
costs 12 % relative AP on the fold that transferred best. In-period AP falls
slightly in all three windows (−0.009, −0.009, −0.025), which is the price of
dropping `weekend_share` and coarsening calendar-day concentration into deciles.

### 7.2 Operational (mean over seeds; each cell v1 / v2)

| fold | P@0.1 % | P@1 % | R@1 % | alerts at frozen threshold | FP/TP | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1->W2 | **0.9872 / 0.8851** | 0.2270 / 0.2193 | 0.3655 / 0.3532 | 1,626 / 1,601 | 2.4 / 2.7 | 0.069 / 0.070 |
| W2->W3 | 0.9915 / 0.9732 | 0.2473 / 0.2385 | 0.3890 / 0.3752 | 2,336 / 2,365 | 3.0 / 3.2 | 0.071 / 0.070 |
| W3->W4 | **0.1746 / 0.2582** | 0.0757 / 0.0766 | 0.1921 / 0.1945 | 44,282 / 46,541 | 75.2 / 72.7 | 0.375 / 0.380 |

The operating point is **not** stabilised: on W3->W4 the frozen threshold still
admits ~13 % of the population under v2 (46,541 alerts) against ~12.5 % under
v1, at FP/TP 72.7 against 75.2. The top of the list improves there
(P@0.1 % 0.175 -> 0.258) and degrades on W1->W2 (0.987 -> 0.885).

### 7.3 Drift (same PSI implementation as R1)

PSI is a property of the two windows' feature distributions, so it barely moves
with the seed; the table quotes **seed 42** and notes where the seeds disagree.

| fold | features > 0.25 | median PSI | `aml_temporal_*` family median |
| --- | ---: | ---: | ---: |
| W1->W2 | 6/48 -> 6/48 (v2: 6/5/6 across seeds) | 0.0991 -> 0.1090 | 0.1127 -> 0.1185 |
| W2->W3 | 0/48 -> 0/48 | 0.0000 -> 0.0000 | 0.0001 -> 0.0001 |
| W3->W4 | **16/48 -> 16/48** | 0.1175 -> **0.1622** | **0.2722 -> 0.2717** |

Largest individual shifts on the problem fold:

* **v1:** `daily_concentration` **2.4934**, `recency_days` 1.5156,
  `amount_top1_share` 1.1735, `span_days` 1.0165, `second_half_share` 0.9615.
* **v2:** `amount_top1_share` **1.1735** (an `aml_txn_v1` column v2 does not
  touch, by design), `span_activity` 0.9122, `active_bin_fraction` 0.8887,
  `window_half_share` 0.8434, `txn_total` 0.7124.

So the transformation **compressed the extreme position drift** - the largest
temporal PSI falls from 2.4934 to 0.9122, and `daily_concentration` and
`recency_days` leave the top five entirely - but **did not reduce the typical
drift**: the family median is unchanged (0.2722 -> 0.2717), the count above 0.25
is unchanged (16 of 48), and the overall median rose.

### 7.4 Entity strata (seed 42)

| fold | seen entities | new entities | first-time-positive population |
| --- | ---: | ---: | ---: |
| W1->W2 | 0.3432 -> 0.2921 | 0.2913 -> 0.2730 | 0.2640 -> 0.2347 |
| W2->W3 | 0.3459 -> 0.3335 | 0.3492 -> 0.3353 | 0.2859 -> 0.2744 |
| W3->W4 | **0.0255 -> 0.0882** | 0.0582 -> 0.0633 | 0.0207 -> 0.0381 |

### 7.5 Seed stability

Seed sd / mean: W1->W2 0.016 -> 0.031, W2->W3 0.010 -> 0.013, W3->W4
**0.202 -> 0.007**. v2 is dramatically more stable on the fold that collapses -
its failure there is systematic rather than seed-dependent.

### 7.6 Leakage verification

* 0 findings above `info` in all three runs (13 `info` findings each,
  identical to R1's three runs): 1 evaluation-regime, 3 calendar-separation,
  3 entity-overlap and 6 detection-window-boundary checks - one boundary check
  for each fold's train and test window.
* `tests/test_temporal_v2.py` (14 tests): the designed invariance (a monotone
  time warp leaves every activity-clock feature unchanged while v1's calendar-day
  features move), independence from `FeatureRequest.as_of`, independence from
  labels, window-locality through the snapshot path, bounds, and the NaN
  conventions.
* `aml_temporal_v2` reads only `FeatureRequest.edge_table` rows inside the
  window. It uses no labels, no rows outside the window, no other window and no
  reference "now". The activity clock is a statistic of the window's own
  transactions - the same class of population-level information
  `aml_network_v1`'s component features already use.

## 8. Decision gate

| condition | required | measured | verdict |
| --- | --- | --- | --- |
| 1. W3->W4 retention | >= 0.50 | **0.192** | **fail** |
| 2. W1->W2 and W2->W3 retention | >= 0.90 each | **0.881** / 0.977 | **fail** (W1->W2) |
| 3. operational behaviour not worse | - | better on W3->W4, worse on W1->W2; alert volume unchanged in kind | not met |
| 4. temporal-family drift decreases on W3->W4 | decrease | 0.2722 -> 0.2717, 16/48 unchanged | **fail** (extremes compressed, typical unchanged) |
| 5. no leakage | none | none | pass |

The pre-registered FAIL condition is met: W3->W4 retention stays below 0.30
**and** the relevant drift is not materially reduced at family level.

## **Decision: FAIL - RETAIN `aml_temporal_v1`**

Directionally v2 did what it was designed to do on the problem fold (+0.0236 AP,
retention 0.111 -> 0.192, lift 10.1 -> 16.1, seed sd/mean 0.202 -> 0.007, largest
temporal PSI 2.49 -> 0.91) and it removed v1's dependence on a reference "now".
It did not come close to the 0.50 target, and it **cost** 0.094 retention and
12 % relative AP on W1->W2 - the comparable-window case a production system
faces most of the time. A representation that helps the case you cannot detect
and hurts the case you can is not a production improvement.

**Why the transformation did not rescue the fold.** After the change, the largest
remaining shift among the model's features is `amount_top1_share` (1.1735), an
`aml_txn_v1` column v2 deliberately does not touch, followed by activity-clock
columns that *still* differ (`span_activity` 0.91, `active_bin_fraction` 0.89).
The activity clock normalises **when** things happened; it cannot normalise
**who is present**. W4 holds 354,447 eligible entities against W3's 235,433 at
1.6x lower prevalence. This is evidence consistent with R1's reading that window
comparability - population composition - dominates, and that calendar encoding
was a symptom rather than the cause.

Per the pre-registered rule, **temporal-feature experimentation stops here**. No
R7/R8 temporal transformation is proposed.

## 9. Production implication

* `aml_temporal_v1` **remains** the production temporal family.
* `aml_temporal_v2` stays in the repository as a registered, tested family that
  is **measured and not adopted**. It is not part of the production feature
  contract, and `docs/system_architecture_decision.md` §5 is unchanged.
* The safeguards R1 already indicated are what production needs, and R6 does not
  change any of them: model and feature versioning (carried by
  `RiskAssessment`), **per-window alert-budget thresholds**, a **PSI gate**
  before applying a frozen model to a new window, and the investigation graph.

## 10. Next step

**One action: implement the production safeguards in the risk-engine path** -
per-window threshold derivation and a PSI gate that refuses, or flags, a frozen
model applied to a window whose temporal-family drift exceeds the level at which
R1 measured transfer failure. Not another feature experiment.
