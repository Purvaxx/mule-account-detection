# The Calendar-Forward Benchmark (IBM HI-Small)

Can behaviour observed up to a fixed date predict which accounts will be
involved in laundering in the days that follow? And under that question, does
graph structure add anything?

This is the experiment `docs/temporal_collapse.md` §13 prescribed before any
2-hop or GNN work: a time-forward split that holds out **time** rather than
**entities chosen by when they were last active**.

Run 2026-09-11. **IBM AMLworld is synthetic.** Nothing here is evidence about
real-world detection rates.

Evidence:

* benchmark artifacts: `outputs/experiments/ibm_cal_*.json`
* design audit (timeline, origin scan, feature shift): `outputs/audits/calendar_design_ibm_aml_hi_small.json`,
  produced by `calendar_design_audit.py`
* generated table: `docs/calendar_forward_table.md`
* implementation: `src/experiments/calendar.py`; tests: `tests/test_calendar_forward.py`

Earlier IBM artifacts are untouched. The standard (`ibm_*`) and survivorship
time-forward (`ibm_tf_*`) numbers quoted here are read from those artifacts,
not recomputed.

---

## 0. The result in four sentences

Under a clean calendar cutoff the supervised models reach average precision
**0.007-0.017** at a prevalence of **0.267 %** (2.5-6.2x lift across candidates
and four model seeds; ROC 0.64-0.70), against a label-free floor of 0.0046
(1.7x). For the 82 % of future positives that had never been on a flagged
transaction before, every model falls to **1.4-2.0x lift**. Graph features add
**nothing measurable**: across four model seeds the paired increment is
**-0.0024 ± 0.0027 AP**, negative in all four. Advanced graph learning is
recorded as **NO-GO** on this benchmark.

---

## 1. What the previous time-forward split could not answer

`time_forward_holdout` with `time_reference="last"` put each **entity** in train,
validation or test by the date of its *last* transaction. On IBM that made the
test partition the accounts still transacting at the end of the file: train
median 3 transactions, test median 37, 48 of 54 features past PSI 0.25. It
compared two populations, not two periods (`docs/temporal_collapse.md` §5, §11).

The calendar-forward regime keeps the population and moves the clock.

---

## 2. The timeline

From the design audit's `daily` block.

| day | weekday | transactions | flagged | flagged rate | flagged rows that are ACH |
| --- | --- | ---: | ---: | ---: | ---: |
| 2022-09-01 | Thu | 1,114,921 | 322 | 0.029 % | 67.1 % |
| 2022-09-02 | Fri | 754,449 | 408 | 0.054 % | 74.8 % |
| 2022-09-03 | Sat | 207,382 | 391 | 0.189 % | 88.0 % |
| 2022-09-04 | Sun | 207,430 | 407 | 0.196 % | 93.6 % |
| 2022-09-05 | Mon | 482,650 | 471 | 0.098 % | 83.4 % |
| 2022-09-06 | Tue | 482,089 | 531 | 0.110 % | 87.6 % |
| 2022-09-07 | Wed | 482,751 | 497 | 0.103 % | 85.5 % |
| 2022-09-08 | Thu | 482,773 | 539 | 0.112 % | 86.3 % |
| 2022-09-09 | Fri | 654,467 | 514 | 0.079 % | 81.3 % |
| 2022-09-10 | Sat | 208,325 | 442 | 0.212 % | 94.1 % |
| 2022-09-11 .. 18 | | 1,108 in total | 655 | 57-73 % per day | 100 % |

Four facts decide the design:

1. **The body is ten days.** Normal traffic runs 2022-09-01 to 2022-09-10.
2. **Laundering volume is roughly stationary in the body** - 391 to 539
   flagged transactions a day from day 2 on. The flagged *rate* swings with the
   weekday only because normal volume does (weekend days carry about 43 % of a
   Monday-Thursday's traffic).
3. **Day 1 is a burn-in.** 1.11 M transactions, 511 thousand of them in the
   first six hours, and 437 thousand of the 515 thousand accounts active on it.
4. **Days 11-18 are a different process.** Normal traffic has stopped; what
   remains is the completion of laundering chains, 57-73 % flagged and 100 %
   ACH. A label drawn from that period describes the simulator winding down,
   not account behaviour.

---

## 3. Design

### 3.1 The prediction target

The project's entity label has always been *"the account is the payer or the
payee of at least one transaction the publisher flagged"*, evaluated over the
whole file. The calendar target is **the same rule evaluated over a prediction
window**:

| element | definition |
| --- | --- |
| unit of prediction | the pair (entity, origin) |
| origin `T` | a fixed calendar instant, the same for every entity |
| observation window | every transaction dated `< T` (expanding history; §3.4) |
| feature availability | all six `aml_*` families, computed from the observation window only; `recency_days` and `tenure_days` measured from `T` |
| prediction window | `[T, T + 2 days)` |
| label | 1 when the entity is payer or payee of >= 1 flagged transaction dated in the prediction window |
| eligibility | >= 1 dated transaction in the observation window. Window positives without one are **excluded and counted**, never scored |
| prior involvement | the entity was on a flagged transaction before `T`. Recorded for the evaluation subgroup; **never a feature** |

This is not a silent change of target - it is the only form of the existing
target that a calendar cutoff admits. It is nonetheless a **different quantity**
from the standard and survivorship benchmarks, whose label is whole-file
involvement, and its prevalence differs accordingly (0.267 % against 1.234 %
and 2.64 %). Average precision is prevalence-dependent, so the three regimes
are compared on lift and ROC as well as AP, and never in one table without
their targets.

About 82 % of test-window positives (1,125 of 1,374) had never been on a
flagged transaction before the origin. The benchmark is therefore dominated by
**first involvement**, and every artifact reports that subgroup separately
(`subgroup_test_metrics.no_prior_involvement`).

### 3.2 Why three origins, not one cutoff

A single cutoff with a random entity split would fit the model on labels from
`[T, T + H)` and score other entities against the *same* window. At scoring
time those training labels would not exist yet, so the design is not
deployable. And any single cutoff whose label runs to the end of the file draws
on the days 11-18 tail: at `T = 2022-09-10` with a three-day horizon, 388 of
1,091 eligible positives (36 %) would be positive *only* through the tail
(origin scan).

So the design is a rolling origin with three roles:

```text
train origin      2022-09-05   features < 09-05   label window [09-05, 09-07)   -> fit
validation origin 2022-09-07   features < 09-07   label window [09-07, 09-09)   -> calibrate, threshold
test origin       2022-09-09   features < 09-09   label window [09-09, 09-11)   -> scored once
```

The plan enforces `T_train + H <= T_val` and `T_val + H <= T_test`, so every
label used to fit or calibrate has been fully observed before the snapshot it
is applied to. `max_label_end = 2022-09-11` makes the exclusion of the tail a
declared, checked parameter: a configuration whose test window crosses it is
refused.

**The tail does make a single cutoff with an open-ended label invalid**, and it
is excluded by construction here rather than modelled.

### 3.3 Origins and horizon

The constraints, from the origin scan:

* Three non-overlapping windows must fit between the train origin and
  2022-09-11.
* The train origin must follow the day-1 burn-in by enough to leave history
  that is not dominated by it.
* Each window needs enough positives to estimate a precision-recall curve.

| horizon | feasible origins | positives per window (eligible) | verdict |
| --- | --- | ---: | --- |
| 1 day | many | 629-844 (origins 09-03 .. 09-10) | feasible, but half the positives per window |
| **2 days** | **09-05 / 09-07 / 09-09** | **1,428 / 1,465 / 1,374** | **chosen** |
| 3 days | 09-02 / 09-05 / 09-08 | 1,694 / 2,029 / 2,095 | train origin one day after the burn-in, history = day 1 alone |

Prevalence under the chosen design is **0.279 % / 0.286 % / 0.267 %** - stable
across the three roles, which the survivorship split was not (0.78 % / 1.18 % /
2.64 %). The origins were fixed from these label-free constraints before any
model was run.

The windows fall on different weekdays (train Mon-Tue, validation Wed-Thu, test
Fri-Sat), and the test window holds a Saturday, whose flagged *rate* is twice a
weekday's. Ten days cannot hold weekday composition fixed across three windows;
this is a stated residual, not a corrected one.

### 3.4 Feature history: expanding, not a fixed lookback

A fixed lookback makes every snapshot the same length, which sounds cleaner.
Measured by the shift it induces between the train and test snapshots - a
criterion that reads no labels and no model - it is worse:

| design | test positives excluded (no history) | features with PSI > 0.25 | median PSI |
| --- | ---: | ---: | ---: |
| **expanding history** | **23 of 1,397** | **14 of 66** | **0.010** |
| 4-day lookback | 379 of 1,397 | 37 of 66 | 0.340 |
| 3-day lookback | 412 of 1,397 | 21 of 66 | 0.149 |
| *survivorship split, for reference* | - | *48 of 54* | - |

A 4-day lookback puts the day-1 burn-in inside the train window only, so the
train snapshot is structurally unlike the later ones; both lookbacks also drop
27-29 % of test positives, whose recent history is empty. Expanding history is the
design.

Its residual shift is concentrated in `aml_temporal_v1` (family median PSI
0.275, every other family below 0.015): `tenure_days` (PSI 7.40), `recency_days`
(2.31) and `span_days` (1.22) grow with the length of the history, which is four
days at the train origin and eight at the test origin. Nearly every account
exists from day 1, so `tenure_days` is close to constant within a snapshot and
behaves like an origin indicator. Unlike `recency_days` under the survivorship
split it does not encode *who* is in which partition - every entity carries the
same offset - but a tree fitted at the train origin sees these three columns
only up to four days.

---

## 4. Temporal leakage audit

**Rule:** `feature_information_time < origin` for every feature of every
snapshot. How it is enforced, not just intended:

| mechanism | where |
| --- | --- |
| Each snapshot's edge table is `date < origin` (and `>= origin - lookback`); undated rows are excluded from every feature window and every label window | `calendar._feature_mask` |
| The rows *actually handed* to the families are re-checked: latest date `< origin`, earliest `>=` lookback start, no undated rows; a violation is a hard error | audit `calendar_information_boundary` |
| Label windows ordered and closed before the next origin; hard error otherwise | plan validation + audit `calendar_label_ordering` |
| Transaction flags live in `ExperimentDataset.transaction_labels`, **beside** the edge table, so no family can read them | `dataset.py`, `sources.build_transaction_dataset` |
| Whole-file labels include the future; a candidate must declare `calendar_compatible` (reads only `ctx.features`, `ctx.y_train`, `ctx.y_val`) or the run is refused. The legacy GraphSAGE reads `ctx.dataset.labels` and is refused | `candidate.py`, `calendar.run_calendar_experiment` |
| Families that need a `networkx` graph or do not honour a time window are refused | `calendar._check_spec` |
| `global` and `train_window` scopes are refused under this split; `origin_window` is refused under every other split | `calendar._check_spec`, `runner._compute_features` |
| Label-column and perfect-separator audits run on the snapshots | `leakage.audit_label_columns`, `leakage.audit_perfect_separators` |

The six families, audited against the code (`src/experiments/aml_features.py`
reads only `request.edge_table`, `request.entity_ids`, `request.as_of` and its
per-request `request.cache`; no family reads `request.graph`,
`request.attributes` or any global state):

| family | reads | depends on "now" | future information possible? |
| --- | --- | --- | --- |
| `aml_txn_v1` | counts and amounts of the windowed rows | no | no |
| `aml_temporal_v1` | timestamps of the windowed rows | `recency_days`, `tenure_days` from `as_of = T` | no; the history-length shift in §3.4 is a distribution effect, not leakage |
| `aml_flow_v1` | in/out pairs and turnaround inside the window | no | no - a turnaround needs both legs before `T` |
| `aml_network_v1` | sparse adjacency built from the windowed rows | no | no - future counterparties are absent |
| `aml_cycle_v1` | the same adjacency; ordered layering inside the window | no | no |
| `aml_context_v1` | payment format and currency of the windowed rows | no | no - post-origin payment formats are absent |

Tests that pin this (`tests/test_calendar_forward.py`): perturbing every amount
at or after the test origin and adding post-origin transactions leaves all three
snapshots bit-identical; re-flagging validation- and test-window rows and adding
post-train traffic leaves the fitted model's inputs identical; recency is
measured from each snapshot's own origin; the audit records an `info`-level
boundary check for every role. `IBM` artifacts carry the same checks: all
`calendar_information_boundary` and `calendar_label_ordering` findings are
`info`, and no run raised a warning.

---

## 5. Results - calendar-forward (regime C)

Test origin 2022-09-09: 514,022 eligible entities, **1,374 positive (0.267 %)**,
23 cold-start positives excluded. Model seed 42. Threshold at a 1 % alert budget
chosen on validation; test alerts are whatever that frozen threshold admits.

| candidate | AP | lift | ROC | P@100 | P@1000 | test alerts | TP | FP | precision | recall | Brier skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| label-free floor | 0.0046 | 1.72x | 0.652 | 0.00 | 0.015 | 4,848 | 39 | 4,809 | 0.8 % | 2.8 % | n/a |
| logistic regression | **0.0162** | 6.06x | 0.642 | 0.22 | 0.046 | 3,775 | 67 | 3,708 | 1.8 % | 4.9 % | +0.0013 |
| boosting, no graph | 0.0144 | 5.40x | **0.694** | 0.19 | 0.046 | 6,948 | 126 | 6,822 | 1.8 % | 9.2 % | +0.0018 |
| boosting + graph | 0.0142 | 5.29x | 0.686 | 0.19 | 0.050 | 3,343 | 74 | 3,269 | 2.2 % | 5.4 % | +0.0017 |

**First involvement** - the 509,308 test entities with no flagged transaction
before the origin, 1,125 of them positive (prevalence 0.221 %):

| candidate | AP | lift | ROC |
| --- | ---: | ---: | ---: |
| label-free floor | 0.0031 | 1.39x | 0.625 |
| logistic regression | 0.0036 | 1.65x | 0.624 |
| boosting, no graph | 0.0042 | 1.88x | 0.668 |
| boosting + graph | 0.0039 | 1.77x | 0.662 |

**A label-using reference, not a candidate.** Ranking test entities by whether
they were *already* on a flagged transaction before the origin - information an
AML team would hold, and which no candidate here is given - reaches AP
**0.0118** (design audit, `persistence_reference_ap`). 249 of the 4,714
already-involved accounts are positive again in the window (5.3 %).

Four readings:

1. **Supervision survives, narrowly.** Boosting and logistic regression reach
   3.1-3.5x the floor's AP at seed 42. Across four model seeds (§8) boosting
   scores 0.0084-0.0167 without graph features and 0.0068-0.0142 with them -
   1.5-3.6x the floor, depending on nothing but the seed.
2. **Most of the top of the ranking is re-identification.** The models' overall
   AP is in the range of the persistence reference, and on first involvement
   it collapses to under 2x lift. Pre-origin behaviour says little about which
   *new* accounts will be drawn into laundering in the next two days.
3. **The probabilities carry almost no information.** Brier skill against a
   constant base-rate forecast is +0.0013 to +0.0018. The small expected
   calibration errors (about 2e-4) reflect near-constant predictions, not good
   calibration.
4. **Average precision is unstable here; ROC is not.** At 0.27 % prevalence AP
   is set by a handful of top-ranked positives (P@100 ranges 0.02-0.22 across
   cells and seeds), while ROC stays within 0.64-0.70. Single-seed AP
   differences of a few thousandths are noise.

---

## 6. Three regimes, side by side

Not one table: three questions.

| | **A - standard** | **B - survivorship time-forward** | **C - calendar-forward** |
| --- | --- | --- | --- |
| split | stratified 4-fold, entity-disjoint | entities placed by last activity | fixed origins, (entity, origin) samples |
| feature scope | `global` (whole file) | `train_window` (before the one boundary) | `origin_window` (before each snapshot's origin) |
| target | involvement anywhere in the file | involvement anywhere in the file | involvement in the 2 days after the origin |
| features overlap the label period? | **yes** | partly | **no** |
| test prevalence | 1.234 % | 2.641 % | 0.267 % |
| features with PSI > 0.25 (train vs test) | - | 48 of 54 | 14 of 66 |

| candidate | A: AP / lift / ROC | B: AP / lift / ROC | C: AP / lift / ROC |
| --- | --- | --- | --- |
| label-free floor | 0.0390 / 3.2x / 0.753 | 0.0389 / 1.5x / 0.552 | 0.0046 / 1.7x / 0.652 |
| logistic regression | 0.1710 / 13.9x / 0.871 | 0.0525 / 2.0x / 0.596 | 0.0162 / 6.1x / 0.642 |
| boosting, no graph | 0.3734 / 30.3x / 0.901 | 0.1120 / 4.2x / 0.592 | 0.0144 / 5.4x / 0.694 |
| boosting + graph | **0.3827 / 31.0x / 0.922** | 0.1110 / 4.2x / 0.619 | 0.0142 / 5.3x / 0.686 |
| best supervised / floor (AP) | 9.8x | 2.9x | 3.1-3.5x at seed 42; boosting 1.5-3.6x across four seeds |

What the comparison shows:

* **A is retrospective detection.** Its features are computed over the same
  period as the label, so they include the flagged transactions themselves -
  their amounts, formats and timing. That is a legitimate question ("given
  everything an account did, was it involved?"), and 0.3827 is a valid answer
  to it. It is not a forecast.
* **C is prospective forecasting.** No feature can see a label-defining
  transaction. Removing the survivorship selection repaired the population
  (14 of 66 shifted features against 48 of 54) and improved ranking quality
  over B (ROC 0.69 against 0.59-0.62), but the question itself is far harder in
  precision terms.
* **B answered neither cleanly.** Its population shift and its drifting label
  were properties of the split (`docs/temporal_collapse.md`).

---

## 7. Feature families

Same model, same origins, model seed 42; only the families change.

| cell | families | A: AP | B: AP (ROC) | **C: AP (ROC)** | C, first involvement: AP |
| --- | --- | ---: | ---: | ---: | ---: |
| A | `aml_txn_v1` | 0.0589 | 0.0363 (0.533) | 0.0157 (0.683) | 0.0040 |
| B | A + `aml_temporal_v1` | 0.2222 | 0.0295 (**0.478**) | 0.0152 (0.676) | 0.0044 |
| C | A + `aml_flow_v1` | 0.1410 | 0.0553 (0.607) | 0.0075 (0.691) | 0.0044 |
| D | A + `aml_network_v1` + `aml_cycle_v1` | 0.1193 | 0.0618 (0.623) | 0.0065 (0.700) | 0.0041 |
| - | all but graph (`boosting_behaviour`) | 0.3734 | 0.1120 (0.592) | 0.0144 (0.694) | 0.0042 |
| E | all six | 0.3827 | 0.1110 (0.619) | 0.0142 (0.686) | 0.0039 |

* **Under C no family improves the top of the ranking over transaction
  features alone.** Cell A has the highest AP (0.0157). The cells are single
  seed; one configuration spans 0.0084-0.0167 across four model seeds
  (standard deviation 0.0035), so the AP differences between A, B and E are
  noise, and C and D sit about two to three seed standard deviations below A.
* **Flow and network structure help the bulk of the ranking, not its top.**
  Adding `aml_flow_v1` or `aml_network_v1` + `aml_cycle_v1` to transaction
  features raises ROC from 0.683 to 0.691 and 0.700 while *lowering* AP. The
  full models (ROC 0.686-0.700 across seeds) are no better than cell D.
* **The temporal inversion was the survivorship split.** Adding
  `aml_temporal_v1` under B drove ROC below chance (0.478); under C it costs
  0.007 ROC and nothing in AP beyond seed noise. The family that dominates A
  (+0.1633) is inert under C - which is what a family that summarises the label
  period's own activity should do when that period is withheld.
* **Network and cycle structure are not the survivors either.** Their ROC gain
  in cell D disappears once the other families are present (§8).

---

## 8. Graph incremental value

`boosting_graph` against `boosting_behaviour`: the same model, the same
origins, the same seed; the only difference is `aml_network_v1` + `aml_cycle_v1`.
Four model seeds, so the increment can be read against a run-to-run spread - a
single holdout has no fold dispersion.

| model seed | no graph: AP | + graph: AP | **Δ AP** | no graph: ROC | + graph: ROC | Δ ROC | Δ AP, first involvement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.0144 | 0.0142 | −0.0003 | 0.6936 | 0.6864 | −0.0072 | −0.0003 |
| 1337 | 0.0084 | 0.0068 | −0.0016 | 0.6980 | 0.6963 | −0.0017 | −0.0001 |
| 7 | 0.0167 | 0.0103 | −0.0064 | 0.7004 | 0.6915 | −0.0089 | +0.0001 |
| 2024 | 0.0136 | 0.0123 | −0.0013 | 0.6966 | 0.6982 | +0.0016 | +0.0001 |
| **mean ± sd** | 0.0133 ± 0.0035 | 0.0109 ± 0.0032 | **−0.0024 ± 0.0027** | 0.6971 ± 0.0028 | 0.6931 ± 0.0053 | −0.0040 ± 0.0049 | −0.0000 ± 0.0002 |

The paired AP difference is **negative in 4 of 4 seeds**; ROC is lower in 3 of
4; on first involvement it is ±0.0001 either way. The seed spread of either
configuration alone (sd ≈ 0.003) is larger than any graph effect, and the
effect it does show points the wrong way.

Across regimes, the graph families' marginal value is:

| regime | Δ AP from network + cycle | dispersion |
| --- | ---: | --- |
| A - standard | **+0.0093** | positive in all four paired folds |
| B - survivorship time-forward | −0.0010 | single holdout |
| C - calendar-forward | **−0.0024 ± 0.0027** | four paired model seeds, negative in all four |

The only positive graph increment the project has measured is in the one regime
whose features summarise the label period itself.

---

## 9. Scientific interpretation

1. **The calendar-forward benchmark is sound.** Its population is fixed, its
   prevalence is stable across roles, its information boundary is enforced on
   the rows actually used, the tail is excluded by declaration, and 97.9-98.4 %
   of window positives (by role) are scoreable from history.
2. **On this benchmark the forecasting problem is mostly not solved by any
   candidate.** 5-6x lift overall at the best seed, under 2x for first
   involvement, and near-zero Brier skill. That is a property of the question
   on this simulator - whether pre-origin behaviour anticipates a
   counterparty's involvement two days later - and not a defect of the split.
3. **The standard benchmark's strength is retrospective.** The 0.3827 headline
   comes from features that summarise the label period itself. The project
   should state which task it is solving - detection over an observed period,
   or forecasting beyond it - because the evidence for the two diverges by more
   than an order of magnitude in lift.
4. **Graph structure adds nothing in any temporal regime** measured on this
   dataset. The only positive graph increment the project has measured is
   +0.0093 AP in regime A, where features see the label period.

---

## 10. Decision: advanced graph learning is **NO-GO** on IBM HI-Small

Recorded in `docs/decisions.md` ("Establish the Calendar-Forward Benchmark and
Record NO-GO for Advanced Graph Learning on IBM HI-Small").

| GO condition | met? | evidence |
| --- | --- | --- |
| labels adequate | yes | 1,374-1,465 positives per window |
| evaluation defensible | **yes, now** | calendar-forward regime, §3-§4 |
| supervised beats the label-free floor | yes, narrowly and seed-dependently | §5, §8 |
| graph holds information tabular features do not | **no** | §8: −0.0024 ± 0.0027 AP under C, negative in 4 of 4 paired seeds; −0.0010 under B; +0.0093 only under A |

* **No edge-aware / GINE-style model is built.** Its justification was relational
  signal the tabular families miss; under the one regime that asks a forecasting
  question cleanly, the tabular graph families contribute nothing, and neither
  does any other family beyond transaction volume.
* **The 2-hop family is not run in this phase** and is no longer on the critical
  path. It remains the cheapest falsification test of the multi-hop claim; if
  run, it should be run under C with at least four model seeds and judged
  against the seed spread in §8, not against a single number.
* **What would reopen the question:** a paired, multi-seed graph increment under
  calendar-forward that exceeds the seed spread; or a dataset - ideally real -
  on which forecasting from history is not already near the floor.

---

## 11. Reproduce

```bash
python calendar_design_audit.py configs/experiments/ibm_cal_boosting_graph.json
python run_experiment.py configs/experiments/ibm_cal_*.json
python compare_experiments.py outputs/experiments/ibm_cal_*.json --report docs/calendar_forward_table.md
```

Run the configs in one invocation: each origin's snapshot is computed once per
process and reused by every config (`feature_summary.snapshot_reuse` records
which blocks were reused).

## 12. What this does not establish

* **Anything about real data.** A simulator's prevalence, horizon and typology
  mix.
* **Tuned performance.** One fixed configuration per candidate, as in the
  earlier regimes.
* **Robustness to the choice of origins.** One rolling-origin design fits in
  ten days. A second test origin would need a shorter horizon.
* **Performance under a label-aware model.** No candidate sees prior
  involvement; the persistence reference suggests such a feature would matter.
  Adding it is a change to the feature contract and needs its own decision.
* **A transaction-level result.** IBM labels transactions; this benchmark, like
  every earlier one, classifies entities.
