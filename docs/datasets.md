# Dataset Audit

Which data this project can actually learn from, and what each dataset is
allowed to be used to claim.

Generated evidence lives beside this document and is regenerated, never edited:

```bash
python audit_datasets.py
```

* machine-readable: `outputs/audits/audit_<name>.json`, one per dataset
* generated table: `docs/dataset_audit_table.md`
* audit implementation: `src/experiments/audit.py`
* loaders: `src/experiments/sources.py`

The prose below is the interpretation. The numbers in it are copied from the
JSON artifacts and are reproducible by re-running the command above.

Audit run: 2026-09-06. Environment: Python 3.11.8, Windows, CPU only.

---

## 1. The headline

**`ibm_aml_hi_small` is a usable AML model-selection benchmark. Nothing else
available is.**

Five labelled transaction datasets are now reachable from this tree:

| dataset | origin | entities | transactions | positives | positive rate | probe AP | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ibm_aml_hi_small` (IBM AMLworld HI-Small) | synthetic | 515,088 | 5,078,345 | 6,357 | 1.234 % | **0.2098** | **MODEL-SELECTION READY** |
| `reference_dkdata` (Azerbaijani Laundromat) | **real** | 3,440 | 16,940 | 4 | 0.116 % | undefined | REGRESSION ONLY |
| `samld` (SAML-D) | synthetic | 855,460 | 9,504,852 | 7,902 | 0.924 % | 0.9971 | STRUCTURAL VALIDATION ONLY |
| `ibm_aml_sample` (pattern extract) | synthetic | 3,183 | 137,936 | 3,183 | 100 % | undefined | BLOCKED |
| `financial_fraud_kaggle` | synthetic | 899,988 | 5,000,000 | 295,613 | 32.85 % | 0.4226 | STRUCTURAL VALIDATION ONLY (card fraud) |

"probe AP" is the average precision an **untuned** gradient-boosted tree reaches
using only fifteen simple structural quantities. It is the measure of whether a
dataset's labels are a function of its structure. At 0.9971 SAML-D's labels are;
at 0.2098 IBM's are not, while still carrying a 17× lift over prevalence. That
gap is what separates a benchmark from a generator you can reverse-engineer.

**The previous phase's conclusion — that no dataset could rank models — is
superseded.** It was correct about the four datasets then available. The IBM
*ledger* was not among them: only its pattern extract was, which has no negative
class. Acquiring the ledger changed the answer.

What has not changed: the only dataset with **real, observed** transactions is
still `reference_dkdata`, and it still has four positives. Every model result in
this project remains a result about synthetic data.

### 1.1 This is a different question from workspace compatibility

`docs/workspaces.md` reports SAML-D and IBM AML as `INCOMPATIBLE` / `BLOCKED`
with `missing_required_fields == ["amount_usd"]`. That is a statement about the
**frozen-checkpoint inference path**, which needs a USD amount because the
checkpoint was trained on one, and which will not invent an FX rate to produce
it.

This document is about the **experiment layer**, which has no such requirement:
`src/experiments/aml_features.py` reads whatever amount column the dataset
carries and the dataset spec records that it is native, not converted. So the
same file can be blocked for scoring against the frozen checkpoint and still be
loadable for model research. Both statements are true and neither weakens the
other.

---

## 2. `reference_dkdata` — the Reference Dataset

**Source.** `dkdata.json`, one directory above the repository. Derived from the
OCCRP Azerbaijani Laundromat publication. Real, observed transactions.

**Labels.** Entity-level. Positive means the `core` node attribute is true,
which marks an entity named as core to the published investigation. It is not a
supervised fraud adjudication, and there are four of them.

**Facts.** 3,440 entities, 16,940 transactions, 4,024 distinct ordered pairs
(4.21 transactions per pair), 33 self-loops, no fully duplicated rows.
Timestamps span 2012-06-30 to 2014-12-31 (914 days), every transaction dated.

**Structural triviality.** Ranking the population by a single integer column
gives average precision **1.0000** — a lift of 860× over prevalence — for each
of: transaction count, out-degree, in-degree, fan-in, fan-out, counterparty
count, incoming volume, degree centrality and velocity per active day.
Outgoing volume gives 0.8042 and total volume 0.8875. The four positives are
simply the four busiest entities.

The combined probe is `None`, with the reason recorded: four positives cannot
be split into a train and a test half.

**Temporal evaluation is impossible.** Cutting the population at the median
first-activity date leaves 2 positives in the later partition; at the 60th,
70th and 80th percentiles it leaves **zero**. There is no forward-looking split
that has anything to score.

**Verdict: REGRESSION ONLY.** Its value is that its numbers are pinned. A
change in them means the pipeline broke. It is not evidence of detection skill
and a perfect score on it distinguishes nothing.

This confirms, through an independent implementation, the finding already
recorded in `docs/architecture.md` §45.7. Do not delete
it, do not alter its historical results, and do not quote them as capability.

---

## 3. `samld` — SAML-D

**Source.** `SAML-D.csv.zip` (Oztas et al., 2023). The name is literal: it is a
**Synthetic** Anti-Money-Laundering Dataset, generated from declared typologies.
`DatasetSpec.synthetic` is `True` and every artifact produced from it carries
that flag.

**Facts.** 9,504,852 transactions, 855,460 accounts, 2022-10-07 to 2023-08-23
(320 days). No missing values in any of the twelve columns. No self-loops. No
duplicated rows. 13 payment currencies (95.7 % GBP), 18 bank locations, 7
payment types.

**Labels.** SAML-D labels *transactions*: 9,873 of 9,504,852 carry
`Is_laundering == 1` (0.104 %). The entity label used here is derived — an
account is positive if it is the sender or the receiver of at least one flagged
transaction — which yields 7,902 positive accounts (0.924 %). The derivation is
written into `DatasetSpec.label_definition` and it has a consequence worth
stating: a launderer's counterparty becomes positive too.

Positives are spread evenly across all eleven months (694 to 1,024 flagged
transactions per month), so a forward-looking split is feasible: cutting at the
median first-activity date leaves 911 positives in the later partition.

**Label alias.** `Laundering_type` names the generating mechanism of every row,
normal rows included, and `Is_laundering == 1` exactly when it does not begin
with `Normal` — verified with **zero** mismatches over all 9,504,852 rows. It is
an exact alias of the label. `src/experiments/sources.py` does not load it.

### 3.1 The structure the generator produced

The graph is far thinner than the transaction count suggests:

* 9,504,852 transactions collapse to **887,497 distinct ordered pairs** — 10.7
  transactions per relationship.
* Median counterparty count is **1**; the 90th percentile is also **1**. Ninety
  per cent of accounts transact with exactly one other account.
* 15,592 weakly connected components, largest 954, median size 51.
* **Zero directed 3-cycles in the entire dataset.** 93,834 reciprocal edges
  (46,917 mutual pairs).
* The `Cycle` typology is 382 transactions (0.004 % of the dataset) and forms
  chains of length 5 to 14; only 36 of them close into a cycle.

### 3.2 The labels are recoverable from that structure

No single simple quantity looks alarming. Ranking the whole population by each
in turn (prevalence floor 0.0092):

| quantity | AP | ROC | direction |
| --- | ---: | ---: | --- |
| `fan_out` | 0.1831 | 0.9229 | descending |
| `counterparty_count` | 0.1820 | 0.9367 | descending |
| `degree_centrality` | 0.1820 | 0.9367 | descending |
| `transaction_count` | 0.1655 | 0.8856 | descending |
| `fan_in` | 0.1633 | 0.8849 | descending |
| `total_volume` | 0.1552 | 0.8896 | descending |
| `outgoing_volume` | 0.1327 | 0.9025 | descending |
| `out_degree` | 0.1305 | 0.8868 | descending |
| `active_span_days` | 0.0658 | 0.8117 | descending |
| `incoming_volume` | 0.0649 | 0.7439 | descending |
| `in_degree` | 0.0625 | 0.6855 | descending |
| `component_size` | 0.0452 | 0.7929 | descending |
| `last_activity` | 0.0416 | 0.8132 | descending |
| `first_activity` | 0.0392 | 0.7852 | ascending |
| `velocity_per_active_day` | 0.0102 | 0.5302 | ascending |

Every one is a lift over prevalence, none is remotely conclusive, and the gap
between ROC (up to 0.94) and AP (up to 0.18) is the ordinary rare-positive
picture: good global ordering, poor precision at the top of the list.

The *combination* is the problem. An untuned `HistGradientBoostingClassifier`
on the fifteen simple quantities alone, fitted on a stratified half and scored
on the other half, reaches:

```text
test average precision  0.9971
test ROC-AUC            0.9999712
test positives          3,951
```

The mechanism is a **hole in the negative distribution**, and the audit reports
it directly (`structural_triviality.class_free_bands`):

| quantity | values no negative takes | positives in them |
| --- | --- | ---: |
| `counterparty_count` | 5, 6, 8 | 619 |
| `fan_out` | 5, 6, 7, 8 | 610 |
| `degree_centrality` | the same values, rescaled | 619 |

(The audit only reports a band holding at least 25 entities, so
`counterparty_count == 7`, with 18 positives and no negatives, is below the
reporting floor rather than absent from the effect.)

The generator builds normal patterns as **stars** — a hub with degree-1 leaves,
which is why the median account has exactly one counterparty — and laundering
typologies as **multi-hop meshes**. An account's own counterparty count together
with the mean counterparty count of its neighbours therefore separates the
classes almost exactly: positives average 5.5 neighbour-counterparties against
44.5 for negatives, distributions that barely overlap and that are *inverted*
relative to what an analyst would expect.

Neither `reference_dkdata` nor `financial_fraud_kaggle` has a single class-free
band. The diagnostic is not flagging everything; it is flagging SAML-D.

Restricting the population does not help. On accounts with at least three,
five or ten counterparties — where prevalence rises to about 32 % — a boosted
model on the full AML feature set still reaches average precision 1.0000.

**Verdict: STRUCTURAL VALIDATION ONLY.** SAML-D is genuinely valuable to this
project: it is the only dataset here that exercises the pipeline at scale, and
every scaling claim in this phase was verified on it. It cannot rank models,
because every reasonable model saturates.

---

---

## 4. `ibm_aml_hi_small` — IBM AMLworld HI-Small ledger — **MODEL-SELECTION READY**

The dataset that unblocks model selection. Acquired 2026-09-06.

### 4.1 Provenance

Not vendored. Fetch it with:

```bash
python acquire_dataset.py ibm_aml_hi_small
```

which writes `data/manifests/ibm_aml_hi_small.json` — the committed provenance
record. What that manifest holds:

| field | value |
| --- | --- |
| file | `HI-Small_Trans.csv` |
| source | `https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml` |
| publisher | Erik Altman (IBM) |
| licence | Community Data License Agreement — Sharing — Version 1.0 |
| acquired | 2026-09-06T11:07:43Z |
| bytes | 475,664,283 |
| sha256 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| citation | Altman, Blanusa, von Niederhäusern, Egressy, Anghel, Atasu. *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023 Datasets and Benchmarks. |

The download needed no Kaggle credentials in this environment. If a future
environment blocks it, `acquire_dataset.py` prints the exact manual step and
`--verify-only` will hash a manually placed file and write the same manifest.

**SYNTHETIC.** IBM's AMLworld is a virtual-world simulator. `DatasetSpec.synthetic`
is `True` and every artifact inherits it. Nothing below is evidence about
real-world detection performance.

### 4.2 What the file contains

5,078,345 transactions, 2022-09-01 00:00 to 2022-09-18 16:18 (**17 days**),
every row dated, **no missing values in any column**.

| property | value |
| --- | --- |
| entities (`(bank, account)` pairs) | 515,088 |
| distinct ordered pairs | 1,015,736 (5.00 transactions per relationship) |
| self-loops | 591,212 (11.64 %) — the `Reinvestment` payment format |
| fully duplicated rows | 9 |
| distinct bank codes | 30,528 sending, 15,850 receiving |
| currencies | 15 (US Dollar 37 %, Euro 23 %) |
| payment formats | 7: Cheque, Credit Card, ACH, Cash, Reinvestment, Wire, Bitcoin |
| rows where amount received ≠ amount paid | 72,158 (currency conversion) |

### 4.3 Two schema traps, both real

**The payer and payee account columns have the same name.** The published
header is `Timestamp,From Bank,Account,To Bank,Account,...` — `Account` twice.
pandas disambiguates the second as `Account.1`. A loader that reads "the
`Account` column" gets the payer twice and builds a graph of self-loops.

**Bank codes are zero-padded text.** Read with inferred dtypes, the file's
30,528 distinct bank codes collapse to 30,470 — **58 collisions**, including
`012` merged with `0012`. `src/experiments/sources.py` pins the identifier
columns to `str`. On this file the collisions happen not to produce colliding
`(bank, account)` pairs, so the entity population is 515,088 either way; that is
luck, not a reason to rely on inference.

### 4.4 Labels

IBM labels **transactions**: 5,177 of 5,078,345 carry `Is Laundering == 1`
(0.102 %). The entity label used here is derived — an entity is positive if it
is the payer or payee of at least one flagged transaction — giving **6,357
positive entities (1.234 %)**. The derivation is written into
`DatasetSpec.label_definition`, and it makes a launderer's counterparty positive
too.

**No label alias exists in the ledger.** Unlike SAML-D, the typology names live
in a *separate* file (`HI-Small_Patterns.txt`), which this loader does not read.
That file's group headers state the mechanism *and its parameter* — `FAN-IN: Max
13-degree Fan-In`, `CYCLE: Max 12 hops` — so it is a description of the
label-generating mechanism and must never become a feature.

The categorical columns correlate with the label but none is deterministic:

| Payment Format | negatives | positives | positive rate |
| --- | ---: | ---: | ---: |
| ACH | 596,314 | 4,483 | 0.746 % |
| Bitcoin | 146,035 | 56 | 0.038 % |
| Cash | 490,783 | 108 | 0.022 % |
| Cheque | 1,864,007 | 324 | 0.017 % |
| Credit Card | 1,323,118 | 206 | 0.016 % |
| Reinvestment | 481,056 | 0 | 0 % |
| Wire | 171,855 | 0 | 0 % |

ACH carries 87 % of the positives at a 0.75 % positive rate — a strong signal,
not an alias. Self-loops are almost never laundering (0.0019 %).

### 4.5 Structural triviality — **the decisive contrast with SAML-D**

Ranking the whole population by each simple quantity (prevalence floor 0.0123):

| quantity | AP | ROC |
| --- | ---: | ---: |
| `last_activity` | 0.1404 | 0.6781 |
| `active_span_days` | 0.1321 | 0.6861 |
| `counterparty_count` | 0.0353 | 0.7569 |
| `fan_in` | 0.0346 | 0.7307 |
| `transaction_count` | 0.0255 | 0.6917 |
| `out_degree` | 0.0244 | 0.6581 |
| `fan_out` | 0.0227 | 0.6577 |
| `component_size` | 0.0170 | 0.6396 |
| `velocity_per_active_day` | 0.0166 | 0.6413 |

The untuned combined probe on all fifteen quantities reaches:

```text
test average precision  0.2098
test ROC-AUC            0.8771
test positives          3,178
```

**`class_free_bands` is empty.** There is no value of any discrete quantity that
no negative takes. Compare SAML-D, where `counterparty_count ∈ {5, 6, 8}` held
619 positives and zero negatives.

| dataset | probe AP | class-free bands | verdict |
| --- | ---: | --- | --- |
| `samld` | **0.9971** | `counterparty_count`, `fan_out`, `degree_centrality` | structural validation only |
| `ibm_aml_hi_small` | **0.2098** | none | **model-selection ready** |

Structure gives a 17× lift over prevalence and leaves most of the problem
unsolved. That is what a usable benchmark looks like: informative, not
determinative.

One correlate to watch: the two strongest single quantities are *temporal*
(`last_activity`, `active_span_days`). Laundering entities are active later and
for longer inside a 17-day window. That is a property of how the simulator
injects typologies over the period, and it is why the time-forward result below
matters more than the transductive one.

### 4.6 Split feasibility

| regime | usable | evidence |
| --- | --- | --- |
| stratified k-fold (entity-disjoint) | **yes** | 6,357 positives over 4 folds ≈ 1,589 test positives per fold |
| entity-disjoint | **yes** | the unit of prediction is the entity and entities are distinct |
| time-forward holdout | **yes, on *last* activity only** | see below |
| component-grouped k-fold | possible | components are the grouping unit; fold sizes follow the component-size distribution |

**Time-forward requires care, and the audit shows why.** Placing entities by
*first* activity is nearly degenerate: half of all entities transact for the
first time within **28 minutes** of the start of the file, so the 60th-percentile
cut falls at 2022-09-01 03:55 and leaves a four-hour training window with
essentially no history to compute features from.

Placing entities by *last* activity works:

| cut | boundary | entities after | positives after | prevalence after |
| --- | --- | ---: | ---: | ---: |
| q60 | 2022-09-09 18:43 | 206,085 | 3,940 | 1.91 % |
| q70 | 2022-09-09 23:29 | 154,729 | 3,367 | 2.18 % |
| q80 | 2022-09-10 06:27 | 103,101 | 2,723 | 2.64 % |

That gives a ~9-day training window and thousands of evaluable positives. The
configs use `time_reference="last"` with `train_quantile=0.6`, `val_quantile=0.8`
and `FeatureSpec.scope="train_window"`, and the artifact records the partition
counts.

### 4.7 Verdict

**MODEL-SELECTION READY.** Enough positives (6,357), labels that are not
recoverable from structure (probe AP 0.2098, no class-free bands), and two
defensible evaluation regimes with real positives in every partition.

Limitations that travel with every result from it:

* **Synthetic.** A simulator's notion of laundering, not an observed one.
* **Derived entity label.** A launderer's counterparty is labelled positive.
* **17 days.** Long-horizon behaviour cannot be studied here.
* Being the best available benchmark is not the same as being realistic. No
  claim of realism is made; the generator's own paper is the authority on what
  it models.

---

## 5. `ibm_aml_sample` — the IBM AML pattern extract

**Source.** `IBM_AML_sample.xlsx`, 137,936 rows, derived from `IBM data.txt`,
which is the laundering-*pattern* file of IBM's synthetic AML data (its lines
are grouped by `BEGIN LAUNDERING ATTEMPT` markers).

**Facts.** 3,183 accounts, 103,550 distinct ordered pairs, 1,709 self-loops,
2022-08-01 to 2023-01-12, 15 currencies, no missing values.

**Labels.** Every one of the 137,936 rows carries `is_suspicious == 1`, so all
3,183 accounts are positive. There is no negative class.

**Verdict: BLOCKED.** A classifier cannot be trained or evaluated on a
single-class population, and no ranking metric is defined. The blocker is
specific and removable: IBM publishes full transaction ledgers
(`HI-Small_Trans.csv` and its siblings) that contain both classes. Those files
are **not present in this tree**. Obtaining one is the concrete next step for
this dataset.

---

## 6. `financial_fraud_kaggle`

**Source.** `financial_fraud_detection_dataset.csv`, 5,000,000 rows.

**Facts.** 899,988 accounts, 2023-01-01 to 2024-01-01, no missing values in the
loaded columns.

**Two disqualifying properties**, both structural rather than statistical:

1. **Wrong task.** The label marks card-not-present fraud — the file's single
   `fraud_type` value is `card_not_present`. Ranking a model on it selects for a
   different problem, however good the score is.
2. **No payment network.** 5,000,000 transactions produce 4,999,985 distinct
   ordered pairs: **1.000003 transactions per relationship**. The account graph
   is a random matching, not a network. Entity-level aggregation of a
   transaction label over such a graph turns 179,553 flagged transactions into
   295,613 positive accounts (32.85 %), which describes the sampling, not the
   accounts.

The file also ships `velocity_score`, `spending_deviation_score` and
`geo_anomaly_score`, which were generated alongside the label. None is loaded.

**Verdict: STRUCTURAL VALIDATION ONLY** (domain mismatch). Its one genuine use
is as a control: its structural probe reaches average precision 0.4226 against a
prevalence floor of 0.3285, which is the only *non-degenerate* probe result in
this table and therefore evidence that the audit is capable of returning
"non-trivial" rather than always returning "trivial".

---

## 7. Datasets present in the tree that are not benchmarks

| file | what it is | status |
| --- | --- | --- |
| `Azerbaijani_Laundromat_Raw_Data.xlsx` | the spreadsheet `dkdata.json` was derived from | same data as `reference_dkdata`; not audited separately |
| `synthetic_aml_demo/` | a demo dataset this project generated for the UI | demo only; never evidence |
| `DEMO XL.xlsx`, `demo 1.xlsx` | demo uploads used to exercise the workspace path | demo only |
| `SAML_D_enriched_test.xlsx` | a 100,000-row SAML-D slice with project columns added | derived from `samld`; inherits its verdict, and its added columns include `payer_core` / `beneficiary_core`, which are project artefacts, not SAML-D labels |

Synthetic and demo datasets must never be presented as evidence of real-world
performance. `DatasetSpec.synthetic` carries that distinction into every
artifact; do not strip it.

---

## 8. What this means for model selection

1. **Model comparison is unblocked, on `ibm_aml_hi_small`.** Results are in
   `docs/model_selection.md`. Every one of them is a result about a simulator.

2. **The other four keep their roles, unchanged.** `reference_dkdata` is the
   regression fixture (`tests/test_reference_regression.py`). `samld` and
   `financial_fraud_kaggle` are scale and sanity checks. `ibm_aml_sample` is
   blocked and is superseded by the ledger.

3. **The remaining gap is real data, and it is a different gap from the one this
   phase closed.**

   | gap | status |
   | --- | --- |
   | no benchmark that can rank models | **closed** — `ibm_aml_hi_small` |
   | IBM AML has no negatives here | **closed** — the ledger has both classes |
   | no dataset with **real** labels and enough positives | **open** — needs a labelled real transaction set or a supervised-adjudication export |
   | SAML-D's entity labels are structurally determined | open; evaluating SAML-D at *transaction* level is untested |
   | no benchmark with a documented realistic benign/suspicious overlap | open — see 4 |

4. **A controlled synthetic benchmark is no longer the immediate next step.**
   The previous phase justified building one because nothing could rank models.
   Something can now. Building a generator of our own would be measuring our own
   assumptions when a published, peer-reviewed benchmark is available. Revisit it
   if IBM's limitations (17 days, derived entity label, one simulator) start
   binding on a specific question.

5. **Being the best available benchmark is not being realistic.** IBM AMLworld
   models a virtual economy. No claim about real-world detection rates,
   prevalence or alert burden may be sourced from it.

## 9. How to re-run and how to extend

```bash
python audit_datasets.py                 # all reachable datasets
python audit_datasets.py samld           # one
python audit_datasets.py --list
```

Adding a dataset is two edits: a loader in `src/experiments/sources.py`
(about twenty lines: a `SourceSpec` column mapping plus a `DatasetSpec` that
states the label rule and the synthetic flag), and one row in `_loaders()` in
`audit_datasets.py` declaring its label domain.

The declared domain is not measured. Whether a publisher's positive class means
"money laundering" is a fact about that publisher's definition, and no statistic
on the file can recover it, so the loader table states it and the audit records
it.
