# Research Findings

## Scope and evidentiary standard

This document records **only** findings that are grounded in material that
actually exists in this repository: the controlled ablation study, the audit
scripts and their captured outputs, and the current implementation.

Per the project's research-integrity rule, no citation, benchmark, or comparison to
external literature appears here unless the supporting material is present in the
repository. **No external literature review has been conducted or recorded for
this project.** That gap is stated explicitly in
[Not yet established](#not-yet-established) rather than filled with plausible
references.

Every quantitative result below is reproducible from the reference dataset
(`../dkdata.json`; 3,440 entities, 16,940 transactions, 4 labelled core entities)
via the `run_exp*.py` and `audit_*.py` scripts.

---

## 1. Controlled ablation study (Experiments 1–5)

**Source:** `docs/architecture.md` §28.3; reproducible via `scripts/legacy/run_exp1.py` …
`scripts/legacy/run_exp5.py`. Experiment 5's captured output is committed at
`outputs/experiments/exp5_report.txt`.

All five experiments used identical GraphSAGE architectures, hyperparameters,
dataset partitioning, and the same fixed 4-fold cross-validation protocol. Audits
(`scripts/legacy/audit_exp3.py`, `scripts/legacy/audit_exp4.py`, `scripts/legacy/audit_gnn.py`) confirmed no target-label
leakage in any configuration.

| Experiment | Feature set | Features | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|---|---|
| 1 | Baseline | 160 | 0.7534 ± 0.4271 | 0.9790 ± 0.0363 | 0.7500 ± 0.4330 |
| 2 | Baseline + Temporal | 166 | 0.7550 ± 0.4244 | 0.9857 ± 0.0247 | 0.6000 ± 0.4243 |
| 3 | Baseline + Directed-Multigraph | 167 | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** |
| 4 | Baseline + Temporal + Multigraph | 173 | 0.7534 ± 0.4271 | 0.9790 ± 0.0363 | 0.6250 ± 0.4146 |
| 5 | Baseline + TGA (time-respecting cycles) | 163 | 0.7534 ± 0.4272 | 0.9788 ± 0.0368 | 0.6250 ± 0.4146 |

### Finding 1.1 — Directed-multigraph structure carries the discriminative signal

Preserving parallel directed edges, rather than collapsing repeated transactions
into a single edge, is what separates core entities on this dataset. This is the
project's central empirical result and the reason Experiment 3 was selected as
the shipping configuration.

### Finding 1.2 — The separation is driven by a single feature

`scripts/legacy/audit_multigraph.py` established that the perfect separation is attributable
primarily to `reciprocal_out_count`:

* the 4 core entities have a **minimum** `reciprocal_out_count` of **8**;
* the 3,436 non-core entities have a **maximum** of **3**.

The classes are linearly separable on this one feature. The GNN is not required
to discover a subtle interaction.

### Finding 1.3 — Temporal features gave no measurable uplift under this protocol

Experiment 2 did not improve meaningfully over Experiment 1, and Experiment 4 did
not preserve Experiment 3's separation. This is an **observed result under a fixed
protocol**, not a demonstration that temporal features are inherently
uninformative. No causal explanation for the Experiment 4 degradation has been
established.

### Finding 1.4 — These metrics are statistically unstable and must not be generalized

With only 4 positive entities, fold-level and threshold-based metrics are
extremely unstable. Perfect metric values do **not** indicate generalization.

> **Standing constraint.** Experiment 3's `1.0000` scores describe perfect
> separation *on this particular dataset*. They must never be presented as
> evidence of real-world AML performance, and the system must not be described as
> solving money laundering.

---

## 2. Entity and country encoding audits

**Source:** `scripts/legacy/analyze_33_ee.py`, `scripts/legacy/analyze_conflicts.py`,
`scripts/legacy/analyze_conflicts_details.py`, `scripts/legacy/feature_audit.py`.

### Finding 2.1 — `bank_country` mixes ISO codes with non-ISO encodings

The reference dataset's bank-country field contains genuine ISO-3166 alpha-2
codes (`EE`, `LV`, `GB`, `DE`) alongside numeric and symbolic codes (`00`–`99`,
`.A`, `.D`, `.K`, `/M`, `G8`, `JH`, `R0`, `T1`, `TA`, `TX`). 151 distinct values
occur.

### Finding 2.2 — The encoding variants were not fully resolvable

Investigation of specific candidate pairs (e.g. whether `33` is an encoding
variant of `EE`) did not produce a defensible complete mapping. Some variants
appear to be alternative encodings of the same country; others represent
genuinely distinct bank locations.

**Consequence for the implementation.** Because the mapping is unresolved, no
normalization is applied. Every raw code is retained as its own indicator column,
and entities are represented multi-valued — an entity with accounts in `EE` and
`LV` sets both columns. Collapsing them would have required an unjustified
assumption. See `docs/decisions.md` → *Replace the Dataset-Derived Country
Vocabulary* for how this set later became a frozen constant.

### Finding 2.3 — Entities genuinely operate across multiple countries

Multi-country entities are common enough that single-valued country encoding
would lose real information. This motivated the multi-hot rather than one-hot
country representation.

---

## 3. Topological / temporal-cycle audit

**Source:** `scripts/legacy/audit_tga.py`, captured output in `scripts/legacy/audit_tga.txt`;
`scripts/legacy/audit_tga_correlations.py`, `scripts/legacy/investigate_cores.py`.

Over all 3,440 nodes (25.37 s runtime):

| Measure | Value |
|---|---|
| Nodes with reciprocal temporal cycles | 132 |
| Nodes with ≥3-edge temporal cycles | 40 |
| Max reciprocal cycle count | 98,092 |
| Max longer-cycle count | 31,335,496 |

Core vs non-core distributions:

| Feature | Core values | Core range | Non-core range |
|---|---|---|---|
| `temporal_reciprocal_count` | 98092, 17545, 24318, 0 | 0 – 98,092 | 0 – 70,004 |
| `temporal_cycle_count` | 4232395, 31335496, 20586329, 34430 | 34,430 – 31,335,496 | 0 – 775,432 |
| `earliest_cycle_completion_days` | 4, 47, 10, 294 | 4 – 294 | −1 – 695 |

### Finding 3.1 — Time-respecting cycle counts do not cleanly separate core entities

Neither topological feature separates the classes. For `temporal_cycle_count` the
ranges overlap: the lowest core value (34,430) sits **below** the highest non-core
value (775,432). `temporal_reciprocal_count` overlaps more severely still — one
core entity scores 0, while non-core entities reach 70,004.

Neither topological feature achieves the clean separation that
`reciprocal_out_count` does. This is consistent with the decision to keep TDA as a
separate research component rather than folding it into the shipped feature set.

### Finding 3.2 — Cycle enumeration is combinatorially expensive

Longer-cycle counts reach 3.1 × 10⁷ on a 3,440-node graph. Any production cycle
detection will require bounding by length, time window, or amount.

> **These are transaction-combination counts, not structure counts.** TGA
> enumerates one ordered choice of transaction per leg, so parallel transactions
> multiply out: three entities with 50 transactions per leg contribute up to
> 125,000 to `temporal_cycle_count`. Layer 10 (`src/detection/cycles.py`) counts
> the same situation as **one** cycle with three legs, and finds 5,589
> entity-level cycles in the whole reference graph at length ≤ 6. The two
> numbers differ by orders of magnitude because they measure different
> quantities; neither corrects the other and they must not be presented as
> conflicting. See `docs/architecture.md` §22.1 for the full comparison.

### Finding 3.3 — Day-level date precision limits causal claims

The reference dataset's dates have **day-level precision only**. Same-day
inflow/outflow co-occurrence therefore cannot be interpreted as causal
pass-through: it is not possible to establish that an inflow preceded an outflow
when both fall on the same calendar day. This constraint is recorded in
`docs/architecture.md` §12.2 and applies to any temporal feature built from this
data.

### Finding 3.4 — TGA features produce no measured uplift in a controlled ablation

**Source:** `scripts/legacy/run_exp5.py`; captured output in `outputs/experiments/exp5_report.txt`.

Experiment 5 adds the three TGA features to the 160-feature baseline (163 total)
under the protocol used for Experiments 1–4. Measured deltas against
Experiment 1:

| Metric | Exp 1 | Exp 5 | Delta |
|---|---|---|---|
| PR-AUC | 0.7534 ± 0.4271 | 0.7534 ± 0.4272 | `-0.0000` |
| ROC-AUC | 0.9790 ± 0.0363 | 0.9788 ± 0.0368 | `-0.0003` |
| F1 | 0.7500 ± 0.4330 | 0.6250 ± 0.4146 | `-0.1250` |
| Precision | 0.7500 ± 0.4330 | 0.5833 ± 0.4330 | `-0.1667` |
| Recall | 0.7500 ± 0.4330 | 0.7500 ± 0.4330 | `+0.0000` |

**The three TGA features add no predictive value over the baseline, and the
threshold-dependent metrics get worse.** Against Experiment 3 the configuration
is worse on every reported metric (`PR-AUC -0.2466`, `ROC-AUC -0.0212`,
`F1 -0.3750`).

This is a **negative result and is recorded as one**. It independently confirms
Finding 3.1, which reached the same conclusion from the distributional overlap
rather than from a trained model. Two facts constrain how far it can be read:

* Per-fold PR-AUC is `0.0135, 1.0000, 1.0000, 1.0000` — a single fold carries
  the entire mean. With one positive entity per test fold, the F1 and precision
  deltas are threshold artefacts on a handful of predictions, not a stable
  measurement of feature quality.
* The result speaks only to *these three cycle-count features*, on *this*
  dataset, under *this* protocol. It is not evidence that topological structure
  is uninformative in general, and persistence-based TDA remains unimplemented
  and unevaluated (`docs/architecture.md` §15).

The Experiment 1 comparison arm inside the same run reproduced the recorded
Experiment 1 metrics exactly, and the whole experiment was executed twice with
bit-identical results, so the deltas are not run-to-run noise.

---

## 4. Known redundancy retained deliberately

**Source:** `docs/architecture.md` §28.2; `src/features/multigraph.py`.

`reciprocal_out_count` and `reciprocal_in_count` are mathematically identical in
the current implementation. Both are retained in the 167-feature input tensor to
test the 7-feature specification exactly as designed. This is a deliberate,
documented redundancy — not an oversight — and neither feature may be removed or
redesigned without invalidating the frozen checkpoint.

Similarly, `parallel_temporal_dispersion` uses transaction dates. In the static
transductive node-classification setting of Experiment 3 this is a retrospective
computation and is not leakage. **Causal temporal prediction in production would
require proper `as_of` temporal filtration**, which is not implemented.

---

## Not yet established

The following are **open**. They are listed so that absent evidence is visible
rather than assumed.

* **No external literature review.** No survey of published graph-based AML
  methods, no benchmark comparison against published baselines, and no citations
  exist in this repository. Nothing in this project's performance has been
  positioned relative to prior work.
* **No generalization evidence.** Findings 1.1-3.4 come from one dataset with 4
  positive entities. Three further labelled datasets have since been audited
  (`docs/datasets.md`) and models have been run on SAML-D, but none of them is
  `MODEL_SELECTION_READY`, so no claim about transfer or generalization is
  supported by any of it.
* **No dataset that can rank models.** Established 2026-09-06 and now the
  project's blocking dependency: the one real dataset has 4 positives, and the
  one large AML dataset has entity labels an untuned probe recovers from graph
  statistics at average precision 0.9971. See `docs/datasets.md` and
  `docs/model_selection.md`. Until this is resolved, no architectural
  comparison in this project can be decided on evidence.
  **Resolved 2026-09-06:** the IBM AMLworld HI-Small ledger is
  `MODEL_SELECTION_READY` and now ranks models (`docs/model_selection.md`
  §I). The findings above this list predate it.
* ~~**No evidence that retrospective detection transfers across periods.**~~
  **Answered 2026-09-12 (R1), with limits.** A model fitted on one completed
  2-day detection window keeps **97-98 %** of its same-period average precision
  on a comparable next window (AP `0.3145 ± 0.0050` and `0.3425 ± 0.0034`,
  17-20x the label-free floor, FP/TP 2-3 at its frozen threshold) and **11 %**
  on a window whose composition differs materially (AP `0.0396`, FP/TP 74).
  Transfer is not entity memorisation: new entities score as well as seen ones,
  and first-time-flagged populations reach AP 0.26-0.29 where transfer holds.
  Verdict **LIMITED TEMPORAL TRANSFER**; record
  `docs/r1_detection_transfer.md`. **R6 closed half of what was open**: the
  collapse is *not* caused by window-composition encoding in `aml_temporal_v1`.
  Re-expressing that family on the window's own activity clock
  (`aml_temporal_v2`) lifted W3->W4 retention only 0.111 -> 0.192 while dropping
  W1->W2 from 0.975 -> 0.881, and left family-level drift unchanged (median
  0.2722 -> 0.2717); `aml_temporal_v1` is retained
  (`docs/r6_temporal_v2.md`). Still open: the tail adjacency and population
  composition of the last window, and everything about real data.
* **No measurement of graph analytics as investigation evidence.** Whether
  cycles, flows and paths around an alert surface laundering more often for
  true than false alerts is roadmap R2. Graph *features* are measured: +0.0093
  AP retrospectively, −0.0024 ± 0.0027 when forecasting.
* **No transaction-level result.** IBM publishes transaction labels; every
  result in this repository classifies entities (roadmap R3).
* **No causal account of the Experiment 4 degradation.** Adding temporal features
  to Experiment 3 destroyed its separation. The mechanism is unexplained.
* **No statistical comparison of external datasets to the training
  distribution.** The compatibility report emits a standing warning to this
  effect: *"Dataset characteristics have not been statistically tested against
  training baseline."*
* **No validation of the frozen country vocabulary against non-reference data.**
  How much signal is lost when an external dataset's countries fall outside the
  151-code vocabulary has not been measured.
* **No evaluation of `ACCOUNT_ID` identity resolution at scale.** It is unit
  tested on a 2-row fixture only; its effect on graph topology and feature
  distributions on a real account-keyed dataset is unmeasured.
* **Cycle and subgraph detection are not evaluated as a detection capability.**
  Experiment 5 (Finding 3.4) measured the predictive contribution of three
  cycle-*derived features* and found none. That is a different question from
  detection quality: no cycle or subgraph has been evaluated as investigator
  output — there is no measurement of what such detection surfaces, its
  precision, or its usefulness — and it is not part of the shipped inference
  path.

---

## Reproducing these findings

The dataset audit and the Layer 6 model experiments are reproduced separately;
see `docs/datasets.md` §8 and `docs/model_selection.md`. The Experiment 1-5
findings above are reproduced by the original scripts:

```bash
python -m scripts.legacy.run_exp1        # Baseline, 160 features
python -m scripts.legacy.run_exp2        # + Temporal, 166
python -m scripts.legacy.run_exp3        # + Directed-Multigraph, 167  (shipping configuration)
python -m scripts.legacy.run_exp4        # + Both, 173
python -m scripts.legacy.run_exp5        # + TGA, 163
python -m scripts.legacy.audit_multigraph
python -m scripts.legacy.audit_tga       # output captured in scripts/legacy/audit_tga.txt
python -m scripts.legacy.feature_audit
```

All read the reference dataset from `../dkdata.json`.

Each script prints its report to stdout. The committed Experiment 5 artifact was
produced by redirecting that output:

```bash
PYTHONIOENCODING=utf-8 python -m scripts.legacy.run_exp5 > outputs/experiments/exp5_report.txt 2>&1
```
