# Experiment specifications

One JSON file per experiment.  Each is a complete, machine-readable
`src.experiments.spec.ExperimentSpec`; nothing about an experiment lives in
Python.

    python run_experiment.py configs/experiments/<name>.json
    python run_experiment.py configs/experiments/samld_*.json
    python compare_experiments.py outputs/experiments/samld_*.json

The `dataset` block's `name` selects the loader in `src/experiments/sources.py`.
The rest of the block states intent only: `run_experiment.py` replaces it with
the identity the loader verified, so a stale version string in a config can
never mislabel an artifact.  A dataset is loaded once per invocation however
many configs name it.

## Reference Dataset (`reference_dkdata`)

Four positives.  These runs are a **regression signal**, not a comparison.  See
`docs/datasets.md` §2 for why, and do not quote their numbers as detection
capability.

| file | question it answers |
| --- | --- |
| `heuristic_floor.json` | What does an unsupervised AML rule score achieve on its own? Any supervised model that cannot beat this has learned nothing. |
| `logistic_aml_core.json` | Does a linear model on interpretable AML features beat the rule score? |
| `boosting_aml_core.json` | Does a gradient-boosted tree ensemble add anything over the linear model? |
| `legacy_graphsage_exp3.json` | How does the Experiment 3 architecture and its 167-column representation score under the same protocol as everything else? |
| `time_forward_boosting.json` | What survives when the model may only use transactions that had already happened? Every test partition here holds zero positives, which is the finding, not a failure. |

## SAML-D (`samld`)

855,460 entities, 9,504,852 transactions, 7,902 positive entities, synthetic.
Classified `STRUCTURAL_VALIDATION_ONLY`: an untuned probe on simple structural
quantities alone reaches average precision 0.9971, so **these runs do not rank
models**.  What they do establish is that the whole stack — loaders, features,
splits, candidates, metrics, leakage audits — is correct and affordable at that
scale, and what the label-free floor is.

### Main comparison — stratified 4-fold, global feature scope

| file | question it answers |
| --- | --- |
| `samld_heuristic_floor.json` | The label-free floor. No labels are read to produce the score. |
| `samld_logistic_aml.json` | What a linear model on the full AML feature set reaches. |
| `samld_boosting_behaviour.json` | What boosting reaches on transaction, temporal, flow and context features **without any graph structure**. |
| `samld_boosting_graph.json` | The same model with `aml_network_v1` and `aml_cycle_v1` added. Also ablation cell E. |

### Dispersion

| file | question it answers |
| --- | --- |
| `samld_heuristic_floor_seed1337.json` | The floor under a second fold assignment. The score has no seed; the folds do. |
| `samld_boosting_graph_seed1337.json` | The full model under a second seed, so the fold-to-fold spread can be read against a run-to-run spread. |

### Time-forward — `time_forward_holdout`, `train_window` feature scope

Entities are placed by **last** activity, and features may only use transactions
before the training boundary.  This is the operational question: can behaviour
up to a boundary predict laundering involvement realised after it?

| file | question it answers |
| --- | --- |
| `samld_tf_boosting_graph.json` | The full model under the operational regime. |
| `samld_tf_heuristic_floor.json` | The floor under the same regime, so the comparison stays within one regime. |

### Ablation — which information source contributes?

Same model, same split, same seed; only the feature families change.

| cell | file | families |
| --- | --- | --- |
| A | `samld_ablation_a_txn.json` | `aml_txn_v1` |
| B | `samld_ablation_b_txn_temporal.json` | `aml_txn_v1`, `aml_temporal_v1` |
| C | `samld_ablation_c_txn_flow.json` | `aml_txn_v1`, `aml_flow_v1` |
| D | `samld_ablation_d_txn_network.json` | `aml_txn_v1`, `aml_network_v1`, `aml_cycle_v1` |
| E | `samld_boosting_graph.json` | all six |

## IBM AMLworld HI-Small (`ibm_aml_hi_small`) — **the model-selection benchmark**

515,088 entities, 5,078,345 transactions, 6,357 positive entities, synthetic,
17 days.  Classified `MODEL_SELECTION_READY`: the structural probe reaches
average precision 0.2098 against a 0.0123 prevalence floor and there are no
class-free bands, so **these runs do rank models**.

The dataset is not committed.  Acquire it first:

```bash
python acquire_dataset.py ibm_aml_hi_small
```

### Main comparison — stratified 4-fold, global feature scope

| file | question it answers |
| --- | --- |
| `ibm_heuristic_floor.json` | The label-free floor. No labels are read to produce the score. |
| `ibm_logistic_aml.json` | What a linear model on the full AML feature set reaches. |
| `ibm_boosting_behaviour.json` | What boosting reaches **without any graph structure**. |
| `ibm_boosting_graph.json` | The same model with `aml_network_v1` and `aml_cycle_v1` added. Also ablation cell E. |

### Dispersion

`ibm_heuristic_floor_seed1337.json`, `ibm_boosting_graph_seed1337.json` — a
second fold assignment, so the fold spread can be read against a run-to-run
spread.

### Time-forward — `time_forward_holdout`, `train_window` feature scope

Entities are placed by **last** activity.  Placing them by *first* activity is
degenerate here: half of all entities first transact within 28 minutes of the
file's start, so a 60th-percentile cut leaves a four-hour training window.

`ibm_tf_heuristic_floor.json`, `ibm_tf_boosting_graph.json`.

### Ablation

| cell | file | families |
| --- | --- | --- |
| A | `ibm_ablation_a_txn.json` | `aml_txn_v1` |
| B | `ibm_ablation_b_txn_temporal.json` | `aml_txn_v1`, `aml_temporal_v1` |
| C | `ibm_ablation_c_txn_flow.json` | `aml_txn_v1`, `aml_flow_v1` |
| D | `ibm_ablation_d_txn_network.json` | `aml_txn_v1`, `aml_network_v1`, `aml_cycle_v1` |
| E | `ibm_boosting_graph.json` | all six |

### Calendar-forward — `calendar_forward`, `origin_window` feature scope

Every entity with history is scored at fixed origins - train 2022-09-05,
validation 2022-09-07, test 2022-09-09 - from the transactions before each
origin. The label is involvement in a flagged transaction in the two days after
the origin, and the last window ends 2022-09-11, before the days 11-18 tail in
which normal traffic has stopped. Design and rationale:
`docs/calendar_forward_benchmark.md`.

| file | question it answers |
| --- | --- |
| `ibm_cal_heuristic_floor.json` | The label-free floor, ranked within each origin's population. |
| `ibm_cal_logistic_aml.json` | A linear model on all six families. |
| `ibm_cal_boosting_behaviour.json` | Boosting without graph structure. |
| `ibm_cal_boosting_graph.json` | Boosting with `aml_network_v1` and `aml_cycle_v1`. Also ablation cell E. |
| `ibm_cal_ablation_{a,b,c,d}_*.json` | The A-D ablation cells under this regime. |
| `ibm_cal_boosting_{graph,behaviour}_seed1337.json` | A second model seed, so the graph increment can be read against a run-to-run spread (a single holdout has no fold dispersion). |

Run them in one invocation - the snapshot for each origin is then computed once
and reused:

```bash
python run_experiment.py configs/experiments/ibm_cal_*.json
```

## Control (`financial_fraud_kaggle`)

Two runs on the one available dataset whose labels are **not** structurally
trivial (structural probe average precision 0.4226 against a prevalence floor of
0.3285).  Its domain is card-not-present fraud, not laundering, so it is
`STRUCTURAL_VALIDATION_ONLY` and its numbers say nothing about AML.

Their purpose is narrower and worth stating: if the same protocol, the same
features and the same candidates land far below 1.0 here while saturating on
SAML-D, then SAML-D's saturation is a property of SAML-D and not of a broken
evaluation stack.

| file | question it answers |
| --- | --- |
| `ffk_heuristic_floor.json` | What the label-free floor reaches on a non-trivial label. |
| `ffk_boosting_graph.json` | What the full supervised model reaches on the same. |

## Before reading any number

1. **Check `dataset.spec.synthetic`.** SAML-D is generated data. No number from
   it is evidence about real-world detection performance.

2. **Check `split.regime` and `features.scope`.** An `entity-disjoint`, a
   `time-forward` and a `calendar-forward` result answer different questions
   and must not share a column. `compare_experiments.py` enforces the grouping.

3. **Check `aggregate.average_precision.n_defined`.** With four positives in the
   Reference Dataset, most folds cannot define a precision-recall curve at all,
   and a mean over the one fold that could is not an estimate of anything.

4. **Check the dataset's verdict in `docs/datasets.md`.** A ranking produced on a
   dataset classified `STRUCTURAL_VALIDATION_ONLY` is not a weak ranking. It is
   not a ranking.
