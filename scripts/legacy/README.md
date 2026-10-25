# Legacy research scripts

The original Experiment 1-5 runs and the research audits behind the findings
in `docs/research.md`. They are kept as a historical record, not as part of the
current pipeline: new experiments go through `run_experiment.py` and
`configs/experiments/`.

Run them from the repository root as modules, so `src` is importable:

```bash
python -m scripts.legacy.run_exp3
```

All of them read the Reference Dataset from `../dkdata.json`.

| Script | Purpose |
|---|---|
| `run_exp1.py` … `run_exp5.py` | Experiments 1-5: baseline, + temporal, + directed multigraph (shipping configuration), + both, + TGA |
| `train_checkpoint.py` | Trains and saves the frozen Experiment 3 checkpoint |
| `audit_exp3.py`, `audit_exp4.py`, `audit_gnn.py` | Label-leakage audits |
| `audit_multigraph.py`, `audit_temporal.py` | Feature-family attribution audits |
| `audit_tga.py`, `audit_tga_correlations.py` | TGA cost and correlation audits (output in `audit_tga.txt`) |
| `analyze_33_ee.py`, `analyze_conflicts*.py`, `feature_audit.py`, `investigate_cores.py` | Entity-resolution and core-entity investigations |
