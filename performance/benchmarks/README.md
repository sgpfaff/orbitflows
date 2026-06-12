# orbitflows benchmarks

Each subdirectory is **one experimental question**. Every benchmark has
the same shape:

```
<benchmark>/
    config.yaml      # what to run (inherits ../configs/_base.yaml)
    run.py           # thin driver: load config -> run_sweep
    analyze.ipynb    # plots from results/summary.parquet
    results/         # gitignored except summary.{parquet,csv}
        summary.parquet
        runs/<run_id>/{config.yaml, loss_curve.npy, lr_curve.npy}
```

The shared machinery lives in `common/`:

| module | role |
| --- | --- |
| `config.py`   | YAML loading, `defaults:` inheritance, sweep expansion, run hashing. |
| `registry.py` | `name -> class` lookup tables for models, layers, conditioners, optimizers, schedulers, potentials, losses. |
| `runner.py`   | `run_one(cfg)` and `run_sweep(cfg)`. Implements the **quickstart protocol** end to end. |
| `metrics.py`  | Held-out energy-conservation metrics (mirrors the quickstart diagnostics). |
| `io.py`       | Append-safe Parquet (+ CSV) results store. |
| `hardware.py` | Captures git SHA, host, torch/CUDA versions, GPU name. |

## Adding a new benchmark

1. `mkdir benchmarks/<name>/results && touch benchmarks/<name>/results/.gitkeep`
2. Copy `scaling_train_size/{config.yaml,run.py,README.md,analyze.ipynb}` and edit:
   - rename `name:` in `config.yaml`
   - change the `sweep:` axes
   - update plotting variables in `analyze.ipynb`
3. `python benchmarks/<name>/run.py`

If the new question can't be expressed by editing `sweep:`/`fixed:` and
needs new logic, add it to `common/` (not the benchmark folder) so other
benchmarks can share it.

## Conventions

- **YAML for what, Python for how.** Configs never `import`; drivers
  never hard-code values.
- **Reproducibility.** Each row carries `git_sha`, `git_dirty`,
  `hostname`, `torch_version`, `device`, `seed`. Don't merge results
  across `git_dirty=True` runs without checking.
- **Idempotence.** Runs are content-addressed by `run_id =
  hash(flat_config)`; reruns skip already-recorded rows. Use
  `--force` to override.
- **Aggregate at plot time.** Always store one row per (config, seed);
  reduce across seeds only in the notebook.
- **Heavy artifacts are local.** `results/runs/` is gitignored;
  `summary.parquet` + `summary.csv` are the only things meant to be
  committed (and even those are optional).

## Suggested benchmark roadmap

| folder | question |
| --- | --- |
| `scaling_train_size/`     | Loss & wall time vs. number of training orbits. *(Included as the worked example.)* |
| `scaling_model_size/`     | Sweep `model.n_layers`, `conditioner.args.projection_dims`. Pareto: parameters vs. error. |
| `hparam_search_flow/`     | Random search over `lr`, `n_layers`, conditioner depth/width. |
| `convergence_vs_steps/`   | Held-out error every N training steps. Adds a callback to `runner.py`. |
| `potential_robustness/`   | Fix model, vary target potential. Bar chart of held-out error per potential. |
| `model_class_comparison/` | `HamiltonianMappingModel` vs. `TorusMappingModel`. |

## Smoke test

A tiny pytest can short-circuit every config (`training.steps=5,
seeds=[0]`) and assert it runs. Add it to `benchmarks/test_smoke.py`
when you start running the full sweeps in CI.
