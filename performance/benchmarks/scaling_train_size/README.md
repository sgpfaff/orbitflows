# scaling_train_size

**Question:** how do final loss, held-out energy-conservation error, and
wall time depend on the size of the SHO training set?

**Protocol:** the quickstart workflow exactly, with `data.n_orbits`
varied. Every other knob is held at the `_base.yaml` default.

## Run

```bash
cd benchmarks/scaling_train_size
python run.py
```

Re-running is idempotent: a run is skipped if its `run_id` already
appears in `results/summary.parquet`. Use `--force` to rerun.

## Outputs

- `results/summary.parquet` (+ `.csv` mirror) — one row per (sweep
  point, seed) with config + metrics + provenance.
- `results/runs/<run_id>/` — `config.yaml`, `loss_curve.npy`,
  `lr_curve.npy`.

## Analysis

Open `analyze.ipynb`.
