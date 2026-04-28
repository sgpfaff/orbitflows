"""Driver for the `scaling_train_size` benchmark.

Usage
-----
    python run.py                  # run every (sweep point, seed)
    python run.py --force          # ignore existing results
    python run.py --dry-run        # print configs without training

Results land in ``./results/summary.parquet`` (+ summary.csv mirror) and
per-run artifacts under ``./results/runs/<run_id>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Make `benchmarks.common` importable when running this script directly.
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import expand_sweep, load_config, run_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--force", action="store_true", help="Re-run even if run_id already exists")
    parser.add_argument("--dry-run", action="store_true", help="Only enumerate configs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        for i, run_cfg in enumerate(expand_sweep(cfg)):
            print(f"[{i}] seed={run_cfg['seed']}  n_orbits={run_cfg['data']['n_orbits']}")
        return

    run_sweep(cfg, results_dir=args.results_dir, progress=True, skip_existing=not args.force)


if __name__ == "__main__":
    main()
