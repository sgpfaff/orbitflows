"""Append-safe storage for benchmark results.

Each row is a flat dict. We persist to **Parquet** (preserves dtypes,
small on disk) and also to a sibling **CSV** for easy git-diffing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_result(path: str | Path, row: dict) -> None:
    """Append a single result row to a Parquet file (and CSV mirror)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])

    if path.exists():
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a results Parquet into a DataFrame."""
    return pd.read_parquet(path)
