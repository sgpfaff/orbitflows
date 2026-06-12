"""Config loading, sweep expansion, and hashing.

Conventions
-----------
- A config is a YAML file with at least three top-level keys:
    ``fixed`` : nested dict of parameters that don't vary.
    ``sweep``  : optional dict of ``"dotted.key": [v1, v2, ...]``;
                 the cartesian product is expanded.
    ``seeds`` : optional list of ints; each sweep point is run once
                 per seed.
- A config may include ``defaults: <relative path>`` which is loaded
  first and then deep-merged with the current file (current wins).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, recursively resolving ``defaults``."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if "defaults" in cfg:
        defaults_path = (path.parent / cfg.pop("defaults")).resolve()
        base = load_config(defaults_path)
        cfg = _deep_merge(base, cfg)

    cfg.setdefault("fixed", {})
    cfg.setdefault("sweep", {})
    cfg.setdefault("seeds", [0])
    return cfg


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict using dotted keys (``a.b.c``)."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def expand_sweep(cfg: dict) -> Iterator[dict]:
    """Yield one resolved config dict per (sweep point, seed)."""
    sweep = cfg.get("sweep", {}) or {}
    seeds = cfg.get("seeds", [0])
    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]

    # Top-level keys (device, dtype, etc.) that are not fixed/sweep/seeds
    # are carried into each run config.
    top_level = {
        k: v for k, v in cfg.items()
        if k not in ("fixed", "sweep", "seeds", "defaults", "name")
    }

    grid = list(itertools.product(*value_lists)) if keys else [()]
    for combo in grid:
        for seed in seeds:
            run_cfg = deepcopy(cfg["fixed"])
            run_cfg.update(deepcopy(top_level))
            for k, v in zip(keys, combo):
                _set_dotted(run_cfg, k, v)
            run_cfg["seed"] = int(seed)
            run_cfg["_name"] = cfg.get("name", "unnamed")
            yield run_cfg


def hash_config(cfg: dict, length: int = 10) -> str:
    """Stable short hash of a config dict (for run IDs)."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]
