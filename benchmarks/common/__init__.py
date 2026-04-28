"""Shared benchmarking machinery for orbitflows.

A benchmark consists of:
    1. A YAML config declaring fixed params + a sweep grid.
    2. A short driver script (``run.py``) that calls :func:`run_sweep`.
    3. A results directory containing ``summary.parquet`` and per-run
       artifacts (loss curve, checkpoint).
    4. An analysis notebook that loads ``summary.parquet`` and plots.

The runner here implements the **quickstart protocol**:
``generate_sho_orbits`` -> ``HamiltonianMappingModel`` -> ``model.train``.
"""

from .config import load_config, expand_sweep, flatten_dict, hash_config
from .registry import POTENTIALS, LAYERS, CONDITIONERS, OPTIMIZERS, SCHEDULERS, ACTIVATIONS
from .runner import run_one, run_sweep
from .io import append_result, load_results
from .hardware import get_hardware_info

__all__ = [
    "load_config",
    "expand_sweep",
    "flatten_dict",
    "hash_config",
    "POTENTIALS",
    "LAYERS",
    "CONDITIONERS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "ACTIVATIONS",
    "run_one",
    "run_sweep",
    "append_result",
    "load_results",
    "get_hardware_info",
]
