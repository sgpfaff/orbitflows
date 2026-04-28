"""End-to-end training driver

A run is fully specified by a flat-ish dict (the resolved config) with
groups: ``data``, ``model``, ``conditioner``, ``training``, ``optimizer``,
``scheduler``, ``potential``, ``eval``, plus a top-level ``seed``.

The runner:
    1. Sets seeds and selects device.
    2. Generates SHO training orbits (``generate_sho_orbits``).
    3. Builds a ``HamiltonianMappingModel`` from the registry.
    4. Trains via ``model.train(...)`` -- the same call the notebook uses.
    5. Computes held-out energy-conservation metrics.
    6. Returns one flat row of (config + metrics + provenance).

Each row is also appended to ``results/summary.parquet`` and the
loss curve / model are saved under ``results/runs/<run_id>/``.
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .config import expand_sweep, flatten_dict, hash_config
from .hardware import get_hardware_info
from .io import append_result
from .metrics import evaluate_metrics, loss_summary
from .registry import (
    ACTIVATIONS,
    CONDITIONERS,
    LAYERS,
    LOSSES,
    MODELS,
    OPTIMIZERS,
    POTENTIALS,
    SCHEDULERS,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_potential(cfg: dict, device: torch.device) -> callable:
    name = cfg["name"]
    kwargs = {k: torch.tensor(v, device=device) for k, v in cfg.get("kwargs", {}).items()}
    pot = POTENTIALS[name]
    return partial(pot, **kwargs) if kwargs else pot


def _build_scheduler(cfg: dict | None):
    if cfg is None:
        return None
    name = cfg["class"]
    kwargs = {k: v for k, v in cfg.items() if k != "class"}
    return partial(SCHEDULERS[name], **kwargs)


def _build_model(cfg: dict, target_potential: callable, device: torch.device):
    model_cls = MODELS[cfg["model"]["class"]]
    layer_cls = LAYERS[cfg["model"]["layer_class"]]
    cond_cls = CONDITIONERS[cfg["model"]["conditioner"]["class"]]
    cond_args = deepcopy(cfg["model"]["conditioner"].get("args", {}))
    if "activation" in cond_args and isinstance(cond_args["activation"], str):
        cond_args["activation"] = ACTIVATIONS[cond_args["activation"]]

    optimizer_cls = OPTIMIZERS[cfg["optimizer"]["class"]]
    scheduler = _build_scheduler(cfg.get("scheduler"))

    model = model_cls(
        targetPotential=target_potential,
        input_dim=cfg["model"]["input_dim"],
        n_layers=cfg["model"]["n_layers"],
        omega=cfg["data"]["omega"],
        layer_class=layer_cls,
        conditioner=cond_cls,
        conditioner_args=cond_args,
        optimizer=optimizer_cls,
        scheduler=scheduler,
    )
    model.flow.to(device)
    return model


def _generate_training_data(cfg: dict, device: torch.device, dtype: torch.dtype):
    from orbitflows.util import generate_sho_orbits

    d = cfg["data"]
    omega = d["omega"]
    guess_ps, true_aa = generate_sho_orbits(
        n_orbits=d["n_orbits"],
        omega=omega,
        t_end=2 * torch.pi / omega,
        n_steps=d["n_steps"],
        r_bounds=torch.tensor(list(d["r_bounds"])),
    )
    return (
        guess_ps.to(device=device, dtype=dtype),
        true_aa.to(device=device, dtype=dtype),
    )


def run_one(cfg: dict, *, results_dir: Path | None = None, progress: bool = True) -> dict:
    """Execute a single training run and return a flat result row."""
    cfg = deepcopy(cfg)
    seed = cfg.get("seed", 0)
    _set_seed(seed)

    device = _resolve_device(cfg.get("device"))
    dtype = getattr(torch, cfg.get("dtype", "float32"))

    target_potential = _build_potential(cfg["potential"], device)
    guess_ps, _ = _generate_training_data(cfg, device, dtype)
    model = _build_model(cfg, target_potential, device)

    # --- Train ----------------------------
    train_kwargs = dict(cfg["training"])
    steps = train_kwargs.pop("steps")
    lr = train_kwargs.pop("lr", 1e-3)
    loss_name = train_kwargs.pop("loss", "scaled_H_std")
    loss_function = LOSSES[loss_name]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    model.train(
        guess_ps,
        steps,
        lr=lr,
        loss_function=loss_function,
        update_frequency=train_kwargs.pop("update_frequency", None),
        nested_progress=progress,
        **train_kwargs,
    )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # --- Evaluate -------------------------------------------------------
    e = cfg.get("eval", {})
    eval_metrics = evaluate_metrics(
        model,
        n_orbits=e.get("n_orbits", 100),
        n_steps=e.get("n_steps", 1000),
        omega=cfg["data"]["omega"],
        r_bounds=tuple(e.get("r_bounds", cfg["data"]["r_bounds"])),
        device=device,
        dtype=dtype,
    )

    # --- Build result row ----------------------------------------------
    row: dict = {}
    row.update(flatten_dict(cfg))
    row.update(loss_summary(model.loss_list))
    row.update(eval_metrics)
    row["wall_time_s"] = wall
    row["steps_per_sec"] = steps / wall if wall > 0 else float("nan")
    if device.type == "cuda":
        row["peak_gpu_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)
    row.update(get_hardware_info(device))
    row["run_id"] = hash_config(flatten_dict(cfg))

    # --- Persist per-run artifacts -------------------------------------
    if results_dir is not None:
        run_dir = Path(results_dir) / "runs" / row["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        np.save(run_dir / "loss_curve.npy", np.asarray(model.loss_list))
        if model.lr_list is not None:
            np.save(run_dir / "lr_curve.npy", np.asarray(model.lr_list))

    return row


def run_sweep(
    cfg: dict,
    *,
    results_dir: str | Path,
    skip_existing: bool = True,
    progress: bool = True,
) -> None:
    """Execute every (sweep point, seed) and append rows to summary.parquet."""
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.parquet"

    runs: Iterable[dict] = list(expand_sweep(cfg))
    iterator = tqdm(runs, desc=cfg.get("name", "sweep"), position=0, leave=True) if progress else runs

    seen_ids: set[str] = set()
    if skip_existing and summary_path.exists():
        import pandas as pd

        seen_ids = set(pd.read_parquet(summary_path)["run_id"].tolist())

    for run_cfg in iterator:
        run_id = hash_config(flatten_dict(run_cfg))
        if run_id in seen_ids:
            continue
        row = run_one(run_cfg, results_dir=results_dir, progress=progress)
        append_result(summary_path, row)
        seen_ids.add(run_id)
