"""Eval metrics that mirror the quickstart notebook diagnostics.

The quickstart evaluates a trained model by generating a held-out set of
SHO orbits, mapping their action-angle pairs to phase space with the
model, computing ``H(target_potential, ps)`` along each orbit, and
reporting the mean / max relative deviation along orbits.
"""

from __future__ import annotations

import torch

from orbitflows.dynamics import H
from orbitflows.util import generate_sho_orbits, max_error_along_orbs, mean_error_along_orbs


@torch.no_grad()
def evaluate_metrics(
    model,
    *,
    n_orbits: int,
    n_steps: int,
    omega: float,
    r_bounds: tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Compute mean/max along-orbit energy errors on a held-out set."""
    _, eval_aa = generate_sho_orbits(
        n_orbits=n_orbits,
        omega=omega,
        t_end=2 * torch.pi / omega,
        n_steps=n_steps,
        r_bounds=torch.tensor(list(r_bounds)),
    )
    #eval_ps = eval_ps.to(device=device, dtype=dtype)
    eval_aa = eval_aa.to(device=device, dtype=dtype)

    model_ps = model.aa_to_ps(eval_aa)
    H_model = H(model_ps, model.targetPotential)

    return {
        "eval_mean_error_mean": float(mean_error_along_orbs(H_model)[1:].mean()),
        "eval_mean_error_max": float(mean_error_along_orbs(H_model)[1:].max()),
        "eval_max_error_mean": float(max_error_along_orbs(H_model)[1:].mean()),
        "eval_max_error_max": float(max_error_along_orbs(H_model)[1:].max()),
    }


def loss_summary(loss_list: list[float], tail_frac: float = 0.1) -> dict:
    """Final / best / tail-mean of the training loss curve."""
    if not loss_list:
        return {"final_loss": float("nan"), "best_loss": float("nan"), "tail_mean_loss": float("nan")}
    n_tail = max(1, int(len(loss_list) * tail_frac))
    return {
        "final_loss": float(loss_list[-1]),
        "best_loss": float(min(loss_list)),
        "tail_mean_loss": float(sum(loss_list[-n_tail:]) / n_tail),
        "tail_std_loss": float(torch.tensor(loss_list[-n_tail:]).std().item()),
    }
