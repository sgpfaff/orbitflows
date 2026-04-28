"""String -> object lookup tables.

Configs reference everything by name so that YAML never has to import
Python. Add new models/potentials/etc. here exactly once.
"""

from __future__ import annotations

import torch

from orbitflows import (
    HamiltonianMappingModel,
    TorusMappingModel,
    SymplecticCouplingLayer,
    SimpleNNConditioner,
)
from orbitflows.dynamics import MWPotential2014_1D
from orbitflows.util import scaled_H_std, mean_H_rms, H_std


MODELS: dict[str, type] = {
    "HamiltonianMappingModel": HamiltonianMappingModel,
    "TorusMappingModel": TorusMappingModel,
}

POTENTIALS: dict[str, callable] = {
    "MWPotential2014_1D": MWPotential2014_1D,
}

LAYERS: dict[str, type] = {
    "SymplecticCouplingLayer": SymplecticCouplingLayer,
}

CONDITIONERS: dict[str, type] = {
    "SimpleNNConditioner": SimpleNNConditioner,
}

OPTIMIZERS: dict[str, type] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}

SCHEDULERS: dict[str, type] = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
}

ACTIVATIONS: dict[str, type] = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "GELU": torch.nn.GELU,
    "SiLU": torch.nn.SiLU,
    "LeakyReLU": torch.nn.LeakyReLU,
}

LOSSES: dict[str, callable] = {
    "scaled_H_std": scaled_H_std,
    "mean_H_rms": mean_H_rms,
    "H_std": H_std,
}
