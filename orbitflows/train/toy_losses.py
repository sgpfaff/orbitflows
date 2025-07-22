'''Loss functions for training with a toy guess.'''

import torch
from ..utils import H

def scaled_H_std(ps, targetPotential):
    orbit_energies = H(ps, targetPotential)
    orbit_std = torch.std(orbit_energies, axis=-1)
    return (orbit_std/torch.abs(torch.mean(orbit_energies, axis=-1))).sum() # made the devision by the energies an absolute value recently

def H_std(ps, targetPotential):
    orbit_energies = H(ps, targetPotential)
    orbit_std = torch.std(orbit_energies, axis=-1)
    return (orbit_std).sum()