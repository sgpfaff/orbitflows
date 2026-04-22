'''Loss functions for training with a toy guess.'''

import torch
from ..dynamics import H

def scaled_H_std(ps, targetPotential):
    orbit_energies = H(ps, targetPotential)
    orbit_std = torch.std(orbit_energies, axis=-1)
    return (orbit_std/torch.abs(torch.mean(orbit_energies, axis=-1))).sum() # made the devision by the energies an absolute value recently

def H_std(ps, targetPotential):
    orbit_energies = H(ps, targetPotential)
    orbit_std = torch.std(orbit_energies, axis=-1)
    return (orbit_std).mean()

def mean_H_rms(ps, targetPotential):
    orbit_energies = H(ps, targetPotential)
    return torch.sqrt(torch.mean(orbit_energies**2, axis=-1)).mean()