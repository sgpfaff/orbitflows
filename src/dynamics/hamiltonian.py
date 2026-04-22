'''
Hamiltonian functions of various potentials, as a function of given phase-space points .
'''

import numpy as np
import torch
import matplotlib.pyplot as plt

def H(ps, potential_for_H):
    '''
    Hamiltonian of a phase-space point in a given potnetial.

    Parameters:
    -----------
    ps_point (array): (x, v) phase-space point

    potential (galpy Potential)

    Returns:
    -------
    float: Hamiltonian value at the given phase-space point
    '''
    if ps.shape[-1] == 2:
        q, p = ps[..., 0], ps[..., 1]
        return 0.5*p**2 + potential_for_H(q)
    elif ps.shape[-1] == 4:
        x, y, vx, vy = ps[..., 0], ps[..., 1], ps[..., 2], ps[..., 3]
        r = np.sqrt(x**2 + y**2)
        return 0.5*(vx**2 + vy**2) + potential_for_H(r)
    elif ps.shape[-1] == 6:
        x, y, z, vx, vy, vz = ps[..., 0], ps[..., 1], ps[..., 2], ps[..., 3], ps[..., 4], ps[..., 5]
        #radius = np.sqrt(x**2 + y**2 + z**2)
        R = torch.sqrt(x**2 + y**2)
        return 0.5*(vx**2 + vy**2 + vz**2) + potential_for_H(z, R)

def H_sho(ps, omega=1):
    '''
    Hamiltonian function for the harmonic oscillator.

    Parameters:
    ps_point (array): (x, v) phase-space point. Shape (Orbit dims, 2) where orbit dims
                        can be the number of ps points the dims separating several orbits

    Returns:
    float: Hamiltonian value at the given phase-space point
    '''
    q, p = ps[...,0], ps[...,1]
    return 0.5*p**2 + 0.5*(omega*q)**2