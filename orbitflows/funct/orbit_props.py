'''
Calculate properties of orbits
'''

import numpy as np
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
    x, v = ps.T
    return 0.5*v**2 + potential_for_H(x)

def H_sho(ps, omega=1):
    '''
    Hamiltonian function for the harmonic oscillator.

    Parameters:
    ps_point (array): (x, v) phase-space point. Shape (Orbit dims, 2) where orbit dims
                        can be the number of ps points the dims separating several orbits

    Returns:
    float: Hamiltonian value at the given phase-space point
    '''
    x, v = ps.T
    return 0.5*v**2 + 0.5*(omega*x)**2
