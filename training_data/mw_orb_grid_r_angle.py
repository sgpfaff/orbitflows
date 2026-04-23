'''
Create a grid of orbits in the vertical MWPotential2014 at 
the solar radius, equally spaced in apocenter radius and angle.
'''

from galpy.actionAngle import actionAngleVerticalInverse 
from galpy.potential import MWPotential2014, toVerticalPotential
import numpy as np
from orbitflows.dynamics import H, MWPotential2014_1D
import torch

def generate_grid(n_orbits, nsteps, zmin, zmax):
    '''
    Generate a grid of orbits in the vertical MWPotential2014 at the solar radius,
    
    Parameters
    ----------
    n_orbits : int
        Number of orbits to generate (equally spaced in apocenter radius)
    nsteps : int
        Number of time steps to integrate each orbit for (equally spaced in angle)
    
    Returns
    -------
    xvs : np.ndarray
        Array of shape (n_orbits, nsteps, 2) containing the (z, vz) coordinates 
        of each orbit at each time step.
    aa : np.ndarray
        Array of shape (n_orbits, nsteps, 2) containing the (angle, action) coordinates 
        of each orbit at each time step.
    '''
    # define the potential
    mw = MWPotential2014
    mwvert = toVerticalPotential(mw, 1) # vertical MW potential at solar radius

    # create grid of radii and velocities
    zs = torch.linspace(zmin, zmax, n_orbits)
    vs = torch.zeros_like(zs)

    # compute the energies at the phase-space points
    Es = H(torch.vstack((zs, vs)).T, MWPotential2014_1D).numpy()

    # compute the actions and frequencies at these energies
    aA1Dinv= actionAngleVerticalInverse(pot=mwvert,nta=4*128,
                                            Es=Es,
                                            use_pointtransform=False)
    actions = np.array([aA1Dinv.J(E) for E in Es])
    action_list = np.repeat(actions, nsteps).reshape(n_orbits, nsteps)     

    # define equally spaced angles
    angles = np.linspace(0, 2 * np.pi, nsteps)
    angle_list = np.repeat(angles, n_orbits).reshape(nsteps, n_orbits).T

    # combine actions and angles in single array
    aa = np.array([angle_list, action_list]).T.swapaxes(0,1)

    # compute (z, vz) coordinates for each (action, angle) pair
    xvs = np.array([np.array(aA1Dinv(action,angles)) for action in actions]).swapaxes(1,2)

    return xvs, aa

if __name__ == "__main__":
    xvs, aa = generate_grid(n_orbits=10, nsteps=1000, zmin=0.025, zmax=0.25)
    torch.save(torch.tensor(xvs), 'mw_ps_r_angle_grid.pt')
    torch.save(torch.tensor(aa), 'mw_aa_r_angle_grid.pt')

