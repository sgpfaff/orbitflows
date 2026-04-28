'''
Create a grid of orbits in the vertical MWPotential2014 at 
the solar radius, equally spaced in action and angle.
'''

from galpy.actionAngle import actionAngleVerticalInverse 
from galpy.potential import MWPotential2014, toVerticalPotential
import numpy as np
from orbitflows.dynamics import H, MWPotential2014_1D
import torch
from tqdm import tqdm

def wrap_angles(aa):
    '''
    Wrap angles to [0, 2π) range
    '''
    below_zero_mask = aa[...,0] > np.pi
    aa[...,0][below_zero_mask] = aa[...,0][below_zero_mask] - 2*np.pi
    return aa

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
    freqs : np.ndarray
        Array of shape (n_orbits,) containing the frequencies of each orbit.
    '''
    # define the potential
    mw = MWPotential2014
    mwvert = toVerticalPotential(mw, 1) # vertical MW potential at solar radius

    # compute min and max energies for the grid based on zmin and zmax
    zs = torch.tensor([zmin, zmax]) # min and max radius
    vs = torch.zeros_like(zs)
    E_min_max = H(torch.vstack((zs, vs)).T, MWPotential2014_1D).numpy()
    
    # compute min and max actions for the grid based on these energies
    aA1Dinv_trial = actionAngleVerticalInverse(pot=mwvert,nta=4*128,
                                        Es=E_min_max,
                                        use_pointtransform=False)
    action_min_max = np.array([aA1Dinv_trial.J(E) for E in E_min_max]) # min and max actions
    
    # create grid of actions
    actions = np.linspace(*action_min_max, n_orbits)
    action_list = np.repeat(actions, nsteps).reshape(n_orbits, nsteps)

    # create grid of angles
    angles = np.linspace(0, 2 * np.pi, nsteps)
    angle_list = np.repeat(angles, n_orbits).reshape(nsteps, n_orbits).T

    # combine actions and angles in single array
    aa = np.array([angle_list, action_list]).T.swapaxes(0,1)

    # compute positions and velocities for each (action, angle) pair
    print("setting up action-angle interpolation...")
    aA1Dinv = actionAngleVerticalInverse(pot=mwvert,nta=4*128,
                                        Es=np.linspace(*E_min_max, 100),
                                        setup_interp=True)
    xvs = []
    freqs = []
    for action in tqdm(actions, desc="Computing (z, vz) and frequency for each (action, angle) pair"):
        freqs.append(aA1Dinv.Freqs(action))
        xvs.append(np.array(aA1Dinv(action, angles)))
    freqs = np.array(freqs)
    xvs = np.array(xvs).swapaxes(1,2)
    
    return xvs, wrap_angles(aa), freqs

if __name__ == "__main__":
    xvs, aa, freqs = generate_grid(n_orbits=500, nsteps=5000, zmin=0.025, zmax=0.25)
    torch.save(torch.tensor(xvs), 'mw_ps_action_angle_grid.pt')
    torch.save(torch.tensor(aa), 'mw_aa_action_angle_grid.pt')
    torch.save(torch.tensor(freqs), 'mw_freqs_action_angle_grid.pt')

