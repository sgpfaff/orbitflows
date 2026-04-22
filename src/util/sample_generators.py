'''
Generate training data and orbit samples
'''

import torch
from ..dynamics import H, H_sho

# galpy
from galpy.actionAngle import actionAngleVertical
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleHarmonic
from galpy.actionAngle import actionAngleHarmonicInverse

import numpy as np
import os
import json

def generate_orbit_ps(n_orbits : int, potential, t_end : float, n_steps : int, 
                      t_start : float = 0.0, r_bounds = torch.tensor([0.1, 0.75])):
    '''
    Generate a set of orbits in phase space for a given potential.
    Integrates the orbits using galpy and returns the phase space values.

    Parameters
    ----------
    n_orbits : int
        number of orbits to generate
    potential : galpy potential object
        potential to integrate the orbits in
    t_end : float
        end time of integration
    n_steps : int
        number of steps in integration
    t_start : float
        start time of integration (default = 0)
    r_bounds : torch.tensor
        bounds of the radius of the orbits (default = [0.1, 0.75])

    Returns
    -------
    qp : torch.tensor
        tensor of shape (n_orbits, n_steps, 2) containing the phase space values of the orbits. 
        The last dimension contains the q and p values, respectively.
    '''
    tl = np.linspace(t_start, t_end, n_steps)
    r = torch.linspace(r_bounds[0], r_bounds[-1], n_orbits)
    ic = np.array([[i,0] for i in r])
    oi = Orbit(ic)
    oi.integrate(tl, potential)
    q = oi.x(tl).flatten()
    p = oi.vx(tl).flatten()
    qp = torch.cat(
        (torch.from_numpy(q)[:,None], 
         torch.from_numpy(p)[:,None]), 
         dim=-1)
    return qp.reshape((n_orbits, n_steps, 2))

def generate_orbits_aa(n_actions : int, n_angles : int, action_min : float, action_max : float):
    '''
    Generate a several orbits of different actions in action-angle space. 
    In practice, this is a grid of points in action-angle space.

    Parameters
    ----------
    n_actions : int
        number of different actions to generate
    n_angles : int
        number of different angles to generate
    action_min : float
        minimum action value to generate
    action_max : float
        maximum action value to generate

    Returns
    -------
    new_batch : torch.tensor
        tensor of shape (n_actions, n_angles, 2) containing the generated action-angle pairs.
        The last dimension contains the angle and action values, respectively.
    '''
    
    new_batch = torch.zeros(n_actions, n_angles, 2)
    angles_to_add = torch.linspace(0, 2*np.pi, n_angles+2)[1:-1]
    actions_to_add = torch.linspace(action_min, action_max, n_actions)
    new_batch[:,:,0] = angles_to_add
    new_batch[:,:,1] = actions_to_add[:, None]
    return new_batch

def generate_orbits(n_orbits : int, potential, t_end : float, n_steps : int, 
                    t_start : float = 0.0, r_bounds = torch.tensor([0.1, 0.75]), 
                    saveData=False, output_dir = None, filename='test_data.pt'):
    '''
    Generate a set of orbits in phase space for a given potential.
    Integrates the orbits using galpy and returns the phase space 
    and action-angle coordinate values.

    Parameters
    ----------
    n_orbits : int
        number of orbits to generate
    potential : galpy potential object
        potential to integrate the orbits in
    t_end : float
        end time of integration
    n_steps : int
        number of steps in integration
    t_start : float
        start time of integration (default = 0)
    r_bounds : torch.tensor
        bounds of the radius of the orbits (default = [0.1, 0.75])
    saveData : bool
        whether to save the generated data (default = False)
    output_dir : str
        directory to save the generated data if saveData is True (default = None)
    filename : str
        name of the file to save the generated data to if saveData is True (default = 'test_data.pt')

    Returns
    -------
    qp : torch.tensor
        tensor of shape (n_orbits, n_steps, 2) containing the phase space values of the orbits. 
        The last dimension contains the q and p values, respectively.
    aa : torch.tensor
        tensor of shape (n_orbits, n_steps, 2) containing the action-angle values of the orbits. 
        The last dimension contains the angle and action values, respectively.
    '''
    tl = np.linspace(t_start, t_end, n_steps)
    aAV = actionAngleVertical(pot=potential)
    r = torch.linspace(r_bounds[0], r_bounds[-1], n_orbits)
    ic = np.array([[i,0] for i in r])
    oi = Orbit(ic)
    oi.integrate(tl, potential)
    
    q = oi.x(tl).flatten()   # Example q values
    p = oi.vx(tl).flatten()   # Example p values
    qp = torch.cat((torch.from_numpy(q)[:,None], torch.from_numpy(p)[:,None]), dim=-1)
    J, _, phi = aAV.actionsFreqsAngles(q, p)
    aa = torch.cat((torch.from_numpy(phi)[:,None], torch.from_numpy(J)[:,None]), dim=-1).to(torch.float32)
        
    if saveData == True:
        model_dir = os.path.join(output_dir, filename)
        os.makedirs(model_dir, exist_ok=True)
        data_path = f'{model_dir}/{filename}.pt'
        torch.save({'phase space':qp, 'action angle':aa}, data_path)
        print(f'Training data saved to {data_path}')

        # Create a configuration dictionary
        potential_str = str(potential)[17:]
        cut_index = potential_str.find('.')
        potential_name = potential_str[:cut_index]
        config = {
            "num_orbits": n_orbits,
            "potential": potential_name,
            "potential_values": vars(potential),
            "integration_start_time": t_start,
            "integration_end_time": t_end,
            "num_steps": n_steps,
            "r_lower_bound": float(r_bounds[0]),
            "r_upper_bound": float(r_bounds[1]),
            "filename": filename
        }
        
        # Save the configuration as a JSON file
        config_path = f'{model_dir}/{filename}_config.json'
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

        print(f"Configuration saved to {config_path}")
    return qp.reshape((n_orbits, n_steps, 2)), aa.reshape((n_orbits, n_steps, 2))

def generate_sho_orbits(n_orbits : int, omega : float, t_end : float, n_steps : int, 
                        t_start : float = 0.0, r_bounds = None, J_bounds = None,
                        split_orbits :bool = False, output_dir = None, saveData : bool =False, filename : str = 'sho_test_data.pt'):
    '''
    Generates set of orbits for a simple harmonic oscillator in phase space and action-angle coordinates.

    Parameters:
    ----------
    n_orbits : int
        number of orbits to generate
    omega : float
        frequency of the SHO
    t_end : float
        end time of integration
    n_steps : int
        number of steps in integration
    t_start : float
        start time of integration (default = 0)
    r_bounds : torch.tensor
        bounds of the radius of the SHO orbits
    J_bounds : torch.tensor
        bounds of the action of the SHO orbits
    split_orbits : bool
        whether to split the output into individual orbits
    saveData : bool
        whether to save the generated data
    filename : str
        name of the file to save the data to
    
    Returns:
    -------
    qp : torch.tensor
        phase space values of the generated orbits [q, p], 
        shape = (num_orbits * n_steps, 2) if split_orbits=True,
        shape = (num_orbits, n_steps, 2) if split_orbits=False
    aa : torch.tensor
        action-angle values of the generated orbits [theta, J],
        shape = (num_orbits * n_steps, 2) if split_orbits=True,
        shape = (num_orbits, n_steps, 2) if split_orbits=False
    '''
    tl = torch.linspace(t_start, t_end, n_steps)
    if r_bounds is not None and J_bounds is None:
        aAH = actionAngleHarmonic(omega=omega)
        r = torch.linspace(r_bounds[0], r_bounds[-1], n_orbits)
        q = torch.stack([A * torch.cos(omega*tl) for A in r]).flatten()
        p = torch.stack([- A * omega * torch.sin(omega*tl) for A in r]).flatten()
        qp = torch.cat((q[:,None], p[:,None]), dim=-1)
        J, _, phi = aAH.actionsFreqsAngles(q, p)
        aa = torch.cat((phi[:, None], J[:, None]), dim=-1).to(torch.float64)
    elif J_bounds is not None and r_bounds is None:
        aAH_inv = actionAngleHarmonicInverse(omega=omega)
        J = torch.linspace(J_bounds[0], J_bounds[-1], n_orbits)
        theta = torch.linspace(-np.pi, np.pi, n_steps+2)[1:-1]
        J_grid, theta_grid = torch.meshgrid(J, theta, indexing='ij')
        q = torch.stack([aAH_inv(j, th)[0] for j, th in zip(J_grid.flatten(), theta_grid.flatten())])
        p = torch.stack([aAH_inv(j, th)[1] for j, th in zip(J_grid.flatten(), theta_grid.flatten())])
        qp = torch.cat((q[:,None], p[:,None]), dim=-1)
        aa = torch.cat((theta_grid.flatten()[:, None], J_grid.flatten()[:, None]), dim=-1).to(torch.float64)
    else:
        raise ValueError('Must provide either r_bounds or J_bounds, not both or neither.')
    
    if split_orbits == True:
        qp = qp.reshape(n_orbits, n_steps, 2)
        aa = aa.reshape(n_orbits, n_steps, 2)
    if saveData == True:
        model_dir = os.path.join(output_dir, filename)
        os.makedirs(model_dir, exist_ok=True)
        data_path = f'{model_dir}/{filename}.pt'
        torch.save({'phase space':qp, 'action angle':aa}, data_path)
        print(f'Training data saved to {data_path}')
        
        config = {
            "num_orbits": n_orbits,
            "potential": 'Simple Harmonic Oscillator',
            "potential_values": {'omega': omega},
            "integration_start_time": t_start,
            "integration_end_time": t_end,
            "num_steps": n_steps,
            "r_lower_bound": float(r_bounds[0]),
            "r_upper_bound": float(r_bounds[1]),
            "filename": filename
            }
        # Save the configuration as a JSON file
        config_path = f'{model_dir}/{filename}_config.json'
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)
        print(f"Configuration saved to {config_path}")
    return qp.reshape((n_orbits, n_steps, 2)), aa.reshape((n_orbits, n_steps, 2))

def guess_aa_pair(qp, transform_type='SHO', omega_guess=1):
    '''
    Parameters
    ----------
    qp : torch.tensor
        phase space values of the orbits. shape = (num_orbits x n_steps, 2)
    transform_type (str): type of guess action angle transformation
    '''
    if transform_type == 'SHO':
        aAH_guess = actionAngleHarmonic(omega=omega_guess)
        J_guess, _, theta_guess = aAH_guess.actionsFreqsAngles(qp[:,0], qp[:,1])
        aa_guess = torch.cat((theta_guess[:, None], J_guess[:, None]), dim=-1).to(torch.float32)
        for i, a in enumerate(aa_guess[:,0]):
            if a < 0:
                aa_guess[:,0][i] = a + 2*np.pi
    else:
        raise ValueError('Unknown transformation type')
    return aa_guess 

