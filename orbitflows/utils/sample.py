'''
Generate training data and orbit samples
'''

import torch
torch.set_default_dtype(torch.float64)
from .hamiltonians import H, H_sho

# galpy
from galpy.actionAngle import actionAngleVertical
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleHarmonic
from galpy.actionAngle import actionAngleHarmonicInverse

import numpy as np
import matplotlib.pyplot as plt
import os
import json

def generate_orbit_ps(num_orbits, potential, t_end, n_steps, t_start=0, r_bounds=torch.tensor([0.1, 0.75]), plot=False, color_by = 'H'):
    # Generate Phase-space values for num_orbits orbits
    tl = np.linspace(t_start, t_end, n_steps)
    r = torch.linspace(r_bounds[0], r_bounds[-1], num_orbits)
    ic = np.array([[i,0] for i in r])
    oi = Orbit(ic)
    oi.integrate(tl, potential)
    q = oi.x(tl).flatten()   # Example q values
    p = oi.vx(tl).flatten()   # Example p values
    z = torch.cat((torch.from_numpy(q)[:,None], torch.from_numpy(p)[:,None]), dim=-1)
    return z.reshape((num_orbits, n_steps, 2))

def generate_orbits_aa(n_actions, n_angles, action_min, action_max):
    '''
    Generate a several orbits of different
    actions in action-angle space. In practice,
    this is a grid of points in action-angle
    space.
    '''
    
    new_batch = torch.zeros(n_actions, n_angles, 2)
    angles_to_add = torch.linspace(0, 2*np.pi, n_angles+2)[1:-1] #- torch.pi
    actions_to_add = torch.linspace(action_min, action_max, n_actions)
    new_batch[:,:,0] = actions_to_add[:, None]
    new_batch[:,:,1] = angles_to_add
    return new_batch

def generate_orbits(num_orbits, potential, t_end, n_steps, t_start=0, pot_type='1D', r_bounds=torch.tensor([0.1, 0.75]), saveData=False, output_dir = None, filename='test_data.pt'):
    # Simple synthetic transformation (this is where you'd put your known transformation)
    #r_bounds = torch.sqrt(2 * E_bounds / omega**2)
    tl = np.linspace(t_start, t_end, n_steps)
    if pot_type == '1D':
        aAV = actionAngleVertical(pot=potential)
        r = torch.linspace(r_bounds[0], r_bounds[-1], num_orbits)
        ic = np.array([[i,0] for i in r])
        oi = Orbit(ic)
        oi.integrate(tl, potential)
        
        q = oi.x(tl).flatten()   # Example q values
        p = oi.vx(tl).flatten()   # Example p values
        z = torch.cat((torch.from_numpy(q)[:,None], torch.from_numpy(p)[:,None]), dim=-1)
        J, _, phi = aAV.actionsFreqsAngles(q, p)
        aa = torch.cat((torch.from_numpy(phi)[:,None], torch.from_numpy(J)[:,None]), dim=-1).to(torch.float32)
            
    if saveData == True:
    
        model_dir = os.path.join(output_dir, filename)
        os.makedirs(model_dir, exist_ok=True)
        data_path = f'{model_dir}/{filename}.pt'
        torch.save({'phase space':z, 'action angle':aa}, data_path)
        print(f'Training data saved to {data_path}')

        # Create a configuration dictionary
        if pot_type == '1D':
            potential_str = str(potential)[17:]
            cut_index = potential_str.find('.')
            potential_name = potential_str[:cut_index]
            config = {
                "num_orbits": num_orbits,
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
    return z.reshape((num_orbits, n_steps, 2)), aa.reshape((num_orbits, n_steps, 2))

def generate_sho_orbits(
        num_orbits, 
        omega, 
        t_end, 
        n_steps, 
        t_start=0, 
        r_bounds=torch.tensor([0.1, 0.75]),
        split_orbits=False,
        output_dir=None,
        saveData=False, 
        filename='sho_test_data.pt'):
    '''
    generates set of orbits for a simple harmonic oscillator in phase space and their action-angle space representations

    Parameters:
    ----------
    num_orbits : int
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
    split_orbits : bool
        whether to split the output into individual orbits
    saveData : bool
        whether to save the generated data
    filename : str
        name of the file to save the data to
    
    Returns:
    -------
    z : torch.tensor
        phase space values of the generated orbits [q, p], 
        shape = (num_orbits * n_steps, 2) if split_orbits=True,
        shape = (num_orbits, n_steps, 2) if split_orbits=False
    aa : torch.tensor
        action-angle values of the generated orbits [theta, J],
        shape = (num_orbits * n_steps, 2) if split_orbits=True,
        shape = (num_orbits, n_steps, 2) if split_orbits=False
    '''
    aAH = actionAngleHarmonic(omega=omega)
    tl = torch.linspace(t_start, t_end, n_steps)
    r = torch.linspace(r_bounds[0], r_bounds[-1], num_orbits)
    q = torch.stack([A * torch.cos(omega*tl) for A in r]).flatten()
    p = torch.stack([- A * omega * torch.sin(omega*tl) for A in r]).flatten()
    z = torch.cat((q[:,None], p[:,None]), dim=-1)
    
    J, _, phi = aAH.actionsFreqsAngles(q, p)
    #phi = phi + torch.pi
    aa = torch.cat((phi[:, None], J[:, None]), dim=-1).to(torch.float64)
    for i, a in enumerate(aa[:,0]):
            if a < 0:
                aa[:,0][i] = a + 2*np.pi
    if split_orbits == True:
        z = z.reshape(num_orbits, n_steps, 2)
        aa = aa.reshape(num_orbits, n_steps, 2)
    if saveData == True:
        model_dir = os.path.join(output_dir, filename)
        os.makedirs(model_dir, exist_ok=True)
        data_path = f'{model_dir}/{filename}.pt'
        torch.save({'phase space':z, 'action angle':aa}, data_path)
        print(f'Training data saved to {data_path}')

        # Create a configuration dictionary
        
        config = {
            "num_orbits": num_orbits,
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
    return z.reshape((num_orbits, n_steps, 2)), aa.reshape((num_orbits, n_steps, 2))

def guess_aa_pair(z, transform_type='SHO', omega_guess=1):
    # I should add some sort of kwargs thing here instead of omega, so that they can identify a custom guess potential/action angle transformation
    '''
    params:
    ------
    z : torch.tensor
        phase space values of the orbits. shape = (num_orbits x n_steps, 2)
    transform_type (str): type of guess action angle transformation
    '''
    if transform_type == 'SHO':
        aAH_guess = actionAngleHarmonic(omega=omega_guess)
        J_guess, _, theta_guess = aAH_guess.actionsFreqsAngles(z[:,0], z[:,1])
        aa_guess = torch.cat((theta_guess[:, None], J_guess[:, None]), dim=-1).to(torch.float32)
        for i, a in enumerate(aa_guess[:,0]):
            if a < 0:
                aa_guess[:,0][i] = a + 2*np.pi
    else:
        raise ValueError('Unknown transformation type')
    
    return aa_guess 

