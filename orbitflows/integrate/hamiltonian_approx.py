'''Approximations of the hamiltonian as a function of the action-angle variables.'''
import torch


def hamiltonian_fixed_angle(model, aa, theta_set=torch.tensor(0.0)):
    '''
    Compute the Hamiltonian at an action, averaged over angles.
        
    Parameters
    ----------
    aa : torch.tensor
        action-angle variables

    theta_set : torch.tensor
        angle to be set in the Hamiltonian.
        
    Returns
    -------
    torch.tensor
        Hamiltonian averaged over the angle
    '''
    _aa = aa.clone()
    _aa[..., 0] = theta_set.clone()
    return model.hamiltonian(_aa)
