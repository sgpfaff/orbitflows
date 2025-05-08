'''Correction functions for integration.'''

import torch

def dH_dx(ps, wrt, _hamiltonian):
    '''
    Compute the derivative of the Hamiltonian with respect to q.
    
    Parameters
    ----------
    ps : torch.tensor
        phase-space point

    wrt : str
        variable to compute the derivative with respect to.
        Should be either 'q' or 'p'.
    
    _hamiltonian : callable
        function to compute the Hamiltonian of the model prediction.
        Should take a phase-space point as input and return a scalar.
        Called internally in Model integration method.
    
    Returns
    -------
    torch.tensor
        derivative of the Hamiltonian with respect to q
    '''
    q0 = ps[..., 0]
    p0 = ps[..., 1]
    if wrt == 'q':
        return - torch.autograd.grad(_hamiltonian(torch.stack([q0, p0], dim=-1)), q0)[0]
    elif wrt == 'p':
        return torch.autograd.grad(_hamiltonian(torch.stack([q0, p0], dim=-1)), p0)[0]

def eulerstep(ps, delta_t, _hamiltonian_err):
    '''
    Euler step

    Parameters
    ----------
    ps : torch.tensor
        phase-space point
    
    delta_t : float
        time step
    
    _hamiltonian_err : callable
        function to compute the Hamiltonian error of the model prediction.
        Should take a phase-space point as input and return a scalar.
        Called internally in Model integration method.
    ''' 
    q0 = ps[..., 0]
    p0 = ps[..., 1]
    p = p0 + delta_t * dH_dx(ps, 'q', _hamiltonian_err)
    q = q0 + delta_t * dH_dx(ps, 'p', _hamiltonian_err)
    return torch.stack([q, p], dim=-1)

def rungekutta4(ps, delta_t, _hamiltonian_err):
    '''4th order runge-kutta step
    
    Parameters
    ----------
    ps : torch.tensor
        phase-space point
    
    delta_t : float
        time step
    
    _hamiltonian_err : callable
        function to compute the Hamiltonian error of the model prediction.
        Should take a phase-space point as input and return a scalar.
        Called internally in Model integration method.
    '''
    def _kfull_x(ps, wrt, kn_q, kn_p):
        '''
        Compute runge-kutta full step k for a given variable. 
        Used for k1 and k4.
        '''
        q0 = ps[..., 0]
        p0 = ps[..., 1]
        return delta_t * dH_dx(torch.stack([q0 + kn_q, p0 + kn_p], dim=-1), wrt, _hamiltonian_err)
    
    def _khalf_x(ps, wrt, kn_q, kn_p):
        '''
        Compute runge-kutta half-step for a given variable. 
        Used for k2 and k3.
        '''
        q0 = ps[..., 0]
        p0 = ps[..., 1]
        return delta_t * dH_dx(torch.stack([q0 + kn_q/2, p0 + kn_p/2], dim=-1), wrt, _hamiltonian_err)
    
   
    
    k1_q, k1_p = _kfull_x(ps, 'p', 0, 0), _kfull_x(ps, 'q', 0, 0)

    k2_q, k2_p = _khalf_x(ps, 'p', k1_q, k1_p), _khalf_x(ps, 'q', k1_q, k1_p)

    k3_q, k3_p = _khalf_x(ps, 'p', k2_q, k2_p), _khalf_x(ps, 'q', k2_q, k2_p)

    k4_q, k4_p = _kfull_x(ps, 'p', k3_q, k3_p), _kfull_x(ps, 'q', k3_q, k3_p)

    q0 = ps[..., 0]
    p0 = ps[..., 1]

    q = q0 + (1/6)*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    p = p0 + (1/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)

    return torch.stack([q, p], dim=-1)
