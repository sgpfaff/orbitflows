'''Top level class for all models.'''

from abc import ABC, abstractmethod
from ..flow import GsympNetFlow
from ..utils import H
import torch
from tqdm import tqdm
from ..integrate import eulerstep, hamiltonian_fixed_angle


class Model(ABC):
    '''Base class for all models.'''
    
    def __init__(self, input_dim : int, hidden_dim : int, num_layers : int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.flow = GsympNetFlow(input_dim, hidden_dim, num_layers)
        self.optimizer = torch.optim.Adam
        self.loss_list = []

        self.training_config = {
            'steps' : None,
            'stepsize' : None,
            'loss_function' : None
            }

    @abstractmethod
    def aa_to_ps(self, aa):
        pass

    @abstractmethod
    def ps_to_aa(self, ps):
        pass

    @abstractmethod
    def train(self, training_data, steps, loss_function):
        '''Train the model.'''
        pass


    def hamiltonian(self, aa):
        '''
        Compute the Hamiltonian for the given action-angle variables, using the
        normalizing flow approximation.

        Parameters
        ----------
        aa : torch.Tensor
            The action-angle variables.
            Should be of shape (n_orbits, n_steps, 2).
            The first dimension is the angle and the second dimension is the action.

        Returns
        -------
        torch.Tensor
            The Hamiltonian value along each orbit.
        '''
        return H(self.aa_to_ps(aa), self.targetPotential)
        

    def frequency(self, aa):
        '''Compute the frequency of the system.'''
        theta, j = aa[..., 0], aa[..., 1]
        #theta.requires_grad = True
        aa = torch.stack((theta, j), dim=-1)
        return torch.autograd.grad(self.hamiltonian(aa), j, allow_unused=True)[0]

 


    def integrate(self, aa, steps, t_end, correction=eulerstep, hamiltonian_tilde=hamiltonian_fixed_angle):
        '''Routine to integrate orbits in AA space and update frequencies regularly, with euler step correction
    
        Parameters
        ----------
        aa : torch.tensor
            initial action-angle variables

        steps : int
            number of steps in action-angle space between frequency updates

        t_end : float
            end time of the integration
        
        correction : callable
            correction function to be used in the integration process.
            Should take the phase-space coordinates, the time step, and the hamiltonian_tilde
            function as arguments.
            The function should return new phase-space coordinates of the same shape as the input.

        hamiltonian_tilde : callable
            Assumption for the correct Hamiltonian as a function of action-angle variables.
            Only arguments should be model and aa, so it's recommended to use a partial function or
            redefine the function such that this is the case.
        
        Returns
        -------
        torch.tensor
            action-angle variables as a function of time
        
        '''

        def _hamiltonian_error(ps):
            '''
            Compute the error of the model prediction in the Hamiltonian.
            
            Parameters
            ----------
            ps : torch.tensor
                phase-space point
            
            Returns
            -------
            torch.tensor
                Hamiltonian error of the model prediction
            '''
            _aa = self.ps_to_aa(ps)
            return H(ps, self.targetPotential) - hamiltonian_tilde(self, _aa)

        def _freq_tilde(aa):
            '''
            Compute the frequency of the system using the hamiltonian_tilde function.
            
            Parameters
            ----------
            aa : torch.tensor
                action-angle variables

            Returns
            -------
            torch.tensor
                frequency of the system assuming the hamiltonian_tilde function
            '''
            theta, j = aa[..., 0], aa[..., 1]
            _aa = torch.stack((theta, j), dim=-1)
            return torch.autograd.grad(hamiltonian_tilde(self, _aa), j, allow_unused=True)[0]
        
        delta_t = torch.tensor(t_end/steps)
        theta_list = torch.zeros(steps)
        j_list = torch.zeros(steps)
        
        theta0 = aa[...,0]
        j0 = aa[...,1]
        freq = _freq_tilde(aa)
        

        for i, t in enumerate(tqdm(torch.linspace(0, t_end, steps))):
            # evolve in action-angle space
            theta_half = theta0 + freq*delta_t/2 # drift 

            # transform to phase space
            aa_half = torch.stack((theta_half, j0))
            ps_half = self.aa_to_ps(aa_half)

            # compute Hamiltonian error

            ps_half_corrected = correction(ps_half, delta_t, _hamiltonian_error)

            # convert back to action-angle space
            aa_half_corrected = self.ps_to_aa(ps_half_corrected)

            # update frequency
            theta_half_corrected = aa_half_corrected[0]
            j = aa_half_corrected[1]
            freq = _freq_tilde(aa_half_corrected)

            # drift
            theta = theta_half_corrected + freq*delta_t/2

            theta_list[i] = theta
            j_list[i] = j
            theta0 = theta
            j0 = j
            #print(theta0, J0)
        return torch.stack([theta_list, j_list], dim=-1)

    @abstractmethod
    def save(self):
        pass
