'''Model with an intermediate toy hamiltonian.'''

import torch
from tqdm import tqdm
from .base import Model
from ..util import actionAngleHarmonic, actionAngleHarmonicInverse, H
from ..train import scaled_H_std

class HamiltonianMappingModel(Model):
    def __init__(self, input_dim : int, hidden_dim : int, num_layers : int, omega=None, isochroneParams=None):
        '''
        Initialize the normalizing flow model with a toy hamiltonian.

        The harmonic oscillator is used as the toy hamiltonian for
        1D systems and the isochrone potential is used for 2D systems.

        Parameters
        ----------

        input_dim : int
            The dimensions of the input data.

        hidden_dim : int
            The dimensions of the hidden layers.

        num_layers : int
            The number of layers in the model.

        omega : float, optional
            The frequency of the harmonic oscillator for the toy hamiltonian.
            If None, defaults to 1.0.

        isochroneParams : list, optional
            The parameters for the isochrone potential.

        Notes
        -----
        - only use omega for systems with one dimension and
        isochroneParams for systems with more than one dimension.
        '''

        Model.__init__(self, input_dim, hidden_dim, num_layers)

        if self.input_dim == 2:
            if isochroneParams is not None:
                raise ValueError("Isochrone parameters are not supported for 1D systems. Please define omega instead.")
            elif omega is not None:
                self.omega = omega
            else:
                self.omega = 1.0
            def _toy_ps_to_aa(ps):
                q, p = ps[..., 0], ps[..., 1]
                j, _, theta = actionAngleHarmonic(omega=self.omega).actionsFreqsAngles(q, p)
                return torch.stack((theta, j), dim=-1)
            def _aa_to_toy_ps(aa):
                theta, j = aa[..., 0], aa[..., 1]
                q, p = actionAngleHarmonicInverse(omega=self.omega)(j, theta)
                return torch.stack((q, p), dim=-1)
            
        
        self.toy_ps_to_aa = _toy_ps_to_aa
        self.aa_to_toy_ps = _aa_to_toy_ps
        self.targetPotential = None
        self.loss_list = []
            
    def train(self, targetPotential, training_data, steps, loss_function=scaled_H_std, learning_rate=1e-4):
        '''
        Train the model.
        
        Parameters
        ----------

        training_data : torch.Tensor
            The training data to use for the model. Should be of shape (n_orbits, n_steps, 2).

        steps : int
            The number of steps to train the model.

        stepsize : float
            The step size for the training.

        loss_function : callable
            The loss function to use for the training.
        '''
        
        self.targetPotential = targetPotential
        
        optimizer = self.optimizer(self.flow.parameters(), lr=learning_rate)
        for epoch in tqdm(range(steps)):
            ps_NF = self.flow(training_data)
            loss = loss_function(ps_NF, targetPotential)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            self.loss_list.append(loss.item())

    def aa_to_ps(self, aa):
        '''
        Compute the phase-space coordinates from the action-angle variables,
        using the normalizing flow approximation.

        Parameters
        ----------
        aa : torch.Tensor
            The action-angle variables.

        Returns
        -------
        torch.Tensor
            The phase-space coordinates.
        '''
        ps_int = self.aa_to_toy_ps(aa) # intermediate solution
        return self.flow(ps_int)
    
    def ps_to_aa(self, ps):
        ps_sho = self.flow.inverse(ps)
        return self.toy_ps_to_aa(ps_sho)
    
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


    def save(self, filename, output_dir=''):
        pass
