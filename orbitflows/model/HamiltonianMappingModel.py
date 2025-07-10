'''Model with an intermediate toy hamiltonian, mapped in phase-space.'''

from .MappingModel import MappingModel
from ..flow import GradientBasedConditioner
from ..flow import SymplecticCouplingLayer


class HamiltonianMappingModel(MappingModel):
    def __init__(self, targetPotential : callable, input_dim : int, num_layers : int, omega=1.0, layer_class : callable = SymplecticCouplingLayer, conditioner : callable = GradientBasedConditioner, conditioner_args : dict = {}):
        '''
        Initialize the normalizing flow model with a toy hamiltonian.

        The harmonic oscillator is used as the toy hamiltonian for
        1D systems and the isochrone potential is used for 2D systems.

        Parameters
        ----------
        targetPotential : callable
            The potential that represents the physical system of interest.
            Should be a callable that takes
            phase-space coordinates as input and returns the potential.

        input_dim : int
            The dimensions of the input data. Should be the same dimension of the phase space
            you wish to transform.

        hidden_dim : int
            The dimensions of the hidden layers.

        num_layers : int
            The number of layers in the normalizing flow.

        omega : float, optional
            The frequency of the harmonic oscillator for the toy hamiltonian.
            If None, defaults to 1.0.

        Notes
        -----
        - TO ADD: only use omega for systems with one dimension and 
        isochroneParams for systems with more than one dimension.
        '''

        MappingModel.__init__(self, targetPotential, input_dim, num_layers, omega, layer_class, conditioner, conditioner_args)  

    def aa_to_ps(self, aa):
        '''
        Transform aciton angle to phase-space coordinates using the 
        normalizing flow and the toy potential.

        Parameters
        ----------
        aa : torch.Tensor
            The action angle coordinates to transform.

        Returns
        -------
        torch.Tensor
            The approximate phase-space coordinates cooresponding to the input.
        '''
        ps_int = self.aa_to_toy_ps(aa) # intermediate solution
        return self.flow(ps_int)
    
    def ps_to_aa(self, ps):
        '''
        Transform phase-space to action angle coordinates using the 
        normalizing flow and the toy potential.

        Parameters
        ----------
        ps : torch.Tensor
            The phase-space coordinates to transform.

        Returns
        -------
        torch.Tensor
            The approximate action angle coordinates cooresponding to the input.
        '''
        ps_sho = self.flow.inverse(ps)
        return self.toy_ps_to_aa(ps_sho)
