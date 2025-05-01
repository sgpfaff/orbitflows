'''Model with an intermediate toy hamiltonian, mapped in action angle space.'''

from .MappingModel import MappingModel


class TorusMappingModel(MappingModel):
    def __init__(self, targetPotential : callable, input_dim : int, hidden_dim : int, num_layers : int, omega=1.0):
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
        - Model is designed to go from the toy to the target system in action-angle space.
        - TO ADD: only use omega for systems with one dimension and 
        isochroneParams for systems with more than one dimension.
        '''

        MappingModel.__init__(self, targetPotential, input_dim, hidden_dim, num_layers, omega)  

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
        aa_toy = self.flow.inverse(aa) # flow goes from toy to target
        return self.aa_to_toy_ps(aa_toy)
    
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
        aa_toy = self.toy_ps_to_aa(ps)
        return self.flow(aa_toy)
