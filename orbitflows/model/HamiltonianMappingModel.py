'''Model with an intermediate toy hamiltonian, mapped in phase-space.'''

from .MappingModel import MappingModel
from ..flow import GradientBasedConditioner
from ..flow import SymplecticCouplingLayer
from ..utils import potential_key_mappings as pm
from ..utils import potential_function_mappings as pfm
from ..utils import layer_key_mappings
from ..utils import conditioner_key_mappings

import json
from functools import partial
import torch
import inspect


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
        
        if isinstance(self.targetPotential, partial):
            self.targetPotentialKey = self.targetPotential.func.__name__
        try:
            self.targetPotentialKey =  self.targetPotential.__name__
        except AttributeError:
            self.targetPotentialKey =  str(self.targetPotential)

        #self.targetPotentialKey = self.targetPotential.__name__#pfm[self.targetPotential]
        self.potential_kwargs = {}
        params = inspect.signature(self.targetPotential).parameters
        for i, param in enumerate(params):
            if i != 0:
                self.potential_kwargs[param] = float(params[param].default)

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

    def to_dict(self):
        return {
            "input_dim" : self.input_dim,
            "num_layers" : self.num_layers,
            "omega" : self.omega,
            "layer_class_key" : self.layer_class_key,
            "conditioner_key" : self.conditioner_key,
            "conditioner_args" : self.conditioner_args,
            "targetPotentialKey" : self.targetPotentialKey,
            "potential_kwargs" : self.potential_kwargs,
            "loss_list" : self.loss_list
        }
    
    @classmethod
    def load(cls, filename):
        '''
        Load model from file.
        '''
        with open(filename+'.json', "r") as f:
            data = json.load(f)

        for key, value in data['potential_kwargs'].items():
            data['potential_kwargs'][key] = torch.tensor(value)
        instance = cls(
            targetPotential = partial(pm[data['targetPotentialKey']], **data['potential_kwargs']), 
            input_dim = data['input_dim'],
            num_layers = data['num_layers'],
            omega = data['omega'],
            layer_class = layer_key_mappings[data['layer_class_key']],
            conditioner = conditioner_key_mappings[data['conditioner_key']],
            conditioner_args = data['conditioner_args']
            )
        instance.flow.load_state_dict(torch.load(filename+'.pt'))
        instance.loss_list = data['loss_list']
        return instance
