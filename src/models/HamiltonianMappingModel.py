'''Model with an intermediate toy hamiltonian, mapped in phase-space.'''

import matplotlib.pyplot as plt
from .MappingModel import MappingModel
from .flow import GradientBasedConditioner
from .flow import SymplecticCouplingLayer
from tqdm import tqdm
from ..dynamics import H
from ..util import scaled_H_std, max_error_along_orbs, mean_error_along_orbs
from ..util import potential_key_mappings as pm
from ..util import potential_function_mappings as pfm
from ..util import optimizer_key_mappings as okm
from ..util import scheduler_key_mappings as skm
from ..util import layer_key_mappings
from ..util import conditioner_key_mappings

import json
from functools import partial
import torch
import inspect


class HamiltonianMappingModel(MappingModel):
    def __init__(self, targetPotential : callable, input_dim : int, n_layers : int, 
                 omega=1.0, layer_class : callable = SymplecticCouplingLayer, 
                 conditioner : callable = GradientBasedConditioner, 
                 conditioner_args : dict = {}, optimizer=None, 
                 scheduler=None):
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

        n_layers : int
            The number of layers in the normalizing flow.

        omega : float, optional
            The frequency of the harmonic oscillator for the toy hamiltonian.
            If None, defaults to 1.0.

        Notes
        -----
        - TO ADD: only use omega for systems with one dimension and 
        isochroneParams for systems with more than one dimension.
        '''

        MappingModel.__init__(self, targetPotential, input_dim, n_layers, omega, layer_class, conditioner, conditioner_args, optimizer, scheduler)
        
        if isinstance(self.targetPotential, partial):
            self.targetPotentialKey = self.targetPotential.func.__name__
        else:
            try:
                self.targetPotentialKey =  self.targetPotential.__name__
            except AttributeError:
                self.targetPotentialKey =  str(self.targetPotential)

        if isinstance(self.scheduler, partial):
            self.schedulerKey = self.scheduler.func.__name__
        else:
            try:
                self.schedulerKey =  self.scheduler.__name__
            except AttributeError:
                self.schedulerKey =  str(self.scheduler)

        if isinstance(self.optimizer, partial):
            self.optimizerKey = self.optimizer.func.__name__
        else:
            try:
                self.optimizerKey =  self.optimizer.__name__
            except AttributeError:
                self.optimizerKey =  str(self.optimizer)

        #self.targetPotentialKey = self.targetPotential.__name__#pfm[self.targetPotential]
        self.potential_kwargs = {}
        params = inspect.signature(self.targetPotential).parameters
        for i, param in enumerate(params):
            if i != 0:
                self.potential_kwargs[param] = float(params[param].default)

        self.scheduler_kwargs = {}
        params = inspect.signature(self.scheduler).parameters
        for i, param in enumerate(params):
            if i != 0:
                if type(params[param].default) is not str:
                    self.scheduler_kwargs[param] = float(params[param].default)
                else:
                    self.scheduler_kwargs[param] = params[param].default

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

    def train(self, 
              training_data, 
              steps, 
              lr=1e-4, 
              loss_function=scaled_H_std, 
              lf_args=None, 
              orbit_batching=False, 
              batching_along_orbits=False, 
              batch_size=None, 
              update_frequency=25,
              update_plots=False,
              nested_progress=False):
        '''
        Train the model.
        
        Parameters
        ----------

        training_data : torch.Tensor
            Training data to use for the model. Should be of shape (n_orbits, n_steps, 2).

        steps : int
            Number of steps to train the model.

        lr : float
            Learning rate for the training.

        loss_function : callable
            Loss function to use for the training.
        
        lf_args : dict
            Arguments to pass to the loss function.
            If the loss function is scaled_H_std, then lf_args should contain
            the target potential to use for the training.
        update_frequency : str
            Frequency of updates. No updates if None.
        update_plots : bool
            Whether to update plots during training.
        '''

        if loss_function == scaled_H_std and lf_args is None:
            lf_args={'targetPotential' : self.targetPotential}

        if batching_along_orbits or orbit_batching:
            if batch_size is None:
                batch_size = training_data.shape[0] // 10
        else:
            training_data_sample = training_data.clone()
        
        optimizer = self.optimizer(self.flow.parameters(), lr=lr)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
        
        if nested_progress:
            pbar = tqdm(range(steps), desc="Training", position=1, leave=False)
        else:
            pbar = tqdm(range(steps), desc="Training", disable=update_frequency is None)
        for epoch in pbar:
            if batching_along_orbits and orbit_batching:
                indices_for_orbits = torch.randperm(training_data.shape[1])[:batch_size]
                indices_along_orbit = torch.randperm(training_data.shape[1])[:batch_size]
                training_data_sample = training_data[indices_for_orbits, indices_along_orbit]
            elif orbit_batching:
                indices = torch.randperm(training_data.shape[0])[:batch_size]
                training_data_sample = training_data[indices]
            elif batching_along_orbits:
                indices = torch.randperm(training_data.shape[1])[:batch_size]
                training_data_sample = training_data[:, indices].requires_grad_(True)
            
            nf_output = self.flow(training_data_sample)
            loss = loss_function(nf_output, **lf_args)
            if update_frequency is not None:
                if epoch % update_frequency == 0:
                    mean_mean_error = mean_error_along_orbs(H(nf_output, self.targetPotential)).mean().item()
                    mean_max_error = max_error_along_orbs(H(nf_output, self.targetPotential)).mean().item()
                    if update_plots:
                        plt.scatter(*training_data_sample.detach().cpu().numpy().T)
                        plt.show()
                    if self.scheduler is not None:
                        pbar.set_postfix(loss=loss.item(), lr = scheduler.optimizer.param_groups[0]['lr'],
                                         mean_max_error=mean_max_error,
                                         mean_mean_error=mean_mean_error)
                    else:
                        pbar.set_postfix(loss=loss.item(), mean_max_error = mean_max_error,
                                         mean_mean_error=mean_mean_error)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.scheduler is not None:
                scheduler.step(loss.detach())
                self.lr_list.append(scheduler.optimizer.param_groups[0]['lr'])
            self.loss_list.append(loss.item())
        
    def _to_dict(self):
        # Handle conditioner_args with activation functions
        serializable_conditioner_args = self.conditioner_args.copy()
        
        # Convert activation function to string if present
        if 'activation' in serializable_conditioner_args:
            activation_func = serializable_conditioner_args['activation']
            if hasattr(activation_func, '__name__'):
                serializable_conditioner_args['activation'] = activation_func.__name__
            elif hasattr(activation_func, '__class__'):
                serializable_conditioner_args['activation'] = activation_func.__class__.__name__
            else:
                serializable_conditioner_args['activation'] = str(activation_func)
        
        return {
            "input_dim" : self.input_dim,
            "n_layers" : self.n_layers,
            "omega" : self.omega,
            "layer_class_key" : self.layer_class_key,
            "conditioner_key" : self.conditioner_key,
            "conditioner_args" : serializable_conditioner_args,
            "targetPotentialKey" : self.targetPotentialKey,
            "potential_kwargs" : self.potential_kwargs,
            "loss_list" : self.loss_list,
            "lr_list" : self.lr_list,
            "optimizerKey" : self.optimizerKey,
            "schedulerKey" : self.schedulerKey,
            "scheduler_kwargs" : self.scheduler_kwargs
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
        
        # Handle activation function reconstruction
        conditioner_args = data['conditioner_args'].copy()
        if 'activation' in conditioner_args and isinstance(conditioner_args['activation'], str):
            activation_name = conditioner_args['activation']
            # Map common activation function names back to PyTorch functions
            activation_mapping = {
                'ReLU': torch.nn.ReLU,
                'Tanh': torch.nn.Tanh,
                'Sigmoid': torch.nn.Sigmoid,
                'LeakyReLU': torch.nn.LeakyReLU,
                'ELU': torch.nn.ELU,
                'GELU': torch.nn.GELU,
                'SiLU': torch.nn.SiLU,
                'Swish': torch.nn.SiLU,  # Swish is the same as SiLU
            }
            
            if activation_name in activation_mapping:
                conditioner_args['activation'] = activation_mapping[activation_name]
            else:
                # Fallback to ReLU if we don't recognize the activation
                print(f"Warning: Unknown activation '{activation_name}', defaulting to ReLU")
                conditioner_args['activation'] = torch.nn.ReLU
        
        instance = cls(
            targetPotential = partial(pm[data['targetPotentialKey']], **data['potential_kwargs']), 
            input_dim = data['input_dim'],
            n_layers = data['n_layers'],
            omega = data['omega'],
            layer_class = layer_key_mappings[data['layer_class_key']],
            conditioner = conditioner_key_mappings[data['conditioner_key']],
            conditioner_args = conditioner_args,
            optimizer = okm[data['optimizerKey']],
            scheduler = partial(skm[data['schedulerKey']], **data['scheduler_kwargs'])
            )
        instance.flow.load_state_dict(torch.load(filename+'.pt'))
        instance.loss_list = data['loss_list']
        instance.lr_list = data['lr_list']
        return instance
