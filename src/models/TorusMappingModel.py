'''Model with an intermediate toy hamiltonian, mapped in action angle space.'''

from .MappingModel import MappingModel
from .flow import GradientBasedConditioner
from .flow import TorusSymplecticCouplingLayer
from tqdm import tqdm
import json
from functools import partial
import torch
import inspect
import matplotlib.pyplot as plt
from ..util import scaled_H_std
from ..util import potential_key_mappings as pm
from ..util import potential_function_mappings as pfm
from ..util import optimizer_key_mappings as okm
from ..util import scheduler_key_mappings as skm
from ..util import layer_key_mappings
from ..util import conditioner_key_mappings


class TorusMappingModel(MappingModel):
    def __init__(self, targetPotential : callable, input_dim : int, n_layers : int, omega=1.0, layer_class : callable = TorusSymplecticCouplingLayer, conditioner : callable = GradientBasedConditioner, conditioner_args : dict = {}, optimizer=None, scheduler=None):
        '''
        Initialize the normalizing flow model with a toy hamiltonian. TorusMappingModel deforms
        toy tori in action-angle space, trained by the outputs conservation of energy after
        transforming to phase space coordinates using the toy transformation.

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
        - Model is designed to go from the toy to the target system in action-angle space.
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
            if self.scheduler is not None:
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
    def _transform_to_periodic(self, aa):
        '''
        Transform action-angle coordinates to periodic coordinates for the flow.

        Parameters
        ----------
        aa : torch.Tensor
            The action-angle coordinates to transform.

        Returns
        -------
        torch.Tensor
            The transformed periodic coordinates.
        '''
        theta, j = aa[...,0], aa[...,1]
        input_data = torch.stack((torch.sin(theta), j), dim=-1)
        return input_data
    
    def _transform_from_periodic(self, periodic_aa):
        '''
        Transform periodic coordinates back to action-angle coordinates.

        Parameters
        ----------
        periodic_aa : torch.Tensor
            The periodic coordinates to transform.

        Returns
        -------
        torch.Tensor
            The transformed action-angle coordinates.
        '''
        sin_theta, j = periodic_aa[...,0], periodic_aa[...,1]
        theta = torch.asin(sin_theta)
        return torch.stack((theta, j), dim=-1)

    def aa_to_ps(self, aa):
        '''
        Transform action angle to phase-space coordinates using the 
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
        aa_toy = self.flow(aa)
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
        return self.flow.inverse(aa_toy)

    def train(self, 
              training_data, 
              steps, 
              lr=1e-4, 
              loss_function=None, 
              lf_args=None, 
              orbit_batching=False, 
              batching_along_orbits=False, 
              batch_size=None, 
              update_frequency=100,
              update_plots=False
              ):
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
        for epoch in tqdm(range(steps)):
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
            
            nf_output = self.aa_to_ps(training_data_sample)
            
            loss = loss_function(nf_output, **lf_args)
            if update_frequency is not None:
                if epoch % update_frequency == 0:
                    tqdm.write(f"\nEpoch {epoch}: {loss.item()}")
                    if self.scheduler is not None:
                        tqdm.write(f"LR : {scheduler.optimizer.param_groups[0]['lr']}")
                    if update_plots:
                        plt.figure(figsize=(6,6))
                        plt.scatter(*nf_output.detach().numpy().T, s=10)
                        plt.show()
                        plt.figure(figsize=(6,6))
                        plt.scatter(*training_data_sample.detach().numpy().T, s=10, alpha=0.1)
                        plt.scatter(*self.flow(training_data_sample).detach().numpy().T, s=10, alpha=0.1)
                        plt.show()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.scheduler is not None:
                scheduler.step(loss)
                self.lr_list.append(scheduler.optimizer.param_groups[0]['lr'])

            self.loss_list.append(loss.item())
    
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
            num_layers = data['num_layers'],
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
            "num_layers" : self.num_layers,
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
