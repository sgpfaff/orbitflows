from .base import Model
from abc import abstractmethod
from tqdm import tqdm
from ..train import scaled_H_std
from ..utils import actionAngleHarmonic, actionAngleHarmonicInverse
import torch

'''Base class for mapping models.'''

class MappingModel(Model):
    '''
    Base class for mapping models, which involve toy systems with
    known analytical transformations.
    '''
    def __init__(self, targetPotential : callable, input_dim, hidden_dim, num_layers, omega):
        '''
        Initialize the mapping model.

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
        '''
        Model.__init__(self, input_dim, hidden_dim, num_layers)
        
        if self.input_dim == 2:
            self.omega = omega
            def _toy_ps_to_aa(ps):
                '''
                Transform phase-space to action-angle coordinates under the toy potential.
                '''
                q, p = ps[..., 0], ps[..., 1]
                j, _, theta = actionAngleHarmonic(omega=self.omega).actionsFreqsAngles(q, p)
                return torch.stack((theta, j), dim=-1)
            def _aa_to_toy_ps(aa):
                '''
                Transform action-angle to phase-space coordinates under the toy potential.
                '''
                theta, j = aa[..., 0], aa[..., 1]
                q, p = actionAngleHarmonicInverse(omega=self.omega)(j, theta)
                return torch.stack((q, p), dim=-1)
        else: 
            # in the future, _aa_to_toy_ps and _toy_ps_to_aa will be defined here 
            # for higher dimensional systems, using the isochrone potential as the toy potential.
            raise ValueError("Only 1D systems are currently supported. 2D and 3D will be added in the future.")
            
        self.toy_ps_to_aa = _toy_ps_to_aa 
        self.aa_to_toy_ps = _aa_to_toy_ps
        self.targetPotential = targetPotential
    
    @abstractmethod
    def ps_to_aa(self, ps):
        pass
    
    @abstractmethod
    def aa_to_ps(self, aa):
        pass
        
    def train(self, training_data, steps, lr=1e-4, loss_function=scaled_H_std, lf_args=None):
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
        '''
        if loss_function == scaled_H_std and lf_args is None:
            lf_args={'targetPotential' : self.targetPotential}
        
        
        optimizer = self.optimizer(self.flow.parameters(), lr=lr)
        for epoch in tqdm(range(steps)):
            nf_output = self.flow(training_data)
            loss = loss_function(nf_output, **lf_args)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            self.loss_list.append(loss.item())

    def save(self, filename, relative_path='../../trained_models/'):
        '''
        Save the model to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the model to.
        
        relative_path : str
            The relative path to the directory to save the model in.
        '''
        pass