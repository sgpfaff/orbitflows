from .BaseModel import BaseModel
from abc import abstractmethod
from ..dynamics import actionAngleHarmonic, actionAngleHarmonicInverse
import torch

'''Base class for mapping models.'''

class MappingModel(BaseModel):
    '''
    Base class for mapping models, which involve toy systems with
    known analytical transformations.
    '''
    def __init__(self, targetPotential : callable, input_dim : int, n_layers : int, 
                 omega : float, layer_class, conditioner, conditioner_args : dict = {}, 
                 optimizer=None, scheduler=None):
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

        n_layers : int
            The number of layers in the normalizing flow.
        '''
        BaseModel.__init__(self, targetPotential, input_dim, n_layers, layer_class, conditioner, conditioner_args, optimizer, scheduler)
        
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
            raise NotImplementedError("Only 1D systems are currently supported. 2D and 3D will be added in the future.")
            
        self.toy_ps_to_aa = _toy_ps_to_aa 
        self.aa_to_toy_ps = _aa_to_toy_ps
        self.targetPotential = targetPotential
    
    @abstractmethod
    def ps_to_aa(self, ps):
        pass
    
    @abstractmethod
    def aa_to_ps(self, aa):
        '''
        aa : (theta, J)
        '''
        pass

    @abstractmethod
    def train(self, training_data, steps, lr, loss_function, lf_args, orbit_batching=False, batching_along_orbits=False, batch_size=None, updates=False, nested_progress=False):
        pass

