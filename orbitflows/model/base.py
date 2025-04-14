'''Top level class for all models.'''

from abc import ABC, abstractmethod
from ..flow import GsympNetFlow
import torch

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
    def train(self, training_data, steps, loss_function):
        '''Train the model.'''
        pass

    @abstractmethod
    def hamiltonian(self, j, theta):
        pass

    @abstractmethod
    def aa_to_ps(self, aa):
        pass

    @abstractmethod
    def ps_to_aa(self, ps):
        pass

    @abstractmethod
    def save(self):
        pass
