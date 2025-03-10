import torch
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from .. import GsympNetFlow
import numpy as np

class DataLoader():
    def __init__(self, filename, relative_path='../../training_data/'):
        self.filename = filename
        self.relative_path = relative_path
        self.data, self.config = self.load(filename, relative_path)
        self.ps = self.data['phase space']
        self.aa = self.data['action angle']
        self.n_orbits = self.config['num_orbits']
        self.n_steps = self.config['num_steps']
        self.potential = self.config['potential']
        self.potential_values = self.config['potential_values'] # a dictionary of values
        self.start_time = self.config['integration_start_time']
        self.end_time = self.config['integration_end_time']
        self.r_upper = self.config['r_upper_bound']
        self.r_lower = self.config['r_lower_bound']
        self.t_ls = torch.linspace(self.start_time, self.end_time, self.n_steps)

    def load(self, filename, relative_path):
        data = torch.load(f'{relative_path}{filename}/{filename}.pt')
        # load integration details:
        with open(f'{relative_path}{filename}/{filename}_config.json', 'r') as config_file:
            config = json.load(config_file)
        

        return data, config

class ModelLoader():
    def __init__(self, filename, relative_path='../../trained_models/'):
        self.filename = filename
        self.relative_path = relative_path
        self.flow, self.config, self.loss_list = self.load(filename, relative_path)
        self.model_name = self.config['model_name']
        self.training_steps = self.config['training_steps']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.input_dim = self.config['input_dim']
        self.potential = self.config['potential']
        self.potential_values = self.config['potential_values']
        self.filename = self.config['filename']
        self.output_frequency = self.config['output_frequency']
        self.optimizer = self.config['optimizer']
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.num_layers = self.config['num_layers']
    

    def load(self, filename, relative_path):
        with open(f'{relative_path}{filename}/{filename}_config.json', 'r') as config_file:
            config = json.load(config_file)
        loaded_model = GsympNetFlow(config['input_dim'], config['hidden_dim'], config['num_layers'])
        loaded_model.load_state_dict(torch.load(f'{relative_path}{filename}/{filename}.pt'))
        # load loss list numpy array
        loss_list = np.load(f'{relative_path}{filename}/{filename}_loss.npy')
        return loaded_model, config, loss_list
