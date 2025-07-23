from ..flow.normFlow import Flow
import json
import torch
import numpy as np

def load(self, model, filename, relative_path):
    with open(f'{relative_path}{filename}/{filename}_config.json', 'r') as config_file:
        config = json.load(config_file)
    model.flow = Flow(config['input_dim'], config['hidden_dim'], config['num_layers'])
    model.flow.load_state_dict(torch.load(f'{relative_path}{filename}/{filename}.pt'))
    loss_list = np.load(f'{relative_path}{filename}/{filename}_loss.npy')
    return model, config, loss_list