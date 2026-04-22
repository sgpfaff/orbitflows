'''Defines the structure of the normalizing flow.'''

import torch.nn as nn

class Flow(nn.Module):
    """Normalizing Flow with symplectic transformations alternating between evolving q and p."""
    
    def __init__(self, input_dim, num_layers, layer_class, conditioner, conditioner_args={}):
        super(Flow, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.layer_class = layer_class
        self.conditioner = conditioner
        
        # Alternate between updating q and p in each layer
        for i in range(num_layers):
            update_q = (i % 2 == 0)  # True for layers updating q, False for layers updating p
            conditioner = self.conditioner(**conditioner_args)
            self.layers.append(self.layer_class(conditioner=conditioner, update_q=update_q))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z