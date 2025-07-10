'''Defines the structure of the normalizing flow.'''

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SymplecticCouplingLayer


# class GsympNetFlow(nn.Module):
#     """Normalizing Flow with symplectic transformations alternating between evolving q and p."""
    
#     def __init__(
#             self, 
#             input_dim, 
#             hidden_dim, 
#             num_layers, 
#             conditioner = Gradient
#             ):
#         super(GsympNetFlow, self).__init__()
#         self.layers = nn.ModuleList()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         # Create default conditioner if none provided
#         if conditioner is None:
#             self.conditioner = GradientBasedConditioner,
#         else:
#             self.conditioner = conditioner
        
#         # Alternate between updating q and p in each layer
#         for i in range(num_layers):
#             update_q = (i % 2 == 0)  # True for layers updating q, False for layers updating p
#             new_conditioner = self.conditioner(
#                 activation=F.relu, 
#                 projection_dims=self.hidden_dim, 
#                 input_dim=self.input_dim
#                 )
#             self.layers.append(SymplecticCouplingLayer(conditioner=new_conditioner, update_q=update_q))
    
#     def forward(self, x):
#         log_det_jacobian = 0
#         for layer in self.layers:
#             x = layer(x)#x, s = layer(x)
#             #log_det_jacobian += s.sum(dim=-1)  # log|det(Jacobian)|
#         return x #, log_det_jacobian
    
#     def inverse(self, z):
#         for layer in reversed(self.layers):
#             z = layer.inverse(z)
#         return z
    

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
        log_det_jacobian = 0
        for layer in self.layers:
            x = layer(x)#x, s = layer(x)
            #log_det_jacobian += s.sum(dim=-1)  # log|det(Jacobian)|
        return x #, log_det_jacobian
    
    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

# class GsympNetFlow(nn.Module):
#     """Normalizing Flow with symplectic transformations alternating between evolving q and p."""
    
#     def __init__(self, input_dim, hidden_dim, num_layers):
#         super(GsympNetFlow, self).__init__()
#         self.layers = nn.ModuleList()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         # Alternate between updating q and p in each layer
#         for i in range(num_layers):
#             update_q = (i % 2 == 0)  # True for layers updating q, False for layers updating p
#             conditioner = GradientBasedConditioner(activation=F.relu, projection_dims=hidden_dim, input_dim=input_dim)
#             self.layers.append(SymplecticCouplingLayer(conditioner=conditioner, update_q=update_q))
    
#     def forward(self, x):
#         log_det_jacobian = 0
#         for layer in self.layers:
#             x = layer(x)#x, s = layer(x)
#             #log_det_jacobian += s.sum(dim=-1)  # log|det(Jacobian)|
#         return x #, log_det_jacobian
    
#     def inverse(self, z):
#         for layer in reversed(self.layers):
#             z = layer.inverse(z)
#         return z