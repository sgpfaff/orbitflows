'''Defines the conditioners that can be used for the layers.'''

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# misc.
from typing import Callable


class GradientBasedConditioner(nn.Module):
    """Gradient-based conditioner module in PyTorch."""
    def __init__(self, activation: Callable[[torch.Tensor], torch.Tensor] = F.relu, projection_dims: int = 10, skip_connections: bool = True, input_dim: int = 2):
        super().__init__()
        self.activation = activation
        self.projection_dims = projection_dims
        self.skip_connections = skip_connections
        self.input_dim = input_dim // 2

        # Parameters
        self.w = nn.Parameter(torch.empty(self.input_dim, projection_dims))
        self.b = nn.Parameter(torch.empty(projection_dims))
        self.a = nn.Parameter(torch.zeros(projection_dims))
        self.gate = nn.Parameter(torch.zeros(self.input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.w, mean=0.0, std=0.01)
        nn.init.normal_(self.b, mean=0.0, std=0.01)
        nn.init.zeros_(self.a)
        nn.init.zeros_(self.gate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Apply the conditioner
        outputs = self.activation(inputs @ self.w + self.b)
        outputs = outputs * self.a
        outputs = outputs @ self.w.T

        # Add skip connections if enabled
        if self.skip_connections:
            outputs += inputs

        # Apply the gating mechanism
        outputs *= self.gate
        return outputs


class NNConditioner(nn.Module):
    """Linear conditioner module using standard PyTorch linear layers."""
    def __init__(self, 
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 projection_dims: int = 10,
                 num_layers: int = 3, 
                 skip_connections: bool = True,
                 input_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim // 2
        self.hidden_dim = projection_dims
        self.num_layers = num_layers
        self.activation = activation
        self.skip_connections = skip_connections
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        else:
            # Single layer case
            layers[0] = nn.Linear(self.input_dim, self.input_dim)
        
        self.layers = nn.ModuleList(layers)
        
        # Optional gating mechanism
        self.gate = nn.Parameter(torch.zeros(self.input_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        
        # Forward pass through layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        # Add skip connections if enabled
        if self.skip_connections:
            x += inputs
        
        # Apply gating mechanism
        x *= self.gate
        
        return x

class LinearConditioner(nn.Module):
    '''Normalizing flow layer consisting of a single linear neural network'''
    def __init__(self, input_dims : int = 2, include_bias : bool = True):
        super().__init__() 
        self.linear = nn.Linear(in_features=input_dims//2, out_features=input_dims//2, bias=include_bias)

    def forward(self, inputs : torch.Tensor):
        return self.linear(inputs)

    
class SimpleNNConditioner(nn.Module):
    '''Normalizing flow layer consisting of a simple neural network'''
    def __init__(
            self, 
            input_dims : int = 2, 
            include_bias : bool = True, 
            projection_dims : int = 10,
            num_layers : int = 3,
            activation = nn.ReLU,
            gate : bool = True
            ):
        # num_layers is the TOTAL number of layers
        super().__init__() 
        self.input_dims = input_dims
        self.num_layers = num_layers
        self.projection_dims = projection_dims
        self.include_bias = include_bias
        self.activation = activation
        self.gate = nn.Parameter(torch.zeros(self.input_dims//2))

        layers = []
        layers.append(nn.Linear(in_features=self.input_dims//2, out_features=self.projection_dims, bias=self.include_bias))
        layers.append(activation())
        for _ in torch.arange(0, self.num_layers - 2):
            layers.append(nn.Linear(in_features=self.projection_dims, out_features=self.projection_dims, bias=self.include_bias))
            layers.append(activation())

        layers.append(nn.Linear(in_features=self.projection_dims, out_features=self.input_dims//2, bias=self.include_bias))
        self.layers = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate)

    def forward(self, inputs : torch.Tensor):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = x * self.gate
        return x
    


class SymmetricNNConditioner(nn.Module):
    '''Normalizing flow layer consisting of a simple neural network that is symmetric across p=0'''
    def __init__(
            self, 
            input_dims : int = 2, 
            include_bias : bool = True, 
            projection_dims : int = 10,
            num_layers : int = 3,
            activation = nn.ReLU):
        # num_layers is the TOTAL number of layers
        super().__init__() 
        self.input_dims = input_dims
        self.num_layers = num_layers
        self.projection_dims = projection_dims
        self.include_bias = include_bias
        self.activation = activation

        layers = []
        layers.append(nn.Linear(in_features=self.input_dims//2, out_features=self.projection_dims, bias=self.include_bias))
        layers.append(activation())
        for _ in torch.arange(0, self.num_layers - 2):
            layers.append(nn.Linear(in_features=self.projection_dims, out_features=self.projection_dims, bias=self.include_bias))
            layers.append(activation())

        layers.append(nn.Linear(in_features=self.projection_dims, out_features=self.input_dims//2, bias=self.include_bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs : torch.Tensor):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x