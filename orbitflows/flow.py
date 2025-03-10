# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# misc.
from typing import Callable


class GradientBasedConditioner(nn.Module):
    """Gradient-based conditioner module in PyTorch."""
    def __init__(self, activation: Callable[[torch.Tensor], torch.Tensor], projection_dims: int, skip_connections: bool = True, input_dim: int = 2):
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


class SymplecticCouplingLayer(nn.Module):
    def __init__(self, conditioner: nn.Module, update_q: bool = True):
        super().__init__()
        self.conditioner = conditioner
        self.update_q = update_q

    def forward(self, x):
        # Split input into two parts
        q, p = x.chunk(2, dim=-1)
        
        # Compute scale and shift from conditioner
        if self.update_q:
            scale_shift = self.conditioner(p)
            shift = scale_shift.chunk(2, dim=-1)
            # Apply symplectic transformation
            q = q + shift[0]

        else:
            scale_shift = self.conditioner(q)
            shift = scale_shift.chunk(2, dim=-1)
            # Apply symplectic transformation
            p = p + shift[0]

        return torch.cat([q, p], dim=-1)
    def inverse(self, y):
        q, p = y.chunk(2, dim=-1)
            
        if self.update_q:
            scale_shift = self.conditioner(p)
            shift = scale_shift.chunk(2, dim=-1)
            q = q - shift[0]
        else:
            scale_shift = self.conditioner(q)
            shift = scale_shift.chunk(2, dim=-1)
            p = p - shift[0]

        return torch.cat([q, p], dim=-1)
    
class GsympNetFlow(nn.Module):
    """Normalizing Flow with symplectic transformations alternating between evolving q and p."""
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GsympNetFlow, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Alternate between updating q and p in each layer
        for i in range(num_layers):
            update_q = (i % 2 == 0)  # True for layers updating q, False for layers updating p
            conditioner = GradientBasedConditioner(activation=F.relu, projection_dims=hidden_dim, input_dim=input_dim)
            self.layers.append(SymplecticCouplingLayer(conditioner=conditioner, update_q=update_q))
    
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