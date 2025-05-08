'''Defines the conditioners that can be used for the layers.'''

# pytorch
import torch
import torch.nn as nn

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
