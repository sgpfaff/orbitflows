'''Defines the different types of layers to be used for the flow.'''

# pytorch
import torch
import torch.nn as nn


class SymplecticCouplingLayer(nn.Module):
    """Symplectic coupling layer."""
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