'''Defines the different types of layers to be used for the flow.'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymplecticCouplingLayer(nn.Module):
    """
    Symplectic coupling layer.
    """
    def __init__(self, conditioner: nn.Module, update_q: bool = True):
        super().__init__()
        self.conditioner = conditioner
        self.update_q = update_q

    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(p)
            q = q + scale_shift
        else:
            scale_shift = self.conditioner(q)
            p = p + scale_shift
        return torch.cat([q, p], dim=-1)
    
    def inverse(self, y):
        q, p = y.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(p)
            q = q - scale_shift
        else:
            scale_shift = self.conditioner(q)
            p = p - scale_shift
        return torch.cat([q, p], dim=-1)
    
class PointTransformationLayer(nn.Module):
    """
    Point transformation layer.
    """
    def __init__(self, conditioner: nn.Module, update_q: bool = True):
        super().__init__()
        self.conditioner = conditioner
        self.update_q = update_q

    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(q)
            q = q + scale_shift 
        else:
            scale_shift = self.conditioner(p)
            p = p + scale_shift 
        return torch.cat([q, p], dim=-1)
    
    def inverse(self, y):
        q, p = y.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(q)
            q = q - scale_shift 
        else:
            scale_shift = self.conditioner(p)
            p = p - scale_shift
        return torch.cat([q, p], dim=-1)
    
class WrappedAnglesCouplingLayer(nn.Module):
    """
    Symplectic coupling layer on a torus. 
    Wraps angles around 2 pi.
    """
    def __init__(self, conditioner: nn.Module, update_q: bool = True):
        super().__init__()
        self.conditioner = conditioner
        self.update_q = update_q

    def forward(self, x):
        q, p = x.chunk(2, dim=-1)

        if self.update_q:
            scale_shift = self.conditioner(p)
            q = ((q + scale_shift + torch.pi) % 
                 (2*torch.pi) - torch.pi)
        else:
            scale_shift = self.conditioner(q)
            p = p + scale_shift 
        return torch.cat([q, p], dim=-1)
    
    def inverse(self, y):
        q, p = y.chunk(2, dim=-1)
            
        if self.update_q:
            scale_shift = self.conditioner(p)
            q = ((q - scale_shift + torch.pi) % 
                 (2*torch.pi) - torch.pi)
        else:
            scale_shift = self.conditioner(q)
            p = p - scale_shift
        return torch.cat([q, p], dim=-1)

class TorusSymplecticCouplingLayer(nn.Module):
    """
    Symplectic coupling layer on a torus. 
    Uses sin(q) to update p.
    """
    def __init__(self, conditioner: nn.Module, update_q: bool = True):
        super().__init__()
        self.conditioner = conditioner
        self.update_q = update_q

    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(p)
            q = q + scale_shift
        else:
            scale_shift = self.conditioner(torch.sin(q))
            p = p + scale_shift
        return torch.cat([q, p], dim=-1)
    def inverse(self, y):
        q, p = y.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(p)
            q = q - scale_shift
        else:
            scale_shift = self.conditioner(torch.sin(q))
            p = p - scale_shift
        return torch.cat([q, p], dim=-1)
    
class PSymmetricSymplecticCouplingLayer(nn.Module):
    """
    Symplectic coupling layer that is symmetric in p.
    """
    def __init__(self, conditioner: nn.Module, update_q: bool = True):
        super().__init__()
        self.conditioner = conditioner
        self.update_q = update_q

    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        if self.update_q:
            sign_q = torch.sign(q)
            scale_shift = self.conditioner(torch.abs(p))
            q = q + scale_shift * sign_q

        else:
            sign_p = torch.sign(p)
            scale_shift = self.conditioner(torch.abs(q))
            p = p + scale_shift * sign_p
        return torch.cat([q, p], dim=-1)
    
    def inverse(self, y):
        q, p = y.chunk(2, dim=-1)
        if self.update_q:
            scale_shift = self.conditioner(p)
            q = q - scale_shift
        else:
            scale_shift = self.conditioner(q)
            p = p - scale_shift
        return torch.cat([q, p], dim=-1)

