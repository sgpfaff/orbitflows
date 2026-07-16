'''
Target potentials used to train the flows.

The Milky Way potential is galpy's ``MWPotential2014`` (Bovy 2015). Evaluating a
galpy potential on torch tensors dispatches to galpy's torch backend, so the
returned potential values are differentiable with respect to the input
coordinates -- which is what the training loop needs. This replaces the previous
hand-written PyTorch reimplementations of MWPotential2014 and its components.
'''

import torch

from galpy.potential import MWPotential2014 as _MWPotential2014
from galpy.potential import evaluatePotentials as _evaluatePotentials


### Analytically tractable toy potentials ###

def sho_potential(x, omega):
    '''Potential of the 1D harmonic oscillator, 0.5 * (omega * x)^2.'''
    return 0.5 * (x**2) * omega**2


def isoDiskPotential(x, amp=1, sigma=0.1):
    '''Self-gravitating isothermal-sheet potential.'''
    _H = sigma / torch.sqrt(torch.tensor([8.0 * torch.pi * amp]))[0]
    _sigma2 = sigma**2
    return 2.0 * _sigma2 * torch.log(torch.cosh(0.5 * x / _H))


### Milky Way potential (galpy torch backend) ###

def _match(value, ref):
    '''Return ``value`` as a tensor sharing ``ref``'s dtype and device.'''
    if torch.is_tensor(value):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def MWPotential2014(z, R=1.0, amp=1.0):
    '''
    galpy's MWPotential2014 (Bovy 2015) evaluated through the torch backend.

    Parameters
    ----------
    z : torch.Tensor
        Vertical coordinate.
    R : float or torch.Tensor
        Cylindrical radius (default 1.0, i.e. the solar radius in galpy's
        internal units).
    amp : float or torch.Tensor
        Overall amplitude multiplying the potential (default 1.0).

    Returns
    -------
    torch.Tensor
        Potential value(s), differentiable with respect to ``z`` (and ``R``).
    '''
    Rt = _match(R, z)
    return _match(amp, z) * _evaluatePotentials(_MWPotential2014, Rt, z, use_physical=False)


def MWPotential2014_1D(z, R=1.0, amp=1.0):
    '''
    Vertical MWPotential2014 at fixed R, i.e. ``Phi(R, z) - Phi(R, 0)``, evaluated
    through galpy's torch backend.

    The signature mirrors the original custom implementation so that models saved
    with ``targetPotentialKey='MWPotential2014_1D'`` and
    ``potential_kwargs={'R': ..., 'amp': ...}`` keep loading.

    Parameters
    ----------
    z : torch.Tensor
        Vertical coordinate.
    R : float or torch.Tensor
        Cylindrical radius, default 1.0 (the solar radius).
    amp : float or torch.Tensor
        Overall amplitude of the potential, default 1.0.

    Returns
    -------
    torch.Tensor
        Potential value(s) relative to the midplane, differentiable w.r.t. ``z``.
    '''
    Rt = _match(R, z)
    ampt = _match(amp, z)
    zero = torch.zeros((), dtype=z.dtype, device=z.device)
    phi_z = _evaluatePotentials(_MWPotential2014, Rt, z, use_physical=False)
    midplane_potential = _evaluatePotentials(_MWPotential2014, Rt, zero, use_physical=False)
    return ampt * (phi_z - midplane_potential)
