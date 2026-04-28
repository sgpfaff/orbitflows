import torch
from functools import partial
from torch import special

### Useful Functions ###

def normalize_(norm, Rforce):
        """
        Normalize a potential in such a way that vc(z=0, R=1)=1., or a fraction of this.
        """
        device = norm.device if hasattr(norm, 'device') else torch.device('cpu')
        dtype = norm.dtype if hasattr(norm, 'dtype') else torch.float64
        return norm / torch.abs(Rforce(torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)))

def gamma(x : torch.tensor):
    # torch.special.gammaln has no working CUDA kernel on Pascal (sm_61) in
    # PyTorch 2.3.x, so fall back to CPU for the special-function call only.
    if hasattr(x, 'is_cuda') and x.is_cuda:
        return torch.special.gammaln(x.cpu()).exp().to(x.device)
    return torch.special.gammaln(x).exp()

def _gammainc(a, x):
    # Same Pascal/CUDA kernel issue as gamma() above; route through CPU.
    a_cuda = hasattr(a, 'is_cuda') and a.is_cuda
    x_cuda = hasattr(x, 'is_cuda') and x.is_cuda
    if a_cuda or x_cuda:
        device = x.device if x_cuda else a.device
        a_in = a.cpu() if a_cuda else a
        x_in = x.cpu() if x_cuda else x
        return special.gammainc(a_in, x_in).to(device)
    return special.gammainc(a, x)

def hyp1f1(a, b, z, max_iter=100, tol=1e-8):
    # Ensure tensors broadcast
    a, b, z = map(lambda t: t.unsqueeze(-1), (a, b, z))
    k = torch.arange(0, max_iter, device=z.device).float()
    # Compute (a)_k / (b)_k for each k
    num = torch.lgamma(a + k) - torch.lgamma(a)
    den = torch.lgamma(b + k) - torch.lgamma(b)
    poch_ratio = torch.exp(num - den)
    # k! via gammaln
    fact = torch.exp(torch.lgamma(k + 1))
    term = poch_ratio * (z**k) / fact  # shape (..., max_iter)
    series = torch.cumsum(term, dim=-1)
    # Early stopping
    last_terms = term[..., -1]
    if torch.max(last_terms.abs()) < tol:
        return series[..., -1]
    else:
        return series[..., -1]
    
### Analytically Tractible Potentials ###

def sho_potential(x, omega):
    return 0.5 * (x**2) * omega**2

### Milky Way Potentials ###

def MWPotential2014(z: torch.tensor, R: torch.tensor, amp: torch.tensor=torch.tensor(1.0).to(torch.float64)):
    '''
    Milky Way potential from Bovy 2015, with the same parameters as galpy's implementation.
    
    Parameters
    ----------
    z : torch.tensor
        vertical coordinate.
    R : torch.tensor
        radial coordinate.
    amp : torch.tensor
        amplitude of the potential, used for normalization. Default is 1.0.
    
    Returns
    -------
    torch.tensor
        potential at the given coordinates.
    '''
    device = z.device
    dtype = z.dtype
    pspc = PowerSphericalPotentialwCutoff(z, R, normalize=0.05, alpha=torch.tensor(1.8, device=device, dtype=dtype), rc=torch.tensor(1.9 / 8.0, device=device, dtype=dtype), amp=amp) 
    mnp = MiyamotoNagaiPotential(z, R, a=torch.tensor(3.0 / 8.0, device=device, dtype=dtype), b=torch.tensor(0.28 / 8.0, device=device, dtype=dtype), normalize=0.6, amp=amp) 
    nfw = NFWPotential(z, R, a=torch.tensor(2.0, device=device, dtype=dtype), normalize=0.35, amp=amp)
    return mnp + nfw + pspc


def MWPotential2014_1D(z : torch.tensor, R:torch.tensor = torch.tensor(1.0), amp:torch.tensor=torch.tensor(1.0)):
    '''
    Vertical Milky Way potential from Bovy 2015, with the same 
    parameters as galpy's implementation.

    Parameters
    ----------
    z : torch.tensor
        vertical coordinate.
    R : torch.tensor
        radial coordinate, default is 1.0 (solar radius).
    amp : torch.tensor
        amplitude of the potential, used for normalization. Default is 1.0.
    
    Returns
    -------
    torch.tensor        
        potential at the given coordinates.
    '''
    midplane_potential = MWPotential2014(torch.tensor(0.0), R, amp=amp)
    return MWPotential2014(z, R=R, amp=amp) - midplane_potential

### Basic 3D Potentials ###

def NFW_Rforce(z : torch.tensor, R : torch.tensor, a : torch.tensor, amp : torch.tensor):
    Rz = R**2.0 + z**2.0
    sqrtRz = torch.sqrt(Rz)
    return amp * R * (
        1.0 / Rz / (a + sqrtRz)
        - torch.log(1.0 + sqrtRz / a) / sqrtRz / Rz
    )

def NFWPotential(z, R, amp=torch.tensor(1.0), a=torch.tensor(1.0), normalize=1.0):
    device = z.device
    dtype = z.dtype
    # Ensure all parameters are on the same device and dtype
    amp = amp.to(device=device, dtype=dtype) if amp.device != device or amp.dtype != dtype else amp
    if isinstance(a, (int, float)):
        a = torch.tensor(a, device=device, dtype=dtype)
    else:
        a = a.to(device=device, dtype=dtype) if a.device != device or a.dtype != dtype else a
    if isinstance(normalize, (int, float)):
        normalize = torch.tensor(normalize, device=device, dtype=dtype)
    else:
        normalize = normalize.to(device=device, dtype=dtype) if normalize.device != device or normalize.dtype != dtype else normalize
    
    amp = amp * normalize_(normalize, partial(NFW_Rforce, a=a, amp=amp))
    r = torch.sqrt(R**2.0 + z**2.0)
    if isinstance(r, (float, int)) and r == 0:
        return - amp / a
    elif isinstance(r, (float, int)):
        return - amp * torch.xlogy(1.0 / r, 1.0 + r / a)  # stable as r -> infty
    else:
        out = - torch.xlogy(1.0 / r, 1.0 + r / a)  # stable as r -> infty
        out[r == 0] = -1.0 / a
        return amp * out

def NFWPotential_1D(z, R=torch.tensor(1.0), amp=torch.tensor(1.0), a=torch.tensor(1.0), normalize=1.0):
    midplane_potential = NFWPotential(torch.tensor(0.0), R, amp=amp, a=a, normalize=normalize)
    return NFWPotential(z, R, amp=amp, a=a, normalize=normalize) - midplane_potential

def MiyamotoNagai_Rforce(z, R, a, b, amp):
     b2 = b**2
     return - amp * R / (R**2.0 + (a + torch.sqrt(z**2.0 + b2)) ** 2.0) ** (
            3.0 / 2.0
        )
     
def MiyamotoNagaiPotential(z, R, a=3.0 / 8.0, b=0.28 / 8.0, normalize=0.6, amp=torch.tensor(1.0)):
    device = z.device
    dtype = z.dtype
    # Ensure all parameters are on the same device and dtype
    if isinstance(a, (int, float)):
        a = torch.tensor(a, device=device, dtype=dtype)
    else:
        a = a.to(device=device, dtype=dtype) if a.device != device or a.dtype != dtype else a
    if isinstance(b, (int, float)):
        b = torch.tensor(b, device=device, dtype=dtype)
    else:
        b = b.to(device=device, dtype=dtype) if b.device != device or b.dtype != dtype else b
    amp = amp.to(device=device, dtype=dtype) if amp.device != device or amp.dtype != dtype else amp
    
    amp = amp * normalize_(normalize, partial(MiyamotoNagai_Rforce, a=a, b=b, amp=amp))
    b2 = b**2.0
    return amp * -1.0 / torch.sqrt(R**2.0 + (a + torch.sqrt(z**2.0 + b2)) ** 2.0)

def MiyamotoNagaiPotential_1D(z, R=torch.tensor(1.0), amp=torch.tensor(1.0), a=torch.tensor(3.0 / 8.0), b=torch.tensor(0.28 / 8.0), normalize=0.6):
    midplane_potential = MiyamotoNagaiPotential(torch.tensor(0.0), R, amp=amp, a=a, b=b, normalize=normalize)
    return MiyamotoNagaiPotential(z, R, amp=amp, a=a, b=b, normalize=normalize) - midplane_potential

def IsochronePotential(z, R, b=torch.tensor(1.0), amp=torch.tensor(1.0)):
    r = torch.sqrt(R**2.0 + z**2.0)
    return - amp / (b + torch.sqrt(r**2.0 + b**2.0))

def PowerSphericalwCutoff_mass_(R, alpha, rc, amp, z=None):
    if z is not None:
        raise AttributeError 
    r2 = R**2.0
    out = 2. * torch.pi * rc ** (3. - alpha) * (gamma(1.5 - 0.5 * alpha) * _gammainc(1.5 - 0.5 * alpha, r2 / rc / rc))
    return out

def PowerSphericalwCutoff_Rforce(z, R, alpha, rc, amp):
    r = torch.sqrt(R * R + z * z)
    mass = PowerSphericalwCutoff_mass_(r, alpha, rc, amp)
    return -amp * mass * R / r**3.0


def PowerSphericalPotentialwCutoff(z, R, amp=torch.tensor(1.0).to(torch.float64), alpha=torch.tensor(1.0).to(torch.float64), rc=torch.tensor(1.0).to(torch.float64), normalize=torch.tensor(1.0).to(torch.float64), r1=torch.tensor(1.0).to(torch.float64)):
    device = z.device
    dtype = z.dtype
    # Ensure all parameters are on the same device and dtype
    amp = amp.to(device=device, dtype=dtype) if hasattr(amp, 'device') else torch.tensor(amp, device=device, dtype=dtype)
    alpha = alpha.to(device=device, dtype=dtype) if hasattr(alpha, 'device') else torch.tensor(alpha, device=device, dtype=dtype)
    rc = rc.to(device=device, dtype=dtype) if hasattr(rc, 'device') else torch.tensor(rc, device=device, dtype=dtype)
    normalize = normalize.to(device=device, dtype=dtype) if hasattr(normalize, 'device') else torch.tensor(normalize, device=device, dtype=dtype)
    r1 = r1.to(device=device, dtype=dtype) if hasattr(r1, 'device') else torch.tensor(r1, device=device, dtype=dtype)
    
    r2 = R**2.0 + z**2.0
    r = torch.sqrt(r2)

    amp = amp * r1**alpha
    amp = amp * normalize_(normalize, partial(PowerSphericalwCutoff_Rforce, alpha=alpha, rc=rc, amp=amp))
    out = (
            2.0
            * torch.pi
            * rc ** (3.0 - alpha)
            * (
                1
                / rc
                * gamma(1.0 - alpha / 2.0)
                * _gammainc(1.0 - alpha / 2.0, (r / rc) ** 2.0)
                - gamma(1.5 - alpha / 2.0)
                * _gammainc(1.5 - alpha / 2.0, (r / rc) ** 2.0)
                / r
            )
        )
    if isinstance(r, (float, int)):
        if r == 0:
            return 0.0
        else:
            return amp * out
    else:
        out[r == 0] = 0.0
        return amp * out

def PowerSphericalPotentialwCutoff_1D(z, R=torch.tensor(1.0).to(torch.float64), amp=torch.tensor(1.0).to(torch.float64), alpha=torch.tensor(1.0).to(torch.float64), rc=torch.tensor(1.0).to(torch.float64), normalize=torch.tensor(1.0).to(torch.float64)): 
    midplane_potential = PowerSphericalPotentialwCutoff(torch.tensor(0.0), R, amp=amp, alpha=alpha, rc=rc, normalize=normalize)
    return PowerSphericalPotentialwCutoff(z, R, amp=amp, alpha=alpha, rc=rc, normalize=normalize) - midplane_potential

# ### Basic 1D Potentials ###

def isoDiskPotential(x, amp=1, sigma=0.1):
    _H = sigma / torch.sqrt(torch.tensor([8.0 * torch.pi * amp]))[0]
    _sigma2 = sigma**2
    return 2.0 * _sigma2 * torch.log(torch.cosh(0.5 * x / _H))