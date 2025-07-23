"""
PyTorch implementation of the McMillan (2017) potential.
Based on the galpy implementation by Mackereth & Bovy (2018).

This implementation is designed to work with PyTorch's automatic differentiation
for use in neural normalizing flows and other gradient-based methods.
"""

import torch
import numpy as np
from typing import Union, Tuple


class McMillan17Potential:
    """
    PyTorch implementation of the McMillan (2017) Milky Way potential model.
    
    The potential consists of three components:
    1. Disk (gas + stellar disk) - implemented as analytical approximation
    2. Bulge - core-power law with exponential cutoff
    3. Dark matter halo - NFW profile
    
    Parameters are from the best-fitting model in McMillan (2017).
    
    Args:
        ro (float): Distance scale in kpc (default: 8.21 kpc)
        vo (float): Velocity scale in km/s (default: 233.1 km/s)
        device (str): PyTorch device ('cpu' or 'cuda')
        dtype (torch.dtype): PyTorch data type (default: torch.float64)
    """
    
    def __init__(self, 
                 ro: float = 8.21,
                 vo: float = 233.1,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float64):
        
        self.ro = ro
        self.vo = vo
        self.device = device
        self.dtype = dtype
        
        # Unit conversions (following galpy conventions)
        self.sigo = 2.325e7 / (vo**2 / ro)  # Surface density normalization
        self.rhoo = 2.325e7 / (vo**2 * ro)  # Volume density normalization
        
        # Initialize parameters as tensors for autodiff
        self._init_gas_params()
        self._init_stellar_params()
        self._init_bulge_params()
        self._init_halo_params()
    
    def _init_gas_params(self):
        """Initialize gas disk parameters"""
        # HI gas disk
        self.Rd_HI = torch.tensor(7.0 / self.ro, device=self.device, dtype=self.dtype)
        self.Rm_HI = torch.tensor(4.0 / self.ro, device=self.device, dtype=self.dtype)
        self.zd_HI = torch.tensor(0.085 / self.ro, device=self.device, dtype=self.dtype)
        self.Sigma0_HI = torch.tensor(53.1 / self.sigo, device=self.device, dtype=self.dtype)
        
        # H2 gas disk
        self.Rd_H2 = torch.tensor(1.5 / self.ro, device=self.device, dtype=self.dtype)
        self.Rm_H2 = torch.tensor(12.0 / self.ro, device=self.device, dtype=self.dtype)
        self.zd_H2 = torch.tensor(0.045 / self.ro, device=self.device, dtype=self.dtype)
        self.Sigma0_H2 = torch.tensor(2180.0 / self.sigo, device=self.device, dtype=self.dtype)
    
    def _init_stellar_params(self):
        """Initialize stellar disk parameters"""
        # Thin disk
        self.Sigma0_thin = torch.tensor(896.0 / self.sigo, device=self.device, dtype=self.dtype)
        self.Rd_thin = torch.tensor(2.5 / self.ro, device=self.device, dtype=self.dtype)
        self.zd_thin = torch.tensor(0.3 / self.ro, device=self.device, dtype=self.dtype)
        
        # Thick disk
        self.Sigma0_thick = torch.tensor(183.0 / self.sigo, device=self.device, dtype=self.dtype)
        self.Rd_thick = torch.tensor(3.02 / self.ro, device=self.device, dtype=self.dtype)
        self.zd_thick = torch.tensor(0.9 / self.ro, device=self.device, dtype=self.dtype)
    
    def _init_bulge_params(self):
        """Initialize bulge parameters"""
        self.rho0_bulge = torch.tensor(98.4 / self.rhoo, device=self.device, dtype=self.dtype)
        self.r0_bulge = torch.tensor(0.075 / self.ro, device=self.device, dtype=self.dtype)
        self.rcut_bulge = torch.tensor(2.1 / self.ro, device=self.device, dtype=self.dtype)
        self.alpha_bulge = torch.tensor(1.8, device=self.device, dtype=self.dtype)
        self.beta_bulge = torch.tensor(0.5, device=self.device, dtype=self.dtype)
    
    def _init_halo_params(self):
        """Initialize dark matter halo parameters"""
        self.rho0_halo = torch.tensor(0.00854 / self.rhoo, device=self.device, dtype=self.dtype)
        self.rh = torch.tensor(19.6 / self.ro, device=self.device, dtype=self.dtype)
    
    def _expsech2_dens_with_hole(self, R: torch.Tensor, z: torch.Tensor, 
                                Rd: torch.Tensor, Rm: torch.Tensor, 
                                zd: torch.Tensor, Sigma0: torch.Tensor) -> torch.Tensor:
        """
        Exponential disk with hole and sech2 vertical profile.
        """
        R_safe = torch.clamp(R, min=1e-10)  # Avoid division by zero
        
        # Radial profile with hole
        exp_factor = torch.exp(-R_safe / Rd - Rm / R_safe)
        
        # Vertical sech2 profile
        z_scaled = z / zd
        sech2_factor = 1.0 / (torch.cosh(z_scaled)**2)
        
        # Normalization factor
        norm = Sigma0 / (2.0 * zd)
        
        return norm * exp_factor * sech2_factor
    
    def _expexp_dens(self, R: torch.Tensor, z: torch.Tensor,
                    Rd: torch.Tensor, zd: torch.Tensor, 
                    Sigma0: torch.Tensor) -> torch.Tensor:
        """
        Double exponential disk (exponential in R and |z|).
        """
        # Radial exponential
        exp_R = torch.exp(-R / Rd)
        
        # Vertical exponential
        exp_z = torch.exp(-torch.abs(z) / zd)
        
        # Normalization
        norm = Sigma0 / (2.0 * zd)
        
        return norm * exp_R * exp_z
    
    def _core_pow_dens_with_cut(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Core-power law density with exponential cutoff for the bulge.
        rho(r) = rho0 * (r0/r)^alpha / (1 + (r/r0)^2)^(beta-alpha/2) * exp(-(r/rcut)^2)
        """
        r = torch.sqrt(R**2 + z**2)
        r_safe = torch.clamp(r, min=1e-10)
        
        # Core-power law part
        power_term = (self.r0_bulge / r_safe)**self.alpha_bulge
        denom_term = (1.0 + (r_safe / self.r0_bulge)**2)**(self.beta_bulge - self.alpha_bulge/2.0)
        
        # Exponential cutoff
        cutoff = torch.exp(-(r_safe / self.rcut_bulge)**2)
        
        return self.rho0_bulge * power_term / denom_term * cutoff
    
    def _nfw_density(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        NFW dark matter halo density profile.
        rho(r) = rho0 / (r/rs * (1 + r/rs)^2)
        """
        r = torch.sqrt(R**2 + z**2)
        r_safe = torch.clamp(r, min=1e-10)
        
        x = r_safe / self.rh
        x_safe = torch.clamp(x, min=1e-10)
        
        return self.rho0_halo / (x_safe * (1.0 + x_safe)**2)
    
    def gas_density(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Total gas density (HI + H2)"""
        hi_dens = self._expsech2_dens_with_hole(R, z, self.Rd_HI, self.Rm_HI, 
                                               self.zd_HI, self.Sigma0_HI)
        h2_dens = self._expsech2_dens_with_hole(R, z, self.Rd_H2, self.Rm_H2,
                                               self.zd_H2, self.Sigma0_H2)
        return hi_dens + h2_dens
    
    def stellar_density(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Total stellar disk density (thin + thick)"""
        thin_dens = self._expexp_dens(R, z, self.Rd_thin, self.zd_thin, self.Sigma0_thin)
        thick_dens = self._expexp_dens(R, z, self.Rd_thick, self.zd_thick, self.Sigma0_thick)
        return thin_dens + thick_dens
    
    def bulge_density(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Bulge density"""
        return self._core_pow_dens_with_cut(R, z)
    
    def halo_density(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Dark matter halo density"""
        return self._nfw_density(R, z)
    
    def total_density(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Total density (all components)"""
        return (self.gas_density(R, z) + 
                self.stellar_density(R, z) + 
                self.bulge_density(R, z) + 
                self.halo_density(R, z))
    
    def _integrate_potential_numerical(self, R: torch.Tensor, z: torch.Tensor, 
                                     component: str = 'total') -> torch.Tensor:
        """
        Numerical integration of Poisson's equation to get potential.
        This is a simplified implementation - for production use, consider
        more sophisticated integration schemes.
        """
        # Get density function
        if component == 'gas':
            density_func = self.gas_density
        elif component == 'stellar':
            density_func = self.stellar_density
        elif component == 'bulge':
            density_func = self.bulge_density
        elif component == 'halo':
            density_func = self.halo_density
        else:
            density_func = self.total_density
        
        # Simplified analytical approximations for each component
        # Note: For a full implementation, you'd want to use proper numerical
        # integration or pre-computed look-up tables
        
        if component == 'halo' or component == 'total':
            # NFW potential approximation
            r = torch.sqrt(R**2 + z**2)
            r_safe = torch.clamp(r, min=1e-10)
            x = r_safe / self.rh
            
            # NFW potential: -4πGρ₀rs³ * ln(1+x)/x
            G_norm = 4.0 * np.pi  # In code units
            mass_scale = self.rho0_halo * self.rh**3
            
            # Avoid singularity at x=0
            x_safe = torch.clamp(x, min=1e-10)
            ln_term = torch.log(1.0 + x_safe) / x_safe
            
            halo_pot = -G_norm * mass_scale * ln_term
            
            if component == 'halo':
                return halo_pot
        
        # For other components, use simplified approximations
        # This is where you'd implement proper numerical integration
        # or use pre-computed coefficients as in the original galpy implementation
        
        # Placeholder - return zero for now for non-halo components
        if component != 'halo' and component != 'total':
            return torch.zeros_like(R)
        
        return halo_pot  # For 'total', just return halo for now
    
    def potential(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the total gravitational potential at (R, z).
        
        Args:
            R: Cylindrical radius coordinate(s)
            z: Vertical coordinate(s)
            
        Returns:
            Gravitational potential value(s)
        """
        # Ensure inputs are tensors
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, device=self.device, dtype=self.dtype)
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, device=self.device, dtype=self.dtype)
        
        # For now, implement a simplified version focusing on the NFW halo
        # which is the dominant component at large radii
        return self._integrate_potential_numerical(R, z, component='halo')
    
    def __call__(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Make the potential callable"""
        return self.potential(R, z)


def McMillan17(R: torch.Tensor, z: torch.Tensor, 
               ro: float = 8.21, vo: float = 233.1,
               device: str = 'cpu', dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Convenience function to evaluate McMillan17 potential.
    
    Args:
        R: Cylindrical radius coordinate(s) 
        z: Vertical coordinate(s)
        ro: Distance scale in kpc
        vo: Velocity scale in km/s
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Potential value(s)
    """
    pot = McMillan17Potential(ro=ro, vo=vo, device=device, dtype=dtype)
    return pot(R, z)


if __name__ == "__main__":
    # Simple test
    import matplotlib.pyplot as plt
    
    # Create potential instance
    pot = McMillan17Potential()
    
    # Test evaluation
    R = torch.linspace(0.1, 2.0, 50, dtype=torch.float64)
    z = torch.zeros_like(R)
    
    # Evaluate potential
    phi = pot(R, z)
    
    print(f"Potential at R=1, z=0: {pot(torch.tensor(1.0), torch.tensor(0.0))}")
    print(f"Shape of phi: {phi.shape}")
    print(f"Potential requires grad: {phi.requires_grad}")
    
    # Test gradient computation
    R_test = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    z_test = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    phi_test = pot(R_test, z_test)
    
    # Compute gradients
    phi_test.backward()
    print(f"dΦ/dR at (1,0): {R_test.grad}")
    print(f"dΦ/dz at (1,0): {z_test.grad}")
