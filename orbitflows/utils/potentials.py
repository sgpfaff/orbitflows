import torch
from functools import partial
from .functions import gamma
from scipy import special

### Useful Functions ###
def normalize_(norm, Rforce):
        """
        Normalize a potential in such a way that vc(z=0, R=1)=1., or a fraction of this.
        """
        return  norm / torch.abs(Rforce(torch.tensor(0.0).to(torch.float64), torch.tensor(1.0).to(torch.float64)))

### Milky Way Potentials ###

# def McMillan17():
#     McMillan_bulge = SCFPotential(Acos=scf_compute_coeffs_axi(bulge_dens, 20, 10, a=0.1)[0], a=0.1, ro=ro, vo=vo)
#     McMillan_disk = DiskSCFPotential(
#         dens=lambda R, z: gas_dens(R, z) + stellar_dens(R, z),
#         Sigma=sigmadict,
#         hz=hzdict,
#         a=2.5,
#         N=30,
#         L=30,
#         ro=ro,
#         vo=vo,
#     )
#     McMillan_halo = NFWPotential(amp=rho0_halo * (4 * numpy.pi * rh**3), a=rh, ro=ro, vo=vo)
#     # Go back to old floating-point warnings settings
#     return McMillan_disk + McMillan_halo + McMillan_bulge

def MWPotential2014(z: torch.tensor, R: torch.tensor, amp: torch.tensor=torch.tensor(1.0).to(torch.float64)):
    #pspc = PowerSphericalPotentialwCutoff(z, R, normalize=0.05, alpha=torch.tensor(1.8).to(torch.float64), rc=torch.tensor(1.9 / 8.0).to(torch.float64), amp=amp) 
    mnp = MiyamotoNagaiPotential(z, R, a=torch.tensor(3.0 / 8.0).to(torch.float64), b=torch.tensor(0.28 / 8.0).to(torch.float64), normalize=0.6, amp=amp) 
    nfw = NFWPotential(z, R, a=torch.tensor(2.0).to(torch.float64), normalize=0.35, amp=amp)
    return mnp + nfw #+ pspc


def MWPotential2014_1D(z : torch.tensor, R:torch.tensor = torch.tensor(1.0), amp:torch.tensor=torch.tensor(1.0)):
    return MWPotential2014(z, R=R, amp=amp)

### Basic 3D Potentials ###

def NFW_Rforce(z : torch.tensor, R : torch.tensor, a : torch.tensor, amp : torch.tensor):
    Rz = R**2.0 + z**2.0
    sqrtRz = torch.sqrt(Rz)
    return amp * R * (
        1.0 / Rz / (a + sqrtRz)
        - torch.log(1.0 + sqrtRz / a) / sqrtRz / Rz
    )

def NFWPotential(z, R, amp=torch.tensor(1.0), a=torch.tensor(1.0), normalize=1.0):
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
    

def MiyamotoNagai_Rforce(z, R, a, b, amp):
     b2 = b**2
     return - amp * R / (R**2.0 + (a + torch.sqrt(z**2.0 + b2)) ** 2.0) ** (
            3.0 / 2.0
        )
     
def MiyamotoNagaiPotential(z, R, a=3.0 / 8.0, b=0.28 / 8.0, normalize=0.6, amp=torch.tensor(1.0)):
    amp = amp * normalize_(normalize, partial(MiyamotoNagai_Rforce, a=a, b=b, amp=amp))
    b2 = b**2.0
    return amp * -1.0 / torch.sqrt(R**2.0 + (a + torch.sqrt(z**2.0 + b2)) ** 2.0)
    

def PowerSphericalwCutoff_mass(R, alpha, rc, z=None):
    if z is not None:
        raise AttributeError  # use general implementation
    R = torch.tensor(R)
    out = torch.ones_like(R)
    out[~torch.isinf(R)] = (
        2.0
        * torch.pi
        * R[~torch.isinf(R)] ** (3.0 - alpha)
        / (1.5 - alpha / 2.0)
        * special.hyp1f1(
            1.5 - alpha / 2.0,
            2.5 - alpha / 2.0,
            -((R[~torch.isinf(R)] / rc) ** 2.0),
        )
    )
    out[torch.isinf(R)] = (
        2.0
        * torch.pi
        * rc ** (3.0 - alpha)
        * gamma(1.5 - alpha / 2.0)
    )
    return out

def PowerSphericalwCutoff_Rforce(z, R, alpha, rc, amp):
    r = torch.sqrt(R * R + z * z)
    mass = PowerSphericalwCutoff_mass(r, alpha, rc)
    return -amp * mass * R / r**3.0


def PowerSphericalPotentialwCutoff(z, R, amp=torch.tensor(1.0).to(torch.float64), alpha=torch.tensor(1.0).to(torch.float64), rc=torch.tensor(1.0).to(torch.float64), normalize=torch.tensor(1.0).to(torch.float64), r1=torch.tensor(1.0).to(torch.float64)):
    r = torch.sqrt(R**2.0 + z**2.0)
    amp = amp * normalize_(normalize, partial(PowerSphericalwCutoff_Rforce, alpha, rc, amp=amp))
    out = (
        2.0
        * torch.pi
        * rc ** (3.0 - alpha)
        * (
            1
            / rc
            * gamma(1.0 - alpha / 2.0)
            * torch.special.gammainc(1.0 - alpha / 2.0, (r / rc) ** 2.0)
            - gamma(1.5 - alpha / 2.0)
            * torch.special.gammainc(1.5 - alpha / 2.0, (r / rc) ** 2.0)
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

# ### Basic 1D Potentials ###

def isoDiskPotential(x, amp=1, sigma=0.1):
    _H = sigma / torch.sqrt(torch.tensor([8.0 * torch.pi * amp]))[0]
    _sigma2 = sigma**2
    return 2.0 * _sigma2 * torch.log(torch.cosh(0.5 * x / _H))
# def NFWPotential1D(z, R=1.0, amp=1.0, a=1.0):
#     r = torch.sqrt(R**2.0 + z**2.0)
#     if isinstance(r, (float, int)) and r == 0:
#         return -1.0 / a
#     elif isinstance(r, (float, int)):
#         return -torch.xlogy(1.0 / r, 1.0 + r / a)  # stable as r -> infty
#     else:
#         out = -torch.xlogy(1.0 / r, 1.0 + r / a)  # stable as r -> infty
#         out[r == 0] = -1.0 / a
#         return out

def SCFPotential():
    '''`Hernquist & Ostriker (1992) <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`_ Self-Consistent-Field-type potential.'''


def DiskSCFPotential(
        z, 
        R=1.0, 
        amp=1.0, 
        a=1.0
        ):
    dens=lambda R, z: 13.5 * torch.exp(-3.0 * R) * torch.exp(-27.0 * torch.fabs(z))
    r = torch.sqrt(R**2.0 + z**2.0)
    _scf = SCFPotential(amp=1.0, Acos=Acos, Asin=Asin, a=a, ro=None, vo=None)
    out = _scf(R, z, phi=phi, use_physical=False)
    for a, s, H in zip(_Sigma_amp, _Sigma, _Hz):
        out += 4.0 * torch.pi * a * s(r) * H(z)
    return amp * out





# class DiskSCFPotential(Potential):
#     """Class that implements a basis-function-expansion technique for solving the Poisson equation for disk (+halo) systems. We solve the Poisson equation for a given density :math:`\\rho(R,\\phi,z)` by introducing *K* helper function pairs :math:`[\\Sigma_i(R),h_i(z)]`, with :math:`h_i(z) = \\mathrm{d}^2 H(z) / \\mathrm{d} z^2` and search for solutions of the form

#         .. math::

#            \\Phi(R,\\phi,z = \\Phi_{\\mathrm{ME}}(R,\\phi,z) + 4\\pi G\\sum_i \\Sigma_i(r)\\,H_i(z)\\,,

#     where :math:`r` is the spherical radius :math:`r^2 = R^2+z^2`. We can solve for :math:`\\Phi_{\\mathrm{ME}}(R,\\phi,z)` by solving

#         .. math::

#            \\frac{\\Delta \\Phi_{\\mathrm{ME}}(R,\\phi,z)}{4\\pi G} = \\rho(R,\\phi,z) - \\sum_i\\left\\{ \\Sigma_i(r)\\,h_i(z) + \\frac{\\mathrm{d}^2 \\Sigma_i(r)}{\\mathrm{d} r^2}\\,H_i(z)+\\frac{2}{r}\\,\\frac{\\mathrm{d} \\Sigma_i(r)}{\\mathrm{d} r}\\left[H_i(z)+z\\,\\frac{\\mathrm{d}H_i(z)}{\\mathrm{d} z}\\right]\\right\\}\\,.

#     We solve this equation by using the :ref:`SCFPotential <scf_potential>` class and methods (:ref:`scf_compute_coeffs_axi <scf_compute_coeffs_axi>` or :ref:`scf_compute_coeffs <scf_compute_coeffs>` depending on whether :math:`\\rho(R,\\phi,z)` is axisymmetric or not). This technique works very well if the disk portion of the potential can be exactly written as :math:`\\rho_{\\mathrm{disk}} = \\sum_i \\Sigma_i(R)\\,h_i(z)`, because the effective density on the right-hand side of this new Poisson equation is then not 'disky' and can be well represented using spherical harmonics. But the technique is general and can be used to compute the potential of any disk+halo potential; the closer the disk is to :math:`\\rho_{\\mathrm{disk}} \\approx \\sum_i \\Sigma_i(R)\\,h_i(z)`, the better the technique works.

#     This technique was introduced by `Kuijken & Dubinski (1995) <http://adsabs.harvard.edu/abs/1995MNRAS.277.1341K>`__ and was popularized by `Dehnen & Binney (1998) <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__. The current implementation is a slight generalization of the technique in those papers and uses the SCF approach of `Hernquist & Ostriker (1992)
#     <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`__ to solve the Poisson equation for :math:`\\Phi_{\\mathrm{ME}}(R,\\phi,z)` rather than solving it on a grid using spherical harmonics and interpolating the solution (as done in `Dehnen & Binney 1998 <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__).

#     """

#     def __init__(
#         self,
#         amp=1.0,
#         normalize=False,
#         dens=lambda R, z: 13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z)),
#         Sigma={"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
#         hz={"type": "exp", "h": 1.0 / 27.0},
#         Sigma_amp=None,
#         dSigmadR=None,
#         d2SigmadR2=None,
#         Hz=None,
#         dHzdz=None,
#         N=10,
#         L=10,
#         a=1.0,
#         radial_order=None,
#         costheta_order=None,
#         phi_order=None,
#         ro=None,
#         vo=None,
#     ):
#         """
#         Initialize a DiskSCFPotential.

#         Parameters
#         ----------
#         amp : float, optional
#             Amplitude to be applied to the potential (default: 1); cannot have units currently.
#         normalize : bool or float, optional
#             If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
#         dens : callable
#             Function of R,z[,phi optional] that gives the density [in natural units, cannot return a Quantity currently].
#         N : int, optional
#             Number of radial basis functions to use in the SCF expansion.
#         L : int, optional
#             Number of angular basis functions to use in the SCF expansion.
#         a : float or Quantity, optional
#             Scale radius for the SCF expansion.
#         radial_order : int, optional
#             Order of the radial basis functions to use in the SCF expansion.
#         costheta_order : int, optional
#             Order of the angular basis functions to use in the SCF expansion.
#         phi_order : int, optional
#             Order of the azimuthal basis functions to use in the SCF expansion.
#         Sigma : dict or callable
#             Either a dictionary of surface density (example: {'type':'exp','h':1./3.,'amp':1.,'Rhole':0.} for amp x exp(-Rhole/R-R/h) ) or a function of R that gives the surface density.
#         hz : dict or callable
#             Either a dictionary of vertical profile, either 'exp' or 'sech2' (example {'type':'exp','h':1./27.} for exp(-|z|/h)/[2h], sech2 is sech^2(z/[2h])/[4h]) or a function of z that gives the vertical profile.
#         Sigma_amp : float, optional
#             Amplitude to apply to all Sigma functions.
#         dSigmadR : callable, optional
#             Function that gives d Sigma / d R.
#         d2SigmadR2 : callable, optional
#             Function that gives d^2 Sigma / d R^2.
#         Hz : callable, optional
#             Function of z such that d^2 Hz(z) / d z^2 = hz.
#         dHzdz : callable, optional
#             Function of z that gives d Hz(z) / d z.
#         ro : float or Quantity, optional
#             Distance scale for translation into internal units (default from configuration file).
#         vo : float or Quantity, optional
#             Velocity scale for translation into internal units (default from configuration file).

#         Notes
#         -----
#         - Either specify (Sigma,hz) or (Sigma_amp,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz)
#         - Written - Bovy (UofT) - 2016-12-26

#         """
#         Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=None)
#         a = conversion.parse_length(a, ro=self._ro)
#         # Parse and store given functions
#         self.isNonAxi = dens.__code__.co_argcount == 3
#         self._parse_Sigma(Sigma_amp, Sigma, dSigmadR, d2SigmadR2)
#         self._parse_hz(hz, Hz, dHzdz)
#         if self.isNonAxi:
#             self._inputdens = dens
#         else:
#             self._inputdens = lambda R, z, phi: dens(R, z)
#         # Solve Poisson equation for Phi_ME
#         if not self.isNonAxi:
#             dens_func = lambda R, z: phiME_dens(
#                 R,
#                 z,
#                 0.0,
#                 self._inputdens,
#                 self._Sigma,
#                 self._dSigmadR,
#                 self._d2SigmadR2,
#                 self._hz,
#                 self._Hz,
#                 self._dHzdz,
#                 self._Sigma_amp,
#             )
#             Acos, Asin = scf_compute_coeffs_axi(
#                 dens_func,
#                 N,
#                 L,
#                 a=a,
#                 radial_order=radial_order,
#                 costheta_order=costheta_order,
#             )
#         else:
#             dens_func = lambda R, z, phi: phiME_dens(
#                 R,
#                 z,
#                 phi,
#                 self._inputdens,
#                 self._Sigma,
#                 self._dSigmadR,
#                 self._d2SigmadR2,
#                 self._hz,
#                 self._Hz,
#                 self._dHzdz,
#                 self._Sigma_amp,
#             )
#             Acos, Asin = scf_compute_coeffs(
#                 dens_func,
#                 N,
#                 L,
#                 a=a,
#                 radial_order=radial_order,
#                 costheta_order=costheta_order,
#                 phi_order=phi_order,
#             )
#         self._phiME_dens_func = dens_func
#         self._scf = SCFPotential(amp=1.0, Acos=Acos, Asin=Asin, a=a, ro=None, vo=None)
#         if not self._Sigma_dict is None and not self._hz_dict is None:
#             self.hasC = True
#             self.hasC_dens = True
#         if normalize or (
#             isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
#         ):
#             self.normalize(normalize)
#         return None

#     def _parse_Sigma(self, Sigma_amp, Sigma, dSigmadR, d2SigmadR2):
#         if isinstance(Sigma, dict):
#             Sigma = [Sigma]
#         try:
#             nsigma = len(Sigma)
#         except TypeError:
#             Sigma_amp = [Sigma_amp]
#             Sigma = [Sigma]
#             dSigmadR = [dSigmadR]
#             d2SigmadR2 = [d2SigmadR2]
#             nsigma = 1
#         self._nsigma = nsigma
#         self._Sigma_amp = Sigma_amp
#         self._Sigma = Sigma
#         self._dSigmadR = dSigmadR
#         self._d2SigmadR2 = d2SigmadR2
#         if isinstance(Sigma[0], dict):
#             self._Sigma_dict = copy.copy(Sigma)
#             self._parse_Sigma_dict()
#         else:
#             self._Sigma_dict = None
#         return None

#     def _parse_Sigma_dict(self):
#         Sigma_amp, Sigma, dSigmadR, d2SigmadR2 = [], [], [], []
#         for ii in range(self._nsigma):
#             ta, ts, tds, td2s = self._parse_Sigma_dict_indiv(self._Sigma[ii])
#             Sigma_amp.append(ta)
#             Sigma.append(ts)
#             dSigmadR.append(tds)
#             d2SigmadR2.append(td2s)
#         self._Sigma_amp = Sigma_amp
#         self._Sigma = Sigma
#         self._dSigmadR = dSigmadR
#         self._d2SigmadR2 = d2SigmadR2
#         return None

#     def _parse_Sigma_dict_indiv(self, Sigma):
#         stype = Sigma.get("type", "exp")
#         if stype == "exp" and not "Rhole" in Sigma:
#             rd = Sigma.get("h", 1.0 / 3.0)
#             ta = Sigma.get("amp", 1.0)
#             ts = lambda R, trd=rd: numpy.exp(-R / trd)
#             tds = lambda R, trd=rd: -numpy.exp(-R / trd) / trd
#             td2s = lambda R, trd=rd: numpy.exp(-R / trd) / trd**2.0
#         elif stype == "expwhole" or (stype == "exp" and "Rhole" in Sigma):
#             rd = Sigma.get("h", 1.0 / 3.0)
#             rm = Sigma.get("Rhole", 0.5)
#             ta = Sigma.get("amp", 1.0)
#             ts = lambda R, trd=rd, trm=rm: numpy.exp(-trm / R - R / trd)
#             tds = lambda R, trd=rd, trm=rm: (trm / R**2.0 - 1.0 / trd) * numpy.exp(
#                 -trm / R - R / trd
#             )
#             td2s = lambda R, trd=rd, trm=rm: (
#                 (trm / R**2.0 - 1.0 / trd) ** 2.0 - 2.0 * trm / R**3.0
#             ) * numpy.exp(-trm / R - R / trd)
#         return (ta, ts, tds, td2s)

#     def _parse_hz(self, hz, Hz, dHzdz):
#         if isinstance(hz, dict):
#             hz = [hz]
#         try:
#             nhz = len(hz)
#         except TypeError:
#             hz = [hz]
#             Hz = [Hz]
#             dHzdz = [dHzdz]
#             nhz = 1
#         if nhz != self._nsigma and nhz != 1:
#             raise ValueError(
#                 "Number of hz functions needs to be equal to the number of Sigma functions or to 1"
#             )
#         if nhz == 1 and self._nsigma > 1:
#             hz = [hz[0] for ii in range(self._nsigma)]
#             if not isinstance(hz[0], dict):
#                 Hz = [Hz[0] for ii in range(self._nsigma)]
#                 dHzdz = [dHzdz[0] for ii in range(self._nsigma)]
#         self._Hz = Hz
#         self._hz = hz
#         self._dHzdz = dHzdz
#         self._nhz = len(self._hz)
#         if isinstance(hz[0], dict):
#             self._hz_dict = copy.copy(hz)
#             self._parse_hz_dict()
#         else:
#             self._hz_dict = None
#         return None

#     def _parse_hz_dict(self):
#         hz, Hz, dHzdz = [], [], []
#         for ii in range(self._nhz):
#             th, tH, tdH = self._parse_hz_dict_indiv(self._hz[ii])
#             hz.append(th)
#             Hz.append(tH)
#             dHzdz.append(tdH)
#         self._hz = hz
#         self._Hz = Hz
#         self._dHzdz = dHzdz
#         return None

#     def _parse_hz_dict_indiv(self, hz):
#         htype = hz.get("type", "exp")
#         if htype == "exp":
#             zd = hz.get("h", 0.0375)
#             th = lambda z, tzd=zd: 1.0 / 2.0 / tzd * numpy.exp(-numpy.fabs(z) / tzd)
#             tH = (
#                 lambda z, tzd=zd: (
#                     numpy.exp(-numpy.fabs(z) / tzd) - 1.0 + numpy.fabs(z) / tzd
#                 )
#                 * tzd
#                 / 2.0
#             )
#             tdH = (
#                 lambda z, tzd=zd: 0.5
#                 * numpy.sign(z)
#                 * (1.0 - numpy.exp(-numpy.fabs(z) / tzd))
#             )
#         elif htype == "sech2":
#             zd = hz.get("h", 0.0375)
#             # th/tH written so as to avoid overflow in cosh
#             th = (
#                 lambda z, tzd=zd: numpy.exp(
#                     -logsumexp(
#                         numpy.array(
#                             [z / tzd, -z / tzd, numpy.log(2.0) * numpy.ones_like(z)]
#                         ),
#                         axis=0,
#                     )
#                 )
#                 / tzd
#             )
#             tH = lambda z, tzd=zd: tzd * (
#                 logsumexp(numpy.array([z / 2.0 / tzd, -z / 2.0 / tzd]), axis=0)
#                 - numpy.log(2.0)
#             )
#             tdH = lambda z, tzd=zd: numpy.tanh(z / 2.0 / tzd) / 2.0
#         return (th, tH, tdH)

#     def _evaluate(self, R, z, phi=0.0, t=0.0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf(R, z, phi=phi, use_physical=False)
#         for a, s, H in zip(self._Sigma_amp, self._Sigma, self._Hz):
#             out += 4.0 * numpy.pi * a * s(r) * H(z)
#         return out

#     def _Rforce(self, R, z, phi=0, t=0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf.Rforce(R, z, phi=phi, use_physical=False)
#         for a, ds, H in zip(self._Sigma_amp, self._dSigmadR, self._Hz):
#             out -= 4.0 * numpy.pi * a * ds(r) * H(z) * R / r
#         return out

#     def _zforce(self, R, z, phi=0, t=0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf.zforce(R, z, phi=phi, use_physical=False)
#         for a, s, ds, H, dH in zip(
#             self._Sigma_amp, self._Sigma, self._dSigmadR, self._Hz, self._dHzdz
#         ):
#             out -= 4.0 * numpy.pi * a * (ds(r) * H(z) * z / r + s(r) * dH(z))
#         return out

#     def _phitorque(self, R, z, phi=0.0, t=0.0):
#         return self._scf.phitorque(R, z, phi=phi, use_physical=False)

#     def _R2deriv(self, R, z, phi=0.0, t=0.0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf.R2deriv(R, z, phi=phi, use_physical=False)
#         for a, ds, d2s, H in zip(
#             self._Sigma_amp, self._dSigmadR, self._d2SigmadR2, self._Hz
#         ):
#             out += (
#                 4.0
#                 * numpy.pi
#                 * a
#                 * H(z)
#                 / r**2.0
#                 * (d2s(r) * R**2.0 + z**2.0 / r * ds(r))
#             )
#         return out

#     def _z2deriv(self, R, z, phi=0.0, t=0.0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf.z2deriv(R, z, phi=phi, use_physical=False)
#         for a, s, ds, d2s, h, H, dH in zip(
#             self._Sigma_amp,
#             self._Sigma,
#             self._dSigmadR,
#             self._d2SigmadR2,
#             self._hz,
#             self._Hz,
#             self._dHzdz,
#         ):
#             out += (
#                 4.0
#                 * numpy.pi
#                 * a
#                 * (
#                     H(z) / r**2.0 * (d2s(r) * z**2.0 + ds(r) * R**2.0 / r)
#                     + 2.0 * ds(r) * dH(z) * z / r
#                     + s(r) * h(z)
#                 )
#             )
#         return out

#     def _Rzderiv(self, R, z, phi=0.0, t=0.0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf.Rzderiv(R, z, phi=phi, use_physical=False)
#         for a, ds, d2s, H, dH in zip(
#             self._Sigma_amp, self._dSigmadR, self._d2SigmadR2, self._Hz, self._dHzdz
#         ):
#             out += (
#                 4.0
#                 * numpy.pi
#                 * a
#                 * (H(z) * R * z / r**2.0 * (d2s(r) - ds(r) / r) + ds(r) * dH(z) * R / r)
#             )
#         return out

#     def _phi2deriv(self, R, z, phi=0.0, t=0.0):
#         return self._scf.phi2deriv(R, z, phi=phi, use_physical=False)

#     def _dens(self, R, z, phi=0.0, t=0.0):
#         r = numpy.sqrt(R**2.0 + z**2.0)
#         out = self._scf.dens(R, z, phi=phi, use_physical=False)
#         for a, s, ds, d2s, h, H, dH in zip(
#             self._Sigma_amp,
#             self._Sigma,
#             self._dSigmadR,
#             self._d2SigmadR2,
#             self._hz,
#             self._Hz,
#             self._dHzdz,
#         ):
#             out += a * (
#                 s(r) * h(z) + d2s(r) * H(z) + 2.0 / r * ds(r) * (H(z) + z * dH(z))
#             )
#         return out

#     def _mass(self, R, z=None, t=0.0):
#         if not z is None:  # pragma: no cover
#             raise AttributeError  # Hack to fall back to general
#         out = self._scf.mass(R, z=None, use_physical=False)
#         r = R

#         def _integrand(theta):
#             # ~ rforce
#             tz = r * numpy.cos(theta)
#             tR = r * numpy.sin(theta)
#             out = 0.0
#             for a, s, ds, H, dH in zip(
#                 self._Sigma_amp, self._Sigma, self._dSigmadR, self._Hz, self._dHzdz
#             ):
#                 out += a * ds(r) * H(tz) * tR**2
#                 out += a * (ds(r) * H(tz) * tz / r + s(r) * dH(tz)) * tz * r
#             return out * numpy.sin(theta)

#         return out + 2.0 * numpy.pi * integrate.quad(_integrand, 0.0, numpy.pi)[0]


# def phiME_dens(R, z, phi, dens, Sigma, dSigmadR, d2SigmadR2, hz, Hz, dHzdz, Sigma_amp):
#     """The density corresponding to phi_ME"""
#     r = numpy.sqrt(R**2.0 + z**2.0)
#     out = dens(R, z, phi)
#     for a, s, ds, d2s, h, H, dH in zip(
#         Sigma_amp, Sigma, dSigmadR, d2SigmadR2, hz, Hz, dHzdz
#     ):
#         out -= a * (s(r) * h(z) + d2s(r) * H(z) + 2.0 / r * ds(r) * (H(z) + z * dH(z)))
#     return out

# def scf_compute_coeffs_axi(dens, N, L, a=1.0, radial_order=None, costheta_order=None):
#     """
#     Numerically compute the expansion coefficients for a given axi-symmetric density

#     Parameters
#     ----------
#     dens : function
#         A density function that takes parameters R and z
#     N : int
#         Size of the Nth dimension of the expansion coefficients
#     L : int
#         Size of the Lth dimension of the expansion coefficients
#     a : float, optional
#         Parameter used to shift the basis functions (default is 1.0)
#     radial_order : int, optional
#         Number of sample points of the radial integral. If None, radial_order=max(20, N + 3/2L + 1) (default is None)
#     costheta_order : int, optional
#         Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1) (default is None)

#     Returns
#     -------
#     tuple
#         (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

#     Notes
#     -----
#     - 2016-05-20 - Written - Aladdin Seaifan (UofT)
#     """
#     numOfParam = 0
#     try:
#         dens(0, 0)
#         numOfParam = 2
#     except:
#         numOfParam = 3
#     param = [0] * numOfParam
#     dens_kw = _scf_compute_determine_dens_kwargs(dens, param)

#     def integrand(xi, costheta):
#         l = numpy.arange(0, L)[numpy.newaxis, :]
#         r = _xiToR(xi, a)
#         R = r * numpy.sqrt(1 - costheta**2.0)
#         z = r * costheta
#         if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
#             PP = lpmn(0, L - 1, costheta)[0].T[numpy.newaxis, :, 0]
#         else:
#             PP = assoc_legendre_p_all(L - 1, 0, costheta, branch_cut=2)[0].T
#         dV = (1.0 + xi) ** 2.0 * numpy.power(1.0 - xi, -4.0)
#         phi_nl = (
#             a**3 * (1.0 + xi) ** l * (1.0 - xi) ** (l + 1.0) * _C(xi, N, L)[:, :] * PP
#         )
#         param[0] = R
#         param[1] = z
#         return phi_nl * dV * dens(*param, **dens_kw)

#     Acos = numpy.zeros((N, L, 1), float)
#     Asin = None

#     ##This should save us some computation time since we're only taking the double integral once, rather then L times
#     Ksample = [max(N + 3 * L // 2 + 1, 20), max(L + 1, 20)]
#     if radial_order != None:
#         Ksample[0] = radial_order
#     if costheta_order != None:
#         Ksample[1] = costheta_order

#     integrated = _gaussianQuadrature(integrand, [[-1, 1], [-1, 1]], Ksample=Ksample) * (
#         2 * numpy.pi
#     )
#     n = numpy.arange(0, N)[:, numpy.newaxis]
#     l = numpy.arange(0, L)[numpy.newaxis, :]
#     K = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
#     # I = -K*(4*numpy.pi)/(2.**(8*l + 6)) * gamma(n + 4*l + 3)/(gamma(n + 1)*(n + 2*l + 3./2)*gamma(2*l + 3./2)**2)
#     ##Taking the ln of I will allow bigger size coefficients
#     lnI = (
#         -(8 * l + 6) * numpy.log(2)
#         + gammaln(n + 4 * l + 3)
#         - gammaln(n + 1)
#         - numpy.log(n + 2 * l + 3.0 / 2)
#         - 2 * gammaln(2 * l + 3.0 / 2)
#     )
#     I = -K * (4 * numpy.pi) * numpy.e ** (lnI)
#     constants = -(2.0 ** (-2 * l)) * (2 * l + 1.0) ** 0.5
#     Acos[:, :, 0] = 2 * I**-1 * integrated * constants
#     return Acos, Asin
