"""Action-angle maps in 3D with exactly symplectic shear flows.

Map: (theta, J) in T^3 x R^3_+  ->  toy 3D harmonic oscillator (one ellipse per axis)  ->  symplectic flow on R^6
->  (x, v). Every toy 3-torus has actions J and the flow is exact-symplectic, so each image torus has actions J
too; training only asks the image tori to be constant-energy. In 3D that is still enough: a Lagrangian torus on
which H is constant is invariant under the flow of H (dH|_L = 0 and L Lagrangian imply X_H is tangent to L).

Flow layers (SympNets, Jin et al. 2020). In n dimensions q <- q + f(p) is symplectic only if f is a gradient,
so the 1D repo conditioners (per-component outputs, per-component gates) do not carry over. Here:
  GradShear  q <- q + g * W (a * tanh(W^T p + b))   = q + grad_p [g sum_k a_k log cosh(w_k.p + b_k)]
             (or the same with p <- p + f(q)), scalar gate g
  LinShear   q <- q + S p  with S symmetric         (or p <- p + S q); products of these give every linear
             symplectic map, so linear parts are represented exactly.

Targets, all with exact ground truth for H(J):
  AnisoHO        sum_i (v_i^2 + w_i^2 x_i^2)/2              the exact map from an isotropic toy is linear
  SeparableQuartic sum_i (v_i^2/2 + w_i^2 x_i^2/2 + l_i x_i^4/4)  E(J) per axis by Gauss-Legendre quadrature
"""
from __future__ import annotations

import math
import time

import torch

F64 = torch.float64


# ============================================================================== targets
class AnisoHO:
    name = "aniso_ho"
    E0 = 0.0

    def __init__(self, omegas=(1.0, 1.4, 1.9)):
        self.w = torch.tensor(omegas, dtype=F64)

    def H(self, xv):
        x, v = xv[..., :3], xv[..., 3:]
        return 0.5 * (v * v).sum(-1) + 0.5 * ((self.w * x) ** 2).sum(-1)

    def E_of_J(self, J):
        return (self.w * J).sum(-1)

    def freqs(self, J):
        return self.w.expand_as(J)


_GL = {}


def _gl(n):
    """Gauss-Legendre nodes and weights on [-1, 1] (Golub-Welsch, in torch: calling numpy's leggauss after torch
    is loaded crashes on this machine, two OpenMP runtimes)."""
    if n not in _GL:
        k = torch.arange(1, n, dtype=F64)
        beta = k / torch.sqrt(4 * k * k - 1)
        x, V = torch.linalg.eigh(torch.diag(beta, 1) + torch.diag(beta, -1))
        _GL[n] = (x, 2 * V[0] ** 2)
    return _GL[n]


class SeparableQuartic:
    """phi_i(x) = w_i^2 x^2 / 2 + l_i x^4 / 4 on each axis. Actions by quadrature with x = x_max sin(u), which
    makes the integrands smooth (the square-root endpoint behaviour cancels against cos u)."""
    name = "sep_quartic"
    E0 = 0.0

    def __init__(self, omegas=(1.0, 1.4, 1.9), lams=(0.5, 0.8, 1.2), n_gl=64):
        self.w = torch.tensor(omegas, dtype=F64)
        self.l = torch.tensor(lams, dtype=F64)
        self.n_gl = n_gl

    def phi(self, x):
        return 0.5 * (self.w * x) ** 2 + 0.25 * self.l * x ** 4

    def H(self, xv):
        x, v = xv[..., :3], xv[..., 3:]
        return 0.5 * (v * v).sum(-1) + self.phi(x).sum(-1)

    def _xmax(self, E):  # root of l x^4/4 + w^2 x^2/2 = E, per axis, in the cancellation-free form
        a = 0.5 * self.w ** 2
        return torch.sqrt(2 * E / (a + torch.sqrt(a * a + self.l * E)))

    def J_and_dJdE(self, E):
        """Per-axis action J(E) = (2/pi) int_0^xm sqrt(2(E - phi)) dx and dJ/dE = 1/Omega, E (..., 3)."""
        u, wq = _gl(self.n_gl)
        u = (u + 1) * (math.pi / 4); wq = wq * (math.pi / 4)               # [-1, 1] -> [0, pi/2]
        xm = self._xmax(E)[..., None]
        x = xm * torch.sin(u)
        # E - phi(x) = (xm^2 - x^2) G with G = w^2/2 + l (xm^2 + x^2)/4, so p = sqrt(2(E - phi)) = xm cos(u) sqrt(2G)
        # and dx = xm cos(u) du: J = (2/pi) int p dx, dJ/dE = (2/pi) int dx / p = (2/pi) int du / sqrt(2G)
        G = 0.5 * self.w[:, None] ** 2 + 0.25 * self.l[:, None] * (xm ** 2 + x ** 2)
        c = torch.cos(u)
        J = (2 / math.pi) * (wq * torch.sqrt(2 * G) * (xm * c) ** 2).sum(-1)
        dJdE = (2 / math.pi) * (wq / torch.sqrt(2 * G)).sum(-1)
        return J, dJdE

    def E_of_J_axes(self, J, iters=60):
        """Per-axis energies E_i(J_i) by Newton on J(E) (dJ/dE = 1/Omega > 0), J (..., 3)."""
        E = self.w * J  # harmonic start
        for _ in range(iters):
            Jc, dJdE = self.J_and_dJdE(E)
            step = (Jc - J) / dJdE
            E = torch.clamp(E - step, min=0.5 * E)
            if float((step / E).abs().max()) < 1e-16:
                break
        return E

    def E_of_J(self, J):
        return self.E_of_J_axes(J).sum(-1)

    def freqs(self, J):
        _, dJdE = self.J_and_dJdE(self.E_of_J_axes(J))
        return 1 / dJdE


class Isochrone:
    """Phi = -GM / (b + sqrt(b^2 + r^2)). Actions (J_r, L, L_z): H(J) = -(GM)^2 / (2 [J_r + (L + sqrt(L^2 + 4 GM b))/2]^2),
    exact. Near the centre it is a harmonic oscillator with omega^2 = GM / (4 b^3)."""
    name = "isochrone"

    def __init__(self, GM=1.0, b=1.0):
        self.GM, self.b = float(GM), float(b)
        self.omega0 = math.sqrt(self.GM / (4 * self.b ** 3))
        self.E0 = -self.GM / (2 * self.b)  # Phi(0): relative errors are measured above it

    def H(self, xv):
        x, v = xv[..., :3], xv[..., 3:]
        r2 = (x * x).sum(-1)
        return 0.5 * (v * v).sum(-1) - self.GM / (self.b + torch.sqrt(self.b ** 2 + r2))

    def _D(self, J):
        L = J[..., 1]
        return J[..., 0] + 0.5 * (L + torch.sqrt(L * L + 4 * self.GM * self.b))

    def E_of_J(self, J):
        return -0.5 * self.GM ** 2 / self._D(J) ** 2

    def freqs(self, J):
        L = J[..., 1]; Or = self.GM ** 2 / self._D(J) ** 3
        OL = 0.5 * (1 + L / torch.sqrt(L * L + 4 * self.GM * self.b)) * Or
        return torch.stack([Or, OL, torch.zeros_like(Or)], -1)


def make_target(spec):
    kind = spec["kind"]
    if kind == "aniso_ho":
        return AnisoHO(spec.get("omegas", (1.0, 1.4, 1.9)))
    if kind == "sep_quartic":
        return SeparableQuartic(spec.get("omegas", (1.0, 1.4, 1.9)), spec.get("lams", (0.5, 0.8, 1.2)))
    if kind == "isochrone":
        return Isochrone(spec.get("GM", 1.0), spec.get("b", 1.0))
    raise ValueError(kind)


# ================================================================================ toy
class Toy3D(torch.nn.Module):
    """Cartesian 3D harmonic oscillator: (theta, J) -> x_i = sqrt(2 J_i / w_i) cos theta_i,
    v_i = -sqrt(2 J_i w_i) sin theta_i. Canonical, with actions J_i; toy energy sum_i w_i J_i."""

    def __init__(self, omegas):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(omegas, dtype=F64).clone())

    def forward(self, aa):
        th, J = aa[..., :3], aa[..., 3:]
        return torch.cat([torch.sqrt(2 * J / self.w) * torch.cos(th), -torch.sqrt(2 * J * self.w) * torch.sin(th)], -1)

    def energy(self, J):
        return (self.w * J).sum(-1)


class SphericalToyHO(torch.nn.Module):
    """Isotropic 3D oscillator (frequency w) in spherical actions (J_r, L, L_z) with angles (t_r, t_L, t_z).
    The orbit is a centred ellipse with semi-axes a, b: a + b = 2 sqrt((J_r + L)/w), a - b = 2 sqrt(J_r/w), so
    E = w (2 J_r + L) and L = w a b. Along it the phase tau = t_r/2 + pi/2 (t_r = 0 at pericentre, radial
    frequency 2w); the major axis sits at g = t_L - tau from the ascending node (so t_L advances at w); the node
    is at longitude t_z, and cos i = L_z / L. The map is checked numerically to be exactly symplectic."""

    def __init__(self, omega):
        super().__init__()
        self.register_buffer("w", torch.tensor(float(omega), dtype=F64))

    def forward(self, aa):
        tr, tL, tz = aa[..., 0], aa[..., 1], aa[..., 2]
        Jr, L, Lz = aa[..., 3], aa[..., 4], aa[..., 5]
        w = self.w
        apb, amb = 2 * torch.sqrt((Jr + L) / w), 2 * torch.sqrt(Jr / w)
        a, b = 0.5 * (apb + amb), 0.5 * (apb - amb)
        tau = 0.5 * tr + 0.5 * math.pi; g = tL - tau
        # in-plane coordinates along the node line (X) and perpendicular to it (Y)
        X0, Y0 = a * torch.cos(tau), b * torch.sin(tau)                  # in the major-axis frame
        U0, V0 = -w * a * torch.sin(tau), w * b * torch.cos(tau)
        cg, sg = torch.cos(g), torch.sin(g)
        X, Y = cg * X0 - sg * Y0, sg * X0 + cg * Y0
        U, V = cg * U0 - sg * V0, sg * U0 + cg * V0
        ci = Lz / L; si = torch.sqrt(torch.clamp(1 - ci * ci, min=0.0))
        cO, sO = torch.cos(tz), torch.sin(tz)
        n = torch.stack([cO, sO, torch.zeros_like(cO)], -1)              # ascending node
        m = torch.stack([-ci * sO, ci * cO, si], -1)                      # in the plane, 90 degrees ahead of n
        x = X[..., None] * n + Y[..., None] * m
        v = U[..., None] * n + V[..., None] * m
        return torch.cat([x, v], -1)

    def energy(self, J):
        return self.w * (2 * J[..., 0] + J[..., 1])


def sobol_spherical_actions(n, Jr_lo, Jr_hi, L_lo, L_hi, cosi_max=0.9, seed=1):
    """n tori scattered quasi-randomly: log-uniform in J_r and L, uniform in cos i; actions (J_r, L, L cos i)."""
    u = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(n).to(F64)
    Jr = Jr_lo * (Jr_hi / Jr_lo) ** u[:, 0]; L = L_lo * (L_hi / L_lo) ** u[:, 1]; c = cosi_max * (2 * u[:, 2] - 1)
    return torch.stack([Jr, L, L * c], -1)


def spherical_action_grid(n_per_axis, Jr_lo, Jr_hi, L_lo, L_hi, cosi_max=0.9, midpoints=False):
    """Tori labelled by (J_r, L, cos i): geometric in J_r and L, uniform in cos i on [-cosi_max, cosi_max];
    returns actions (J_r, L, L_z = L cos i). midpoints=True gives the held-out tori between them."""
    k = torch.arange(n_per_axis - 1 if midpoints else n_per_axis, dtype=F64)
    u = (k + 0.5 * midpoints) / (n_per_axis - 1)
    Jr = Jr_lo * (Jr_hi / Jr_lo) ** u; L = L_lo * (L_hi / L_lo) ** u; c = cosi_max * (2 * u - 1)
    g = torch.meshgrid(Jr, L, c, indexing="ij")
    Jr, L, c = (t.reshape(-1) for t in g)
    return torch.stack([Jr, L, L * c], -1)


# ============================================================================= layers
class GradShear(torch.nn.Module):
    """q <- q + g W (a * tanh(W^T p + b)) (update_q) or p <- p + g W (a * tanh(W^T q + b)): the gradient of a
    scalar, so exactly symplectic in any dimension. Scalar gate g ~ N(0, gate_std) (an all-zero start makes the
    identity an exact critical point); random biases (zero biases trap odd activations on symmetric targets)."""

    def __init__(self, n=3, width=32, update_q=True, gate_std=1e-3, gen=None):
        super().__init__()
        self.update_q = update_q
        self.W = torch.nn.Parameter(torch.randn(n, width, generator=gen, dtype=F64) / math.sqrt(n))
        self.b = torch.nn.Parameter((2 * torch.rand(width, generator=gen, dtype=F64) - 1))
        self.a = torch.nn.Parameter(torch.randn(width, generator=gen, dtype=F64) / math.sqrt(width))
        self.g = torch.nn.Parameter(gate_std * torch.randn((), generator=gen, dtype=F64))

    def shift(self, u):
        return self.g * ((self.a * torch.tanh(u @ self.W + self.b)) @ self.W.T)

    def forward(self, z):
        q, p = z[..., :3], z[..., 3:]
        return torch.cat([q + self.shift(p), p], -1) if self.update_q else torch.cat([q, p + self.shift(q)], -1)

    def inverse(self, z):
        q, p = z[..., :3], z[..., 3:]
        return torch.cat([q - self.shift(p), p], -1) if self.update_q else torch.cat([q, p - self.shift(q)], -1)


class LinShear(torch.nn.Module):
    """q <- q + S p or p <- p + S q, S = (M + M^T)/2 symmetric (started at zero)."""

    def __init__(self, n=3, update_q=True):
        super().__init__()
        self.update_q = update_q
        self.M = torch.nn.Parameter(torch.zeros(n, n, dtype=F64))

    def shift(self, u):
        return u @ (0.5 * (self.M + self.M.T))

    def forward(self, z):
        q, p = z[..., :3], z[..., 3:]
        return torch.cat([q + self.shift(p), p], -1) if self.update_q else torch.cat([q, p + self.shift(q)], -1)

    def inverse(self, z):
        q, p = z[..., :3], z[..., 3:]
        return torch.cat([q - self.shift(p), p], -1) if self.update_q else torch.cat([q, p - self.shift(q)], -1)


class Flow6(torch.nn.Module):
    def __init__(self, n_lin=2, n_blocks=8, width=32, gate_std=1e-3, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(1000 + seed)
        layers = []
        for _ in range(n_lin):  # linear pairs first: they carry the linear part of the map
            layers += [LinShear(3, True), LinShear(3, False)]
        for _ in range(n_blocks):
            layers += [GradShear(3, width, True, gate_std, gen), GradShear(3, width, False, gate_std, gen)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, z):
        for L in self.layers:
            z = L(z)
        return z

    def inverse(self, z):
        for L in reversed(self.layers):
            z = L.inverse(z)
        return z


class AAMap3D(torch.nn.Module):
    def __init__(self, toy, flow):
        super().__init__()
        self.toy, self.flow = toy, flow

    def forward(self, aa):
        return self.flow(self.toy(aa))


def build_model(mcfg, toy, seed=0):
    """toy: a list of three frequencies (Cartesian oscillator) or {"kind": "spherical_ho", "omega": w}."""
    torch.manual_seed(seed)
    flow = Flow6(mcfg.get("n_lin", 2), mcfg.get("n_blocks", 8), mcfg.get("width", 32), mcfg.get("gate_std", 1e-3), seed)
    if isinstance(toy, dict) and toy.get("kind") == "spherical_ho":
        return AAMap3D(SphericalToyHO(toy["omega"]), flow)
    return AAMap3D(Toy3D(toy["omegas"] if isinstance(toy, dict) else toy), flow)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# =============================================================================== data
def action_grid(n_per_axis, J_lo, J_hi, midpoints=False):
    """Product grid of tori, geometric in each action from J_lo to J_hi (n points per axis, ends included);
    midpoints=True gives the n-1 points per axis halfway (in log J) between them: the held-out tori."""
    k = torch.arange(n_per_axis - 1 if midpoints else n_per_axis, dtype=F64)
    u = (k + 0.5 * midpoints) / (n_per_axis - 1)
    J_lo, J_hi = torch.as_tensor(J_lo, dtype=F64), torch.as_tensor(J_hi, dtype=F64)
    J1 = J_lo[:, None] * (J_hi / J_lo)[:, None] ** u
    g = torch.meshgrid(J1[0], J1[1], J1[2], indexing="ij")
    return torch.stack([t.reshape(-1) for t in g], -1)  # (n^3, 3)


def sobol_actions(n, J_lo, J_hi, seed=1):
    """n tori scattered quasi-randomly (scrambled Sobol) and log-uniformly through the action box."""
    u = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(n).to(F64)
    J_lo, J_hi = torch.as_tensor(J_lo, dtype=F64), torch.as_tensor(J_hi, dtype=F64)
    return J_lo * (J_hi / J_lo) ** u


def angle_samples(n, seed):
    """n points on T^3: a scrambled Sobol sequence (fixed per seed)."""
    s = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    return (2 * math.pi * s.draw(n)).to(F64)


def aa_points(J, th):
    """(nO, 3) actions x (nS, 3) angles -> (nO, nS, 6) [theta, J]."""
    nO, nS = J.shape[0], th.shape[0]
    return torch.cat([th[None].expand(nO, nS, 3), J[:, None].expand(nO, nS, 3)], -1)


# =============================================================================== loss
def residuals(model, aa, target, escale=None):
    """(H - <H>) / scale per point; scale = fixed toy energy per torus (training), or <H> - E0, the torus energy
    above the potential minimum (the relative metric; E0 = 0 for the oscillator-like targets, as in 1D)."""
    nO, nS = aa.shape[:2]
    H = target.H(model(aa.reshape(-1, 6))).view(nO, nS)
    m = H.mean(1, keepdim=True)
    return (H - m) / (escale if escale is not None else (m - target.E0).abs())


class Evaluator:
    def __init__(self, target, J_lo=None, J_hi=None, n_train_per_axis=None, n_angles=1024, seed=12345, J=None):
        self.target = target
        # held-out tori: between the training ones (or given)
        self.J = J if J is not None else action_grid(n_train_per_axis, J_lo, J_hi, midpoints=True)
        self.th = angle_samples(n_angles, seed)                     # held-out angles: an independent sequence
        self.aa = aa_points(self.J, self.th)
        self.E_true = target.E_of_J(self.J).reshape(-1, 1)
        self.E_rel = self.E_true - target.E0  # energy above the potential minimum

    @torch.no_grad()
    def quick(self, model):
        e = residuals(model, self.aa, self.target).abs()
        H = self.target.H(model(self.aa.reshape(-1, 6))).view(*self.aa.shape[:2])
        return dict(eval_mean=float(e.mean()), eval_max=float(e.max()),
                    eval_mean_trueE=float(((H - H.mean(1, keepdim=True)) / self.E_rel).abs().mean()),
                    energy_offset_max=float(((H.mean(1, keepdim=True) - self.E_true) / self.E_rel).abs().max()),
                    origin_image_norm=float(model.flow(torch.zeros(1, 6, dtype=F64)).norm()))

    def action_freq(self, model, n=40):
        """Torus-mean energy against the exact H(J) (the image torus has actions J exactly) and model
        frequencies d<H>/dJ_i against the exact ones."""
        J = self.J[:: max(1, len(self.J) // n)].clone().requires_grad_(True)
        Hm = self.target.H(model(aa_points(J, self.th).reshape(-1, 6))).view(J.shape[0], -1).mean(-1)
        dHdJ = torch.autograd.grad(Hm.sum(), J)[0]
        Et = self.target.E_of_J(J.detach()); Om = self.target.freqs(J.detach())
        scale = Om.abs().max(-1, keepdim=True).values  # some frequencies are zero (L_z in a spherical potential)
        return dict(rel_E_err=((Hm.detach() - Et) / (Et - self.target.E0)).abs().tolist(),
                    rel_freq_err=((dHdJ - Om).abs() / scale).max(-1).values.tolist())


# =========================================================================== trainers
def train_adam(model, aa, target, steps, lr, escale, record=None, every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    for k in range(steps):
        loss = residuals(model, aa, target, escale).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if record and (k % every == 0 or k == steps - 1):
            record(k, float(loss))


def train_lm(model, aa, target, escale, max_iters, lam=1e-3, record=None, every=10, chunk=8, geodesic=True,
             geo_h=0.1, geo_alpha=0.75, wall_cap=None):
    """Levenberg-Marquardt on r / sqrt(N) with Marquardt scaling, optional geodesic acceleration (Transtrum &
    Sethna 2012) with a fallback to the plain step. Per-point Jacobians by vmap(jacrev) w.r.t. the parameter
    dict, chained through H and the per-torus mean; J^T J accumulated over chunks of tori."""
    from torch.func import functional_call, jacrev, vmap
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [p.numel() for _, p in model.named_parameters()]
    nO, nS = aa.shape[:2]; N = nO * nS; sN = math.sqrt(N)

    def unflat(v):
        out, i = {}, 0
        for n, s_, k in zip(names, shapes, sizes):
            out[n] = v[i:i + k].view(s_); i += k
        return out

    def resid(v):
        with torch.no_grad():
            H = target.H(functional_call(model, unflat(v), (aa.reshape(-1, 6),))).view(nO, nS)
            return ((H - H.mean(1, keepdim=True)) / escale / sN).reshape(-1)

    def jtv(v, u):
        vv = v.detach().clone().requires_grad_(True)
        H = target.H(functional_call(model, unflat(vv), (aa.reshape(-1, 6),))).view(nO, nS)
        r = ((H - H.mean(1, keepdim=True)) / escale / sN).reshape(-1)
        return torch.autograd.grad(r, vv, grad_outputs=u)[0]

    f_point = lambda p, x: functional_call(model, p, (x[None],))[0]

    def normal_eqs(v):
        P = v.numel(); A = torch.zeros(P, P, dtype=F64); g = torch.zeros(P, dtype=F64); rs = []
        pd = unflat(v)
        for o0 in range(0, nO, chunk):
            x = aa[o0:o0 + chunk].reshape(-1, 6); k = x.shape[0] // nS
            Jd = vmap(jacrev(f_point), in_dims=(None, 0))(pd, x)
            Jy = torch.cat([Jd[n].reshape(x.shape[0], 6, -1) for n in names], -1); del Jd
            with torch.no_grad():
                y = functional_call(model, pd, (x,))
            y = y.detach().requires_grad_(True); Hf = target.H(y)
            dHdy = torch.autograd.grad(Hf.sum(), y)[0]
            H = Hf.detach().view(k, nS); es = escale[o0:o0 + k]
            r = ((H - H.mean(1, keepdim=True)) / es / sN).reshape(-1)
            dH = (dHdy[:, :, None] * Jy).sum(1).view(k, nS, -1); del Jy
            Jm = ((dH - dH.mean(1, keepdim=True)) / es[..., None] / sN).reshape(k * nS, -1)
            A += Jm.T @ Jm; g += Jm.T @ r; rs.append(r)
        return torch.cat(rs), g, A

    v = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    t0 = time.perf_counter(); why = "max_iters"; k = -1
    for k in range(max_iters):
        r, g, A = normal_eqs(v); f = 0.5 * float(r @ r)
        dg = torch.diagonal(A).clone(); accepted = False
        for _ in range(30):
            L, info = torch.linalg.cholesky_ex(A + lam * torch.diag(dg + 1e-12 * dg.max()))
            if int(info) != 0:
                lam *= 10; continue
            d = -torch.cholesky_solve(g[:, None], L)[:, 0]
            trials = [d]
            if geodesic:
                Jt_rdd = (2 / geo_h) * ((jtv(v, resid(v + geo_h * d)) - g) / geo_h - A @ d)
                a = -torch.cholesky_solve(Jt_rdd[:, None], L)[:, 0]
                if 2 * float((dg * a * a).sum().sqrt()) <= geo_alpha * float((dg * d * d).sum().sqrt()):
                    trials = [d + 0.5 * a, d]
            for dd in trials:
                fn = 0.5 * float(resid(v + dd).pow(2).sum())
                if math.isfinite(fn) and fn < f:
                    v, f, accepted = v + dd, fn, True
                    break
            if accepted:
                lam = max(lam / 3, 1e-15); break
            lam *= 4
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                n = p.numel(); p.copy_(v[i:i + n].view_as(p)); i += n
        stop = not accepted or bool(wall_cap and time.perf_counter() - t0 > wall_cap)
        if record and (k % every == 0 or stop):
            record(k, 2 * f, lam)
        if stop:
            why = "no-descent" if not accepted else "walltime"
            break
    return k + 1, why, lam
