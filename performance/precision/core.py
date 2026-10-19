"""Core library for orbitflows precision runs (portable: laptop, geir, Trillium).

Model: (theta, J) -> toy SHO ellipse with frequency omega(J) -> symplectic shear flow -> (q, p).
  * toy 'fixed'  : omega(J) = omega0 (nu0 = sqrt(Phi''(0)) for MW, exact as J -> 0).
  * toy 'learned': ln omega(J) = ln omega0 + Chebyshev series in ln J. The raw ellipse is then not
    area-preserving (det = 1 - kappa cos 2theta, kappa = dln omega/dln J); the canonical angle is
    theta_c = theta - (kappa/2) sin 2theta, so (theta_c, J) -> (q, p) is exactly symplectic for |kappa| < 1.
    Training samples theta uniformly (the energy loss is per-torus, so the angle density is immaterial);
    theta_c only matters when angles are reported.
  * conditioners: SimpleNNConditioner (repo) or its odd part f(x) = [g(x) - g(-x)]/2, which fixes the
    phase-space origin exactly (correct for reflection-symmetric potentials).
  * init: random biases (zero biases trap odd activations on symmetric data) and gates ~ N(0, gate_std)
    (all-zero gates make the identity an exact critical point of the loss).
Loss: mean over tori of mean over angles of r^2, r = (H - <H>)/E_ref(J), least squares -> Levenberg-Marquardt.
  E_ref = omega0*J (the fixed toy energy; parameter-independent). The older normalisation by the torus's own
  mean energy <H> has a degenerate direction: a constant shear offset translates a torus by c, which inflates
  <H> ~ c^2/2 while the variation grows only ~ c*p, so the relative loss -> 0 as 1/c^2 without fitting
  anything (an LM step can jump the barrier at c ~ 1). loss_norm="mean" keeps the old behaviour.
H is always evaluated in float64 on the CPU; the model can run in float32 on a GPU ('mixed') or float64.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import json
import math
import time

import torch

from orbitflows.models.flow import Flow, SymplecticCouplingLayer, SimpleNNConditioner
from orbitflows.dynamics.potentials import MWPotential2014

HOST = torch.device("cpu")
F64 = torch.float64
ACTS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU, "softplus": torch.nn.Softplus}
RO_KPC = 8.0  # galpy natural length unit


# ============================================================================ targets
class MWVertical:
    """Vertical MWPotential2014 at R = 1 (natural units), Phi(0) = 0 computed in float64."""
    name = "mw"

    def __init__(self):
        self.R = torch.tensor(1.0, dtype=F64)
        self.amp = torch.tensor(1.0, dtype=F64)
        self.phi0 = MWPotential2014(torch.zeros((), dtype=F64), self.R, amp=self.amp).detach()
        z = torch.zeros((), dtype=F64, requires_grad=True)
        d1 = torch.autograd.grad(self.phi(z), z, create_graph=True)[0]
        self.nu0 = float(torch.autograd.grad(d1, z)[0]) ** 0.5

    def phi(self, q):
        return MWPotential2014(q, self.R, amp=self.amp) - self.phi0

    def H(self, ps):
        return 0.5 * ps[..., 1] ** 2 + self.phi(ps[..., 0])


class SHO:
    name = "sho"

    def __init__(self, omega=1.0):
        self.omega = omega
        self.nu0 = omega

    def phi(self, q):
        return 0.5 * self.omega ** 2 * q * q

    def H(self, ps):
        return 0.5 * ps[..., 1] ** 2 + self.phi(ps[..., 0])


class Teacher:
    """Realisable target H_T(x) = H_toy(T^-1(x)) for a fixed shear flow T (zero approximation error for a
    student containing T's architecture). No separable potential: energy metrics only.

    T is either a random flow (n_layers, width, gate, seed), or a trained model's flow loaded from
    `checkpoint` (a file written by save_teacher, path relative to this directory). The second gives a
    realisable target with the complexity of a real problem: e.g. the best MW model, whose tori are within
    ~1e-5 of the MW potential's, so a student of the same architecture and toy can fit it exactly."""
    name = "teacher"

    def __init__(self, omega=1.5, n_layers=4, width=16, gate=0.25, seed=123, checkpoint=None):
        if checkpoint is not None:
            ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint), weights_only=False)
            mc = ck["model"]
            cond = OddNNConditioner if mc.get("odd", False) else SimpleNNConditioner
            self.T = Flow(2, mc["n_layers"], SymplecticCouplingLayer, cond,
                          dict(num_layers=mc.get("depth", 2), projection_dims=mc["width"],
                               activation=ACTS[mc.get("act", "tanh")], include_bias=True)).to(F64)
            self.T.load_state_dict(ck["flow_state"])
            for p in self.T.parameters():
                p.requires_grad_(False)
            self.omega = self.nu0 = float(ck["omega"])
            return
        torch.manual_seed(seed)
        self.T = Flow(2, n_layers, SymplecticCouplingLayer, SimpleNNConditioner,
                      dict(num_layers=2, projection_dims=width, activation=torch.nn.Tanh, include_bias=True)).to(F64)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for m in self.T.modules():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    m.bias.copy_((2 * torch.rand(m.bias.shape, generator=g, dtype=F64) - 1) * 0.3)
            for layer in self.T.layers:
                c = layer.conditioner
                c.gate.copy_(gate * torch.sign(2 * torch.rand(c.gate.shape, generator=g, dtype=F64) - 1))
        for p in self.T.parameters():
            p.requires_grad_(False)
        self.omega = omega
        self.nu0 = omega

    def H(self, ps):
        z = self.T.inverse(ps)
        return 0.5 * z[..., 1] ** 2 + 0.5 * (self.omega * z[..., 0]) ** 2


def save_teacher(job_dir, out_path):
    """Freeze a finished fixed-toy job (config.json + state.pt) as a teacher file for Teacher(checkpoint=...)."""
    cfg = json.load(open(os.path.join(job_dir, "config.json")))
    assert cfg["toy"].get("mode", "fixed") == "fixed", "a teacher needs a fixed toy frequency"
    st = torch.load(os.path.join(job_dir, "state.pt"), weights_only=False)["state"]
    flow_state = {k[len("flow."):]: v.to(F64) for k, v in st.items() if k.startswith("flow.")}
    omega = float(st["toy.ln_w0"].double().exp())  # exactly the frequency the teacher was trained with
    torch.save(dict(model=cfg["model"], omega=omega, flow_state=flow_state, source=cfg["name"]), out_path)
    return omega


def make_target(cfg):
    t = cfg["target"]
    if t == "mw":
        return MWVertical()
    if t == "sho":
        return SHO(cfg.get("omega_target", 1.0))
    if t == "teacher":
        return Teacher(**cfg.get("teacher", {}))
    raise ValueError(t)


# ======================================================= ground truth (1D, even potential)
_GL = {}


def _gl(n):
    """Gauss-Legendre on [0, pi/2] via Golub-Welsch (numpy's leggauss can clash with torch's OpenMP)."""
    if n not in _GL:
        k = torch.arange(1, n, dtype=F64)
        beta = k / torch.sqrt(4 * k * k - 1)
        x, V = torch.linalg.eigh(torch.diag(beta, 1) + torch.diag(beta, -1))
        _GL[n] = ((x + 1) * math.pi / 4, 2 * V[0, :] ** 2 * math.pi / 4)
    return _GL[n]


@torch.no_grad()
def zmax_of_E(E, phi, zhi=5.0, iters=200):
    lo, hi = torch.zeros_like(E), torch.full_like(E, zhi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = phi(mid) - E
        lo = torch.where(f < 0, mid, lo); hi = torch.where(f >= 0, mid, hi)
    return 0.5 * (lo + hi)


@torch.no_grad()
def true_action(E, phi, n_quad=128):
    """J(E) = (2/pi) int_0^zmax sqrt(2(E - phi)) dz, z = zmax sin u, Gauss-Legendre."""
    zm = zmax_of_E(E, phi)
    u, w = _gl(n_quad)
    z = zm[:, None] * torch.sin(u)[None, :]
    v = torch.sqrt(2 * (E[:, None] - phi(z)).clamp_min(0.0))
    return (2 / math.pi) * (v * zm[:, None] * torch.cos(u)[None, :] * w).sum(-1)


@torch.no_grad()
def true_action_freq(E, phi, rel_h=1e-3):
    """Omega = 1/(dJ/dE) by a 5-point stencil on the well-conditioned action integral."""
    h = rel_h * E
    Jk = [true_action(E + k * h, phi) for k in (-2, -1, 0, 1, 2)]
    return Jk[2], 12 * h / (Jk[0] - 8 * Jk[1] + 8 * Jk[3] - Jk[4])


@torch.no_grad()
def E_of_J(J, phi, iters=90):
    lo, hi = torch.full_like(J, -20.0), torch.full_like(J, 2.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = true_action(mid.exp(), phi) - J
        lo = torch.where(f < 0, mid, lo); hi = torch.where(f >= 0, mid, hi)
    return (0.5 * (lo + hi)).exp()


def J_of_zmax(z, phi):
    return true_action(phi(torch.as_tensor(z, dtype=F64).reshape(-1)), phi)


def zmax_kpc_of_J(J, phi):
    return zmax_of_E(E_of_J(J, phi), phi) * RO_KPC


# ================================================================== collocation grids
def j_grid(n, J_lo, J_hi, omega_ref, spacing="asinh", a0=0.03, offset=0.0, dtype=F64):
    """Tori spaced in the reference amplitude a = sqrt(2J/omega_ref): 'asinh' (even spacing for a << a0,
    geometric for a >> a0), 'linear' or 'log'. offset=0 includes both ends; 0.5 gives cell midpoints."""
    a_lo, a_hi = math.sqrt(2 * J_lo / omega_ref), math.sqrt(2 * J_hi / omega_ref)
    f, finv = {"asinh": (lambda a: math.asinh(a / a0), lambda u: a0 * torch.sinh(u)),
               "linear": (lambda a: a, lambda u: u),
               "log": (math.log, torch.exp)}[spacing]
    u0, u1 = f(a_lo), f(a_hi)
    if offset == 0.0:
        u = torch.linspace(u0, u1, n, dtype=dtype)
    else:
        u = u0 + (u1 - u0) / n * (torch.arange(n, dtype=dtype) + offset)
    return omega_ref * finv(u) ** 2 / 2


def train_grid(d, J_lo, J_hi, omega_ref):
    """Training tori from a data config: d["n_orbits"] tori spaced by d["spacing"] (offset 0, both ends included).
    Optional d["inner"] = {"n": n_in, "split_pc": z}: the tori below toy amplitude a = z (~ z_max near the midplane,
    natural units with R0 = 8 kpc) are replaced by n_in tori evenly spaced in ln J from J_lo up to that amplitude
    (the outer tori are unchanged). The held-out grid (Evaluator) always uses the base spacing."""
    J = j_grid(d["n_orbits"], J_lo, J_hi, omega_ref, d.get("spacing", "asinh"), d.get("a0", 0.03), offset=0.0)
    inner = d.get("inner")
    if inner:
        a_split = inner["split_pc"] / 1000 / RO_KPC
        J_split = omega_ref * a_split ** 2 / 2
        outer = J[J >= J_split]
        lnJ = math.log(J_lo) + (math.log(J_split) - math.log(J_lo)) * torch.arange(inner["n"], dtype=J.dtype) / inner["n"]
        J = torch.cat([lnJ.exp(), outer])
    return J


def angles(n, offset=0.5, dtype=F64):
    return 2 * math.pi * (torch.arange(n, dtype=dtype) + offset) / n


def aa_points(J, th):
    """(nO,) actions x (nS,) angles -> (nO, nS, 2) [theta, J]."""
    return torch.stack(torch.broadcast_tensors(th[None, :], J[:, None]), dim=-1)


# ======================================================================== the model
class ToyMap(torch.nn.Module):
    """Toy SHO ellipse with frequency nu0 (fixed) or ln omega(J) = ln nu0 + Chebyshev series in ln J (learned).

    anchor (learned only) pins the series at the smallest training action J_lo, where the true frequency is
    nu0 (to ~5e-5 at z_max = 5 pc) but the loss cannot fix the level of omega (a change of toy frequency and a
    linear stretch of the flow are interchangeable near the midplane):
      None          free series, k = 0..n-1
      "value"       omega(J_lo) = nu0                 (basis T_k(s) - T_k(-1), k = 1..n-1)
      "value_slope" omega(J_lo) = nu0, kappa(J_lo) = 0 (also - T_k'(-1)(s + 1), k = 2..n-1); with the linear
                    continuation below J_lo this gives omega = nu0 for every smaller action
    """

    def __init__(self, omega0, learned=False, J_lo=1e-6, J_hi=1.0, n_cheb=6, anchor=None, var="lnJ"):
        super().__init__()
        self.register_buffer("ln_w0", torch.tensor(math.log(omega0), dtype=torch.float64))
        self.learned = learned
        self.lnJ_lo, self.lnJ_hi = math.log(J_lo), math.log(J_hi)
        # the series variable: ln J (resolution spread evenly over the decades of J) or the amplitude sqrt(J)
        # (resolution where the orbits are large; the region near the midplane is a sliver of the range)
        assert var in ("lnJ", "amp"), var
        self.var = var
        self.x_lo, self.x_hi = (self.lnJ_lo, self.lnJ_hi) if var == "lnJ" else (math.sqrt(J_lo), math.sqrt(J_hi))
        assert anchor in (None, "value", "value_slope"), anchor
        self.anchor = anchor
        self.k0 = {None: 0, "value": 1, "value_slope": 2}[anchor]  # pinned modes carry no parameter
        self.n_cheb = n_cheb
        self.c = torch.nn.Parameter(torch.zeros(n_cheb - self.k0)) if learned else None

    def _series(self, s):
        """sum_k c_k phi_k(s) and its s-derivative over the free modes k = k0..n-1 (T_k' = k U_{k-1})."""
        T = [torch.ones_like(s), s]; U = [torch.ones_like(s), 2 * s]
        for _ in range(2, self.n_cheb):
            T.append(2 * s * T[-1] - T[-2]); U.append(2 * s * U[-1] - U[-2])
        val, der = 0.0, 0.0
        for i, k in enumerate(range(self.k0, self.n_cheb)):
            phi, dphi = T[k], (k * U[k - 1] if k else torch.zeros_like(s))
            if self.anchor:  # T_k(-1) = (-1)^k
                phi = phi - (-1) ** k
            if self.anchor == "value_slope":  # T_k'(-1) = (-1)^(k+1) k^2
                d0 = (-1) ** (k + 1) * k * k
                phi, dphi = phi - d0 * (s + 1), dphi - d0
            val, der = val + self.c[i] * phi, der + self.c[i] * dphi
        return val, der

    def ln_omega_kappa(self, J):
        if not self.learned:
            return self.ln_w0.expand_as(J), torch.zeros_like(J)
        scale = 2.0 / (self.x_hi - self.x_lo)
        x = torch.log(J) if self.var == "lnJ" else torch.sqrt(J)
        s = scale * (x - self.x_lo) - 1
        sc = s.clamp(-1.0, 1.0)  # beyond the trained range: continue linearly in the series variable (C1)
        val, der = self._series(sc)
        dx_dlnJ = 1.0 if self.var == "lnJ" else 0.5 * x  # kappa = d ln omega / d ln J = (d/ds)(ds/dx)(dx/d ln J)
        kappa = der * scale * dx_dlnJ
        return self.ln_w0 + val + der * (s - sc), kappa

    def forward(self, aa):
        th, J = aa[..., 0], aa[..., 1]
        lnw, _ = self.ln_omega_kappa(J)
        w = torch.exp(lnw)
        return torch.stack([torch.sqrt(2 * J / w) * torch.cos(th), -torch.sqrt(2 * J * w) * torch.sin(th)], dim=-1)

    def canonical_angle(self, th, J):
        """The model's input angle is the ellipse parameter; this is the conjugate (canonical) angle."""
        _, kappa = self.ln_omega_kappa(J)
        return th - 0.5 * kappa * torch.sin(2 * th)

    def angle_weights(self, J, th):
        """d theta_c / d theta for (nO,) actions x (nS,) input angles: the weight of each sample in a torus
        (time) average. Its mean over uniform angles is 1; identically 1 for the fixed toy."""
        _, kappa = self.ln_omega_kappa(J)
        return 1 - kappa[:, None] * torch.cos(2 * th)[None, :]


class OddNNConditioner(SimpleNNConditioner):
    """Odd part of the repo conditioner: f(-x) = -f(x), so every shear fixes the origin."""

    def forward(self, inputs):
        return 0.5 * (super().forward(inputs) - super().forward(-inputs))


class AAMap(torch.nn.Module):
    def __init__(self, toy, flow):
        super().__init__()
        self.toy, self.flow = toy, flow

    def forward(self, aa):
        return self.flow(self.toy(aa))


def build_model(mcfg, toy_cfg, omega0, J_lo, J_hi, seed, dtype=F64, device=HOST):
    torch.manual_seed(seed)
    cond = OddNNConditioner if mcfg.get("odd", False) else SimpleNNConditioner
    flow = Flow(2, mcfg["n_layers"], SymplecticCouplingLayer, cond,
                dict(num_layers=mcfg.get("depth", 2), projection_dims=mcfg["width"],
                     activation=ACTS[mcfg.get("act", "tanh")], include_bias=True))
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        if mcfg.get("bias_init", "torch") == "torch":  # PyTorch default U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            for m in flow.modules():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    b = 1.0 / math.sqrt(m.in_features)
                    m.bias.copy_((2 * torch.rand(m.bias.shape, generator=g) - 1) * b)
        gs = mcfg.get("gate_std", 1e-3)
        for n, p in flow.named_parameters():
            if n.endswith("gate"):
                p.copy_(gs * torch.randn(p.shape, generator=g))
    toy = ToyMap(omega0, toy_cfg.get("mode", "fixed") == "learned", J_lo, J_hi, toy_cfg.get("n_cheb", 6),
                 toy_cfg.get("anchor"), toy_cfg.get("var", "lnJ"))
    return AAMap(toy, flow).to(dtype).to(device)


def state_f64(model):
    return {k: v.detach().cpu().double() for k, v in model.state_dict().items()}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ================================================================ residuals and metrics
def to_host64(t):
    return t.detach().cpu().double() if t.dtype != F64 or t.device != HOST else t


def residual_host(model, aa_dev, target, nO, nS, escale=None):
    """Per-point energy deviation (H - <H>) / scale, H in float64 on the host; differentiable through the
    model. scale = escale (nO, 1) if given (training: fixed toy energy), else |<H>| (the relative metric)."""
    y = model(aa_dev.reshape(-1, 2))
    y64 = y.cpu().double() if (y.dtype != F64 or y.device != HOST) else y
    H = target.H(y64).view(nO, nS)
    m = H.mean(1, keepdim=True)
    return (H - m) / (escale if escale is not None else m.abs())


PCTS = {"p90": 0.90, "p99": 0.99, "p999": 0.999}


def torus_stats(e, w=None):
    """Per-torus mean, max and percentiles of the per-angle errors e (n_tori, n_angles) along each orbit.
    w: per-sample weights (dtheta_c/dtheta for a learned toy, so the statistics are time averages), or None.
    The p-th percentile is the error the orbit spends a fraction p of its time below."""
    if w is None:
        w = torch.ones_like(e)
    w = w / w.sum(-1, keepdim=True)
    es, idx = e.sort(-1)
    cw = w.gather(-1, idx).cumsum(-1)
    q = {}
    for name, p in PCTS.items():
        k = torch.searchsorted(cw, torch.full((e.shape[0], 1), p, dtype=cw.dtype)).clamp(max=e.shape[1] - 1)
        q[name] = es.gather(-1, k).squeeze(-1)
    return (w * e).sum(-1), e.max(-1).values, q


class Evaluator:
    """Held-out tori (midpoints between training tori, angles offset from training angles), training
    tori, z_max bands for MW, torus-mean action / frequency errors and the image of the origin.
    Percentiles along each orbit use the same held-out tori at 4096 angles (the top 0.1% of an orbit is then
    ~4 samples; with the smooth per-angle errors this is converged to <1%); eval_p90/p99/p999 are their
    means over tori. The logged eval_mean / eval_max keep the 512-angle grid of earlier runs."""

    def __init__(self, cfg, target, J_lo, J_hi, omega_ref, train_J, n_train_steps):
        d, e = cfg["data"], cfg.get("eval", {})
        self.target = target
        self.J = j_grid(e.get("n_orbits", 200), J_lo, J_hi, omega_ref, d.get("spacing", "asinh"), d.get("a0", 0.03),
                        offset=0.5)
        self.th = angles(e.get("n_steps", 512), offset=0.25)
        self.aa = aa_points(self.J, self.th)
        self.th_pct = angles(e.get("n_steps_pct", 4096), offset=0.25)
        self.aa_train = aa_points(train_J, angles(n_train_steps, 0.5))
        self.mw = target.name == "mw"
        self.mw_range = "zmin_pc" in d  # tori labelled by MW z_max (MW itself, or a teacher made from an MW model)
        self.z = zmax_kpc_of_J(self.J, (target if self.mw else MWVertical()).phi) if self.mw_range else None
        # exact energy of each held-out torus: quadrature for MW; omega*J for the SHO and the teacher
        self.E_true = (E_of_J(self.J, target.phi) if self.mw else
                       (target.omega * self.J)).reshape(-1, 1)
        self.bands = [(0.005, 0.15), (0.15, 1.0), (1.0, 9.0)]

    @torch.no_grad()
    def orbit_stats(self, m64, chunk=25):
        """Per-torus mean, max and percentiles along each held-out orbit, at the fine angle grid."""
        em, ex, q = [], [], {}
        for i in range(0, len(self.J), chunk):
            Jc = self.J[i:i + chunk]
            e = residual_host(m64, aa_points(Jc, self.th_pct), self.target, len(Jc), len(self.th_pct)).abs()
            m_, x_, q_ = torus_stats(e, m64.toy.angle_weights(Jc, self.th_pct))
            em.append(m_); ex.append(x_)
            for k, v in q_.items():
                q.setdefault(k, []).append(v)
        return torch.cat(em), torch.cat(ex), {k: torch.cat(v) for k, v in q.items()}

    @torch.no_grad()
    def quick(self, m64):
        e = residual_host(m64, self.aa, self.target, *self.aa.shape[:2]).abs()
        et = residual_host(m64, self.aa_train, self.target, *self.aa_train.shape[:2]).abs()
        e_true = residual_host(m64, self.aa, self.target, *self.aa.shape[:2], escale=self.E_true).abs()
        H = self.target.H(m64(self.aa)).mean(-1, keepdim=True)
        out = dict(eval_mean=float(e.mean()), eval_max=float(e.max()), train_mean=float(et.mean()),
                   train_max=float(et.max()),
                   eval_mean_trueE=float(e_true.mean()),  # |H - <H>| / E_true(J): cannot be gamed
                   energy_offset_max=float((H / self.E_true - 1).abs().max()),  # ~0 unless tori are displaced
                   origin_image_norm=float(m64.flow(torch.zeros(1, 2, dtype=F64)).norm()))
        _, _, q = self.orbit_stats(m64)
        out.update({f"eval_{k}": float(v.mean()) for k, v in q.items()})  # mean over tori of each orbit's percentile
        if self.mw_range:
            em = e.mean(-1)
            for lo, hi in self.bands:
                sel = (self.z >= lo * 0.999) & (self.z <= hi)
                out[f"band_{lo:g}_{hi:g}"] = float(em[sel].mean()) if sel.any() else None
        return out

    def full(self, m64, n_af=60):
        out = self.quick(m64)
        with torch.no_grad():
            em, ex, q = self.orbit_stats(m64)  # all five statistics from the same fine angle grid
            out.update(per_torus=dict(J=self.J.tolist(), mean=em.tolist(), max=ex.tolist(),
                                      **{k: v.tolist() for k, v in q.items()}, n_angles=len(self.th_pct),
                                      z_kpc=self.z.tolist() if self.mw_range else None))
            out["origin_image"] = m64.flow(torch.zeros(1, 2, dtype=F64))[0].tolist()
        if isinstance(self.target, (MWVertical, SHO)):
            J = self.J[:: max(1, len(self.J) // n_af)].clone().requires_grad_(True)
            th = angles(512, 0.25)
            # torus mean in the canonical angle (a time average); differs from the plain mean for a learned toy
            Hm = (m64.toy.angle_weights(J, th) * self.target.H(m64(aa_points(J, th)))).mean(-1)
            dEdJ = torch.autograd.grad(Hm.sum(), J)[0]
            Jt, Om = true_action_freq(Hm.detach(), self.target.phi)
            out["action_freq"] = dict(J=J.detach().tolist(), rel_action_err=((J.detach() - Jt) / Jt).abs().tolist(),
                                      rel_freq_err=((dEdJ - Om) / Om).abs().tolist())
        return out


# ========================================================================= trainers
def train_adam(model, aa_dev, target, nO, nS, steps, lr, record, every=100, escale=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    for k in range(steps):
        loss = residual_host(model, aa_dev, target, nO, nS, escale).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sch.step()
        if k % every == 0 or k == steps - 1:
            record(k, float(loss), opt.param_groups[0]["lr"])


def train_lm(model, aa_dev, target, nO, nS, max_iters, lam=1e-3, record=None, ckpt=None, every=25,
             wall_cap=None, stall_window=None, stall_rtol=None, chunk_orbits=16, it0=0, escale=None,
             geodesic=False, geo_h=0.1, geo_alpha=0.75, weight_fn=None, objective=None):
    """Levenberg-Marquardt on r_i / sqrt(N). Per-point Jacobians of the model outputs (vmap of jacrev) on
    the model's device/dtype, chained analytically through H (float64, host) and the per-torus mean;
    J^T J and J^T r are accumulated over chunks of tori (memory O(chunk * P + P^2)) and solved in float64.
    Every step is accepted only if the float64 loss decreases. Returns (iterations done, stop reason, lam).

    geodesic: add the geodesic-acceleration correction of Transtrum & Sethna (2012), aimed at sloppy problems
    whose LM steps are limited by curvature of the residual manifold. With the LM step d, the second
    directional derivative r_dd ~ (2/h)[(r(v + h d) - r(v))/h - J d] gives a = -(J^T J + lam D)^-1 J^T r_dd,
    and the step becomes d + a/2 when 2|a|_D/|d|_D <= geo_alpha. If the correction is too large, or the corrected
    step does not lower the loss, the plain step d is tried before lam is raised, so a geodesic step is never
    worse than a plain one. Costs one or two extra residual evaluations and one vector-Jacobian product per trial.

    weight_fn(r, k) -> w: optional per-point weights (nO, nS), recomputed from the current residuals r (nO, nS)
    at the start of every step k and held fixed within it; the step then minimises sum_i w_i r_i^2. This gives
    iteratively reweighted least squares for other objectives (L_p, minimax by Lawson's method, per-torus
    balance). The stall test compares objectives across steps, so pass stall_window=None with weights.
    objective(r) -> float: optional objective of the unweighted residual vector r (N,), used to accept steps and
    reported as the loss (default: half the weighted sum of squares). With reweighting, accepting on the true
    objective (e.g. the L_p norm) keeps it monotone; the weighted step is then only a proposed direction."""
    from torch.func import functional_call, jacrev, vmap
    dev, dt = next(model.parameters()).device, next(model.parameters()).dtype
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [p.numel() for _, p in model.named_parameters()]
    N = nO * nS; sN = math.sqrt(N)
    to_dev = lambda t: t.to(dt).to(dev)

    def unflat(v):
        out, i = {}, 0
        for n, s_, k in zip(names, shapes, sizes):
            out[n] = v[i:i + k].view(s_); i += k
        return out

    def f_point(p, x):
        return functional_call(model, p, (x[None, :],))[0]

    def resid(v):
        with torch.no_grad():
            y = to_host64(functional_call(model, unflat(v), (aa_dev.reshape(-1, 2),)))
            H = target.H(y).view(nO, nS)
            m = H.mean(1, keepdim=True)
            return ((H - m) / (escale if escale is not None else m.abs()) / sN).reshape(-1)

    def jtv(v, u):
        """J^T u for the residual Jacobian at v (one backward pass)."""
        vv = v.detach().clone().requires_grad_(True)
        y = to_host64(functional_call(model, unflat(vv), (aa_dev.reshape(-1, 2),)))
        H = target.H(y).view(nO, nS)
        m = H.mean(1, keepdim=True)
        r = ((H - m) / (escale if escale is not None else m.abs()) / sN).reshape(-1)
        return to_host64(torch.autograd.grad(r, vv, grad_outputs=u.to(r.dtype))[0])

    sw = None  # sqrt of the per-point weights of the current step (host, flat), or None

    def resid_w(v):
        r = resid(v)
        return r if sw is None else r * sw

    def jtv_w(v, u):
        return jtv(v, u if sw is None else sw * u)

    def normal_eqs(v):
        P = v.numel()
        A = torch.zeros(P, P, dtype=dt, device=dev); g = torch.zeros(P, dtype=dt, device=dev); rs = []
        for o0 in range(0, nO, chunk_orbits):
            sl = aa_dev[o0:o0 + chunk_orbits]; k = sl.shape[0]
            x = sl.reshape(-1, 2)
            # Jacobian w.r.t. the parameter dict, then concatenated: identical to differentiating w.r.t. the
            # flat vector v, but ~20x cheaper (slicing v makes every per-point backward build a dense (n, P) array)
            Jd = vmap(jacrev(f_point), in_dims=(None, 0))(unflat(v), x)
            Jy = torch.cat([Jd[n].reshape(x.shape[0], 2, -1) for n in names], -1)  # (k*nS, 2, P)
            del Jd
            with torch.no_grad():
                y = functional_call(model, unflat(v), (x,))
            y64 = to_host64(y).requires_grad_(True)
            Hf = target.H(y64)
            dHdy = torch.autograd.grad(Hf.sum(), y64)[0]
            H = Hf.detach().view(k, nS)
            m = H.mean(1, keepdim=True)
            if escale is not None:  # fixed scale: dr = (dH - d<H>) / E_ref
                es = escale[o0:o0 + k]
                r = ((H - m) / es / sN).reshape(-1)
                c1 = to_dev(1 / es / sN); c2 = torch.zeros_like(to_dev(H))
            else:                   # relative to <H>: extra quotient-rule term
                r = ((H - m) / m.abs() / sN).reshape(-1)
                c1 = to_dev(1 / m.abs() / sN); c2 = to_dev((H - m) * torch.sign(m) / m ** 2 / sN)
            dH = (to_dev(dHdy)[:, :, None] * Jy).sum(1).view(k, nS, -1)
            del Jy
            dm = dH.mean(1, keepdim=True)
            Jm = ((dH - dm) * c1[..., None] - c2[..., None] * dm).reshape(k * nS, -1)
            if sw is not None:
                swc = sw[o0 * nS:(o0 + k) * nS]
                r = r * swc; Jm = Jm * to_dev(swc)[:, None]
            A += Jm.T @ Jm; g += Jm.T @ to_dev(r)
            rs.append(r)
        return torch.cat(rs), to_host64(g), to_host64(A)

    v = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    hist, why, k = [], "max_iters", -1
    t_start = time.perf_counter()
    for k in range(max_iters):
        if weight_fn is not None:
            sw = weight_fn(resid(v).view(nO, nS) * sN, k).reshape(-1).to(F64).sqrt()
        r, g, A = normal_eqs(v)
        f = 0.5 * float(r @ r) if objective is None else objective(resid(v))
        dg = torch.diagonal(A).clone()
        accepted = False
        for _ in range(30):
            L, info = torch.linalg.cholesky_ex(A + lam * torch.diag(dg + 1e-12 * dg.max()))
            if int(info) != 0:
                lam *= 10; continue
            d = -torch.cholesky_solve(g[:, None], L)[:, 0]
            trials = [d]
            if geodesic:
                Jd = A @ d  # J^T J d
                Jt_rdd = (2 / geo_h) * ((jtv_w(v, resid_w(v + to_dev(geo_h * d))) - g) / geo_h - Jd)
                a = -torch.cholesky_solve(Jt_rdd[:, None], L)[:, 0]
                # use the corrected step only while the quadratic correction is small next to the step;
                # otherwise (and if it fails to descend) fall back to the plain LM step before raising lam
                if 2 * float((dg * a * a).sum().sqrt()) <= geo_alpha * float((dg * d * d).sum().sqrt()):
                    trials = [d + 0.5 * a, d]
            for dd in trials:
                vn = v + to_dev(dd)
                fn = 0.5 * float(resid_w(vn).pow(2).sum()) if objective is None else objective(resid(vn))
                if math.isfinite(fn) and fn < f:
                    v, f, accepted = vn, fn, True
                    break
            if accepted:
                lam = max(lam / 3, 1e-15)
                break
            lam *= 4
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                n = p.numel(); p.copy_(v[i:i + n].view_as(p)); i += n
        hist.append(f)
        stalled = bool(stall_window and len(hist) > stall_window
                       and (hist[-stall_window - 1] - f) / hist[-stall_window - 1] < stall_rtol)
        out_of_time = bool(wall_cap and time.perf_counter() - t_start > wall_cap)
        stop = (not accepted) or stalled or out_of_time
        if record and (k % every == 0 or stop):
            record(it0 + k, 2 * f, lam)
        if ckpt and (k % every == 0 or stop):
            ckpt(it0 + k + 1, lam)
        if stop:
            why = "no-descent" if not accepted else ("converged" if stalled else "walltime")
            break
    return k + 1, why, lam
