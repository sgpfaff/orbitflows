"""Checks for the halo benchmark pieces (laptop, a few seconds each): the isochrone toy is an exact action-angle map,
the quadrature H(J) reproduces the isochrone's closed form, MWPotential2014 matches galpy, the NFW frequencies match
orbit integration, and galpy's own action-angle routines agree with the toy and the NFW ground truth. Writes
results/halo_checks.json;  --galpy2 (galpy >= 2.0) compares with galpy's torch backend -> results/halo_galpy2.json."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import math
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core3d as C  # noqa: E402

F64 = torch.float64


def toy_checks(GM=1.0, b=1.0, n=400, seed=0):
    """Symplectic Jacobian, energy = E(J), |L| = L, L_z, and Hamilton's flow tangent to the torus at the exact frequencies."""
    g = torch.Generator().manual_seed(seed)
    th = 2 * math.pi * torch.rand(n, 3, generator=g, dtype=F64)
    Jr = 10 ** (-3 + 3 * torch.rand(n, generator=g, dtype=F64)); L = 10 ** (-2 + 2.5 * torch.rand(n, generator=g, dtype=F64))
    Lz = L * (2 * torch.rand(n, generator=g, dtype=F64) - 1) * 0.95
    aa = torch.cat([th, torch.stack([Jr, L, Lz], -1)], -1)
    toy, iso = C.IsochroneToy(GM, b), C.Isochrone(GM, b)
    xv = toy(aa)
    E = iso.H(xv); Et = iso.E_of_J(aa[:, 3:])
    Lv = torch.cross(xv[:, :3], xv[:, 3:], dim=-1)
    M = torch.func.vmap(torch.func.jacfwd(lambda a: toy(a[None])[0]))(aa)       # d(x, v)/d(theta, J)
    Om = torch.zeros(6, 6, dtype=F64); Om[:3, 3:] = torch.eye(3); Om[3:, :3] = -torch.eye(3)
    symp = (M.transpose(1, 2) @ Om @ M - Om).abs().amax((1, 2))
    # invariance: X_H(z) = sum_i Omega_i dz/dtheta_i with the exact frequencies
    X = C.hamiltonian_field(iso, xv)
    Xp = torch.einsum("nai,ni->na", M[:, :, :3], toy.freqs(aa[:, 3:]))
    inv = ((X - Xp).norm(dim=-1) / X.norm(dim=-1))
    return dict(symplectic_max=float(symp.max()), energy_rel_max=float(((E - Et) / (Et - iso.E0)).abs().max()),
                L_rel_max=float((Lv.norm(dim=-1) / L - 1).abs().max()), Lz_rel_max=float(((Lv[:, 2] - Lz) / L).abs().max()),
                invariance_rel_max=float(inv.max()), n=n, Jr_range=[float(Jr.min()), float(Jr.max())], L_range=[float(L.min()), float(L.max())])


def quadrature_checks():
    """SphericalPotential (quadrature) against the isochrone's exact H(J) and frequencies."""
    q, iso = C.SphericalIsochrone(1.0, 1.0), C.Isochrone(1.0, 1.0)
    J = C.spherical_action_grid(5, 0.001, 0.5, 0.01, 1.5, 0.9, midpoints=True)
    E, Et = q.E_of_J(J), iso.E_of_J(J)
    f, ft = q.freqs(J), iso.freqs(J)
    return dict(E_rel_max=float(((E - Et) / (Et - iso.E0)).abs().max()), freq_rel_max=float(((f - ft)[:, :2] / ft[:, :2]).abs().max()), n=len(J))


def mw_checks(n=2000, seed=1):
    """Potential and forces against galpy's MWPotential2014 at random points, 0.05 < r < 5 (natural units)."""
    import numpy as np
    from galpy.potential import MWPotential2014, evaluatePotentials, evaluateRforces, evaluatezforces
    rng = np.random.default_rng(seed)
    r = 10 ** rng.uniform(-1.3, 0.7, n); ct = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n)
    R, z = r * np.sqrt(1 - ct ** 2), r * ct
    x = torch.tensor(np.stack([R * np.cos(ph), R * np.sin(ph), z], -1), dtype=F64, requires_grad=True)
    mw = C.MWPotential2014(); P = mw.pot(x); g = torch.autograd.grad(P.sum(), x)[0].detach().numpy()
    Pg = np.array([evaluatePotentials(MWPotential2014, a, b) for a, b in zip(R, z)])
    FR = np.array([evaluateRforces(MWPotential2014, a, b) for a, b in zip(R, z)]); Fz = np.array([evaluatezforces(MWPotential2014, a, b) for a, b in zip(R, z)])
    FR_t = -(g[:, 0] * np.cos(ph) + g[:, 1] * np.sin(ph)); Fz_t = -g[:, 2]
    F = np.sqrt(FR ** 2 + Fz ** 2)
    E0 = float(evaluatePotentials(MWPotential2014, 0.0, 0.0))
    return dict(pot_rel_max=float(np.max(np.abs(P.detach().numpy() - Pg) / np.abs(Pg))),
                force_rel_max=float(np.max(np.sqrt((FR_t - FR) ** 2 + (Fz_t - Fz) ** 2) / F)),
                E0=mw.E0, E0_galpy=E0, vc2_at_R0=float(-evaluateRforces(MWPotential2014, 1.0, 0.0)), n=n)


def nfw_orbit_checks(n=8, seed=2):
    """NFW frequencies from quadrature against orbit integration: start at pericentre, integrate one radial period
    (Yoshida 6th-order leapfrog) and check that r returns to r_p with zero radial velocity and that the in-plane angle
    has advanced by 2 pi Omega_psi / Omega_r."""
    t = C.NFW(2.0, 1.0)
    g = torch.Generator().manual_seed(seed)
    J = torch.stack([0.02 + 0.3 * torch.rand(n, generator=g, dtype=F64), 0.2 + 1.3 * torch.rand(n, generator=g, dtype=F64), torch.zeros(n, dtype=F64)], -1)
    E = t.E_of_J(J); Om = t.freqs(J); L = J[:, 1]
    rp, _ = t._orbit(E, L)
    x = torch.stack([rp, 0 * rp, 0 * rp], -1); v = torch.stack([0 * rp, L / rp, 0 * rp], -1)
    T = 2 * math.pi / Om[:, 0]; N = 4000; dt = T / N
    w1 = 1 / (2 - 2 ** (1 / 3)); w0 = 1 - 2 * w1           # Yoshida's 4th-order composition of leapfrog

    def acc(x):  # -dPhi/dr x/r for the NFW
        r = x.norm(dim=-1, keepdim=True)
        return -t.A * (torch.log1p(r / t.a) / r ** 2 - 1 / (r * (r + t.a))) * x / r
    for _ in range(N):
        for c in (w1, w0, w1):
            h = (c * dt)[:, None]
            x = x + 0.5 * h * v; v = v + h * acc(x); x = x + 0.5 * h * v
    r = x.norm(dim=-1); vr = (x * v).sum(-1) / r
    ang = torch.atan2(x[:, 1], x[:, 0]) % (2 * math.pi)
    want = (2 * math.pi * Om[:, 1] / Om[:, 0]) % (2 * math.pi)
    dang = torch.remainder(ang - want + math.pi, 2 * math.pi) - math.pi
    return dict(r_return_rel_max=float(((r - rp) / rp).abs().max()), vr_rel_max=float((vr / (L / rp)).abs().max()),
                angle_err_max=float(dang.abs().max()), n=n, steps_per_period=N)


def _cyl(xv):
    x, y, z, vx, vy, vz = [xv[:, i].numpy() for i in range(6)]
    import numpy as np
    R = np.hypot(x, y)
    return R, (x * vx + y * vy) / R, (x * vy - y * vx) / R, z, vz, np.arctan2(y, x)


def galpy_aa_checks(n=300, seed=3):
    """Independent of the code here: galpy's actionAngleIsochrone recovers (J, theta, Omega) from the toy's points, and
    its actionAngleSpherical reproduces the NFW quadrature (galpy's own radial integrals are good to ~1e-9)."""
    import numpy as np
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import IsochronePotential, NFWPotential
    cfg = json.load(open(os.path.join(HERE, "configs", "halo", "halo_mw_w32_s0.json"))); GM, b = cfg["toy"]["GM"], cfg["toy"]["b"]
    g = torch.Generator().manual_seed(seed)
    th = 2 * math.pi * torch.rand(n, 3, generator=g, dtype=F64)
    Jr = 10 ** (-2 + 1.7 * torch.rand(n, generator=g, dtype=F64)); L = 10 ** (-0.7 + 1.0 * torch.rand(n, generator=g, dtype=F64))
    Lz = L * 0.9 * (2 * torch.rand(n, generator=g, dtype=F64) - 1)
    aa = torch.cat([th, torch.stack([Jr, L, Lz], -1)], -1); toy = C.IsochroneToy(GM, b)
    jr, lz, jz, Or, Op, Oz, ar, ap, az = actionAngleIsochrone(ip=IsochronePotential(amp=GM, b=b)).actionsFreqsAngles(*_cyl(toy(aa)))
    fr = toy.freqs(aa[:, 3:]).numpy(); Ln = L.numpy()
    wrap = lambda d: float(np.max(np.abs((d + np.pi) % (2 * np.pi) - np.pi)))
    out = dict(toy_actions=float(max(np.max(np.abs(jr - Jr.numpy()) / (Jr.numpy() + Ln)), np.max(np.abs(lz - Lz.numpy()) / Ln), np.max(np.abs(jz - (L - Lz.abs()).numpy()) / Ln))),
               toy_freqs=float(max(np.max(np.abs(Or / fr[:, 0] - 1)), np.max(np.abs(Oz / fr[:, 1] - 1)))),
               toy_angles=max(wrap(ar - th[:, 0].numpy()), wrap(az - th[:, 1].numpy()), wrap(ap - (th[:, 2] + torch.sign(Lz) * th[:, 1]).numpy())), n=n)
    t = C.NFW(2.0, 1.0); nfw = NFWPotential(a=2.0, normalize=1.0)
    J = C.spherical_action_grid(5, 0.01, 0.5, 0.2, 2.0, 0.9, midpoints=True)[::3]
    E = t.E_of_J(J); Om = t.freqs(J).numpy(); rp, ra = t._orbit(E, J[:, 1])
    R = (0.5 * (rp + ra)).numpy(); vT = J[:, 1].numpy() / R; vR = np.sqrt(2 * (E.numpy() - t.phi(torch.tensor(R)).numpy()) - vT ** 2)
    jr2, _, _, Or2, Op2, _ = actionAngleSpherical(pot=nfw).actionsFreqs(R, vR, vT, 0 * R, 0 * R, 0 * R)
    out.update(nfw_Jr=float(np.max(np.abs(jr2 / J[:, 0].numpy() - 1))), nfw_freqs=float(max(np.max(np.abs(Or2 / Om[:, 0] - 1)), np.max(np.abs(Op2 / Om[:, 1] - 1)))), nfw_n=len(R))
    return out


def galpy2_backend_check(n=4000, seed=1):
    """galpy >= 2.0 only: its torch backend's MWPotential2014 (values and autograd forces) against the potential
    written here, and against galpy's own numpy path."""
    import numpy as np
    import galpy
    from galpy.potential import MWPotential2014, evaluatePotentials
    rng = np.random.default_rng(seed)
    r = 10 ** rng.uniform(-1.3, 0.7, n); ct = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n); R, z = r * np.sqrt(1 - ct ** 2), r * ct
    x = torch.tensor(np.stack([R * np.cos(ph), R * np.sin(ph), z], -1), dtype=F64)
    mw = C.MWPotential2014(); mg = C.MWPotential2014(backend="galpy")
    xa = x.clone().requires_grad_(True); Pn = mw.pot(xa); gn = torch.autograd.grad(Pn.sum(), xa)[0]
    xb = x.clone().requires_grad_(True); Pg = mg.pot(xb); gg = torch.autograd.grad(Pg.sum(), xb)[0]
    Pnp = evaluatePotentials(MWPotential2014, R, z)
    return dict(galpy=galpy.__version__, torch=torch.__version__, n=n,
                pot_vs_native=float(((Pn - Pg).abs() / (Pg - mw.E0).abs()).max()), force_vs_native=float(((gn - gg).norm(dim=-1) / gg.norm(dim=-1)).max()),
                pot_vs_numpy=float(np.max(np.abs(Pg.detach().numpy() - Pnp) / np.abs(Pnp - mw.E0))))


def identity_check():
    """Isochrone toy onto the same isochrone with the flow switched off (gates and linear shears zero): the identity
    is the exact answer, so the held-out error and the errors against the exact H(J) should be at round-off."""
    cfg = json.load(open(os.path.join(HERE, "configs", "halo", "halo_nfw_w32_s0.json"))); toy, d = cfg["toy"], cfg["data"]
    t = C.Isochrone(toy["GM"], toy["b"]); m = C.build_model(cfg["model"], toy)
    with torch.no_grad():
        for n_, p in m.named_parameters():
            if n_.endswith(".g") or n_.endswith(".M"):
                p.zero_()
    Jh = C.spherical_action_grid(d["n_per_axis"], d["Jr"][0], d["Jr"][1], d["L"][0], d["L"][1], d["cosi_max"], True)
    ev = C.Evaluator(t, J=Jh, n_angles=1024); q = ev.quick(m); af = ev.action_freq(m)
    return dict(held_out_mean=q["eval_mean"], held_out_max=q["eval_max"], E_vs_exact_max=max(af["rel_E_err"]), freq_vs_exact_max=max(af["rel_freq_err"]))


if __name__ == "__main__":
    if "--galpy2" in sys.argv:  # run in an environment with galpy >= 2.0 (torch backend)
        out = galpy2_backend_check(); print(json.dumps(out, indent=1))
        json.dump(out, open(os.path.join(HERE, "results", "halo_galpy2.json"), "w"), indent=1); sys.exit()
    out = dict(toy=toy_checks(), quadrature=quadrature_checks(), mw=mw_checks(), identity=identity_check(), galpy_aa=galpy_aa_checks())
    print(json.dumps(out, indent=1), flush=True)
    if "--orbits" in sys.argv:
        out["nfw_orbits"] = nfw_orbit_checks(); print(json.dumps(out["nfw_orbits"], indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "halo_checks.json"), "w"), indent=1)
