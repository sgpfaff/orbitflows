"""Recompute every number and curve on the 3D progress page from the saved models and logs -> report/data.json.

Sanity checks are re-run here (not copied), so the page always reflects the current code.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import itertools
import json
import math
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import core3d as C  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
F = torch.float64
RES = os.path.join(ROOT, "results")


def load(name, key="model.pt"):
    ck = torch.load(os.path.join(RES, name, key) if key else os.path.join(RES, name), weights_only=False)
    c = ck["config"]; t = C.make_target(c["target"]); m = C.build_model(c["model"], c["toy"]); m.load_state_dict(ck["state"])
    return c, t, m


def grids(c):
    d = c["data"]
    if d.get("kind") == "spherical":
        sg = lambda mid: C.spherical_action_grid(d["n_per_axis"], *d["Jr"], *d["L"], d.get("cosi_max", 0.9), mid)
        Jt, Jh = sg(False), sg(True)
        if d.get("tori") == "sobol":
            Jt = C.sobol_spherical_actions(d["n_tori"], *d["Jr"], *d["L"], d.get("cosi_max", 0.9), seed=1 + c.get("seed", 0))
    else:
        Jt, Jh = C.action_grid(d["n_per_axis"], d["J_lo"], d["J_hi"]), C.action_grid(d["n_per_axis"], d["J_lo"], d["J_hi"], midpoints=True)
        if d.get("tori") == "sobol":
            Jt = C.sobol_actions(d["n_tori"], d["J_lo"], d["J_hi"], seed=1 + c.get("seed", 0))
    return Jt, Jh, C.angle_samples(d["n_angles"], c.get("seed", 0)), C.angle_samples(1024, 12345)


# --------------------------------------------------------------------------------------------- sanity checks
def sanity():
    out = []
    add = lambda group, what, value, target, ok: out.append(dict(group=group, what=what, value=value, target=target, ok=bool(ok)))
    torch.manual_seed(0)
    Om6 = torch.zeros(6, 6, dtype=F); Om6[:3, 3:] = torch.eye(3, dtype=F); Om6[3:, :3] = -torch.eye(3, dtype=F)

    def sympl(f, pts):
        return max(float((M.T @ Om6 @ M - Om6).abs().max()) for M in (torch.autograd.functional.jacobian(f, z) for z in pts))
    th = 2 * math.pi * torch.rand(12, 3, dtype=F)
    add("Toy maps", "Cartesian oscillator toy is symplectic: max |MᵀΩM − Ω|", sympl(C.Toy3D([1.0, 1.4, 1.9]), torch.cat([th, torch.rand(12, 3, dtype=F) + 0.05], -1)), "≈ 1e-16", True)
    sph = C.SphericalToyHO(0.5)
    Jr = 0.5 * torch.rand(12, dtype=F) + 1e-3; L = torch.rand(12, dtype=F) + 0.05; Lz = L * (1.8 * torch.rand(12, dtype=F) - 0.9)
    Js = torch.stack([Jr, L, Lz], -1); pts = torch.cat([th, Js], -1)
    add("Toy maps", "Spherical-action oscillator toy is symplectic: max |MᵀΩM − Ω|", sympl(sph, pts), "≈ 1e-15", True)
    xv = sph(pts); x, v = xv[:, :3], xv[:, 3:]
    E = 0.5 * (v * v).sum(-1) + 0.125 * (x * x).sum(-1)
    add("Toy maps", "Spherical toy energy equals ω(2J_r + L)", float((E / sph.energy(Js) - 1).abs().max()), "≈ 1e-16", True)
    Lv = torch.cross(x, v, dim=-1)
    add("Toy maps", "Spherical toy angular momentum: |x × v| = L and (x × v)_z = L_z (relative to L)", max(float((Lv.norm(dim=-1) / L - 1).abs().max()), float(((Lv[:, 2] - Lz) / L).abs().max())), "≈ 1e-16", True)
    tt = 0.7; xe = x * math.cos(0.5 * tt) + v / 0.5 * math.sin(0.5 * tt); ve = -x * 0.5 * math.sin(0.5 * tt) + v * math.cos(0.5 * tt)
    xt = sph(torch.cat([th + tt * torch.tensor([1.0, 0.5, 0.0], dtype=F), Js], -1))
    add("Toy maps", "Advancing the toy angles at (2ω, ω, 0) reproduces the exact oscillator motion", float((xt - torch.cat([xe, ve], -1)).abs().max()), "≈ 1e-16", True)

    q = C.SeparableQuartic(lams=(1.5, 2.5, 4.0)); J = torch.tensor([[1e-4, 0.05, 0.3], [0.2, 0.01, 1e-3], [1.0, 0.5, 0.8]], dtype=F)
    Eq = q.E_of_J_axes(J); Jb, _ = q.J_and_dJdE(Eq)
    add("Ground truth", "Separable quartic: J(E(J)) round trip", float((Jb / J - 1).abs().max()), "≈ 1e-16", True)
    h = 1e-4; fd = (q.E_of_J_axes(J * (1 + h)) - q.E_of_J_axes(J * (1 - h))) / (2 * h * J)
    add("Ground truth", "Separable quartic: frequencies vs finite differences of E(J) (h = 1e-4, error ∝ h²)", float((q.freqs(J) / fd - 1).abs().max()), "≈ 1e-11", True)
    w0, l0 = float(q.w[0]), float(q.l[0]); E0 = float(Eq[2, 0]); xm = float(q._xmax(Eq[2])[0])
    xs = torch.linspace(-xm, xm, 400001, dtype=F); pp = torch.sqrt(torch.clamp(2 * (E0 - (0.5 * w0 ** 2 * xs ** 2 + 0.25 * l0 * xs ** 4)), min=0))
    add("Ground truth", "Separable quartic: action vs a brute-force area integral (limited by the brute force)", abs(float(torch.trapezoid(pp, xs) / math.pi) / float(J[2, 0]) - 1), "≈ 1e-8", True)
    iso = C.Isochrone(); Om0 = iso.freqs(torch.tensor([[1e-12, 1e-12, 0.0]], dtype=F))[0]
    add("Ground truth", "Isochrone: small-action frequencies equal the toy's (2ω₀, ω₀, 0) = (1, 0.5, 0)", float((Om0 - torch.tensor([1.0, 0.5, 0.0], dtype=F)).abs().max()), "≈ 1e-11", True)

    def zero_flow(m):
        with torch.no_grad():
            for n, p in m.flow.named_parameters():
                if n.endswith(".g") or n.endswith(".M"):
                    p.zero_()
        return m
    th2 = C.angle_samples(128, 0)
    for lab, toy, tgt, Jg in (("Cartesian", [1.0, 1.4, 1.9], C.AnisoHO((1.0, 1.4, 1.9)), C.action_grid(3, [1e-3] * 3, [1.0] * 3)),
                              ("spherical", dict(kind="spherical_ho", omega=0.5), C.AnisoHO((0.5, 0.5, 0.5)), C.spherical_action_grid(3, 1e-3, 0.2, 1e-2, 0.5, 0.9))):
        m = zero_flow(C.build_model(dict(n_lin=1, n_blocks=1, width=8), toy)); aa = C.aa_points(Jg, th2)
        R, _ = C.invariance_residuals(m, aa, tgt, C.invariance_scale(m, aa))
        add("Invariance residual", f"Vanishes when target = toy and flow = identity ({lab})", float(R.pow(2).sum(-1).mean().sqrt()), "≈ 1e-15", True)

    # the analytic LM Jacobian of the combined residual vs finite differences, on a small problem
    m = C.build_model(dict(n_lin=1, n_blocks=2, width=8), dict(kind="spherical_ho", omega=0.5), seed=3)
    with torch.no_grad():
        for n, p in m.flow.named_parameters():
            if n.endswith(".g"):
                p.mul_(100.0)  # a visibly non-trivial map
    Jg = C.spherical_action_grid(3, 1e-3, 0.2, 1e-2, 0.5, 0.9); aa = C.aa_points(Jg, C.angle_samples(32, 0))
    unflat, resid, normal_eqs = C.lm_residual_system(m, aa, iso, m.toy.energy(Jg).reshape(-1, 1), C.invariance_scale(m, aa), 1.0, chunk=4)
    v = torch.cat([p.detach().reshape(-1) for p in m.parameters()]); r, g, A = normal_eqs(v); worst = 0.0
    for _ in range(3):
        u = torch.randn_like(v); u /= u.norm(); Ju = (resid(v + 1e-6 * u) - resid(v - 1e-6 * u)) / 2e-6
        worst = max(worst, abs(float(g @ u) / float(Ju @ r) - 1), abs(float(u @ A @ u) / float(Ju @ Ju) - 1))
    add("Optimiser", "Analytic Jacobian of energy + invariance residuals vs finite differences (relative)", worst, "≈ 1e-8", True)
    for o in out:
        o["ok"] = o["value"] < {"≈ 1e-16": 1e-14, "≈ 1e-15": 1e-13, "≈ 1e-11": 1e-9, "≈ 1e-8": 1e-6}[o["target"]]
    return out


# ------------------------------------------------------------------------------------------------- validation
def curves():
    out = {}
    for name in ("s1_aniso_ho", "s2_sep_quartic", "s2b_sep_quartic_sobol", "s3_isochrone"):
        p = os.path.join(RES, name, "log.jsonl")
        if not os.path.exists(p):
            continue
        L = [json.loads(l) for l in open(p) if l.strip()]
        out[name] = [dict(stage=r["stage"], it=r["it"], ev=r["eval_mean"], ev_max=r["eval_max"], t=r["t"]) for r in L]
    return out


def generalisation():
    out = []
    for name in ("s2_sep_quartic", "s2b_sep_quartic_sobol", "s3_isochrone"):
        if not os.path.exists(os.path.join(RES, name, "model.pt")):
            continue
        c, t, m = load(name); Jt, Jh, tht, thh = grids(c)
        with torch.no_grad():
            f = lambda J, th: float(C.residuals(m, C.aa_points(J, th), t).abs().mean())
            out.append(dict(name=name, train=f(Jt, tht), new_angles=f(Jt, thh), new_tori=f(Jh, tht), held_out=f(Jh, thh)))
    return out


# ------------------------------------------------------------------------------------------- intuitive pictures
def quartic_portraits(n=360):
    """Per-axis phase portraits (x_i, v_i) on two held-out tori: exact level curve, toy ellipse, learned points."""
    c, t, m = load("s2_sep_quartic"); _, Jh, _, _ = grids(c)
    toy = C.build_model(c["model"], c["toy"]); out = []
    th = C.angle_samples(n, 7)
    for idx in (len(Jh) // 2, len(Jh) - 1):
        J = Jh[idx:idx + 1]; aa = C.aa_points(J, th).reshape(-1, 6)
        with torch.no_grad():
            z = m(aa); z0 = toy.toy(aa)
        Ei = t.E_of_J_axes(J)[0]; axes = []
        for i in range(3):
            xm = float(t._xmax(t.E_of_J_axes(J))[0, i]); xs = torch.linspace(-xm, xm, 241, dtype=F)
            vs = torch.sqrt(torch.clamp(2 * (Ei[i] - (0.5 * t.w[i] ** 2 * xs ** 2 + 0.25 * t.l[i] * xs ** 4)), min=0))
            axes.append(dict(curve_x=xs.tolist() + xs.flip(0).tolist(), curve_v=vs.tolist() + (-vs.flip(0)).tolist(),
                             learned_x=z[:, i].tolist(), learned_v=z[:, 3 + i].tolist(), toy_x=z0[:, i].tolist(), toy_v=z0[:, 3 + i].tolist(),
                             theta=th[:, i].tolist(),
                             dev=((0.5 * z[:, 3 + i] ** 2 + 0.5 * t.w[i] ** 2 * z[:, i] ** 2 + 0.25 * t.l[i] * z[:, i] ** 4 - Ei[i]) / Ei.sum()).tolist()))
        out.append(dict(J=J[0].tolist(), axes=axes))
    return out


def isochrone_pictures(name="s3_isochrone", key="model.pt", n=600):
    """(r, p_r) portraits on held-out tori vs the exact curve, and the |L| wander of the worst torus against the
    resonant angle theta_r - 2 theta_L."""
    if key == "model.pt":
        c, t, m = load(name)
    else:
        c, t, m = load(name, key)
    _, Jh, _, thh = grids(c)
    out = dict(portraits=[])
    th = C.angle_samples(n, 7)
    with torch.no_grad():
        for idx in (5, len(Jh) // 2, len(Jh) - 3):
            J = Jh[idx:idx + 1]; z = m(C.aa_points(J, th).reshape(-1, 6)); x, v = z[:, :3], z[:, 3:]
            r = x.norm(dim=-1); pr = (x * v).sum(-1) / r
            E = float(t.E_of_J(J)); L = float(J[0, 1])
            rr = torch.linspace(float(r.min()) * 0.9, float(r.max()) * 1.05, 600, dtype=F)
            p2 = 2 * (E - t.pot(torch.stack([rr, 0 * rr, 0 * rr], -1))) - L * L / rr ** 2; ok = p2 > 0
            rc, pc = rr[ok], torch.sqrt(p2[ok])
            out["portraits"].append(dict(J=J[0].tolist(), r=r.tolist(), pr=pr.tolist(), curve_r=rc.tolist() + rc.flip(0).tolist(),
                                         curve_p=pc.tolist() + (-pc.flip(0)).tolist()))
        # |L| wander around each torus, relative to the torus's total action J_r + L (dividing by L alone inflates
        # the nearly radial, small-L tori, which are also the ones nearest Omega_r/Omega_L = 2)
        xv = m(C.aa_points(Jh, thh).reshape(-1, 6)).view(Jh.shape[0], -1, 6); Lm = torch.cross(xv[..., :3], xv[..., 3:], dim=-1).norm(dim=-1)
        scale = Jh[:, 0] + Jh[:, 1]; wander = Lm.std(1) / scale
        H = t.H(xv); ener = ((H - H.mean(1, keepdim=True)) / (H.mean(1, keepdim=True) - t.E0)).abs().mean(1)
        Om = t.freqs(Jh); out["per_torus"] = dict(wander=wander.tolist(), energy=ener.tolist(), ratio=(Om[:, 0] / Om[:, 1]).tolist(), L=Jh[:, 1].tolist())
        i = int(wander.argmax()); y = (Lm[i] - Jh[i, 1]) / scale[i]
        # which angular pattern: least squares on modes k = (k_r, k_L, k_z), |k| <= 3
        ks = [k for k in itertools.product(range(0, 4), range(-3, 4), range(-3, 4)) if k != (0, 0, 0) and (k[0] > 0 or k[1] > 0 or (k[1] == 0 and k[2] > 0))]
        K = torch.tensor(ks, dtype=F); ph = thh @ K.T
        B = torch.cat([torch.ones(thh.shape[0], 1, dtype=F), torch.cos(ph), torch.sin(ph)], 1)
        coef = torch.linalg.lstsq(B, y[:, None]).solution[:, 0]
        amp = (coef[1:1 + len(ks)] ** 2 + coef[1 + len(ks):] ** 2).sqrt(); top = torch.argsort(amp, descending=True)[:4].tolist()
        frac = 1 - float((y - B @ coef).pow(2).sum() / (y - y.mean()).pow(2).sum())
        out["worst"] = dict(J=Jh[i].tolist(), phi=torch.remainder(thh[:, 0] - 2 * thh[:, 1], 2 * math.pi).tolist(), thz=thh[:, 2].tolist(),
                            dL=y.tolist(), ratio=float(Om[i, 0] / Om[i, 1]), wander=float(wander[i]), energy=float(ener[i]),
                            explained=frac, modes=[dict(k=list(ks[j]), amp=float(amp[j]), kOmega=float(K[j] @ Om[i])) for j in top])
    return out


def main():
    t0 = time.perf_counter()
    data = dict(generated=time.strftime("%Y-%m-%d %H:%M"), sanity=sanity(), curves=curves(), generalisation=generalisation(),
                quartic=quartic_portraits(), isochrone=isochrone_pictures())
    p = os.path.join(RES, "exp_invariance.json")
    if os.path.exists(p):
        data["invariance"] = json.load(open(p))
        for lab in ("energy_only", "energy_plus_invariance"):
            if os.path.exists(os.path.join(RES, f"s3_isochrone_{lab}.pt")):
                data[f"isochrone_{lab}"] = isochrone_pictures(f"s3_isochrone_{lab}.pt", key=None)
    json.dump(data, open(os.path.join(HERE, "data.json"), "w"))
    print(f"data.json written ({time.perf_counter() - t0:.0f} s): {sum(s['ok'] for s in data['sanity'])}/{len(data['sanity'])} sanity checks pass")
    for s in data["sanity"]:
        print(f"  {'PASS' if s['ok'] else 'FAIL'}  {s['value']:.1e}  {s['what']}")


if __name__ == "__main__":
    main()
