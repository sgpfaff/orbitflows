"""Curved-path test: is a stalled student joined to its teacher T by a low-error curve in parameter space?

A quadratic Bezier curve theta(t) = (1-t)^2 S + 2t(1-t) C + t^2 T from a student S (aligned to T with the exact network
symmetries) to T; the control point C (started at the straight line's midpoint) is fitted by Levenberg-Marquardt over
K values of t, stratified with a fresh random jitter at every iteration (Garipov et al. 2018 sample t uniformly; the
jitter keeps the fit from hiding peaks between fixed t values). The residual at t is the energy residual of theta(t)
normalised by each torus's mean energy (--norm mean, the normalisation of the plotted error eps, bounded where the map
is far off) or by the toy energy nu0 J as in training (--norm toy, which explodes on the inner tori away from the
endpoints). --objective log minimises sum_t log |r(t)|^2 (Gauss-Newton with each t weighted by 1/|r(t)|^2, so every
point of the curve counts equally); --objective mean minimises the mean of |r(t)|^2 (dominated by the worst t).
After the fit, eps is profiled along the curve on --fine points: a maximum near the endpoints' level means S and T lie
in one connected low-error valley; a maximum near the straight line's means the pits are separated, at least for this
family of curves.

    python landscape_bezier.py [--student NAME] [--gauge min_shift] [--k 16] [--iters 400] [--wall 3000] [--device cuda]
        a 16 x 64 student of the MW teacher (landscape_lab), -> RES/bezier_<student>_<gauge>_<objective>.json
    python landscape_bezier.py --mini L W SEED [...]      a same-size student of the mini test bed (mini_lab, $MINI_RES)
    python landscape_bezier.py --check        the device Jacobian (--norm toy) against gauge_lab.energy_jacobian at A
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

F64 = torch.float64


class Fit:
    """Energy residual r(v) and its Jacobian for a flat parameter vector v of `m`, on `device`."""

    def __init__(self, m, T, omega, aa, es, device, chunk, norm):
        self.dev, self.chunk, self.norm = torch.device(device), chunk, norm
        self.m, self.T = copy.deepcopy(m).to(self.dev), copy.deepcopy(T).to(self.dev)
        self.w = float(omega)
        self.aa, self.es = aa.to(self.dev), es.to(self.dev)
        self.nO, self.nS = aa.shape[:2]
        self.names = [n for n, _ in self.m.named_parameters()]
        self.shapes = [p.shape for p in self.m.parameters()]

    def H(self, y):  # H_T(y) = H_toy(T^-1(y)), as core.Teacher.H
        z = self.T.inverse(y)
        return 0.5 * (z[:, 1] ** 2 + (self.w * z[:, 0]) ** 2)

    def params(self, v):
        out, i = {}, 0
        v = v.to(self.dev)
        for n, s in zip(self.names, self.shapes):
            k = int(np.prod(s)); out[n] = v[i:i + k].view(s); i += k
        return out

    def _r(self, Hk, es):
        mH = Hk.mean(1, keepdim=True)
        scale = es if self.norm == "toy" else mH.abs()
        return (Hk - mH) / scale / math.sqrt(self.nO * self.nS)

    def resid(self, v):
        p = self.params(v)
        with torch.no_grad():
            H = self.H(functional_call(self.m, p, (self.aa.reshape(-1, 2),))).view(self.nO, self.nS)
        return self._r(H, self.es).reshape(-1)

    def jac(self, v):
        p = self.params(v); sN = math.sqrt(self.nO * self.nS)
        f_point = lambda q, x: functional_call(self.m, q, (x[None, :],))[0]
        rows, res = [], []
        for o0 in range(0, self.nO, self.chunk):
            x = self.aa[o0:o0 + self.chunk].reshape(-1, 2); k = x.shape[0] // self.nS
            Jd = vmap(jacrev(f_point), in_dims=(None, 0))(p, x)
            Jy = torch.cat([Jd[n].reshape(x.shape[0], 2, -1) for n in self.names], -1)
            with torch.no_grad():
                y = functional_call(self.m, p, (x,))
            y = y.detach().requires_grad_(True)
            Hy = self.H(y)
            dHdy = torch.autograd.grad(Hy.sum(), y)[0]
            dH = (dHdy[:, :, None] * Jy).sum(1).view(k, self.nS, -1)
            Hk = Hy.detach().view(k, self.nS); es = self.es[o0:o0 + k]
            dm = dH.mean(1, keepdim=True); mH = Hk.mean(1, keepdim=True)
            if self.norm == "toy":
                J = (dH - dm) / es[..., None] / sN
            else:  # d[(H - m)/|m|] = (dH - dm)/|m| - (H - m) sign(m) dm / m^2
                J = ((dH - dm) / mH.abs()[..., None] - ((Hk - mH) * torch.sign(mH) / mH ** 2)[..., None] * dm) / sN
            rows.append(J.reshape(k * self.nS, -1))
            res.append(self._r(Hk, es).reshape(-1))
        return torch.cat(rows), torch.cat(res)


def run(S, T, fit, eps_fn, nT, a, path, label):
    dev = fit.dev
    Sd, Td = S.to(dev), T.to(dev)
    P = S.numel()
    bez = lambda C, t: (1 - t) ** 2 * Sd + 2 * t * (1 - t) * C + t ** 2 * Td
    t_fine = torch.linspace(0, 1, a.fine, dtype=F64)

    def profile(C, n=None):
        ts = t_fine if n is None else torch.linspace(0, 1, n, dtype=F64)
        return np.asarray(eps_fn(torch.stack([bez(C, float(t)).cpu() for t in ts])))

    C = 0.5 * (Sd + Td)
    e_line = profile(C)
    print(f"{label}: |S-T|/|T| {float((S - T).norm()) / nT:.3f}; straight line max eps {e_line.max():.3e} at t = "
          f"{float(t_fine[e_line.argmax()]):.3f}; endpoints {e_line[0]:.2e}, {e_line[-1]:.2e}", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    K, lam, t0, hist = a.k, a.lam0, time.perf_counter(), []
    for it in range(1, a.iters + 1):
        ts = [(i + float(torch.rand(1, generator=g))) / K for i in range(K)]
        A = torch.zeros(P, P, dtype=F64, device=dev); gr = torch.zeros(P, dtype=F64, device=dev); fk = []
        for t in ts:
            Jm, r = fit.jac(bez(C, t)); b = 2 * t * (1 - t); f = float(r.pow(2).sum()); fk.append(f)
            wt = 1.0 / f if a.objective == "log" else 1.0
            A += wt * b * b * (Jm.T @ Jm); gr += wt * b * (Jm.T @ r)
        obj = (lambda fs: sum(math.log(x) for x in fs) / K) if a.objective == "log" else (lambda fs: sum(fs) / K)
        f0 = obj(fk)
        D = torch.diagonal(A).clone(); D = D + 1e-12 * D.max()
        acc = False
        for _ in range(40):
            Lc, info = torch.linalg.cholesky_ex(A + lam * torch.diag(D))
            if int(info) != 0:
                lam *= 10; continue
            d = -torch.cholesky_solve(gr[:, None], Lc)[:, 0]
            fn = obj([float(fit.resid(bez(C + d, t)).pow(2).sum()) for t in ts])
            if math.isfinite(fn) and fn < f0:
                C, acc = C + d, True; lam = max(lam / 3, 1e-15); break
            lam *= 4
        hist.append(dict(it=it, obj=f0, obj_new=fn if acc else None, worst_t_loss=max(fk), lam=lam, t=time.perf_counter() - t0))
        if it % 10 == 0 or not acc:
            e = profile(C, 21)
            hist[-1]["max_eps_21"] = float(e.max())
            print(f"  it {it:4d}: objective {f0:.4e} -> {fn:.4e}  lam {lam:.1e}  max eps (21 pts) {e.max():.3e}  "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if not acc or time.perf_counter() - t0 > a.wall:
            break
    e_bz = profile(C)
    Cc = C.cpu()
    out = dict(label=label, objective=a.objective, norm=a.norm, K=K, iters=len(hist), dist_S_T=float((S - T).norm()) / nT,
               dist_C_mid=float((Cc - 0.5 * (S + T)).norm()) / nT, t_fine=t_fine.tolist(),
               line=dict(eps=e_line.tolist(), max=float(e_line.max()), t_at_max=float(t_fine[e_line.argmax()])),
               bezier=dict(eps=e_bz.tolist(), max=float(e_bz.max()), t_at_max=float(t_fine[e_bz.argmax()])),
               endpoints=dict(S=float(e_line[0]), T=float(e_line[-1])), hist=hist, wall_s=time.perf_counter() - t0)
    json.dump(out, open(path, "w"), indent=1)
    np.save(path.replace(".json", "_C.npy"), Cc.numpy())
    print(f"Bezier: max eps {e_bz.max():.3e} at t = {out['bezier']['t_at_max']:.3f} (straight line {e_line.max():.3e}); "
          f"|C - mid|/|T| {out['dist_C_mid']:.3f}; wrote {path}", flush=True)


def big(a):
    import landscape_lab as LL
    L = LL.load_all()
    name = a.student or L.A
    S = dict(min_shift=L.al_min_shift, min=L.al_min, gate=L.al, raw=L.raw)[a.gauge][name].to(F64)
    fit = Fit(L.m, L.t.T, L.t.omega, L.aa, L.es, a.device, a.chunk, a.norm)
    run(S, L.vT.to(F64), fit, LL.eps_batch, L.nT, a, os.path.join(LL.RES, f"bezier_{name}_{a.gauge}_{a.objective}_{a.norm}.json"),
        f"{name} ({a.gauge})")


def mini(a):
    import mini_lab as ML
    L, W, seed = a.mini
    jd = os.path.join(ML.RES, "students", f"{L}x{W}", f"s{seed}")
    bed = ML.Bed(ML.make_cfg("bz", "bz", L, W, 0, teacher=ML.teacher_file(L, W)))
    m = bed.build(); lay = ML.Layout(L, W)
    vT = torch.cat([p.detach().reshape(-1) for p in bed.t.T.parameters()])
    m.load_state_dict(torch.load(os.path.join(jd, "state.pt"), weights_only=False)["state"])
    S = ML.align(lay, ML.get_flat(m), vT, "min", True)

    def eps_fn(V):
        out = []
        for v in V:
            ML.set_flat(m, v); out.append(bed.heldout(m)[0])
        return out
    fit = Fit(m, bed.t.T, bed.t.omega, bed.aa, bed.es, a.device, a.chunk, a.norm)
    run(S, vT, fit, eps_fn, float(vT.norm()), a, os.path.join(ML.RES, f"bezier_{L}x{W}_s{seed}_{a.objective}_{a.norm}.json"),
        f"mini {L}x{W} s{seed}")


def check():
    import landscape_lab as LL
    L = LL.load_all()
    v = L.al_min_shift[L.A]
    J0, r0 = LL.energy_jac(v)
    J1, r1 = Fit(L.m, L.t.T, L.t.omega, L.aa, L.es, "cpu", 16, "toy").jac(v)
    print(f"Jacobian rel. diff {float((J1 - J0).norm() / J0.norm()):.2e}; residual rel. diff {float((r1 - r0).norm() / r0.norm()):.2e}")
    fm = Fit(L.m, L.t.T, L.t.omega, L.aa, L.es, "cpu", 16, "mean")  # mean-normalised Jacobian by finite differences
    Jm, _ = fm.jac(v); d = torch.randn(v.numel(), dtype=F64, generator=torch.Generator().manual_seed(1)); d /= d.norm()
    h = 1e-7 * float(v.norm())
    fd = (fm.resid(v + h * d) - fm.resid(v - h * d)) / (2 * h)
    print(f"mean-norm Jacobian vs central difference: rel. diff {float((Jm @ d - fd).norm() / fd.norm()):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=None); ap.add_argument("--gauge", default="min_shift", choices=["min_shift", "min", "gate", "raw"])
    ap.add_argument("--mini", type=int, nargs=3, default=None, metavar=("L", "W", "SEED"))
    ap.add_argument("--objective", default="log", choices=["log", "mean"]); ap.add_argument("--norm", default="mean", choices=["mean", "toy"])
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--iters", type=int, default=400); ap.add_argument("--wall", type=float, default=3000)
    ap.add_argument("--fine", type=int, default=61); ap.add_argument("--lam0", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--chunk", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    (mini if a.mini else big)(a)


if __name__ == "__main__":
    main()
