"""Curved-path test on the 16x64 MW-teacher test bed: is a stalled student joined to the teacher T by a low-error curve?

A quadratic Bezier curve theta(t) = (1-t)^2 S + 2t(1-t) C + t^2 T from a student S (aligned to T with the exact network
symmetries, landscape_lab gauges) to T; the control point C (3104 parameters, started at the straight line's midpoint)
is fitted by Levenberg-Marquardt to the mean LM training loss over K values of t, stratified with a fresh random
jitter at every iteration (Garipov et al. 2018 sample t uniformly; jitter keeps the fit from hiding peaks between fixed
t values). The residual of the curve at t is the training energy residual of theta(t), so its Jacobian with respect to
C is 2t(1-t) J_E(theta(t)). After the fit, eps (landscape_lab's metric, as in the landscape figures) is profiled along
the curve on --fine points: a maximum near the endpoints' level means S and T lie in one connected low-error valley; a
maximum near the straight line's (0.6-0.7) means the pits are separated, at least for this family of curves.

    python landscape_bezier.py [--student NAME] [--gauge min_shift] [--k 16] [--iters 400] [--wall 3000] [--device cuda]
    python landscape_bezier.py --check        the device Jacobian against gauge_lab.energy_jacobian at A (CPU)
Writes RES/bezier_<student>_<gauge>.json (profiles of the line and the curve, history) and _C.npy.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

import landscape_lab as LL

F64 = torch.float64


class Fit:
    """Training residual r(v) and Jacobian J_E(v) of a 16x64 parameter vector on `device` (LM's normalisation)."""

    def __init__(self, device, chunk):
        L = LL.load_all()
        self.dev, self.chunk = torch.device(device), chunk
        import copy
        self.m = copy.deepcopy(L.m).to(self.dev)
        self.T = copy.deepcopy(L.t.T).to(self.dev)
        self.w = float(L.t.omega)
        self.aa, self.es = L.aa.to(self.dev), L.es.to(self.dev)
        self.nO, self.nS = L.nO, L.nS
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

    def resid(self, v):
        p = self.params(v); sN = math.sqrt(self.nO * self.nS)
        with torch.no_grad():
            y = functional_call(self.m, p, (self.aa.reshape(-1, 2),))
            H = self.H(y).view(self.nO, self.nS)
        return ((H - H.mean(1, keepdim=True)) / self.es / sN).reshape(-1)

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
            es = self.es[o0:o0 + k]
            rows.append(((dH - dH.mean(1, keepdim=True)) / es[..., None] / sN).reshape(k * self.nS, -1))
            Hk = Hy.detach().view(k, self.nS)
            res.append(((Hk - Hk.mean(1, keepdim=True)) / es / sN).reshape(-1))
        return torch.cat(rows), torch.cat(res)


def check():
    L = LL.load_all()
    v = L.al_min_shift[L.A]
    J0, r0 = LL.energy_jac(v)
    J1, r1 = Fit("cpu", 16).jac(v)
    print(f"Jacobian rel. diff {float((J1 - J0).norm() / J0.norm()):.2e}; residual rel. diff {float((r1 - r0).norm() / r0.norm()):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=None); ap.add_argument("--gauge", default="min_shift", choices=["min_shift", "min", "gate", "raw"])
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--iters", type=int, default=400); ap.add_argument("--wall", type=float, default=3000)
    ap.add_argument("--fine", type=int, default=61); ap.add_argument("--lam0", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--chunk", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    L = LL.load_all()
    name = a.student or L.A
    S = dict(min_shift=L.al_min_shift, min=L.al_min, gate=L.al, raw=L.raw)[a.gauge][name].to(F64)
    T = L.vT.to(F64)
    fit = Fit(a.device, a.chunk)
    dev = fit.dev
    Sd, Td = S.to(dev), T.to(dev)
    bez = lambda C, t: (1 - t) ** 2 * Sd + 2 * t * (1 - t) * C + t ** 2 * Td
    t_fine = torch.linspace(0, 1, a.fine, dtype=F64)

    def profile(C, n=None):
        ts = t_fine if n is None else torch.linspace(0, 1, n, dtype=F64)
        V = torch.stack([bez(C, float(t)).cpu() for t in ts])
        return ts.numpy(), LL.eps_batch(V).numpy()

    C = 0.5 * (Sd + Td)
    _, e_line = profile(C)
    print(f"{name} ({a.gauge}): |S-T|/|T| {float((S - T).norm()) / L.nT:.3f}; straight line max eps {e_line.max():.3e} "
          f"at t = {float(t_fine[e_line.argmax()]):.3f}; endpoints {e_line[0]:.2e}, {e_line[-1]:.2e}", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    K, lam, t0, hist = a.k, a.lam0, time.perf_counter(), []
    for it in range(1, a.iters + 1):
        ts = [(i + float(torch.rand(1, generator=g))) / K for i in range(K)]
        F = lambda C_: sum(float(fit.resid(bez(C_, t)).pow(2).sum()) for t in ts) / K
        A = torch.zeros(LL.P, LL.P, dtype=F64, device=dev); gr = torch.zeros(LL.P, dtype=F64, device=dev); f = 0.0
        for t in ts:
            Jm, r = fit.jac(bez(C, t)); b = 2 * t * (1 - t)
            A += b * b * (Jm.T @ Jm); gr += b * (Jm.T @ r); f += float(r.pow(2).sum())
        A /= K; gr /= K; f /= K
        D = torch.diagonal(A).clone(); D = D + 1e-12 * D.max()
        acc = False
        for _ in range(40):
            Lc, info = torch.linalg.cholesky_ex(A + lam * torch.diag(D))
            if int(info) != 0:
                lam *= 10; continue
            d = -torch.cholesky_solve(gr[:, None], Lc)[:, 0]
            fn = F(C + d)
            if math.isfinite(fn) and fn < f:
                C, acc = C + d, True; lam = max(lam / 3, 1e-15); break
            lam *= 4
        hist.append(dict(it=it, f=f, f_new=fn if acc else None, lam=lam, t=time.perf_counter() - t0))
        if it % 10 == 0 or not acc:
            _, e = profile(C, 21)
            hist[-1]["max_eps_21"] = float(e.max())
            print(f"  it {it:4d}: mean loss over t {f:.3e} -> {fn:.3e}  lam {lam:.1e}  max eps (21 pts) {e.max():.3e}  "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if not acc or time.perf_counter() - t0 > a.wall:
            break
    _, e_bz = profile(C)
    Cc = C.cpu()
    out = dict(student=name, gauge=a.gauge, K=K, iters=len(hist), dist_S_T=float((S - T).norm()) / L.nT,
               dist_C_mid=float((Cc - 0.5 * (S + T)).norm()) / L.nT, t_fine=t_fine.tolist(),
               line=dict(eps=e_line.tolist(), max=float(e_line.max()), t_at_max=float(t_fine[e_line.argmax()])),
               bezier=dict(eps=e_bz.tolist(), max=float(e_bz.max()), t_at_max=float(t_fine[e_bz.argmax()])),
               endpoints=dict(S=float(e_line[0]), T=float(e_line[-1])), hist=hist, wall_s=time.perf_counter() - t0)
    path = os.path.join(LL.RES, f"bezier_{name}_{a.gauge}.json")
    json.dump(out, open(path, "w"), indent=1)
    np.save(path.replace(".json", "_C.npy"), Cc.numpy())
    print(f"Bezier: max eps {e_bz.max():.3e} at t = {out['bezier']['t_at_max']:.3f} (straight line {e_line.max():.3e}); "
          f"|C - mid|/|T| {out['dist_C_mid']:.3f}; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
