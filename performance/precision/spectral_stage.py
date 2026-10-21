"""Diagnostic: refine a trained map with exactly canonical generating-function stages (classical torus-mapping
correction), to see how far and how fast the remaining error can be removed when the correction is expressed
in a basis suited to it.

    python spectral_stage.py <job dir> [--rounds 4] [--harmonics 48] [--degree 40] [--stage-orbits N]

Model after k stages: (theta', J') -> G_k -> ... -> G_1 -> base map. Each G is the canonical map of the type-2
generating function F(theta, J') = theta J' + S(theta, J'):  J = J' + dS/dtheta,  theta' = theta + dS/dJ'
(theta found from theta' by fixed-point iteration, to round-off). S(theta, J) = sum_n 2 Re[s_n(J) e^{i n theta}]
with s_n(J) = J times a Chebyshev series in the asinh amplitude coordinate. First-order fit, per training torus:
Omega dS/dtheta = -(H - <H>)  =>  s_n = -c_n / (i n Omega), then least squares across tori for the J dependence.
Ported from the multistage branch to the precision-optim core (training grid from core.train_grid). Fixed toy only:
a learned toy's angle is not the canonical one, so its torus averages and harmonics would need the angle weights.
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

F64 = torch.float64


class GenStage:
    def __init__(self, coef, s_of_J, nh):
        self.coef, self.s_of_J, self.nh = coef, s_of_J, nh  # coef: complex (nh, K)

    def S(self, th, J):
        s = self.s_of_J(J)
        T = [torch.ones_like(s), s]
        for _ in range(2, self.coef.shape[1]):
            T.append(2 * s * T[-1] - T[-2])
        T = torch.stack(T, -1).to(self.coef.dtype)                       # (..., K)
        sn = J[..., None] * (T @ self.coef.T)                            # (..., nh) complex; s_n(J) = J x series
        n = torch.arange(1, self.nh + 1, dtype=F64)
        return 2 * (sn * torch.exp(1j * n * th[..., None])).real.sum(-1)

    def grads(self, th, J):
        with torch.enable_grad():
            th = th.detach().requires_grad_(True); J = J.detach().requires_grad_(True)
            dth, dJ = torch.autograd.grad(self.S(th, J).sum(), (th, J))
        return dth.detach(), dJ.detach()

    @property
    def n_coef(self):
        return 2 * self.coef.numel()  # real coefficients

    def inverse(self, aa):
        """(theta, J) of the previous model -> (theta', J'): solve J' = J - dS/dtheta(theta, J'), then
        theta' = theta + dS/dJ'(theta, J')."""
        th, J = aa[..., 0], aa[..., 1]
        Jp = J.clone()
        for _ in range(6):
            dth, _ = self.grads(th, Jp)
            Jp = J - dth
        _, dJ = self.grads(th, Jp)
        return torch.stack([th + dJ, Jp], -1)

    def __call__(self, aa):
        """(theta', J') -> (theta, J) of the previous model."""
        thp, Jp = aa[..., 0], aa[..., 1]
        th = thp.clone()
        for _ in range(6):
            _, dJ = self.grads(th, Jp)
            th = thp - dJ
        dth, _ = self.grads(th, Jp)
        return torch.stack([th, Jp + dth], -1)


BANDS = [(0.005, 0.025), (0.025, 0.15), (0.15, 1), (1, 3), (3, 9)]


def load(job):
    """A trained fixed-toy job: its config, target, base model, ranges, training grid and held-out evaluator."""
    cfg = json.load(open(os.path.join(job, "config.json")))
    t = C.make_target(cfg); d = cfg["data"]
    if t.name == "mw":
        J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, t.phi)[0]); J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, t.phi)[0])
    else:
        J_lo, J_hi = d["J_lo"], d["J_hi"]
    tw = cfg["toy"].get("omega", "nu0"); w0 = t.nu0 if tw == "nu0" else float(tw)
    assert cfg["toy"]["mode"] == "fixed", "fixed toy only (see the docstring)"
    base = C.build_model(cfg["model"], cfg["toy"], w0, J_lo, J_hi, cfg.get("seed", 0))
    base.load_state_dict(torch.load(os.path.join(job, "state.pt"), weights_only=False)["state"])
    J_train = C.train_grid(d, J_lo, J_hi, w0)
    ev = C.Evaluator(cfg, t, J_lo, J_hi, w0, J_train, d["n_steps"])
    return dict(cfg=cfg, t=t, d=d, J_lo=J_lo, J_hi=J_hi, w0=w0, a0=d.get("a0", 0.03), base=base, J_train=J_train, ev=ev)


class Corrected:
    """The base model behind a stack of generating-function stages: (theta', J') -> stages -> base -> (q, p)."""
    def __init__(self, base, stages):
        self.base, self.stages, self.toy = base, stages, base.toy

    def __call__(self, aa, need_grad=False):
        shape = aa.shape[:-1]; x = aa.reshape(-1, 2)
        if not need_grad and len(x) > 32768:  # evaluate in chunks: the stages' (points x harmonics) arrays are large
            return torch.cat([self._apply(c, False) for c in x.split(32768)]).reshape(*shape, 2)
        return self._apply(x, need_grad).reshape(*shape, 2)

    def _apply(self, x, need_grad):
        for g in reversed(self.stages):   # newest stage acts first on the input
            x = g(x)
        with torch.set_grad_enabled(need_grad):
            return self.base(x)


def fit_round(ctx, model, J_fit, th_fit, harmonics, degree):
    """One generating-function stage fitted to the current model's energy residual on (J_fit x th_fit)."""
    t, w0, a0, nS = ctx["t"], ctx["w0"], ctx["a0"], len(th_fit)
    uf = lambda J: torch.asinh(torch.sqrt(2 * J / w0) / a0)
    u_lo, u_hi = float(uf(torch.tensor(ctx["J_lo"], dtype=F64))), float(uf(torch.tensor(ctx["J_hi"], dtype=F64)))
    s_of_J = lambda J: 2 * (uf(J) - u_lo) / (u_hi - u_lo) - 1
    Jg = J_fit.clone().requires_grad_(True)
    H = t.H(model(C.aa_points(Jg, th_fit), need_grad=True))
    Om = torch.autograd.grad(H.mean(-1).sum(), Jg)[0].detach()            # dE/dJ of the current model
    H = H.detach()
    c = torch.fft.rfft(H - H.mean(-1, keepdim=True), dim=-1) / nS         # c_n, n = 0..nS/2
    nh = min(harmonics, nS // 2 - 1)
    n = torch.arange(1, nh + 1, dtype=F64)
    # rfft is relative to the sample angles th = 2 pi (k + 0.5)/nS: undo that phase
    cn = c[:, 1:nh + 1] * torch.exp(-1j * n * th_fit[0])
    sn = -cn / (1j * n * Om[:, None])                                    # (nO, nh)
    s = s_of_J(J_fit)
    T = [torch.ones_like(s), s]
    for _ in range(2, degree):
        T.append(2 * s * T[-1] - T[-2])
    T = torch.stack(T, -1)                                               # (nO, K)
    # fit s_n / J, so every torus counts by its relative error (J spans ~6.5 decades on MW)
    coef = torch.linalg.lstsq(T.to(torch.complex128), sn / J_fit[:, None]).solution.T  # (nh, K)
    return GenStage(coef, s_of_J, nh)


def fit(ctx, rounds=1, harmonics=48, degree=40, stage_orbits=None, stage_angles=None, log=None):
    """Fit `rounds` stages. The energy loss needs no ground truth, so the stages can be fitted on any tori and
    angles (default: the job's own training grid). Returns the corrected model and the fitting grid."""
    d = ctx["d"]
    J_fit = (C.j_grid(stage_orbits, ctx["J_lo"], ctx["J_hi"], ctx["w0"], d.get("spacing", "asinh"), ctx["a0"], offset=0.0)
             if stage_orbits else ctx["J_train"])
    th_fit = C.angles(stage_angles or d["n_steps"], 0.5)
    model = Corrected(ctx["base"], [])
    for r in range(rounds):
        model.stages.append(fit_round(ctx, model, J_fit, th_fit, harmonics, degree))
        if log:
            log(r + 1, model)
    return model, J_fit, th_fit


def errors(ctx, model, J_fit=None, th_fit=None):
    """Held-out mean and max relative energy error, the same on the fitting grid, and the mean against the true E."""
    t, ev = ctx["t"], ctx["ev"]
    H = t.H(model(ev.aa)); m = H.mean(-1, keepdim=True)
    e = ((H - m) / m.abs()).abs(); et = ((H - ev.E_true) / ev.E_true).abs()
    out = dict(held_out=float(e.mean()), held_out_max=float(e.max()), vs_true_E=float(et.mean()),
               held_out_p99=float(torch.quantile(e.reshape(-1), 0.99)))
    if J_fit is not None:
        Ht = t.H(model(C.aa_points(J_fit, th_fit))); mt = Ht.mean(-1, keepdim=True)
        out["train"] = float(((Ht - mt) / mt.abs()).abs().mean())
    zz = torch.as_tensor(ctx["ev"].z, dtype=F64).reshape(-1); em, ex = e.mean(-1).reshape(-1), e.max(-1).values.reshape(-1)
    out["bands"] = [dict(lo=lo, hi=hi, mean=float(em[(zz >= lo) & (zz < hi)].mean()), max=float(ex[(zz >= lo) & (zz < hi)].max()))
                    for lo, hi in BANDS]
    return out


def checks(ctx, model):
    """Invertibility, canonicity and size of the corrected input map (theta', J') -> (theta, J), on held-out points."""
    aa = ctx["ev"].aa[::7, ::16].reshape(-1, 2)
    fwd = aa
    for g in reversed(model.stages):
        fwd = g(fwd)
    back = fwd
    for g in model.stages:
        back = g.inverse(back)
    d_th = ((back[:, 0] - aa[:, 0] + math.pi) % (2 * math.pi) - math.pi).abs().max()
    d_J = ((back[:, 1] - aa[:, 1]) / aa[:, 1]).abs().max()

    def fmap(x):
        for g in reversed(model.stages):
            x = g(x)
        return x
    dets = torch.stack([torch.det(torch.autograd.functional.jacobian(lambda z: fmap(z[None])[0], x)) for x in aa[::4]])
    return dict(round_trip_theta=float(d_th), round_trip_J=float(d_J), det_dev=float((dets - 1).abs().max()),
                size=float(((fwd[:, 1] - aa[:, 1]) / aa[:, 1]).abs().max()), n_points=len(aa), n_det=len(dets))


def fmt_bands(b):
    return "  ".join(f"{x['lo']:g}-{x['hi']:g} kpc {x['mean']:.1e} (max {x['max']:.0e})" for x in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--harmonics", type=int, default=48); ap.add_argument("--degree", type=int, default=40)
    ap.add_argument("--stage-orbits", type=int, default=None, help="tori used to fit the stages (default: the job's)")
    ap.add_argument("--stage-angles", type=int, default=None, help="angles per torus used to fit the stages (default: the job's)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = load(a.job); name = ctx["cfg"]["name"]
    J0 = C.j_grid(a.stage_orbits, ctx["J_lo"], ctx["J_hi"], ctx["w0"], ctx["d"].get("spacing", "asinh"), ctx["a0"], offset=0.0) \
        if a.stage_orbits else ctx["J_train"]
    th0 = C.angles(a.stage_angles or ctx["d"]["n_steps"], 0.5)
    t0 = time.perf_counter(); rows = []
    e = errors(ctx, Corrected(ctx["base"], []), J0, th0); rows.append(dict(round=0, wall_s=0.0, **e))
    print(f"{name}  base      held-out {e['held_out']:.2e} (max {e['held_out_max']:.1e})  train {e['train']:.2e}  vs true E {e['vs_true_E']:.2e}", flush=True)
    print("    bands: " + fmt_bands(e["bands"]), flush=True)

    def log(r, model):
        e = errors(ctx, model, J0, th0); dt = time.perf_counter() - t0; rows.append(dict(round=r, wall_s=round(dt, 1), **e))
        print(f"{'':16s}  round {r}   held-out {e['held_out']:.2e} (max {e['held_out_max']:.1e})  train {e['train']:.2e}  vs true E {e['vs_true_E']:.2e}  t={dt:.1f}s", flush=True)
    model, _, _ = fit(ctx, a.rounds, a.harmonics, a.degree, a.stage_orbits, a.stage_angles, log=log)
    print("    bands: " + fmt_bands(rows[-1]["bands"]), flush=True)
    ck = checks(ctx, model)
    print(f"round trip forward then inverse, {ck['n_points']} points: max |d theta| {ck['round_trip_theta']:.1e}, max |dJ/J| {ck['round_trip_J']:.1e}")
    print(f"canonicity: max |det d(theta,J)/d(theta',J') - 1| over {ck['n_det']} points = {ck['det_dev']:.1e}; "
          f"size of the correction: max |J - J'|/J' = {ck['size']:.1e}; {sum(g.n_coef for g in model.stages):,} real coefficients")
    if a.out:
        json.dump(dict(job=name, harmonics=a.harmonics, degree=a.degree, rows=rows, checks=ck), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
