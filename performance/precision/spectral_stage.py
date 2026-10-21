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
        th = th.detach().requires_grad_(True); J = J.detach().requires_grad_(True)
        dth, dJ = torch.autograd.grad(self.S(th, J).sum(), (th, J))
        return dth.detach(), dJ.detach()

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--harmonics", type=int, default=48); ap.add_argument("--degree", type=int, default=40)
    ap.add_argument("--stage-orbits", type=int, default=None, help="tori used to fit the stages (default: the job's)")
    ap.add_argument("--stage-angles", type=int, default=None, help="angles per torus used to fit the stages (default: the job's)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.job, "config.json")))
    t = C.make_target(cfg); d = cfg["data"]
    if t.name == "mw":
        J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, t.phi)[0]); J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, t.phi)[0])
    else:
        J_lo, J_hi = d["J_lo"], d["J_hi"]
    tw = cfg["toy"].get("omega", "nu0"); w0 = t.nu0 if tw == "nu0" else float(tw)
    base = C.build_model(cfg["model"], cfg["toy"], w0, J_lo, J_hi, cfg.get("seed", 0))
    base.load_state_dict(torch.load(os.path.join(a.job, "state.pt"), weights_only=False)["state"])
    assert cfg["toy"]["mode"] == "fixed", "fixed toy only (see the docstring)"
    a0 = d.get("a0", 0.03)
    J_tr = C.train_grid(d, J_lo, J_hi, w0)
    nS = d["n_steps"]; th_tr = C.angles(nS, 0.5)
    ev = C.Evaluator(cfg, t, J_lo, J_hi, w0, J_tr, nS)
    if a.stage_orbits:  # the energy loss needs no ground truth, so the stages can be fitted on any tori
        J_tr = C.j_grid(a.stage_orbits, J_lo, J_hi, w0, d.get("spacing", "asinh"), a0, offset=0.0)
    nO = len(J_tr)
    if a.stage_angles:
        nS = a.stage_angles; th_tr = C.angles(nS, 0.5)
    uf = lambda J: torch.asinh(torch.sqrt(2 * J / w0) / a0)
    u_lo, u_hi = float(uf(torch.tensor(J_lo, dtype=F64))), float(uf(torch.tensor(J_hi, dtype=F64)))
    s_of_J = lambda J: 2 * (uf(J) - u_lo) / (u_hi - u_lo) - 1
    stages = []

    def model(aa, need_grad=False):
        for g in reversed(stages):   # newest stage acts first on the input
            aa = g(aa)
        with torch.set_grad_enabled(need_grad):
            return base(aa.reshape(-1, 2)).reshape(*aa.shape[:-1], 2)

    BANDS = [(0.005, 0.025), (0.025, 0.15), (0.15, 1), (1, 3), (3, 9)]

    def bands():
        H = t.H(model(ev.aa)); e = ((H - H.mean(-1, keepdim=True)) / H.mean(-1, keepdim=True).abs()).abs()
        em, ex = e.mean(-1).reshape(-1), e.max(-1).values.reshape(-1); zz = torch.as_tensor(ev.z, dtype=F64).reshape(-1)
        return "  ".join(f"{lo:g}-{hi:g} kpc {float(em[(zz >= lo) & (zz < hi)].mean()):.1e} (max {float(ex[(zz >= lo) & (zz < hi)].max()):.0e})" for lo, hi in BANDS)

    def errors():
        H = t.H(model(ev.aa)); m = H.mean(-1, keepdim=True)
        e = ((H - m) / m.abs()).abs(); et = ((H - ev.E_true) / ev.E_true).abs()
        Ht = t.H(model(C.aa_points(J_tr, th_tr))); mt = Ht.mean(-1, keepdim=True)
        return float(e.mean()), float(e.max()), float(((Ht - mt) / mt.abs()).abs().mean()), float(et.mean())

    t0 = time.perf_counter(); rows = []
    e = errors(); rows.append(dict(round=0, held_out=e[0], held_out_max=e[1], train=e[2], vs_true_E=e[3], wall_s=0.0))
    print(f"{cfg['name']}  base      held-out {e[0]:.2e} (max {e[1]:.1e})  train {e[2]:.2e}  vs true E {e[3]:.2e}", flush=True)
    print("    bands: " + bands(), flush=True)
    for r in range(a.rounds):
        # residual of the current model on the training grid, in its input coordinates
        Jg = J_tr.clone().requires_grad_(True)
        H = t.H(model(C.aa_points(Jg, th_tr), need_grad=True))
        Hm = H.mean(-1)
        Om = torch.autograd.grad(Hm.sum(), Jg)[0].detach()               # dE/dJ of the current model
        H = H.detach()
        c = torch.fft.rfft(H - H.mean(-1, keepdim=True), dim=-1) / nS     # c_n, n = 0..nS/2
        nh = min(a.harmonics, nS // 2 - 1)
        n = torch.arange(1, nh + 1, dtype=F64)
        # rfft is relative to the sample angles th_tr = 2 pi (k + 0.5)/nS: undo that phase
        cn = c[:, 1:nh + 1] * torch.exp(-1j * n * th_tr[0])
        sn = -cn / (1j * n * Om[:, None])                                # (nO, nh)
        s = s_of_J(J_tr)
        T = [torch.ones_like(s), s]
        for _ in range(2, a.degree):
            T.append(2 * s * T[-1] - T[-2])
        T = torch.stack(T, -1)                                           # (nO, K)
        # fit s_n / J, so every torus counts by its relative error (J spans ~6.5 decades on MW)
        coef = torch.linalg.lstsq(T.to(torch.complex128), sn / J_tr[:, None]).solution.T  # (nh, K)
        stages.append(GenStage(coef, s_of_J, nh))
        e = errors(); dt = time.perf_counter() - t0
        rows.append(dict(round=r + 1, held_out=e[0], held_out_max=e[1], train=e[2], vs_true_E=e[3], wall_s=round(dt, 1)))
        print(f"{'':16s}  round {r + 1}   held-out {e[0]:.2e} (max {e[1]:.1e})  train {e[2]:.2e}  vs true E {e[3]:.2e}  t={dt:.1f}s", flush=True)
    print("    bands: " + bands(), flush=True)
    # invertibility and canonicity of the corrected input map (theta', J') -> (theta, J), on held-out points
    aa = ev.aa[::7, ::16].reshape(-1, 2)
    fwd = aa
    for g in reversed(stages):
        fwd = g(fwd)
    back = fwd
    for g in stages:
        back = g.inverse(back)
    d_th = ((back[:, 0] - aa[:, 0] + math.pi) % (2 * math.pi) - math.pi).abs().max()
    d_J = ((back[:, 1] - aa[:, 1]) / aa[:, 1]).abs().max()
    print(f"round trip forward then inverse, {len(aa)} points: max |d theta| {float(d_th):.1e}, max |dJ/J| {float(d_J):.1e}")
    def fmap(x):
        for g in reversed(stages):
            x = g(x)
        return x
    dets = torch.stack([torch.det(torch.autograd.functional.jacobian(lambda z: fmap(z[None])[0], x)) for x in aa[::4]])
    print(f"canonicity: max |det d(theta,J)/d(theta',J') - 1| over {len(dets)} points = {float((dets - 1).abs().max()):.1e}; "
          f"size of the correction: max |J - J'|/J' = {float(((fwd[:, 1] - aa[:, 1]) / aa[:, 1]).abs().max()):.1e}")
    if a.out:
        json.dump(dict(job=cfg["name"], harmonics=a.harmonics, degree=a.degree, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
