"""Write the halo-benchmark configs (configs/halo): isochrone toy -> NFW halo and -> MWPotential2014.

One action box for both targets, in galpy's natural units (R0 = 8 kpc, v0 = 220 km/s, so 1 = 1760 kpc km/s):
J_r 0.01-0.5, L 0.2-2, |cos i| <= 0.9. In the NFW that spans pericentres from 0.9 kpc to apocentres of 27 kpc. The toy
isochrone (GM, b) is the least-squares fit to each target's rotation curve (angle-averaged radial force for the
axisymmetric one) over 0.8-36 kpc. Stage 1 trains a 16-layer, 32-wide flow from scratch; stage 2 grows it to 128 wide.
configs/halo_geir holds stage 1 for CPUs under the same run names, so an H100 job can pick up its checkpoints."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import math
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core3d as C  # noqa: E402

F64 = torch.float64
BOX = dict(kind="spherical", n_per_axis=5, Jr=[0.01, 0.5], L=[0.2, 2.0], cosi_max=0.9, n_angles=256)
R_FIT = (0.1, 4.5)


def vc2_target(t, r, n_mu=32):
    """r <dPhi/dr> averaged over directions (Gauss-Legendre in cos theta)."""
    mu, w = C._gl(n_mu)
    rr = r[:, None].expand(-1, n_mu).clone().requires_grad_(True); st = torch.sqrt(1 - mu * mu)
    x = torch.stack([rr * st, 0 * rr, rr * mu], -1)
    g = torch.autograd.grad(t.pot(x).sum(), rr)[0]
    return r * (g * w).sum(-1) / 2


def fit_isochrone(t, steps=4000):
    r = torch.exp(torch.linspace(math.log(R_FIT[0]), math.log(R_FIT[1]), 200, dtype=F64))
    y = torch.log(vc2_target(t, r)).detach()
    p = torch.zeros(2, dtype=F64, requires_grad=True)  # log GM, log b
    opt = torch.optim.Adam([p], lr=0.02)
    for _ in range(steps):
        GM, b = torch.exp(p[0]), torch.exp(p[1]); a = torch.sqrt(b * b + r * r)
        loss = (torch.log(GM * r * r / ((b + a) ** 2 * a)) - y).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(torch.exp(p[0])), float(torch.exp(p[1])), math.sqrt(float(loss))


def main():
    out = os.path.join(HERE, "configs", "halo"); os.makedirs(out, exist_ok=True)
    fits = {}
    for key, spec in (("nfw", dict(kind="nfw", a=2.0, normalize=1.0)), ("mw", dict(kind="mw2014"))):
        GM, b, rms = fit_isochrone(C.make_target(spec)); fits[key] = dict(GM=GM, b=b, rms_ln_vc2=rms)
        toy = dict(kind="isochrone", GM=round(GM, 6), b=round(b, 6))
        for s in (0, 1):
            base = dict(toy=toy, data=BOX, target=spec, seed=s, threads=24, device="cuda")
            lm = dict(invariance_beta=0.0, inv_angles=64, geodesic=True, every=10, ckpt_every=50, lam0=1e-3, plateau=[500, 0.01])
            n1 = f"halo_{key}_w32_s{s}"
            json.dump(dict(base, name=n1, model=dict(n_lin=2, n_blocks=8, width=32, gate_std=1e-3), adam=dict(steps=1000, lr=1e-3),
                           lm=dict(lm, max_iters=20000, wall_hours=0.75, chunk=128)), open(os.path.join(out, n1 + ".json"), "w"), indent=1)
            geir = os.path.join(HERE, "configs", "halo_geir"); os.makedirs(geir, exist_ok=True)  # same run on CPUs: the H100 job resumes it
            json.dump(dict(base, name=n1, device="cpu", threads=4, model=dict(n_lin=2, n_blocks=8, width=32, gate_std=1e-3), adam=dict(steps=1000, lr=1e-3),
                           lm=dict(lm, max_iters=20000, wall_hours=68.0, chunk=4, ckpt_every=10)), open(os.path.join(geir, n1 + ".json"), "w"), indent=1)
            n2 = f"halo_{key}_w128_s{s}"
            json.dump(dict(base, name=n2, model=dict(n_lin=2, n_blocks=8, width=128, gate_std=1e-3), adam=dict(steps=0),
                           init_from=n1, grow=True, lm=dict(lm, max_iters=20000, wall_hours=1.0, chunk=64)),
                      open(os.path.join(out, n2 + ".json"), "w"), indent=1)
    json.dump(fits, open(os.path.join(out, "toy_fits.json"), "w"), indent=1)
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
