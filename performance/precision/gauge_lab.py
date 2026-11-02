"""Starting states and diagnostics for the twist-gauge tests on the 16-layer MW-teacher test bed.

The energy loss is invariant under twists theta -> theta + g(J), so its exact solutions on the teacher target are
T o Toy o twist_g for any g(J); only g = 0 is in the student class. The stalled 16 x 64 students sit at g = 1.8-2.9 rad,
in the time-reversed sector of the teacher (their turning-point angle has the opposite sign), where the best they can
do is about the teacher's own asymmetric error (R T R scores 2.05e-5 on the teacher target).

    python gauge_lab.py perturb SRC_JOB OUT_DIR --mode sloppy|random [--target 1e-5] [--rel 1e-4] [--n-steps 256]
        the teacher itself as the student, moved off the exact solution: along a random combination of the right
        singular vectors of the energy-residual Jacobian with sigma/sigma_max < --cut (sloppy, scaled until the
        held-out error is --target), or by relative parameter noise --rel (random). Tests whether LM gets back to
        machine-level accuracy from inside the teacher's own sector.
    python gauge_lab.py flip SRC_JOB OUT_DIR
        the time-reversal conjugate R S R of a stalled student (W1 -> -W1 in the q-shears, gate -> -gate in the
        p-shears; exact), which moves it into the teacher's sector without new parameters.
    python gauge_lab.py twist JOB_DIR [JOB_DIR ...]
        the twist g(J) of a student relative to the teacher (angle of T^-1 o S in the teacher's toy coordinates), its
        held-out error and the turning-point angle by z_max.
OUT_DIR gets config.json (SRC_JOB's, with name/batch changed) and state.pt, ready for optim_lab.py.
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402
import optim_lab as OL  # noqa: E402

F64 = torch.float64


def load(jd, n_steps=None):
    cfg, t, aa, nO, nS, es, Ev, build = OL.setup(jd, n_steps=n_steps)
    m = build(cfg.get("seed", 0))
    m.load_state_dict({k: v.double() for k, v in torch.load(os.path.join(jd, "state.pt"), weights_only=False)["state"].items()})
    return cfg, t, aa, nO, nS, es, Ev, m


def write(out, cfg, m, note):
    os.makedirs(out, exist_ok=True)
    cfg = dict(cfg, name=os.path.basename(os.path.normpath(out)), batch=os.path.basename(os.path.dirname(os.path.normpath(out))), note=note)
    cfg["teacher"] = dict(cfg["teacher"], checkpoint=os.path.relpath(cfg["teacher"]["checkpoint"], HERE)
                          if os.path.isabs(cfg["teacher"]["checkpoint"]) else cfg["teacher"]["checkpoint"])
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    torch.save(dict(state=C.state_f64(m)), os.path.join(out, "state.pt"))


def energy_jacobian(m, aa, t, nO, nS, es, chunk=16):
    """Rows of the energy-residual Jacobian (nO nS, P), exactly as train_lm builds them (escale normalisation)."""
    from torch.func import functional_call, jacrev, vmap
    names = [n for n, _ in m.named_parameters()]
    params = {n: p.detach() for n, p in m.named_parameters()}
    f_point = lambda p, x: functional_call(m, p, (x[None, :],))[0]
    rows = []
    for o0 in range(0, nO, chunk):
        x = aa[o0:o0 + chunk].reshape(-1, 2); k = x.shape[0] // nS
        Jd = vmap(jacrev(f_point), in_dims=(None, 0))(params, x)
        Jy = torch.cat([Jd[n].reshape(x.shape[0], 2, -1) for n in names], -1)
        with torch.no_grad():
            y = m(x)
        y = y.detach().requires_grad_(True)
        dHdy = torch.autograd.grad(t.H(y).sum(), y)[0]
        dH = (dHdy[:, :, None] * Jy).sum(1).view(k, nS, -1)
        rows.append(((dH - dH.mean(1, keepdim=True)) / es[o0:o0 + k][..., None] / math.sqrt(nO * nS)).reshape(k * nS, -1))
    return torch.cat(rows)


def set_flat(m, v):
    with torch.no_grad():
        i = 0
        for p in m.parameters():
            n = p.numel(); p.copy_(v[i:i + n].view_as(p)); i += n


def perturb(a):
    cfg, t, aa, nO, nS, es, Ev, m = load(a.src, a.n_steps)
    m.flow.load_state_dict(t.T.state_dict())  # the student is now the teacher: zero energy and coordinate residual
    v0 = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
    print(f"teacher as student: held-out {Ev.quick(m)['eval_mean']:.2e}")
    g = torch.Generator().manual_seed(a.seed)
    if a.mode == "random":
        set_flat(m, v0 + a.rel * v0.abs().clamp(min=1e-3) * torch.randn(v0.shape, generator=g, dtype=F64))
        note = f"teacher + relative parameter noise {a.rel:g}"
    else:
        Jm = energy_jacobian(m, aa, t, nO, nS, es)
        U, S, Vh = torch.linalg.svd(Jm, full_matrices=False)
        sel = S / S[0] < a.cut
        print(f"Jacobian {tuple(Jm.shape)}: sigma/sigma_max from {float(S[-1] / S[0]):.1e}; {int(sel.sum())} directions below {a.cut:g}")
        d = Vh[sel].T @ torch.randn(int(sel.sum()), generator=g, dtype=F64)
        d = d / d.norm() * v0.norm()
        err = lambda s: (set_flat(m, v0 + s * d), Ev.quick(m)["eval_mean"])[1]
        lo, hi = 0.0, 1.0
        while err(hi) < a.target:
            lo, hi = hi, 2 * hi
        for _ in range(40):  # bisection in log scale once bracketed
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 2
            lo, hi = (mid, hi) if err(mid) < a.target else (lo, mid)
            if hi / max(lo, 1e-300) < 1.02:
                break
        e = err(hi)
        note = f"teacher moved along {int(sel.sum())} sloppy directions (sigma/sigma_max < {a.cut:g}), step {hi:.3e} |v|"
        print(f"step {hi:.3e} |v|: held-out {e:.2e}")
    q = Ev.quick(m)
    print(f"start: held-out {q['eval_mean']:.3e}, max {q['eval_max']:.2e}")
    write(a.out, cfg, m, note)


def flip(a):
    cfg, t, aa, nO, nS, es, Ev, m = load(a.src)
    print(f"{os.path.basename(a.src)}: held-out {Ev.quick(m)['eval_mean']:.3e}")
    with torch.no_grad():
        for L in m.flow.layers:
            if L.update_q:
                L.conditioner.layers[0].weight.mul_(-1)
            else:
                L.conditioner.gate.mul_(-1)
    print(f"R S R: held-out {Ev.quick(m)['eval_mean']:.3e}")
    write(a.out, cfg, m, f"time-reversal conjugate R S R of {os.path.basename(os.path.normpath(a.src))}")


def twist(a):
    for jd in a.jobs:
        cfg, t, aa, nO, nS, es, Ev, m = load(jd)
        J_lo, J_hi = Ev.J_range; w0 = t.nu0
        Jh = C.j_grid(40, J_lo, J_hi, w0, "asinh", 0.03, offset=0.5); nth = 256
        A = C.aa_points(Jh, C.angles(nth, 0.25)).reshape(-1, 2)
        with torch.no_grad():
            X = t.T.inverse(m(A)); q, p = X[:, 0], X[:, 1]
            d = torch.remainder(torch.atan2(-p / math.sqrt(w0), q * math.sqrt(w0)) - A[:, 0] + math.pi, 2 * math.pi) - math.pi
            d = d.view(len(Jh), nth)
            gJ = torch.atan2(torch.sin(d).mean(1), torch.cos(d).mean(1))
        z = C.zmax_kpc_of_J(Jh, C.MWVertical().phi)
        print(f"{os.path.basename(os.path.normpath(jd))}: held-out {Ev.quick(m)['eval_mean']:.3e}; twist g(J) {float(gJ.min()):+.4f} .. "
              f"{float(gJ.max()):+.4f} rad (range {float(gJ.max() - gJ.min()):.2e}); by z_max "
              + " ".join(f"{float(zz) * 1000:.0f}pc:{float(gg):+.3f}" for zz, gg in list(zip(z, gJ))[::8]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("perturb"); p1.add_argument("src"); p1.add_argument("out")
    p1.add_argument("--mode", choices=["sloppy", "random"], default="sloppy"); p1.add_argument("--target", type=float, default=1e-5)
    p1.add_argument("--cut", type=float, default=1e-8); p1.add_argument("--rel", type=float, default=1e-4)
    p1.add_argument("--seed", type=int, default=0); p1.add_argument("--n-steps", type=int, default=256)
    p2 = sp.add_parser("flip"); p2.add_argument("src"); p2.add_argument("out")
    p3 = sp.add_parser("twist"); p3.add_argument("jobs", nargs="+")
    a = ap.parse_args()
    {"perturb": perturb, "flip": flip, "twist": twist}[a.cmd](a)
