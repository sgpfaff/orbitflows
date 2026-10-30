"""Is the optimisation plateau spectral bias? Angular-harmonic content of the residual Jacobian's singular directions.

    python spectral_bias_diag.py JOB_DIR [--out diag.json]

At a stalled checkpoint, the residual Jacobian J (rows: training points, (torus, angle); columns: parameters) is
decomposed as J = U S V^T. Each left singular vector u_k is a pattern of energy deviations over the training tori and
angles; its angular power spectrum (FFT along each torus) gives its typical harmonic n_k, and its torus weights give
where in z_max it lives. If the plateau is spectral bias, the directions with small sigma_k / sigma_max carry the high
harmonics, and the residual r (which LM cannot remove) lies in them. Prints, per decade of sigma_k / sigma_max: the
number of directions, their median harmonic and median z_max, and the share of the residual they carry; and the
residual's own harmonic spectrum.
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
from torch.func import functional_call, jacrev, vmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("job"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.job, "config.json")))
    if cfg["target"] == "teacher" and not os.path.isabs(cfg["teacher"].get("checkpoint", "/")):
        cfg["teacher"]["checkpoint"] = os.path.join(HERE, cfg["teacher"]["checkpoint"])
    t = C.make_target(cfg); d = cfg["data"]; mw = C.MWVertical()
    J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, mw.phi)[0]); J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, mw.phi)[0])
    w0 = t.nu0
    J_tr = C.train_grid(d, J_lo, J_hi, w0); nS = d["n_steps"]
    aa = C.aa_points(J_tr, C.angles(nS, 0.5)); nO = len(J_tr); N = nO * nS
    es = (w0 * J_tr).reshape(-1, 1)
    m = C.build_model(cfg["model"], cfg["toy"], w0, J_lo, J_hi, cfg.get("seed", 0))
    m.load_state_dict(torch.load(os.path.join(a.job, "state.pt"), weights_only=False)["state"])
    names = [n for n, _ in m.named_parameters()]; shapes = [p.shape for _, p in m.named_parameters()]
    sizes = [p.numel() for _, p in m.named_parameters()]
    v0 = torch.cat([p.detach().reshape(-1) for p in m.parameters()]); P = v0.numel()

    def unflat(v):
        out, i = {}, 0
        for n, s_, k in zip(names, shapes, sizes):
            out[n] = v[i:i + k].view(s_); i += k
        return out
    Js, rs = [], []
    for o0 in range(0, nO, 16):
        x = aa[o0:o0 + 16].reshape(-1, 2); k = x.shape[0] // nS
        Jd = vmap(jacrev(lambda p, xi: functional_call(m, p, (xi[None],))[0]), in_dims=(None, 0))(unflat(v0), x)
        Jy = torch.cat([Jd[n].reshape(x.shape[0], 2, -1) for n in names], -1)
        y = functional_call(m, unflat(v0), (x,)).detach().requires_grad_(True)
        Hf = t.H(y); dHdy = torch.autograd.grad(Hf.sum(), y)[0]
        H = Hf.detach().view(k, nS)
        dH = (dHdy[:, :, None] * Jy).sum(1).view(k, nS, -1)
        Js.append(((dH - dH.mean(1, keepdim=True)) / es[o0:o0 + k, :, None] / math.sqrt(N)).reshape(k * nS, -1))
        rs.append(((H - H.mean(1, keepdim=True)) / es[o0:o0 + k] / math.sqrt(N)).reshape(-1))
    Jr = torch.cat(Js).detach(); r = torch.cat(rs)
    print(f"{cfg['name']}: N = {N} residuals ({nO} tori x {nS} angles), P = {P} parameters; rms residual {float(r.norm()):.3e}", flush=True)
    U, S, Vh = torch.linalg.svd(Jr, full_matrices=False)
    rel = S / S[0]
    # angular spectrum of each left singular vector: FFT along the angles of every torus, power summed over tori
    Uc = U.T.reshape(-1, nO, nS)                                         # (K, nO, nS)
    pw = torch.fft.rfft(Uc, dim=-1).abs() ** 2                          # (K, nO, nS/2+1)
    n = torch.arange(pw.shape[-1], dtype=torch.float64)
    pn = pw.sum(1); nbar = (pn * n).sum(-1) / pn.sum(-1)                # power-weighted mean harmonic
    pt = pw.sum(-1); z = C.zmax_kpc_of_J(J_tr, mw.phi)
    zbar = torch.exp((pt * torch.log(z)).sum(-1) / pt.sum(-1))         # power-weighted geometric mean z_max
    ur = U.T @ r
    print(f"share of the residual in range(J): {float(ur.norm() / r.norm()):.4f}")
    rows = []
    print(f"{'sigma/sigma_max':>18s} {'count':>6s} {'median harmonic':>16s} {'median z_max (kpc)':>19s} {'share of residual':>18s}")
    for e in range(0, 17):
        lo, hi = 10.0 ** (-(e + 1)), 10.0 ** (-e)
        sel = (rel <= hi) & (rel > lo)
        if not sel.any():
            continue
        share = float((ur[sel] ** 2).sum() / (r ** 2).sum())
        row = dict(decade=-e, count=int(sel.sum()), nbar=float(nbar[sel].median()), zbar=float(zbar[sel].median()), share=share)
        rows.append(row)
        print(f"{'1e' + str(-(e + 1)) + '..1e' + str(-e):>18s} {row['count']:6d} {row['nbar']:16.1f} {row['zbar']:19.3f} {share:18.3e}")
    rr = r.view(nO, nS); rp = torch.fft.rfft(rr, dim=-1).abs() ** 2
    rpn = rp.sum(0); cum = torch.cumsum(rpn, 0) / rpn.sum()
    bands = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
    spec = {f"{a_}-{b_}": float(rpn[a_:b_ + 1].sum() / rpn.sum()) for a_, b_ in bands if b_ < rpn.numel()}
    print("residual power by harmonic band: " + "  ".join(f"n={k}: {v:.2e}" for k, v in spec.items()))
    print("harmonic below which 50% / 90% / 99% of the residual power lies: "
          + " / ".join(str(int(torch.searchsorted(cum, torch.tensor(q, dtype=cum.dtype)))) for q in (0.5, 0.9, 0.99)))
    if a.out:
        json.dump(dict(job=cfg["name"], N=N, P=P, decades=rows, residual_spectrum=spec), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
