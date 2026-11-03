"""1D profiles of the energy landscape through the teacher T and through the stalled student A (task "profiles").

Built on landscape_lab (LL): same landscape set (100 tori x 128 angles), same eps (mean |dE/E|), same parameter layout.
A direction d is a unit vector in raw parameter space; the profile is eps(O + t |theta_T| d) for t in TS (both signs).

    python landscape_profiles.py twist   twist subspace at T: J_C restricted to the energy-null space of J_E, and
                                         least-squares realisations of a constant / ramp twist -> RES/profiles_twist.npz
    python landscape_profiles.py T       profiles from T (SVD, random, toward A, symmetry and twist-like directions),
                                         with jets (first/second t-derivatives of the residual, twist fraction)
                                         -> RES/profiles_T.npz
    python landscape_profiles.py A       profiles from A (raw, gate and min+shift gauges) -> RES/profiles_A.npz
    python landscape_profiles.py dist    distance from T vs eps vs LM outcome -> RES/profiles_dist.npz
    python landscape_profiles.py summary slopes, crossings, tables -> RES/profiles_summary.json
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NTHREADS", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402
import landscape_lab as LL  # noqa: E402

torch.set_num_threads(int(os.environ["NTHREADS"]))
F64 = torch.float64
RES = LL.RES
TS = np.logspace(-14, 0.5, 80)
SV_TARGETS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
NULL_CUTS = (1e-6, 1e-8, 1e-10, 1e-12)
FIT_CUTS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)


def log(*a):
    print(*a, flush=True)


# ===================================================================== evaluation helpers
def _params(v):
    L = LL.load_all(); out, i = {}, 0
    for n, p in zip(LL.NAMES, L.m.parameters()):
        k = p.numel(); out["flow." + n] = v[i:i + k].view_as(p); i += k
    return out


def _map(x):
    from torch.func import functional_call
    L = LL.load_all()
    return lambda v: functional_call(L.m, _params(v), (x,))


@torch.no_grad()
def metrics(V, chunk=32):
    """(eps, rms) of the relative energy residual (H - <H>)/|<H>| on the landscape set for V (K, P): eps = mean |.|
    (the landscape height, = LL.eps_batch up to summation order), rms = root mean square (the least-squares view)."""
    L = LL.load_all()
    V = torch.as_tensor(V, dtype=F64); V = V[None] if V.dim() == 1 else V
    E, R = [], []
    for k0 in range(0, V.shape[0], chunk):
        y = LL._forward(V[k0:k0 + chunk], L.x_land)
        H = L.t.H(y.reshape(-1, 2)).view(-1, LL.LAND_TORI, LL.LAND_ANGLES)
        mH = H.mean(-1, keepdim=True); r = (H - mH) / mH.abs()
        E.append(r.abs().mean((-1, -2))); R.append(r.pow(2).mean((-1, -2)).sqrt())
    return torch.cat(E).numpy(), torch.cat(R).numpy()


def coord_scale_land():
    L = LL.load_all()
    return C.coord_scale(L.t.nu0 * L.J_land, float(L.t.nu0)).view(LL.LAND_TORI, 1, 2)


def input_tangents(v0, x, nO, nS, s):
    """Scaled d y / d theta and J d y / d J of the map at v0 on x (nO nS, 2): (nO, nS, 2) each, divided by s."""
    from torch.func import jvp
    from torch.func import functional_call
    L = LL.load_all(); p = _params(v0)
    f = lambda xx: functional_call(L.m, p, (xx,))
    e_th = torch.stack([torch.ones(len(x), dtype=F64), torch.zeros(len(x), dtype=F64)], -1)
    e_J = torch.stack([torch.zeros(len(x), dtype=F64), x[:, 1]], -1)
    _, Th = jvp(f, (x,), (e_th,)); _, Jd = jvp(f, (x,), (e_J,))
    return Th.view(nO, nS, 2) / s, Jd.view(nO, nS, 2) / s


def twist_fractions(D, Th, Jd=None):
    """D (..., nO, nS, 2) scaled coordinate changes; Th, Jd (nO, nS, 2) scaled theta- and J-tangents. Per torus fit
    D ~ g Th (twist) and D ~ g Th + h Jd (torus-preserving to first order). Returns f_tw, g (..., nO), f_tp."""
    tot = (D * D).sum((-1, -2, -3))
    tt = (Th * Th).sum((-1, -2)); dt = (D * Th).sum((-1, -2))
    g = dt / tt
    f_tw = (g * dt).sum(-1) / tot
    if Jd is None:
        return f_tw, g, None
    jj = (Jd * Jd).sum((-1, -2)); tj = (Th * Jd).sum((-1, -2)); dj = (D * Jd).sum((-1, -2))
    det = tt * jj - tj * tj
    a = (dt * jj - dj * tj) / det; b = (dj * tt - dt * tj) / det
    f_tp = (a * dt + b * dj).sum(-1) / tot
    return f_tw, g, f_tp


def jets(v0, d):
    """Taylor coefficients in t (step t |theta_T| d) of the relative residual r = (H - <H>)/<H> on the landscape set:
    r0, r1, r2 (nO, nS); the scaled first-order coordinate change dx (nO, nS, 2), its twist fraction (per-torus fit
    g(J) dx/dtheta), its torus-preserving fraction (g dx/dtheta + h J dx/dJ) and g (nO,)."""
    from torch.func import jvp
    L = LL.load_all(); x = L.x_land; nO, nS = LL.LAND_TORI, LL.LAND_ANGLES
    v0 = torch.as_tensor(v0, dtype=F64); u = torch.as_tensor(d, dtype=F64) * L.nT
    F = _map(x)
    Hf = lambda v: L.t.H(F(v))
    y0, y1 = jvp(F, (v0,), (u,))
    _, y2 = jvp(lambda v: jvp(F, (v,), (u,))[1], (v0,), (u,))
    H0, H1 = jvp(Hf, (v0,), (u,))
    _, H2 = jvp(lambda v: jvp(Hf, (v,), (u,))[1], (v0,), (u,))
    H0, H1, H2 = H0.view(nO, nS), H1.view(nO, nS), 0.5 * H2.view(nO, nS)
    Dk = [h.mean(1, keepdim=True) for h in (H0, H1, H2)]
    Nk = [h - m for h, m in zip((H0, H1, H2), Dk)]
    r0 = Nk[0] / Dk[0]; r1 = (Nk[1] - r0 * Dk[1]) / Dk[0]; r2 = (Nk[2] - r1 * Dk[1] - r0 * Dk[2]) / Dk[0]
    s = coord_scale_land()
    dx = y1.view(nO, nS, 2) / s
    Th, Jd = input_tangents(v0, x, nO, nS, s)
    f_tw, g, f_tp = twist_fractions(dx, Th, Jd)
    return dict(r0=r0.detach(), r1=r1.detach(), r2=r2.detach(), dx=dx.detach(), d2x=(0.5 * y2.view(nO, nS, 2) / s).detach(),
                f_tw=float(f_tw), f_tp=float(f_tp), g=g.detach(), dx_rms=float(dx.pow(2).sum(-1).mean().sqrt()))


def predict(j, ts, sign):
    t = torch.as_tensor(sign * ts, dtype=F64)[:, None, None]
    return (j["r0"][None] + j["r1"][None] * t + j["r2"][None] * t * t).abs().mean((-1, -2)).numpy()


def profile(v0, d, ts=TS):
    """eps and rms (2, nt) for signs (+, -) along O + s t |theta_T| d."""
    L = LL.load_all()
    v0 = torch.as_tensor(v0, dtype=F64); d = torch.as_tensor(d, dtype=F64)
    st = torch.as_tensor(np.concatenate([ts, -ts]), dtype=F64)
    V = v0[None] + (st * L.nT)[:, None] * d[None]
    e, r = metrics(V)
    return e.reshape(2, -1), r.reshape(2, -1)


def unit(v):
    v = torch.as_tensor(v, dtype=F64)
    return v / v.norm()


def rand_dirs():
    L = LL.load_all(); out = {}
    for s in range(3):
        g = torch.Generator().manual_seed(s)
        out[f"rand{s}"] = unit(torch.randn(LL.P, generator=g, dtype=F64))
    g = torch.Generator().manual_seed(100)
    out["relrand0"] = unit(L.vT.abs().clamp(min=1e-3) * torch.randn(LL.P, generator=g, dtype=F64))
    return out


def sv_pick(S, Vh, targets):
    rel = S / S[0]; out = {}
    out["sv1"] = (0, float(rel[0]), torch.from_numpy(np.ascontiguousarray(Vh[0])))
    for c in targets:
        i = int(np.argmin(np.abs(np.log(np.maximum(rel, 1e-300)) - math.log(c))))
        out[f"sv_{c:.0e}"] = (i, float(rel[i]), torch.from_numpy(np.ascontiguousarray(Vh[i])))
    return out


# ===================================================================== twist subspace at T
def train_tangent_T():
    """Scaled d y/d theta on the training set at T, (nO, nS, 2), and the per-point scale (N, 2)."""
    L = LL.load_all()
    s = C.coord_scale(L.es, float(L.t.nu0))[:, None, :]  # (nO, 1, 2)
    Th, _ = input_tangents(L.vT, L.aa.reshape(-1, 2), L.nO, L.nS, s)
    return Th.detach(), s


def cmd_twist():
    L = LL.load_all(); t0 = time.perf_counter()
    N = L.nO * L.nS; sN = math.sqrt(N)
    Jc, _ = LL.coord_jac(L.vT); log(f"J_C {tuple(Jc.shape)} {time.perf_counter() - t0:.0f} s")
    Je, _ = LL.energy_jac(L.vT); log(f"J_E {tuple(Je.shape)} {time.perf_counter() - t0:.0f} s")
    Th, s = train_tangent_T()
    z = LL.svd("T_energy"); SE = torch.from_numpy(z["S"]); VhE = torch.from_numpy(z["Vh"])
    out = {}
    # --- (a) energy-null subspace: how much can it move the coordinates, and is that motion a twist?
    for c in NULL_CUTS:
        sel = SE / SE[0] < c
        Nb = VhE[sel].T.contiguous()
        M = Jc @ Nb
        U, sM, W = torch.linalg.svd(M, full_matrices=False)
        K = min(64, len(sM))
        Dk = (U[:, :K] * sM[:K]).T.reshape(K, L.nO, L.nS, 2) * sN  # scaled dx of the top K
        f_tw, _, _ = twist_fractions(Dk, Th)
        dirs = (Nb @ W[:K].T).T  # (K, P) unit vectors
        eresp = (Je @ dirs.T).norm(dim=0)
        tag = f"{c:.0e}"
        out[f"null_{tag}_n"] = int(sel.sum()); out[f"null_{tag}_sM"] = sM.numpy()
        out[f"null_{tag}_ftw"] = f_tw.numpy(); out[f"null_{tag}_eresp"] = eresp.numpy()
        if c == 1e-8:
            out["twnull_dirs"] = dirs[:8].numpy()
        log(f"energy-null cut {c:.0e}: {int(sel.sum())} dims; s_M/sigma_C,max top "
            + " ".join(f"{float(x):.2e}" for x in (sM[:12] / 1260.8)) + f"; counts s_M/1260.8 > 1e-2/1e-4/1e-6/1e-8: "
            + "/".join(str(int((sM / 1260.8 > q).sum())) for q in (1e-2, 1e-4, 1e-6, 1e-8))
            + "; f_tw top12 " + " ".join(f"{float(x):.4f}" for x in f_tw[:12]) + f"; {time.perf_counter() - t0:.0f} s")
        del M, U, W
    # --- (b) realise an explicit twist g(J) dx/dtheta by least squares on J_C (minimum norm, SVD cutoff)
    U, S, Vh = torch.linalg.svd(Jc, full_matrices=False); log(f"SVD J_C {time.perf_counter() - t0:.0f} s")
    out["S_C"] = S.numpy()
    lnJ = torch.log(L.es[:, 0] / L.t.nu0)
    ramp = 2 * (lnJ - lnJ.min()) / (lnJ.max() - lnJ.min()) - 1
    for gname, gJ in (("g1", torch.ones(L.nO, dtype=F64)), ("ramp", ramp)):
        b = (gJ[:, None, None] * Th).reshape(-1) / sN
        Ub = U.T @ b; nb2 = float(b @ b)
        rows, D = [], []
        for c in FIT_CUTS:
            keep = S / S[0] > c
            d = Vh[keep].T @ (Ub[keep] / S[keep])
            res = Jc @ d - b
            rows.append([c, float(d.norm()) / L.nT, float(res @ res) / nb2, float((Je @ d).norm()), float((Jc @ d).norm())])
            D.append(d)
        out[f"fit_{gname}_table"] = np.array(rows)  # cut, |d|/|theta_T| per unit twist, residual fraction, |J_E d|, |J_C d|
        out[f"fit_{gname}_dirs"] = torch.stack(D).numpy()
        out[f"fit_{gname}_g"] = gJ.numpy()
        for r in rows:
            log(f"twist fit {gname} cut {r[0]:.0e}: |d|/|T| {r[1]:.3e} per rad, residual fraction {r[2]:.2e}, |J_E d| {r[3]:.2e}, "
                f"|J_C d| {r[4]:.3e} (target |b| {math.sqrt(nb2):.3e})")
    out["sigma_E_max"] = float(SE[0]); out["sigma_C_max"] = float(S[0])
    np.savez(f"{RES}/profiles_twist.npz", **out)
    log(f"saved {RES}/profiles_twist.npz, {time.perf_counter() - t0:.0f} s")


# ===================================================================== profiles from T
def T_directions():
    L = LL.load_all()
    z = LL.svd("T_energy"); dirs, meta = {}, {}
    for k, (i, rel, v) in sv_pick(z["S"], z["Vh"], SV_TARGETS).items():
        dirs[k] = v; meta[k] = dict(kind="J_E singular vector at T", index=i, sigma_rel=rel)
    for k, v in rand_dirs().items():
        dirs[k] = v; meta[k] = dict(kind="random" if k.startswith("rand") else "random, relative (|theta_T| * N(0,1))")
    dirs["toA"] = unit(L.al_min_shift[L.A] - L.vT); meta["toA"] = dict(kind="toward A (min+shift gauge)", dist=LL.rdist(L.al_min_shift[L.A]))
    dirs["toA_raw"] = unit(L.raw[L.A] - L.vT); meta["toA_raw"] = dict(kind="toward A (raw)", dist=LL.rdist(L.raw[L.A]))
    dirs["toB"] = unit(L.al_min_shift[L.B] - L.vT); meta["toB"] = dict(kind="toward B (min+shift gauge)", dist=LL.rdist(L.al_min_shift[L.B]))
    g0 = torch.zeros(LL.P, dtype=F64); b = LL.blocks(g0); bT = LL.blocks(L.vT)
    b[0, LL.G] = bT[0, LL.G]; b[0, LL.W2] = -bT[0, LL.W2]; b[0, LL.B2] = -bT[0, LL.B2]
    dirs["gate0"] = unit(g0); meta["gate0"] = dict(kind="gate-scale symmetry tangent, layer 0 (exact to first order only)")
    dirs["shift0"] = unit(LL._shift_basis(L.vT)[:, 0]); meta["shift0"] = dict(kind="translation gauge 0 (exact along the line)")
    tw = np.load(f"{RES}/profiles_twist.npz")
    for gname, cuts in (("g1", (1e-6, 1e-8)), ("ramp", (1e-8,))):
        tab = tw[f"fit_{gname}_table"]
        for c in cuts:
            i = int(np.argmin(np.abs(np.log(tab[:, 0]) - math.log(c))))
            k = f"twfit_{gname}_{c:.0e}"
            dirs[k] = unit(torch.from_numpy(tw[f"fit_{gname}_dirs"][i]))
            meta[k] = dict(kind=f"least-squares twist g={gname} (SVD cut {c:.0e})", norm_per_rad=float(tab[i, 1]), fit_resid=float(tab[i, 2]))
    for i in range(3):
        k = f"twnull{i + 1}"
        dirs[k] = unit(torch.from_numpy(tw["twnull_dirs"][i]))
        meta[k] = dict(kind=f"top-{i + 1} coordinate-moving direction in the J_E-null space (sigma/sigma_max < 1e-8)")
    return dirs, meta


def run_profiles(tag, v0, dirs, meta, Je=None, extra=None):
    L = LL.load_all(); t0 = time.perf_counter()
    out = dict(ts=TS, names=np.array(list(dirs)), origin=v0.numpy())
    for k, d in dirs.items():
        j = jets(v0, d)
        e, r = profile(v0, d)
        out[f"{k}__d"] = d.numpy(); out[f"{k}__eps"] = e; out[f"{k}__rms"] = r
        out[f"{k}__pred"] = np.stack([predict(j, TS, +1), predict(j, TS, -1)])
        out[f"{k}__g"] = j["g"].numpy()
        sc = dict(c0=float(j["r0"].abs().mean()), c1=float(j["r1"].abs().mean()), c2=float(j["r2"].abs().mean()),
                  rms1=float(j["r1"].pow(2).mean().sqrt()), rms2=float(j["r2"].pow(2).mean().sqrt()),
                  dx_rms=j["dx_rms"], f_tw=j["f_tw"], f_tp=j["f_tp"],
                  d2x_rms=float(j["d2x"].pow(2).sum(-1).mean().sqrt()))
        if Je is not None:
            sc["rms1_train"] = float((Je @ d).norm()) * L.nT
        meta[k].update(sc)
        log(f"{tag} {k:18s} c1 {sc['c1']:.3e} c2 {sc['c2']:.3e} t*=c1/c2 {sc['c1'] / max(sc['c2'], 1e-300):.2e}  dx {sc['dx_rms']:.3e} "
            f"f_tw {sc['f_tw']:.6f} f_tp {sc['f_tp']:.6f}  eps(+) at t=1e-10,1e-6,1e-3,1: "
            + " ".join(f"{float(np.interp(math.log(q), np.log(TS), e[0])):.2e}" for q in (1e-10, 1e-6, 1e-3, 1.0))
            + f"  [{time.perf_counter() - t0:.0f} s]")
    if extra:
        out.update(extra)
    out["meta"] = json.dumps(meta)
    np.savez(f"{RES}/profiles_{tag}.npz", **out)
    log(f"saved {RES}/profiles_{tag}.npz [{time.perf_counter() - t0:.0f} s]")


def cmd_T():
    L = LL.load_all()
    Je, _ = LL.energy_jac(L.vT)
    dirs, meta = T_directions()
    if len(sys.argv) > 2:
        keep = sys.argv[2].split(","); dirs = {k: v for k, v in dirs.items() if k in keep}
    # straight segments T -> A (min+shift and raw) and T -> B, in units of the segment
    sg = np.linspace(0, 1.2, 49)
    extra = dict(seg_s=sg)
    for nm, w in (("A", L.al_min_shift[L.A]), ("A_raw", L.raw[L.A]), ("B", L.al_min_shift[L.B])):
        V = L.vT[None] + torch.as_tensor(sg, dtype=F64)[:, None] * (w - L.vT)[None]
        e, r = metrics(V); extra[f"seg_{nm}_eps"] = e; extra[f"seg_{nm}_rms"] = r
        log(f"segment T->{nm}: eps at s=0.1,0.5,0.9,1.0: " + " ".join(f"{e[np.argmin(np.abs(sg - q))]:.2e}" for q in (0.1, 0.5, 0.9, 1.0)))
    run_profiles("T" if len(sys.argv) <= 2 else "T_part", L.vT, dirs, meta, Je=Je, extra=extra)


# ===================================================================== profiles from A
def cmd_A():
    L = LL.load_all(); t0 = time.perf_counter()
    origins = {"raw": L.raw[L.A], "gate": L.al[L.A], "ms": L.al_min_shift[L.A]}
    rd = rand_dirs()
    dirs, meta = {}, {}
    for oname, v0 in origins.items():
        if oname == "gate":
            z = LL.svd("A_energy"); S, Vh = z["S"], z["Vh"]
        elif oname == "ms":
            z = LL.svd("A_energy_min_shift"); S, Vh = z["S"], z["Vh"]
        else:
            Jm, _ = LL.energy_jac(v0)
            _, St, Vht = torch.linalg.svd(Jm, full_matrices=False); S, Vh = St.numpy(), Vht.numpy(); del Jm
            np.save(f"{RES}/profiles_A_raw_S.npy", S)
            log(f"SVD J_E at raw A: sigma_max {S[0]:.1f} [{time.perf_counter() - t0:.0f} s]")
        for k, (i, rel, v) in sv_pick(S, Vh, (1e-4, 1e-8)).items():
            dirs[f"{oname}:{k}"] = v; meta[f"{oname}:{k}"] = dict(origin=oname, kind="J_E singular vector at A", index=i, sigma_rel=rel)
        for k in ("rand0", "rand1", "rand2"):
            dirs[f"{oname}:{k}"] = rd[k]; meta[f"{oname}:{k}"] = dict(origin=oname, kind="random")
        dirs[f"{oname}:toT"] = unit(L.vT - v0); meta[f"{oname}:toT"] = dict(origin=oname, kind="toward T", dist=LL.rdist(v0))
    out = dict(ts=TS, names=np.array(list(dirs)))
    eA = {}
    for oname, v0 in origins.items():
        e0, r0 = metrics(v0); eA[oname] = (float(e0[0]), float(r0[0]))
        out[f"origin_{oname}"] = v0.numpy()
    for k, d in dirs.items():
        oname = k.split(":")[0]; v0 = origins[oname]
        j = jets(v0, d)
        e, r = profile(v0, d)
        out[f"{k}__d"] = d.numpy(); out[f"{k}__eps"] = e; out[f"{k}__rms"] = r
        out[f"{k}__pred"] = np.stack([predict(j, TS, +1), predict(j, TS, -1)])
        meta[k].update(eps0=eA[oname][0], rms0=eA[oname][1], c0=float(j["r0"].abs().mean()), c1_signed=float((j["r0"].sign() * j["r1"]).mean()),
                       rms_lin=float((j["r0"] * j["r1"]).mean() / j["r0"].pow(2).mean().sqrt()),
                       rms2_coef=float(j["r1"].pow(2).mean()), dx_rms=j["dx_rms"], f_tw=j["f_tw"], f_tp=j["f_tp"])
        log(f"A {k:16s} eps0 {eA[oname][0]:.4e}  d eps/dt {meta[k]['c1_signed']:+.2e}  dx {j['dx_rms']:.2e} f_tw {j['f_tw']:.4f}  "
            f"eps(+) at t=1e-8,1e-6,1e-4,1e-2,1: " + " ".join(f"{float(np.interp(math.log(q), np.log(TS), e[0])):.3e}" for q in (1e-8, 1e-6, 1e-4, 1e-2, 1.0))
            + f"  [{time.perf_counter() - t0:.0f} s]")
    out["meta"] = json.dumps(meta)
    np.savez(f"{RES}/profiles_A.npz", **out)
    log(f"saved {RES}/profiles_A.npz [{time.perf_counter() - t0:.0f} s]")


# ===================================================================== distance vs outcome
OUTCOME = {  # start point -> (outcome, final point name or None)
    "rand4_start": ("converged", "rand4_final"), "sloppy_start": ("converged", "sloppy_final"),
    "rsr_geo_s0_start": ("stalled", "rsr_geo_s0"), "rsr_geo_s1_start": ("stalled", "rsr_geo_s1"),
}


@torch.no_grad()
def map_distances(v):
    """Gauge-invariant distances of the map S_v from T on the landscape set: rms |S - T| / torus amplitude, and in the
    teacher's toy coordinates of T^-1 S: the twist g(J) (circular mean of the angle offset per torus), the rms angle
    distortion after removing g(J), and rms dJ/J."""
    L = LL.load_all(); nO, nS = LL.LAND_TORI, LL.LAND_ANGLES
    x = L.x_land; s = coord_scale_land()
    y = LL._forward(torch.as_tensor(v, dtype=F64)[None], x)[0]
    yT = L.t.aa_map(x)
    dC = float(((y - yT).view(nO, nS, 2) / s).pow(2).sum(-1).mean().sqrt())
    X = L.t.T.inverse(y); w = L.t.omega
    th2 = torch.atan2(-X[:, 1] / math.sqrt(w), X[:, 0] * math.sqrt(w)); J2 = (X[:, 1] ** 2 + (w * X[:, 0]) ** 2) / (2 * w)
    dth = torch.remainder(th2 - x[:, 0] + math.pi, 2 * math.pi).view(nO, nS) - math.pi
    g = torch.atan2(torch.sin(dth).mean(1), torch.cos(dth).mean(1))
    dist = torch.remainder(dth - g[:, None] + math.pi, 2 * math.pi) - math.pi
    dJ = (J2.view(nO, nS) - x[:, 1].view(nO, nS)) / x[:, 1].view(nO, nS)
    return dict(coord_rms=dC, twist_min=float(g.min()), twist_max=float(g.max()), twist_mean=float(g.mean()),
                angle_distortion_rms=float(dist.pow(2).mean().sqrt()), dJ_over_J_rms=float(dJ.pow(2).mean().sqrt()))


def cmd_dist():
    import optim_lab as OL
    L = LL.load_all(); t0 = time.perf_counter()
    A = json.load(open(f"{RES}/aligned.json"))
    rows = {}
    for n in L.names:
        r = A["rows"][n]
        rows[n] = dict(dist_raw=r["dist_raw"], dist_gate=r["aligned"]["dist"], dist_min=r["aligned_min"]["dist"],
                       dist_min_shift=r["aligned_min_shift"]["dist"], eps_land=r["eps_land"], heldout=r["heldout_eval_mean"],
                       **map_distances(L.raw[n]))
    # the originals' starting points: build(seed) (Adam 3000 steps, then LM, from this init)
    cfg0 = json.load(open(os.path.join(LL.BASE, "config.json")))
    for n in LL.ORIGINALS:
        c = json.load(open(os.path.join(LL.JOBS[n], "config.json")))
        assert all(c[k] == cfg0[k] for k in ("model", "toy", "data", "teacher")), n
    _, _, _, _, _, _, _, build = OL.setup(LL.BASE)
    for seed in range(3):
        v = LL.to_vec(build(seed))
        e, _ = metrics(v)
        nm = f"init_s{seed}"
        al = LL.align(v, gauge="min", shifts=True)
        rows[nm] = dict(dist_raw=LL.rdist(v), dist_gate=LL.rdist(LL.align(v, gauge="gate")), dist_min=LL.rdist(LL.align(v, gauge="min")),
                        dist_min_shift=LL.rdist(al), eps_land=float(e[0]), heldout=None, **map_distances(v))
        log(f"{nm}: dist raw {rows[nm]['dist_raw']:.3f} min+shift {rows[nm]['dist_min_shift']:.3f}, eps {e[0]:.3e} [{time.perf_counter() - t0:.0f} s]")
    e, _ = metrics(L.vT)
    rows["teacher"] = dict(dist_raw=0.0, dist_gate=0.0, dist_min=0.0, dist_min_shift=0.0, eps_land=float(e[0]), heldout=A["heldout_T"],
                           **map_distances(L.vT))
    for n, r in rows.items():
        if n in OUTCOME:
            r["role"] = "start"; r["outcome"], r["final"] = OUTCOME[n]
            r["final_eps_land"] = rows[r["final"]]["eps_land"]
        elif n in ("rand4_final", "sloppy_final"):
            r["role"] = "end"; r["outcome"] = "converged"
        elif n.startswith("init_s"):
            r["role"] = "start"; r["outcome"] = "stalled"
            r["final"] = [m for m in LL.ORIGINALS if m.endswith(n[-2:])]
            r["final_eps_land"] = [rows[m]["eps_land"] for m in r["final"]]
        elif n == "teacher":
            r["role"] = "exact"; r["outcome"] = "exact"
        else:
            r["role"] = "end"; r["outcome"] = "stalled"
        log(f"{n:17s} {r['role']:5s} {r['outcome']:9s} dist raw {r['dist_raw']:.3e} ms {r['dist_min_shift']:.3e} eps {r['eps_land']:.3e} "
            f"coord {r['coord_rms']:.3e} twist {r['twist_min']:+.3f}..{r['twist_max']:+.3f} distortion {r['angle_distortion_rms']:.2e} "
            f"dJ/J {r['dJ_over_J_rms']:.2e}")
    json.dump(rows, open(f"{RES}/profiles_dist.json", "w"), indent=1)
    keys = ["dist_raw", "dist_gate", "dist_min", "dist_min_shift", "eps_land", "coord_rms", "twist_min", "twist_max", "angle_distortion_rms", "dJ_over_J_rms"]
    names = list(rows)
    np.savez(f"{RES}/profiles_dist.npz", names=np.array(names), role=np.array([rows[n]["role"] for n in names]),
             outcome=np.array([rows[n]["outcome"] for n in names]), **{k: np.array([rows[n][k] for n in names], dtype=float) for k in keys})
    log(f"saved {RES}/profiles_dist.{{npz,json}} [{time.perf_counter() - t0:.0f} s]")


# ===================================================================== summary
FLOOR = 5.8127e-16


def first_cross(ts, e, level):
    """First t on the grid where e >= level, log-log interpolated with the previous point; None if never."""
    idx = np.nonzero(e >= level)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(ts[0])
    a, b = np.log(e[i - 1]), np.log(e[i])
    f = (math.log(level) - a) / (b - a) if b > a else 1.0
    return float(np.exp(np.log(ts[i - 1]) + f * (np.log(ts[i]) - np.log(ts[i - 1]))))


def fit_slope(ts, e, lo, hi):
    """Least-squares log-log slope over the grid points before e first exceeds hi, with lo < e (contiguous run)."""
    stop = np.nonzero(e >= hi)[0]; stop = int(stop[0]) if len(stop) else len(e)
    m = np.zeros(len(e), bool); m[:stop] = e[:stop] > lo
    if m.sum() < 3:
        return None, int(m.sum()), None
    x, y = np.log10(ts[m]), np.log10(e[m])
    p = np.polyfit(x, y, 1)
    return float(p[0]), int(m.sum()), [float(ts[m][0]), float(ts[m][-1])]


def local_slope_at(ts, e, level):
    """Local log-log slope (centred, over +-2 grid points) where e first crosses level."""
    idx = np.nonzero(e >= level)[0]
    if len(idx) == 0:
        return None
    i = int(np.clip(idx[0], 2, len(e) - 3))
    return float((np.log10(e[i + 2]) - np.log10(e[i - 2])) / (np.log10(ts[i + 2]) - np.log10(ts[i - 2])))


def cmd_summary():
    L = LL.load_all()
    eA = L.eps_land[L.A]
    out = dict(grid=dict(t="logspace(-14, 0.5, 80) in units of |theta_T| (raw parameter space, unit directions)", theta_T_norm=L.nT,
                         landscape="100 tori x 128 angles (landscape_lab LAND set)", eps_floor_T=FLOOR, eps_A=eA, A=L.A),
               windows=dict(low="eps in (30 floor = 1.7e-14, 1e-11]", high="eps in (1e-11, 1e-7]", full="eps in (30 floor, 1e-7]"))
    # ---- T
    z = np.load(f"{RES}/profiles_T.npz"); meta = json.loads(str(z["meta"])); ts = z["ts"]
    rowsT = {}
    for k in [str(s) for s in z["names"]]:
        e = z[f"{k}__eps"]; pr = z[f"{k}__pred"]; m = meta[k]
        row = dict(m)
        e_star = m["c1"] ** 2 / m["c2"] if m["c2"] > 0 else float("inf")  # eps where the t and t^2 terms are equal
        for si, sg in enumerate(("+", "-")):
            ee = e[si]
            row[f"slope_low{sg}"], row[f"n_low{sg}"], _ = fit_slope(ts, ee, 30 * FLOOR, 1e-11)
            row[f"slope_high{sg}"], row[f"n_high{sg}"], _ = fit_slope(ts, ee, 1e-11, 1e-7)
            row[f"slope_full{sg}"], row[f"n_full{sg}"], row[f"t_range_full{sg}"] = fit_slope(ts, ee, 30 * FLOOR, 1e-7)
            # jet-informed windows: first order below 0.1 eps*, second order from 10 eps* up to 1e-5
            row[f"slope_1st{sg}"], row[f"n_1st{sg}"], row[f"t_range_1st{sg}"] = fit_slope(ts, ee, 30 * FLOOR, min(0.1 * e_star, 1e-5))
            row[f"slope_2nd{sg}"], row[f"n_2nd{sg}"], row[f"t_range_2nd{sg}"] = fit_slope(ts, ee, max(30 * FLOOR, 10 * e_star), 1e-5) \
                if 10 * e_star < 1e-6 else (None, 0, None)
            for lvl, nm in ((eA, "eps_A"), (2 * eA, "2eps_A"), (10 * eA, "10eps_A"), (1e-10, "1e-10")):
                row[f"t_{nm}{sg}"] = first_cross(ts, ee, lvl)
            row[f"local_slope_at_1e-12{sg}"] = local_slope_at(ts, ee, 1e-12)
            row[f"local_slope_at_eps_A{sg}"] = local_slope_at(ts, ee, eA)
            ok = (ee > 100 * FLOOR) & (ee < 1e-6)
            row[f"pred_ratio_med{sg}"] = float(np.median(ee[ok] / pr[si][ok])) if ok.any() else None
            row[f"pred_ratio_minmax{sg}"] = [float((ee[ok] / pr[si][ok]).min()), float((ee[ok] / pr[si][ok]).max())] if ok.any() else None
            row[f"eps_max{sg}"] = float(ee.max())
        row["sign_asym_max_below_1e-7"] = float(np.max(np.abs(np.log10(e[0][(e[0] < 1e-7) & (e[1] < 1e-7)] / e[1][(e[0] < 1e-7) & (e[1] < 1e-7)])))) \
            if ((e[0] < 1e-7) & (e[1] < 1e-7)).any() else None
        row["t_sea_0.1+"], row["t_sea_0.1-"] = first_cross(ts, e[0], 0.1), first_cross(ts, e[1], 0.1)
        null_dir = m["dx_rms"] < 1e-12  # symmetry tangent: no first-order coordinate change, f_tw undefined
        row["first_order_null_coordinates"] = bool(null_dir)
        row["c1_over_nontwist_dx"] = None if null_dir or m["f_tw"] >= 1 else m["c1"] / (m["dx_rms"] * math.sqrt(1 - m["f_tw"]))
        row["one_minus_f_tw"] = None if null_dir else 1 - m["f_tw"]
        row["t_star_c1_over_c2"] = m["c1"] / m["c2"] if m["c2"] > 0 else None
        row["eps_at_t_star"] = m["c1"] ** 2 / m["c2"] if m["c2"] > 0 else None
        row["energy_per_coord"] = m["c1"] / m["dx_rms"] if m["dx_rms"] > 0 else None
        rowsT[k] = row
    out["T"] = rowsT
    out["T_segments"] = {nm: dict(s=z["seg_s"].tolist(), eps=z[f"seg_{nm}_eps"].tolist()) for nm in ("A", "A_raw", "B")}
    # ---- twist subspace
    tw = np.load(f"{RES}/profiles_twist.npz")
    sCm = float(tw["sigma_C_max"])
    out["twist_subspace"] = {f"{c:.0e}": dict(dims=int(tw[f"null_{c:.0e}_n"]),
                                              sM_rel_top16=(tw[f"null_{c:.0e}_sM"][:16] / sCm).tolist(),
                                              counts_sM_rel_gt={f"{q:g}": int((tw[f"null_{c:.0e}_sM"] / sCm > q).sum()) for q in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8)},
                                              f_tw_top64=tw[f"null_{c:.0e}_ftw"].tolist(),
                                              energy_resp_top16=tw[f"null_{c:.0e}_eresp"][:16].tolist()) for c in NULL_CUTS}
    out["twist_fit"] = {g: [dict(cut=float(r[0]), norm_per_rad_over_thetaT=float(r[1]), resid_frac=float(r[2]), JE_d=float(r[3]), JC_d=float(r[4]))
                            for r in tw[f"fit_{g}_table"]] for g in ("g1", "ramp")}
    # ---- A
    if os.path.exists(f"{RES}/profiles_A.npz"):
        za = np.load(f"{RES}/profiles_A.npz"); ma = json.loads(str(za["meta"]))
        rowsA = {}
        for k in [str(s) for s in za["names"]]:
            e = za[f"{k}__eps"]; m = ma[k]; row = dict(m)
            e0 = m["eps0"]
            for si, sg in enumerate(("+", "-")):
                for f in (1.1, 2, 10):
                    row[f"t_{f:g}eps_A{sg}"] = first_cross(ts, e[si], f * e0)
                row[f"eps_min{sg}"] = float(e[si].min())
                row[f"t_eps_min{sg}"] = float(ts[int(np.argmin(e[si]))])
                row[f"t_sea_0.1{sg}"] = first_cross(ts, e[si], 0.1)
            rowsA[k] = row
        out["A"] = rowsA
        # A / T width ratios at the same absolute levels (geometric mean over the two signs)
        gm = lambda a: float(np.exp(np.mean(np.log(a)))) if a and all(x is not None for x in a) else None
        ratios = {}
        for oname in ("raw", "gate", "ms"):
            for dn, tn in (("sv1", "sv1"), ("sv_1e-04", "sv_1e-04"), ("sv_1e-08", "sv_1e-08"), ("rand0", "rand0"), ("rand1", "rand1"),
                           ("rand2", "rand2"), ("toT", "toA" if oname == "ms" else "toA_raw")):
                ra, rt = rowsA[f"{oname}:{dn}"], rowsT[tn]
                ratios[f"{oname}:{dn}"] = {lvl: (lambda a, b: a / b if a and b else None)(gm([ra[f"t_{lvl}+"], ra[f"t_{lvl}-"]]),
                                                                                         gm([rt[f"t_{lvl}+"], rt[f"t_{lvl}-"]]))
                                           for lvl in ("2eps_A", "10eps_A")}
                ratios[f"{oname}:{dn}"]["sea_0.1"] = (lambda a, b: a / b if a and b else None)(gm([ra["t_sea_0.1+"], ra["t_sea_0.1-"]]),
                                                                                             gm([rt["t_sea_0.1+"], rt["t_sea_0.1-"]]))
        out["width_ratio_A_over_T"] = ratios
        # widths: geometric mean of the two signs (min over signs as well)
        comp = {}
        for oname in ("raw", "gate", "ms"):
            for dn in ("sv1", "sv_1e-04", "sv_1e-08", "rand0", "rand1", "rand2", "toT"):
                k = f"{oname}:{dn}"
                if k not in rowsA:
                    continue
                r = rowsA[k]
                comp[k] = {f"{f}eps_A": [r[f"t_{f}eps_A+"], r[f"t_{f}eps_A-"]] for f in ("2", "10")}
        for dn in ("sv1", "sv_1e-04", "sv_1e-08", "rand0", "rand1", "rand2", "toA", "toA_raw"):
            r = rowsT[dn]
            comp[f"T:{dn}"] = {"2eps_A": [r["t_2eps_A+"], r["t_2eps_A-"]], "10eps_A": [r["t_10eps_A+"], r["t_10eps_A-"]],
                               "eps_A": [r["t_eps_A+"], r["t_eps_A-"]]}
        out["width_comparison"] = comp
    if os.path.exists(f"{RES}/profiles_dist.json"):
        out["dist"] = json.load(open(f"{RES}/profiles_dist.json"))
    json.dump(out, open(f"{RES}/profiles_summary.json", "w"), indent=1, default=lambda o: None)
    # console table
    log(f"{'dir':18s} {'sig_rel':>8s} {'f_tw':>9s} {'c1/dx':>8s} {'eps*':>8s} {'s_lo':>6s} {'s_hi':>6s} {'s_1st':>6s} {'s_2nd':>6s} {'s@epsA':>6s} "
        f"{'t_epsA':>9s} {'t_2epsA':>9s} {'t_10epsA':>9s} {'pred':>6s}")
    fmt = lambda v, f=".2e": ("None" if v is None else format(v, f))
    for k, r in rowsT.items():
        for sg in ("+", "-"):
            head = (f"{k:18s} {fmt(r.get('sigma_rel')):>8s} {r['f_tw']:9.6f} {fmt(r['energy_per_coord']):>8s} {fmt(r['eps_at_t_star'], '.1e'):>8s}"
                    if sg == "+" else f"{'':18s} {'(-)':>8s} {'':9s} {'':8s} {'':8s}")
            log(f"{head} {fmt(r['slope_low' + sg], '.3f'):>6s} {fmt(r['slope_high' + sg], '.3f'):>6s} {fmt(r['slope_1st' + sg], '.3f'):>6s} "
                f"{fmt(r['slope_2nd' + sg], '.3f'):>6s} {fmt(r['local_slope_at_eps_A' + sg], '.2f'):>6s} "
                f"{fmt(r['t_eps_A' + sg]):>9s} {fmt(r['t_2eps_A' + sg]):>9s} {fmt(r['t_10eps_A' + sg]):>9s} {fmt(r['pred_ratio_med' + sg], '.3f'):>6s}")
    if "A" in out:
        for k, r in out["A"].items():
            log(f"A {k:16s} eps0 {r['eps0']:.4e}  t(1.1/2/10 eps0) + {fmt(r['t_1.1eps_A+'])} {fmt(r['t_2eps_A+'])} {fmt(r['t_10eps_A+'])}  "
                f"- {fmt(r['t_1.1eps_A-'])} {fmt(r['t_2eps_A-'])} {fmt(r['t_10eps_A-'])}  min {r['eps_min+']:.4e}/{r['eps_min-']:.4e}")
    log(f"saved {RES}/profiles_summary.json")


if __name__ == "__main__":
    {"twist": cmd_twist, "T": cmd_T, "A": cmd_A, "dist": cmd_dist, "summary": cmd_summary}[sys.argv[1]]()
