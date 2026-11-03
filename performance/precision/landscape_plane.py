"""Task "plane": 2D slices, log scans and straight paths through the 16 x 64 MW-teacher energy landscape (needle test).

Built on landscape_lab (LL): eps(v) = LL.eps_batch (mean |dE/E| on the 100 x 128 landscape set); the training residual
norm |r| (LM objective, (H - <H>) / (nu0 J) / sqrt(N) on the 96 x 128 training set) by train_rnorm below. Points are the
teacher T = theta_T and students in the distance-minimising gauge with translation shifts (LL "aligned_min_shift"),
unless a name says otherwise. Plane coordinates are in units of |theta_T|: x = O + nT (a u + b w), u, w orthonormal.

    python landscape_plane.py scan               log scans from T, A, B along +-u, +-w (+ random) -> RES/plane_scan.npz
    python landscape_plane.py pick_r             cutout half-width r from the scans -> RES/plane_r.json
    python landscape_plane.py paths [NAMES]      straight paths (T->A in four gauges, T->B, A->B) -> RES/plane_paths.npz
    python landscape_plane.py grid NAME [BUDGET] resumable grid evaluation (main, rsr, cutT, cutT100, cutA, cutB, cutA100,
                                                 cutB100, main_raw)
    python landscape_plane.py summary            -> RES/plane_<grid>.npz (final) and RES/plane_summary.json
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

import landscape_lab as LL

F64 = torch.float64
RES = LL.RES
PFX = f"{RES}/plane_"
GRIDS = ("main", "rsr", "cutT", "cutT100", "cutA", "cutB", "cutA100", "cutB100", "main_raw")
SCAN_T = np.logspace(-12, 0, 97)  # 8 per decade, units of |theta_T|


# ======================================================================== helpers
@torch.no_grad()
def train_rnorm(V, chunk=32):
    """Training residual norm |r| = sqrt(mean ((H - <H>) / es)^2) on the LM training set (96 x 128), as LL.energy_jac's r,
    for V (K, P) or (P,)."""
    L = LL.load_all()
    V = torch.as_tensor(V, dtype=F64)
    one = V.dim() == 1
    V = V[None] if one else V
    X = L.aa.reshape(-1, 2)
    out = []
    for k0 in range(0, V.shape[0], chunk):
        y = LL._forward(V[k0:k0 + chunk], X)
        H = L.t.H(y.reshape(-1, 2)).view(-1, L.nO, L.nS)
        res = (H - H.mean(-1, keepdim=True)) / L.es
        out.append(res.pow(2).mean((-1, -2)).sqrt())
    out = torch.cat(out)
    return out[0] if one else out


def pts(gauge="aligned_min_shift"):
    """T, A, B and the rsr finals as (P,) vectors in a gauge ('raw', 'aligned', 'aligned_min', 'aligned_min_shift')."""
    L = LL.load_all()
    d = dict(raw=L.raw, aligned=L.al, aligned_min=L.al_min, aligned_min_shift=L.al_min_shift)[gauge]
    return dict(T=L.vT, A=d[L.A], B=d[L.B], C=d["rsr_geo_s0"], D=d["rsr_geo_s1"])


def basis(O, P1, P2):
    """Orthonormal u = (P1 - O)/|P1 - O|, w = Gram-Schmidt(P2 - O) (two passes), and the plane coordinates (units of
    |theta_T|) of P1, P2 plus the out-of-plane residual of P2."""
    nT = LL.load_all().nT
    d1, d2 = P1 - O, P2 - O
    u = d1 / d1.norm()
    w = d2 - (d2 @ u) * u
    w = w / w.norm()
    w = w - (w @ u) * u
    w = w / w.norm()
    c1 = (float(d1.norm()) / nT, 0.0)
    c2 = (float(d2 @ u) / nT, float(d2 @ w) / nT)
    off = float((d2 - (d2 @ u) * u - (d2 @ w) * w).norm()) / nT
    return u, w, c1, c2, off


def axis(lo, hi, n, node=None):
    """n nodes s_i = h (i - k0) covering about [lo, hi], with 0 a node (index k0) and, if node (> 0) is given, node a
    node too (h = node / k). Returns s, h, k0, k_node."""
    h = (hi - lo) / (n - 1)
    k = None
    if node is not None:
        k = max(1, int(round(node / h)))
        h = node / k
    k0 = min(max(int(round(-lo / h)), 0), n - 1)
    s = h * (np.arange(n) - k0)
    return s, h, k0, (None if k is None else k0 + k)


def crossing(t, e, level):
    """First t at which e(t) reaches level (log-log interpolation), or None."""
    t, e = np.asarray(t), np.asarray(e)
    idx = np.nonzero(e >= level)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t[0])
    lt0, lt1, le0, le1 = np.log(t[i - 1]), np.log(t[i]), np.log(e[i - 1]), np.log(e[i])
    return float(np.exp(lt0 + (np.log(level) - le0) * (lt1 - lt0) / (le1 - le0)))


def loglog_slope(t, e, lo, hi):
    """Least-squares slope of log e vs log t over the points with lo <= e <= hi (None if fewer than 3)."""
    t, e = np.asarray(t), np.asarray(e)
    k = (e >= lo) & (e <= hi)
    if k.sum() < 3:
        return None
    return float(np.polyfit(np.log10(t[k]), np.log10(e[k]), 1)[0])


# ======================================================================== scans
def cmd_scan():
    L = LL.load_all()
    t0 = time.perf_counter()
    p = pts()
    u, w, cA, cB, off = basis(p["T"], p["A"], p["B"])
    g = torch.Generator().manual_seed(12345)
    rnd = [torch.randn(LL.P, generator=g, dtype=F64) for _ in range(2)]
    rnd = [r / r.norm() for r in rnd]
    dirs = {"+u": u, "-u": -u, "+w": w, "-w": -w, "rand0": rnd[0], "rand1": rnd[1]}
    plan = [("T", k) for k in dirs] + [("A", k) for k in dirs] + [("B", k) for k in ("+u", "-u", "+w", "-w")]
    ts = torch.as_tensor(SCAN_T, dtype=F64)
    out = dict(t=SCAN_T, u=u.numpy(), w=w.numpy(), rand0=rnd[0].numpy(), rand1=rnd[1].numpy())
    centre_vals = {}
    for c in ("T", "A", "B"):
        centre_vals[c] = (float(LL.eps_batch(p[c])), float(train_rnorm(p[c])))
        out[f"{c}_eps0"], out[f"{c}_r0"] = centre_vals[c]
    for c, k in plan:
        V = p[c][None] + (ts[:, None] * L.nT) * dirs[k][None]
        e = LL.eps_batch(V).numpy(); r = train_rnorm(V).numpy()
        out[f"{c}{k}_eps"], out[f"{c}{k}_r"] = e, r
        print(f"{c} {k:5s}: eps at t=1e-12,1e-9,1e-6,1e-3,1: " + " ".join(f"{e[i]:.2e}" for i in (0, 24, 48, 72, 96))
              + f"; min eps {e.min():.3e} at t={SCAN_T[e.argmin()]:.1e}; |r| min {r.min():.3e} (centre {centre_vals[c][1]:.3e})"
              f"  [{time.perf_counter() - t0:.0f} s]", flush=True)
    np.savez(f"{PFX}scan.npz", **out, plan=np.array([f"{c}{k}" for c, k in plan]), gauge="aligned_min_shift",
             note="points centre + t |theta_T| d for t in t (logspace(-12, 0, 97)); eps = LL.eps_batch, r = train_rnorm")
    print(f"scan done in {time.perf_counter() - t0:.0f} s")


def cmd_pick_r():
    z = np.load(f"{PFX}scan.npz")
    t = z["t"]; eA = float(z["A_eps0"]); eT = float(z["T_eps0"])
    rows = {}
    for k in ("+u", "-u", "+w", "-w", "rand0", "rand1"):
        e = z[f"T{k}_eps"]
        rows[k] = {"t_reach_epsA": crossing(t, e, eA), "t_reach_1e-10": crossing(t, e, 1e-10),
                   "slope_1e-13_to_1e-7": loglog_slope(t, e, 1e-13, 1e-7), "slope_1e-10_to_1e-6": loglog_slope(t, e, 1e-10, 1e-6)}
    inplane = [rows[k]["t_reach_epsA"] for k in ("+u", "-u", "+w", "-w")]
    tmax = max(inplane)
    # round up to 1, 2 or 5 x 10^n so every in-plane edge midpoint reaches at least eps_A
    ex = math.floor(math.log10(tmax)); m = tmax / 10 ** ex
    r = next(c for c in (1, 2, 5, 10) if c >= m - 1e-12) * 10 ** ex
    out = dict(r=r, eps_A=eA, eps_T=eT, t_reach_epsA_inplane=dict(zip(("+u", "-u", "+w", "-w"), inplane)), rows=rows,
               rule="r = max over +-u, +-w of the step at which eps(T + t nT d) first reaches eps_A, rounded up to 1/2/5 x 10^n")
    json.dump(out, open(f"{PFX}r.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ======================================================================== paths
def path_t():
    """401 uniform t in [0, 1] plus 60 log-spaced t and 1 - t in [1e-10, 1e-1]. Returns (t, s = 1 - t, from_end) with
    from_end True for points built as P1 + s (P0 - P1) (accurate near t = 1)."""
    lg = np.logspace(-10, -1, 60)
    t = np.concatenate([np.linspace(0, 1, 401), lg, 1 - lg])
    s = np.concatenate([1 - np.linspace(0, 1, 401), 1 - lg, lg])
    o = np.argsort(t, kind="stable")
    t, s = t[o], s[o]
    return t, s, t > 0.5


PATHS = {"TA_raw": ("raw", "T", "A"), "TA_gate": ("aligned", "T", "A"), "TA_min": ("aligned_min", "T", "A"),
         "TA_min_shift": ("aligned_min_shift", "T", "A"), "TB_raw": ("raw", "T", "B"),
         "TB_min_shift": ("aligned_min_shift", "T", "B"), "AB_min_shift": ("aligned_min_shift", "A", "B")}


def cmd_paths():
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PATHS)
    f = f"{PFX}paths.npz"
    old = dict(np.load(f)) if os.path.exists(f) else {}
    t, s, fe = path_t()
    T_, S_, FE = (torch.as_tensor(a) for a in (t, s, fe))
    t0 = time.perf_counter()
    for n in names:
        g, a, b = PATHS[n]
        p = pts(g); P0, P1 = p[a], p[b]
        V = torch.where(FE[:, None], P1[None] + S_[:, None].to(F64) * (P0 - P1)[None], P0[None] + T_[:, None].to(F64) * (P1 - P0)[None])
        e = LL.eps_batch(V).numpy(); r = train_rnorm(V).numpy()
        old[f"{n}_eps"], old[f"{n}_r"] = e, r
        old[f"{n}_len"] = LL.rdist(P0, P1)
        i = int(e.argmax())
        print(f"{n:13s} |P1-P0|/|T| {LL.rdist(P0, P1):.4f}: eps ends {e[0]:.3e} / {e[-1]:.3e}; barrier {e[i]:.4f} at t={t[i]:.4f}; "
              f"|r| ends {r[0]:.3e} / {r[-1]:.3e}, max {r.max():.3e}  [{time.perf_counter() - t0:.0f} s]", flush=True)
    old.update(t=t, s=s, from_end=fe)
    np.savez(f, **old)


# ======================================================================== grids
def grid_spec(name):
    """Grid definition: O (P,), unit u, w (P,), s1 (slow, along u), s2 (along w) in units of |theta_T|, node indices
    and the plane coordinates of the named points."""
    L = LL.load_all()
    if name in ("main", "main_raw", "cutT", "cutT100", "cutA", "cutB", "cutA100", "cutB100"):
        p = pts("raw" if name == "main_raw" else "aligned_min_shift")
        u, w, cA, cB, off = basis(p["T"], p["A"], p["B"])
        named = dict(T=(0.0, 0.0), A=cA, B=cB)
        n1, n2 = (121, 121) if name == "main" else (41, 41) if name == "main_raw" else (61, 61) if name in ("cutA100", "cutB100") else (101, 101)
        if name in ("main", "main_raw"):
            xs, ys = [0.0, cA[0], cB[0]], [0.0, cA[1], cB[1]]
            mx, my = 0.2 * (max(xs) - min(xs)), 0.2 * (max(ys) - min(ys))
            s1, h1, i0, iA = axis(min(xs) - mx, max(xs) + mx, n1, node=cA[0])
            s2, h2, j0, _ = axis(min(ys) - my, max(ys) + my, n2)
            O, centre = p["T"], "T"
            nodes = dict(T=[i0, j0], A=[iA, j0])
        else:
            r = json.load(open(f"{PFX}r.json"))["r"] * (0.01 if name.endswith("100") else 1.0)
            s1, h1, i0, _ = axis(-r, r, n1); s2, h2, j0, _ = axis(-r, r, n2)
            centre = name[3]
            O = p[centre]
            nodes = {centre: [i0, j0]}
            named = {k: (v[0] - named[centre][0], v[1] - named[centre][1]) for k, v in named.items()}
        gauge = "raw" if name == "main_raw" else "aligned_min_shift"
        return dict(name=name, O=O, u=u, w=w, s1=s1, s2=s2, h1=h1, h2=h2, centre=centre, nodes=nodes, named=named,
                    off_plane_B=off, gauge=gauge, P0="T", P1="A", P2="B")
    if name == "rsr":
        p = pts("aligned_min_shift")
        u, w, cC, cD, off = basis(p["T"], p["C"], p["D"])
        xs, ys = [0.0, cC[0], cD[0]], [0.0, cC[1], cD[1]]
        mx, my = 0.2 * (max(xs) - min(xs)), 0.2 * (max(ys) - min(ys))
        s1, h1, i0, iC = axis(min(xs) - mx, max(xs) + mx, 81, node=cC[0])
        s2, h2, j0, _ = axis(min(ys) - my, max(ys) + my, 81)
        return dict(name=name, O=p["T"], u=u, w=w, s1=s1, s2=s2, h1=h1, h2=h2, centre="T", nodes=dict(T=[i0, j0], C=[iC, j0]),
                    named=dict(T=(0.0, 0.0), C=cC, D=cD), off_plane_B=off, gauge="aligned_min_shift", P0="T", P1="rsr_geo_s0",
                    P2="rsr_geo_s1")
    raise KeyError(name)


def _vectors(spec, rows):
    nT = LL.load_all().nT
    return LL.plane(spec["O"], nT * spec["u"], nT * spec["w"], spec["s1"][rows], spec["s2"])


def cmd_grid():
    name = sys.argv[2]
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    spec = grid_spec(name)
    n1, n2 = len(spec["s1"]), len(spec["s2"])
    f = f"{PFX}{name}_partial.npz"
    sig = np.array([float(spec["O"].sum()), float(spec["u"] @ torch.arange(LL.P, dtype=F64)), float(spec["w"].sum())])
    if os.path.exists(f):
        z = np.load(f)
        assert np.array_equal(z["s1"], spec["s1"]) and np.array_equal(z["s2"], spec["s2"]) and np.array_equal(z["sig"], sig), "spec changed"
        E = z["eps"].copy(); dt = z["dt"].copy()
    else:
        E = np.full((n1, n2), np.nan); dt = np.full(n1, np.nan)
    t0 = time.perf_counter()
    todo = [i for i in range(n1) if np.isnan(E[i]).any()]
    print(f"{name}: {n1}x{n2}, {len(todo)} rows to do, h = ({spec['h1']:.4e}, {spec['h2']:.4e}) |T|", flush=True)
    per_row = None
    for i in todo:
        if per_row is not None and time.perf_counter() - t0 + per_row > budget:
            break
        ts = time.perf_counter()
        E[i] = LL.eps_batch(_vectors(spec, [i])).numpy()
        dt[i] = time.perf_counter() - ts
        per_row = dt[i] if per_row is None else max(per_row, dt[i])
        np.savez(f, eps=E, dt=dt, s1=spec["s1"], s2=spec["s2"], sig=sig)
    left = int(np.isnan(E).any(1).sum())
    print(f"{name}: {len(todo) - left} rows in {time.perf_counter() - t0:.0f} s; {left} rows left", flush=True)


# ======================================================================== summary
def local_minima(E):
    """Interior nodes lower than all 8 neighbours: list of (i, j, eps)."""
    out = []
    for i in range(1, E.shape[0] - 1):
        for j in range(1, E.shape[1] - 1):
            nb = E[i - 1:i + 2, j - 1:j + 2].copy(); c = nb[1, 1]; nb[1, 1] = np.inf
            if c < nb.min():
                out.append((i, j, float(c)))
    return sorted(out, key=lambda x: x[2])


def cmd_summary():
    L = LL.load_all()
    S = dict(units="parameter distances and plane coordinates in units of |theta_T|", theta_T_norm=L.nT,
             gauge_default="aligned_min_shift (LL: gate scale chosen to minimise |S - T|, unit sign/permutation matching, 14 translation gauges)",
             eps="LL.eps_batch: mean |dE/E| on 100 tori x 128 angles (landscape set)",
             train_r="|r| = sqrt(mean((H-<H>)/(nu0 J))^2) on the 96 x 128 LM training set", A=L.A, B=L.B)
    p = pts(); pr = pts("raw")
    S["points"] = {k: dict(eps=float(LL.eps_batch(v)), train_r=float(train_rnorm(v))) for k, v in p.items()}
    S["points_raw_eps"] = {k: float(LL.eps_batch(v)) for k, v in pr.items() if k != "T"}
    S["dist_min_shift"] = dict(AT=LL.rdist(p["A"]), BT=LL.rdist(p["B"]), AB=LL.rdist(p["A"], p["B"]), CT=LL.rdist(p["C"]),
                               DT=LL.rdist(p["D"]), CD=LL.rdist(p["C"], p["D"]))
    S["dist_raw"] = dict(AT=LL.rdist(pr["A"]), BT=LL.rdist(pr["B"]), AB=LL.rdist(pr["A"], pr["B"]))
    eA = S["points"]["A"]["eps"]
    for name in GRIDS:
        f = f"{PFX}{name}_partial.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f); E = z["eps"]
        if np.isnan(E).any():
            S[f"grid_{name}"] = dict(incomplete_rows=int(np.isnan(E).any(1).sum()))
            continue
        spec = grid_spec(name)
        # exact named points (not nodes in general): eps of the stored vectors, and of the plane reconstruction
        nT = L.nT
        exact = {}
        for k, (a, b) in spec["named"].items():
            vk = spec["O"] + nT * (a * spec["u"] + b * spec["w"])
            exact[k] = dict(coords=[a, b], eps_plane_reconstruction=float(LL.eps_batch(vk)))
        lg = np.log10(E)
        mins = local_minima(E)
        info = dict(shape=list(E.shape), h1=spec["h1"], h2=spec["h2"], s1_range=[float(spec["s1"][0]), float(spec["s1"][-1])],
                    s2_range=[float(spec["s2"][0]), float(spec["s2"][-1])], centre=spec["centre"], nodes=spec["nodes"],
                    named=exact, gauge=spec["gauge"], off_plane_P2=spec["off_plane_B"],
                    log10eps_min=float(lg.min()), log10eps_median=float(np.median(lg)), log10eps_max=float(lg.max()),
                    argmin=[int(x) for x in np.unravel_index(E.argmin(), E.shape)], eps_min=float(E.min()),
                    frac_le_2epsA=float((E <= 2 * eA).mean()), frac_within_2x_epsA=float(((E <= 2 * eA) & (E >= eA / 2)).mean()),
                    frac_le_10epsA=float((E <= 10 * eA).mean()), frac_le_1em3=float((E <= 1e-3).mean()),
                    frac_le_1em2=float((E <= 1e-2).mean()), n_local_minima=len(mins), local_minima_lowest10=mins[:10],
                    seconds=float(np.nansum(z["dt"])))
        for k, (i, j) in spec["nodes"].items():
            info[f"eps_node_{k}"] = float(E[i, j])
        # 4 edge midpoints and corners
        c1, c2 = E.shape[0] // 2, E.shape[1] // 2
        info["eps_edges"] = {"-u": float(E[0, c2]), "+u": float(E[-1, c2]), "-w": float(E[c1, 0]), "+w": float(E[c1, -1])}
        info["eps_corners"] = [float(E[0, 0]), float(E[0, -1]), float(E[-1, 0]), float(E[-1, -1])]
        S[f"grid_{name}"] = info
        np.savez(f"{PFX}{name}.npz", log10eps=lg, eps=E, s1=spec["s1"], s2=spec["s2"], h1=spec["h1"], h2=spec["h2"],
                 O=spec["O"].numpy(), u=spec["u"].numpy(), w=spec["w"].numpy(), nT=nT, centre=spec["centre"],
                 node_names=np.array(list(spec["nodes"])), node_ij=np.array(list(spec["nodes"].values())),
                 named_names=np.array(list(exact)), named_coords=np.array([exact[k]["coords"] for k in exact]),
                 named_eps=np.array([exact[k]["eps_plane_reconstruction"] for k in exact]), gauge=spec["gauge"],
                 axes="eps[i, j] at O + nT (s1[i] u + s2[j] w); s1 (slow) along u, s2 along w; units |theta_T|",
                 P0=spec["P0"], P1=spec["P1"], P2=spec["P2"])
        print(f"{name}: log10 eps min/median/max {lg.min():.2f}/{np.median(lg):.2f}/{lg.max():.2f}; frac <= 2 eps_A "
              f"{info['frac_le_2epsA']:.4f}; local minima {len(mins)}", flush=True)
    if all(f"grid_{n}" in S and "shape" in S[f"grid_{n}"] for n in ("cutT", "cutT100")):
        a, b = np.load(f"{PFX}cutT.npz")["eps"], np.load(f"{PFX}cutT100.npz")["eps"]
        m = np.ones_like(a, bool); m[a.shape[0] // 2, a.shape[1] // 2] = False
        q = a[m] / (100 * b[m])
        S["scale_free_cutT_over_100cutT100"] = dict(min=float(q.min()), max=float(q.max()),
                                                    note="eps on the r grid / (100 x eps on the r/100 grid), all nodes but T")
    if os.path.exists(f"{PFX}dirs.json"):
        S["dirs"] = json.load(open(f"{PFX}dirs.json"))
    if os.path.exists(f"{PFX}r.json"):
        S["cutout_r"] = json.load(open(f"{PFX}r.json"))
    if os.path.exists(f"{PFX}paths.npz"):
        z = np.load(f"{PFX}paths.npz"); t = z["t"]
        S["paths"] = {}
        for n in PATHS:
            if f"{n}_eps" not in z.files:
                continue
            e, r = z[f"{n}_eps"], z[f"{n}_r"]; i = int(e.argmax())
            S["paths"][n] = dict(length=float(z[f"{n}_len"]), eps_t0=float(e[0]), eps_t1=float(e[-1]), barrier_eps=float(e[i]),
                                 barrier_t=float(t[i]), barrier_over_eps_A=float(e[i] / eA), train_r_max=float(r.max()),
                                 train_r_t_at_max=float(t[int(r.argmax())]),
                                 eps_at_t=dict((f"{tt:g}", float(np.interp(tt, t, e))) for tt in (1e-8, 1e-6, 1e-4, 1e-2, 0.25, 0.5, 0.75)),
                                 eps_at_1mt=dict((f"{ss:g}", float(e[np.argmin(np.abs(z["s"] - ss))])) for ss in (1e-8, 1e-6, 1e-4, 1e-2)),
                                 n_local_minima_interior=int(((e[1:-1] < e[:-2]) & (e[1:-1] < e[2:])).sum()))
    if os.path.exists(f"{PFX}scan.npz"):
        z = np.load(f"{PFX}scan.npz"); t = z["t"]
        S["scans"] = {}
        for k in z["plan"]:
            k = str(k); e, r = z[f"{k}_eps"], z[f"{k}_r"]; c = k[0]
            S["scans"][k] = dict(eps_at=dict((f"{tt:g}", float(e[np.argmin(np.abs(np.log(t / tt)))])) for tt in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1)),
                                 t_reach_epsA=crossing(t, e, eA) if c == "T" else crossing(t, e, 2 * float(z[f"{c}_eps0"])),
                                 level="eps_A" if c == "T" else f"2 eps_{c}",
                                 eps_min=float(e.min()), t_at_eps_min=float(t[e.argmin()]), centre_eps=float(z[f"{c}_eps0"]),
                                 r_min=float(r.min()), t_at_r_min=float(t[r.argmin()]), centre_r=float(z[f"{c}_r0"]),
                                 slope_eps_1em13_1em7=loglog_slope(t, e, 1e-13, 1e-7) if c == "T" else None)
    json.dump(S, open(f"{PFX}summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if not k.startswith(("grid_", "scans", "paths"))}, indent=1))


# ======================================================================== directional stiffness
def cmd_dirs():
    """|J_X d| (training energy Jacobian at X in T, A, B, aligned_min_shift) for the plane directions u, w, the scan's
    random directions and the radial direction (X - T)/|X - T|, against the RMS over random directions
    sqrt(tr(J^T J)/P); and how each direction's squared norm splits over parameter types -> RES/plane_dirs.json."""
    z = np.load(f"{PFX}scan.npz")
    d = {k: torch.from_numpy(z[k]) for k in ("u", "w", "rand0", "rand1")}
    p = pts()
    out = dict(note=__doc__ and cmd_dirs.__doc__)
    types = dict(gate=[LL.G], W1=list(range(1, 65)), b1=list(range(65, 129)), W2=list(range(129, 193)), b2=[LL.B2])
    for k in ("u", "w"):
        bl = LL.blocks(d[k].clone()).pow(2)
        out[f"type_fraction_{k}"] = {n: float(bl[:, ix].sum()) for n, ix in types.items()}
        out[f"layer_fraction_{k}"] = [float(x) for x in bl.sum(1)]
    for X in ("T", "A", "B"):
        Jm, r = LL.energy_jac(p[X])
        rms = float(Jm.pow(2).sum().sqrt()) / math.sqrt(LL.P)
        dd = dict(d)
        if X != "T":
            dd["radial"] = (p[X] - p["T"]) / (p[X] - p["T"]).norm()
        row = {k: float((Jm @ v).norm()) for k, v in dd.items()}
        out[X] = dict(gain=row, rms_random=rms, gain_over_rms={k: v / rms for k, v in row.items()},
                      grad_norm=float((Jm.T @ r).norm()), r_norm=float(r.norm()),
                      grad_dot={k: float((Jm.T @ r) @ v) for k, v in dd.items()})
        # gain of each parameter type at X (RMS over unit vectors supported on that type)
        cn = Jm.pow(2).sum(0).view(LL.N_LAYERS, LL.PER)
        out[X]["type_rms_gain"] = {n: float(cn[:, ix].sum().sqrt()) / math.sqrt(cn[:, ix].numel()) for n, ix in types.items()}
        print(X, json.dumps(out[X]), flush=True)
        del Jm
    json.dump(out, open(f"{PFX}dirs.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("type")}, indent=1))


if __name__ == "__main__":
    {"scan": cmd_scan, "pick_r": cmd_pick_r, "paths": cmd_paths, "grid": cmd_grid, "summary": cmd_summary,
     "dirs": cmd_dirs}[sys.argv[1]]()
