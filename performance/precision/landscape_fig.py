"""Paper figures for the loss-landscape study (reads only results/landscape/, writes audit/figs/landscape/).

    python3 landscape_fig.py [numbers] [fig1] [figS1] [fig2] [json]      (no argument = all)

numbers  results/landscape/fig_numbers.json: every derived number quoted on the figures or in the captions (widths by
         log-interpolation of the saved scans, slopes, contour shape, grid statistics, spectrum statistics, twist medians,
         margins), computed from the saved npz/json only.
fig1     3D log10 mean|dE/E| over the T-A-B plane (aligned min+shift gauge), with the sub-grid pits drawn as needles,
         equal-scale cutouts around T and A, and the same cross-sections through T, A, B on a linear eps axis.
figS1    2D companion maps: three planes (T-A-B min+shift, T-C-D min+shift, T-A-B raw) and cutouts around T, A, B.
fig2     sloppiness and the structure of T's basin: J spectra (with the true curvature at A), twist fraction of the J_E
         singular vectors at T, 1D profiles from T with the LM start/end points, straight-line half-widths at equal
         absolute error.
json     results/landscape/interactive.json (grids + markers + metadata for an interactive 3D page).

No computation of eps happens here: every number comes from the npz/json files written by landscape_lab.py,
landscape_plane.py, landscape_sloppy.py, landscape_profiles.py, and from the verification checks copied into
results/landscape/verify/ (gn_A.json, valley.json, rplane.json, hess_TA.json, eps_direct.json, slope_twnull1.json,
indep_gn_fdhess_A.json).
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

RES = "/Users/gabrielpfaffman/Repos/orbitflows/performance/precision/results/landscape"
FIG = "/Users/gabrielpfaffman/Repos/orbitflows/performance/audit/figs/landscape"

# ---------------------------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
CMAP = "viridis"
ZLO, ZHI = -16.0, 0.0          # one colour normalisation for every panel of figure 1
NORM = mcolors.Normalize(ZLO, ZHI)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "lines.linewidth": 1.0, "legend.frameon": False, "savefig.dpi": 200,
    "pdf.fonttype": 42, "mathtext.default": "regular",
})


def load(name):
    return np.load(os.path.join(RES, name), allow_pickle=True)


def jload(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def sci(x, d=1):
    """Mathtext a x 10^b."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return rf"{m:.{d}f}\times10^{{{e}}}"


def sci0(x):
    """Mathtext 10^b for a power of ten, else a x 10^b with one digit."""
    e = int(np.round(np.log10(abs(x))))
    return rf"10^{{{e}}}" if abs(x / 10 ** e - 1) < 1e-9 else sci(x)


def save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), dpi=200)
    print("wrote", os.path.join(FIG, stem + ".{png,pdf}"))


def replace_node(L, i, j):
    """Copy of L with node (i, j) set to the mean of its 8 neighbours (for drawing a sub-grid pit as a needle)."""
    L = L.copy()
    nb = [L[i + a, j + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    L[i, j] = float(np.mean(nb))
    return L


def bilinear(s1, s2, L, x, y):
    i = int(np.searchsorted(s1, x) - 1); j = int(np.searchsorted(s2, y) - 1)
    fx = (x - s1[i]) / (s1[i + 1] - s1[i]); fy = (y - s2[j]) / (s2[j + 1] - s2[j])
    return float((1 - fx) * (1 - fy) * L[i, j] + fx * (1 - fy) * L[i + 1, j] + (1 - fx) * fy * L[i, j + 1] + fx * fy * L[i + 1, j + 1])


def needle(ax, x, y, ztop, zbot, lw=1.5):
    """A vertical line from ztop down to zbot, coloured along its length with the surface colormap, over a dark halo."""
    zs = np.linspace(ztop, zbot, 200)
    seg = [[(x, y, zs[k]), (x, y, zs[k + 1])] for k in range(len(zs) - 1)]
    halo = Line3DCollection(seg, colors="#1b1b1b", linewidths=lw + 1.4, zorder=20)
    ax.add_collection3d(halo)
    lc = Line3DCollection(seg, cmap=CMAP, norm=NORM, linewidths=lw, zorder=21)
    lc.set_array(0.5 * (zs[:-1] + zs[1:]))
    ax.add_collection3d(lc)


def style3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = GRID
        a._axinfo["grid"]["linewidth"] = 0.4
        a.line.set_linewidth(0.5); a.line.set_color(INK2)
    ax.tick_params(axis="both", which="major", pad=0, labelsize=6.2)
    ax.tick_params(axis="z", pad=1)


def surface(ax, X, Y, Z, rc=None):
    n1, n2 = Z.shape
    Zc = np.clip(Z, ZLO, ZHI)
    fc = plt.get_cmap(CMAP)(NORM(Zc))
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=55)
    # gentle hill-shading so the ruggedness of the surface is visible even where the colour hardly changes
    sh = ls.shade_rgb(fc[..., :3], Zc, vert_exag=1.0, blend_mode="soft", fraction=0.6)
    fc = np.concatenate([0.55 * fc[..., :3] + 0.45 * sh, np.ones(Z.shape + (1,))], -1)
    srf = ax.plot_surface(X, Y, Zc, facecolors=fc, rcount=rc or n1, ccount=rc or n2, linewidth=0, antialiased=False,
                          shade=False, zorder=1)
    srf.set_rasterized(True)   # avoids hairline seams between the 14k facets in PDF viewers


# ---------------------------------------------------------------------------------------------- derived numbers
def _reach(t, e, lev):
    """First t at which e(t) >= lev, log-log interpolated between scan points (None if never)."""
    k = np.nonzero(e >= lev)[0]
    if len(k) == 0:
        return None
    k = int(k[0])
    if k == 0:
        return float(t[0])
    lt, le = np.log10(t), np.log10(e)
    f = (np.log10(lev) - le[k - 1]) / (le[k] - le[k - 1])
    return float(10 ** (lt[k - 1] + f * (lt[k] - lt[k - 1])))


def numbers(write=True):
    """Every derived number used on the figures and in the captions (from saved data only)."""
    from scipy.interpolate import RegularGridInterpolator
    ps = jload("plane_summary.json"); P = ps["points"]
    epsT, epsA, epsB = P["T"]["eps"], P["A"]["eps"], P["B"]["eps"]
    ed = jload("verify/eps_direct.json"); ck = jload("check.json")
    N = dict(eps_T_batched=epsT, eps_T_direct=ed["T"], eps_T_heldout=ck["T"]["heldout"], eps_A=epsA, eps_B=epsB,
             eps_A_heldout=ck["A"]["heldout"], eps_B_heldout=ck["B"]["heldout"],
             theta_T_norm=ps["theta_T_norm"])
    # ---- reaches along the in-plane and random scans (plane_scan.npz, 97 points per scan, 8 per decade)
    z = load("plane_scan.npz"); t = z["t"]
    lv = {"epsA": epsA, "2epsA": 2 * epsA, "10epsA": 10 * epsA}
    reach = {}
    for c in ("T", "A", "B"):
        for d in ("+u", "-u", "+w", "-w", "rand0", "rand1"):
            k = f"{c}{d}"
            if k + "_eps" not in z:
                continue
            reach[k] = {ln: _reach(t, z[k + "_eps"], v) for ln, v in lv.items()}
            if c == "B":
                reach[k]["2epsB"] = _reach(t, z[k + "_eps"], 2 * epsB)
    N["reach"] = reach
    half = lambda c, ax, ln: 0.5 * (reach[f"{c}+{ax}"][ln] + reach[f"{c}-{ax}"][ln])
    N["half_2epsA"] = {c: {ax: half(c, ax, "2epsA") for ax in "uw"} for c in "TAB"}
    N["half_10epsA"] = {c: {ax: half(c, ax, "10epsA") for ax in "uw"} for c in "TAB"}
    N["half_epsA_T"] = {ax: half("T", ax, "epsA") for ax in "uw"}
    N["half_2epsB_B"] = {ax: half("B", ax, "2epsB") for ax in "uw"}
    # ---- linearity from T: fit over t in [1e-12, 1e-6]; t where eps first departs 1% from c t
    lin = {}
    for d in ("+u", "-u", "+w", "-w", "rand0", "rand1"):
        e = z[f"T{d}_eps"]; s = (t >= 1e-12) & (t <= 1e-6)
        p = np.polyfit(np.log10(t[s]), np.log10(e[s]), 1)
        res = np.log10(e[s]) - np.polyval(p, np.log10(t[s]))
        c = e[0] / t[0]; dev = np.abs(e / (c * t) - 1); k = np.nonzero(dev > 0.01)[0]
        lin[d] = {"slope_1e-12_1e-6": float(p[0]), "max_log10_resid": float(np.abs(res).max()), "eps_over_t": float(c),
                  "t_1pct": float(t[k[0]]) if len(k) else None, "eps_1pct": float(e[k[0]]) if len(k) else None}
    N["linearity_T"] = lin
    # ---- grid geometry and statistics
    gm = ps["grid_main"]; h1, h2 = gm["h1"], gm["h2"]
    N["cell"] = [h1, h2]
    hw = N["half_2epsA"]
    ratios = [h1 / hw[c]["u"] for c in "TAB"] + [h2 / hw[c]["w"] for c in "TAB"]
    N["cell_over_half_2epsA"] = dict(min=float(min(ratios)), max=float(max(ratios)),
                                     T=[h1 / hw["T"]["u"], h2 / hw["T"]["w"]], A=[h1 / hw["A"]["u"], h2 / hw["A"]["w"]],
                                     B=[h1 / hw["B"]["u"], h2 / hw["B"]["w"]])
    m = load("plane_main.npz"); E = m["eps"]
    other = np.ones_like(E, bool)
    nb = {}
    for nm, (i, j) in zip(m["node_names"], m["node_ij"]):
        other[i, j] = False
        n3 = [E[i + a, j + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
        nb[str(nm)] = [float(min(n3)), float(max(n3))]
    N["main"] = dict(median=float(np.median(E)), p1=float(np.percentile(E, 1)), p99=float(np.percentile(E, 99)),
                     frac_above_0p1=float((E > 0.1).mean()), lowest_other=float(E[other].min()),
                     nbr8_range=nb, frac_le_2epsA=float((E <= 2 * epsA).mean()))
    # nearest node to B
    bu, bw = m["named_coords"][2]
    i = int(np.argmin(np.abs(m["s1"] - bu))); j = int(np.argmin(np.abs(m["s2"] - bw)))
    N["main"]["B_nearest_node_eps"] = float(E[i, j])
    for f, k in (("plane_rsr.npz", "rsr"), ("plane_main_raw.npz", "raw")):
        g = load(f); Eg = g["eps"]; ot = np.ones_like(Eg, bool)
        for (i, j) in g["node_ij"]:
            ot[i, j] = False
        N[k] = dict(median=float(np.median(Eg)), lowest_other=float(Eg[ot].min()))
    pa = ps["paths"]
    N["barriers"] = {k: float(v["barrier_eps"]) for k, v in pa.items()}
    # ---- T's eps_A contour in the cutout (linear regime): reach along every angle vs the axis-aligned ellipse
    c = load("plane_cutT.npz"); f = RegularGridInterpolator((c["s1"], c["s2"]), c["eps"])
    th = np.linspace(0, 2 * np.pi, 721)[:-1]; rho = 2e-7
    rch = epsA * rho / f(np.c_[rho * np.cos(th), rho * np.sin(th)])
    a, b = N["half_epsA_T"]["u"], N["half_epsA_T"]["w"]
    ell = 1 / np.sqrt(np.cos(th) ** 2 / a ** 2 + np.sin(th) ** 2 / b ** 2)
    M = np.c_[np.cos(th) ** 2, 2 * np.cos(th) * np.sin(th), np.sin(th) ** 2]
    q, *_ = np.linalg.lstsq(M, 1 / rch ** 2, rcond=None)
    w_, V_ = np.linalg.eigh(np.array([[q[0], q[1]], [q[1], q[2]]]))
    tilt = float(np.degrees(np.arctan2(V_[1, 0], V_[0, 0])))   # direction of the major semi-axis (smallest eigenvalue)
    tilt = (tilt + 90) % 180 - 90
    N["contour_T_epsA"] = dict(ratio_to_axis_ellipse=[float((rch / ell).min()), float((rch / ell).max())],
                               fit_semi_axes=[float(1 / np.sqrt(w_[0])), float(1 / np.sqrt(w_[1]))], fit_tilt_deg=tilt,
                               fit_max_dev=float(np.abs(rch * np.sqrt(M @ q) - 1).max()))
    # ---- linear cross-sections of the cutouts: (edge - centre) / r along +-u, +-w
    vs = {}
    for k, fn in (("T", "plane_cutT.npz"), ("A", "plane_cutA.npz"), ("B", "plane_cutB.npz")):
        cc = load(fn); Ec = cc["eps"]; e0 = Ec[50, 50]; r = float(cc["s1"][-1])
        vs[k] = dict(u=[(Ec[100, 50] - e0) / r, (Ec[0, 50] - e0) / r], w=[(Ec[50, 100] - e0) / r, (Ec[50, 0] - e0) / r],
                     max_over_centre=float(Ec.max() / e0))
        vs[k]["frac_le_2epsA"] = float((Ec <= 2 * epsA).mean())
    N["vslopes"] = vs
    # ---- verification checks
    gn = jload("verify/gn_A.json"); ig = jload("verify/indep_gn_fdhess_A.json")
    rel = lambda h: [h[k + 1] / h[k] - 1 for k in range(len(h) - 1)]
    N["gn_A"] = dict(verifier_hist=gn["iter_hist"], verifier_rel_steps=rel(gn["iter_hist"]), verifier_moved=gn["iter_moved"],
                     indep_hist=ig["gn"]["r_hist"], indep_rel_steps=rel(ig["gn"]["r_hist"]), indep_moved=ig["gn"]["moved_over_nT"])
    hs = jload("verify/hess_TA.json")
    N["hess_A"] = [dict(name=r["name"], sigma_rel=r["sigma_rel"], GN=r["GN"], resid=r["resid_term"], full=r["full"])
                   for r in hs["A"]["rows"] if r["name"].startswith("sv")]
    N["hess_A_fd"] = ig["hess_A"]["rows"]
    vl = jload("verify/valley.json")
    N["valley"] = {k: {f"{r['d']:.0e}": r["ratio"] for r in vl[k]["rows"] if r["sign"] == 1} for k in ("A", "B")}
    rp = jload("verify/rplane.json")
    N["rplane_r_equiv"] = {k: rp[k]["r_equiv"] for k in ("T", "A_raw", "A_ms")}
    tn = jload("verify/slope_twnull1.json")
    N["twnull1_local_slope"] = {f"{a:.2e}": [float(e), float(s)] for a, e, s in
                                zip(tn["ts"][:-1], tn["eps_plus"][:-1], tn["local_slope_eps_plus"])}
    # ---- spectra
    fl = jload("sloppy_floor.json"); F = load("sloppy_floor.npz")
    sp = {}
    for k in ("T_energy", "T_coord", "A_energy"):
        S = F[f"S_{k}"]; r = S / S[0]; n = fl[k]["n_resolved"]; lr = np.log10(r[:n])
        sp[k] = dict(n_resolved=n, n_unresolved=int(len(S) - n), floor_rel=fl[k]["floor_rel"], span_decades=float(-lr[-1]),
                     max_gap_decades=float((-np.diff(lr)).max()), frac_unresolved=float(1 - n / len(S)),
                     n_above_1em4=int((r >= 1e-4).sum()),
                     per_decade=[int(((r >= 10.0 ** (d - 1)) & (r < 10.0 ** d)).sum()) for d in range(0, -17, -1)])
    N["spectra"] = sp
    N["spectra"]["T_energy"]["fresh_gesvd_first_10pct_disagreement_index"] = 1313
    # ---- twist fraction medians (fig 2b): resolved per direction AND above the SVD-reproducibility floor of panel a
    tw = load("sloppy_twist_T.npz")
    S0 = float(tw["S"][0]); reld = tw["sig_dir"] / S0
    okd = tw["sig_dir"] > 10 * np.maximum(tw["sig_noise"], tw["sig_bound"])
    flr = fl["T_energy"]["floor_rel"]; ok2 = okd & (reld >= flr)
    ftw = tw["twist"] / tw["tot"]
    meds = []
    for e0 in range(-17, 0):
        sel = ok2 & (reld >= 10.0 ** e0) & (reld < 10.0 ** (e0 + 1))
        if sel.sum() >= 5:
            meds.append(dict(decade=f"[1e{e0},1e{e0 + 1})", n=int(sel.sum()), median=float(np.median(ftw[sel]))))
    N["twist_T"] = dict(n_resolved_per_direction=int(okd.sum()), n_below_svd_floor=int((okd & (reld < flr)).sum()),
                        n_plotted_coloured=int(ok2.sum()), medians=meds,
                        random_median=jload("sloppy_twist_T.json")["baselines"]["random200"]["twist_frac_median"])
    # ---- straight-line widths at 2 eps_A around T (profiles) and the extra directions (sloppy_line)
    prs = jload("profiles_summary.json"); T = prs["T"]
    w2 = lambda en: float(np.mean([en[f"t_2eps_A{s}"] for s in "+-" if en.get(f"t_2eps_A{s}") is not None]))
    N["width_T_2epsA"] = {k: w2(T[k]) for k in T if k != "shift0" and any(T[k].get(f"t_2eps_A{s}") for s in "+-")}
    sl = jload("sloppy_line.json"); ts = np.array(sl["t_over_thetaT"])
    N["width_T_2epsA_line"] = {k: _reach(ts, np.array(sl[k]["eps"]), 2 * epsA)
                               for k in ("redundancy", "gate_scale_tangent_l4", "twist_like_1e-08", "twist_like_1e-14")}
    WC = prs["width_comparison"]
    N["width_A_2epsA"] = {k: float(np.mean(v["2eps_A"])) for k, v in WC.items() if not k.startswith("T:")}
    dist = jload("profiles_dist.json")
    ends = {k: r["dist_min_shift"] for k, r in dist.items() if r["role"] == "end" and r["outcome"] == "stalled"}
    dmin = min(ends.values())
    sym = {"gate0", "redundancy", "gate_scale_tangent_l4"}
    allw = {**{f"T:{k}": v for k, v in N["width_T_2epsA"].items()}, **{f"T:{k}": v for k, v in N["width_T_2epsA_line"].items()},
            **{f"A:{k}": v for k, v in N["width_A_2epsA"].items()}}
    nonsym = {k: v for k, v in allw.items() if k.split(":", 1)[1] not in sym}
    kmax = max(nonsym, key=nonsym.get)
    N["margin"] = dict(students_min_dist=dmin, students_min_name=min(ends, key=ends.get),
                       students_max_dist=max(ends.values()),
                       largest_nonsym_width=[kmax, nonsym[kmax]], decades_nonsym=float(np.log10(dmin / nonsym[kmax])),
                       gate0=N["width_T_2epsA"]["gate0"], decades_gate0=float(np.log10(dmin / N["width_T_2epsA"]["gate0"])),
                       redundancy=N["width_T_2epsA_line"]["redundancy"],
                       decades_redundancy=float(np.log10(dmin / N["width_T_2epsA_line"]["redundancy"])),
                       sloppy_final_dist=dist["sloppy_final"]["dist_min_shift"], sloppy_final_eps=dist["sloppy_final"]["eps_land"],
                       rand4_final_eps=dist["rand4_final"]["eps_land"])
    if write:
        with open(os.path.join(RES, "fig_numbers.json"), "w") as fh:
            json.dump(N, fh, indent=1)
        print("wrote", os.path.join(RES, "fig_numbers.json"))
    return N


# ---------------------------------------------------------------------------------------------- figure 1
def fig1():
    N = numbers(write=False)
    ps = jload("plane_summary.json")
    m = load("plane_main.npz"); cT = load("plane_cutT.npz"); cA = load("plane_cutA.npz")
    s1, s2, L = m["s1"], m["s2"], m["log10eps"]
    xy = m["named_coords"]
    epsT, epsA, epsB = N["eps_T_batched"], N["eps_A"], N["eps_B"]
    (iT, jT), (iA, jA) = [tuple(v) for v in m["node_ij"]]
    Ld = replace_node(replace_node(L, iT, jT), iA, jA)
    span_u, span_w = s1[-1] - s1[0], s2[-1] - s2[0]
    r = float(ps["cutout_r"]["r"])
    lzA = np.log10(epsA)
    h2 = N["half_2epsA"]

    W, H = 7.0, 9.15
    fig = plt.figure(figsize=(W, H))
    F = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]   # inches -> figure fraction

    # ---------------- (a) main plane
    ax = fig.add_axes(F(-0.53, 4.85, 5.74, 4.42), projection="3d", computed_zorder=False)
    style3d(ax)
    X, Y = np.meshgrid(s1, s2, indexing="ij")
    surface(ax, X, Y, Ld)
    bx = [s1[0], s1[-1], s1[-1], s1[0], s1[0]]; by = [s2[0], s2[0], s2[-1], s2[-1], s2[0]]
    ax.plot_surface(np.array([[s1[0], s1[0]], [s1[-1], s1[-1]]]), np.array([[s2[0], s2[-1]], [s2[0], s2[-1]]]),
                    np.full((2, 2), lzA), color="#9c9a92", alpha=0.18, linewidth=0, shade=False, zorder=0)
    ax.plot(bx, by, [lzA] * 5, color=MUTED, lw=0.6, zorder=0)
    ax.text(s1[0], s2[0], lzA - 0.35, r"plane $\epsilon=\epsilon_A$", color=INK2, fontsize=6.0,
            ha="right", va="top", zorder=30)
    pts = {"T": (xy[0], epsT), "A": (xy[1], epsA), "B": (xy[2], epsB)}
    for k, ((x, y), e) in pts.items():
        ztop = bilinear(s1, s2, Ld, x, y)
        needle(ax, x, y, ztop, np.log10(e))
        ax.scatter([x], [y], [np.log10(e)], s=14, color=plt.get_cmap(CMAP)(NORM(np.log10(e))), edgecolor="#1b1b1b",
                   linewidth=0.6, zorder=25, depthshade=False)
    ax.text(xy[0][0] + 0.25, xy[0][1] - 0.1, np.log10(epsT) - 0.1,
            "teacher T\n" + r"$\epsilon\approx5\times10^{-16}$" + "\n(float64 rounding\nof an exact zero)",
            fontsize=6.4, ha="left", va="center", zorder=30, linespacing=1.25)
    ax.text(xy[1][0] + 0.12, xy[1][1] - 0.1, np.log10(epsA) - 1.5,
            "student A\n(std LM, seed 1)\n" + rf"$\epsilon={sci(epsA)}$" + "\nwhere LM stalled", fontsize=6.4,
            ha="left", va="center", zorder=30, linespacing=1.25)
    ax.text(xy[2][0] + 0.15, xy[2][1], np.log10(epsB) + 0.2,
            "student B\n(geodesic LM, seed 0)\n" + rf"$\epsilon={sci(epsB)}$" + ", LM stalled", fontsize=6.4,
            ha="left", va="center", zorder=30, linespacing=1.25)
    ax.set_xlim(s1[0], s1[-1]); ax.set_ylim(s2[0], s2[-1]); ax.set_zlim(ZLO, ZHI)
    ax.set_box_aspect((span_u, span_w, 4.8), zoom=0.95)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-62)
    ax.set_xlabel(r"$u$  [$|\theta_T|$]", labelpad=-5); ax.set_ylabel(r"$w$  [$|\theta_T|$]", labelpad=-5)
    ax.set_zlabel(r"$\log_{10}$ mean $|\Delta E/E|$", labelpad=-2)
    ax.set_zticks([-15, -10, -5, 0])
    ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 1, 2, 3, 4, 5])
    fig.text(0.015, 1 - 0.1 / H, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.045, 1 - 0.115 / H, "Plane through T, A, B in parameter space (aligned min+shift gauge), 121 x 121 grid",
             fontsize=7.2, va="top")

    mn = N["main"]
    note = (r"$\epsilon$ = mean $|\Delta E/E|$ (100 tori x 128 angles)" "\n"
            rf"$\epsilon_A={sci(epsA)}$ (student A)" "\n\n"
            r"$\bf{Away\ from\ T,\ A,\ B}$" "\n"
            rf"$\epsilon$ = {mn['p1']:.3f}-{mn['p99']:.2f} (1st-99th percentile)," "\n"
            f"median {mn['median']:.2f}: the level of an untrained\nnetwork (0.25-0.26). At grid resolution\n"
            f"no node other than T and A is below\n{mn['lowest_other']:.3f}: no plateau at the students'\n"
            f"level. Sub-cell pits would not show\n(B's nearest node reads {mn['B_nearest_node_eps']:.2f}).\n"
            f"Straight segments T-A, T-B, A-B peak\nat {min(N['barriers'].values()):.2f}-{max(N['barriers'].values()):.2f}.\n\n"
            r"$\bf{Pits\ far\ below\ grid\ scale}$" "\n"
            f"Cell {N['cell'][0]:.3f} x {N['cell'][1]:.3f} " r"$|\theta_T|$; each pit is" "\n"
            "a vertical line to its own measured " r"$\epsilon$." "\n"
            r"Half-width where $\epsilon$ reaches $2\epsilon_A$ [$|\theta_T|$]:" "\n"
            rf"  T   ${sci(h2['T']['u'])}$ (u),  ${sci(h2['T']['w'])}$ (w)" "\n"
            rf"  A   ${sci(h2['A']['u'])}$ (u),  ${sci(h2['A']['w'])}$ (w)" "\n"
            rf"  B   ${sci(h2['B']['u'])}$ (u),  ${sci(h2['B']['w'])}$ (w)")
    fig.text(0.715, 1 - 0.22 / H, note, fontsize=6.2, va="top", ha="left", linespacing=1.32, color=INK)

    # shared colour bar for a-c
    cax = fig.add_axes(F(5.15, 5.80, 1.62, 0.075))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-16, -12, -8, -4, 0]); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(r"$\log_{10}$ mean $|\Delta E/E|$ (panels a-c)", fontsize=6.3, labelpad=2)

    # ---------------- (b) cutout around T and (c) cutout around A, same window and z-range
    zlo2, zhi2 = -16.0, -4.0
    zoom_u = span_u / (2 * r)
    for k, (c, lab, pos) in enumerate(((cT, "b", F(-0.25, 2.04, 3.78, 2.68)), (cA, "c", F(3.32, 2.04, 3.78, 2.68)))):
        bx_ = fig.add_axes(pos, projection="3d", computed_zorder=False)
        style3d(bx_)
        q1, q2, Q = c["s1"] / 1e-7, c["s2"] / 1e-7, c["log10eps"]
        ic, jc = [tuple(v) for v in c["node_ij"]][0]
        Xc, Yc = np.meshgrid(q1, q2, indexing="ij")
        if k == 0:
            Qd = replace_node(Q, ic, jc)
            surface(bx_, Xc, Yc, Qd)
            needle(bx_, 0.0, 0.0, Qd[ic, jc], Q[ic, jc])
            bx_.scatter([0], [0], [Q[ic, jc]], s=14, color=plt.get_cmap(CMAP)(NORM(Q[ic, jc])), edgecolor="#1b1b1b",
                        linewidth=0.6, zorder=25, depthshade=False)
            cs = plt.figure().add_subplot().contour(q1, q2, Q.T, levels=[lzA]); plt.close(cs.axes.figure)
            for path in cs.allsegs[0]:
                bx_.plot(path[:, 0], path[:, 1], np.full(len(path), lzA), color="white", lw=0.8, zorder=22)
            bx_.text(0.6, 0.6, Q[ic, jc] + 0.2, "T  " + r"$\epsilon\approx5\times10^{-16}$", fontsize=6.4, ha="left",
                     va="bottom", zorder=30)
            title = (rf"Cutout around T: half-width ${sci(r, 0)}\,|\theta_T|$ (zoom $\times{sci(zoom_u)}$ in $u$)."
                     + "\n" + r"White contour: $\epsilon=\epsilon_A$. On a log axis an exact zero looks" + "\n"
                     + "like this at every zoom; the needle is depth, not narrowness.")
        else:
            surface(bx_, Xc, Yc, Q)
            bx_.scatter([0], [0], [Q[ic, jc]], s=14, color=plt.get_cmap(CMAP)(NORM(Q[ic, jc])), edgecolor="#1b1b1b",
                        linewidth=0.6, zorder=25, depthshade=False)
            bx_.text(-1.0, -1.0, -8.6, "A  " + rf"$\epsilon={sci(10 ** Q[ic, jc])}$ (lowest node of this window)" + "\n"
                     + rf"window maximum ${sci(10 ** Q.max())}$ = {10 ** Q.max() / 10 ** Q[ic, jc]:.1f} $\epsilon_A$" + "\n"
                     + "u and w are sloppy directions at both A and T,\nso both basins look wide here (Fig. 2d)",
                     fontsize=6.2, ha="center", va="top", zorder=30, linespacing=1.3)
            title = "Cutout around A: same window, z-range and\ncolours as b. Flat only on this 12-decade axis:\nin linear " + r"$\epsilon$" + " it is a V like T's (panel d)."
        bx_.set_xlim(q1[0], q1[-1]); bx_.set_ylim(q2[0], q2[-1]); bx_.set_zlim(zlo2, zhi2)
        bx_.set_box_aspect((1, 1, 1.0), zoom=0.9)
        bx_.set_proj_type("ortho")
        bx_.view_init(elev=27, azim=-62)
        bx_.set_xlabel(r"$u$  [$10^{-7}|\theta_T|$]", labelpad=-5)
        bx_.set_ylabel(r"$w$  [$10^{-7}|\theta_T|$]", labelpad=-5)
        bx_.set_zlabel(r"$\log_{10}$ mean $|\Delta E/E|$", labelpad=-3)
        bx_.set_xticks([-5, 0, 5]); bx_.set_yticks([-5, 0, 5]); bx_.set_zticks([-16, -12, -8, -4])
        x0 = (0.10 if k == 0 else 3.67)
        fig.text(x0 / W, 4.97 / H, lab, fontsize=9, fontweight="bold", va="top")
        fig.text((x0 + 0.21) / W, 4.955 / H, title, fontsize=6.5, va="top", linespacing=1.3)

    # ---------------- (d) the same cross-sections through T, A, B on a linear eps axis
    z = load("plane_scan.npz"); t = z["t"]
    e0 = {"T": float(z["T_eps0"]), "A": float(z["A_eps0"]), "B": float(z["B_eps0"])}
    col = {"T": CAT[0], "A": CAT[1], "B": CAT[2]}
    lab_ = {"T": "through T", "A": "through A", "B": "through B"}
    xmax = 10.0   # units of 1e-7 |theta_T|
    fig.text(0.10 / W, 2.02 / H, "d", fontsize=9, fontweight="bold", va="top")
    fig.text(0.31 / W, 2.005 / H, r"The same cross-sections on a linear $\epsilon$ axis: straight lines through T, A and B along $u$ (left) and $w$ (right)",
             fontsize=6.8, va="top")
    vs = N["vslopes"]
    for k_, (axn, pos) in enumerate((("u", F(0.62, 0.36, 2.55, 1.28)), ("w", F(3.86, 0.36, 2.55, 1.28)))):
        a2 = fig.add_axes(pos)
        for c in "TAB":
            tp = t[t <= xmax * 1e-7 * 1.05]
            ep = z[f"{c}+{axn}_eps"][: len(tp)]; em = z[f"{c}-{axn}_eps"][: len(tp)]
            xx = np.concatenate([-tp[::-1], [0.0], tp]) / 1e-7
            yy = np.concatenate([em[::-1], [e0[c]], ep]) / 1e-5
            a2.plot(xx, yy, color=col[c], lw=1.2, label=lab_[c], zorder=3, clip_on=True)
            a2.plot([0], [e0[c] / 1e-5], "o", ms=3.2, color=col[c], mec="white", mew=0.4, zorder=4)
        # direct labels: T below its V on the left arm, A above its V, B beside its spike
        a2.text(-0.80 * xmax, 0.80 * xmax * 1e-7 * np.mean(np.abs(vs["T"][axn])) / 1e-5 - 0.45, "T", color=col["T"],
                fontsize=6.6, fontweight="bold", ha="center", va="top")
        a2.text(-0.80 * xmax, e0["A"] / 1e-5 + 0.80 * xmax * 1e-7 * np.mean(np.abs(vs["A"][axn])) / 1e-5 + 0.25, "A",
                color=col["A"], fontsize=6.6, fontweight="bold", ha="center", va="bottom")
        a2.text(0.35, 7.6, "B", color=col["B"], fontsize=6.6, fontweight="bold", ha="left", va="center")
        a2.axhline(epsA / 1e-5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        a2.axhline(2 * epsA / 1e-5, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        a2.text(-xmax * 0.98, epsA / 1e-5 + 0.12, r"$\epsilon_A$", fontsize=5.8, color=INK2, va="bottom")
        a2.text(-xmax * 0.98, 2 * epsA / 1e-5 + 0.12, r"$2\epsilon_A$", fontsize=5.8, color=INK2, va="bottom")
        a2.set_xlim(-xmax, xmax); a2.set_ylim(0, 9.0)
        a2.set_xlabel(rf"distance along ${axn}$  [$10^{{-7}}|\theta_T|$]", labelpad=1)
        a2.set_ylabel(r"$\epsilon$  [$10^{-5}$]", labelpad=1)
        a2.grid(True, color=GRID, lw=0.4); a2.set_axisbelow(True)
        sT = np.mean(np.abs(vs["T"][axn])); sA = np.mean(np.abs(vs["A"][axn]))
        a2.set_title(rf"along ${axn}$: slopes T {sT:.0f}, A {sA:.0f} per $|\theta_T|$; B reaches $2\epsilon_A$ within ${sci(N['half_2epsA']['B'][axn])}$",
                     fontsize=6.0, color=INK2, loc="left", pad=2.5)
        a2.set_xticks([-10, -5, 0, 5, 10])
    save(fig, "fig1_landscape_3d")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure S1
def figS1():
    ps = jload("plane_summary.json")
    planes = [("plane_main.npz", "plane T, A, B (min+shift gauge)", ("T", "A", "B")),
              ("plane_rsr.npz", "plane T, C, D (min+shift gauge)", ("T", "C", "D")),
              ("plane_main_raw.npz", "plane T, A, B (raw parameters)", ("T", "A", "B"))]
    cuts = [("plane_cutT.npz", "around T"), ("plane_cutA.npz", "around A"), ("plane_cutB.npz", "around B")]
    off = {"T": (5, 4), "A": (-5, 4), "B": (5, 4), "C": (-5, 4), "D": (5, 4)}
    ha = {"T": "left", "A": "right", "B": "left", "C": "right", "D": "left"}
    epsP = {k: v["eps"] for k, v in ps["points"].items()}
    epsA = epsP["A"]
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.04], left=0.065, right=0.915, top=0.93, bottom=0.085,
                          wspace=0.34, hspace=0.38)
    cm = plt.get_cmap(CMAP).with_extremes(under="#1b1b1b")
    n_top = mcolors.Normalize(-2.6, 0.0)
    for k, (f, title, who) in enumerate(planes):
        z = load(f); ax = fig.add_subplot(gs[0, k])
        im = ax.pcolormesh(z["s1"], z["s2"], z["log10eps"].T, cmap=cm, norm=n_top, shading="nearest", rasterized=True)
        for nm, (x, y) in zip(who, z["named_coords"]):
            ax.plot([x], [y], "o", ms=5.5, mfc="none", mec="white", mew=1.6)
            ax.plot([x], [y], "o", ms=5.5, mfc="none", mec=INK, mew=0.7)
            ax.annotate(nm, (x, y), xytext=off[nm], textcoords="offset points", fontsize=6.5, fontweight="bold",
                        color=INK, ha=ha[nm], va="bottom", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        g = z["log10eps"]; other = np.ones_like(g, bool)
        for (i, j) in z["node_ij"]:
            other[i, j] = False
        ax.set_title(rf"$\bf{{{'abc'[k]}}}$  {title}" + "\n" + f"{g.shape[0]}x{g.shape[1]}; median {10 ** np.median(g):.2g}; "
                     f"next-lowest node {10 ** g[other].min():.2g}", fontsize=6.2, pad=3, loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$u$  [$|\theta_T|$]", labelpad=1); ax.set_ylabel(r"$w$  [$|\theta_T|$]", labelpad=1)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), extend="min")
    cb.set_label(r"$\log_{10}$ mean $|\Delta E/E|$", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    n_bot = mcolors.Normalize(-6.5, -1.5)
    lev_mult = [1, 2, 10, 100]
    for k, (f, title) in enumerate(cuts):
        z = load(f); ax = fig.add_subplot(gs[1, k])
        q1, q2, g = z["s1"] / 1e-7, z["s2"] / 1e-7, z["log10eps"]
        im2 = ax.pcolormesh(q1, q2, g.T, cmap=cm, norm=n_bot, shading="nearest", rasterized=True)
        levs = [np.log10(m_ * epsA) for m_ in lev_mult]
        cs = ax.contour(q1, q2, g.T, levels=levs, colors="white", linewidths=0.6, negative_linestyles="solid")
        ax.clabel(cs, fmt={l: f"{m_}" + r"$\epsilon_A$" for l, m_ in zip(levs, lev_mult)}, fontsize=5.2, inline_spacing=1)
        c = str(z["centre"])
        ax.plot([0], [0], "o", ms=4.5, mfc="none", mec="white", mew=1.4)
        ax.plot([0], [0], "o", ms=4.5, mfc="none", mec=INK, mew=0.6)
        ax.set_aspect("equal")
        n2 = int((10 ** g <= 2 * epsA).sum())
        frac = f"{100 * n2 / g.size:.1f}% of nodes" if n2 > 5 else f"{n2} node" + ("s" if n2 != 1 else "")
        ev = 10 ** g[50, 50]
        elab = r"\approx5\times10^{-16}" if c == "T" else f"={sci(ev)}"
        ax.set_title(rf"$\bf{{{'def'[k]}}}$  cutout {title}, " + rf"$\epsilon_{{{c}}}{elab}$" + "\n"
                     + rf"$\epsilon\leq2\epsilon_A$ on {frac}", fontsize=6.2, pad=3, loc="left")
        ax.set_xlabel(r"$u$  [$10^{-7}|\theta_T|$]", labelpad=1); ax.set_ylabel(r"$w$  [$10^{-7}|\theta_T|$]", labelpad=0)
        ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
    cb2 = fig.colorbar(im2, cax=fig.add_subplot(gs[1, 3]), extend="min")
    cb2.set_label(r"$\log_{10}$ mean $|\Delta E/E|$", fontsize=6.5, labelpad=2); cb2.ax.tick_params(labelsize=6)
    cb2.outline.set_linewidth(0.4)
    save(fig, "figS1_landscape_maps")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 2
def _w2(entry, key="t_2eps_A"):
    """Mean over the two signs of a width (None if neither sign reached the level)."""
    v = [entry.get(key + sg) for sg in "+-"]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _line_label(ax, x, y, text, dx=0.0, dy=0.25, slope=1.0, **kw):
    """Text placed just above a log-log line at (x, y), rotated to the line's on-screen angle (slope in decades)."""
    p0 = ax.transData.transform((x, y)); p1 = ax.transData.transform((x * 10, y * 10 ** slope))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x * 10 ** dx, y * 10 ** dy, text, rotation=ang, rotation_mode="anchor", ha="left", va="bottom",
            fontsize=5.8, color=INK, **kw)


def fig2():
    N = numbers(write=False)
    fl = jload("sloppy_floor.json"); F = load("sloppy_floor.npz")
    tw = load("sloppy_twist_T.npz")
    pr = load("profiles_T.npz"); prs = jload("profiles_summary.json")
    dist = jload("profiles_dist.json")
    sl = jload("sloppy_line.json")
    epsA = N["eps_A"]; epsT = N["eps_T_batched"]
    W, H = 7.0, 8.85
    fig = plt.figure(figsize=(W, H))

    def rect(x0, y0, w, h):  # inches -> figure fraction
        return [x0 / W, y0 / H, w / W, h / H]

    # ---------------- (a) spectra
    ax = fig.add_axes(rect(0.55, 6.30, 2.75, 2.25))
    series = [("T_energy", r"$J_E$ at T (energy)", CAT[0]), ("T_coord", r"$J_C$ at T (coordinates)", CAT[1]),
              ("A_energy", r"$J_E$ at A (Gauss-Newton: LM's model)", CAT[2])]
    floor_max = max(fl[k]["floor_rel"] for k, _, _ in series)
    ax.axhspan(1e-18, floor_max, color="#e1e0d9", lw=0, zorder=0)
    ax.text(3080, 4.5e-17, "not reproducible in float64:\n" + " / ".join(f"{3104 - fl[k]['n_resolved']}" for k, _, _ in series)
            + " of 3104", fontsize=5.6, color=INK2, va="center", ha="right", linespacing=1.15)
    for k, lab, col in series:
        S = F[f"S_{k}"]; rel = S / S[0]; n = fl[k]["n_resolved"]
        idx = np.arange(1, len(S) + 1)
        ax.plot(idx[:n], rel[:n], color=col, lw=1.1, label=lab)
    # true curvature of the LM objective at A along sampled v_i: sqrt(v_i^T Hess f v_i) / sigma_max
    hA = N["hess_A"]; s0 = np.sqrt(hA[0]["GN"])
    ix = np.array([int(h["name"][2:]) + 1 for h in hA]); yc = np.array([np.sqrt(max(h["full"], 0.0)) / s0 for h in hA])
    ax.plot(ix, yc, ls="none", marker="o", ms=3.3, mfc="white", mec=CAT[2], mew=0.9, zorder=5,
            label=r"A: true curvature $\sqrt{v_i^T\nabla^2 f\, v_i}/\sigma_{\max}$")
    ax.set_yscale("log")
    ax.set_xlim(0, 3104); ax.set_ylim(1e-17, 2)
    ax.set_yticks(10.0 ** np.arange(0, -17, -2))
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.minorticks_off()
    ax.set_xlabel("singular value index (3104 parameters)")
    ax.set_ylabel(r"$\sigma_i / \sigma_{\max}$")
    ax.legend(loc="upper right", handlelength=1.5, borderaxespad=0.3, fontsize=5.8, labelspacing=0.3)
    ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
    spE = N["spectra"]["T_energy"]; pdc = spE["per_decade"]
    band = pdc[5:15]   # decades [1e-6,1e-5) ... [1e-15,1e-14)
    ax.text(0.985, 0.115, rf"$J_E$ at T: {spE['n_above_1em4']} values above $10^{{-4}}$;" + "\n"
            + rf"{min(band)}-{max(band)} per decade from $10^{{-15}}$ to $10^{{-5}}$;" + "\n"
            + rf"no gap wider than {spE['max_gap_decades']:.2f} decades." + "\n"
            + r"$J_C$ is as sloppy as $J_E$." + "\n"
            + r"At A the residual term $r\cdot\nabla^2 r$ sets" + "\n" + r"the curvature below $\sim10^{-9}$.",
            transform=ax.transAxes, fontsize=5.6, color=INK2, va="bottom", ha="right", linespacing=1.25)
    ax.set_title(r"$\bf{a}$  Jacobian spectra (training set, 96 tori x 128 angles)", loc="left", fontsize=7.2)

    # ---------------- (b) twist fraction of the J_E singular vectors at T
    ax = fig.add_axes(rect(4.05, 6.30, 2.80, 2.25))
    S0 = float(tw["S"][0]); rel_d = tw["sig_dir"] / S0
    flE = np.maximum(tw["sig_noise"], tw["sig_bound"]); ok = tw["sig_dir"] > 10 * flE
    flr = fl["T_energy"]["floor_rel"]; ok2 = ok & (rel_d >= flr); grey = ok & (rel_d < flr)
    ftw = tw["twist"] / tw["tot"]; dxr = tw["dx"] / float(tw["sC"])
    lo, hi = 1e-3, 1 - 1.5e-7
    yv = np.clip(ftw, lo, hi)
    cn = mcolors.LogNorm(1e-14, 1)
    ax.scatter(rel_d[grey], yv[grey], color="#b4b2aa", s=3.0, lw=0, rasterized=True, zorder=1)
    sc = ax.scatter(rel_d[ok2], yv[ok2], c=np.clip(dxr[ok2], 1e-14, 1), cmap="cividis", norm=cn, s=3.5, lw=0,
                    rasterized=True, zorder=2)
    ax.axvline(flr, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axvspan(1e-17, 1e-13, color="#efeee9", lw=0, zorder=0)
    md = N["twist_T"]["medians"]
    xm = [10 ** (int(d["decade"][3:].split(",")[0]) + 0.5) for d in md]; ym = [d["median"] for d in md]
    ax.plot(xm, np.clip(ym, lo, hi), "-o", color=INK, ms=2.8, lw=0.9, zorder=3, label="median per decade")
    base = N["twist_T"]["random_median"]
    ax.axhline(base, color=CAT[1], lw=0.9, zorder=1, label=f"random directions (median {base:.2f})")
    ax.set_xscale("log"); ax.set_yscale("logit")
    ax.set_xlim(1e-17, 2); ax.set_ylim(lo, hi)
    ax.set_xticks(10.0 ** np.arange(-16, 1, 4))
    ax.set_yticks([0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])
    ax.set_yticklabels(["0.01", "0.1", "0.5", "0.9", "0.99", "0.999", "0.9999", "0.99999"])
    ax.minorticks_off()
    ax.set_xlabel(r"energy cost of the direction  $|J_E v_i| / \sigma_{\max}$")
    ax.set_ylabel(r"twist fraction of $|\delta x|^2$ (normalised coords)")
    ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.legend(loc="lower right", borderaxespad=0.3, handlelength=1.4, fontsize=5.8)
    ax.text(flr * 1.25, 0.999995, r"$\theta$-dependent slips:" + "\nenergy-free only on\n96 training tori", fontsize=5.3,
            color=INK2, va="top", ha="left", linespacing=1.15)
    ax.text(flr * 1.3, 0.00115, "panel-a floor", fontsize=5.0, color=INK2, va="bottom", ha="left")
    cax = ax.inset_axes([0.42, 0.88, 0.36, 0.03])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([1e-14, 1e-10, 1e-6, 1e-2]); cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(r"coordinate change $|\delta x|/\sigma_{C,\max}$", fontsize=5.8, labelpad=1.5)
    cb.outline.set_linewidth(0.4)
    ax.set_title(r"$\bf{b}$  Twist content of the $J_E$ directions at T", loc="left", fontsize=7.2)

    # ---------------- (c) 1D profiles from T and the LM start / end points
    ax = fig.add_axes(rect(0.55, 3.10, 4.35, 2.55))
    ts = pr["ts"]
    lines = [("sv1", r"stiffest $J_E$ direction"), ("rand0", "random direction"), ("toA", "toward student A (u)"),
             ("relrand0", r"relative random ($\propto|\theta_{T,i}|$)"), ("twnull1", "twist in the $J_E$ null space"),
             ("gate0", "gate-scale symmetry tangent")]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-14, 12); ax.set_ylim(1e-16, 3)
    ax.axhspan(1.6e-5, 2.8e-5, color="#d6d4cc", lw=0, zorder=0)
    ax.axhline(epsT, color=MUTED, lw=0.6, zorder=0)
    for k, (n, lab) in enumerate(lines):
        e = pr[n + "__eps"][0]
        ax.plot(ts, e, color=CAT[k], lw=1.0, label=lab, zorder=2)
    tl = np.array(sl["t_over_thetaT"]); el = np.array(sl["redundancy"]["eps"])
    ax.plot(tl, el, color=CAT[6], lw=1.0, label="least-coordinate-change redundancy", zorder=2)
    g1x = np.array([1e-13, 1e-10]); ax.plot(g1x, 1e-6 * (g1x / 1e-13), color=INK, lw=0.7, zorder=1)
    _line_label(ax, 1.4e-13, 1e-6 * 1.4, "slope 1", dy=0.2)
    g2x = np.array([1e-6, 1e-5]); ax.plot(g2x, 1.5e-13 * (g2x / 1e-6) ** 2, color=INK, lw=0.7, zorder=1)
    _line_label(ax, 1.15e-6, 1.5e-13 * 1.15 ** 2, "slope 2", dy=0.2, slope=2.0)
    sty = {("start", "converged"): dict(marker="o", mfc="white", mec=INK, ms=4.4, mew=0.9),
           ("end", "converged"): dict(marker="o", mfc=INK, mec=INK, ms=4.4, mew=0.9),
           ("start", "stalled"): dict(marker="s", mfc="white", mec=INK2, ms=3.8, mew=0.8),
           ("end", "stalled"): dict(marker="s", mfc=INK2, mec="white", ms=3.8, mew=0.4)}
    for nm, row in dist.items():
        if row["role"] == "exact":
            continue
        ax.plot([row["dist_min_shift"]], [row["eps_land"]], ls="none", zorder=5, **sty[(row["role"], row["outcome"])])
    for a_, b_ in (("rand4_start", "rand4_final"), ("sloppy_start", "sloppy_final")):
        A_, B_ = dist[a_], dist[b_]
        ax.annotate("", xy=(B_["dist_min_shift"], B_["eps_land"]), xytext=(A_["dist_min_shift"], A_["eps_land"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, shrinkA=3, shrinkB=3, mutation_scale=6), zorder=4)
    ax.annotate("perturbed teacher\n(noise / sloppy step)", (dist["rand4_start"]["dist_min_shift"], dist["rand4_start"]["eps_land"]),
                xytext=(-6, 8), textcoords="offset points", fontsize=5.6, ha="right", va="bottom", color=INK)
    ax.annotate("random init", (dist["init_s0"]["dist_min_shift"], dist["init_s0"]["eps_land"]),
                xytext=(-5, 3), textcoords="offset points", fontsize=5.6, ha="right", va="bottom", color=INK)
    ax.set_xticks(10.0 ** np.arange(-14, 1, 2)); ax.set_yticks(10.0 ** np.arange(-16, 1, 2))
    ax.minorticks_off()
    ax.set_xlabel(r"distance from T  $t$  [$|\theta_T|$]")
    ax.set_ylabel(r"mean $|\Delta E/E|$  ($\epsilon$)")
    ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    leg1 = ax.legend(h1, l1, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.0, handlelength=1.6,
                     labelspacing=0.35, title=r"$\epsilon(T + t\,|\theta_T|\,d)$, unit raw-parameter $d$ =", title_fontsize=6.2,
                     borderaxespad=0.0)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    hp = [Line2D([], [], ls="none", **sty[("start", "converged")]), Line2D([], [], ls="none", **sty[("end", "converged")]),
          Line2D([], [], ls="none", **sty[("start", "stalled")]), Line2D([], [], ls="none", **sty[("end", "stalled")]),
          Patch(color="#d6d4cc", lw=0), Line2D([], [], color=MUTED, lw=0.6)]
    leg2 = ax.legend(hp, ["start, LM converged", "end, LM converged", "start, LM stalled", "end, LM stalled",
                          r"trained students' $\epsilon$ (1.6-2.8$\times10^{-5}$)",
                          r"$\epsilon(T)\approx5\times10^{-16}$ (float64 rounding)"],
                     loc="upper left", bbox_to_anchor=(1.02, 0.40), fontsize=6.0, handlelength=1.0, labelspacing=0.35,
                     title="LM runs (Euclidean distance from T\nafter min+shift alignment)", title_fontsize=6.2, borderaxespad=0.0)
    leg2._legend_box.align = "left"
    ax.set_title(r"$\bf{c}$  Energy error along straight lines from T, and where LM started and ended", loc="left", fontsize=7.2)

    # ---------------- (d) straight-line half-widths at equal absolute error
    ax = fig.add_axes(rect(2.0, 0.42, 4.85, 2.05))
    T = prs["T"]; WC = prs["width_comparison"]
    rows = [("stiffest $J_E$ direction", ["sv1"], ["sv1"]),
            (r"$J_E$ direction, $\sigma/\sigma_{\max}\approx10^{-4}$", ["sv_1e-04"], ["sv_1e-04"]),
            (r"$J_E$ direction, $\sigma/\sigma_{\max}\approx10^{-8}$", ["sv_1e-08"], ["sv_1e-08"]),
            ("random (3 directions)", ["rand0", "rand1", "rand2"], ["rand0", "rand1", "rand2"]),
            ("along the segment T-A", ["toA", "toA_raw"], ["toT"]),
            ("twist-like (5 directions, T only)", ["twnull1", "twnull2", "twnull3", "twfit_g1_1e-08", "twfit_ramp_1e-08"], []),
            ("gate-scale symmetry tangent (T only)", ["gate0"], []),
            ("least-coordinate-change redundancy (T only)", ["@redundancy"], []),
            ("LM endpoint from a perturbed teacher", ["@sloppy_final"], [])]
    gst = {"raw": dict(marker="^", ms=4.0), "gate": dict(marker="s", ms=3.4), "ms": dict(marker="D", ms=3.2)}
    yl = []
    for r_, (lab, tn, an) in enumerate(rows):
        y = len(rows) - 1 - r_
        yl.append((y, lab))
        if tn == ["@sloppy_final"]:
            ax.plot([N["margin"]["sloppy_final_dist"]], [y], ls="none", marker="*", ms=6.5, color=INK, mec="white", mew=0.4, zorder=4)
            ax.annotate(r"its distance from T (min+shift), where $\epsilon=2.4\times10^{-10}\ll2\epsilon_A$:" + "\n"
                        + "T's low-error set reaches at least this far (curved path)",
                        (N["margin"]["sloppy_final_dist"], y), xytext=(-7, 0), textcoords="offset points", fontsize=5.4,
                        ha="right", va="center", color=INK2, linespacing=1.15)
            continue
        if tn == ["@redundancy"]:
            tv = [N["width_T_2epsA_line"]["redundancy"]]
        else:
            tv = [_w2(T[n]) for n in tn]
        ax.plot(tv, [y + 0.13] * len(tv), "o", color=CAT[0], ms=4.0, mec="white", mew=0.4, zorder=4, ls="none")
        for gname in ("raw", "gate", "ms"):
            av = [float(np.mean(WC[f"{gname}:{n}"]["2eps_A"])) for n in an]
            ax.plot(av, [y - 0.15] * len(av), ls="none", mfc="none", mec=CAT[1], mew=0.8, zorder=3, **gst[gname])
    mg = N["margin"]
    ax.axvspan(mg["students_min_dist"], mg["students_max_dist"], color="#d6d4cc", lw=0, zorder=0)
    ax.text(np.sqrt(mg["students_min_dist"] * mg["students_max_dist"]), -0.55, "trained\nstudents'\ndistance\nfrom T",
            fontsize=5.6, color=INK2, ha="center", va="bottom")
    ax.set_yticks([y for y, _ in yl]); ax.set_yticklabels([l for _, l in yl], fontsize=6.0)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xscale("log"); ax.set_xlim(1e-11, 40)
    ax.set_xticks(10.0 ** np.arange(-11, 2, 1)); ax.minorticks_off()
    ax.set_xticklabels([rf"$10^{{{e}}}$" if e % 2 == 1 else "" for e in range(-11, 2)])
    ax.set_xlabel(r"half-width: $t$ at which $\epsilon$ first reaches $2\epsilon_A=3.6\times10^{-5}$ on the straight line  [$|\theta_T|$]")
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
    hd = [Line2D([], [], ls="none", marker="o", color=CAT[0], ms=4.0, mec="white", mew=0.4),
          Line2D([], [], ls="none", mfc="none", mec=CAT[1], mew=0.8, **gst["raw"]),
          Line2D([], [], ls="none", mfc="none", mec=CAT[1], mew=0.8, **gst["gate"]),
          Line2D([], [], ls="none", mfc="none", mec=CAT[1], mew=0.8, **gst["ms"])]
    ax.legend(hd, ["around T (teacher)", "around A, raw parameters", "around A, gate gauge", "around A, min+shift gauge"],
              loc="upper left", bbox_to_anchor=(0.62, 1.0), fontsize=5.8, handlelength=1.0, labelspacing=0.25,
              borderaxespad=0.3)
    ax.set_title(r"$\bf{d}$  Straight-line half-widths at the same absolute error (lower bounds on each basin's extent)",
                 loc="left", fontsize=7.2, x=-0.27)
    save(fig, "fig2_sloppy")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- interactive.json
def _r4(a):
    """Round to 4 significant digits (nested lists, JSON-safe)."""
    a = np.asarray(a, float)
    return [float(f"{x:.4g}") for x in a.ravel()] if a.ndim == 1 else [_r4(row) for row in a]


def _grid(z, max_n=121):
    L = z["log10eps"]; s1, s2 = z["s1"], z["s2"]
    st1 = int(np.ceil(L.shape[0] / max_n)); st2 = int(np.ceil(L.shape[1] / max_n))
    i1 = np.arange(0, L.shape[0], st1); i2 = np.arange(0, L.shape[1], st2)
    nodes = {str(n): [int(i), int(j)] for n, (i, j) in zip(z["node_names"], z["node_ij"])}
    return dict(shape=[len(i1), len(i2)], downsample=[st1, st2], h1=float(z["h1"]) * st1, h2=float(z["h2"]) * st2,
                s1=[float(f"{x:.6g}") for x in s1[i1]], s2=[float(f"{x:.6g}") for x in s2[i2]],
                log10eps=_r4(L[np.ix_(i1, i2)]), log10eps_min=float(f"{L.min():.4g}"), log10eps_max=float(f"{L.max():.4g}"),
                log10eps_median=float(f"{np.median(L):.4g}"), centre=str(z["centre"]), nodes=nodes,
                layout="log10eps[i][j] at O + |theta_T| (s1[i] u + s2[j] w); i (rows) along u, j along w")


def write_json():
    import json as _json
    N = numbers(write=False)
    ps = jload("plane_summary.json")
    P = ps["points"]
    g4 = lambda x: float(f"{x:.4g}")
    lab = {"T": "teacher T", "A": "student A (std_s1: standard LM, seed 1; a point where LM stalled)",
           "B": "student B (geo_s0: geodesic LM, seed 0; a point where LM stalled)"}
    out = dict(meta=dict(
        metric="eps = mean over 100 tori x 128 angles of |H_T(S(toy(theta,J))) - <H_T>_theta| / |<H_T>_theta| (landscape set)",
        values="log10(eps), rounded to 4 significant digits; no value is clipped (max log10 eps on any grid = -0.003)",
        units="plane coordinates s1 (along u) and s2 (along w) in units of |theta_T| = %.6g (raw parameter norm of T)" % ps["theta_T_norm"],
        plane="x(s1, s2) = O + |theta_T| (s1 u + s2 w); u = (A - T)/|A - T|, w = Gram-Schmidt(B - T) against u; O = T for "
              "'main', O = T / A / B for the cutouts (same u, w)",
        gauge="students aligned to T with exact symmetries: per-layer gate scale chosen to minimise |S - T|, hidden-unit "
              "permutation and tanh sign (Hungarian), 14 translation gauges ('aligned_min_shift')",
        A="std_s1", B="geo_s0",
        eps_T_note="eps at T is float64 rounding of an exact zero and depends on the evaluation path: %.2g batched (the value "
                   "stored here), %.2g by core.residual_host, %.2g held-out. Quote it as ~5e-16." % (
                       N["eps_T_batched"], N["eps_T_direct"], N["eps_T_heldout"]),
        sub_grid_pits="T and A are single-node pits of the main grid and B is not a grid node (nearest node eps = %.3g). "
                      "Half-widths where eps reaches 2 eps_A (|theta_T|): T %.2g (u) / %.2g (w), A %.2g / %.2g, B %.2g / %.2g; "
                      "the main grid cell is %.3g x %.3g, i.e. %.0e to %.0e times larger. An interactive page should draw the "
                      "pits as needles to the marker eps values, as figure 1 does." % (
                          N["main"]["B_nearest_node_eps"], N["half_2epsA"]["T"]["u"], N["half_2epsA"]["T"]["w"],
                          N["half_2epsA"]["A"]["u"], N["half_2epsA"]["A"]["w"], N["half_2epsA"]["B"]["u"], N["half_2epsA"]["B"]["w"],
                          N["cell"][0], N["cell"][1], N["cell_over_half_2epsA"]["min"], N["cell_over_half_2epsA"]["max"]),
        caveats=["At grid resolution no node other than T and A is below %.3g: there is no plateau at the students' level "
                 "(~2e-5). Sub-cell pits like B's would not show on this grid." % N["main"]["lowest_other"],
                 "The needle at T is the log image of eps ~ |t| at an exact zero; its depth is float64 rounding. At equal "
                 "absolute error T's and A's cross-sections in this plane are comparable (see widths).",
                 "A and B are points where LM stalled, not proven local minima: truncated Gauss-Newton steps from A still lower "
                 "the training |r| by about 4e-6 relative per step.",
                 "Barriers (%.2f-%.2f) are measured on straight segments only; curved low-error paths are not excluded." % (
                     min(N["barriers"].values()), max(N["barriers"].values()))],
        sources=["plane_main.npz", "plane_cutT.npz", "plane_cutA.npz", "plane_cutB.npz", "plane_summary.json", "plane_scan.npz",
                 "fig_numbers.json"]))
    out["markers"] = [dict(name=k, label=lab[k], u=float(xy[0]), w=float(xy[1]), eps=float(f"{P[k]['eps']:.6g}"),
                           log10eps=float(f"{np.log10(P[k]['eps']):.4g}"), train_r=float(f"{P[k]['train_r']:.6g}"))
                      for k, xy in zip(("T", "A", "B"), load("plane_main.npz")["named_coords"])]
    for key, f in (("main", "plane_main.npz"), ("cutT", "plane_cutT.npz"), ("cutA", "plane_cutA.npz"), ("cutB", "plane_cutB.npz")):
        out[key] = _grid(load(f))
    cr = ps["cutout_r"]; sc = ps["scans"]; ct = N["contour_T_epsA"]; li = N["linearity_T"]
    out["widths"] = dict(
        units="|theta_T|",
        cutout_half_width=float(cr["r"]),
        T_reach_epsA_inplane={k: g4(v) for k, v in cr["t_reach_epsA_inplane"].items()},
        half_width_2epsA={c: {ax: g4(v) for ax, v in d.items()} for c, d in N["half_2epsA"].items()},
        half_width_10epsA={c: {ax: g4(v) for ax, v in d.items()} for c, d in N["half_10epsA"].items()},
        T_eps_over_t={k: float(f"{sc[k]['eps_at']['1e-09'] / 1e-9:.4g}") for k in ("T+u", "T+w", "Trand0", "Trand1")},
        main_panel_span=[float(f"{ps['grid_main']['s1_range'][1] - ps['grid_main']['s1_range'][0]:.4g}"),
                         float(f"{ps['grid_main']['s2_range'][1] - ps['grid_main']['s2_range'][0]:.4g}")],
        note=("T's eps = eps_A contour is approximately elliptical: within %.0f%% of the axis-aligned ellipse with semi-axes "
              "T_reach_epsA_inplane['+u'], ['+w'] (%.0f%% for a best-fit ellipse tilted by %.0f deg). eps grows linearly from T: "
              "along u and w the log-log slope is 1.000000 (+-4e-7) for t in [1e-12, 1e-6]; along the two random directions eps "
              "is linear to 1%% up to t = %.1e / %.1e (eps ~ 1e-2). Half-widths at 2 eps_A are log-interpolated from plane_scan.npz.") % (
            100 * (1 - ct["ratio_to_axis_ellipse"][0]), 100 * ct["fit_max_dev"], abs(ct["fit_tilt_deg"]),
            li["rand0"]["t_1pct"], li["rand1"]["t_1pct"]))
    out["levels"] = dict(eps_T=P["T"]["eps"], eps_T_direct=N["eps_T_direct"], eps_A=P["A"]["eps"], eps_B=P["B"]["eps"],
                         log10_eps_A=float(f"{np.log10(P['A']['eps']):.4g}"),
                         students_range=[1.6e-5, 2.8e-5], suggested_z_range=[ZLO, ZHI])
    path = os.path.join(RES, "interactive.json")
    with open(path, "w") as fh:
        _json.dump(out, fh, separators=(",", ":"))
    print("wrote", path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    which = sys.argv[1:] or ["numbers", "fig1", "figS1", "fig2", "json"]
    for w in which:
        {"numbers": numbers, "fig1": fig1, "figS1": figS1, "fig2": fig2, "json": write_json}[w]()
