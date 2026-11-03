"""Terrain version of the landscape figure (companion to landscape_fig.py's Fig 1), from results/landscape data:
(a) the T-A-B plane at its own height range, hill-shaded, with the three sub-cell pits drawn as drop lines;
(b) the same plane from above; (c) median and range of the error against distance in the plane around T and A; (d, e) log-polar cutouts around T and A (radius = log
distance, same scales); (f) linear-error cross-sections.

    python landscape_terrain_fig.py OUT_STEM        writes OUT_STEM.png and OUT_STEM.pdf
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, Normalize  # noqa: E402

RES = "/Users/gabrielpfaffman/Repos/orbitflows/performance/precision/results/landscape"
ORANGE, BLUE, GREEN = "#D85A30", "#2a6fdb", "#1a9e77"


def plane_coords():
    al = np.load(f"{RES}/aligned.npz", allow_pickle=True)
    names = list(al["names"]); vT = al["vT"]; nT = float(al["nT"]); V = al["aligned_min_shift"]
    A, B = str(al["A"]), str(al["B"])
    d1, d2 = V[names.index(A)] - vT, V[names.index(B)] - vT
    u = d1 / np.linalg.norm(d1); w = d2 - (d2 @ u) * u; w /= np.linalg.norm(w)
    return (np.linalg.norm(d1) / nT, 0.0), ((d2 @ u) / nT, (d2 @ w) / nT)


def main(stem):
    N = json.load(open(f"{RES}/fig_numbers.json"))
    eT, eA, eB = N["eps_T_direct"], N["eps_A"], N["eps_B"]
    cA, cB = plane_coords()
    main_ = np.load(f"{RES}/plane_main_partial.npz")
    lpT, lpA = np.load(f"{RES}/logpolar_T.npz"), np.load(f"{RES}/logpolar_A.npz")
    scan = np.load(f"{RES}/plane_scan.npz", allow_pickle=True)
    ls = LightSource(azdeg=315, altdeg=40)
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(15.5, 10.2))

    # (a) terrain at its own height range; pits far narrower than a cell drawn as drop lines to their measured value
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    X, Y = np.meshgrid(main_["s1"], main_["s2"], indexing="ij")
    Z = np.clip(np.log10(main_["eps"]), -3, 0.3)
    for (x, y) in ((0, 0), cA):  # the T and A nodes: shown with their neighbours' median, the pit is a line
        i = np.unravel_index(np.argmin((X - x) ** 2 + (Y - y) ** 2), X.shape)
        Z[i] = np.median(Z[max(i[0] - 1, 0):i[0] + 2, max(i[1] - 1, 0):i[1] + 2])
    norm = Normalize(-3, 0.3)
    rgb = ls.shade(Z, cmap=plt.cm.viridis, norm=norm, vert_exag=2, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rcount=Z.shape[0], ccount=Z.shape[1], linewidth=0, antialiased=False, shade=False)
    for (x, y), e, lab, c in (((0, 0), eT, "T", ORANGE), (cA, eA, "A", "k"), (cB, eB, "B", "k")):
        i = np.unravel_index(np.argmin((X - x) ** 2 + (Y - y) ** 2), X.shape)
        ax.plot([x, x], [y, y], [Z[i], -3], color=c, lw=2.2, zorder=10)
        ax.text(x, y, -3.3, f"{lab}: {e:.1e}", fontsize=9, ha="center", zorder=11,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_zlim(-3.3, 0.3); ax.view_init(elev=32, azim=-115)
    ax.set_xlabel("u  [|θ_T|]"); ax.set_ylabel("w  [|θ_T|]"); ax.set_zlabel("log10 mean|ΔE/E|")
    ax.set_title("(a) plane through teacher T and students A, B\nheight range -3..0; each pit (~1e-6 wide) is a drop line\n"
                 "to its measured error", fontsize=9.5)

    # (b) top view
    ax = fig.add_subplot(2, 3, 2)
    rgb2 = ls.shade(np.clip(np.log10(main_["eps"]), -3, 0.3).T, cmap=plt.cm.viridis, norm=norm, vert_exag=2, blend_mode="soft")
    ax.imshow(rgb2, origin="lower", extent=(main_["s1"][0], main_["s1"][-1], main_["s2"][0], main_["s2"][-1]), interpolation="nearest")
    for (x, y), lab in (((0, 0), "T"), (cA, "A"), (cB, "B")):
        ax.plot(x, y, "o", ms=7, mfc=ORANGE if lab == "T" else "white", mec="k")
        ax.annotate(lab, (x, y), xytext=(7, 5), textcoords="offset points", color="w", fontsize=12, weight="bold")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="log10 mean|ΔE/E| (shown -3..0)", shrink=0.8)
    ax.set_xlabel("u  [|θ_T|]"); ax.set_ylabel("w  [|θ_T|]")
    ax.set_title("(b) the same plane from above (hill-shaded)\nmedian 0.27, the level of an untrained network", fontsize=9.5)

    # (c, d) log-polar cutouts with identical scales
    def logpolar(ax, lp, title):
        rho = np.log10(lp["r"]) + 16
        R, P = np.meshgrid(rho, lp["phi"], indexing="ij")
        Zp = np.log10(lp["eps"])
        n2 = Normalize(-16, -3)
        rgb = ls.shade(Zp, cmap=plt.cm.viridis, norm=n2, vert_exag=0.3, blend_mode="soft")
        ax.plot_surface(R * np.cos(P), R * np.sin(P), Zp, facecolors=rgb, rcount=Zp.shape[0], ccount=Zp.shape[1],
                        linewidth=0, antialiased=False, shade=False)
        th = np.linspace(0, 2 * np.pi, 100)
        for k, lab in ((2, "1e-14"), (5, "1e-11"), (8, "1e-8"), (11, "1e-5")):
            ax.plot(k * np.cos(th), k * np.sin(th), -16 * np.ones_like(th), color="0.5", lw=0.6)
            ax.text(k * np.cos(-0.9), k * np.sin(-0.9), -16, lab, fontsize=7, color="0.3")
        ax.set_zlim(-16, -3); ax.set_zlabel("log10 mean|ΔE/E|"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5); ax.view_init(elev=22, azim=-60)
    logpolar(fig.add_subplot(2, 3, 4, projection="3d"), lpT,
             "(d) around T in the plane; radius = log10 distance\n(rings on the floor: 1e-14 ... 1e-5 |θ_T|). An exact zero\n"
             "is a cone on this axis at every zoom")
    logpolar(fig.add_subplot(2, 3, 5, projection="3d"), lpA,
             "(e) around A, same scales: flat at its own error\n(1.8e-5) out to ~1e-7 |θ_T|, then the same rise as T")

    # (e) median and range over directions in the plane
    ax = fig.add_subplot(2, 3, 3)
    for lp, lab, c in ((lpT, "around teacher T", ORANGE), (lpA, "around student A", BLUE)):
        e = lp["eps"]
        ax.fill_between(lp["r"], e.min(1), e.max(1), color=c, alpha=0.25, lw=0)
        ax.loglog(lp["r"], np.median(e, 1), color=c, lw=2, label=lab)
    ax.axhline(eT, color="gray", ls=":", lw=0.8); ax.text(2e-16, eT * 1.8, "float64 rounding of an exact zero", fontsize=8, color="gray")
    ax.set_xlabel("distance from the centre, in the plane  [|θ_T|]"); ax.set_ylabel("mean|ΔE/E|"); ax.set_ylim(1e-16, 1e-2)
    ax.set_title("(c) the two pits: median and range over\ndirections in the plane", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")

    # (f) linear-error cross-sections through T, A, B along u
    ax = fig.add_subplot(2, 3, 6)
    t = scan["t"]; k = t <= 1.2e-6
    for c0, lab, col in (("T", "T", BLUE), ("A", "A", ORANGE), ("B", "B", GREEN)):
        x = np.concatenate([-t[k][::-1], [0.0], t[k]]) / 1e-7
        y = np.concatenate([scan[f"{c0}-u_eps"][k][::-1], [float(scan[f"{c0}_eps0"])], scan[f"{c0}+u_eps"][k]]) / 1e-5
        ax.plot(x, y, color=col, lw=1.8, label=lab)
    ax.axhline(eA / 1e-5, color="k", ls="--", lw=0.7); ax.text(-11.5, eA / 1e-5 + 0.15, "ε_A", fontsize=8)
    ax.set_xlim(-12, 12); ax.set_ylim(0, 9)
    ax.set_xlabel("distance along u  [1e-7 |θ_T|]"); ax.set_ylabel("mean|ΔE/E|  [1e-5]")
    ax.set_title("(f) the same pits on a linear axis: T and A are V's of\nsimilar slope (T reaches zero); B is ~400x narrower here", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=200); fig.savefig(stem + ".pdf")


if __name__ == "__main__":
    main(sys.argv[1])
