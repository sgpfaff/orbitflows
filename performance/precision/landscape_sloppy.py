"""Sloppiness of the energy loss on the 16-layer MW-teacher test bed, and the twist decomposition of its null directions.

Uses landscape_lab (training set 96 tori x 128 angles, cached SVDs of the energy Jacobian J_E at T and at A, and of the
coordinate Jacobian J_C at T). Every output goes to RES/sloppy_*.{npz,json}.

    python landscape_sloppy.py floor     spectra of J_E(T), J_C(T), J_E(A); float64 resolution of each, from the same
                                         Jacobian at a symmetry copy of the point (hidden units permuted and sign-flipped:
                                         the same map with different rounding), from a row-permuted SVD and from 1-ulp
                                         entry noise; the exact gate-scale and translation tangents -> sloppy_floor.*
    python landscape_sloppy.py units     per-layer redundancy: the 1D shear function f_k(s) of each layer and its
                                         194-parameter Jacobian on the layer's inputs at the training points; dead,
                                         saturated, linear and duplicate hidden units -> sloppy_units.*
    python landscape_sloppy.py twist T|A|A_min_shift
                                         for every right singular vector v_i of J_E at the point, the coordinate change
                                         dx_i = J_C v_i on the training points (normalised by the torus amplitudes, as the
                                         coordinate loss), split per torus into a twist g(J) dx/dtheta, the remaining
                                         tangential part and the normal part; the rounding noise of dx_i and of |J_E v_i|
                                         from the symmetry copy; twist-profile ranks; for T also J_E vs J_C null spaces
                                         -> sloppy_twist_<pt>.{npz,json}
    python landscape_sloppy.py line      eps on the landscape set along lines from T: a twist-like null direction, a
                                         generic direction, the stiffest direction (log-log slopes) -> sloppy_line.*
    python landscape_sloppy.py summary   -> sloppy_summary.json

Decomposition (per torus o, angles j, normalised coordinates u = (q / s_q, p / s_p) / sqrt(N), t_oj = du/dtheta):
    a_oj = <dx_oj, t_oj> / |t_oj|^2 (tangential coefficient), g_o = sum_j <dx_oj, t_oj> / sum_j |t_oj|^2 (twist),
    |dx|^2 = sum g_o^2 |t_oj|^2 (twist) + sum (a_oj - g_o)^2 |t_oj|^2 (other tangential) + sum n_oj^2 (normal),
    n_oj = (dx_q t_p - dx_p t_q) / |t_oj| (each part computed directly, not by subtraction). In normalised coordinates
    grad_u H is perpendicular to t (grad H . dx/dtheta = 0 on a level set), so the normal part carries all of the
    first-order energy change at an exact solution.
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
EPS64 = float(torch.finfo(F64).eps)  # 2.2e-16
CUTS = (1e-4, 1e-8, 1e-12, 1e-15)
DECADES = list(range(0, -21, -1))  # bins [10^(d-1), 10^d) of sigma / sigma_max


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda x: float(x) if np.isscalar(x) else np.asarray(x).tolist())


# ======================================================================= symmetry copy (rounding perturbation)
def sym_perm(seed):
    """The signed permutation of LL.random_symmetry(., seed): new[i] = s[i] * old[pi[i]]. Returns (pi, s)."""
    u = torch.arange(1, LL.P + 1, dtype=F64)
    w = LL.random_symmetry(u, seed)
    pi = (w.abs() - 1).round().long(); s = torch.sign(w)
    assert torch.equal(u[pi] * s, w)
    return pi, s


def to_sym(V, pi, s):
    """Tangent vectors V (P,) or (P, K) at v -> the same tangents at the symmetry copy R v."""
    return (V[pi].T * s).T if V.dim() == 2 else V[pi] * s


def from_sym_cols(Jp, pi, s):
    """Jacobian Jp at R v (columns in the copy's order) -> the same matrix in v's column order (Jp R)."""
    out = torch.empty_like(Jp)
    out[:, pi] = Jp * s
    return out


# ============================================================================== tangents, decomposition
def tangent(v):
    """dx/dtheta (nO, nS, 2) at the training points for the student at v (raw coordinates)."""
    L = LL.load_all(); LL.set_vec(L.m, v)
    X = L.aa.reshape(-1, 2).clone().requires_grad_(True)
    y = L.m(X)
    tq = torch.autograd.grad(y[:, 0].sum(), X, retain_graph=True)[0][:, 0]
    tp = torch.autograd.grad(y[:, 1].sum(), X)[0][:, 0]
    return torch.stack([tq, tp], -1).detach().view(L.nO, L.nS, 2)


def grad_H(v):
    """grad H_T (nO, nS, 2) at the student's images of the training points, and the images y."""
    L = LL.load_all(); LL.set_vec(L.m, v)
    with torch.no_grad():
        y = L.m(L.aa.reshape(-1, 2))
    y = y.detach().requires_grad_(True)
    g = torch.autograd.grad(L.t.H(y).sum(), y)[0]
    return g.detach().view(L.nO, L.nS, 2), y.detach().view(L.nO, L.nS, 2)


def norm_scale():
    """Per-torus (nO, 1, 2) factor s_pt sqrt(N) that turns the normalised rows of coord_jac back into raw coordinates."""
    L = LL.load_all()
    return C.coord_scale(L.es, float(L.t.nu0))[:, None, :] * math.sqrt(L.nO * L.nS)


def decompose(D, Tn):
    """D (nO, nS, 2, K) normalised coordinate changes, Tn (nO, nS, 2) normalised tangents. Returns dict of (K,) squared
    norms tot, twist, other, normal and the twist profiles g (nO, K) (radians per unit step) and the isometric twist
    amplitudes Gw = g * sqrt(sum_j |t_oj|^2) (nO, K)."""
    tt = (Tn ** 2).sum(-1)                                   # (nO, nS)
    proj = torch.einsum("osc,osck->osk", Tn, D)              # <dx, t> per point
    a = proj / tt[..., None]
    g = proj.sum(1) / tt.sum(1)[:, None]                     # (nO, K)
    twist = (g ** 2 * tt.sum(1)[:, None]).sum(0)
    other = ((a - g[:, None, :]) ** 2 * tt[..., None]).sum((0, 1))
    cross = D[:, :, 0, :] * Tn[:, :, 1, None] - D[:, :, 1, :] * Tn[:, :, 0, None]
    normal = (cross ** 2 / tt[..., None]).sum((0, 1))
    tot = (D ** 2).sum((0, 1, 2))
    return dict(tot=tot, twist=twist, other=other, normal=normal, g=g, Gw=g * tt.sum(1).sqrt()[:, None])


def energy_from_dx(D, gH, scale):
    """The first-order training energy residual of coordinate changes D (nO, nS, 2, K) (normalised): rows
    (dH - <dH>) / es / sqrt(N), and the squared norm of the torus-mean part <dH> (same normalisation)."""
    L = LL.load_all(); sN = math.sqrt(L.nO * L.nS)
    dH = torch.einsum("osc,osck->osk", gH * scale, D)       # raw dH per point
    mH = dH.mean(1, keepdim=True)
    es = L.es.reshape(-1, 1, 1)
    r = (dH - mH) / es / sN
    return (r ** 2).sum((0, 1)), ((mH / es / sN) ** 2).sum((0, 1)) * L.nS


# ================================================================================ floor
def svals(Jm):
    try:
        return torch.linalg.svdvals(Jm)
    except Exception:  # gesdd fails on many repeated values (e.g. principal-angle cosines ~ 1): QR-iteration driver
        import scipy.linalg as sla
        return torch.from_numpy(sla.svd(Jm.numpy(), compute_uv=False, lapack_driver="gesvd"))


def plateau(S):
    """The value at which LAPACK gesdd (torch.linalg.svd) deflates the tail: the most repeated singular value, if it
    repeats more than 20 times (else None), and its multiplicity."""
    u, c = torch.unique(S, return_counts=True)
    i = int(c.argmax())
    return (float(u[i]), int(c[i])) if int(c[i]) > 20 else (None, 1)


def floor_compare(S, S2, win=20):
    """Two spectra of the same matrix under different rounding (index-wise): the median relative difference in a
    window of +-win indices around each level of sigma/sigma_max, and the resolution floor: the largest sigma/sigma_max
    at which the running median relative difference exceeds 0.1 (or the deflation plateau of either, if higher)."""
    s0 = float(S[0]); rel = S / s0
    d = ((S2 - S).abs() / S).numpy()
    n = len(S)
    run = np.array([np.median(d[max(i - win, 0):i + win + 1]) for i in range(n)])
    pl = [p for p in (plateau(S)[0], plateau(S2)[0]) if p is not None]
    bad = run > 0.1
    if pl:
        bad |= (S.numpy() <= max(pl) * (1 + 1e-12)) | (S2.numpy() <= max(pl) * (1 + 1e-12))
    i_fl = int(np.argmax(bad)) if bad.any() else n
    lv = {}
    for l in (1e-8, 1e-12, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-20):
        i = int(torch.searchsorted(-rel.flip(0), torch.tensor(-l, dtype=F64)))  # not used directly
        i = int((rel >= l).sum())
        if i < n:
            lv[f"{l:g}"] = float(run[i])
    return dict(floor_idx=i_fl, floor_sigma_rel=float(rel[min(i_fl, n - 1)]), n_resolved=i_fl, median_rel_diff_at=lv)


def cmd_floor():
    import scipy.linalg as sla
    L = LL.load_all(); t0 = time.perf_counter()
    out, arrs = {}, {}
    pts = {"T": L.vT, "A": L.al[L.A]}
    for key in ("T_energy", "T_coord", "A_energy"):
        z = LL.svd(key); S = torch.from_numpy(z["S"]); rel = S / S[0]
        pv, pc = plateau(S)
        out[key] = dict(sigma_max=float(S[0]), cond=float(S[0] / S[-1]), counts_below={f"{c:g}": int((rel < c).sum()) for c in CUTS},
                        counts_below_extra={f"{c:g}": int((rel < c).sum()) for c in (1e-14, 1e-16, 1e-17, 1e-18)},
                        per_decade={f"[1e{d - 1},1e{d})": int(((rel < 10.0 ** d) & (rel >= 10.0 ** (d - 1))).sum()) for d in DECADES},
                        gesdd_plateau_rel=None if pv is None else pv / float(S[0]), gesdd_plateau_count=pc)
        arrs[f"S_{key}"] = S.numpy()
    for pname, v in pts.items():
        kinds = [("energy", lambda w: LL.energy_jac(w)[0])] + ([("coord", lambda w: LL.coord_jac(w)[0])] if pname == "T" else [])
        for kind, jf in kinds:
            key = f"{pname}_{kind}"; res = {}
            Jm = jf(v)
            S = torch.from_numpy(LL.svd(key)["S"]); Vh = torch.from_numpy(LL.svd(key)["Vh"]); nJ = float(S[0]); rel = S / nJ
            res["recomputed_vs_cache_max_abs_over_smax"] = float((svals(Jm) - S).abs().max() / nJ)
            # (a) the Jacobian at a symmetry copy (same map, permuted summation order) -> rounding of the Jacobian itself
            pi, s = sym_perm(0)
            Jp = from_sym_cols(jf(LL.random_symmetry(v, 0)), pi, s)
            E = Jp - Jm
            res["sym_copy"] = dict(E_2norm_over_smax=float(torch.linalg.matrix_norm(E, 2)) / nJ,
                                   E_fro_over_J_fro=float(E.norm() / Jm.norm()), **floor_compare(S, svals(Jp)))
            del E
            # (b) SVD algorithm: gesvd (QR iteration) instead of gesdd (divide and conquer); rows permuted
            Sg = torch.from_numpy(sla.svd(Jm.numpy(), compute_uv=False, lapack_driver="gesvd"))
            arrs[f"S_{key}_gesvd"] = Sg.numpy()
            res["gesvd"] = floor_compare(S, Sg)
            res["gesvd_counts_below"] = {f"{c:g}": int((Sg / Sg[0] < c).sum()) for c in CUTS + (1e-16, 1e-17, 1e-18, 1e-20)}
            g = torch.Generator().manual_seed(7)
            res["rowperm"] = floor_compare(S, svals(Jm[torch.randperm(Jm.shape[0], generator=g)]))
            # (c) 1-ulp relative noise on every entry
            res["ulp_noise"] = floor_compare(S, svals(Jm * (1 + EPS64 * torch.randn(Jm.shape, generator=g, dtype=F64))))
            res["floor_rel"] = max(res[k]["floor_sigma_rel"] for k in ("sym_copy", "gesvd", "rowperm", "ulp_noise"))
            res["n_resolved"] = int((rel > res["floor_rel"]).sum())
            # column grading: parameter columns of J span many decades
            cn = Jm.norm(dim=0)
            res["column_norm_over_smax_quantiles"] = dict(zip(("min", "q01", "q10", "q50", "q90", "max"),
                                                             (cn / nJ).quantile(torch.tensor([0, .01, .1, .5, .9, 1], dtype=F64)).tolist()))
            Sb = svals(Jm / cn.clamp(min=1e-300)); relb = Sb / Sb[0]
            arrs[f"S_{key}_colnorm"] = Sb.numpy()
            res["column_normalised"] = dict(sigma_max=float(Sb[0]), counts_below={f"{c:g}": int((relb < c).sum()) for c in CUTS})
            # (d) refined tail: J restricted to the span of the right singular vectors below 1e-15, re-SVD'd (its own
            # floor is eps |J V0|, far below eps sigma_max); compare with the same at the symmetry copy
            V0 = Vh[rel < 1e-15].T.contiguous()
            P0 = Jm @ V0
            _, S0, W0 = torch.linalg.svd(P0, full_matrices=False)
            Vr = V0 @ W0.T                                       # refined tail directions (columns)
            Pr = Jm @ Vr; Prs = Jp @ Vr
            Prn = (Jm * (1 + EPS64 * torch.randn(Jm.shape, generator=g, dtype=F64))) @ Vr
            val = Pr.norm(dim=0); noise_s = (Prs - Pr).norm(dim=0); noise_n = (Prn - Pr).norm(dim=0)
            bound = EPS64 * (Jm.abs() @ Vr.abs()).norm(dim=0)    # componentwise rounding scale of J v
            fl_dir = torch.maximum(torch.maximum(noise_s, noise_n), bound)
            S0s = svals(Jp @ V0)
            fc = floor_compare(S0, S0s)
            res["refined_tail"] = dict(dim=V0.shape[1], top_rel=float(S0[0]) / nJ, sym_copy_floor_rel=fc["floor_sigma_rel"] * float(S0[0]) / nJ,
                                       counts_below={f"{c:g}": int((S0 / nJ < c).sum()) for c in (1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21)},
                                       noise_symcopy_over_smax_median=float((noise_s / nJ).median()),
                                       noise_ulp_over_smax_median=float((noise_n / nJ).median()),
                                       eps_bound_over_smax_median=float((bound / nJ).median()),
                                       value_over_smax_median=float((val / nJ).median()),
                                       n_value_gt_10x_floor=int((val > 10 * fl_dir).sum()), n_value_lt_10x_floor=int((val <= 10 * fl_dir).sum()),
                                       n_value_lt_floor=int((val <= fl_dir).sum()),
                                       value_over_floor_quantiles=(val / fl_dir).quantile(torch.tensor([0, .1, .5, .9, 1], dtype=F64)).tolist())
            arrs[f"S_{key}_refined_tail"] = S0.numpy(); arrs[f"S_{key}_refined_tail_sym"] = S0s.numpy()
            arrs[f"refined_tail_{key}_val"] = val.numpy(); arrs[f"refined_tail_{key}_floor"] = fl_dir.numpy()
            del Jp, P0, Pr, Prs, Prn
            # exact symmetry tangents
            D = symmetry_tangents(v)["D"]
            relv = (Jm @ D).norm(dim=0) / (nJ * D.norm(dim=0))
            res["symmetry_tangents"] = dict(gate_scale_max_rel=float(relv[:16].max()), gate_scale_per_layer=relv[:16].tolist(),
                                            translation_max_rel=float(relv[16:].max()), rank_of_30_generators=int(torch.linalg.matrix_rank(D)),
                                            max_rel_with_column_weights=float(((Jm @ D).norm(dim=0) / ((Jm.abs() @ D.abs()).norm(dim=0))).max()))
            out[key].update(res)
            print(f"{key}: floors (sigma/smax): sym copy {res['sym_copy']['floor_sigma_rel']:.1e}, gesvd {res['gesvd']['floor_sigma_rel']:.1e}, "
                  f"rowperm {res['rowperm']['floor_sigma_rel']:.1e}, ulp {res['ulp_noise']['floor_sigma_rel']:.1e} -> {res['floor_rel']:.1e}, "
                  f"{res['n_resolved']} resolved; |E|_2/smax {res['sym_copy']['E_2norm_over_smax']:.1e}; median rel diff sym copy "
                  f"{res['sym_copy']['median_rel_diff_at']}; gesvd counts {res['gesvd_counts_below']}; column norms {res['column_norm_over_smax_quantiles']}; "
                  f"column-normalised counts {res['column_normalised']['counts_below']}; refined tail {res['refined_tail']}; "
                  f"symmetry tangents gate {res['symmetry_tangents']['gate_scale_max_rel']:.1e} transl {res['symmetry_tangents']['translation_max_rel']:.1e} "
                  f"(column-weighted {res['symmetry_tangents']['max_rel_with_column_weights']:.1e})  [{time.perf_counter() - t0:.0f} s]", flush=True)
            del Jm
    np.savez(f"{RES}/sloppy_floor.npz", **arrs)
    out["seconds"] = time.perf_counter() - t0
    jdump(out, f"{RES}/sloppy_floor.json")


def symmetry_tangents(v):
    """(P, 30) exact symmetry tangents at v: 16 gate-scale generators d_k = (gate, -W2, -b2) on layer k's block, and the
    14 translation generators of landscape_lab._shift_basis."""
    Bv = LL.blocks(v)
    Dg = torch.zeros(LL.N_LAYERS, LL.N_LAYERS, LL.PER, dtype=F64)
    for k in range(LL.N_LAYERS):
        Dg[k, k, LL.G] = Bv[k, LL.G]; Dg[k, k, LL.W2] = -Bv[k, LL.W2]; Dg[k, k, LL.B2] = -Bv[k, LL.B2]
    return dict(D=torch.cat([Dg.view(LL.N_LAYERS, LL.P).T, LL._shift_basis(v)], 1))


# ================================================================================ per-layer redundancy
def layer_inputs(v):
    """Inputs s_k (N,) of each layer's conditioner at the training points for the flow at v (list of 16), and update_q."""
    L = LL.load_all(); LL.set_vec(L.m, v)
    with torch.no_grad():
        x = L.m.toy(L.aa.reshape(-1, 2))
        ins = []
        for lay in L.m.flow.layers:
            q, p = x[:, :1], x[:, 1:]
            ins.append((p if lay.update_q else q)[:, 0].clone())
            x = lay(x)
    return ins, [lay.update_q for lay in L.m.flow.layers]


def layer_fun_jac(b, s):
    """Jacobian (N, 194) of f(s) = g (W2 tanh(W1 s + b1) + b2) w.r.t. the block b (NAMES order) at inputs s (N,), and h."""
    g, w1, b1, w2, b2 = b[LL.G], b[LL.W1], b[LL.B1], b[LL.W2], b[LL.B2]
    h = torch.tanh(s[:, None] * w1[None, :] + b1[None, :])
    dh = 1 - h * h
    cols = [(h @ w2 + b2)[:, None], g * w2[None, :] * dh * s[:, None], g * w2[None, :] * dh, g * h, torch.full_like(s, float(g))[:, None]]
    return torch.cat(cols, 1), h


def cmd_units():
    L = LL.load_all(); t0 = time.perf_counter()
    out = dict(layers=[]); arrs = {}
    pts = {"T": L.vT, "A": L.al[L.A]}
    for pname, v in pts.items():
        ins, upd = layer_inputs(v)
        Bv = LL.blocks(v)
        tot = {f"{c:g}": 0 for c in CUTS}; tot_h = {f"{c:g}": 0 for c in CUTS}
        rows = []
        for k in range(LL.N_LAYERS):
            s = ins[k]; b = Bv[k]
            Dk, h = layer_fun_jac(b, s)
            Sk = torch.linalg.svdvals(Dk); relk = Sk / Sk[0]
            Hh = torch.cat([h, torch.ones_like(s)[:, None]], 1)
            Sh = torch.linalg.svdvals(Hh); relh = Sh / Sh[0]
            arrs[f"S_layer_{pname}_{k}"] = Sk.numpy(); arrs[f"S_feat_{pname}_{k}"] = Sh.numpy()
            pre = s[:, None] * b[LL.W1][None, :] + b[LL.B1][None, :]
            hstd = h.std(0)
            f = h @ b[LL.W2] + b[LL.B2]
            contrib = (b[LL.W2].abs() * hstd) / f.std()
            sat = (pre.abs().min(0).values > 18.0)               # tanh within 1e-15 of +-1 at every input: constant
            lin = (pre.abs().max(0).values < 1e-4)               # tanh linear to 1e-8 relative at every input
            small = contrib < 1e-10
            U = torch.stack([b[LL.W1], b[LL.B1]], 1); Un = U / U.norm(dim=1, keepdim=True)
            cs = (Un @ Un.T).abs(); cs.fill_diagonal_(0)
            dup = int(((cs > 1 - 1e-12) & ((U[:, None, :].norm(dim=-1) - U[None, :, :].norm(dim=-1)).abs() < 1e-8 * U.norm(dim=1)[:, None])).triu().sum())
            row = dict(layer=k, update_q=bool(upd[k]), input_range=[float(s.min()), float(s.max())], gate=float(b[LL.G]),
                       f_std=float(f.std() * b[LL.G].abs()), n_saturated=int(sat.sum()), n_linear=int(lin.sum()), n_small_contrib=int(small.sum()),
                       n_duplicate_pairs=dup, min_contrib=float(contrib.min()), max_preact=float(pre.abs().max()),
                       fun_jac_counts_below={f"{c:g}": int((relk < c).sum()) for c in CUTS},
                       feature_counts_below={f"{c:g}": int((relh < c).sum()) for c in CUTS},
                       fun_jac_sigma_max=float(Sk[0]))
            for c in CUTS:
                tot[f"{c:g}"] += row["fun_jac_counts_below"][f"{c:g}"]; tot_h[f"{c:g}"] += row["feature_counts_below"][f"{c:g}"]
            rows.append(row)
            print(f"{pname} layer {k:2d} ({'q' if upd[k] else 'p'}-shear) in [{float(s.min()):+.3f}, {float(s.max()):+.3f}] gate {float(b[LL.G]):+.3e}: "
                  f"saturated {row['n_saturated']}, linear {row['n_linear']}, small {row["n_small_contrib"]}, dup {dup}; "
                  f"f-Jacobian (194) below 1e-4/1e-8/1e-12/1e-15: " + "/".join(str(row['fun_jac_counts_below'][f'{c:g}']) for c in CUTS)
                  + "; features (65): " + "/".join(str(row['feature_counts_below'][f'{c:g}']) for c in CUTS), flush=True)
        out[pname] = dict(layers=rows, sum_fun_jac_counts_below=tot, sum_feature_counts_below=tot_h)
        print(f"{pname}: per-layer function-Jacobian near-null counts summed over 16 layers: {tot}; features: {tot_h}", flush=True)
    out.pop("layers")
    out["seconds"] = time.perf_counter() - t0
    np.savez(f"{RES}/sloppy_units.npz", **arrs)
    jdump(out, f"{RES}/sloppy_units.json")


# ================================================================================ twist decomposition
def alt_sets():
    """Point sets other than the training set, for the energy cost of the same directions off the training points:
    training tori with angles shifted by half a step; the cell midpoints between training tori with the training
    angles; the landscape set (100 held-out-style tori x 128 angles at offset 0.25). Each: (aa, es)."""
    import gauge_lab  # noqa: F401  (energy_jacobian)
    L = LL.load_all(); w0 = float(L.t.nu0)
    J_tr = L.es.reshape(-1) / w0
    J_lo, J_hi = L.Ev.J_range
    Jg = C.j_grid(len(J_tr), J_lo, J_hi, w0, "asinh", 0.03, offset=0.0)
    assert torch.allclose(Jg, J_tr, rtol=1e-12, atol=0), "training tori are not the plain asinh grid"
    J_mid = C.j_grid(len(J_tr) - 1, J_lo, J_hi, w0, "asinh", 0.03, offset=0.5)
    return {"train_tori_shifted_angles": (C.aa_points(J_tr, C.angles(L.nS, 0.0)), (w0 * J_tr).reshape(-1, 1)),
            "mid_tori_train_angles": (C.aa_points(J_mid, C.angles(L.nS, 0.5)), (w0 * J_mid).reshape(-1, 1)),
            "landscape": (L.aa_land, (w0 * L.J_land).reshape(-1, 1))}


def energy_cost_alt(v, V):
    """|J_E(alt set) v_i| for the columns of V (P, K), on each alternative point set (dict of (K,))."""
    import gauge_lab as GL
    L = LL.load_all(); LL.set_vec(L.m, v)
    out = {}
    for name, (aa, es) in alt_sets().items():
        Jm = GL.energy_jacobian(L.m, aa, L.t, aa.shape[0], aa.shape[1], es)
        out[name] = (Jm @ V).norm(dim=0)
        del Jm
    return out


def decade_summary(rel, comp, dx_rel, resolved, sig_rec_rel=None, sig_ok=None):
    """Per decade of rel = sigma/sigma_max: counts, medians and aggregated fractions of the decomposition. With sig_ok,
    only directions whose energy cost is resolved go in the decades, and a last row collects the others."""
    rows = []
    ok = torch.ones_like(rel, dtype=torch.bool) if sig_ok is None else sig_ok
    for d in DECADES + ([None] if sig_ok is not None else []):
        if d is None:
            sel = ~ok
        else:
            sel = (rel < 10.0 ** d) & (rel >= 10.0 ** (d - 1)) & ok
            if d == DECADES[-1]:
                sel = (rel < 10.0 ** d) & ok
        n = int(sel.sum())
        if n == 0:
            continue
        tot = comp["tot"][sel]
        fr = {k: comp[k][sel] / tot for k in ("twist", "other", "normal")}
        rs = sel & resolved
        frr = {k: comp[k][rs] / comp["tot"][rs] for k in ("twist", "other", "normal")}
        row = dict(decade="sigma unresolved" if d is None else (f"[1e{d - 1},1e{d})" if d != DECADES[-1] else f"<1e{d}"), n=n, n_dx_resolved=int(rs.sum()),
                   dx_rel_median=float(dx_rel[sel].median()), dx_rel_max=float(dx_rel[sel].max()),
                   **{f"{k}_frac_median": float(fr[k].median()) for k in fr},
                   **{f"{k}_frac_aggregate": float(comp[k][sel].sum() / tot.sum()) for k in fr},
                   **({f"{k}_frac_median_resolved": float(frr[k].median()) for k in frr} if int(rs.sum()) else {}),
                   **({"twist_frac_min_resolved": float(frr["twist"].min())} if int(rs.sum()) else {}))
        if sig_rec_rel is not None:
            row["sigma_from_dx_over_sigma_median"] = float(sig_rec_rel[sel].median())
        rows.append(row)
    return rows


def cmd_twist(pname):
    L = LL.load_all(); t0 = time.perf_counter()
    v = {"T": L.vT, "A": L.al[L.A], "A_min_shift": L.al_min_shift[L.A]}[pname]
    z = LL.svd({"T": "T_energy", "A": "A_energy", "A_min_shift": "A_energy_min_shift"}[pname]); S = torch.from_numpy(z["S"]); Vh = torch.from_numpy(z["Vh"])
    assert (pname == "T" and str(z["gauge"]) == "teacher") or (pname == "A" and str(z["gauge"]) == "aligned") or \
        (pname == "A_min_shift" and str(z["gauge"]) == "aligned_min_shift"), z["gauge"]
    rel = S / S[0]
    nO, nS, P = L.nO, L.nS, LL.P
    JE, _ = LL.energy_jac(v)
    JC, _ = LL.coord_jac(v)
    sC = float(torch.linalg.matrix_norm(JC, 2))
    print(f"{pname}: Jacobians in {time.perf_counter() - t0:.0f} s; sigma_E max {float(S[0]):.4e}, sigma_C max {sC:.4e}", flush=True)
    # |J_E v_i| recomputed against the cached S
    sig_dir = (JE @ Vh.T).norm(dim=0)
    # symmetry copy: per-direction rounding noise of J_E v_i and J_C v_i
    pi, s = sym_perm(0)
    vs = LL.random_symmetry(v, 0)
    JEs, _ = LL.energy_jac(vs); JCs, _ = LL.coord_jac(vs)
    Vs = to_sym(Vh.T.contiguous(), pi, s)                    # (P, P) columns = v_i at the copy
    noiseE = (JEs @ Vs - JE @ Vh.T).norm(dim=0)
    del JEs
    Dmat = JC @ Vh.T                                           # (nO nS 2, P): normalised coordinate change of every v_i
    noiseC = (JCs @ Vs - Dmat).norm(dim=0)
    Ds_mat = JCs @ Vs
    del JCs, Vs
    dx = Dmat.norm(dim=0)
    VhTa = Vh.T.abs()
    boundE = EPS64 * (JE.abs() @ VhTa).norm(dim=0)            # componentwise rounding scale of J_E v_i
    boundC = EPS64 * (JC.abs() @ VhTa).norm(dim=0)
    del VhTa
    flE = torch.maximum(noiseE, boundE); flC = torch.maximum(noiseC, boundC)
    print(f"   products done [{time.perf_counter() - t0:.0f} s]", flush=True)
    # tangents and decomposition
    scale = norm_scale()                                       # (nO, 1, 2)
    Tn = tangent(v) / scale                                    # normalised like the rows of J_C
    gH, _ = grad_H(v)
    comp = {k: [] for k in ("tot", "twist", "other", "normal", "g", "Gw")}
    comp_s = {k: [] for k in ("tot", "twist", "other", "normal")}
    eE2, eM2 = [], []
    for k0 in range(0, P, 388):
        D = Dmat[:, k0:k0 + 388].reshape(nO, nS, 2, -1)
        c = decompose(D, Tn)
        for k in comp:
            comp[k].append(c[k])
        cs = decompose(Ds_mat[:, k0:k0 + 388].reshape(nO, nS, 2, -1), Tn)
        for k in comp_s:
            comp_s[k].append(cs[k])
        a, b = energy_from_dx(D, gH, scale)
        eE2.append(a); eM2.append(b)
    comp = {k: torch.cat(vv, -1) for k, vv in comp.items()}
    comp_s = {k: torch.cat(vv, -1) for k, vv in comp_s.items()}
    sig_from_dx = torch.cat(eE2).sqrt(); mean_part = torch.cat(eM2).sqrt()
    del Ds_mat
    print(f"   decomposition done [{time.perf_counter() - t0:.0f} s]", flush=True)
    dx_rel = dx / sC
    resolved = dx > 10 * flC                                   # coordinate change resolved above its rounding floor
    sig_resolved = sig_dir > 10 * flE
    frac = {k: comp[k] / comp["tot"] for k in ("twist", "other", "normal")}
    rel_d = sig_dir / S[0]                                     # direct energy cost of each direction (the SVD plateaus below ~1e-16)
    frac_s = {k: comp_s[k] / comp_s["tot"] for k in ("twist", "other", "normal")}
    # normal part vs energy singular value (sigma ~ |grad_u H| / es * |n|)
    nrm = comp["normal"].sqrt()
    out = dict(point=pname, gauge=str(z["gauge"]), sigma_E_max=float(S[0]), sigma_C_max=sC,
               sigma_dir_vs_cache_max_abs_over_smax=float((sig_dir - S).abs().max() / S[0]),
               sigma_from_dx_vs_sigma_max_rel_top100=float(((sig_from_dx - S).abs() / S)[:100].max()),
               counts_below={f"{c:g}": int((rel < c).sum()) for c in CUTS},
               n_sigma_resolved=int(sig_resolved.sum()), sigma_resolved_smallest_rel=float(rel[sig_resolved].min()),
               first_unresolved_sigma_idx=int(torch.nonzero(~sig_resolved)[0]) if bool((~sig_resolved).any()) else P,
               n_dx_resolved=int(resolved.sum()), dx_floor_median_over_sC=float((flC / sC).median()),
               dx_eps_bound_median_over_sC=float((boundC / sC).median()), sigma_eps_bound_median_over_smax=float((boundE / S[0]).median()),
               dx_noise_median_over_sC=float((noiseC / sC).median()),
               dx_noise_max_over_sC=float((noiseC / sC).max()), sigma_noise_median_over_smax=float((noiseE / S[0]).median()),
               decades_by_svd_sigma=decade_summary(rel, comp, dx_rel, resolved, sig_from_dx / S),
               decades=decade_summary(rel_d, comp, dx_rel, resolved, sig_from_dx / sig_dir, sig_ok=sig_resolved))
    # the theory test: directions with tiny energy cost but resolved coordinate change
    tests = {}
    for cut in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        for dxmin in (1e-6, 1e-9, 1e-12):
            sel = (rel_d < cut) & resolved & (dx_rel > dxmin)
            n = int(sel.sum())
            key = f"sigma<{cut:g},dx>{dxmin:g}"
            if n == 0:
                tests[key] = dict(n=0); continue
            tw = frac["twist"][sel]
            tests[key] = dict(n=n, twist_frac_min=float(tw.min()), twist_frac_median=float(tw.median()),
                              n_twist_frac_gt_0p99=int((tw > 0.99).sum()), n_twist_frac_gt_0p9=int((tw > 0.9).sum()),
                              other_frac_max=float(frac["other"][sel].max()), other_frac_median=float(frac["other"][sel].median()),
                              normal_frac_max=float(frac["normal"][sel].max()), normal_frac_median=float(frac["normal"][sel].median()),
                              twist_frac_agg=float(comp["twist"][sel].sum() / comp["tot"][sel].sum()),
                              frac_resolution_median=float(((flC[sel] / dx[sel]) ** 2).median()),
                              sigma_E_resolved=int(sig_resolved[sel].sum()),
                              symcopy_twist_frac_max_absdiff=float((frac_s["twist"][sel] - tw).abs().max()))
    out["null_direction_tests"] = tests
    # stiff directions for contrast
    st = rel > 1e-4
    out["stiff_sigma>1e-4"] = dict(n=int(st.sum()), twist_frac_median=float(frac["twist"][st].median()),
                                   normal_frac_median=float(frac["normal"][st].median()),
                                   other_frac_median=float(frac["other"][st].median()))
    both = (nrm > 0) & sig_resolved & resolved & (rel > 1e-12)
    ratio = sig_dir[both] / nrm[both]
    out["sigma_over_normal_norm"] = dict(n=int(both.sum()), median=float(ratio.median()), p05=float(ratio.quantile(0.05)),
                                         p95=float(ratio.quantile(0.95)))
    out["torus_mean_part_of_dH_over_sigma"] = dict(median=float((mean_part / sig_from_dx).median()),
                                                   max_top1000=float((mean_part / sig_from_dx)[:1000].max()))
    # twist profiles: how many independent g(J) at first order, from the energy-cheap directions
    prof = {}
    Gw = comp["Gw"]                                            # (nO, P)
    for cut in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        sel = rel < cut
        Sg = torch.linalg.svdvals(Gw[:, sel]) / sC
        prof[f"sigma<{cut:g}"] = dict(n_dirs=int(sel.sum()), sv_over_sC=Sg.tolist(),
                                      rank_above={f"{c:g}": int((Sg > c).sum()) for c in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)})
    Sg_all = torch.linalg.svdvals(Gw) / sC
    prof["all"] = dict(n_dirs=P, sv_over_sC=Sg_all.tolist(), rank_above={f"{c:g}": int((Sg_all > c).sum()) for c in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)})
    out["twist_profiles"] = prof
    for cut in (1e-8, 1e-12):
        print(f"   twist-profile singular values (sigma_E<{cut:g}) / sigma_C: " + " ".join(f"{x:.1e}" for x in prof[f'sigma<{cut:g}']['sv_over_sC'][:12]), flush=True)
    # baselines: the twist content of the stiffest coordinate directions (right singular vectors of J_C) and of random
    # parameter directions, so that "mostly twist" can be judged against what a generic coordinate change looks like
    _, SCv, VhC = torch.linalg.svd(JC, full_matrices=False)
    gB = torch.Generator().manual_seed(11)
    R = torch.randn(P, 200, generator=gB, dtype=F64); R = R / R.norm(dim=0)
    base = {}
    for bname, Vb in (("coord_top50", VhC[:50].T), ("coord_1e-4_band", VhC[(SCv / SCv[0] < 1e-3) & (SCv / SCv[0] >= 1e-5)].T), ("random200", R)):
        cb = decompose((JC @ Vb).reshape(nO, nS, 2, -1), Tn)
        eb = (JE @ Vb).norm(dim=0) / S[0]
        base[bname] = dict(n=Vb.shape[1], twist_frac_median=float((cb["twist"] / cb["tot"]).median()),
                           other_frac_median=float((cb["other"] / cb["tot"]).median()), normal_frac_median=float((cb["normal"] / cb["tot"]).median()),
                           twist_frac_agg=float(cb["twist"].sum() / cb["tot"].sum()), energy_cost_rel_median=float(eb.median()))
        print(f"   baseline {bname}: {base[bname]}", flush=True)
    out["baselines"] = base
    del VhC
    # energy cost of the same directions off the training points (tests whether energy-null means energy-null elsewhere)
    alt = energy_cost_alt(v, Vh.T.contiguous())
    cls = dict(twist_dominated=frac["twist"] > 0.9, other_dominated=frac["other"] > 0.5)
    offt = {}
    for lo_, hi_ in ((1e-8, 1e-4), (1e-12, 1e-8), (1e-14, 1e-12), (1e-16, 1e-14), (0.0, 1e-16)):
        band = (rel_d < hi_) & (rel_d >= lo_) & resolved & sig_resolved
        for cn_, cm in cls.items():
            sel = band & cm
            if int(sel.sum()) == 0:
                continue
            offt[f"[{lo_:g},{hi_:g}) {cn_}"] = dict(n=int(sel.sum()), dx_rel_median=float(dx_rel[sel].median()),
                                                     **{f"{k}_over_train_median": float((a[sel] / sig_dir[sel]).median()) for k, a in alt.items()},
                                                     **{f"{k}_over_train_p90": float((a[sel] / sig_dir[sel]).quantile(0.9)) for k, a in alt.items()})
    out["off_training_energy_cost"] = offt
    for k, r in offt.items():
        print(f"   off-training energy cost {k}: {r}", flush=True)
    arrs = dict(S=S.numpy(), rel=rel.numpy(), **{f"sig_alt_{k}": a.numpy() for k, a in alt.items()}, sig_dir=sig_dir.numpy(), sig_noise=noiseE.numpy(), sig_from_dx=sig_from_dx.numpy(),
                dx=dx.numpy(), dx_noise=noiseC.numpy(), dx_bound=boundC.numpy(), sig_bound=boundE.numpy(), sC=sC, twist=comp["twist"].numpy(), other=comp["other"].numpy(),
                normal=comp["normal"].numpy(), tot=comp["tot"].numpy(), twist_frac_symcopy=frac_s["twist"].numpy(),
                g=comp["g"].numpy(), Gw=comp["Gw"].numpy(), J_train=L.es.reshape(-1).numpy() / float(L.t.nu0),
                zmax_kpc=C.zmax_kpc_of_J(L.es.reshape(-1) / float(L.t.nu0), C.MWVertical().phi).numpy(),
                mean_part=mean_part.numpy())
    if pname == "T":
        cmp, extra = compare_null_spaces(JE, JC, S, Vh, Dmat, Tn, sC)
        out["energy_vs_coord"] = cmp; arrs.update(extra)
    np.savez(f"{RES}/sloppy_twist_{pname}.npz", **arrs)
    out["seconds"] = time.perf_counter() - t0
    jdump(out, f"{RES}/sloppy_twist_{pname}.json")
    for r in out["decades"]:
        print(f"   {r['decade']:>16s} n {r['n']:4d} (dx resolved {r['n_dx_resolved']:4d}) |dx|/sC med {r['dx_rel_median']:.1e} max {r['dx_rel_max']:.1e}; "
              f"twist {r['twist_frac_median']:.3f} other {r['other_frac_median']:.2e} normal {r['normal_frac_median']:.2e} (median); "
              f"agg twist {r['twist_frac_aggregate']:.3f}; resolved twist med {r.get('twist_frac_median_resolved', float('nan')):.4f} "
              f"min {r.get('twist_frac_min_resolved', float('nan')):.3f}", flush=True)
    for k, r in tests.items():
        print(f"   {k}: {r}", flush=True)
    print(f"   sigma/|normal| {out['sigma_over_normal_norm']}; stiff {out['stiff_sigma>1e-4']}  [{out['seconds']:.0f} s]", flush=True)


def compare_null_spaces(JE, JC, S_E, VhE, Dmat, Tn, sC):
    """J_E vs J_C at T: near-null counts, principal angles between the small-sigma subspaces, J_C restricted to the J_E
    near-null subspace (and its top coordinate patterns decomposed), J_E restricted to the J_C near-null subspace."""
    zc = LL.svd("T_coord"); S_C = torch.from_numpy(zc["S"]); VhC = torch.from_numpy(zc["Vh"])
    relE, relC = S_E / S_E[0], S_C / S_C[0]
    nO, nS = Tn.shape[:2]
    out, extra = {}, {}
    for cut in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        NE, NC = VhE[relE < cut], VhC[relC < cut]
        kE, kC = NE.shape[0], NC.shape[0]
        cos = svals(NE @ NC.T)                                  # min(kE, kC) cosines of principal angles
        # coordinate singular values restricted to the energy near-null subspace
        DN = Dmat[:, relE < cut]
        U2, S2, W2 = torch.linalg.svd(DN, full_matrices=False)
        S2r = S2 / sC
        # energy singular values restricted to the coordinate near-null subspace
        SE_C = svals(JE @ NC.T) / S_E[0]
        # decompose the coordinate patterns of the top coordinate-active energy-null directions
        m = int((S2r > cut).sum())
        mm = min(max(m, 1), 200)
        pat = (U2[:, :mm] * S2[:mm]).reshape(nO, nS, 2, -1)
        c = decompose(pat, Tn)
        tw = c["twist"] / c["tot"]
        oth = c["other"] / c["tot"]; nor = c["normal"] / c["tot"]
        ecost = (JE @ (NE.T @ W2[:mm].T)).norm(dim=0) / S_E[0]   # energy cost of each pattern, sigma_E-relative
        ratio = S2r[:mm] / ecost.clamp(min=1e-300)             # coordinate change per energy cost (both relative)
        out[f"{cut:g}"] = dict(k_E=kE, k_C=kC, excess=kE - kC,
                               cos_principal_gt_0p999=int((cos > 0.999).sum()), cos_principal_gt_0p9=int((cos > 0.9).sum()),
                               cos_principal_min=float(cos.min()) if cos.numel() else None,
                               n_coord_active_in_E_null=m, coord_sv_in_E_null_top=S2r[:40].tolist(),
                               E_sv_in_C_null_max=float(SE_C.max()) if SE_C.numel() else None,
                               n_E_active_in_C_null=int((SE_C > cut).sum()),
                               top_patterns_twist_frac=tw[:min(mm, 60)].tolist(),
                               top_patterns_other_frac=oth[:min(mm, 60)].tolist(), top_patterns_normal_frac=nor[:min(mm, 60)].tolist(),
                               top_patterns_energy_cost_rel=ecost[:min(mm, 60)].tolist(), top_patterns_dx_over_cost=ratio[:min(mm, 60)].tolist(),
                               top_patterns_twist_frac_min_over_active=float(tw[:max(m, 1)].min()),
                               top_patterns_twist_frac_median_over_active=float(tw[:max(m, 1)].median()))
        extra[f"cmp_cos_{cut:g}"] = cos.numpy(); extra[f"cmp_coordsv_{cut:g}"] = S2r.numpy(); extra[f"cmp_Esv_Cnull_{cut:g}"] = SE_C.numpy()
        print(f"   J_E vs J_C at {cut:g}: k_E {kE} k_C {kC} excess {kE - kC}; principal cos>0.999 {out[f'{cut:g}']['cos_principal_gt_0p999']}, "
              f">0.9 {out[f'{cut:g}']['cos_principal_gt_0p9']}; coord-active (sv/sC > {cut:g}) in E-null {m}; max E sv in C-null "
              f"{out[f'{cut:g}']['E_sv_in_C_null_max']:.1e} ({out[f'{cut:g}']['n_E_active_in_C_null']} above cut); twist frac of "
              f"active patterns min {out[f'{cut:g}']['top_patterns_twist_frac_min_over_active']:.4f} median "
              f"{out[f'{cut:g}']['top_patterns_twist_frac_median_over_active']:.4f}", flush=True)
        print("      top patterns (dx/sC, Ecost/smaxE, twist, other, normal): " + "; ".join(
            f"{float(S2r[j]):.1e},{float(ecost[j]):.1e},{float(tw[j]):.3f},{float(oth[j]):.3f},{float(nor[j]):.1e}" for j in range(min(mm, 8))), flush=True)
    return out, extra


# ================================================================================ lines from T
def cmd_line():
    """eps(T + t |theta_T| d) and the RMS normalised coordinate deviation from T on the training set, for unit directions
    d: the most coordinate-active direction inside the J_E near-null space sigma_E/sigma_max < 1e-8 (a first-order twist)
    and < 1e-14 (its first-order coordinate change is ~1e-11 sC, so it acts as a redundancy), the least coordinate-active
    direction (a redundancy), the gate-scale tangent of layer 4, a random unit direction and the stiffest right singular
    vector; log-log slopes over the first decades where eps is above 1e3 x the floor. Twist signature: coordinate slope 1
    with energy slope 2; a redundancy or symmetry tangent has slope 2 in both; a generic direction slope 1 in both."""
    L = LL.load_all(); t0 = time.perf_counter()
    z = LL.svd("T_energy"); S = torch.from_numpy(z["S"]); Vh = torch.from_numpy(z["Vh"]); rel = S / S[0]
    JC, _ = LL.coord_jac(L.vT)
    dirs = {}
    for cut in (1e-14, 1e-8):
        NE = Vh[rel < cut]
        U2, S2, W2 = torch.linalg.svd(JC @ NE.T, full_matrices=False)
        dirs[f"twist_like_{cut:g}"] = NE.T @ W2[0]
    # a parametrisation redundancy: the direction of least coordinate change (refined J_C tail), and an exact symmetry
    # tangent (gate scale of layer 4), whose straight line leaves the symmetry orbit at second order
    zc = LL.svd("T_coord"); relC = torch.from_numpy(zc["S"]) / float(zc["S"][0]); V0 = torch.from_numpy(zc["Vh"])[relC < 1e-15].T.contiguous()
    _, _, W0 = torch.linalg.svd(JC @ V0, full_matrices=False)
    dirs["redundancy"] = V0 @ W0[-1]
    dg = symmetry_tangents(L.vT)["D"][:, 4]; dirs["gate_scale_tangent_l4"] = dg / dg.norm()
    g = torch.Generator().manual_seed(3)
    d = torch.randn(LL.P, generator=g, dtype=F64); dirs["random"] = d / d.norm()
    dirs["stiffest"] = Vh[0].clone()
    Tn = tangent(L.vT) / norm_scale()
    X = L.aa.reshape(-1, 2)
    s_pt = C.coord_scale(L.es, float(L.t.nu0)).repeat_interleave(L.nS, 0)
    with torch.no_grad():
        yT = LL._forward(L.vT[None], X)[0]
    coord_err = lambda V: torch.cat([((LL._forward(V[k:k + 16], X) - yT) / s_pt).pow(2).mean((1, 2)).sqrt() for k in range(0, V.shape[0], 16)])
    ts = torch.logspace(-9, -1, 25, dtype=F64)
    out = dict(t_over_thetaT=ts.tolist(), eps_T=float(LL.eps_batch(L.vT)),
               eps_T_train=float(LL.eps_batch(L.vT, x=L.aa.reshape(-1, 2), nO=L.nO, nS=L.nS)))
    for name, dv in dirs.items():
        c = decompose((JC @ dv).reshape(L.nO, L.nS, 2, 1), Tn)
        sigE = float((LL.energy_jac(L.vT)[0] @ dv).norm()) / float(S[0])
        V = L.vT[None] + (ts[:, None] * L.nT) * dv[None]
        e = LL.eps_batch(V)
        e_tr = LL.eps_batch(V, x=L.aa.reshape(-1, 2), nO=L.nO, nS=L.nS)  # the same on the training set
        with torch.no_grad():
            ce = coord_err(V)                                        # RMS normalised coordinate deviation from T's map
        okc = ce > 1e-13
        sl_c = float(np.polyfit(ts.log10()[okc][:6].numpy(), ce.log10()[okc][:6].numpy(), 1)[0]) if int(okc.sum()) >= 6 else None
        ok = e > 1e3 * out["eps_T"]
        lt, le = ts.log10()[ok], e.log10()[ok]
        sl_lo = float(np.polyfit(lt[:6].numpy(), le[:6].numpy(), 1)[0]) if int(ok.sum()) >= 6 else None
        loc = [float((le[i + 1] - le[i]) / (lt[i + 1] - lt[i])) for i in range(len(lt) - 1)]
        out[name] = dict(sigma_E_rel=sigE, dx_over_sC=float(c["tot"].sqrt()) / float(torch.linalg.matrix_norm(JC, 2)),
                         twist_frac=float(c["twist"] / c["tot"]), other_frac=float(c["other"] / c["tot"]), normal_frac=float(c["normal"] / c["tot"]),
                         eps=e.tolist(), eps_train=e_tr.tolist(), slope_first6_above_floor=sl_lo, local_slopes=loc,
                         coord_err=ce.tolist(), coord_slope_first6=sl_c)
        print(f"{name}: sigma_E/smax {sigE:.1e}, twist frac {out[name]['twist_frac']:.6f}; eps " + " ".join(f"{float(x):.1e}" for x in e[::3])
              + f"; slope (first 6 above 1e3 floor) {sl_lo}; coord err " + " ".join(f"{float(x):.1e}" for x in ce[::3]) + f"; coord slope {sl_c}", flush=True)
    out["seconds"] = time.perf_counter() - t0
    jdump(out, f"{RES}/sloppy_line.json")


# ================================================================================ summary
def cmd_summary():
    fl = json.load(open(f"{RES}/sloppy_floor.json"))
    un = json.load(open(f"{RES}/sloppy_units.json"))
    tw = {p: json.load(open(f"{RES}/sloppy_twist_{p}.json")) for p in ("T", "A", "A_min_shift") if os.path.exists(f"{RES}/sloppy_twist_{p}.json")}
    ln = json.load(open(f"{RES}/sloppy_line.json")) if os.path.exists(f"{RES}/sloppy_line.json") else None
    keep_fl = ("sigma_max", "cond", "counts_below", "counts_below_extra", "per_decade", "gesdd_plateau_rel", "gesdd_plateau_count",
               "gesvd_counts_below", "floor_rel", "n_resolved", "column_norm_over_smax_quantiles", "column_normalised",
               "refined_tail", "symmetry_tangents")
    spectra = {}
    for k in ("T_energy", "T_coord", "A_energy"):
        d = {kk: fl[k][kk] for kk in keep_fl if kk in fl[k]}
        d["floor_tests"] = {t: dict(floor_sigma_rel=fl[k][t]["floor_sigma_rel"], n_resolved=fl[k][t]["n_resolved"],
                                    median_rel_diff_at=fl[k][t]["median_rel_diff_at"]) for t in ("sym_copy", "gesvd", "rowperm", "ulp_noise")}
        d["sym_copy_E_2norm_over_smax"] = fl[k]["sym_copy"]["E_2norm_over_smax"]
        spectra[k] = d
    units = {p: dict(sum_layer_fun_jac_counts_below=un[p]["sum_fun_jac_counts_below"], sum_layer_feature_counts_below=un[p]["sum_feature_counts_below"],
                     saturated_units=sum(r["n_saturated"] for r in un[p]["layers"]), small_contribution_units=sum(r["n_small_contrib"] for r in un[p]["layers"]),
                     linear_units=sum(r["n_linear"] for r in un[p]["layers"]), duplicate_pairs=sum(r["n_duplicate_pairs"] for r in un[p]["layers"]),
                     per_layer_fun_jac_below_1em15=[r["fun_jac_counts_below"]["1e-15"] for r in un[p]["layers"]]) for p in ("T", "A")}
    twist = {}
    for p, v in tw.items():
        d = dict(gauge=v["gauge"], sigma_E_max=v["sigma_E_max"], sigma_C_max=v["sigma_C_max"], n_sigma_resolved=v["n_sigma_resolved"],
                 n_dx_resolved=v["n_dx_resolved"], sigma_over_normal_norm=v["sigma_over_normal_norm"],
                 decades=[{kk: r[kk] for kk in ("decade", "n", "n_dx_resolved", "dx_rel_median", "twist_frac_median", "other_frac_median",
                                                 "normal_frac_median", "twist_frac_aggregate") if kk in r} for r in v["decades"]],
                 null_direction_tests={k: {kk: r[kk] for kk in ("n", "twist_frac_min", "twist_frac_median", "n_twist_frac_gt_0p99", "n_twist_frac_gt_0p9",
                                                                 "other_frac_max", "normal_frac_max", "twist_frac_agg") if kk in r}
                                       for k, r in v["null_direction_tests"].items()},
                 baselines=v.get("baselines"), off_training_energy_cost=v.get("off_training_energy_cost"),
                 twist_profile_ranks={k: dict(n_dirs=pp["n_dirs"], rank_above=pp["rank_above"], top_sv_over_sC=pp["sv_over_sC"][:12])
                                      for k, pp in v["twist_profiles"].items()})
        if "energy_vs_coord" in v:
            d["energy_vs_coord"] = {k: {kk: c[kk] for kk in ("k_E", "k_C", "excess", "cos_principal_gt_0p999", "cos_principal_gt_0p9",
                                                             "n_coord_active_in_E_null", "E_sv_in_C_null_max", "n_E_active_in_C_null",
                                                             "top_patterns_twist_frac_median_over_active")}
                                    | dict(top8_patterns=[dict(dx_over_sC=c["coord_sv_in_E_null_top"][j], energy_cost_rel=c["top_patterns_energy_cost_rel"][j],
                                                               twist=c["top_patterns_twist_frac"][j], other=c["top_patterns_other_frac"][j],
                                                               normal=c["top_patterns_normal_frac"][j]) for j in range(min(8, len(c["top_patterns_twist_frac"])))])
                                    for k, c in v["energy_vs_coord"].items()}
        twist[p] = d
    line = None
    if ln:
        ts = np.array(ln["t_over_thetaT"])
        line = dict(eps_T=ln["eps_T"], eps_T_train=ln["eps_T_train"], t_over_thetaT=ln["t_over_thetaT"])
        for k, v in ln.items():
            if not isinstance(v, dict):
                continue
            e = np.array(v["eps"]); ok = e > 1e3 * ln["eps_T"]
            line[k] = dict(sigma_E_rel=v["sigma_E_rel"], twist_frac=v["twist_frac"], other_frac=v["other_frac"], normal_frac=v["normal_frac"],
                           dx_over_sC=v["dx_over_sC"], slope_first6_above_floor=v["slope_first6_above_floor"],
                           local_slopes=v["local_slopes"], eps=v["eps"], eps_train=v["eps_train"],
                           coord_err=v.get("coord_err"), coord_slope_first6=v.get("coord_slope_first6"),
                           t_at_eps_2em5=float(np.interp(np.log10(2e-5), np.log10(e[ok]), np.log10(ts[ok]))) if ok.sum() > 2 else None)
            if line[k]["t_at_eps_2em5"] is not None:
                line[k]["t_at_eps_2em5"] = 10 ** line[k]["t_at_eps_2em5"]
            if ok.sum() > 2 and e[ok][0] > 2e-5:  # already above 2e-5 at the smallest step: extrapolate with the first local slope
                sl0 = v["local_slopes"][0]
                line[k]["t_at_eps_2em5"] = float(ts[ok][0] * (2e-5 / e[ok][0]) ** (1 / sl0))
                line[k]["t_at_eps_2em5_extrapolated"] = True
    out = dict(spectra=spectra, units=units, twist=twist, line=line,
               notes=dict(units="sigma/sigma_max relative to each Jacobian's own largest singular value; |dx|/sC = RMS normalised coordinate "
                                "change per unit parameter step over sigma_max of J_C; eps = landscape-set mean |dE/E|; t in units of |theta_T|",
                          files=dict(floor=f"{RES}/sloppy_floor.json", units=f"{RES}/sloppy_units.json",
                                     twist={p: f"{RES}/sloppy_twist_{p}.json" for p in tw}, line=f"{RES}/sloppy_line.json")))
    jdump(out, f"{RES}/sloppy_summary.json")
    print(json.dumps(dict(spectra={k: {kk: v[kk] for kk in ("counts_below", "gesvd_counts_below", "floor_rel", "n_resolved")} for k, v in spectra.items()},
                          units={p: u["sum_layer_fun_jac_counts_below"] for p, u in units.items()},
                          line={k: (v["slope_first6_above_floor"], v.get("t_at_eps_2em5")) for k, v in (line or {}).items() if isinstance(v, dict)}), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "twist":
        cmd_twist(sys.argv[2])
    else:
        {"floor": cmd_floor, "units": cmd_units, "line": cmd_line, "summary": cmd_summary}[cmd]()
