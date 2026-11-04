"""Shared tools for pictures of the energy-loss landscape on the 16-layer MW-teacher test bed (is the exact solution a needle?).

Every 16 x 64 student of the teacher T (teachers/mw_fixed_std_s1.pt) is a point in the same 3104-dimensional space as T:
16 shear layers of [gate, W1 (64), b1 (64), W2 (64), b2] (194 each, NAMES order, the same for m.flow and t.T). Points
are compared after removing exact parameter symmetries (align), per layer: the gate scale (gate, W2, b2) ->
(a gate, W2/a, b2/a), the odd symmetry of tanh per hidden unit (w1, b1, w2) -> -(w1, b1, w2), and the order of the 64
hidden units (Hungarian matching to the teacher's units, cost the squared difference of (w1, b1, w2) minimised over the
sign). gauge="gate" takes a = gate_T/gate_S; gauge="min" picks a with the matching to minimise |S - T| (the gate gauge
moves factors up to ~150 into W2 and inflates the distances, so planes should use "min"). shifts=True also removes the
14 translation gauges (a shift c of q after q-shear k is absorbed by b1 -= W1 c in p-shear k+1 and undone by q-shear
k+2; likewise for p), exact as well but not per layer. aligned.npz holds every student in the three gauges "aligned"
(gate), "aligned_min" and "aligned_min_shift", with the raw vectors (the scratchpad job dirs in JOBS may not survive).

The landscape height is eps(v) = mean over tori and angles of |H_T(S_v(theta, J)) - <H_T>| / |<H_T>| on a fixed set
(LAND_TORI held-out-style tori x LAND_ANGLES angles, as in gauge_lab.twist); eps_batch(V) evaluates a stack of
parameter vectors by vmap over the student module (~0.12 s per vector with 2 threads). Its float64 floor is eps(T) =
5.8e-16 (4.6e-16 by residual_host: at T the value is rounding noise and depends on the evaluation path).

    python landscape_lab.py align   aligned vectors of every student, their eps and distances -> RES/aligned.{npz,json}
    python landscape_lab.py check   eps_batch against residual_host and Ev.quick; eps(T); timing -> RES/check.json
    python landscape_lab.py svd [NAMES]  energy Jacobians at T and A, coordinate Jacobian at T, SVDs -> RES/svd_*.npz
API: load_all(), to_vec(obj), set_vec(m, v), job_vec(dir), blocks(v), align(v, ref, gauge, shifts), random_symmetry(v),
eps_batch(V), eps_per_torus(V), plane(O, D1, D2, s1, s2), aligned(), svd(name), rdist(v, w), energy_jac(v), coord_jac(v).
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NTHREADS", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402
import gauge_lab as GL  # noqa: E402

torch.set_num_threads(int(os.environ["NTHREADS"]))
F64 = torch.float64

REPO = "/Users/gabrielpfaffman/Repos/orbitflows"
RES = os.environ.get("LANDSCAPE_RES", f"{REPO}/performance/precision/results/landscape")  # cluster: env
FIGS = f"{REPO}/performance/audit/figs/landscape"
ORIG = os.environ.get("LANDSCAPE_ORIG", f"{REPO}/performance/precision/results/node_final_1019/mwteacher_v2")
SCR = "/private/tmp/claude-501/-Users-gabrielpfaffman-Repos-orbitflows/83657895-663a-4aa9-9083-bad2d0d05aa6/scratchpad"
BASE = f"{ORIG}/mwt_std_s1"  # training set, evaluator and student module (all six originals share data/model/eval)
JOBS = {**{f"{k}_s{s}": f"{ORIG}/mwt_{k}_s{s}" for k in ("geo", "std") for s in range(3)},
        "rsr_geo_s0": f"{SCR}/gauge_res/final/rsr_geo_s0", "rsr_geo_s1": f"{SCR}/gauge_res/final/rsr_geo_s1",
        "rsr_geo_s0_start": f"{SCR}/gauge_starts/gauge_v2/rsr_geo_s0", "rsr_geo_s1_start": f"{SCR}/gauge_starts/gauge_v2/rsr_geo_s1",
        "untw_end": f"{SCR}/untwist_res/untw_end",
        "rand4_start": f"{SCR}/gauge_starts/gauge_v2/teacher_rand4", "rand4_final": f"{SCR}/gauge_res/final/teacher_rand4",
        "sloppy_start": f"{SCR}/gauge_starts/gauge_v2/teacher_sloppy", "sloppy_final": f"{SCR}/gauge_res/final/teacher_sloppy"}
ORIGINALS = [n for n in JOBS if n.startswith(("geo", "std"))]
ALIGNED = f"{RES}/aligned.npz"
LAND_TORI, LAND_ANGLES = 100, 128

N_LAYERS, WIDTH = 16, 64
NAMES = [f"layers.{k}.conditioner.{p}" for k in range(N_LAYERS)
         for p in ("gate", "layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias")]
PER = 3 * WIDTH + 2  # gate, W1, b1, W2, b2 = 194
P = N_LAYERS * PER   # 3104
# offsets inside a layer's block
G, W1, B1, W2, B2 = 0, slice(1, 1 + WIDTH), slice(1 + WIDTH, 1 + 2 * WIDTH), slice(1 + 2 * WIDTH, 1 + 3 * WIDTH), 3 * WIDTH + 1

_L = None


# ===================================================================== parameter vectors
def to_vec(obj):
    """Flat float64 (P,) vector of a flow's parameters in NAMES order: from an AAMap / Flow module, or a state dict
    (student "flow.*" keys or teacher keys)."""
    if isinstance(obj, torch.nn.Module):
        d = dict(obj.named_parameters())
        d = {k[len("flow."):] if k.startswith("flow.") else k: v for k, v in d.items()}
    else:
        d = {k[len("flow."):]: v for k, v in obj.items() if k.startswith("flow.")} or dict(obj)
    v = torch.cat([d[n].detach().reshape(-1).double() for n in NAMES])
    assert v.numel() == P
    return v


def set_vec(m, v):
    """Load a (P,) vector into the student module m (an AAMap; its parameters are exactly the flow's, in NAMES order)."""
    GL.set_flat(m, torch.as_tensor(v, dtype=F64))


def job_vec(jd):
    return to_vec(torch.load(os.path.join(jd, "state.pt"), weights_only=False)["state"])


def rdist(v, w=None):
    """|v - w| / |theta_T| (w defaults to the teacher)."""
    L = load_all()
    return float((torch.as_tensor(v) - (L.vT if w is None else torch.as_tensor(w))).norm() / L.nT)


def blocks(v):
    """(N_LAYERS, PER) view of a flat vector."""
    return v.view(N_LAYERS, PER)


# ==================================================================== setup and caches
def load_all():
    """Cached setup: L.cfg, L.t (teacher target), L.aa / L.nO / L.nS / L.es (training set 96 x 128, as LM), L.Ev,
    L.m (student module, float64 CPU; its parameters get overwritten by set_vec), L.vT and L.nT = |theta_T|,
    L.J_land / L.aa_land / L.x_land (landscape set), and, once aligned.npz exists, L.names, L.raw, L.al, L.al_min,
    L.al_min_shift (dicts name -> (P,) vector, one per gauge), L.eps_land, L.A / L.B (main-plane students)."""
    global _L
    if _L is not None:
        return _L
    cfg, t, aa, nO, nS, es, Ev, m = GL.load(BASE)
    sn = [n for n, _ in m.named_parameters()]
    assert [n[len("flow."):] for n in sn] == NAMES and all(n.startswith("flow.") for n in sn), sn[:6]
    assert [n for n, _ in t.T.named_parameters()] == NAMES
    assert all(p.shape == q.shape for p, q in zip(m.parameters(), t.T.parameters()))
    assert math.isclose(float(m.toy.ln_w0.exp()), t.omega, rel_tol=1e-15), (float(m.toy.ln_w0.exp()), t.omega)
    vT = to_vec(t.T)
    J_lo, J_hi = Ev.J_range
    J_land = C.j_grid(LAND_TORI, J_lo, J_hi, t.nu0, "asinh", 0.03, offset=0.5)
    aa_land = C.aa_points(J_land, C.angles(LAND_ANGLES, 0.25))
    _L = SimpleNamespace(cfg=cfg, t=t, aa=aa, nO=nO, nS=nS, es=es, Ev=Ev, m=m, vT=vT, nT=float(vT.norm()),
                         J_land=J_land, aa_land=aa_land, x_land=aa_land.reshape(-1, 2), names=None, raw=None, al=None)
    if os.path.exists(ALIGNED):
        try:
            _L.__dict__.update(aligned())
        except KeyError as e:  # an older cache (rebuild it with: python landscape_lab.py align)
            print(f"landscape_lab: {ALIGNED} is incomplete ({e}); ignoring it", flush=True)
    return _L


def aligned():
    """Contents of RES/aligned.npz: names; raw, al (gauge "gate"), al_min, al_min_shift: dicts name -> torch (P,);
    eps_land (name -> eps of the raw vector; every gauge has the same map); A, B."""
    z = np.load(ALIGNED, allow_pickle=False)
    names = [str(s) for s in z["names"]]
    mk = lambda key: {n: torch.from_numpy(z[key][i].copy()) for i, n in enumerate(names)}
    return dict(names=names, raw=mk("raw"), al=mk("aligned"), al_min=mk("aligned_min"), al_min_shift=mk("aligned_min_shift"),
                eps_land={n: float(e) for n, e in zip(names, z["eps_land"])}, A=str(z["A"]), B=str(z["B"]))


def svd(name):
    """Cached SVD npz (name in 'T_energy', 'T_coord', 'A_energy' (A in gauge "gate"), 'A_energy_min_shift'): S (P,),
    Vh (P, P) (rows = right singular vectors, in NAMES order), Ur = U^T r (P,) (the residual on the left singular
    vectors; the Gauss-Newton step is -Vh^T (Ur / S)), r_norm, shape, nT, at, gauge, nO, nS."""
    z = np.load(f"{RES}/svd_{name}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


# ====================================================================== alignment
def _match(Us, Ut):
    """Hungarian matching of student units Us (64, 3) to teacher units Ut, cost min over sign of |s u - t|^2.
    Returns perm, sign with new[j] = sign[j] * Us[perm[j]]."""
    from scipy.optimize import linear_sum_assignment
    dot = Us @ Ut.T
    cost = (Us * Us).sum(1)[:, None] + (Ut * Ut).sum(1)[None, :] - 2 * dot.abs()
    r, c = linear_sum_assignment(cost.numpy())
    perm = torch.empty(WIDTH, dtype=torch.long); perm[torch.from_numpy(c)] = torch.from_numpy(r)
    sgn = torch.sign(dot[perm, torch.arange(WIDTH)]); sgn[sgn == 0] = 1.0
    return perm, sgn


def _shift_basis(v):
    """(P, 14) generators of the translation gauges at v: column k (k = 0..13) shifts the shear variable of layer k
    by +1 (b2_k += 1/gate_k), compensates the next layer (b1_{k+1} -= W1_{k+1}) and undoes it at layer k+2
    (b2_{k+2} -= 1/gate_{k+2}). Exact for any coefficient (the map is unchanged), independent of b1, b2."""
    Bv = blocks(v); Z = torch.zeros(N_LAYERS - 2, N_LAYERS, PER, dtype=F64)
    for k in range(N_LAYERS - 2):
        Z[k, k, B2] = 1 / Bv[k, G]
        Z[k, k + 1, B1] = -Bv[k + 1, W1]
        Z[k, k + 2, B2] = -1 / Bv[k + 2, G]
    return Z.view(N_LAYERS - 2, P).T


def _best_scale(g, u, gT, uT):
    """The scale a minimising (a g - gT)^2 + |u / a - uT|^2 (u = (W2, b2)): a real root of
    g^2 a^4 - g gT a^3 + (u.uT) a - |u|^2 = 0 (there is always one of each sign)."""
    f = lambda a: (a * g - gT) ** 2 + float(((u / a - uT) ** 2).sum())
    rts = np.roots([g * g, -g * gT, 0.0, float(u @ uT), -float(u @ u)])
    return min((float(r.real) for r in rts if abs(r.imag) <= 1e-8 * abs(r) and r.real != 0), key=f)


def _align_layer(b, r, gauge, max_it=50):
    """One layer's block b (PER,) in the gauge of the reference block r. gauge="gate": a = gate_r / gate_b, then the
    matching; gauge="min": a, the signs and the order alternately minimise |b' - r|^2 (from a = gate_r / gate_b and from
    a = 1, keeping the better). Returns (new block, a, perm, sign)."""
    g, w1, b1, w2, b2 = b[G], b[W1], b[B1], b[W2], b[B2]
    Ut = torch.stack([r[W1], r[B1], r[W2]], 1); uT = torch.cat([r[W2], r[B2:B2 + 1]])
    best = None
    for a in ([float(r[G] / g)] if gauge == "gate" else [float(r[G] / g), 1.0]):
        prev = None
        for _ in range(1 if gauge == "gate" else max_it):
            perm, sgn = _match(torch.stack([w1, b1, w2 / a], 1), Ut)
            if gauge == "gate":
                break
            a_new = _best_scale(float(g), torch.cat([sgn * w2[perm], b2.view(1)]), float(r[G]), uT)
            done = prev is not None and torch.equal(perm, prev[0]) and torch.equal(sgn, prev[1]) and abs(a_new - a) <= 1e-13 * abs(a)
            prev, a = (perm, sgn), (a if done else a_new)
            if done:
                break
        new = b.clone()
        new[G] = r[G] if gauge == "gate" else a * g
        new[W1], new[B1], new[W2], new[B2] = sgn * w1[perm], sgn * b1[perm], sgn * w2[perm] / a, b2 / a
        D = float((new - r).pow(2).sum())
        if best is None or D < best[0]:
            best = (D, new, a, perm, sgn)
    return best[1:]


def align(v, ref=None, gauge="gate", shifts=False, iters=10, info=False):
    """Student vector v (P,) in the gauge of ref (default theta_T), removing exact symmetries only. Per layer (layers are
    not permutable): the gate scale (gate, W2, b2) -> (a gate, W2/a, b2/a), the tanh sign of each hidden unit and the
    order of the 64 units (Hungarian matching). gauge="gate": a = gate_ref/gate_v (the gates become the teacher's);
    gauge="min": a chosen with the matching to minimise the distance (the distance on the quotient by these symmetries,
    up to the alternating search). shifts=True: then the 14 translation gauges by least squares, alternating with the
    per-layer step until the distance stops falling (at most iters rounds). info=True also returns stats of round 0."""
    ref = load_all().vT if ref is None else torch.as_tensor(ref, dtype=F64)
    Rb = blocks(ref)
    Vb = blocks(torch.as_tensor(v, dtype=F64).clone())
    st = dict(a=[0.0] * N_LAYERS, units_moved=[0] * N_LAYERS, units_sign_flipped=[0] * N_LAYERS)
    D_prev, n_rounds = float("inf"), 0
    for rnd in range(iters if shifts else 1):
        for k in range(N_LAYERS):
            Vb[k], a, perm, sgn = _align_layer(Vb[k], Rb[k], gauge)
            if rnd == 0:
                st["a"][k], st["units_moved"][k], st["units_sign_flipped"][k] = a, int((perm != torch.arange(WIDTH)).sum()), int((sgn < 0).sum())
        n_rounds = rnd + 1
        if not shifts:
            break
        Bm = _shift_basis(Vb.reshape(-1))
        c = torch.linalg.lstsq(Bm, (ref - Vb.reshape(-1))[:, None]).solution[:, 0]
        Vb = blocks(Vb.reshape(-1) + Bm @ c).clone()
        D = float((Vb.reshape(-1) - ref).norm())
        if D > (1 - 1e-10) * D_prev:
            break
        D_prev = D
    out = Vb.reshape(-1).clone()
    if not info:
        return out
    aa = torch.tensor(st["a"], dtype=F64)
    return out, dict(st, a_abs_min=float(aa.abs().min()), a_abs_max=float(aa.abs().max()), gate_sign_flips=int((aa < 0).sum()),
                     rounds=n_rounds)


def random_symmetry(v, seed=0):
    """v under a random exact symmetry with no rescaling (a random permutation and random signs of the hidden units of
    every layer): the same map up to summation order, i.e. the float64 noise floor of map comparisons."""
    g = torch.Generator().manual_seed(seed)
    Vb = blocks(torch.as_tensor(v, dtype=F64).clone())
    for k in range(N_LAYERS):
        perm = torch.randperm(WIDTH, generator=g); sgn = torch.randint(0, 2, (WIDTH,), generator=g).to(F64) * 2 - 1
        Vb[k, W1], Vb[k, B1], Vb[k, W2] = sgn * Vb[k, W1][perm], sgn * Vb[k, B1][perm], sgn * Vb[k, W2][perm]
    return Vb.reshape(-1).clone()


# ================================================================== batched evaluator
def _forward(V, x):
    """Images (K, n, 2) of the points x (n, 2) [theta, J] under the K students V (K, P), by vmap of the module."""
    from torch.func import functional_call, vmap
    L = load_all()
    sh = [p.shape for p in L.m.parameters()]; sz = [p.numel() for p in L.m.parameters()]
    params, i = {}, 0
    for n, s_, k in zip(NAMES, sh, sz):
        params["flow." + n] = V[:, i:i + k].reshape(V.shape[0], *s_); i += k
    return vmap(lambda p: functional_call(L.m, p, (x,)))(params)


@torch.no_grad()
def eps_per_torus(V, chunk=32, x=None, nO=None, nS=None):
    """(K, nO) per-torus mean |H_T - <H_T>| / |<H_T>| on the landscape set (or on x (nO nS, 2), nO tori x nS angles)
    for parameter vectors V (K, P) or (P,). Chunks of `chunk` vectors (~13 MB of activations per vector)."""
    L = load_all()
    V = torch.as_tensor(V, dtype=F64)
    V = V[None] if V.dim() == 1 else V
    x = L.x_land if x is None else x
    nO, nS = (LAND_TORI, LAND_ANGLES) if nO is None else (nO, nS)
    out = []
    for k0 in range(0, V.shape[0], chunk):
        y = _forward(V[k0:k0 + chunk], x)
        H = L.t.H(y.reshape(-1, 2)).view(-1, nO, nS)
        mH = H.mean(-1, keepdim=True)
        out.append(((H - mH) / mH.abs()).abs().mean(-1))
    return torch.cat(out)


def plane(O, D1, D2, s1, s2):
    """Grid vectors O + a D1 + b D2 for a in s1 (slow index), b in s2: (len(s1) len(s2), P)."""
    a, b = torch.meshgrid(torch.as_tensor(s1, dtype=F64), torch.as_tensor(s2, dtype=F64), indexing="ij")
    return O + a.reshape(-1, 1) * D1 + b.reshape(-1, 1) * D2


def eps_batch(V, chunk=32, **kw):
    """(K,) mean |dE/E| on the landscape set for V (K, P) (or a scalar tensor for a (P,) vector)."""
    e = eps_per_torus(V, chunk, **kw).mean(-1)
    return e[0] if torch.as_tensor(V).dim() == 1 else e


# ================================================================== Jacobians
def energy_jac(v):
    """Training energy-residual Jacobian (nO nS, P) at v, exactly as LM builds it (gauge_lab.energy_jacobian), with the
    residual vector r (nO nS,) at v in the same normalisation ((H - <H>) / (nu0 J) / sqrt(N))."""
    L = load_all()
    set_vec(L.m, v)
    with torch.no_grad():
        r = (C.residual_host(L.m, L.aa, L.t, L.nO, L.nS, L.es) / math.sqrt(L.nO * L.nS)).reshape(-1)
    return GL.energy_jacobian(L.m, L.aa, L.t, L.nO, L.nS, L.es), r


def coord_jac(v, y_ref=None, chunk=16):
    """Coordinate-residual Jacobian (nO nS 2, P) at v, rows (point, q/p) as train_lm loss="coord": residual
    (y - y_ref) / s / sqrt(N), s = coord_scale(es, nu0) per torus; y_ref defaults to the teacher's own map.
    Returns (J, r)."""
    from torch.func import functional_call, jacrev, vmap
    L = load_all(); m = L.m; set_vec(m, v)
    N = L.nO * L.nS; sN = math.sqrt(N)
    X = L.aa.reshape(-1, 2)
    with torch.no_grad():
        y_ref = L.t.aa_map(X) if y_ref is None else y_ref
        s_pt = C.coord_scale(L.es, float(L.t.nu0)).repeat_interleave(L.nS, 0)
        r = ((m(X) - y_ref) / s_pt / sN).reshape(-1)
    params = {n: p.detach() for n, p in m.named_parameters()}
    f_point = lambda p, x: functional_call(m, p, (x[None, :],))[0]
    rows = []
    for o0 in range(0, L.nO, chunk):
        x = L.aa[o0:o0 + chunk].reshape(-1, 2)
        Jd = vmap(jacrev(f_point), in_dims=(None, 0))(params, x)
        Jy = torch.cat([Jd["flow." + n].reshape(x.shape[0], 2, -1) for n in NAMES], -1)
        sl = slice(o0 * L.nS, o0 * L.nS + x.shape[0])
        rows.append((Jy / s_pt[sl][..., None] / sN).reshape(-1, P))
    return torch.cat(rows), r


# ===================================================================== scripts
def _heldout(v):
    """Ev.quick eval_mean (held-out 200 x 512) of vector v."""
    L = load_all(); set_vec(L.m, v)
    return L.Ev.quick(L.m)["eval_mean"]


def _map_dev(v, w, x):
    """(max over points of |S_v(x) - S_w(x)| / |S_w(x)|, max |S_v - S_w| / max |S_w|)."""
    L = load_all()
    with torch.no_grad():
        set_vec(L.m, v); y1 = L.m(x)
        set_vec(L.m, w); y2 = L.m(x)
    d = (y1 - y2).norm(dim=-1)
    return float((d / y2.norm(dim=-1)).max()), float(d.max() / y2.norm(dim=-1).max())


GAUGES = {"aligned": dict(gauge="gate"), "aligned_min": dict(gauge="min"), "aligned_min_shift": dict(gauge="min", shifts=True)}


def cmd_align():
    L = load_all()
    t0 = time.perf_counter()
    names = list(JOBS)
    raw = {n: job_vec(JOBS[n]) for n in names}
    X_ho = C.aa_points(*_ho_grid()).reshape(-1, 2)  # 200 x 512 held-out points
    rows, vecs = {}, {g: {} for g in GAUGES}
    for n in names:
        rows[n] = dict(dir=JOBS[n], dist_raw=rdist(raw[n]), norm_over_T=float(raw[n].norm()) / L.nT,
                       map_rel_dev_baseline=dict(land=_map_dev(random_symmetry(raw[n]), raw[n], L.x_land),
                                                 heldout=_map_dev(random_symmetry(raw[n]), raw[n], X_ho)))
        for gname, kw in GAUGES.items():
            a, inf = align(raw[n], info=True, **kw)
            vecs[gname][n] = a
            rows[n][gname] = dict(dist=rdist(a), cos_T=float(a @ L.vT / a.norm() / L.nT), norm_over_T=float(a.norm()) / L.nT,
                                  map_rel_dev=dict(land=_map_dev(a, raw[n], L.x_land), heldout=_map_dev(a, raw[n], X_ho)), **inf)
        r = rows[n]
        print(f"{n:17s} |S-T|/|T| raw {r['dist_raw']:.4f}  gate {r['aligned']['dist']:.4f}  min {r['aligned_min']['dist']:.4f}  "
              f"min+shift {r['aligned_min_shift']['dist']:.4f};  map dev (point, global) gate {max(r['aligned']['map_rel_dev']['land'][0], r['aligned']['map_rel_dev']['heldout'][0]):.1e} "
              f"min {max(r['aligned_min']['map_rel_dev']['land'][0], r['aligned_min']['map_rel_dev']['heldout'][0]):.1e} "
              f"min+shift {max(r['aligned_min_shift']['map_rel_dev']['land'][0], r['aligned_min_shift']['map_rel_dev']['heldout'][0]):.1e} "
              f"baseline {max(r['map_rel_dev_baseline']['land'][0], r['map_rel_dev_baseline']['heldout'][0]):.1e}; "
              f"units moved {sum(r['aligned']['units_moved'])}, flipped {sum(r['aligned']['units_sign_flipped'])}", flush=True)
    V = torch.stack([raw[n] for n in names])
    e_raw = eps_batch(V)
    E = {g: eps_batch(torch.stack([vecs[g][n] for n in names])) for g in GAUGES}
    eT = float(eps_batch(L.vT)); hoT = _heldout(L.vT)
    for i, n in enumerate(names):
        ho = _heldout(raw[n])
        rows[n].update(eps_land=float(e_raw[i]), heldout_eval_mean=ho, ratio_heldout_over_land=ho / float(e_raw[i]))
        for g in GAUGES:
            rows[n][g].update(eps_land=float(E[g][i]), eps_rel_change=float((E[g][i] - e_raw[i]).abs() / e_raw[i]))
        print(f"{n:17s} eps_land {float(e_raw[i]):.4e} (rel change aligned: " + ", ".join(f"{g} {rows[n][g]['eps_rel_change']:.1e}" for g in GAUGES)
              + f"); Ev.quick {ho:.4e}; ratio {ho / float(e_raw[i]):.4f}", flush=True)
    seeds = {n: json.load(open(os.path.join(JOBS[n], "config.json")))["seed"] for n in ORIGINALS}
    ranked = sorted(ORIGINALS, key=lambda n: rows[n]["heldout_eval_mean"])
    A = ranked[0]; B = next(n for n in ranked[1:] if seeds[n] != seeds[A])
    os.makedirs(RES, exist_ok=True)
    np.savez(ALIGNED, names=np.array(names), raw=V.numpy(), vT=L.vT.numpy(), nT=L.nT, eps_land=e_raw.numpy(), eps_T=eT, A=A, B=B,
             **{g: torch.stack([vecs[g][n] for n in names]).numpy() for g in GAUGES},
             dist_raw=np.array([rows[n]["dist_raw"] for n in names]),
             **{f"dist_{g}": np.array([rows[n][g]["dist"] for n in names]) for g in GAUGES})
    json.dump(dict(theta_T_norm=L.nT, eps_land_T=eT, heldout_T=hoT, ratio_heldout_over_land_T=hoT / eT, A=A, B=B,
                   seeds=seeds, ranked_originals=ranked, dist_A_B={g: rdist(vecs[g][A], vecs[g][B]) for g in GAUGES},
                   dist_A_B_raw=rdist(raw[A], raw[B]),
                   landscape_set=dict(tori=LAND_TORI, angles=LAND_ANGLES, j_grid="asinh a0=0.03 offset=0.5", angle_offset=0.25,
                                      J_range=list(L.Ev.J_range)),
                   gauges={g: str(kw) for g, kw in GAUGES.items()}, rows=rows, seconds=time.perf_counter() - t0),
              open(f"{RES}/aligned.json", "w"), indent=1)
    print(f"T: eps_land {eT:.3e}, Ev.quick {hoT:.3e}; |theta_T| {L.nT:.4f}; A = {A}, B = {B}; |A-B|/|T| raw {rdist(raw[A], raw[B]):.4f}, "
          + ", ".join(f"{g} {rdist(vecs[g][A], vecs[g][B]):.4f}" for g in GAUGES) + f"; {time.perf_counter() - t0:.0f} s")


def _ho_grid():
    """The Evaluator's held-out grid (200 tori, 512 angles at offset 0.25), rebuilt from the config."""
    L = load_all(); e = L.cfg.get("eval", {}); d = L.cfg["data"]
    J = C.j_grid(e.get("n_orbits", 200), *L.Ev.J_range, L.t.nu0, d.get("spacing", "asinh"), d.get("a0", 0.03), offset=0.5)
    return J, C.angles(e.get("n_steps", 512), 0.25)


def cmd_check():
    L = load_all()
    A, B, As, Bs = L.al[L.A], L.al[L.B], L.al_min_shift[L.A], L.al_min_shift[L.B]
    g = torch.Generator().manual_seed(0)
    u = torch.rand(2, generator=g, dtype=F64) * 2 - 0.5
    pts = dict(T=L.vT, A=A, B=B, A_min_shift=As, mid_TA=0.5 * (L.vT + A), mid_TA_min_shift=0.5 * (L.vT + As),
               rand_TAB_min_shift=L.vT + u[0] * (As - L.vT) + u[1] * (Bs - L.vT), rand4_final=L.al["rand4_final"],
               sloppy_start=L.al["sloppy_start"])
    out = dict(rand_TAB_coeffs=u.tolist())
    for n, v in pts.items():
        e_b = float(eps_batch(v))
        set_vec(L.m, v)
        with torch.no_grad():
            e_d = float(C.residual_host(L.m, L.aa_land, L.t, LAND_TORI, LAND_ANGLES).abs().mean())
        ho = L.Ev.quick(L.m)["eval_mean"]
        out[n] = dict(eps_batch=e_b, eps_direct=e_d, rel_diff=abs(e_b - e_d) / e_d, heldout=ho, ratio_heldout_over_land=ho / e_b)
        print(f"{n:13s} eps_batch {e_b:.6e}  residual_host {e_d:.6e}  rel diff {abs(e_b - e_d) / e_d:.1e}  Ev.quick {ho:.4e}  "
              f"ratio {ho / e_b:.4f}", flush=True)
    # a batch of mixed points evaluated together must agree with the single evaluations
    Vm = torch.stack(list(pts.values()))
    eb = eps_batch(Vm)
    out["batch_vs_single_max_rel"] = max(abs(float(eb[i]) - out[n]["eps_batch"]) / out[n]["eps_batch"] for i, n in enumerate(pts))
    # timing at K = 64 (plane points around T, A, B)
    c = torch.rand(64, 2, generator=g, dtype=F64)
    V = L.vT + c[:, :1] * (As - L.vT) + c[:, 1:] * (Bs - L.vT)
    for ch in (16, 32, 64):
        t0 = time.perf_counter(); eps_batch(V, chunk=ch); dt = time.perf_counter() - t0
        out[f"time_K64_chunk{ch}_s"] = dt
        print(f"K = 64, chunk {ch}: {dt:.2f} s ({dt / 64 * 1000:.0f} ms per vector)", flush=True)
    t0 = time.perf_counter(); ept = eps_per_torus(L.vT); out["time_single_s"] = time.perf_counter() - t0
    out["eps_T_per_torus_min_max"] = [float(ept.min()), float(ept.max())]
    out["eps_T_per_torus_first5_last5"] = [float(x) for x in torch.cat([ept[0, :5], ept[0, -5:]])]
    print(f"eps(T) {out['T']['eps_batch']:.3e}; per torus {float(ept.min()):.1e} .. {float(ept.max()):.1e}; "
          f"batch vs single max rel {out['batch_vs_single_max_rel']:.1e}")
    json.dump(out, open(f"{RES}/check.json", "w"), indent=1)


def cmd_svd():
    L = load_all()
    jobs = [("T_energy", "teacher", lambda: energy_jac(L.vT)), ("A_energy", "aligned", lambda: energy_jac(L.al[L.A])),
            ("A_energy_min_shift", "aligned_min_shift", lambda: energy_jac(L.al_min_shift[L.A])),
            ("T_coord", "teacher", lambda: coord_jac(L.vT))]
    if len(sys.argv) > 2:
        jobs = [j for j in jobs if j[0] in sys.argv[2].split(",")]
    for name, gauge, f in jobs:
        t0 = time.perf_counter()
        Jm, r = f(); t1 = time.perf_counter()
        U, S, Vh = torch.linalg.svd(Jm, full_matrices=False); t2 = time.perf_counter()
        Ur = U.T @ r
        rel = S / S[0]
        cnt = {f"{c:g}": int((rel < c).sum()) for c in (1e-4, 1e-8, 1e-10, 1e-12, 1e-14, 1e-15)}
        np.savez(f"{RES}/svd_{name}.npz", S=S.numpy(), Vh=Vh.numpy(), Ur=Ur.numpy(), r_norm=float(r.norm()), shape=np.array(Jm.shape),
                 nT=L.nT, at=L.A if name.startswith("A") else "teacher", gauge=gauge, nO=L.nO, nS=L.nS, sec_jac=t1 - t0, sec_svd=t2 - t1)
        print(f"{name}: J {tuple(Jm.shape)} in {t1 - t0:.0f} s, SVD {t2 - t1:.0f} s; sigma_max {float(S[0]):.3e}, "
              f"sigma_min/sigma_max {float(rel[-1]):.1e}, cond {float(S[0] / S[-1]):.1e}; |r| {float(r.norm()):.2e}; "
              f"counts below: {cnt}", flush=True)
        del Jm, U, S, Vh


if __name__ == "__main__":
    {"align": cmd_align, "check": cmd_check, "svd": cmd_svd}[sys.argv[1]]()
