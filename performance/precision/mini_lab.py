"""Mini teacher-student test bed: small fixed-toy MW models as teachers, same-size students trained from scratch.

Purpose: iterate quickly on optimisation questions that are slow at 16 x 64 (6-28 h per student): at which sizes do
same-size students of an exactly representable teacher reach it (held-out < 1e-10), at which do they stall, how do the
two kinds of run train, and is a stalled student in an isolated pit or in a curved canyon that leads to the teacher.

Data, toy and recipe are the test bed's (make_jobs.py mwteacher): MW range 5 pc - 8.3 kpc, 96 tori x 128 angles
(asinh a0 0.03), fixed toy nu0 (the teacher's own frequency), float64; Adam 3000 (lr 1e-3 cosine) then geodesic LM
(lam0 1e-3, stall rule 1000 steps / 1e-5, retry_lam 1e-11) with an iteration cap.

    python mini_lab.py cost L W [L W ...]             time one Adam step, one LM step and one held-out evaluation
    python mini_lab.py teacher L W [--seed S] [--cap N] train an MW model L x W and freeze it as teachers/mini_LxW.pt
    python mini_lab.py student L W SEED [--cap N] [--wall S]   a same-size student of teachers/mini_LxW.pt
    python mini_lab.py analyse [--sizes 4x8,...]       success rates, alignment over training, twist, figure, summary
    python mini_lab.py bezier L W SEED [--k 12] [--iters 300]  curved-path test student -> teacher
Every run writes RES/<kind>/<LxW>/s<seed>/{config.json, state.pt, curve.npz, result.json}. curve.npz holds the learning
curve (phase 0 Adam / 1 LM, step, training loss, lr or lam, held-out, train_mean, wall) and the parameter vector at every
record point (every 100 Adam steps and every 25 LM steps).
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

F64 = torch.float64
RES = os.environ.get("MINI_RES", "/Users/gabrielpfaffman/Repos/orbitflows/performance/precision/results/mini")  # cluster: $MINI_RES
MW_DATA = dict(zmin_pc=5.0, zmax_kpc=8.32, n_orbits=96, n_steps=128, spacing="asinh", a0=0.03)
SUCCESS = 1e-10


def teacher_file(L, W):
    return f"teachers/mini_{L}x{W}.pt"  # relative to HERE (core.Teacher resolves it there)


def make_cfg(name, batch, L, W, seed, teacher=None):
    c = dict(name=name, batch=batch, target="teacher" if teacher else "mw", seed=seed, loss_norm="toy",
             model=dict(n_layers=L, width=W, depth=2, act="tanh", bias_init="torch", gate_std=1e-3, odd=False),
             toy=dict(mode="fixed", omega="nu0"), data=dict(MW_DATA), eval=dict(n_orbits=200, n_steps=512))
    if teacher:
        c["teacher"] = dict(checkpoint=teacher)
    return c


class Bed:
    """Target, training set, evaluator and model builder of a config (as run.py builds them)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.t = C.make_target(cfg)
        d = cfg["data"]; mw = C.MWVertical()
        self.J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, mw.phi)[0])
        self.J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, mw.phi)[0])
        self.w0 = self.t.nu0
        self.J_tr = C.train_grid(d, self.J_lo, self.J_hi, self.w0)
        self.aa = C.aa_points(self.J_tr, C.angles(d["n_steps"], 0.5)); self.nO, self.nS = self.aa.shape[:2]
        self.es = (self.w0 * self.J_tr).reshape(-1, 1)
        self.ev = C.Evaluator(cfg, self.t, self.J_lo, self.J_hi, self.w0, self.J_tr, self.nS)
        self.L, self.W = cfg["model"]["n_layers"], cfg["model"]["width"]

    def build(self, seed=None):
        return C.build_model(self.cfg["model"], self.cfg["toy"], self.w0, self.J_lo, self.J_hi,
                             self.cfg.get("seed", 0) if seed is None else seed)

    @torch.no_grad()
    def heldout(self, m):
        """eval_mean of Evaluator.quick (200 x 512 held-out points), and train_mean."""
        e = C.residual_host(m, self.ev.aa, self.t, *self.ev.aa.shape[:2]).abs().mean()
        et = C.residual_host(m, self.ev.aa_train, self.t, *self.ev.aa_train.shape[:2]).abs().mean()
        return float(e), float(et)

    @torch.no_grad()
    def loss(self, m):
        """Training loss as LM reports it: mean over points of ((H - <H>) / (nu0 J))^2."""
        return float(C.residual_host(m, self.aa, self.t, self.nO, self.nS, self.es).pow(2).mean())


# ================================================================= parameter vectors
def get_flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()


def set_flat(m, v):
    with torch.no_grad():
        i = 0
        for p in m.parameters():
            n = p.numel(); p.copy_(torch.as_tensor(v, dtype=F64)[i:i + n].view_as(p)); i += n


def check_layout(m, L, W):
    """Student parameters are exactly the flow's, per layer [gate, W1 (W), b1 (W), W2 (W), b2]."""
    names = [n for n, _ in m.named_parameters()]
    want = [f"flow.layers.{k}.conditioner.{p}" for k in range(L)
            for p in ("gate", "layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias")]
    assert names == want, (names[:6], want[:6])


# ===================================================================== training
def train(cfg, jdir, adam_steps=3000, lm_cap=3000, stall=(1000, 1e-5), wall=None, geodesic=True, retry=1e-11):
    os.makedirs(jdir, exist_ok=True)
    json.dump(cfg, open(os.path.join(jdir, "config.json"), "w"), indent=1)
    bed = Bed(cfg)
    m = bed.build(); check_layout(m, bed.L, bed.W)
    rows, snaps = [], []
    t0 = time.perf_counter()

    def rec(phase, it, loss, lr):
        ho, tr = bed.heldout(m)
        rows.append((phase, it, loss, lr, ho, tr, time.perf_counter() - t0)); snaps.append(get_flat(m).numpy())
        if phase == 1 and (it % 250 == 0):
            print(f"[{cfg['name']}] LM {it:5d} loss {loss:.3e} held-out {ho:.3e} lam {lr:.1e} {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)

    C.train_adam(m, bed.aa, bed.t, bed.nO, bed.nS, adam_steps, 1e-3, lambda k, l, lr: rec(0, k, l, lr), every=100, escale=bed.es)
    t_adam = time.perf_counter() - t0
    print(f"[{cfg['name']}] Adam done: held-out {rows[-1][4]:.3e} ({t_adam:.0f} s)", flush=True)
    n, why, lam = C.train_lm(m, bed.aa, bed.t, bed.nO, bed.nS, max_iters=lm_cap, record=lambda it, l, lm: rec(1, it, l, lm),
                             every=25, wall_cap=wall, stall_window=stall[0], stall_rtol=stall[1], escale=bed.es,
                             geodesic=geodesic, retry_lam=retry)
    wall_s = time.perf_counter() - t0
    torch.save(dict(state=C.state_f64(m)), os.path.join(jdir, "state.pt"))
    q = bed.ev.quick(m)
    R = np.array(rows, dtype=np.float64)
    np.savez(os.path.join(jdir, "curve.npz"), phase=R[:, 0], it=R[:, 1], loss=R[:, 2], lr_lam=R[:, 3], heldout=R[:, 4],
             train_mean=R[:, 5], t=R[:, 6], snaps=np.stack(snaps))
    res = dict(name=cfg["name"], L=bed.L, W=bed.W, seed=cfg["seed"], n_params=C.n_params(m), lm_iters=n, why=why, lam_end=lam,
               wall_s=wall_s, adam_s=t_adam, lm_s_per_step=(wall_s - t_adam) / max(n, 1), final=q)
    json.dump(res, open(os.path.join(jdir, "result.json"), "w"), indent=1)
    print(f"DONE {cfg['name']}: {n} LM its ({why}), held-out {q['eval_mean']:.3e} max {q['eval_max']:.2e} "
          f"train {q['train_mean']:.2e}; {wall_s / 60:.1f} min", flush=True)
    return res, m, bed


def cmd_cost(a):
    out = {}
    sizes = list(zip(a.sizes[::2], a.sizes[1::2]))
    for L, W in sizes:
        bed = Bed(make_cfg("cost", "cost", L, W, 0))
        m = bed.build()
        t0 = time.perf_counter(); C.train_adam(m, bed.aa, bed.t, bed.nO, bed.nS, 200, 1e-3, lambda *x: None, every=1000, escale=bed.es)
        adam = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter(); bed.heldout(m); ho = time.perf_counter() - t0
        t0 = time.perf_counter(); bed.ev.quick(m); quick = time.perf_counter() - t0
        for geo in (False, True):
            t0 = time.perf_counter()
            n, why, _ = C.train_lm(m, bed.aa, bed.t, bed.nO, bed.nS, max_iters=20, escale=bed.es, geodesic=geo, retry_lam=1e-11)
            out[f"{L}x{W}_lm_{'geo' if geo else 'plain'}_s"] = (time.perf_counter() - t0) / n
        out[f"{L}x{W}"] = dict(P=C.n_params(m), adam_step_s=adam, heldout_s=ho, quick_s=quick,
                               lm_plain_s=out.pop(f"{L}x{W}_lm_plain_s"), lm_geo_s=out.pop(f"{L}x{W}_lm_geo_s"))
        print(L, W, out[f"{L}x{W}"], flush=True)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "step_costs.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out); json.dump(old, open(path, "w"), indent=1)


def cmd_teacher(a):
    L, W = a.L, a.W
    jdir = os.path.join(RES, "teachers", f"{L}x{W}", f"s{a.seed}")
    cfg = make_cfg(f"mini_T_{L}x{W}_s{a.seed}", "mini_teachers", L, W, a.seed)
    # the MW teachers' recipe (make_jobs rec_fixed_std: LM until < 0.1% in 300 steps), geodesic + retry, capped
    res, m, bed = train(cfg, jdir, lm_cap=a.cap, stall=(300, 1e-3), wall=a.wall)
    out = os.path.join(HERE, teacher_file(L, W))
    omega = C.save_teacher(jdir, out)
    # sanity: the teacher's own weights in a student of its target score at the float64 floor
    sb = Bed(make_cfg("chk", "chk", L, W, 0, teacher=teacher_file(L, W)))
    ms = sb.build(); ms.flow.load_state_dict(sb.t.T.state_dict())
    e_self = sb.heldout(ms)[0]
    res.update(teacher_file=out, omega=omega, heldout_self=e_self, mw_heldout=res["final"]["eval_mean"])
    json.dump(res, open(os.path.join(jdir, "result.json"), "w"), indent=1)
    print(f"teacher {L}x{W}: MW held-out {res['final']['eval_mean']:.3e}; omega {omega:.15f}; teacher-as-student held-out {e_self:.2e}",
          flush=True)


def cmd_student(a):
    L, W = a.L, a.W
    jdir = os.path.join(RES, "students", f"{L}x{W}", f"s{a.seed}")
    if os.path.exists(os.path.join(jdir, "result.json")) and not a.force:
        print("done:", jdir); return
    cfg = make_cfg(f"mini_S_{L}x{W}_s{a.seed}", "mini_students", L, W, a.seed, teacher=teacher_file(L, W))
    train(cfg, jdir, lm_cap=a.cap, stall=(1000, 1e-5), wall=a.wall)


# ===================================================================== alignment (size generic)
class Layout:
    """Flat vector layout of an L x W flow: per layer [gate, W1 (W), b1 (W), W2 (W), b2]."""

    def __init__(self, L, W):
        self.L, self.W, self.PER = L, W, 3 * W + 2
        self.P = L * self.PER
        self.G, self.W1, self.B1 = 0, slice(1, 1 + W), slice(1 + W, 1 + 2 * W)
        self.W2, self.B2 = slice(1 + 2 * W, 1 + 3 * W), 3 * W + 1

    def blocks(self, v):
        return v.view(self.L, self.PER)


def _match(Us, Ut):
    """Hungarian matching of student units Us (W, 3) to teacher units Ut, cost min over sign of |s u - t|^2."""
    from scipy.optimize import linear_sum_assignment
    W = Us.shape[0]
    dot = Us @ Ut.T
    cost = (Us * Us).sum(1)[:, None] + (Ut * Ut).sum(1)[None, :] - 2 * dot.abs()
    r, c = linear_sum_assignment(cost.numpy())
    perm = torch.empty(W, dtype=torch.long); perm[torch.from_numpy(c)] = torch.from_numpy(r)
    sgn = torch.sign(dot[perm, torch.arange(W)]); sgn[sgn == 0] = 1.0
    return perm, sgn


def _best_scale(g, u, gT, uT):
    """a minimising (a g - gT)^2 + |u / a - uT|^2 (as landscape_lab._best_scale)."""
    f = lambda a: (a * g - gT) ** 2 + float(((u / a - uT) ** 2).sum())
    rts = np.roots([g * g, -g * gT, 0.0, float(u @ uT), -float(u @ u)])
    cands = [float(r.real) for r in rts if abs(r.imag) <= 1e-8 * abs(r) and r.real != 0]
    return min(cands, key=f) if cands else gT / g


def _align_layer(lay, b, r, gauge, max_it=50):
    g, w1, b1, w2, b2 = b[lay.G], b[lay.W1], b[lay.B1], b[lay.W2], b[lay.B2]
    Ut = torch.stack([r[lay.W1], r[lay.B1], r[lay.W2]], 1); uT = torch.cat([r[lay.W2], r[lay.B2:lay.B2 + 1]])
    best = None
    for a in ([float(r[lay.G] / g)] if gauge == "gate" else [float(r[lay.G] / g), 1.0]):
        prev = None
        for _ in range(1 if gauge == "gate" else max_it):
            perm, sgn = _match(torch.stack([w1, b1, w2 / a], 1), Ut)
            if gauge == "gate":
                break
            a_new = _best_scale(float(g), torch.cat([sgn * w2[perm], b2.view(1)]), float(r[lay.G]), uT)
            done = prev is not None and torch.equal(perm, prev[0]) and torch.equal(sgn, prev[1]) and abs(a_new - a) <= 1e-13 * abs(a)
            prev, a = (perm, sgn), (a if done else a_new)
            if done:
                break
        new = b.clone()
        new[lay.G] = r[lay.G] if gauge == "gate" else a * g
        new[lay.W1], new[lay.B1], new[lay.W2], new[lay.B2] = sgn * w1[perm], sgn * b1[perm], sgn * w2[perm] / a, b2 / a
        D = float((new - r).pow(2).sum())
        if best is None or D < best[0]:
            best = (D, new, a, perm, sgn)
    return best[1:]


def _shift_basis(lay, v):
    """(P, L-2) translation gauges (landscape_lab._shift_basis for any L x W)."""
    Bv = lay.blocks(v); n = max(lay.L - 2, 0)
    Z = torch.zeros(n, lay.L, lay.PER, dtype=F64)
    for k in range(n):
        Z[k, k, lay.B2] = 1 / Bv[k, lay.G]
        Z[k, k + 1, lay.B1] = -Bv[k + 1, lay.W1]
        Z[k, k + 2, lay.B2] = -1 / Bv[k + 2, lay.G]
    return Z.view(n, lay.P).T


def align(lay, v, ref, gauge="min", shifts=True, iters=10):
    """v in the gauge of ref, exact symmetries only: per-layer gate scale, tanh sign per unit, Hungarian unit order;
    shifts=True also the L-2 translation gauges (least squares, alternating with the per-layer step)."""
    ref = torch.as_tensor(ref, dtype=F64); Rb = lay.blocks(ref)
    Vb = lay.blocks(torch.as_tensor(v, dtype=F64).clone())
    D_prev = float("inf"); flips = 0
    for rnd in range(iters if shifts else 1):
        for k in range(lay.L):
            Vb[k], a, perm, sgn = _align_layer(lay, Vb[k], Rb[k], gauge)
            if rnd == 0:
                flips += int((sgn < 0).sum())
        if not shifts or lay.L < 3:
            break
        Bm = _shift_basis(lay, Vb.reshape(-1))
        c = torch.linalg.lstsq(Bm, (ref - Vb.reshape(-1))[:, None]).solution[:, 0]
        Vb = lay.blocks(Vb.reshape(-1) + Bm @ c).clone()
        D = float((Vb.reshape(-1) - ref).norm())
        if D > (1 - 1e-10) * D_prev:
            break
        D_prev = D
    return Vb.reshape(-1).clone()


def random_symmetry(lay, v, seed=0):
    g = torch.Generator().manual_seed(seed)
    Vb = lay.blocks(torch.as_tensor(v, dtype=F64).clone())
    for k in range(lay.L):
        perm = torch.randperm(lay.W, generator=g); sgn = torch.randint(0, 2, (lay.W,), generator=g).to(F64) * 2 - 1
        Vb[k, lay.W1], Vb[k, lay.B1], Vb[k, lay.W2] = sgn * Vb[k, lay.W1][perm], sgn * Vb[k, lay.B1][perm], sgn * Vb[k, lay.W2][perm]
    return Vb.reshape(-1).clone()


@torch.no_grad()
def map_dev(m, v, w, x):
    """max over points of |S_v(x) - S_w(x)| / |S_w(x)|."""
    set_flat(m, v); y1 = m(x)
    set_flat(m, w); y2 = m(x)
    return float(((y1 - y2).norm(dim=-1) / y2.norm(dim=-1)).max())


@torch.no_grad()
def twist(bed, m):
    """Twist g(J) of student m relative to the teacher (gauge_lab.twist): angle of T^-1 o S in the teacher's toy
    coordinates, circular mean over 256 angles on 40 tori. Returns (J, g)."""
    w0 = bed.t.nu0
    Jh = C.j_grid(40, bed.J_lo, bed.J_hi, w0, "asinh", 0.03, offset=0.5); nth = 256
    A = C.aa_points(Jh, C.angles(nth, 0.25)).reshape(-1, 2)
    X = bed.t.T.inverse(m(A)); q, p = X[:, 0], X[:, 1]
    d = torch.remainder(torch.atan2(-p / math.sqrt(w0), q * math.sqrt(w0)) - A[:, 0] + math.pi, 2 * math.pi) - math.pi
    d = d.view(len(Jh), nth)
    return Jh, torch.atan2(torch.sin(d).mean(1), torch.cos(d).mean(1))


# ===================================================================== analysis
def load_run(jdir):
    z = np.load(os.path.join(jdir, "curve.npz"))
    return json.load(open(os.path.join(jdir, "result.json"))), {k: z[k] for k in z.files}


def runs_of(L, W):
    base = os.path.join(RES, "students", f"{L}x{W}")
    if not os.path.isdir(base):
        return []
    out = []
    for s in sorted(os.listdir(base), key=lambda x: int(x[1:])):
        jd = os.path.join(base, s)
        if os.path.exists(os.path.join(jd, "result.json")):
            out.append(jd)
    return out


def cmd_analyse(a):
    sizes = [tuple(int(x) for x in s.split("x")) for s in a.sizes.split(",")]
    summary = dict(success_threshold=SUCCESS, step_costs=json.load(open(os.path.join(RES, "step_costs.json")))
                   if os.path.exists(os.path.join(RES, "step_costs.json")) else None, sizes={})
    for L, W in sizes:
        tres = os.path.join(RES, "teachers", f"{L}x{W}", "s0", "result.json")
        T_info = json.load(open(tres)) if os.path.exists(tres) else None
        jds = runs_of(L, W)
        if not jds:
            continue
        lay = Layout(L, W)
        bed = Bed(make_cfg("an", "an", L, W, 0, teacher=teacher_file(L, W)))
        m = bed.build(); vT = torch.cat([p.detach().reshape(-1) for p in bed.t.T.parameters()]); nT = float(vT.norm())
        X_ho = bed.ev.aa.reshape(-1, 2)
        per = []
        for jd in jds:
            res, cv = load_run(jd)
            S = torch.from_numpy(cv["snaps"][-1])
            Sa = align(lay, S, vT, "min", True); Sg = align(lay, S, vT, "gate", False)
            dev_al = max(map_dev(m, Sa, S, X_ho), map_dev(m, Sg, S, X_ho))
            dev_base = map_dev(m, random_symmetry(lay, S), S, X_ho)
            # aligned distance over training (every record point; Adam points every 100 steps)
            dist = np.array([float((align(lay, torch.from_numpy(v), vT, "min", True) - vT).norm()) / nT for v in cv["snaps"]])
            set_flat(m, S); Jh, gJ = twist(bed, m)
            e = res["final"]["eval_mean"]
            lm = cv["phase"] == 1
            row = dict(dir=jd, seed=res["seed"], heldout=e, heldout_max=res["final"]["eval_max"], train_mean=res["final"]["train_mean"],
                       success=bool(e < SUCCESS), lm_iters=res["lm_iters"], why=res["why"], lam_end=res["lam_end"], wall_s=res["wall_s"],
                       lm_s_per_step=res["lm_s_per_step"], dist_final_min_shift=float((Sa - vT).norm()) / nT,
                       dist_final_gate=float((Sg - vT).norm()) / nT, dist_raw=float((S - vT).norm()) / nT,
                       align_map_dev=dev_al, align_map_dev_baseline=dev_base,
                       twist_min=float(gJ.min()), twist_max=float(gJ.max()), twist_mean=float(torch.atan2(torch.sin(gJ).mean(), torch.cos(gJ).mean())),
                       heldout_adam_end=float(cv["heldout"][~lm][-1]), dist_adam_end=float(dist[~lm][-1]))
            for k in (0, 100, 250, 500, 1000, 2000):
                sel = np.where(lm & (cv["it"] <= k))[0]
                if len(sel):
                    i = sel[-1]
                    row[f"at{k}"] = dict(heldout=float(cv["heldout"][i]), loss=float(cv["loss"][i]), lam=float(cv["lr_lam"][i]), dist=float(dist[i]))
            # first LM step where held-out < 1e-8 and < SUCCESS
            for thr, key in ((1e-6, "it_below_1e-6"), (1e-8, "it_below_1e-8"), (SUCCESS, "it_below_1e-10")):
                sel = np.where(lm & (cv["heldout"] < thr))[0]
                row[key] = int(cv["it"][sel[0]]) if len(sel) else None
            np.savez(os.path.join(jd, "analysis.npz"), dist_min_shift=dist, twist_J=Jh.numpy(), twist_g=gJ.numpy())
            per.append(row)
            print(f"{L}x{W} s{res['seed']}: held-out {e:.2e} ({'OK' if e < SUCCESS else 'FAIL'}), {res['lm_iters']} LM its ({res['why']}), "
                  f"dist {row['dist_final_min_shift']:.3e} (gate {row['dist_final_gate']:.3e}), twist {row['twist_min']:+.3f}..{row['twist_max']:+.3f}, "
                  f"align map dev {dev_al:.1e} (baseline {dev_base:.1e})", flush=True)
        ok = [r for r in per if r["success"]]
        summary["sizes"][f"{L}x{W}"] = dict(L=L, W=W, P=lay.P, teacher=None if T_info is None else dict(
            mw_heldout=T_info["final"]["eval_mean"], lm_iters=T_info["lm_iters"], why=T_info["why"], heldout_self=T_info.get("heldout_self")),
            n=len(per), n_success=len(ok), success_rate=len(ok) / len(per), runs=per)
    json.dump(summary, open(os.path.join(RES, "pilot_summary.json"), "w"), indent=1)
    print(json.dumps({k: (v["n_success"], v["n"]) for k, v in summary["sizes"].items()}))


def cmd_figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = json.load(open(os.path.join(RES, "pilot_summary.json")))
    focus = a.focus
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ok_c, bad_c = "#1b7837", "#b2182b"
    runs = S["sizes"][focus]["runs"]
    for r in runs:
        _, cv = load_run(r["dir"]); an = np.load(os.path.join(r["dir"], "analysis.npz"))
        lm = cv["phase"] == 1
        x = cv["it"][lm] + 1
        c = ok_c if r["success"] else bad_c
        lab = f"s{r['seed']} {'reached T' if r['success'] else 'stalled'} ({r['heldout']:.1e})"
        ax[0, 0].loglog(x, cv["heldout"][lm], color=c, lw=1.2, alpha=0.85, label=lab)
        ax[0, 1].loglog(x, cv["lr_lam"][lm], color=c, lw=1.0, alpha=0.8)
        ax[1, 0].loglog(x, an["dist_min_shift"][lm], color=c, lw=1.2, alpha=0.85)
    ax[0, 0].axhline(SUCCESS, color="k", ls=":", lw=0.8)
    ax[0, 0].set(xlabel="LM step + 1", ylabel="held-out mean |dE/E| vs teacher", title=f"{focus} students: learning curves (LM phase)")
    ax[0, 0].legend(fontsize=7, ncol=2)
    ax[0, 1].set(xlabel="LM step + 1", ylabel="LM damping lambda", title="damping history")
    ax[1, 0].set(xlabel="LM step + 1", ylabel="|S - T| / |T| after alignment (gauge min + shifts)", title="aligned distance to the teacher")
    # success rate and final error by size
    names = list(S["sizes"])
    for i, k in enumerate(names):
        for r in S["sizes"][k]["runs"]:
            ax[1, 1].semilogy(i + (r["seed"] - 3) * 0.04, max(r["heldout"], 1e-17), "o", ms=5,
                              color=ok_c if r["success"] else bad_c, alpha=0.8)
    ax[1, 1].axhline(SUCCESS, color="k", ls=":", lw=0.8)
    ax[1, 1].set_xticks(range(len(names)))
    ax[1, 1].set_xticklabels([f"{k}\n{S['sizes'][k]['n_success']}/{S['sizes'][k]['n']}" for k in names])
    ax[1, 1].set(ylabel="final held-out vs teacher", title="same-size students by size (successes / seeds)")
    for a_ in ax.flat:
        a_.grid(alpha=0.3, which="both", lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "pilot_curves.png"), dpi=130)
    print("wrote", os.path.join(RES, "pilot_curves.png"))


# ===================================================================== curved-path test
def energy_jacobian(m, bed, chunk=16):
    """Training energy-residual Jacobian (N, P) and residual (N,) at m's parameters, normalised as train_lm."""
    from torch.func import functional_call, jacrev, vmap
    names = [n for n, _ in m.named_parameters()]
    params = {n: p.detach() for n, p in m.named_parameters()}
    f_point = lambda p, x: functional_call(m, p, (x[None, :],))[0]
    rows = []; sN = math.sqrt(bed.nO * bed.nS)
    for o0 in range(0, bed.nO, chunk):
        x = bed.aa[o0:o0 + chunk].reshape(-1, 2); k = x.shape[0] // bed.nS
        Jd = vmap(jacrev(f_point), in_dims=(None, 0))(params, x)
        Jy = torch.cat([Jd[n].reshape(x.shape[0], 2, -1) for n in names], -1)
        with torch.no_grad():
            y = m(x)
        y = y.detach().requires_grad_(True)
        dHdy = torch.autograd.grad(bed.t.H(y).sum(), y)[0]
        dH = (dHdy[:, :, None] * Jy).sum(1).view(k, bed.nS, -1)
        rows.append(((dH - dH.mean(1, keepdim=True)) / bed.es[o0:o0 + k][..., None] / sN).reshape(k * bed.nS, -1))
    with torch.no_grad():
        r = (C.residual_host(m, bed.aa, bed.t, bed.nO, bed.nS, bed.es) / sN).reshape(-1)
    return torch.cat(rows), r


def cmd_bezier(a):
    L, W = a.L, a.W
    jd = os.path.join(RES, "students", f"{L}x{W}", f"s{a.seed}")
    lay = Layout(L, W)
    bed = Bed(make_cfg("bz", "bz", L, W, 0, teacher=teacher_file(L, W)))
    m = bed.build()
    vT = torch.cat([p.detach().reshape(-1) for p in bed.t.T.parameters()]); nT = float(vT.norm())
    st = torch.load(os.path.join(jd, "state.pt"), weights_only=False)["state"]
    m.load_state_dict(st); S_raw = get_flat(m)
    S = align(lay, S_raw, vT, a.gauge, a.gauge == "min")
    X_ho = bed.ev.aa.reshape(-1, 2)
    print(f"student {L}x{W} s{a.seed}: held-out {bed.heldout(m)[0]:.3e}; aligned |S-T|/|T| {float((S - vT).norm()) / nT:.3e}; "
          f"map dev of alignment {map_dev(m, S, S_raw, X_ho):.1e}", flush=True)
    K = a.k
    ts = torch.tensor([(i + 0.5) / K for i in range(K)], dtype=F64)  # stratified interior points
    t_fine = torch.linspace(0, 1, a.fine, dtype=F64)

    def bez(Cc, t):
        return (1 - t) ** 2 * S + 2 * t * (1 - t) * Cc + t ** 2 * vT

    def F(Cc):
        tot = 0.0
        for t in ts:
            set_flat(m, bez(Cc, t)); tot += bed.loss(m)
        return tot / K

    def profile(Cc):
        h, l = [], []
        for t in t_fine:
            set_flat(m, bez(Cc, t)); h.append(bed.heldout(m)[0]); l.append(bed.loss(m))
        return np.array(h), np.array(l)

    Cc = 0.5 * (S + vT)  # the straight line
    h_line, l_line = profile(Cc)
    print(f"straight line: max held-out {h_line.max():.3e} at t = {float(t_fine[h_line.argmax()]):.3f}; "
          f"endpoints {h_line[0]:.2e}, {h_line[-1]:.2e}", flush=True)
    lam, f = a.lam0, F(Cc); hist = [(0, f, lam)]; t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        A = torch.zeros(lay.P, lay.P, dtype=F64); g = torch.zeros(lay.P, dtype=F64)
        for t in ts:
            set_flat(m, bez(Cc, t))
            Jm, r = energy_jacobian(m, bed)
            b = float(2 * t * (1 - t))
            A += b * b * (Jm.T @ Jm); g += b * (Jm.T @ r)
        A /= K; g /= K  # F = (1/K) sum |r_k|^2 (mean r^2 per point x N... same scale as LM's 2f)
        dg = torch.diagonal(A).clone(); D = dg + 1e-12 * dg.max()
        acc = False
        for _ in range(40):
            Lc, info = torch.linalg.cholesky_ex(A + lam * torch.diag(D))
            if int(info) != 0:
                lam *= 10; continue
            d = -torch.cholesky_solve(g[:, None], Lc)[:, 0]
            fn = F(Cc + d)
            if math.isfinite(fn) and fn < f:
                Cc, f, acc = Cc + d, fn, True; lam = max(lam / 3, 1e-15); break
            lam *= 4
        hist.append((it, f, lam))
        if it % 10 == 0 or not acc:
            print(f"  bezier it {it:4d}: mean loss over t {f:.3e}  lam {lam:.1e}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if not acc or (a.wall and time.perf_counter() - t0 > a.wall):
            break
    h_bz, l_bz = profile(Cc)
    out = dict(L=L, W=W, seed=a.seed, gauge=a.gauge, K=K, iters=len(hist) - 1, t_fine=t_fine.tolist(),
               dist_S_T=float((S - vT).norm()) / nT, dist_C_mid=float((Cc - 0.5 * (S + vT)).norm()) / nT,
               line=dict(heldout=h_line.tolist(), loss=l_line.tolist(), max_heldout=float(h_line.max()),
                         t_at_max=float(t_fine[h_line.argmax()])),
               bezier=dict(heldout=h_bz.tolist(), loss=l_bz.tolist(), max_heldout=float(h_bz.max()), t_at_max=float(t_fine[h_bz.argmax()])),
               endpoints=dict(S=float(h_line[0]), T=float(h_line[-1])), hist=hist, wall_s=time.perf_counter() - t0)
    path = os.path.join(RES, f"bezier_{L}x{W}_s{a.seed}_{a.gauge}.json")
    json.dump(out, open(path, "w"), indent=1)
    np.save(path.replace(".json", "_C.npy"), Cc.numpy())
    print(f"Bezier: max held-out {h_bz.max():.3e} at t = {out['bezier']['t_at_max']:.3f} (line {h_line.max():.3e}); "
          f"S {h_line[0]:.2e}; |C - mid|/|T| {out['dist_C_mid']:.3e}; wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cost"); p.add_argument("sizes", type=int, nargs="+")
    p = sp.add_parser("teacher"); p.add_argument("L", type=int); p.add_argument("W", type=int)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--cap", type=int, default=3000); p.add_argument("--wall", type=float, default=None)
    p = sp.add_parser("student"); p.add_argument("L", type=int); p.add_argument("W", type=int); p.add_argument("seed", type=int)
    p.add_argument("--cap", type=int, default=3000); p.add_argument("--wall", type=float, default=None); p.add_argument("--force", action="store_true")
    p = sp.add_parser("analyse"); p.add_argument("--sizes", default="2x8,2x16,4x8,4x16,8x16")
    p = sp.add_parser("figure"); p.add_argument("--focus", default="4x8")
    p = sp.add_parser("bezier"); p.add_argument("L", type=int); p.add_argument("W", type=int); p.add_argument("seed", type=int)
    p.add_argument("--k", type=int, default=12); p.add_argument("--iters", type=int, default=300); p.add_argument("--fine", type=int, default=61)
    p.add_argument("--lam0", type=float, default=1e-3); p.add_argument("--wall", type=float, default=None)
    p.add_argument("--gauge", default="min", choices=["min", "gate"])
    a = ap.parse_args()
    {"cost": cmd_cost, "teacher": cmd_teacher, "student": cmd_student, "analyse": cmd_analyse, "figure": cmd_figure,
     "bezier": cmd_bezier}[a.cmd](a)


if __name__ == "__main__":
    main()
