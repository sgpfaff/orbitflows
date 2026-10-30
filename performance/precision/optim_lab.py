"""Optimisation experiments on an exactly representable target: the 16-layer MW teacher (teachers/mw_fixed_std_s1.pt).

The first bundle's 16 x 64 students of this teacher (mwteacher_v2) contain it exactly, yet all six stalled near 2e-5
held-out after ~20k LM steps and stayed there for another 60-75k. Any residual is optimisation, so this is the test
bed for changes to the optimiser.

    python optim_lab.py damping  STUDENT_JOB_DIR [--iters 400] [--variants marquardt,levenberg,sqrt,tensor,floor6]
    python optim_lab.py homotopy STUDENT_JOB_DIR [--stages 20] [--stage-iters 150] [--final-iters 3000] [--seed 0]

damping: from the stalled student's checkpoint, the same number of LM iterations (geodesic, retry_lam 1e-11) with each
  damping matrix (core.train_lm damping=; floor6 is Marquardt with damp_floor 1e-6). Escaping the stall shows as a
  drop in the held-out error against the teacher.
homotopy: a fresh student (same architecture and seed rule), no Adam, trained by LM along the family of targets
  H_s = (1 - s) H_SHO + s H_teacher, s = 0 .. 1 in equal stages, each stage started from the previous one's solution
  (and its damping). At s = 0 the target is the toy itself, so the answer is the identity, near which the flow starts.
  The last stage (s = 1) runs on. The held-out error against the teacher is logged after every stage and every 50 steps
  of the last one, with the cumulative LM step count, to compare with the students' learning curves.
Writes OUT (default optim_lab_<mode>.json next to this file).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402


class Blend:
    """H_s = (1 - s) (p^2 + nu0^2 q^2) / 2 + s H_teacher."""
    name = "blend"

    def __init__(self, teacher, s):
        self.t, self.s, self.nu0 = teacher, s, teacher.nu0

    def H(self, ps):
        sho = 0.5 * (ps[..., 1] ** 2 + self.nu0 ** 2 * ps[..., 0] ** 2)
        return sho if self.s == 0 else (1 - self.s) * sho + self.s * self.t.H(ps)


def setup(jd, device="cpu"):
    cfg = json.load(open(os.path.join(jd, "config.json")))
    cfg["teacher"]["checkpoint"] = os.path.join(HERE, cfg["teacher"]["checkpoint"]) \
        if not os.path.isabs(cfg["teacher"]["checkpoint"]) else cfg["teacher"]["checkpoint"]
    t = C.make_target(cfg); d = cfg["data"]; mw = C.MWVertical()
    J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, mw.phi)[0]); J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, mw.phi)[0])
    w0 = t.nu0
    J_tr = C.train_grid(d, J_lo, J_hi, w0)
    aa = C.aa_points(J_tr, C.angles(d["n_steps"], 0.5)); nO, nS = aa.shape[:2]; aa = aa.to(device)
    es = (w0 * J_tr).reshape(-1, 1)
    ev0 = C.Evaluator(cfg, t, J_lo, J_hi, w0, J_tr, nS)
    build = lambda seed, dev=device: C.build_model(cfg["model"], cfg["toy"], w0, J_lo, J_hi, seed, C.F64, dev)

    class Ev:  # the evaluator works on a float64 CPU copy of the model
        @staticmethod
        def cpu(m):
            if next(m.parameters()).device.type == "cpu":
                return m
            mc = build(0, "cpu"); mc.load_state_dict(C.state_f64(m)); return mc

        @staticmethod
        def quick(m):
            return ev0.quick(Ev.cpu(m))

        @staticmethod
        def vs(m, target):
            """Held-out mean of |H - <H>| / |<H>| against any target (e.g. a homotopy stage)."""
            with torch.no_grad():
                return float(C.residual_host(Ev.cpu(m), ev0.aa, target, *ev0.aa.shape[:2]).abs().mean())
    return cfg, t, aa, nO, nS, es, Ev, build


def damping(a):
    cfg, t, aa, nO, nS, es, ev, build = setup(a.job, a.device)
    st = torch.load(os.path.join(a.job, "state.pt"), weights_only=False)
    rows = []
    for v in a.variants.split(","):
        m = build(cfg.get("seed", 0)); m.load_state_dict({k: v.to(a.device) for k, v in st["state"].items()})
        mode, floor = ("marquardt", 1e-6) if v == "floor6" else (v, 1e-12)
        e0 = ev.quick(m); curve = [(0, e0["eval_mean"], e0["train_mean"], float("nan"))]; t0 = time.perf_counter()

        def rec(it, loss, lam):
            if it % 25 == 0:
                q = ev.quick(m); curve.append((it, q["eval_mean"], q["train_mean"], lam))
        n, why, lam = C.train_lm(m, aa, t, nO, nS, max_iters=a.iters, lam=a.lam0, record=rec, every=1, escale=es, geodesic=True,
                                 retry_lam=1e-11, damping=mode, damp_floor=floor, chunk_orbits=a.chunk)
        q = ev.quick(m); curve.append((n, q["eval_mean"], q["train_mean"], lam))
        rows.append(dict(variant=v, iters=n, why=why, s=time.perf_counter() - t0, start=e0["eval_mean"], end=q["eval_mean"],
                         end_max=q["eval_max"], train_end=q["train_mean"], curve=curve))
        print(f"{v:10s} {n:4d} its ({why:10s}, {time.perf_counter() - t0:5.0f} s): held-out {e0['eval_mean']:.3e} -> {q['eval_mean']:.3e} "
              f"(x{e0['eval_mean'] / q['eval_mean']:.2f})  train {q['train_mean']:.2e}  lam {lam:.1e}", flush=True)
        json.dump(dict(mode="damping", job=a.job, rows=rows), open(a.out, "w"), indent=1)


def homotopy(a):
    cfg, t, aa, nO, nS, es, ev, build = setup(a.job, a.device)
    m = build(a.seed)
    log, tot, lam, t0 = [], 0, a.lam0, time.perf_counter()
    q = ev.quick(m); log.append(dict(stage=0, s=0.0, it=0, eval_mean=q["eval_mean"], t=0.0))
    print(f"start: held-out vs teacher {q['eval_mean']:.3e}", flush=True)
    for k in range(1, a.stages + 1):
        s = k / a.stages; last = k == a.stages
        target = t if last else Blend(t, s)

        trace = []; own0 = ev.vs(m, target)  # held-out error against this stage's target, before its first step

        def rec(it, loss, lam_, tot0=tot):
            trace.append((it, loss, lam_))
            if last and it % 50 == 0:
                q = ev.quick(m); log.append(dict(stage=k, s=s, it=tot0 + it, eval_mean=q["eval_mean"], train_mean=q["train_mean"],
                                                 loss=loss, lam=lam_, t=time.perf_counter() - t0))
                print(f"   s=1 step {tot0 + it:6d}: held-out {q['eval_mean']:.3e}  train {q['train_mean']:.2e}  lam {lam_:.1e}  "
                      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        n, why, lam = C.train_lm(m, aa, target, nO, nS, max_iters=a.final_iters if last else a.stage_iters, lam=lam,
                                 record=rec, every=1, escale=es, geodesic=True, retry_lam=1e-11, damping=a.damping,
                                 stall_window=None if last else 40, stall_rtol=None if last else 1e-3, chunk_orbits=a.chunk)
        tot += n
        q = ev.quick(m); own = ev.vs(m, target)
        lams = [x[2] for x in trace]
        st = dict(loss_start=trace[0][1], loss_end=trace[-1][1], own_heldout_start=own0, own_heldout=own, lam_min=min(lams), lam_max=max(lams),
                  loss_trace=[(x[0], x[1]) for x in trace[::max(1, len(trace) // 30)]])
        log.append(dict(stage=k, s=s, it=tot, eval_mean=q["eval_mean"], train_mean=q["train_mean"], why=why, lam=lam,
                        t=time.perf_counter() - t0, **st))
        print(f"stage {k:2d} s={s:.3f}: {n:4d} its ({why}), total {tot:6d}; own target: loss {st['loss_start']:.2e} -> {st['loss_end']:.2e}, "
              f"held-out {own0:.2e} -> {own:.2e}; vs teacher {q['eval_mean']:.3e}; lam {st['lam_min']:.0e}..{st['lam_max']:.0e}; "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        json.dump(dict(mode="homotopy", job=a.job, seed=a.seed, stages=a.stages, stage_iters=a.stage_iters, damping=a.damping, log=log),
                  open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["damping", "homotopy"]); ap.add_argument("job")
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--variants", default="marquardt,levenberg,sqrt,tensor,floor6")
    ap.add_argument("--lam0", type=float, default=1e-11)
    ap.add_argument("--stages", type=int, default=20); ap.add_argument("--stage-iters", type=int, default=150)
    ap.add_argument("--final-iters", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--damping", default="marquardt"); ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--chunk", type=int, default=16, help="tori per Jacobian chunk")
    a = ap.parse_args()
    a.out = a.out or os.path.join(HERE, f"optim_lab_{a.mode}.json")
    (damping if a.mode == "damping" else homotopy)(a)


if __name__ == "__main__":
    main()
