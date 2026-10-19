"""Fine-tune a trained MW model under a chosen objective (iteratively reweighted least squares inside LM).

    python finetune_objective.py OBJECTIVE STEPS OUT_DIR [--model teachers/mw_32x32_a256.pt] [--angles 256]

OBJECTIVE is one of
  l2        plain least squares (the control): minimises the mean of r^2
  l4, l8    the mean of r^4 or r^8, by reweighting with (|r| + delta)^(p-2); aimed at the tails (99th pct, max)
  balanced  the sum over tori of log(mean over the torus of r^2), by reweighting each torus with 1 / its mean r^2;
            every torus's relative improvement counts the same, so the fit cannot trade one band for another
Steps are accepted only if the objective itself decreases (core.train_lm(objective=...)); geodesic LM throughout.
The held-out set is the usual one (200 tori between the training tori x 512 angles), never used for steps.
Writes OUT_DIR/log.jsonl (held-out mean, max, percentiles and z_max bands every 25 steps), state.pt, summary.json.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import math
import time

import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, HERE)
import core as C  # noqa: E402


def make_objective(name, nO, nS):
    """(weight_fn, objective) for core.train_lm; residuals arrive as (nO, nS) for weights and flat for objectives."""
    if name == "l2":
        return None, None
    if name in ("l4", "l8"):
        p = int(name[1:])

        def w(r, k):
            a = r.abs(); rms = a.pow(2).mean().sqrt()
            ww = (a + 1e-3 * rms) ** (p - 2)
            return ww / ww.mean()
        return w, (lambda r: float(r.abs().pow(p).mean()))
    if name == "balanced":
        def w(r, k):
            mt = r.pow(2).mean(1, keepdim=True)
            ww = (1 / (mt + 1e-6 * mt.mean())).expand(nO, nS)
            return ww / ww.mean()
        return w, (lambda r: float(torch.log(r.view(nO, nS).pow(2).mean(1)).sum()))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objective"); ap.add_argument("steps", type=int); ap.add_argument("out")
    ap.add_argument("--model", default="teachers/mw_32x32_a256.pt"); ap.add_argument("--angles", type=int, default=256)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--init", default=None, help="continue from a state.pt written by an earlier run of this script")
    ap.add_argument("--hours", type=float, default=None, help="wall-time cap")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--lam0", type=float, default=1e-4)
    ap.add_argument("--damp-floor", type=float, default=1e-12)
    ap.add_argument("--grow", default=None, help="insert a (q, p) pair of new layers after each of these old layer indices, e.g. 3,7,11")
    ap.add_argument("--grow-gate", type=float, default=0.02, help="gate of the inserted layers (their output layer starts at zero)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--toy-learned", default=None, choices=["lnJ", "amp"],
                    help="switch to a learned toy frequency in this variable, starting at nu0 (same map at the start)")
    a = ap.parse_args()
    ck = torch.load(os.path.join(HERE, a.model), weights_only=False)
    mw = C.MWVertical(); w0 = float(ck["omega"])
    MW = dict(zmin_pc=5.0, zmax_kpc=8.32, n_orbits=96, n_steps=a.angles, spacing="asinh", a0=0.03)
    cfg = dict(name=f"ft_{a.objective}", target="mw", model=ck["model"], toy=dict(mode="fixed", omega="nu0"), data=MW,
               eval=dict(n_orbits=200, n_steps=512), source=ck.get("source"), objective=a.objective)
    Jlo = float(C.J_of_zmax(MW["zmin_pc"] / 1000 / C.RO_KPC, mw.phi)[0]); Jhi = float(C.J_of_zmax(MW["zmax_kpc"] / C.RO_KPC, mw.phi)[0])
    m = C.build_model(ck["model"], cfg["toy"], w0, Jlo, Jhi, 0)
    with torch.no_grad():
        m.flow.load_state_dict(ck["flow_state"])
    it0 = 0
    if a.init:
        sti = torch.load(a.init, weights_only=False); m.load_state_dict(sti["state"]); it0 = int(sti.get("it", 0))
        print(f"continuing from {a.init} (step {it0})", flush=True)
    if a.grow:  # deepen the model without changing its map: new layer pairs with tiny gates, parity of old layers kept
        after = sorted(int(x) for x in a.grow.split(",")); L0 = m.flow.num_layers
        mc = dict(cfg["model"], n_layers=L0 + 2 * len(after)); cfg["model"] = mc; cfg["grown_from"] = dict(layers=L0, after=after)
        g = C.build_model(mc, cfg["toy"], w0, Jlo, Jhi, a.seed); gs = g.state_dict(); old = m.state_dict(); pos, new = 0, []
        for j in range(L0):
            for k, v in old.items():
                if k.startswith(f"flow.layers.{j}."):
                    gs[k.replace(f"flow.layers.{j}.", f"flow.layers.{pos}.", 1)] = v
            pos += 1
            if j in after:
                new += [pos, pos + 1]; pos += 2
        for k, v in old.items():
            if k.startswith("toy."):
                gs[k] = v
        # zero the inserted layers' output weights and bias, so each new shear is exactly zero and the map is unchanged,
        # but keep an ordinary gate: the output weights then have well-scaled gradients from the first step (with zero
        # gates instead, the new weights start with no gradient at all and LM's Marquardt scaling mishandles them)
        gen = torch.Generator().manual_seed(a.seed)
        for i in new:
            k = f"flow.layers.{i}.conditioner.gate"
            gs[k] = a.grow_gate * torch.sign(torch.randn(gs[k].shape, generator=gen, dtype=gs[k].dtype))
            for kk in (f"flow.layers.{i}.conditioner.layers.2.weight", f"flow.layers.{i}.conditioner.layers.2.bias"):
                gs[kk] = torch.zeros_like(gs[kk])
        g.load_state_dict(gs); m = g
        print(f"grown from {L0} to {mc['n_layers']} layers ({C.n_params(m)} parameters); new layers at {new}", flush=True)
    if a.toy_learned:  # learned omega(J) with all coefficients zero: omega = nu0 and kappa = 0, so the map is unchanged
        cfg["toy"] = dict(mode="learned", omega="nu0", n_cheb=6, var=a.toy_learned)
        g = C.build_model(cfg["model"], cfg["toy"], w0, Jlo, Jhi, a.seed)
        missing, unexpected = g.load_state_dict(m.state_dict(), strict=False)
        assert list(missing) == ["toy.c"] and not unexpected, (missing, unexpected)
        m = g
        print(f"learned toy frequency in {a.toy_learned} ({C.n_params(m)} parameters)", flush=True)
    Jt = C.train_grid(MW, Jlo, Jhi, w0); nO, nS = len(Jt), a.angles
    aa = C.aa_points(Jt, C.angles(nS, 0.5)); es = (w0 * Jt).reshape(-1, 1)
    ev = C.Evaluator(cfg, mw, Jlo, Jhi, w0, Jt, nS)
    os.makedirs(a.out, exist_ok=True); json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "a"); t0 = time.perf_counter()

    def record(it, loss, lam):
        q = ev.quick(m)
        rec = dict(it=it, objective=loss, lam=lam, t=time.perf_counter() - t0, **q)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{a.objective}] step {it:4d}  held-out {q['eval_mean']:.3e}  p99 {q['eval_p99']:.2e}  max {q['eval_max']:.2e}  "
              f"bands {q['band_0.005_0.15']:.1e} / {q['band_0.15_1']:.1e} / {q['band_1_9']:.1e}  train {q['train_mean']:.2e}  "
              f"{rec['t'] / 60:.1f} min", flush=True)
    record(it0, float("nan"), float("nan"))
    wfn, obj = make_objective(a.objective, nO, nS)

    def save(it, lam):
        if (it - it0) % a.ckpt_every == 0:
            tmp = os.path.join(a.out, "state.pt.tmp")
            torch.save(dict(state=C.state_f64(m), it=it, lam=lam), tmp); os.replace(tmp, os.path.join(a.out, "state.pt"))
    n, why, _ = C.train_lm(m, aa, mw, nO, nS, max_iters=a.steps, lam=a.lam0, escale=es, geodesic=True, weight_fn=wfn, objective=obj,
                           record=lambda it, l, lam: record(it + 1, l, lam) if (it + 1 - it0) % a.every == 0 else None, every=1,
                           ckpt=save, it0=it0, wall_cap=a.hours * 3600 if a.hours else None, damp_floor=a.damp_floor)
    n += it0
    record(n, float("nan"), float("nan"))
    torch.save(dict(state=C.state_f64(m), it=n), os.path.join(a.out, "state.pt"))
    json.dump(dict(objective=a.objective, steps=n, stop=why, wall_s=time.perf_counter() - t0), open(os.path.join(a.out, "summary.json"), "w"))
    print(f"[{a.objective}] done: {n} steps ({why}), {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
