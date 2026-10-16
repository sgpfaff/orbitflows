"""Train one 3D action-angle map from a JSON config:  python run3d.py configs/<name>.json [--out results]

Writes results/<name>/{config.json, log.jsonl, model.pt, result.json}. Config keys:
  target  {"kind": "aniso_ho" | "sep_quartic", "omegas": [...], "lams": [...]}
  toy     {"omegas": [w, w, w]}                         Cartesian 3D oscillator
  data    {"n_per_axis": 5, "J_lo": [...], "J_hi": [...], "n_angles": 256}
  model   {"n_lin": 2, "n_blocks": 4, "width": 16, "gate_std": 1e-3}
  adam    {"steps": 300, "lr": 1e-3}
  lm      {"max_iters": 300, "geodesic": true, "wall_hours": 1.0}
  init_from   name of an earlier run in the same results folder: its matching layers are loaded (grow a model)
  seed, threads
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core3d as C  # noqa: E402


def main(cfg, out_root):
    torch.set_num_threads(int(cfg.get("threads", os.environ.get("NTHREADS", 4))))
    jdir = os.path.join(out_root, cfg["name"]); os.makedirs(jdir, exist_ok=True)
    json.dump(cfg, open(os.path.join(jdir, "config.json"), "w"), indent=1)
    t = C.make_target(cfg["target"]); d = cfg["data"]
    if d.get("kind", "cartesian") == "spherical":  # tori labelled by (J_r, L, cos i)
        sg = lambda mid: C.spherical_action_grid(d["n_per_axis"], d["Jr"][0], d["Jr"][1], d["L"][0], d["L"][1], d.get("cosi_max", 0.9), mid)
        Jt, J_eval = sg(False), sg(True)
        if d.get("tori") == "sobol":
            Jt = C.sobol_spherical_actions(d["n_tori"], d["Jr"][0], d["Jr"][1], d["L"][0], d["L"][1], d.get("cosi_max", 0.9), seed=1 + cfg.get("seed", 0))
    else:
        Jt, J_eval = C.action_grid(d["n_per_axis"], d["J_lo"], d["J_hi"]), C.action_grid(d["n_per_axis"], d["J_lo"], d["J_hi"], midpoints=True)
        if d.get("tori") == "sobol":  # scattered training tori; the held-out grid stays the same
            Jt = C.sobol_actions(d["n_tori"], d["J_lo"], d["J_hi"], seed=1 + cfg.get("seed", 0))
    aa = C.aa_points(Jt, C.angle_samples(d["n_angles"], cfg.get("seed", 0)))
    m = C.build_model(cfg["model"], cfg["toy"], seed=cfg.get("seed", 0))
    if cfg.get("init_from"):  # grow: load every matching layer of an earlier model (e.g. a linear-only fit)
        src = torch.load(os.path.join(out_root, cfg["init_from"], "model.pt"), weights_only=False)["state"]
        own = m.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        m.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)}/{len(own)} tensors from {cfg['init_from']}", flush=True)
    esc = m.toy.energy(Jt).reshape(-1, 1)
    ev = C.Evaluator(t, J=J_eval, n_angles=d.get("n_eval_angles", 1024))
    logf = open(os.path.join(jdir, "log.jsonl"), "w"); t0 = time.perf_counter()

    def record(stage, it, loss, lam=None):
        rec = dict(stage=stage, it=it, loss=loss, lam=lam, t=round(time.perf_counter() - t0, 2), **ev.quick(m))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(f"[{cfg['name']}] {stage} it={it} loss={loss:.3e} held-out={rec['eval_mean']:.3e} max={rec['eval_max']:.3e} "
              f"vs-true-E={rec['eval_mean_trueE']:.3e} t={rec['t']/60:.1f}min", flush=True)

    print(f"{cfg['name']}: {C.n_params(m)} parameters, {aa.shape[0]} tori x {aa.shape[1]} angles", flush=True)
    record("init", 0, float(C.residuals(m, aa, t, esc).pow(2).mean()))
    A = cfg.get("adam", {})
    if A.get("steps", 0):
        C.train_adam(m, aa, t, A["steps"], A.get("lr", 1e-3), esc, record=lambda k, l: record("adam", k, l), every=A.get("every", 100))
    L = cfg.get("lm", {})
    why = None
    if L.get("max_iters", 0):
        wall = L.get("wall_hours"); wall = wall * 3600 if wall else None
        n, why, lam = C.train_lm(m, aa, t, esc, L["max_iters"], lam=L.get("lam0", 1e-3), geodesic=L.get("geodesic", True),
                                 record=lambda k, l, lam: record("lm", k, l, lam), every=L.get("every", 10), wall_cap=wall,
                                 chunk=L.get("chunk", 8))
    torch.save(dict(state=m.state_dict(), config=cfg), os.path.join(jdir, "model.pt"))
    res = dict(config=cfg, n_params=C.n_params(m), lm_stop=why, wall_s=time.perf_counter() - t0, final=ev.quick(m), action_freq=ev.action_freq(m))
    json.dump(res, open(os.path.join(jdir, "result.json"), "w"))
    f = res["final"]
    print(f"DONE {cfg['name']}: held-out {f['eval_mean']:.3e} max {f['eval_max']:.3e}; torus-mean E vs exact H(J) "
          f"max {max(res['action_freq']['rel_E_err']):.1e}; frequencies max {max(res['action_freq']['rel_freq_err']):.1e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    main(json.load(open(a.config)), a.out)
