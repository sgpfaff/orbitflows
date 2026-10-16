"""Train (or resume) one 3D action-angle map from a JSON config:  python run3d.py configs/<name>.json [--out results]

Writes results/<name>/{config.json, log.jsonl, state.pt, model.pt, result.json, DONE}. state.pt is written every
`lm.ckpt_every` LM steps (atomically); rerunning the same command resumes from it, so a session can end at any time.

Config keys:
  target  {"kind": "aniso_ho" | "sep_quartic" | "isochrone", ...}
  toy     {"omegas": [w, w, w]} (Cartesian oscillator) or {"kind": "spherical_ho", "omega": w}
  data    {"n_per_axis": 5, "J_lo": [...], "J_hi": [...], "n_angles": 256}  or, for spherical toys,
          {"kind": "spherical", "n_per_axis": 5, "Jr": [lo, hi], "L": [lo, hi], "cosi_max": 0.9, "n_angles": 256}
  model   {"n_lin": 2, "n_blocks": 4, "width": 16, "gate_std": 1e-3}
  adam    {"steps": 300, "lr": 1e-3}
  lm      {"max_iters": 300, "geodesic": true, "wall_hours": 1.0, "every": 10}
          with "invariance_beta": b (0 included), the run uses plain LM on energy + b * invariance residuals
          (core3d.train_lm_inv), "inv_angles" angles per torus for the invariance rows, checkpoints every "ckpt_every"
  init_from   a .pt file (path relative to this directory) or the name of an earlier run in the results folder:
              its matching layers are loaded (warm start / grow a model)
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


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def spherical_diagnostics(m, t, Jh, th, n_inv=256):
    """|L| wander around each held-out torus relative to J_r + L, and the invariance residual (on n_inv angles)."""
    aa = C.aa_points(Jh, th)
    with torch.no_grad():
        xv = m(aa.reshape(-1, 6)).view(Jh.shape[0], -1, 6)
        Lm = torch.cross(xv[..., :3], xv[..., 3:], dim=-1).norm(dim=-1)
        w = Lm.std(1) / (Jh[:, 0] + Jh[:, 1])
    ai = aa[:, :n_inv]
    R, _ = C.invariance_residuals(m, ai, t, C.invariance_scale(m, ai))
    return dict(L_wander_median=float(w.median()), L_wander_max=float(w.max()), invariance_rms=float(R.detach().pow(2).sum(-1).mean().sqrt()))


def main(cfg, out_root):
    torch.set_num_threads(int(os.environ.get("NTHREADS", cfg.get("threads", 4))))
    jdir = os.path.join(out_root, cfg["name"]); os.makedirs(jdir, exist_ok=True)
    if os.path.exists(os.path.join(jdir, "DONE")):
        print("already done:", cfg["name"]); return
    json.dump(cfg, open(os.path.join(jdir, "config.json"), "w"), indent=1)
    t = C.make_target(cfg["target"]); d = cfg["data"]
    spherical = d.get("kind", "cartesian") == "spherical"
    if spherical:  # tori labelled by (J_r, L, cos i)
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
    st_path = os.path.join(jdir, "state.pt")
    st = torch.load(st_path, weights_only=False) if os.path.exists(st_path) else None
    if st:
        m.load_state_dict(st["state"])
        print(f"resuming {cfg['name']} at {st['stage']} step {st['it']} ({st['t'] / 3600:.2f} h so far)", flush=True)
    elif cfg.get("init_from"):  # warm start: every matching layer of an earlier model
        src_path = cfg["init_from"]
        src_path = os.path.join(HERE, src_path) if src_path.endswith(".pt") else os.path.join(out_root, src_path, "model.pt")
        src = torch.load(src_path, weights_only=False)["state"]
        own = m.state_dict(); keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        m.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)}/{len(own)} tensors from {cfg['init_from']}", flush=True)
    esc = m.toy.energy(Jt).reshape(-1, 1)
    ev = C.Evaluator(t, J=J_eval, n_angles=d.get("n_eval_angles", 1024))
    t_off = st["t"] if st else 0.0
    logf = open(os.path.join(jdir, "log.jsonl"), "a"); t0 = time.perf_counter()
    elapsed = lambda: t_off + time.perf_counter() - t0

    def record(stage, it, loss, lam=None):
        rec = dict(stage=stage, it=it, loss=loss, lam=lam, t=round(elapsed(), 2), **ev.quick(m))
        if spherical:
            rec.update(spherical_diagnostics(m, t, ev.J, ev.th))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        extra = f" |L|-wander={rec['L_wander_median']:.2e} inv={rec['invariance_rms']:.2e}" if spherical else ""
        print(f"[{cfg['name']}] {stage} it={it} loss={loss:.3e} held-out={rec['eval_mean']:.3e} max={rec['eval_max']:.3e}{extra} t={rec['t'] / 3600:.2f}h", flush=True)

    print(f"{cfg['name']}: {C.n_params(m)} parameters, {aa.shape[0]} tori x {aa.shape[1]} angles", flush=True)
    A = cfg.get("adam", {})
    if not st:
        record("init", 0, float(C.residuals(m, aa, t, esc).pow(2).mean()))
        if A.get("steps", 0):
            C.train_adam(m, aa, t, A["steps"], A.get("lr", 1e-3), esc, record=lambda k, l: record("adam", k, l), every=A.get("every", 100))
    L = cfg.get("lm", {})
    why, n_done = None, (st["it"] if st else 0)
    save = lambda it, lam: atomic_save(dict(state=m.state_dict(), stage="lm", it=it, lam=lam, t=elapsed()), st_path)
    if L.get("max_iters", 0) and n_done < L["max_iters"]:
        wall = L.get("wall_hours"); wall = max(wall * 3600 - t_off, 1.0) if wall else None
        lam0 = st["lam"] if st and st.get("lam") else L.get("lam0", 1e-3)
        if "invariance_beta" in L:
            isc = C.invariance_scale(m, aa)
            n, why, lam = C.train_lm_inv(m, aa, t, esc, isc, L["invariance_beta"], L["max_iters"] - n_done, lam=lam0,
                                         record=lambda k, l, lam: record("lm", k, l, lam), every=L.get("every", 10), wall_cap=wall,
                                         chunk=L.get("chunk", 4), inv_angles=L.get("inv_angles"), ckpt=save,
                                         ckpt_every=L.get("ckpt_every", 10), it0=n_done)
        else:
            n, why, lam = C.train_lm(m, aa, t, esc, L["max_iters"] - n_done, lam=lam0, geodesic=L.get("geodesic", True),
                                     record=lambda k, l, lam: record("lm", n_done + k, l, lam), every=L.get("every", 10), wall_cap=wall,
                                     chunk=L.get("chunk", 8))
            save(n_done + n, lam)
    torch.save(dict(state=m.state_dict(), config=cfg), os.path.join(jdir, "model.pt"))
    res = dict(config=cfg, n_params=C.n_params(m), lm_stop=why, wall_s=elapsed(), final=ev.quick(m), action_freq=ev.action_freq(m))
    if spherical:
        res["final"].update(spherical_diagnostics(m, t, ev.J, ev.th))
    json.dump(res, open(os.path.join(jdir, "result.json"), "w"))
    open(os.path.join(jdir, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    f = res["final"]
    print(f"DONE {cfg['name']}: held-out {f['eval_mean']:.3e} max {f['eval_max']:.3e}; torus-mean E vs exact H(J) "
          f"max {max(res['action_freq']['rel_E_err']):.1e}; frequencies max {max(res['action_freq']['rel_freq_err']):.1e}", flush=True)


def _exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)  # state.pt is atomic, so the job resumes from its last checkpoint


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, _exit_on_signal); signal.signal(signal.SIGTERM, _exit_on_signal)
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    main(json.load(open(a.config)), a.out)
