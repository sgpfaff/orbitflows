"""Run (or resume) one job:  python run.py jobs/<batch>.jsonl --line K [--out results]

Job directory <out>/<batch>/<name>/ holds
  state.pt    float64 weights + position (stage, iterations done, LM damping, cumulative wall time);
              written atomically at every stage boundary and every `every` LM steps
  log.jsonl   one record per log point (held-out / training errors, z_max bands)
  result.json final metrics (per-torus errors, z_max, action/frequency errors, origin image)
  DONE        marker; a finished job is skipped
Killing the process at any point loses at most `every` LM steps (or the current Adam stage); rerunning
the same command resumes from state.pt.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import platform
import socket
import sys
import time
import traceback

import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return name


def main(spec, out_root):
    cfg = spec
    jdir = os.path.join(out_root, cfg["batch"], cfg["name"])
    os.makedirs(jdir, exist_ok=True)
    if os.path.exists(os.path.join(jdir, "DONE")):
        print("already done:", cfg["name"]); return
    json.dump(cfg, open(os.path.join(jdir, "config.json"), "w"), indent=1)
    t_session = time.perf_counter()

    target = C.make_target(cfg)
    d = cfg["data"]
    if target.name == "mw":
        J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, target.phi)[0])
        J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, target.phi)[0])
    else:
        J_lo, J_hi = d["J_lo"], d["J_hi"]
    tw = cfg["toy"].get("omega", "nu0")
    omega0 = target.nu0 if tw == "nu0" else float(tw)
    J_train = C.j_grid(d["n_orbits"], J_lo, J_hi, omega0, d.get("spacing", "asinh"), d.get("a0", 0.03), offset=0.0)
    aa_train = C.aa_points(J_train, C.angles(d["n_steps"], 0.5))
    nO, nS = aa_train.shape[:2]
    ev = C.Evaluator(cfg, target, J_lo, J_hi, omega0, J_train, nS)

    def fresh(dtype=C.F64, device="cpu"):
        return C.build_model(cfg["model"], cfg["toy"], omega0, J_lo, J_hi, cfg.get("seed", 0), dtype, device)

    st_path = os.path.join(jdir, "state.pt")
    st = torch.load(st_path, weights_only=False) if os.path.exists(st_path) else None
    model = fresh()
    if st:
        model.load_state_dict(st["state"])
        print(f"resuming {cfg['name']} at stage {st['stage']} it {st['it']} (wall so far {st['t_total']/3600:.2f} h)")
    stage0, it_done, lam = (st["stage"], st["it"], st.get("lam")) if st else (0, 0, None)
    t_off = st["t_total"] if st else 0.0
    stage_info = st.get("stage_info", []) if st else []
    logf = open(os.path.join(jdir, "log.jsonl"), "a")
    t_now = lambda: t_off + time.perf_counter() - t_session

    def save(stage, it, lam_):
        atomic_save(dict(state=C.state_f64(model), stage=stage, it=it, lam=lam_, t_total=t_now(),
                         stage_info=stage_info), st_path)

    def as_f64(m):
        if next(m.parameters()).dtype == C.F64 and next(m.parameters()).device.type == "cpu":
            return m
        m64 = fresh(); m64.load_state_dict(C.state_f64(m)); return m64

    stages = cfg["stages"]
    for si in range(stage0, len(stages)):
        S = stages[si]
        dev = resolve_device(S.get("device", "cpu")); dt = getattr(torch, S.get("dtype", "float64"))
        sd = C.state_f64(model)
        model = fresh(dt, dev)
        model.load_state_dict({k: v.to(dt) for k, v in sd.items()})
        aa_dev = aa_train.to(dt).to(dev)
        label = f"{S['kind']}-{dev}-{S.get('dtype', 'float64')}"

        def record(it, loss, lr_or_lam, label=label):
            rec = dict(t=round(t_now(), 2), stage=si, label=label, it=it, loss=loss, lr=lr_or_lam,
                       **ev.quick(as_f64(model)))
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"[{cfg['name']}] {label} it={it} loss={loss:.3e} held-out={rec['eval_mean']:.3e} "
                  f"max={rec['eval_max']:.3e} t={rec['t']/60:.1f}min", flush=True)

        t_stage = time.perf_counter()
        if S["kind"] == "adam":
            save(si, 0, None)  # an interrupted Adam stage restarts from here
            C.train_adam(model, aa_dev, target, nO, nS, S["steps"], S.get("lr", 1e-3), record, S.get("every", 100))
            stage_info.append(dict(stage=si, label=label, iters=S["steps"], why="done", wall=time.perf_counter() - t_stage))
        else:
            lam0 = lam if (lam is not None and it_done > 0) else S.get("lam0", 1e-3)
            wall = S.get("wall_hours"); wall = wall * 3600 if wall else None
            n, why, lam = C.train_lm(model, aa_dev, target, nO, nS, max_iters=S["max_iters"] - it_done, lam=lam0,
                                     record=record, ckpt=lambda it, l: save(si, it, l), every=S.get("every", 25),
                                     wall_cap=wall, stall_window=S.get("stall_window"), stall_rtol=S.get("stall_rtol"),
                                     chunk_orbits=S.get("chunk_orbits", 16), it0=it_done)
            stage_info.append(dict(stage=si, label=label, iters=it_done + n, why=why, wall=time.perf_counter() - t_stage))
        it_done, lam = 0, None
        save(si + 1, 0, None)

    m64 = as_f64(model)
    res = dict(config=cfg, n_params=C.n_params(m64), J_range=[J_lo, J_hi], omega0=omega0, stages=stage_info,
               wall_total_s=t_now(), host=socket.gethostname(), torch=torch.__version__, python=platform.python_version(),
               final=ev.full(m64))
    json.dump(res, open(os.path.join(jdir, "result.json"), "w"))
    open(os.path.join(jdir, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    f = res["final"]
    print(f"DONE {cfg['name']}: held-out mean {f['eval_mean']:.3e} max {f['eval_max']:.3e} train {f['train_mean']:.3e} "
          f"params {res['n_params']} wall {res['wall_total_s']/3600:.2f} h", flush=True)


def _exit_on_signal(signum, _frame):
    # Explicit handlers: a process started in the background from a non-interactive shell (nohup, &)
    # inherits SIGINT=ignore, which Python honours; this makes Ctrl-C / kill stop the job either way.
    # state.pt is written atomically, so the job resumes from its last checkpoint.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, _exit_on_signal)
    signal.signal(signal.SIGTERM, _exit_on_signal)
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs"); ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    spec = json.loads(open(a.jobs).read().splitlines()[a.line])
    try:
        main(spec, a.out)
    except Exception:
        traceback.print_exc(); sys.exit(1)
