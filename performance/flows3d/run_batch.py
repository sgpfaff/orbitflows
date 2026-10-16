"""Run several run3d.py configs in parallel:  python run_batch.py configs/sweep_beta/*.json --workers 6 --threads 3

Finished jobs (DONE) are skipped and interrupted ones resume from state.pt, so rerun the same command after a session
ends. Each job's output goes to results/<name>/stdout.log; a status table is printed every --status seconds and written
to results/batch_status.txt.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def last_record(jdir):
    p = os.path.join(jdir, "log.jsonl")
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as fh:
        try:
            fh.seek(-4000, os.SEEK_END)
        except OSError:
            fh.seek(0)
        for ln in reversed(fh.read().decode(errors="ignore").strip().splitlines()):
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                continue
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results")); ap.add_argument("--status", type=int, default=120)
    a = ap.parse_args()
    jobs = [(p, json.load(open(p))["name"]) for p in a.configs]
    jdir = lambda n: os.path.join(a.out, n)
    pending = [j for j in jobs if not os.path.exists(os.path.join(jdir(j[1]), "DONE"))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} done, {len(pending)} to run with {a.workers} workers x {a.threads} threads", flush=True)
    env = dict(os.environ, NTHREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               OPENBLAS_NUM_THREADS=str(a.threads), KMP_DUPLICATE_LIB_OK="TRUE")
    running = {}

    def stop_all(*_):
        print("\nstopping (jobs resume from their last checkpoint on the next run)", flush=True)
        for p in running.values():
            p.send_signal(signal.SIGTERM)
        time.sleep(3)
        for p in running.values():
            if p.poll() is None:
                p.kill()
        sys.exit(130)
    signal.signal(signal.SIGINT, stop_all); signal.signal(signal.SIGTERM, stop_all)
    t_status = 0.0
    while pending or running:
        while pending and len(running) < a.workers:
            path, name = pending.pop(0); os.makedirs(jdir(name), exist_ok=True)
            log = open(os.path.join(jdir(name), "stdout.log"), "a")
            running[name] = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "run3d.py"), path, "--out", a.out], stdout=log, stderr=subprocess.STDOUT, env=env)
            print(f"{time.strftime('%H:%M:%S')} start {name}", flush=True)
        for name in list(running):
            if running[name].poll() is not None:
                ok = os.path.exists(os.path.join(jdir(name), "DONE"))
                print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED (see stdout.log)'} {name}", flush=True)
                del running[name]
        if time.time() - t_status > a.status:
            t_status = time.time(); rows = []
            for _, name in jobs:
                r = last_record(jdir(name))
                state = "done" if os.path.exists(os.path.join(jdir(name), "DONE")) else ("running" if name in running else "pending")
                rows.append(f"{name:22s} {state:8s} it={r.get('it', ''):<6} held-out={r.get('eval_mean', float('nan')):.2e} "
                            f"L-wander={r.get('L_wander_median', float('nan')):.2e} inv={r.get('invariance_rms', float('nan')):.2e} t={r.get('t', 0) / 3600:.2f}h")
            table = f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n" + "\n".join(rows)
            print(table, flush=True); open(os.path.join(a.out, "batch_status.txt"), "w").write(table + "\n")
        time.sleep(5)
    print("batch finished", flush=True)


if __name__ == "__main__":
    main()
