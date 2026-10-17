"""Work through one or more job files with W parallel workers of T threads each.

    python queue_runner.py jobs/recipe.jsonl [jobs/ladder.jsonl ...] --workers 4 --threads 8

Run it inside `screen` / `tmux`. Finished jobs (DONE marker) are skipped and interrupted jobs resume
from their last checkpoint, so Ctrl-C, a dropped session or an expired allocation are all safe:
rerun the same command. Each job's output goes to <out>/<batch>/<name>/stdout.log; a status table is
printed every --status seconds (and written to <out>/queue_status.txt).
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
        return None
    with open(p, "rb") as fh:
        try:
            fh.seek(-4000, os.SEEK_END)
        except OSError:
            fh.seek(0)
        lines = fh.read().decode(errors="ignore").strip().splitlines()
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except json.JSONDecodeError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--status", type=int, default=120)
    ap.add_argument("--only", default=None, help="substring filter on job names")
    ap.add_argument("--pin", action="store_true",
                    help="pin each worker slot to its own block of --threads cores (Linux; keeps a job's threads on one L3)")
    a = ap.parse_args()

    todo = []
    for jf in a.jobs:
        for k, ln in enumerate(open(jf).read().splitlines()):
            if ln.strip():
                spec = json.loads(ln)
                if a.only and a.only not in spec["name"]:
                    continue
                todo.append((jf, k, spec))
    jdir = lambda s: os.path.join(a.out, s["batch"], s["name"])
    pending = [t for t in todo if not os.path.exists(os.path.join(jdir(t[2]), "DONE"))]
    print(f"{len(todo)} jobs, {len(todo) - len(pending)} already done, {len(pending)} to run "
          f"with {a.workers} workers x {a.threads} threads", flush=True)

    env = dict(os.environ, NTHREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               OPENBLAS_NUM_THREADS=str(a.threads), KMP_DUPLICATE_LIB_OK="TRUE")
    running, failed = {}, []
    cpus = sorted(os.sched_getaffinity(0)) if a.pin else []
    free_slots = list(range(a.workers))

    def pin_to(slot):
        cores = cpus[slot * a.threads:(slot + 1) * a.threads]
        if len(cores) < a.threads:
            return None
        return lambda: os.sched_setaffinity(0, cores)

    def stop_all(*_):
        print("\nstopping workers (jobs resume from their last checkpoint on the next run)", flush=True)
        for p, *_ in running.values():
            p.send_signal(signal.SIGTERM)
        time.sleep(3)
        for p, *_ in running.values():
            if p.poll() is None:
                p.kill()
        sys.exit(130)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    t_status = 0.0
    while pending or running:
        while pending and len(running) < a.workers:
            jf, k, spec = pending.pop(0)
            os.makedirs(jdir(spec), exist_ok=True)
            log = open(os.path.join(jdir(spec), "stdout.log"), "a")
            slot = free_slots.pop(0)
            p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "run.py"), jf, "--line", str(k), "--out", a.out],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=pin_to(slot) if a.pin else None)
            running[spec["name"]] = (p, spec, slot)
            print(f"{time.strftime('%H:%M:%S')} start {spec['name']}", flush=True)
        for name in list(running):
            p, spec, slot = running[name]
            if p.poll() is not None:
                free_slots.append(slot)
                ok = p.returncode == 0 and os.path.exists(os.path.join(jdir(spec), "DONE"))
                print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED (see stdout.log)'} {name}", flush=True)
                if not ok:
                    failed.append(name)
                del running[name]
        if time.time() - t_status > a.status:
            t_status = time.time()
            rows = []
            for _, _, s in todo:
                d = jdir(s)
                state = "done" if os.path.exists(os.path.join(d, "DONE")) else (
                    "running" if s["name"] in running else ("FAILED" if s["name"] in failed else "pending"))
                r = last_record(d) or {}
                rows.append(f"{s['name']:44s} {state:8s} {r.get('label', ''):22s} it={r.get('it', ''):<6} "
                            f"held-out={r.get('eval_mean', float('nan')):.2e} t={r.get('t', 0) / 3600:.2f}h")
            table = f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n" + "\n".join(rows)
            print(table, flush=True)
            open(os.path.join(a.out, "queue_status.txt"), "w").write(table + "\n")
        time.sleep(5)
    print(f"queue finished; {len(failed)} failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
