"""Run the mini teacher-student pilot on one CPU node: per size, train the teacher, then its students in parallel.

    MINI_RES=results/mini python mini_queue.py --sizes 4x8,4x16,8x8,8x16,8x32,16x16 --seeds 12 \
        --teacher-cap 8000 --teacher-wall 1800 --student-cap 30000 --student-wall 4500 --workers 90 --threads 2

Each task is one `python mini_lab.py ...` subprocess with NTHREADS threads; a size's students start as soon as its
teacher file exists. Logs go to $MINI_RES/logs/<task>.log. On SIGTERM the children are terminated (finished runs keep
their result.json; mini_lab student skips runs that already have one).
"""
import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4x8,4x16,8x8,8x16,8x32,16x16")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--teacher-cap", type=int, default=8000); ap.add_argument("--teacher-wall", type=float, default=1800)
    ap.add_argument("--student-cap", type=int, default=30000); ap.add_argument("--student-wall", type=float, default=4500)
    ap.add_argument("--workers", type=int, default=90); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    res = os.environ["MINI_RES"]
    os.makedirs(os.path.join(res, "logs"), exist_ok=True)
    env = dict(os.environ, NTHREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               KMP_DUPLICATE_LIB_OK="TRUE")
    sizes = [tuple(int(x) for x in s.split("x")) for s in a.sizes.split(",")]
    pending = [("teacher", L, W, None) for L, W in sizes]
    pending += [("student", L, W, s) for L, W in sizes for s in range(a.seeds)]
    running = {}

    def ready(task):
        kind, L, W, _ = task
        return kind == "teacher" or os.path.exists(os.path.join(HERE, "teachers", f"mini_{L}x{W}.pt"))

    def launch(task):
        kind, L, W, s = task
        name = f"{kind}_{L}x{W}" + ("" if s is None else f"_s{s}")
        if kind == "teacher":
            cmd = [sys.executable, "mini_lab.py", "teacher", str(L), str(W), "--cap", str(a.teacher_cap), "--wall", str(a.teacher_wall)]
        else:
            cmd = [sys.executable, "mini_lab.py", "student", str(L), str(W), str(s), "--cap", str(a.student_cap), "--wall", str(a.student_wall)]
        log = open(os.path.join(res, "logs", name + ".log"), "w")
        running[name] = (subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
        print(f"{time.strftime('%H:%M:%S')} start {name}", flush=True)

    def stop(*_):
        for p, log, _ in running.values():
            p.terminate()
        sys.exit(1)
    signal.signal(signal.SIGTERM, stop)

    while pending or running:
        for name, (p, log, t0) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[name]
                print(f"{time.strftime('%H:%M:%S')} end {name} (exit {p.returncode}, {(time.time() - t0) / 60:.1f} min)", flush=True)
                if name.startswith("teacher") and p.returncode != 0:  # no teacher, no students of that size
                    L, W = name.split("_")[1].split("x")
                    pending[:] = [t for t in pending if not (t[0] == "student" and (t[1], t[2]) == (int(L), int(W)))]
        for task in list(pending):
            if len(running) >= a.workers:
                break
            if ready(task):
                pending.remove(task); launch(task)
        time.sleep(5)
    print(f"{time.strftime('%H:%M:%S')} all done", flush=True)


if __name__ == "__main__":
    main()
