"""Time float64 Levenberg-Marquardt iterations on a GPU against the CPU, on real trained checkpoints.

    python bench_lm_device.py RESULTS_DIR [--iters 8] [--cpu-threads 6] [--chunks 16,64] [--out bench.json]

For each model (the final checkpoints of the third bundle), the training set is rebuilt exactly as run.py does, and
core.train_lm runs a few iterations from lam 1e-11 (geodesic, retry_lam 1e-11), once with the model on CUDA in
float64 (for each chunk size: tori per Jacobian chunk) and once on the CPU with the given number of threads. The
first iteration of each run is a warm-up and is not timed. The loss after the same iterations is compared between
devices, which checks that the GPU path computes the same steps.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

MODELS = ["ctrl3_v2/c3_L32w32_lrnA_std_a256_s0", "depth_v2/dep_L64_w32_s1",
          "ctrl3_v2/c3_L48w32_fix_a512_s1", "deep3_v2/d3_L64w32_lrnA_a512_s0"]


def setup(jd, device):
    cfg = json.load(open(os.path.join(jd, "config.json"))); t = C.make_target(cfg); d = cfg["data"]
    J_lo = float(C.J_of_zmax(d["zmin_pc"] / 1000 / C.RO_KPC, t.phi)[0]); J_hi = float(C.J_of_zmax(d["zmax_kpc"] / C.RO_KPC, t.phi)[0])
    w0 = t.nu0
    J_train = C.train_grid(d, J_lo, J_hi, w0)
    aa = C.aa_points(J_train, C.angles(d["n_steps"], 0.5)); nO, nS = aa.shape[:2]
    es = (w0 * J_train).reshape(-1, 1)
    m = C.build_model(cfg["model"], cfg["toy"], w0, J_lo, J_hi, cfg.get("seed", 0), C.F64, device)
    m.load_state_dict({k: v.to(device) for k, v in torch.load(os.path.join(jd, "state.pt"), weights_only=False)["state"].items()})
    return cfg, t, m, aa.to(device), nO, nS, es


def run(jd, device, iters, chunk):
    cfg, t, m, aa, nO, nS, es = setup(jd, device)
    stamps = []

    def rec(it, loss, lam):
        if device == "cuda":
            torch.cuda.synchronize()
        stamps.append((time.perf_counter(), loss, lam))
    t0 = time.perf_counter()
    n, why, _ = C.train_lm(m, aa, t, nO, nS, max_iters=iters, lam=1e-11, record=rec, every=1, escale=es, geodesic=True,
                           retry_lam=1e-11, chunk_orbits=chunk)
    times = [b[0] - a[0] for a, b in zip(stamps[:-1], stamps[1:])]   # iterations 2..n (the first is a warm-up)
    return dict(model=cfg["name"], L=cfg["model"]["n_layers"], points=nO * nS, params=C.n_params(m), device=device,
                chunk=chunk, iters=n, why=why, s_per_iter=sum(times) / max(len(times), 1), first_s=stamps[0][0] - t0,
                loss_end=stamps[-1][1], lam_end=stamps[-1][2], losses=[x[1] for x in stamps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cpu-threads", type=int, default=6); ap.add_argument("--chunks", default="16,64")
    ap.add_argument("--out", default="bench_lm_device.json")
    a = ap.parse_args()
    rows = []
    gpu = torch.cuda.is_available()
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if gpu else 'none'}", flush=True)
    for rel in MODELS:
        jd = os.path.join(a.results, rel)
        if gpu:
            for ch in [int(x) for x in a.chunks.split(",")]:
                r = run(jd, "cuda", a.iters, ch); rows.append(r)
                print(f"{r['model']:28s} {r['points']:6d} pts {r['params']:5d} par  cuda chunk {ch:3d}: {r['s_per_iter']:7.3f} s/iter "
                      f"(first {r['first_s']:.1f} s)  loss {r['loss_end']:.6e}  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(a.cpu_threads)
        r = run(jd, "cpu", min(a.iters, 4), 16); rows.append(r)
        g = [x for x in rows if x['model'] == r['model'] and x['device'] == 'cuda']
        agree = max(abs(x['losses'][len(r['losses']) - 1] - r['loss_end']) / r['loss_end'] for x in g) if g else float('nan')
        print(f"{r['model']:28s} {r['points']:6d} pts {r['params']:5d} par  cpu {a.cpu_threads:2d} threads: {r['s_per_iter']:7.3f} s/iter "
              f"loss {r['loss_end']:.6e}  (GPU loss after the same iterations differs by {agree:.1e} relative)", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
