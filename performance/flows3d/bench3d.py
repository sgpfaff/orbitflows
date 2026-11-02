"""Time the pieces of one 3D LM step on a device:  python bench3d.py [--device cuda] [--out results/bench3d_<host>.json]

For the depth experiment's isochrone setup (125 tori x 256 angles, invariance rows on 64 angles), both depths
(16 x 32, 32 x 16) and both weights (beta = 0, 0.01), at several Jacobian chunk sizes (tori per chunk): the normal
equations, one residual evaluation, one J^T u (geodesic acceleration), and 3 full geodesic LM steps. Also checks the
device against the CPU on the first normal equations (relative difference of J^T r and J^T J).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import platform
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core3d as C  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(dev, f, *a):
    sync(dev); t0 = time.perf_counter(); out = f(*a); sync(dev)
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="auto"); ap.add_argument("--out", default=None)
    ap.add_argument("--chunks", default="4,16,32,64"); ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    dev = C.resolve_device(a.device); torch.set_num_threads(int(os.environ.get("NTHREADS", 8)))
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or "cpu"
    print(f"device {dev} ({name}), torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)
    base = json.load(open(os.path.join(HERE, "configs/depth/depth_d16_b0p01_s0.json"))); d = base["data"]
    t_cpu = C.make_target(base["target"]); t = C.target_on(t_cpu, dev)
    J = C.spherical_action_grid(d["n_per_axis"], *d["Jr"], *d["L"], d["cosi_max"])
    aa = C.aa_points(J, C.angle_samples(d["n_angles"], 0))
    models = {"16x32": dict(n_lin=2, n_blocks=8, width=32, gate_std=1e-3), "32x16": dict(n_lin=2, n_blocks=16, width=16, gate_std=1e-3)}
    ck = torch.load(os.path.join(HERE, "checkpoints/s3_isochrone.pt"), weights_only=False)["state"]
    results = []
    for mtag, mc in models.items():
        for beta in (0.0, 0.01):
            for chunk in [int(c) for c in a.chunks.split(",")]:
                m = C.build_model(mc, base["toy"], device=dev)
                if mtag == "16x32":
                    m.load_state_dict({k: v.to(dev) for k, v in ck.items()})  # a trained state: realistic acceptance
                aad = aa.to(dev); es = m.toy.energy(J.to(dev)).reshape(-1, 1); isc = C.invariance_scale(m, aad)
                unflat, resid, normal_eqs = C.lm_residual_system(m, aad, t, es, isc, beta, chunk, 64)
                v = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
                try:
                    normal_eqs(v)  # warm-up (kernel compilation, allocator)
                    (r, g, A), t_ne = timed(dev, normal_eqs, v)
                    _, t_r = timed(dev, resid, v)
                    _, t_j = timed(dev, resid.jtv, v, r)
                    check = None
                    if dev.type != "cpu" and chunk == 4:  # the device against the CPU, same model
                        mc_ = C.build_model(mc, base["toy"]); mc_.load_state_dict({k: p.detach().cpu() for k, p in m.state_dict().items()})
                        _, _, ne_c = C.lm_residual_system(mc_, aa, t_cpu, es.cpu(), isc.cpu(), beta, chunk, 64)
                        rc, gc, Ac = ne_c(v.cpu())
                        check = dict(g=float((g.cpu() - gc).norm() / gc.norm()), A=float((A.cpu() - Ac).norm() / Ac.norm()))
                    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None
                    _, t_steps = timed(dev, C.train_lm_inv, m, aad, t, es, isc, beta, a.steps, 1e-6, None, 10, chunk, None, 64, None, 10, 0, True)
                    row = dict(model=mtag, beta=beta, chunk=chunk, normal_eqs_s=t_ne, resid_s=t_r, jtv_s=t_j, lm_step_s=t_steps / a.steps,
                               rows=r.numel(), params=v.numel(), check_vs_cpu=check, peak_gpu_gb=peak)
                except torch.cuda.OutOfMemoryError:
                    row = dict(model=mtag, beta=beta, chunk=chunk, error="out of memory"); torch.cuda.empty_cache()
                results.append(row)
                print(json.dumps(row), flush=True)
    out = a.out or os.path.join(HERE, "results", f"bench3d_{platform.node().split('.')[0]}_{dev.type}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(device=str(dev), name=name, torch=torch.__version__, threads=torch.get_num_threads(), results=results), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
