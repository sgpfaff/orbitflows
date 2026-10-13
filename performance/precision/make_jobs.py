"""Write job files for the precision campaign:  python make_jobs.py --profile cpu|gpu [--toy fixed|learned] [--odd]

profile cpu : everything float64 on the CPU (Trillium CPU nodes, geir CPU). Adam (cosine, 3000 steps), then
              Levenberg-Marquardt until it converges (<stall_rtol improvement over stall_window steps),
              stops descending, or hits wall_hours.
profile gpu : Adam and a first LM stage in float32 on the GPU (CUDA or Apple MPS; H stays float64 on the
              host), then the float64 LM polish on the CPU. Float32 LM stalls near 1e-4; the polish continues.

Batches
  recipe   : 16 layers x 64 tanh, z_max 5 pc - 8.3 kpc, 3 seeds x toy {fixed nu0, learned omega(J)} x
             conditioner {standard, odd}  -> 12 jobs. Settles learned omega and odd conditioners with error bars.
  capacity : layers {4, 8, 16, 32, 64} x width {16, 32, 64} x 3 seeds -> 45 jobs (<= 12.4k parameters, so
             the LM normal equations stay <= ~1.2 GB). Use the recipe winner via --toy / --odd.
  anchor   : learned omega(J) pinned to nu0 at the smallest training action (value, or value and slope),
             standard conditioners, 3 seeds -> 6 jobs. The recipe batch found learned omega best above ~1 kpc but
             ~10x worse near the midplane, where omega drifted to ~1.2 instead of nu0; this tests the fix.
             LM wall cap 68 h, to fit one 72 h Trillium OnDemand session.
  ladder   : (A) SHO with the exact toy: the answer is the identity, which the flow represents exactly;
             (B) SHO with a wrong toy frequency: the answer is a linear rescaling, which tanh shears reach
                 only in a limit (small inner weights, large outer ones);
             (C) teacher-student: the answer is a fixed 4-layer tanh flow, which the student contains exactly.
             3 seeds each -> 9 jobs. A and C have zero approximation error, so they measure how far
             optimisation alone gets (machine precision?).
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MW_DATA = dict(zmin_pc=5.0, zmax_kpc=8.32, n_orbits=96, n_steps=128, spacing="asinh", a0=0.03)
LADDER_DATA = dict(J_lo=1.5 * 0.025 ** 2 / 2, J_hi=1.5 * 0.75 ** 2 / 2, n_orbits=96, n_steps=128, spacing="asinh",
                   a0=0.1)
BASE_MODEL = dict(n_layers=16, width=64, depth=2, act="tanh", bias_init="torch", gate_std=1e-3, odd=False)


def stages(profile, polish_hours=60.0, stall_window=300, stall_rtol=1e-3):
    lm64 = dict(kind="lm", device="cpu", dtype="float64", max_iters=200000, wall_hours=polish_hours,
                stall_window=stall_window, stall_rtol=stall_rtol, every=25)
    if profile == "cpu":
        return [dict(kind="adam", device="cpu", dtype="float64", steps=3000, lr=1e-3), lm64]
    return [dict(kind="adam", device="auto", dtype="float32", steps=3000, lr=1e-3),
            dict(kind="lm", device="auto", dtype="float32", max_iters=20000, wall_hours=2.0, every=50), lm64]


def job(name, batch, profile, model=None, toy=None, data=None, target="mw", seed=0, **kw):
    return dict(name=name, batch=batch + "_v2", target=target, seed=seed, loss_norm="toy",
                model=dict(BASE_MODEL, **(model or {})),
                toy=toy or dict(mode="fixed", omega="nu0"), data=data or MW_DATA, eval=dict(n_orbits=200, n_steps=512),
                stages=kw.pop("stages", stages(profile)), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--toy", choices=["fixed", "learned"], default="fixed", help="toy map for the capacity batch")
    ap.add_argument("--odd", action="store_true", help="odd conditioners for the capacity batch")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    p = a.profile
    batches = {}

    batches["recipe"] = [
        job(f"rec_{toy}_{'odd' if odd else 'std'}_s{s}", "recipe", p, model=dict(odd=odd),
            toy=dict(mode=toy, omega="nu0", n_cheb=6), seed=s)
        for toy in ("fixed", "learned") for odd in (False, True) for s in range(3)]

    batches["anchor"] = [
        job(f"anc_{tag}_s{s}", "anchor", p, toy=dict(mode="learned", omega="nu0", n_cheb=6, anchor=anc), seed=s,
            stages=stages(p, polish_hours=68.0))
        for anc, tag in (("value", "val"), ("value_slope", "vs")) for s in range(3)]

    toy = dict(mode=a.toy, omega="nu0", n_cheb=6)
    batches["capacity"] = [
        job(f"cap_L{L}_w{w}_{a.toy}{'_odd' if a.odd else ''}_s{s}", "capacity", p,
            model=dict(n_layers=L, width=w, odd=a.odd), toy=toy, seed=s)
        for L in (4, 8, 16, 32, 64) for w in (16, 32, 64) for s in range(3)]

    lad = lambda: stages(p, stall_window=1000, stall_rtol=1e-5)
    batches["ladder"] = (
        [job(f"lad_A_sho_identity_s{s}", "ladder", p, target="sho", omega_target=1.0, toy=dict(mode="fixed", omega=1.0),
             data=LADDER_DATA, seed=s, stages=lad()) for s in range(3)]
        + [job(f"lad_B_sho_linear_s{s}", "ladder", p, target="sho", omega_target=1.0, toy=dict(mode="fixed", omega=1.5),
               data=LADDER_DATA, seed=s, stages=lad()) for s in range(3)]
        + [job(f"lad_C_teacher_s{s}", "ladder", p, target="teacher", toy=dict(mode="fixed", omega=1.5),
               teacher=dict(omega=1.5, n_layers=4, width=16, gate=0.25, seed=123), data=LADDER_DATA, seed=s,
               stages=lad()) for s in range(3)])

    for b, jobs in batches.items():
        path = os.path.join(HERE, "jobs", f"{b}_{p}.jsonl")
        with open(path, "w") as fh:
            for j in jobs:
                fh.write(json.dumps(j) + "\n")
        print(f"{path}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
