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
  mwteacher: fresh 16 x 64 students (geodesic and plain LM, 3 seeds each) trained on an MW teacher, the Hamiltonian of the best MW model
             (teachers/mw_fixed_std_s1.pt, 1.2e-5 from the real MW tori), same data, toy and recipe as the MW runs,
             3 seeds, 68 h LM cap. The student can represent the teacher exactly, so this measures how far the
             optimiser gets on a map of MW's complexity: if students go far below 1e-5, the MW error is
             approximation and a capacity scan is meaningful above that floor; if they stall near 1e-5, it is
             the optimiser.
  mwgeo    : the real MW target with the same optimiser (fixed nu0, standard conditioners, geodesic LM), 3 seeds,
             68 h: the comparison for mwteacher, and the next best-model attempt.
  ladder2  : rung C again, now exactly realisable (the old runs' toy frequency was float32-rounded, 1.2e-8 off
             the teacher's, which the student had to absorb as a tiny linear squeeze), standard vs geodesic LM,
             3 seeds each, 68 h, plus rung B with geodesic LM. The precision-track baseline.
  anchor_geo, toygeo : the toy-frequency study again with geodesic LM, to compare with mwgeo (fixed nu0):
             anchor_geo = pinned omega (value, value_slope) x 3 seeds; toygeo = free learned omega x 3 seeds.
  capprobe : a small capacity probe with geodesic LM, depth vs width at ~1.5k (8 x 64, 16 x 32), ~3.1k (8 x 128,
             32 x 32, and mwgeo's 16 x 64) and ~6.2k parameters (16 x 128, 32 x 64), 2 seeds.
             Read it together with mwteacher: it only measures the model if the students show the optimiser
             is not the limit at these error levels.
  Second node bundle (after the stall diagnosis: the 32-layer models were limited by 128 angles per orbit in the
  outer disc; denser angles restart training). All geodesic LM, 68 h.
  cont32   : the four 32-layer checkpoints of capprobe (32 x 32 and 32 x 64, seeds 0-1) continued at 256 and 512 angles.
  fresh32  : 32 x 32 from scratch at 256 angles, 3 seeds.
  depth    : 48 x 32, 64 x 32 and 48 x 64 at 256 angles, 2 seeds: does depth keep paying once data is not the limit?
  mwt32    : 32 x 32 students on a 32 x 32 MW teacher (the capprobe model fine-tuned at 256 angles), 256 angles.
  mwt16deep: 32 x 64 students on the 16 x 64 MW teacher, same data as the 16-layer students (128 angles): both
             depths represent it exactly, so any gain over the 16-layer students is depth easing optimisation.
  layout   : tori evenly spaced in ln J below 150 pc (24 tori replacing 16; 104 in all) and 256 angles, for the
             fixed toy and learned omega(J) written in ln J (free) or in amplitude (free; pinned value+slope),
             at 16 x 64 (1 seed) and 32 x 32 (2 seeds).
  Third node bundle (the second found: depth keeps paying, 48 x 32 from scratch reached 3.2e-8; the learned
  frequency written in amplitude is the best toy, 1.5-4x below fixed nu0 at 32 layers). All geodesic LM, 68 h caps.
  deep3    : 48 x 32 and 64 x 32 with the learned frequency (amplitude series) on the ln J layout, 256 and 512 angles.
  ctrl3    : controls: 48 x 32 fixed nu0 at 512 angles (angles at 48 layers); the learned frequency on the standard
             layout at 32, 48 and 64 layers (separates the toy from the layout).
  resume3  : the best second-bundle jobs, resumed from their checkpoints (same names, so they continue with their saved
             damping and the remaining LM time): 48 x 32 and 64 x 32 (depth), the 32 x 32 layout runs with fixed and
             learned frequency, and the 32 x 32 continuations at 512 angles.
  Fourth bundle (the third: most runs stopped on one failed LM iteration while still improving, and a restart
  from the usual damping continues them).
  cont4    : every third-bundle job continued from its final checkpoint (same names, results in cont4_v2/), LM from
             lam 1e-11 with retry_lam 1e-11: an iteration that finds no descent step restarts its damping sweep
             from 1e-11 before the run stops. Jobs that a restart from 1e-11 could not move, or moved by < 5e-5 per
             step, are left out (jobs/cont4_skip.json): 13 jobs.
  fourier1 : 16 x 64 students of the 16-layer MW teacher with Fourier-term conditioners (12 frequencies, 1-100), from
             scratch on H100s, seeds 0-1 (paired with the first bundle's mwt_geo_s0/s1: same initial network).
  fourier2 : the same with amplitude-modulated Fourier terms (A_j(x) = a_j + sum_k C_jk tanh(U_k x + V_k), 8 shared
             tanh units per layer), seed 0.
  cont4g   : the same 13 jobs on one H100 each (Trillium GPU cluster), LM in float64 on CUDA; results in cont4g_v2/.
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


def stages(profile, polish_hours=60.0, stall_window=300, stall_rtol=1e-3, geodesic=False):
    lm64 = dict(kind="lm", device="cpu", dtype="float64", max_iters=200000, wall_hours=polish_hours,
                stall_window=stall_window, stall_rtol=stall_rtol, every=25, geodesic=geodesic)
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

    long_geo = lambda: stages(p, polish_hours=68.0, stall_window=1000, stall_rtol=1e-5, geodesic=True)
    long_std = lambda: stages(p, polish_hours=68.0, stall_window=1000, stall_rtol=1e-5)
    batches["mwteacher"] = [
        job(f"mwt_{tag}_s{s}", "mwteacher", p, target="teacher", teacher=dict(checkpoint="teachers/mw_fixed_std_s1.pt"),
            seed=s, stages=st())
        for st, tag in ((long_geo, "geo"), (long_std, "std")) for s in range(3)]
    batches["mwgeo"] = [job(f"mw_geo_s{s}", "mwgeo", p, seed=s, stages=long_geo()) for s in range(3)]

    batches["anchor_geo"] = [
        job(f"anc_geo_{tag}_s{s}", "anchor_geo", p, toy=dict(mode="learned", omega="nu0", n_cheb=6, anchor=anc), seed=s,
            stages=long_geo())
        for anc, tag in (("value", "val"), ("value_slope", "vs")) for s in range(3)]
    batches["toygeo"] = [
        job(f"lrn_geo_s{s}", "toygeo", p, toy=dict(mode="learned", omega="nu0", n_cheb=6), seed=s, stages=long_geo())
        for s in range(3)]
    # depth vs width at three fixed sizes (~1.5k, ~3.1k, ~6.2k parameters; mwgeo is the 16 x 64 point)
    batches["capprobe"] = [
        job(f"cap_L{L}_w{w}_geo_s{s}", "capprobe", p, model=dict(n_layers=L, width=w), seed=s, stages=long_geo())
        for L, w in ((8, 64), (16, 32), (8, 128), (32, 32), (16, 128), (32, 64)) for s in range(2)]

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

    teacher_C = dict(omega=1.5, n_layers=4, width=16, gate=0.25, seed=123)
    batches["ladder2"] = [
        job(f"lad2_C_{tag}_s{s}", "ladder2", p, target="teacher", toy=dict(mode="fixed", omega=1.5), teacher=teacher_C,
            data=LADDER_DATA, seed=s, stages=stages(p, polish_hours=68.0, stall_window=1000, stall_rtol=1e-5, geodesic=geo))
        for geo, tag in ((False, "std"), (True, "geo")) for s in range(3)]
    batches["ladder2"] += [  # rung B with geodesic LM (the old ladder has B with plain LM)
        job(f"lad2_B_geo_s{s}", "ladder2", p, target="sho", omega_target=1.0, toy=dict(mode="fixed", omega=1.5),
            data=LADDER_DATA, seed=s, stages=stages(p, polish_hours=68.0, stall_window=1000, stall_rtol=1e-5, geodesic=True))
        for s in range(3)]

    # ---- second node bundle ----
    A256 = dict(MW_DATA, n_steps=256)
    batches["cont32"] = [
        job(f"c32_L32w{w}s{s}_a{nA}", "cont32", p, model=dict(n_layers=32, width=w), data=dict(MW_DATA, n_steps=nA), seed=s,
            init={"from": f"capprobe_v2/cap_L32_w{w}_geo_s{s}", "stage": 1}, stages=long_geo())
        for w in (32, 64) for s in (0, 1) for nA in (256, 512)]
    batches["fresh32"] = [job(f"f32_s{s}", "fresh32", p, model=dict(n_layers=32, width=32), data=A256, seed=s, stages=long_geo())
                          for s in range(3)]
    batches["depth"] = [job(f"dep_L{L}_w{w}_s{s}", "depth", p, model=dict(n_layers=L, width=w), data=A256, seed=s, stages=long_geo())
                        for L, w in ((48, 32), (64, 32), (48, 64)) for s in range(2)]
    batches["mwt32"] = [job(f"mwt32_s{s}", "mwt32", p, target="teacher", teacher=dict(checkpoint="teachers/mw_32x32_a256.pt"),
                            model=dict(n_layers=32, width=32), data=A256, seed=s, stages=long_geo()) for s in range(3)]
    batches["mwt16deep"] = [job(f"mwt16d_s{s}", "mwt16deep", p, target="teacher", teacher=dict(checkpoint="teachers/mw_fixed_std_s1.pt"),
                                model=dict(n_layers=32, width=64), seed=s, stages=long_geo()) for s in range(3)]
    LAY = dict(MW_DATA, n_steps=256, inner=dict(n=24, split_pc=150))
    TOYS = {"fix": dict(mode="fixed", omega="nu0"), "lrn": dict(mode="learned", omega="nu0", n_cheb=6),
            "lrnA": dict(mode="learned", omega="nu0", n_cheb=6, var="amp"),
            "pinA": dict(mode="learned", omega="nu0", n_cheb=6, var="amp", anchor="value_slope")}
    batches["layout"] = [job(f"lay_{tag}_L{L}w{w}_s{s}", "layout", p, model=dict(n_layers=L, width=w), toy=toy, data=LAY, seed=s,
                             stages=long_geo())
                         for tag, toy in TOYS.items() for L, w, seeds in ((16, 64, (0,)), (32, 32, (0, 1))) for s in seeds]

    # ---- third node bundle ----
    LRNA = TOYS["lrnA"]
    batches["deep3"] = [
        job(f"d3_L{L}w32_lrnA_a{nA}_s{s}", "deep3", p, model=dict(n_layers=L, width=32), toy=LRNA,
            data=dict(LAY, n_steps=nA), seed=s, stages=long_geo())
        for L, seeds in ((48, (0, 1, 2)), (64, (0, 1))) for nA in (256, 512) for s in seeds]
    batches["ctrl3"] = (
        [job(f"c3_L48w32_fix_a512_s{s}", "ctrl3", p, model=dict(n_layers=48, width=32), data=dict(MW_DATA, n_steps=512),
             seed=s, stages=long_geo()) for s in range(2)]
        + [job(f"c3_L{L}w32_lrnA_std_a256_s{s}", "ctrl3", p, model=dict(n_layers=L, width=32), toy=LRNA, data=A256,
               seed=s, stages=long_geo()) for L in (32, 48, 64) for s in range(2)])
    keep = lambda j: (j["name"] in ("dep_L48_w32_s0", "dep_L48_w32_s1", "dep_L64_w32_s0", "dep_L64_w32_s1",
                                    "lay_lrnA_L32w32_s0", "lay_lrnA_L32w32_s1", "lay_fix_L32w32_s0", "lay_fix_L32w32_s1",
                                    "c32_L32w32s0_a512", "c32_L32w32s1_a512"))
    batches["resume3"] = [j for b in ("depth", "layout", "cont32") for j in batches[b] if keep(j)]

    # ---- fourth bundle: continue the third bundle's jobs, with the damping reset on a failed iteration ----
    # left out (laptop probe, 2026-10-21: two LM iterations from lam 1e-11 on each final checkpoint): "dead" found no
    # descent step, or only at lam > 1e2; "slow" lowered the loss by < 5e-5 per step, < 10% over a node-day
    SKIP = json.load(open(os.path.join(HERE, "jobs", "cont4_skip.json")))
    DEAD = set(SKIP["dead"]) | set(SKIP["slow"])

    def cont(j):
        lm = dict(j["stages"][-1], lam0=1e-11, retry_lam=1e-11)
        src = f"{j['batch']}/{j['name']}"
        return dict(j, batch="cont4_v2", init={"from": src, "stage": len(j["stages"]) - 1}, stages=j["stages"][:-1] + [lm])
    batches["cont4"] = [cont(j) for b in ("deep3", "ctrl3", "resume3") for j in batches[b] if j["name"] not in DEAD]
    # the same 13 continuations on the Trillium GPU cluster (trillium_gpu_cont4.sh: one H100 per job, a Slurm array):
    # LM in float64 on CUDA with 96 tori per Jacobian chunk (benchmark: 4-10x faster per job than 6 CPU threads)
    batches["cont4g"] = [dict(j, batch="cont4g_v2", stages=j["stages"][:-1] + [dict(j["stages"][-1], device="cuda", chunk_orbits=96)])
                         for j in batches["cont4"]]

    # ---- breaking the optimisation floor: Fourier-term conditioners (H100s, trillium_gpu_array.sh) ----
    # students of the 16-layer MW teacher, as the first bundle's mwt_geo (same seeds, so the same initial network plus zero
    # Fourier terms), with FourierNNConditioner: 12 frequencies log-spaced from 1 to 100. LM on CUDA with the retry rule.
    gpu_stages = [dict(kind="adam", device="cuda", dtype="float64", steps=3000, lr=1e-3),
                  dict(kind="lm", device="cuda", dtype="float64", max_iters=200000, wall_hours=68.0, stall_window=1000,
                       stall_rtol=1e-5, every=25, geodesic=True, retry_lam=1e-11, chunk_orbits=96)]
    batches["fourier1"] = [job(f"mwt_ff12_s{s}", "fourier1", p, target="teacher", teacher=dict(checkpoint="teachers/mw_fixed_std_s1.pt"),
                               model=dict(fourier=dict(n=12, max=100.0)), seed=s, stages=gpu_stages) for s in (0, 1)]
    # the same with amplitude-modulated Fourier terms (8 shared tanh units per layer), seed 0 (paired with mwt_geo_s0)
    batches["fourier2"] = [job(f"mwt_ffm12_s{s}", "fourier2", p, target="teacher", teacher=dict(checkpoint="teachers/mw_fixed_std_s1.pt"),
                               model=dict(fourier=dict(n=12, max=100.0, modulate=8)), seed=s, stages=gpu_stages) for s in (0,)]

    for b, jobs in batches.items():
        path = os.path.join(HERE, "jobs", f"{b}_{p}.jsonl")
        with open(path, "w") as fh:
            for j in jobs:
                fh.write(json.dumps(j) + "\n")
        print(f"{path}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
