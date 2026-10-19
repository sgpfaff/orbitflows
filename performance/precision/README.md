# Precision runs (portable: geir, Trillium, laptop)

This directory runs the precision campaign for the 1D vertical Milky Way map. It is self-contained: it imports `orbitflows` only for the flow layers and `MWPotential2014`. Each job checkpoints as it goes, so you can stop it with Ctrl-C, lose the session, or hit the end of an allocation, and rerunning the same command resumes it.

## The recipe

These are the defaults, all settable per job:

- **Model:**
  - 16 shear layers, width 64, tanh.
  - Random biases: zero biases trap odd activations on a symmetric potential.
  - Gates drawn from N(0, 10⁻³): all-zero gates make the identity an exact critical point.
  - Optional *odd* conditioners, f(x) = [g(x) − g(−x)]/2, which fix the phase-space origin exactly.
- **Toy map:**
  - `fixed` uses frequency ν₀ = √Φ″(0), which is exact as J → 0.
  - `learned` uses ln ω(J) = ln ν₀ + a Chebyshev series in ln J. The network's input angle θ is the ellipse parameter; the conjugate angle is θ_c = θ − (κ/2) sin 2θ, with κ = d ln ω / d ln J, and (θ_c, J) → (q, p) is exactly canonical. Torus averages in the evaluator are taken in θ_c, weighting each sample by 1 − κ cos 2θ.
  - `anchor` (learned only) pins the series at the smallest training action, where the true frequency is ν₀: `"value"` sets ω(J_lo) = ν₀, and `"value_slope"` also sets κ(J_lo) = 0, so ω = ν₀ for every smaller action. Without it, ω drifts near the midplane, because there a change of toy frequency and a linear stretch of the flow are interchangeable.
- **Data:**
  - z_max from 5 pc to 8.3 kpc (J = 5.3×10⁻⁷ to 0.42 in natural units, R₀ = 8 kpc).
  - 96 tori spaced evenly in asinh(a/0.03), about 9 pc apart near the midplane, × 128 angles.
- **Held-out set:** 200 tori at the midpoints between training tori × 512 angles offset from the training angles.
- **Loss:** the mean over tori of mean[(H − ⟨H⟩)² / E_ref²], with a fixed energy scale E_ref = ω₀J. H is always evaluated in float64.
  - Do not normalise by the orbit's own ⟨H⟩. A constant shear offset translates a torus and inflates ⟨H⟩, which drives that relative loss to zero without fitting anything, and a large LM step finds this. `loss_norm: "mean"` restores the old behaviour for comparison.
  - Every log point also records percentiles of the error along each orbit (`eval_p90`, `eval_p99`, `eval_p999`: the error each held-out orbit spends 90 / 99 / 99.9% of its time below, at 4,096 angles, averaged over orbits); `result.json` stores them per torus with the mean and max.
  - Every log point also records the error relative to the true energy E(J) (`eval_mean_trueE`, which cannot be gamed this way), the largest |⟨H⟩/E(J) − 1| over orbits, and where the origin is mapped.
- **Training:** Adam (cosine schedule, 3000 steps), then Levenberg–Marquardt (optionally with geodesic acceleration, `"geodesic": true` in the LM stage) until one of these happens:
  - it converges (less than 0.1% improvement over 300 steps);
  - it can no longer descend;
  - it reaches `wall_hours` of LM time, counted across resumes.

### Geodesic acceleration

The residual Jacobian is extremely sloppy: of 3,104 singular values, only tens lie above 10⁻⁸ of the largest. Plain LM then creeps. With `"geodesic": true`, each LM step d gets the second-order correction of Transtrum & Sethna (2012): a = −(JᵀJ + λD)⁻¹ Jᵀ r_dd, with the directional second derivative r_dd from one extra residual evaluation, and the step d + a/2 is tried first while 2|a| ≤ 0.75|d|. If it is too large or fails to descend, the plain step is tried before λ is raised, so a step is never worse than plain LM. In tests on the ladder's teacher target, about 15% more time per step gave 5.7× lower error in 150 steps from a checkpoint (plain LM: 1.1×), and 7× lower error after 60 steps from a fresh start.

### Teachers

`target: "teacher"` takes either a random flow (`n_layers`, `width`, `gate`, `seed`) or `checkpoint`, a frozen trained model written by `core.save_teacher(job_dir, path)`. `teachers/mw_fixed_std_s1.pt` is the best fixed-ν₀ MW model (1.2×10⁻⁵ from the real MW tori). A 16 × 64 student with the same toy represents it exactly (its own weights score 5×10⁻¹⁶), so training on it measures how far the optimiser gets on a map with MW's complexity.

## Setup (once per machine)

```bash
cd ~/orbitflows && git pull            # or rsync this directory over
pip install -e .                       # orbitflows importable; needs torch >= 2.0 (torch.func)
cd performance/precision
python make_jobs.py --profile cpu      # writes jobs/{recipe,capacity,ladder}_cpu.jsonl
```

Choose the profile by hardware:

- **`--profile cpu`**: everything runs in float64 on the CPU. Use this on Trillium CPU bookings, and on geir if it has plenty of cores.
- **`--profile gpu`**: Adam and a first LM stage run in float32 on a GPU (CUDA, or Apple MPS), then the float64 LM polish runs on the CPU.
  - Float32 LM stalls near 10⁻⁴; the gradient there is mostly rounding error. The float64 polish continues from that point.
  - geir's TITAN Xp runs float64 at 1/32 of its float32 speed, so use it only for the float32 phase.

## Run

```bash
screen -S prec                                     # or tmux
python queue_runner.py jobs/recipe_cpu.jsonl jobs/ladder_cpu.jsonl --workers 12 --threads 8
# detach: Ctrl-a d    reattach: screen -r prec
```

Keep workers × threads at or below the number of cores. The LM step is dominated by dense linear algebra, so 6–8 threads per job is a good balance.

The runner prints a status table every 2 minutes and writes it to `results/queue_status.txt`. Each job writes:

```
results/<batch>/<name>/config.json   the job spec
                      log.jsonl      held-out / training error and z_max bands at each log point
                      state.pt       float64 weights + position (atomic; resume point)
                      result.json    final metrics: per-torus errors vs z_max, action and frequency
                                     errors, image of the origin, stage stop reasons, wall time
                      stdout.log
                      DONE
```

To stop, press Ctrl-C (or kill it, or let the allocation end). To resume, run the same command: finished jobs are skipped and the others continue from their last checkpoint, losing at most 25 LM steps.

### Second bundle: denser angles, depth, a 32-layer teacher, a new layout

A stall diagnosis of the first bundle's 32-layer models showed that their held-out error was set by the angular sampling of the outer orbits (error between the 128 training angles 50× the error at them above 3 kpc), and fine-tuning with 256 angles per orbit restarted training (4.9 → 2.2×10⁻⁷ held-out in 200 steps). The second bundle (`trillium_bundle2.sh`, 35 jobs, all geodesic LM) builds on that:

- `cont32`: the four 32-layer checkpoints continued at 256 and 512 angles (`init` starts a job from another job's `state.pt`, at a given stage).
- `fresh32`: 32 × 32 from scratch at 256 angles.
- `depth`: 48 × 32, 64 × 32, 48 × 64 at 256 angles.
- `mwt32`: students on a 32 × 32 MW teacher (`teachers/mw_32x32_a256.pt`, the fine-tuned model; 2.2×10⁻⁷ from the real MW).
- `mwt16deep`: 32 × 64 students on the 16 × 64 teacher with the 16-layer students' data, isolating the effect of depth on optimisation.
- `layout`: `data.inner = {n: 24, split_pc: 150}` replaces the tori below 150 pc (toy amplitude) with 24 evenly spaced in ln J (104 tori in all; the held-out grid is unchanged), with 256 angles, for the fixed toy and for learned ω(J) written in ln J or, with `toy.var = "amp"`, in amplitude √J (free, or pinned value + slope).

## A whole Trillium node

Trillium schedules only whole nodes (192 cores, at most 24 h per job). One LM step barely speeds up with more threads (1.8 s on 1 thread, 1.2 s on 4), so a node is best used by many 4-thread jobs. `trillium_node.sh` runs the current bundle, 45 jobs, in priority order:

- `anchor`, `anchor_geo`, `toygeo`: the toy-frequency study, with pinned, free and fixed frequency (fixed is `mwgeo`).
- `mwgeo`, `mwteacher`: the real MW and the MW teacher.
- `ladder2`: rungs C and B.
- `capprobe`: the capacity probe.

Each job gets its own 4 cores, two per 8-core L3 cache. The 68 h jobs span three chained 24 h segments:

```bash
sbatch -p debug --time=00:30:00 trillium_node.sh     # optional 30-minute test
sbatch trillium_node.sh
sbatch --dependency=singleton trillium_node.sh       # x2: each waits for the previous segment
```

Unfinished jobs resume from their checkpoints at the start of each segment. Slurm's output goes to `results/slurm_<jobid>.out`; the queue's status table is in `results/queue_status.txt`.

## Collect and copy back

```bash
python collect.py                      # results/summary.json + summary.csv, prints a table
# on the laptop:
rsync -av trillium:~/orbitflows/performance/precision/results/ performance/precision/results/
```

## Batches

| file (results go to `results/<batch>_v2/`) | jobs | question |
|---|---|---|
| `recipe` | 12 | 3 seeds × toy {fixed ν₀, learned ω(J)} × conditioner {standard, odd}. Do learned ω and odd conditioners help? (Done, Oct 2026: learned ω wins above ~1 kpc and loses ~10× near the midplane; odd conditioners hurt.) |
| `mwteacher` | 3 | Fresh students (geodesic LM, 68 h) on the MW teacher. Far below 1e-5 means the MW error is approximation and a capacity scan is meaningful above that floor; near 1e-5 means the optimiser is the limit. |
| `mwgeo` | 3 | The real MW target with the same optimiser: the comparison for `mwteacher`, and the next best-model attempt. |
| `ladder2` | 6 | Rung C, now exactly realisable, standard vs geodesic LM × 3 seeds, 68 h. The old ladder's toy frequency was float32-rounded (1.2×10⁻⁸ off the teacher's 1.5), so its C runs had to absorb a tiny linear squeeze; taking that toy away makes the trained s0 8× worse. |
| `anchor` | 6 | Learned ω(J) anchored to ν₀ at the smallest action (`value`, `value_slope`) × 3 seeds, standard conditioners, 68 h LM cap. Does the anchor keep the fixed toy's accuracy near the midplane together with the learned toy's in the outer disc? |
| `ladder` | 9 | (A) SHO with the exact toy, (B) SHO with the wrong toy frequency, (C) teacher–student. How far does optimisation alone go? Machine precision? |
| `capacity` | 45 | layers {4…64} × width {16, 32, 64} × 3 seeds (≤ 12.4k parameters). Error vs model size, each run converged. Regenerate with the recipe winner: `make_jobs.py --toy learned --odd`. |

Cost guide: a 16 × 64 model (3,104 parameters) on 96 × 128 points takes about 1.3 s per LM step on 4 laptop (M-series) threads; Trillium's 4 threads took about 3.5× longer before the Jacobian speedup, which made the step 5× faster. The cost scales as points × parameters², so the largest capacity models (12.4k parameters) are about 16× slower per step.
