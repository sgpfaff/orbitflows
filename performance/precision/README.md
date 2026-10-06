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
  - `learned` uses ln ω(J) = ln ν₀ + a Chebyshev series in ln J. It stays exactly canonical through the angle correction θ_c = θ − (κ/2) sin 2θ, with κ = d ln ω / d ln J.
- **Data:**
  - z_max from 5 pc to 8.3 kpc (J = 5.3×10⁻⁷ to 0.42 in natural units, R₀ = 8 kpc).
  - 96 tori spaced evenly in asinh(a/0.03), about 9 pc apart near the midplane, × 128 angles.
- **Held-out set:** 200 tori at the midpoints between training tori × 512 angles offset from the training angles.
- **Loss:** the mean over tori of mean[(H − ⟨H⟩)² / E_ref²], with a fixed energy scale E_ref = ω₀J. H is always evaluated in float64.
  - Do not normalise by the orbit's own ⟨H⟩. A constant shear offset translates a torus and inflates ⟨H⟩, which drives that relative loss to zero without fitting anything, and a large LM step finds this. `loss_norm: "mean"` restores the old behaviour for comparison.
  - Every log point also records the error relative to the true energy E(J) (`eval_mean_trueE`, which cannot be gamed this way), the largest |⟨H⟩/E(J) − 1| over orbits, and where the origin is mapped.
- **Training:** Adam (cosine schedule, 3000 steps), then Levenberg–Marquardt until one of these happens:
  - it converges (less than 0.1% improvement over 300 steps);
  - it can no longer descend;
  - it reaches `wall_hours`.

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

## Collect and copy back

```bash
python collect.py                      # results/summary.json + summary.csv, prints a table
# on the laptop:
rsync -av trillium:~/orbitflows/performance/precision/results/ performance/precision/results/
```

## Batches

| file (results go to `results/<batch>_v2/`) | jobs | question |
|---|---|---|
| `recipe` | 12 | 3 seeds × toy {fixed ν₀, learned ω(J)} × conditioner {standard, odd}. Do learned ω and odd conditioners help? |
| `ladder` | 9 | (A) SHO with the exact toy, (B) SHO with the wrong toy frequency, (C) teacher–student. How far does optimisation alone go? Machine precision? |
| `capacity` | 45 | layers {4…64} × width {16, 32, 64} × 3 seeds (≤ 12.4k parameters). Error vs model size, each run converged. Regenerate with the recipe winner: `make_jobs.py --toy learned --odd`. |

Cost guide: a 16 × 64 model (3,104 parameters) on 96 × 128 points takes about 2–3 s per LM step with 8–12 fast cores. The cost scales as points × parameters², so the largest capacity models (12.4k parameters) are about 16× slower per step.
