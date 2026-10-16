# 3D action-angle maps (exploratory)

Branch `flows-3d`. Goal: learn the action-angle map of a 3D potential with an exactly symplectic flow, the 3D
version of what `performance/precision/` does in 1D, starting from targets whose answer is known exactly.

```
(theta, J) in T^3 x R^3_+  ->  toy: 3D harmonic oscillator, one ellipse per axis  ->  6D symplectic flow  ->  (x, v)
```

Every toy 3-torus has actions J, and the flow is exact-symplectic, so every image torus has actions J as
well. Training only asks that the energy be constant on each image torus. In 3D that is still enough: a
Lagrangian torus on which H is constant is invariant under H's flow, because dH = 0 on the torus forces the
Hamiltonian vector field to be tangent to it.

## Decisions

| Decision | Why |
|---|---|
| **New 6D layers** (`core3d.py`), not the repo's conditioners. Nonlinear gradient shears q ← q + g·W(a ⊙ tanh(Wᵀp + b)) and linear symmetric shears q ← q + Sp, alternating with the same updates on p. | In n dimensions, q ← q + f(p) is symplectic only if f is a gradient. The repo's `SimpleNNConditioner` outputs independent components, and `GradientBasedConditioner` has the gradient form but multiplies each component by its own gate, which breaks symplecticity for n > 1. The layers here are SympNet modules (Jin et al. 2020) with a scalar gate. The linear symmetric shears generate the whole linear symplectic group, so linear parts of a map are represented exactly, avoiding the slow approximate linear maps seen in 1D rung B. |
| **Initialisation as in 1D:** gates drawn from N(0, 10⁻³), random biases, linear shears at zero. | The two traps found in 1D: zero gates make the identity an exact critical point, and zero biases freeze odd activations on symmetric targets. |
| **Loss:** per-torus energy variance divided by a fixed energy scale, the toy energy Σᵢ ωᵢJᵢ. | Dividing by the torus's own ⟨H⟩ allows the translation loophole found in 1D. |
| **Angles:** a scrambled Sobol set on T³, fixed per seed. Held-out tori sit at log-midpoints between training tori, with an independent Sobol set of 1,024 angles. | A grid of 16³ angles per torus would be too many residuals for LM. |
| **Optimiser:** Adam, then Levenberg–Marquardt with the fast Jacobian and geodesic acceleration (with the plain-step fallback), all float64. | Ported from `precision-optim`. |
| **Potentials written directly in torch for now, not galpy.** | PR 1 (Jo Bovy) routes potentials through galpy's torch backend. The released galpy 1.12.0 doesn't have that backend (handing it a tensor that requires gradients fails); it lives on galpy's `backend/*` development branches and needs building from source. These first targets are simple enough to write in torch. The potential is a plain `H(xv)` method, so galpy's torch backend can be dropped in later, for MWPotential2014 in 3D for example. |
| **Ground truth without numpy:** Gauss–Legendre nodes come from Golub–Welsch in torch. | On this machine, calling numpy's `leggauss` after torch is loaded crashes (two OpenMP runtimes). |

## Targets, in order

1. **`aniso_ho`:** an anisotropic oscillator (ω = 1, 1.4, 1.9) fitted from an isotropic toy (ω₀ = 1.4). The exact map is linear. This tests the 6D machinery, the 3-torus loss and the optimiser.
2. **`sep_quartic`:** a separable anharmonic potential Σᵢ(ωᵢ²xᵢ²/2 + λᵢxᵢ⁴/4). E(J) is exact by quadrature on each axis (checked: the J(E(J)) round trip is good to 4×10⁻¹⁶, and the frequencies agree with finite differences at O(h²)). The exact map is a product of three nonlinear 1D maps, which the coupled 6D flow has to discover.
3. **Next: a spherical potential** (Plummer, Hernquist or isochrone). A Cartesian-oscillator toy has the wrong torus topology for a spherical potential: its tori are labelled by (J_x, J_y, J_z), while a spherical potential's are labelled by (J_r, L, L_z), with different degenerate edges. So this needs a spherical toy, either the isochrone or an oscillator in spherical actions. Ground truth comes from galpy's `actionAngleSpherical` (numpy, works in 1.12).

## Findings so far

**Stage 1, the anisotropic oscillator (exact map linear).** 125 tori × 256 angles, laptop, float64.

| Flow | Result |
|---|---|
| 4 linear shears only (36 parameters) | **2.6×10⁻¹⁶ (machine precision) by LM step 15** |
| 2 linear shears only | stuck at 7×10⁻²: too few factors to represent the map |
| 4 linear + 8 nonlinear layers (684 parameters) | 2.7×10⁻⁸ after 300 LM steps, still falling; torus-mean energy within 1.5×10⁻¹⁰ of the exact H(J), frequencies within 4×10⁻⁹ |

Unneeded nonlinear layers slow the approach to an exact solution, because LM has to drive their gates to zero. This is the same kind of stall as 1D rung A near the identity. **Decision:** fit the linear part first, then add nonlinear layers on top (grow the model).

**Stage 2, the separable quartic (λ = 1.5, 2.5, 4; frequencies shift by 30–57% at J = 1).** 2,612 parameters; 125 grid tori × 256 angles; 1,000 Adam steps then 200 LM steps; 22 min on the laptop.

- Held-out error **7.2×10⁻⁴**, against 4×10⁻⁶ on the training tori.
- Torus-mean energy within 1.7×10⁻⁵ of the exact H(J); frequencies within 7.7×10⁻⁵.
- The 100× gap comes entirely from the tori, not the angles:
  - training tori at new angles: 4.9×10⁻⁶ (the angles generalise);
  - new tori at the training angles: 7.1×10⁻⁴.
- So 5 tori per axis is in the radial-interpolation regime the audit found in 1D. Next test: the same 32k-residual budget spent as 1,000 quasi-random tori × 32 angles.

## Run

```bash
python run3d.py configs/s1_aniso_ho.json          # -> results/s1_aniso_ho/{config.json, log.jsonl, model.pt, result.json}
python run3d.py configs/s2_sep_quartic.json        # 125 grid tori x 256 angles
python run3d.py configs/s2b_sep_quartic_sobol.json # 1000 scattered tori x 32 angles, same residual budget
python run3d.py configs/s3_isochrone.json          # spherical: isochrone target, spherical-oscillator toy
```
