"""Diagnostic: an explicitly invertible correction stage built from ridge shears, fitted by one linear solve.

    python ridge_stage.py <job dir> [--degree 60] [--stage-orbits 400] [--stage-angles 512]

The spectral stage (spectral_stage.py) is a type-2 generating function, so each direction of its map needs a short
fixed-point solve. Here the correction is instead a product of K ridge shears in the Cartesian-like coordinates
X = sqrt(2J) (cos theta, sin theta) / R (R = sqrt(2 J_hi), so the tori lie in the unit disc):

    X -> X + f_k(u_k . X) u_k_perp,   u_k = (cos a_k, sin a_k),  a_k = pi k / K,  f_k = sum_m c_km P_m (Legendre)

Each shear has determinant 1 and leaves u_k . X unchanged, so it is exactly canonical and its inverse is the same
shear with -f_k. With K = D + 1 directions and degree D, the shears' generators span every polynomial of degree <= D
in X. To first order the product's displacement is the sum of the shears' displacements, so the coefficients solve
one linear least-squares problem: cancel the base model's energy deviation along each fitting torus,
dH = grad_X H . sum_k f_k(u_k . X) u_k_perp = -(H - <H>), relative to each torus's energy. The ordering of the shears
only enters at second order in the correction (~1e-13 for the 48 x 32 flow). Fixed toy only, as spectral_stage.py.
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402
import spectral_stage as SP  # noqa: E402

F64 = torch.float64


def legendre(s, D):
    P = [torch.ones_like(s), s]
    for m in range(1, D):
        P.append(((2 * m + 1) * s * P[-1] - m * P[-2]) / (m + 1))
    return torch.stack(P[:D + 1], -1)                                   # (..., D + 1)


class RidgeStage:
    def __init__(self, coef, R):
        self.coef, self.R = coef, R                                     # coef: (K, D + 1)
        K = coef.shape[0]; a = math.pi * torch.arange(K, dtype=F64) / K
        self.u = torch.stack([a.cos(), a.sin()], -1); self.up = torch.stack([-a.sin(), a.cos()], -1)

    @property
    def n_coef(self):
        return self.coef.numel()

    def to_X(self, aa):
        th, J = aa[..., 0], aa[..., 1]; r = torch.sqrt(2 * J) / self.R
        return torch.stack([r * th.cos(), r * th.sin()], -1)

    def from_X(self, X, like):
        th = torch.atan2(X[..., 1], X[..., 0]); th = th + 2 * math.pi * torch.round((like[..., 0] - th) / (2 * math.pi))
        return torch.stack([th, 0.5 * self.R ** 2 * (X ** 2).sum(-1)], -1)

    def _shears(self, X, sign, order):
        D = self.coef.shape[1] - 1
        for k in order:
            s = X @ self.u[k]
            X = X + sign * (legendre(s, D) @ self.coef[k])[..., None] * self.up[k]
        return X

    def __call__(self, aa):
        """(theta', J') -> (theta, J): the shears in order k = 0..K-1."""
        return self.from_X(self._shears(self.to_X(aa), 1.0, range(self.coef.shape[0])), aa)

    def inverse(self, aa):
        """(theta, J) -> (theta', J'): the same shears with -f_k, in reverse order (closed form)."""
        return self.from_X(self._shears(self.to_X(aa), -1.0, reversed(range(self.coef.shape[0]))), aa)


def fit(ctx, degree, stage_orbits, stage_angles, lam=1e-13):
    t, d = ctx["t"], ctx["d"]
    J_fit = C.j_grid(stage_orbits, ctx["J_lo"], ctx["J_hi"], ctx["w0"], d.get("spacing", "asinh"), ctx["a0"], offset=0.0)
    th_fit = C.angles(stage_angles, 0.5); nO, nS = len(J_fit), len(th_fit)
    R = math.sqrt(2 * ctx["J_hi"]); K, D = degree + 1, degree
    st = RidgeStage(torch.zeros(K, D + 1, dtype=F64), R)
    P = K * (D + 1); A = torch.zeros(P, P, dtype=F64); b = torch.zeros(P, dtype=F64)
    for o0 in range(0, nO, 8):                                          # whole tori per chunk (per-torus means)
        Jc = J_fit[o0:o0 + 8]; k = len(Jc)
        X = st.to_X(C.aa_points(Jc, th_fit)).reshape(-1, 2).requires_grad_(True)
        H = t.H(ctx["base"](st.from_X(X, C.aa_points(Jc, th_fit).reshape(-1, 2))))
        g = torch.autograd.grad(H.sum(), X)[0].detach(); H = H.detach().view(k, nS)
        E = H.mean(1, keepdim=True)
        r = ((H - E) / E).reshape(-1)
        Xd = X.detach()
        # design: d(H)/E for each coefficient c_km = (g . u_k_perp) P_m(u_k . X) / E, minus its torus mean
        cols = (legendre(Xd @ st.u.T, D) * (g @ st.up.T)[..., None]).reshape(-1, P)   # (k nS, K (D + 1))
        cols = cols.view(k, nS, P); cols = (cols - cols.mean(1, keepdim=True)) / E[..., None]; cols = cols.reshape(-1, P)
        A += cols.T @ cols; b -= cols.T @ r
    dg = torch.diagonal(A).clone()
    c = torch.linalg.solve(A + lam * dg.max() * torch.eye(P, dtype=F64), b)
    st.coef = c.view(K, D + 1)
    return st, J_fit, th_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--degree", type=int, default=60)
    ap.add_argument("--stage-orbits", type=int, default=400); ap.add_argument("--stage-angles", type=int, default=512)
    a = ap.parse_args()
    ctx = SP.load(a.job); base = SP.Corrected(ctx["base"], [])
    eb = SP.errors(ctx, base)
    t0 = time.perf_counter(); st, J_fit, th_fit = fit(ctx, a.degree, a.stage_orbits, a.stage_angles); t_fit = time.perf_counter() - t0
    model = SP.Corrected(ctx["base"], [st])
    e = SP.errors(ctx, model, J_fit, th_fit); ck = SP.checks(ctx, model)
    x = ctx["ev"].aa.reshape(-1, 2)[:32768].clone()
    with torch.no_grad():
        t1 = time.perf_counter(); ctx["base"](x); tb = time.perf_counter() - t1
        t1 = time.perf_counter(); ctx["base"](st(x)); tc = time.perf_counter() - t1
    print(f"{ctx['cfg']['name']}: ridge stage, degree {a.degree} ({st.coef.shape[0]} directions, {st.n_coef:,} coefficients), "
          f"fitted on {len(J_fit)} tori x {len(th_fit)} angles in {t_fit:.0f} s")
    print(f"  held-out {eb['held_out']:.2e} -> {e['held_out']:.2e} (max {eb['held_out_max']:.1e} -> {e['held_out_max']:.1e}); "
          f"fitting grid {e['train']:.2e}")
    print("  bands: " + SP.fmt_bands(e["bands"]))
    print(f"  explicit inverse round trip: max |d theta| {ck['round_trip_theta']:.1e}, max |dJ/J| {ck['round_trip_J']:.1e}; "
          f"|det - 1| <= {ck['det_dev']:.1e}; correction size {ck['size']:.1e}")
    print(f"  evaluation: flow {tb / len(x) * 1e6:.1f} us/point, flow + ridge stage {tc / len(x) * 1e6:.1f} us/point "
          f"({torch.get_num_threads()} threads)")


if __name__ == "__main__":
    main()
