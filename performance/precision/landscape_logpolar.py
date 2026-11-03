"""python landscape_logpolar.py T|A OUT.npz -- log-polar cutouts of the energy error in the T-A-B plane (min+shift gauge), around T or around A:
points C + r |theta_T| (cos(phi) u + sin(phi) w), r log-spaced, so the shape of each pit is visible over many decades."""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "/Users/gabrielpfaffman/Repos/orbitflows/.claude/worktrees/precision-optim/performance/precision")
import numpy as np, torch
torch.set_num_threads(2)
import landscape_lab as LL
L = LL.load_all()
centre = sys.argv[1]
T, A, B = L.vT, L.al_min_shift[L.A], L.al_min_shift[L.B]
d1, d2 = A - T, B - T
u = d1 / d1.norm(); w = d2 - (d2 @ u) * u; w = w / w.norm(); w = w - (w @ u) * u; w = w / w.norm()
C = dict(T=T, A=A)[centre]
r = np.logspace(-16, -5, 45); phi = np.linspace(0, 2 * np.pi, 49)
R, P = np.meshgrid(r, phi, indexing="ij")
D = torch.as_tensor(np.cos(P).reshape(-1, 1)) * u[None] + torch.as_tensor(np.sin(P).reshape(-1, 1)) * w[None]
V = C[None] + torch.as_tensor(R.reshape(-1, 1)) * L.nT * D
eps = LL.eps_batch(V).numpy().reshape(R.shape)
np.savez(sys.argv[2], r=r, phi=phi, eps=eps, eps_centre=float(LL.eps_batch(C)), centre=centre)
print(centre, "eps centre", float(LL.eps_batch(C)), "eps range", eps.min(), eps.max())
