"""Numbers and curves for the note "Why Energy Suffices" (report/note.html) -> report/note_data.json.

1. Linear response on exact isochrone tori: deform a torus by one Fourier mode of a generating function,
   I = I0 + eps k cos(k.theta), and measure the spread of H, |L| and L_z on it against the first-order prediction.
2. The rigid tilt of each trained isochrone model, and L_z wander before and after removing it.
3. Per-torus energy error and |L| wander of trained models against the detuning of the 1:-2 resonance."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import math
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import core3d as C  # noqa: E402

F = torch.float64
RES = os.path.join(ROOT, "results")


def modes(eps=1e-4):
    toy, iso = C.IsochroneToy(1.0, 1.0), C.Isochrone(1.0, 1.0)
    th = C.angle_samples(4096, 5); out = []
    for L in (0.02, 0.05, 0.1, 0.2, 0.5):
        J0 = torch.tensor([0.05, L, 0.4 * L], dtype=F); Om = iso.freqs(J0[None])[0]
        for k in ([1, 0, 0], [0, 1, 0], [1, -1, 0], [1, -2, 0], [2, -3, 0], [3, -5, 0], [2, -4, 1], [0, 0, 1], [0, 0, 2]):
            kt = torch.tensor(k, dtype=F); e = eps * min(0.05, L) / max(abs(x) for x in k)      # small against every action
            J = J0[None] + e * kt[None] * torch.cos(th @ kt)[:, None]
            xv = toy(torch.cat([th, J], -1)); H = iso.H(xv); Lv = torch.cross(xv[:, :3], xv[:, 3:], dim=-1)
            out.append(dict(L=L, k=k, kOm=float((kt * Om).sum()), Om_r=float(Om[0]), dH=float(H.std()) / e, dL=float(Lv.norm(dim=-1).std()) / e,
                            dLz=float(Lv[:, 2].std()) / e))
    return out


def load(name):
    d = os.path.join(RES, name); c = json.load(open(os.path.join(d, "config.json")))
    m = C.build_model(c["model"], c["toy"], seed=c.get("seed", 0))
    p = os.path.join(d, "state.pt"); m.load_state_dict(torch.load(p if os.path.exists(p) else os.path.join(d, "model.pt"), weights_only=False)["state"])
    dd = c["data"]; J = C.spherical_action_grid(dd["n_per_axis"], dd["Jr"][0], dd["Jr"][1], dd["L"][0], dd["L"][1], dd["cosi_max"], True)
    return c, m, J


def tilt(name, n=2048):
    c, m, J = load(name); t = C.make_target(c["target"]); th = C.angle_samples(n, 777)
    with torch.no_grad():
        xv = m(C.aa_points(J, th).reshape(-1, 6)).view(len(J), n, 6)
    H = t.H(xv); e = ((H - H.mean(1, keepdim=True)).abs() / (H.mean(1, keepdim=True) - t.E0)).mean()
    Lv = torch.cross(xv[..., :3], xv[..., 3:], dim=-1); sc = J[:, 0] + J[:, 1]
    ok = J[:, 2].abs() > 0.2 * J[:, 1]                                  # near-polar tori define the axis poorly
    ax = Lv[ok].mean(1) * torch.sign(J[ok, 2])[:, None]; ax = ax / ax.norm(dim=-1, keepdim=True)
    a0 = ax.mean(0); a0 = a0 / a0.norm()
    La = Lv @ a0
    return dict(name=name, width=c["model"]["width"], seed=c.get("seed", 0), grown=bool(c.get("grow")), energy=float(e),
                tilt_deg=math.degrees(math.acos(float(a0[2]))), axes_agree_deg=float(torch.rad2deg(torch.acos(torch.clamp(ax @ a0, -1, 1))).max()),
                L_wander=float((Lv.norm(dim=-1).std(1) / sc).median()), Lz_wander=float((Lv[..., 2].std(1) / sc).median()),
                Laxis_wander=float((La.std(1) / sc).median()), Laxis_offset=float(((La.mean(1) - J[:, 2]).abs() / sc).max()))


def per_torus(name, n=1024):
    c, m, J = load(name); t = C.make_target(c["target"]); th = C.angle_samples(n, 12345)
    with torch.no_grad():
        xv = m(C.aa_points(J, th).reshape(-1, 6)).view(len(J), n, 6)
    H = t.H(xv); e = ((H - H.mean(1, keepdim=True)).abs() / (H.mean(1, keepdim=True) - t.E0)).mean(1)
    w = torch.cross(xv[..., :3], xv[..., 3:], dim=-1).norm(dim=-1).std(1) / (J[:, 0] + J[:, 1])
    Om = t.freqs(J); det = (Om[:, 0] - 2 * Om[:, 1]).abs() / Om[:, 0]
    return dict(name=name, Jr=J[:, 0].tolist(), L=J[:, 1].tolist(), energy=e.tolist(), wander=w.tolist(), detuning=det.tolist())


if __name__ == "__main__":
    names = ["depth_d16_b0_s0", "depth_d16_b0_s1", "width_w128_b0_s1", "grow_w64_from_d16_b0_s0", "grow_w64_from_d16_b0_s1", "grow_w128_from_d16_b0_s0",
             "grow_w128_from_d16_b0_s1", "width_w64_b0_s0"]
    data = dict(modes=modes(), tilt=[tilt(n) for n in names], per_torus=[per_torus(n) for n in ("s3_isochrone", "depth_d16_b0_s1", "grow_w128_from_d16_b0_s1")])
    json.dump(data, open(os.path.join(HERE, "note_data.json"), "w"))
    for r in data["tilt"]:
        print(f"{r['name']:26s} tilt {r['tilt_deg']:6.2f} deg  E {r['energy']:.1e}  |L| {r['L_wander']:.1e}  Lz {r['Lz_wander']:.1e} -> {r['Laxis_wander']:.1e}  offset {r['Laxis_offset']:.1e}")
    import statistics as st
    for p in data["per_torus"]:
        r = [w / e for w, e in zip(p["wander"], p["energy"])]
        lo = [x for x, d in zip(r, p["detuning"]) if d < 0.03]; hi = [x for x, d in zip(r, p["detuning"]) if d > 0.1]
        print(p["name"], "median wander/energy", round(st.median(r), 2), "| detuning<0.03:", round(st.median(lo), 2), len(lo), "| >0.1:", round(st.median(hi), 2), len(hi))
