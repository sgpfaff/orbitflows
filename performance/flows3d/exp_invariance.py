"""Fine-tune the stage-3 isochrone model from its checkpoint with plain LM: energy loss only (control) vs energy +
beta * invariance. Held-out diagnostics after each: energy error, invariance residual, |L| wander around each torus,
torus-mean energy vs exact H(J), frequency error. Writes results/exp_invariance.json."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core3d as C

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
STEPS = int(os.environ.get("STEPS", 30))


def diagnostics(m, t, Jh, th):
    aa = C.aa_points(Jh, th)
    with torch.no_grad():
        e = C.residuals(m, aa, t).abs()
        xv = m(aa.reshape(-1, 6)).view(Jh.shape[0], -1, 6); Lm = torch.cross(xv[..., :3], xv[..., 3:], dim=-1).norm(dim=-1)
        wander = Lm.std(1) / Jh[:, 1]
    R, Om = C.invariance_residuals(m, aa, t, C.invariance_scale(m, aa))
    Omt = t.freqs(Jh)
    af = C.Evaluator(t, J=Jh, n_angles=th.shape[0]).action_freq(m)
    return dict(energy_err=float(e.mean()), energy_err_max=float(e.max()), invariance_rms=float(R.pow(2).sum(-1).mean().sqrt()),
                L_wander_median=float(wander.median()), L_wander_max=float(wander.max()),
                omega_fit_err=float(((Om[:, :2] - Omt[:, :2]) / Omt[:, :2]).abs().median()),
                torus_E_err_max=max(af["rel_E_err"]), freq_err_max=max(af["rel_freq_err"]),
                per_torus=dict(wander=wander.tolist(), energy=e.mean(1).tolist(), ratio=(Omt[:, 0] / Omt[:, 1]).tolist()))


def main():
    ck = torch.load(os.path.join(HERE, "results/s3_isochrone/model.pt"), weights_only=False); c = ck["config"]; d = c["data"]
    t = C.make_target(c["target"])
    sg = lambda mid: C.spherical_action_grid(d["n_per_axis"], *d["Jr"], *d["L"], d["cosi_max"], mid)
    Jt, Jh = sg(False), sg(True); aa = C.aa_points(Jt, C.angle_samples(d["n_angles"], c["seed"])); th_h = C.angle_samples(1024, 12345)
    out = {}
    m0 = C.build_model(c["model"], c["toy"]); m0.load_state_dict(ck["state"]); out["start"] = diagnostics(m0, t, Jh, th_h)
    print("start:", {k: f"{v:.2e}" for k, v in out["start"].items() if k != "per_torus"}, flush=True)
    for lab, beta in (("energy_only", 0.0), ("energy_plus_invariance", 1.0)):
        m = C.build_model(c["model"], c["toy"]); m.load_state_dict(ck["state"])
        es = m.toy.energy(Jt).reshape(-1, 1); isc = C.invariance_scale(m, aa); log = []
        t0 = time.perf_counter()
        n, why, _ = C.train_lm_inv(m, aa, t, es, isc, beta, STEPS, lam=1e-6, every=5,
                                   record=lambda k, l, lam: (log.append((k, l)), print(f"  [{lab}] LM {k} loss {l:.3e} ({time.perf_counter() - t0:.0f} s)", flush=True)))
        out[lab] = dict(diagnostics(m, t, Jh, th_h), steps=n, stop=why, s_per_step=(time.perf_counter() - t0) / n, loss=log)
        print(f"{lab}:", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in out[lab].items() if k not in ("per_torus", "loss")}, flush=True)
        torch.save(dict(state=m.state_dict(), config=c), os.path.join(HERE, f"results/s3_isochrone_{lab}.pt"))
    json.dump(out, open(os.path.join(HERE, "results/exp_invariance.json"), "w"))


if __name__ == "__main__":
    main()
