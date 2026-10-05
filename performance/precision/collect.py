"""Summarise every job under results/ into results/summary.json and results/summary.csv.

    python collect.py [--out results]

For finished jobs: final held-out / training errors, z_max-band errors, action and frequency errors,
origin image, wall time, stage stop reasons, plus the per-torus curves (for plots). For unfinished jobs:
the latest log record. Copy results/ back to the laptop (rsync) and the audit report can ingest summary.json.
"""
import argparse
import csv
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def last_record(path):
    rec = None
    if os.path.exists(path):
        for ln in open(path):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                pass
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "results")); a = ap.parse_args()
    rows, full = [], []
    for cfg_path in sorted(glob.glob(os.path.join(a.out, "*", "*", "config.json"))):
        d = os.path.dirname(cfg_path); cfg = json.load(open(cfg_path))
        m = cfg["model"]
        row = dict(batch=cfg["batch"], name=cfg["name"], target=cfg["target"], seed=cfg.get("seed", 0),
                   layers=m["n_layers"], width=m["width"], act=m.get("act"), odd=m.get("odd", False),
                   toy=cfg["toy"].get("mode"), status="done" if os.path.exists(os.path.join(d, "DONE")) else "running/stopped")
        res_path = os.path.join(d, "result.json")
        if os.path.exists(res_path):
            r = json.load(open(res_path)); f = r["final"]
            af = f.get("action_freq") or {}
            row.update(params=r["n_params"], wall_h=round(r["wall_total_s"] / 3600, 3), eval_mean=f["eval_mean"],
                       eval_max=f["eval_max"], train_mean=f["train_mean"],
                       action_med=statistics.median(af["rel_action_err"]) if af else None,
                       freq_med=statistics.median(af["rel_freq_err"]) if af else None,
                       stop=";".join(f"{s['label']}:{s['why']}@{s['iters']}" for s in r["stages"]),
                       **{k: v for k, v in f.items() if k.startswith("band_")})
            full.append(dict(row=row, per_torus=f.get("per_torus"), action_freq=af, origin=f.get("origin_image")))
        else:
            rec = last_record(os.path.join(d, "log.jsonl")) or {}
            row.update(eval_mean=rec.get("eval_mean"), eval_max=rec.get("eval_max"), train_mean=rec.get("train_mean"),
                       stop=f"at {rec.get('label')} it {rec.get('it')}", wall_h=round(rec.get("t", 0) / 3600, 3))
        rows.append(row)
    json.dump(dict(rows=rows, full=full), open(os.path.join(a.out, "summary.json"), "w"))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("batch", "name"), k))
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    fmt = lambda v: f"{v:.2e}" if isinstance(v, float) else str(v)
    for r in rows:
        print(f"{r['name']:44s} {r['status']:15s} held-out {fmt(r.get('eval_mean'))} max {fmt(r.get('eval_max'))} "
              f"train {fmt(r.get('train_mean'))} {r.get('stop', '')}")
    print(f"{len(rows)} jobs -> {os.path.join(a.out, 'summary.json')}")


if __name__ == "__main__":
    main()
