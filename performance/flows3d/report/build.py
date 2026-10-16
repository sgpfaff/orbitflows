"""Build the 3D progress page: data.json (from make_data.py) + page.html -> flows3d_progress.html."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
page = open(os.path.join(HERE, "page.html")).read()
blob = json.dumps(data, separators=(",", ":"), allow_nan=True).replace("NaN", "null").replace("Infinity", "1e308")
out = page.replace("__DATA__", blob).replace("__GENERATED__", data["generated"])
open(os.path.join(HERE, "flows3d_progress.html"), "w").write(out)
print(f"flows3d_progress.html: {len(out) / 1e6:.2f} MB")
