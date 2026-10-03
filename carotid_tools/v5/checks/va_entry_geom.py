# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\va_entry_geom.py (= /sp/v5_va/va_entry_geom.py in the build containers)
import json, glob, os
import numpy as np
def load(root, key):
    for f in glob.glob(os.path.join(root, "Centrelines_comb", "*.json")):
        if key in os.path.basename(f).upper():
            d = json.load(open(f)); m = d["markups"][0]
            p = np.array([c["position"] for c in m["controlPoints"]], float)
            r = None
            for meas in m.get("measurements", []):
                if "radius" in str(meas.get("name", "")).lower() and meas.get("controlPointValues"):
                    r = np.array(meas["controlPointValues"], float)
            return p, r
    return None, None
def arcl(p): return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
def seg_dist(q, P):
    a, b = P[:-1], P[1:]; ab = b - a; L = (ab * ab).sum(1) + 1e-12
    t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(2) / L[None], 0, 1)
    c = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(q[:, None, :] - c, axis=2).min(1)
host = "/repo/eve_bench/data/dualdevicenav"
for key, mm in (("RVA", 60.0), ("LVA", 35.0)):
    hp, hr = load(host, key)
    hs = arcl(hp); H = hp[hs <= mm + 5]
    for ver in ("v4", "v5"):
        dmax = []
        for root in sorted(glob.glob("/repo/carotid_data/anatomies_%s/*/" % ver)):
            p, r = load(root, key)
            if p is None: continue
            q = p[arcl(p) <= mm]
            dmax.append(seg_dist(q, H).max())
        dmax = np.array(dmax)
        print("%s %s first %d mm vs host: max deviation median %.3f, p90 %.3f, max %.3f mm; identical (<0.05 mm) %d / %d"
              % (ver, key, mm, np.median(dmax), np.percentile(dmax, 90), dmax.max(), (dmax < 0.05).sum(), len(dmax)))
