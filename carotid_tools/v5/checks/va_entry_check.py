# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\va_entry_check.py (= /sp/v5_va/va_entry_check.py in the build containers)
import json, glob, os, hashlib
import numpy as np
def va(root, key):
    for f in glob.glob(os.path.join(root, "Centrelines_comb", "*.json")):
        if key in os.path.basename(f).upper():
            d = json.load(open(f))
            cps = d["markups"][0]["controlPoints"]
            p = np.array([c["position"] for c in cps], float)
            return p
    return None
def first_mm(p, mm):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return p[s <= mm]
for ver in ("v4", "v5"):
    roots = sorted(glob.glob("/repo/carotid_data/anatomies_%s/*/Centrelines_comb" % ver))
    for key in ("RVA", "LVA"):
        sigs = {}
        for r in roots:
            p = va(os.path.dirname(r), key)
            if p is None: continue
            q = np.round(first_mm(p, 40.0), 3)
            h = hashlib.md5(q.tobytes()).hexdigest()[:10]
            sigs.setdefault(h, []).append(os.path.basename(os.path.dirname(r)))
        sizes = sorted((len(v) for v in sigs.values()), reverse=True)
        print("%s %s: %d anatomies, %d distinct first-40-mm VA paths; largest groups %s" % (ver, key, sum(sizes), len(sigs), sizes[:5]))
