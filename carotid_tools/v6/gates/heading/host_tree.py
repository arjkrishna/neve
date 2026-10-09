# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\heading\host_tree.py (= /sp/v6/heading/host_tree.py in the build containers)
import json, glob, os, numpy as np
H = "/repo/eve_bench/data/dualdevicenav/Centrelines_comb"
def j2b(p):
    p = np.asarray(p, float); return np.stack([p[:, 1], -p[:, 2], -p[:, 0]], 1)
C = {}
for f in sorted(glob.glob(H + "/*.mrk.json")):
    d = json.load(open(f)); p = j2b([c["position"] for c in d["markups"][0]["controlPoints"]])
    C[os.path.basename(f).replace("Centerline curve ", "").replace(".mrk.json", "")] = p
for n, p in C.items():
    L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    print("%-8s n %4d len %6.1f start %s end %s" % (n, len(p), L, p[0].round(1), p[-1].round(1)))
# which curves pass near each named route's start
print()
for n in ("- RCCA", "- LCCA", "- RVA", "- LVA"):
    p = C[n]
    near = []
    for m, q in C.items():
        if m == n: continue
        d0 = np.linalg.norm(q - p[0], axis=1).min()
        if d0 < 3: near.append((m, round(float(d0), 2), int(np.argmin(np.linalg.norm(q - p[0], axis=1))), len(q)))
    print(n, "start", p[0].round(1), "touches:", near)
