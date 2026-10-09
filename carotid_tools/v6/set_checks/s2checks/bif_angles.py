#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\bif_angles.py (= /sp/v6/s2checks/bif_angles.py in the build containers)
"""bif_angles.py (s2_checks): the right bifurcation's take-off angles on a tree (information for the SOFA reach report):
angle between the CCA's last 8 mm direction and the ICA's (RCCA continuation) / ECA's (RECA) first 8 mm, at the RECA
take-off on the RCCA.   python3 bif_angles.py <out.json> <tag>=<root> ... -- <names...>"""
import json
import os
import sys

import numpy as np

sys.path[:0] = ["/repo/eve", "/repo/eve_bench"]
from eve_bench.dualdevicenav import load_branches  # noqa: E402


def arcl(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def d(p, s, a, b):
    i, j = int(np.argmin(np.abs(s - a))), int(np.argmin(np.abs(s - b)))
    v = p[j] - p[i]
    return v / max(np.linalg.norm(v), 1e-9)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


args = sys.argv[1:]
out = args[0]
k = args.index("--")
trees = dict(a.split("=", 1) for a in args[1:k])
names = args[k + 1:]
res = {}
for n in names:
    res[n] = {}
    for t, root in trees.items():
        br = {str(b.name): b for b in load_branches(os.path.join(root, n, "Centrelines_comb"))}
        rc = [b for k_, b in br.items() if k_.endswith("RCCA.mrk")][0]
        re_ = [b for k_, b in br.items() if k_.endswith("RECA.mrk")][0]
        p, pe = np.asarray(rc.coordinates, float), np.asarray(re_.coordinates, float)
        s, se = arcl(p), arcl(pe)
        s0 = float(s[int(np.argmin(np.linalg.norm(p - pe[0], axis=1)))])
        cca, ica, eca = d(p, s, s0 - 8, s0), d(p, s, s0, s0 + 8), d(pe, se, 0, 8)
        res[n][t] = {"eca_takeoff_mm": round(s0, 1), "cca_to_ica_deg": round(ang(cca, ica), 1), "cca_to_eca_deg": round(ang(cca, eca), 1),
                     "ica_to_eca_deg": round(ang(ica, eca), 1),
                     "r_ica_mm": round(float(np.interp(s0 + 5, s, np.asarray(rc.radii, float))), 2),
                     "r_eca_mm": round(float(np.interp(5, se, np.asarray(re_.radii, float))), 2)}
json.dump(res, open(out, "w"), indent=1)
for n, v in res.items():
    print(n, json.dumps(v))
