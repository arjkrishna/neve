#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\select_sofa.py (= /sp/v5_va/checks/select_sofa.py in the build containers)
"""v5_checks item 4: pick the SOFA set = the 15 most changed anatomies + 10 random (neve-build-meshlab, read-only).

"Most changed", stated rule:
  1. every flagged anatomy of the bake: carotid_changed (3), OBJ re-decimated (obj_fix, 3), a VA kept from v4 by the
     bake (2), the largest tube window (case_w_008_right__topcow_mr_020, 9.14 mm), the worst right-route ratio
     (case_w_050_left__topcow_mr_004_L, fork 0.82);
  2. filled to 15 by the largest RVA course change against v4: the max distance of the v5 RVA centreline (past the
     (17) take-off) from the v4 RVA polyline (anatomies whose RVA is replaced);
  3. 10 random from the other 208 (numpy default_rng(20261002)).
Also writes the RVA course change of every anatomy.   -> /sp/v5_va/checks/sofa_selection.json
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

A5, V4 = "/repo/carotid_data/anatomies_v5", "/repo/carotid_data/anatomies_v4"


def pts(root, tag):
    d = json.load(open(os.path.join(root, "Centrelines_comb", "Centerline curve - %s.mrk.json" % tag)))
    P = []
    for mk in d["markups"]:
        if mk["type"] == "Curve":
            for cp in mk["controlPoints"]:
                x, y, z = cp["position"]
                P.append((y, -z, -x))
    return np.asarray(P, float)


def dense(p, step=0.25):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    g = np.arange(0, s[-1], step)
    return np.stack([np.interp(g, s, p[:, k]) for k in range(3)], 1), g


bake = json.load(open("/sp/v5_va/results/v5_bake_full.json"))["anatomies"]
names = sorted(a["name"] for a in bake)
by = {a["name"]: a for a in bake}
flag = []
for a in bake:
    why = []
    if "carotid_changed" in (a.get("flags") or []):
        why.append("carotid_changed")
    if a.get("obj_fix"):
        why.append("obj_fix")
    if a.get("v4_va_kept_by_bake"):
        why.append("v4_va_kept_by_bake %s" % a["v4_va_kept_by_bake"])
    if why:
        flag.append((a["name"], why))
extra = [("case_w_008_right__topcow_mr_020", ["largest window 9.14 mm (RVA, Song 002 R)"]),
         ("case_w_050_left__topcow_mr_004_L", ["worst right-route ratio (fork 0.82)"])]
chg = {}
for n in names:
    p5, p4 = pts(os.path.join(A5, n), "RVA"), pts(os.path.join(V4, n), "RVA")
    q5, s5 = dense(p5)
    q4, _ = dense(p4)
    p17 = pts(os.path.join(A5, n), "RVA")  # noqa: F841
    # (17) take-off: the RVA point nearest the start of branch (17)
    d17 = json.load(open(os.path.join(A5, n, "Centrelines_comb", "Centerline curve (17).mrk.json")))
    c = [cp["position"] for mk in d17["markups"] if mk["type"] == "Curve" for cp in mk["controlPoints"]][0]
    c = np.array([c[1], -c[2], -c[0]])
    s17 = s5[int(np.argmin(np.linalg.norm(q5 - c, axis=1)))]
    m = s5 > s17
    d = cKDTree(q4).query(q5[m])[0]
    chg[n] = {"rva_max_dev_mm": round(float(d.max()), 1), "rva_mean_dev_mm": round(float(d.mean()), 1),
              "rva_len_v5_mm": round(float(s5[-1]), 1), "rva_replaced": not np.array_equal(p5, p4)}
sel, why = [], {}
for n, w in flag + extra:
    if n not in why:
        sel.append(n)
        why[n] = w
    else:
        why[n] += w
rank = sorted((n for n in names if chg[n]["rva_replaced"] and n not in why), key=lambda n: -chg[n]["rva_max_dev_mm"])
for n in rank:
    if len(sel) >= 15:
        break
    sel.append(n)
    why[n] = ["RVA course change: max %.1f mm from v4's RVA" % chg[n]["rva_max_dev_mm"]]
rest = [n for n in names if n not in why]
rng = np.random.default_rng(20261002)
rnd = sorted(rng.choice(rest, 10, replace=False).tolist())
out = {"most_changed_15": sel, "why": why, "random_10": rnd, "rule": __doc__, "rva_change": chg}
json.dump(out, open("/sp/v5_va/checks/sofa_selection.json", "w"), indent=1)
for n in sel:
    print("MC", n, why[n], chg[n])
for n in rnd:
    print("RND", n, chg[n])
