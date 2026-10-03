#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\ostium_lumen.py (= /sp/v5_va/checks/ostium_lumen.py in the build containers)
"""v5_checks item 4 diagnostic: the RVA lumen in the OBJ around the VA ostium (s_o - 3 .. s_o + 10 mm) for the 25 SOFA
anatomies, v5 and v4 (v4's ostium = the (17) take-off): min wall distance and declared radius there (v4 audit's exact
point-triangle distance).   -> /sp/v5_va/checks/ostium_lumen.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/sp/v4_build/audit")
import topo_audit as TA  # noqa: E402
sel = json.load(open("/sp/v5_va/checks/sofa_selection.json"))
names = sel["most_changed_15"] + sel["random_10"]
S = json.load(open("/sp/v5_va/results/v5_checks.json"))["item4b_sofa"]["per_anatomy"]
out = []
for n in names:
    row = {"name": n}
    for v, root in (("v5", "/repo/carotid_data/anatomies_v5/" + n), ("v4", "/repo/carotid_data/anatomies_v4/" + n)):
        br = {t: (p, r) for t, p, r in TA.load_branches(os.path.join(root, "Centrelines_comb"))}
        p, r = br["RVA"]
        s = TA.arclen(p)
        rv = S[n][v]["rva"]
        so = rv["ostium_mm"] if v == "v5" else rv["s_17_takeoff_mm"]
        m = (s >= so - 3.0) & (s <= so + 10.0)
        V, F, _ = TA.read_obj(os.path.join(root, "vessel_architecture_collision.obj"))
        d = TA.surface_dist(p[m], V, F)
        k = int(np.argmin(d))
        row[v] = {"ostium_mm": so, "lumen_min_mm": round(float(d.min()), 3), "at_s_mm": round(float(s[m][k]), 1),
                  "r_there_mm": round(float(r[m][k]), 2), "r_min_mm": round(float(r[m].min()), 2),
                  "entered": S[n][v]["entered_va_20mm_past_ostium"]}
    out.append(row)
json.dump(out, open("/sp/v5_va/checks/ostium_lumen.json", "w"), indent=1)
for v in ("v5", "v4"):
    for e in (True, False):
        x = [r[v]["lumen_min_mm"] for r in out if r[v]["entered"] == e]
        if x:
            print(v, "entered" if e else "not entered", len(x), "ostium lumen_min median %.3f min %.3f" % (np.median(x), min(x)))
