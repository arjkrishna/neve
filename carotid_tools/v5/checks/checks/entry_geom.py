#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\entry_geom.py (= /sp/v5_va/checks/entry_geom.py in the build containers)
"""v5_checks item 4 diagnostic: the RVA entry geometry the SOFA schedule meets at fork 2 (the VA leaving the
subclavian), v5 vs v4, all 223: the turn from the parent's axis (RVA s(17)-8 -> s(17)) to the VA's first 8 mm past its
ostium; the angle between the VA's first 8 mm and the subclavian continuation (17)'s first 8 mm; the junction stub
length (ostium - s(17)).   -> /sp/v5_va/checks/entry_geom.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/sp/v4_build/audit")
import topo_audit as TA  # noqa: E402
A5, V4 = "/repo/carotid_data/anatomies_v5", "/repo/carotid_data/anatomies_v4"


def d(p, s, a, b):
    i, j = int(np.argmin(abs(s - a))), int(np.argmin(abs(s - b)))
    v = p[j] - p[i]
    return v / np.linalg.norm(v)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


out = []
for n in sorted(x for x in os.listdir(A5) if x.startswith("case_")):
    row = {"name": n}
    for tag, root in (("v5", os.path.join(A5, n)), ("v4", os.path.join(V4, n))):
        br = {t: p for t, p, r in TA.load_branches(os.path.join(root, "Centrelines_comb"))}
        p, p17 = br["RVA"], br["(17)"]
        s, s17a = TA.arclen(p), TA.arclen(p17)
        k17 = float(s[int(np.argmin(np.linalg.norm(p - p17[0], axis=1)))])
        prov = json.load(open(os.path.join(root, "provenance.json")))
        ost = k17
        if tag == "v5" and "vaC_R" in prov["xform"]:
            ost = max(float(prov["v5_va"]["right"].get("ostium_at_mm", k17)), k17)
        t_par, d_va, d_17 = d(p, s, k17 - 8, k17), d(p, s, ost, ost + 8), d(p17, s17a, 0, 8)
        row[tag] = {"s17_mm": round(k17, 1), "ostium_mm": round(ost, 1), "stub_mm": round(ost - k17, 1),
                    "turn_parent_to_va_deg": round(ang(t_par, d_va), 1), "va_vs_17_deg": round(ang(d_va, d_17), 1),
                    "turn_parent_to_17_deg": round(ang(t_par, d_17), 1)}
    out.append(row)
json.dump(out, open("/sp/v5_va/checks/entry_geom.json", "w"), indent=1)
for tag in ("v5", "v4"):
    for k in ("turn_parent_to_va_deg", "va_vs_17_deg", "stub_mm"):
        x = np.array([r[tag][k] for r in out])
        print(tag, k, "p10/median/p90/max", np.percentile(x, [10, 50, 90, 100]).round(1).tolist())
