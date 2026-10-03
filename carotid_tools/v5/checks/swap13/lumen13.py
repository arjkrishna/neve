#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\lumen13.py (= /sp/v5_va/swap13/lumen13.py in the build containers)
"""v5_swap13: RCCA / LCCA route min lumen on the UNDECIMATED unions with bake_meshes_v4.route_report (unchanged), as
route13/lumen_iso.py: now (iso_v5 after the swap + anatomies_v5 centrelines), before the swap
(backup_pre_route13/_iso_v5 + backup centrelines) and v4 (vr13/v4iso rebuilt union + anatomies_v4 centrelines).
    python3 lumen13.py   -> /sp/v5_va/swap13/lumen13.json"""
import json
import os
import sys
sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
import pyvista as pv  # noqa: E402
from eve_bench.dualdevicenav import load_branches  # noqa: E402
BK = "/sp/v5_va/backup_pre_route13"
names = [x.strip() for x in open("/sp/v5_va/swap13/names13.txt") if x.strip()]
out = {}
for n in names:
    row = {}
    for tag, iso, root in (("now", "/repo/carotid_data/va_sources/v5_reg/iso_v5/%s.vtp" % n, "/repo/carotid_data/anatomies_v5/" + n),
                           ("before_swap", "%s/_iso_v5/%s.vtp" % (BK, n), "%s/%s" % (BK, n)),
                           ("v4", "/sp/v5_va/vr13/v4iso/%s.vtp" % n, "/repo/carotid_data/anatomies_v4/" + n)):
        if not os.path.exists(iso):
            row[tag] = None
            continue
        br = load_branches(os.path.join(root, "Centrelines_comb"))
        rr, ok = B.route_report(pv.read(iso), br)
        row[tag] = {k: {kk: rr[k].get(kk) for kk in ("lumen_min_mm", "lumen_min_at_mm", "navigable")} for k in ("RCCA", "LCCA")}
    row["now_equals_before"] = row["now"] == row["before_swap"]
    row["now_equals_v4"] = row["v4"] is not None and all(row["now"][k]["lumen_min_mm"] == row["v4"][k]["lumen_min_mm"] for k in ("RCCA", "LCCA"))
    out[n] = row
    print(n, json.dumps(row), flush=True)
json.dump(out, open("/sp/v5_va/swap13/lumen13.json", "w"), indent=1)
print("DONE lumen13")
