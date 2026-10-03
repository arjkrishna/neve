#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\lumen_iso.py (= /sp/v5_va/route13/lumen_iso.py in the build containers)
"""v5_route13: RCCA / LCCA route min lumen on the UNDECIMATED unions (v5 as baked: iso_v5/<name>.vtp; chosen variant:
route13/<variant>_iso/<name>.vtp), with bake_meshes_v4.route_report (unchanged), to separate real lumen changes from
the decimated-OBJ scatter that mesh_v5.json's lumen_min_mm carries.  Read-only.  Inside neve-build-meshlab:
    python3 /sp/v5_va/route13/lumen_iso.py            -> /sp/v5_va/route13/lumen_iso.json"""
import json
import os
import sys

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
import pyvista as pv  # noqa: E402
from eve_bench.dualdevicenav import load_branches  # noqa: E402

R13 = "/repo/carotid_data/va_sources/v5_reg/route13"
ISO5 = "/repo/carotid_data/va_sources/v5_reg/iso_v5"
A5 = "/repo/carotid_data/anatomies_v5"
res = json.load(open("/sp/v5_va/results/v5_route13.json"))
out = {}
for n, e in res["anatomies"].items():
    ch = e["chosen"]["variant"]
    row = {"chosen": ch}
    for tag, iso, root in (("v5", os.path.join(ISO5, n + ".vtp"), os.path.join(A5, n)),
                           (ch, os.path.join(R13, ch + "_iso", n + ".vtp"), os.path.join(R13, ch, n))):
        br = load_branches(os.path.join(root, "Centrelines_comb"))
        m = pv.read(iso)
        rr, ok = B.route_report(m, br)
        row[tag] = {k: {kk: rr[k].get(kk) for kk in ("lumen_min_mm", "lumen_min_at_mm", "navigable")} for k in ("RCCA", "LCCA")}
    out[n] = row
    print(n[5:40], json.dumps(row), flush=True)
with open("/sp/v5_va/route13/lumen_iso.json", "w") as f:
    json.dump(out, f, indent=1)
