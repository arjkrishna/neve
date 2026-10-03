#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\check_inputs.py (= /sp/v5_va/bake_full/check_inputs.py in the build containers)
"""v5_bake_full: are the NON-VA bake inputs of every v5 anatomy byte-identical to v4's? (every centreline file except
RVA / LVA, and every provenance xform entry except va_R / va_L / vaC_*). If so, and the carotid sections are baked
pinned to v4 (no 'carotid_changed' flag), any right-route difference from v4 can only come from the VAs' new course.
Writes /sp/v5_va/bake_full/check_inputs.json.     python3 check_inputs.py   (neve-build-meshlab)
"""
import filecmp
import glob
import json
import os

A5, A4 = "/repo/carotid_data/anatomies_v5", "/repo/carotid_data/anatomies_v4"
out, bad = {}, []
for d in sorted(glob.glob(A5 + "/*/provenance.json")):
    n = os.path.basename(os.path.dirname(d))
    c5, c4 = os.path.join(A5, n, "Centrelines_comb"), os.path.join(A4, n, "Centrelines_comb")
    f5, f4 = sorted(os.listdir(c5)), sorted(os.listdir(c4))
    diff = [f for f in f4 if f not in f5 or not filecmp.cmp(os.path.join(c4, f), os.path.join(c5, f), shallow=False)]
    extra = [f for f in f5 if f not in f4]
    nonva = [f for f in diff + extra if not (f.endswith("RVA.mrk.json") or f.endswith("LVA.mrk.json"))]
    x5, x4 = json.load(open(d))["xform"], json.load(open(os.path.join(A4, n, "provenance.json")))["xform"]
    xd = [k for k in set(x4) | set(x5) if not k.startswith("va") and x4.get(k) != x5.get(k)]
    out[n] = {"centreline_files_changed": diff + extra, "non_va_changed": nonva, "xform_non_va_changed": xd}
    if nonva or xd:
        bad.append(n)
json.dump({"anatomies": len(out), "non_va_inputs_changed": bad, "per_anatomy": out},
          open("/sp/v5_va/bake_full/check_inputs.json", "w"), indent=1)
print("anatomies", len(out), "with non-VA input changes:", bad)
print("centreline change patterns:", sorted({tuple(v["centreline_files_changed"]) for v in out.values()}))
