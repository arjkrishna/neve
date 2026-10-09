#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\gate_summary.py (= /sp/v6/wall/gate_summary.py in the build containers)
"""gate_summary.py (v6_wall): one table of every gate per variant, from the result files.
    python3 gate_summary.py  -> /v6w/gate_summary.json + printed table"""
import json
import os

import numpy as np

V = ["v5ref", "half", "plateau_g", "plateau"]
out = {}
for v in V:
    f = "/v6w/seg/%s/gate_ct.json" % v
    if os.path.exists(f):
        out.setdefault(v, {})["gate_ct"] = json.load(open(f))["summary"]
ss = "/v6w/surf_stats.json"
if os.path.exists(ss):
    S = json.load(open(ss))
    for v in ["v5", "half", "plateau_g", "plateau"]:
        rows = [r[v] for r in S.values() if v in r]
        if not rows:
            continue
        g = [r["genus_if_one"] for r in rows if r.get("genus_if_one") is not None]
        fl = [r["floored_mm"] for r in rows if "floored_mm" in r]
        out.setdefault("v5ref" if v == "v5" else v, {})["surfaces"] = {
            "sides": len(rows), "genus_median": float(np.median(g)) if g else None, "genus_sum": float(np.sum(g)) if g else None,
            "multi_component": sum(1 for r in rows if r["components"] != 1),
            "open_or_nonmanifold": sum(1 for r in rows if r["open_or_nonmanifold_edges"]),
            "floored_mm_total": float(np.sum(fl)) if fl else None, "floored_mm_median_per_side": float(np.median(fl)) if fl else None,
            "sides_with_floor": sum(1 for x in fl if x > 0) if fl else None,
            "misr_median_of_medians": float(np.median([r["misr_median"] for r in rows if "misr_median" in r])) if fl else None,
            "area_sum_mm2": float(np.sum([r["area_mm2"] for r in rows]))}
wd = "/v6w/wd/wd_collect.json"
if os.path.exists(wd):
    W = json.load(open(wd))["summary"]
    for k, v in W.items():
        if isinstance(v, dict) and "p10_med_p90_min_max" in v:
            out.setdefault({"v5": "v5ref"}.get(k, k), {})["wall_detail_proxy"] = v
    out["wd_calibration"] = W.get("calibration_v5proxy_vs_measured")
json.dump(out, open("/v6w/gate_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
