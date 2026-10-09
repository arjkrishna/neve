#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\junction_records_d17.py (= /sp/v6/wall_d17/junction_records_d17.py in the build containers)
"""junction_records_d17.py (s2_harden_wall_bake, D16 + D17; answers wall/requests.json va-D16-1 / va-D17-1): the VA
tool's junction records for the D17 walls = v6_va's sc_post3.py (imported unchanged) with its folders pointed at:
  WALL  /repo/carotid_data/v6_work/wall/d17/local   (local_sub 1.1 on the D17 request runs: the donor subclavian to
        the end of its segmentation + the junction, D15 zones; D16 tube on ia172_164 L / ia172_071 L; .junction_v3)
  PATHS the VA tool's own D17 paths /repo/carotid_data/v6_work/va_s2/paths_d17 (same code as stage 1, no distal cut;
        checked here point for point against this agent's wall/d17/paths up to the shorter end -- the arclength the
        records use); fallback wall/d17/paths when absent or different.
  OUT   /v6w/d17/_junction_records/<donor>/<R|L>/junction.json (published over donors_local/_junction_records by
        publish_d17.py after the checks)
Added per record (the VA agent's schema): d17 {covered_s_mm, label_end_past_ostium_mm, path_file, surface, how, ...};
d16_tube {tube_mm, tube_r_mm, from_xyz, to_xyz, bottleneck_ratio_before, bottleneck_ratio_after, how} on D16 sides.
    python3 junction_records_d17.py [donor ...]        (neve-build-stagea)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/va")
import sc_post3 as S3  # noqa: E402

VERSION = "junction_records_d17 1.0 (2026-10-04, s2_harden_wall_bake)"
MINE = "/repo/carotid_data/v6_work/wall/d17/paths"
VAS = "/repo/carotid_data/v6_work/va_s2/paths_d17"
S3.WALL = "/repo/carotid_data/v6_work/wall/d17/local"
S3.OUT = "/v6w/d17/_junction_records"


def same_paths():
    rows, ok = {}, os.path.isdir(VAS)
    if not ok:
        return False, {"why": "no %s" % VAS}
    for f in sorted(os.listdir(MINE)):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        g = os.path.join(VAS, f)
        if not os.path.isfile(g):
            ok = False
            rows[f[:-5]] = "missing in the VA tool's paths"
            continue
        a = np.asarray(json.load(open(os.path.join(MINE, f)))["path"], float)
        b = np.asarray(json.load(open(g))["path"], float)
        n = min(len(a), len(b))
        d = float(np.abs(a[:n] - b[:n]).max())
        rows[f[:-5]] = {"n_common": n, "n_wall": len(a), "n_va": len(b), "maxdiff_mm": d}
        if d > 1e-6:
            ok = False
    return ok, rows


if __name__ == "__main__":
    ok, cmp_ = same_paths()
    S3.PATHS = VAS if ok else MINE
    print("PATHS", S3.PATHS, "(VA tool's paths identical to the wall's up to the shorter end: %s)" % ok, flush=True)
    os.makedirs(S3.OUT, exist_ok=True)
    json.dump({"script": VERSION, "paths_used": S3.PATHS, "identical": ok, "per_side": cmp_},
              open(os.path.join(S3.OUT, "_paths_check.json"), "w"), indent=1)
    S3.main()
    for dn in sorted(os.listdir(S3.OUT)):
        for sd in ("R", "L"):
            f = os.path.join(S3.OUT, dn, sd, "junction.json")
            if not os.path.isfile(f):
                continue
            o = json.load(open(f))
            pf = os.path.join(S3.PATHS, "%s_%s.json" % (dn, sd))
            pj = json.load(open(pf))
            mj = json.load(open(os.path.join(MINE, "%s_%s.json" % (dn, sd))))
            cov = o.get("covered_s_mm") or [None, None]
            past = (cov[1] - o["s_ostium_mm"]) if cov[1] is not None else None
            o["d17"] = {"covered_s_mm": cov, "label_end_past_ostium_mm": past, "path_file": pf, "surface": o.get("surface"),
                        "how": "D17 (user): the donor's own subclavian label + junction to the end of its segmentation (wall "
                               "agent request s_to = the end of the Arterial xSA chain, clipped where its calibre collapses "
                               "below 0.5 x the subclavian's own about the ostium (%s)); D15 local wall (plateau rule in the "
                               "artefact zones, the v5-rule coarse wall outside them); ONE closed surface (junction + the "
                               "whole subclavian stretch)" % mj["d17"]["end_how"],
                        "request_s_to_mm": mj["path_len_mm"], "path_len_mm": pj.get("path_len_mm"),
                        "tube_extension_needed": bool(past is None or past < 30.0), "script": VERSION}
            jb = o.get("junction_bridge") or {}
            d16 = jb.get("d16")
            if d16:
                o["d16_tube"] = {"tube_mm": d16["tubed_mm"], "tube_r_mm": d16["r_mm"], "from_xyz": d16["from_xyz"],
                                 "to_xyz": d16["to_xyz"], "bottleneck_ratio_before": d16["ratio_before"],
                                 "bottleneck_ratio_after": d16["ratio_after"], "S_mm": d16["S_mm"], "steps": d16["steps"],
                                 "how": d16["rule"] + "; frame " + d16.get("frame", "donor RAS mm")}
            json.dump(o, open(f, "w"), indent=1)
            print(dn, sd, "covered", [None if v is None else round(v, 1) for v in cov], "past %.1f" % (past if past is not None else -1),
                  "failed", o.get("junction_failed"), "d16" if d16 else "", flush=True)
    print("D17 records done", flush=True)
