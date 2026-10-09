#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\publish_d17.py (= /sp/v6/wall_d17/publish_d17.py in the build containers)
"""publish_d17.py (s2_harden_wall_bake; requests va-D16-1 / va-D17-1): publish the D16 + D17 junction records over
the D15 ones where the VA tool reads them (donors_local/_junction_records/<donor>/<side>/junction.json), after checks:
every side present, status ok, the D16 sides not failed, covered_s_mm reaching past stage 1's coverage, surfaces
closed (open / non-manifold edges 0). The D15-only records are kept in donors_local/_junction_records_d15only/ (copied
once). Atomic per file (tmp + replace).   python3 publish_d17.py [--dry]   -> /v6w/d17/publish.json"""
import json
import os
import shutil
import sys
import time

SRC = "/v6w/d17/_junction_records"
DST = "/v6w/donors_local/_junction_records"
BAK = "/v6w/donors_local/_junction_records_d15only"
dry = "--dry" in sys.argv
rows, problems = {}, []
for d in sorted(os.listdir(SRC)):
    if not os.path.isdir(os.path.join(SRC, d)):
        continue
    for sd in ("R", "L"):
        f = os.path.join(SRC, d, sd, "junction.json")
        g = os.path.join(DST, d, sd, "junction.json")
        if not os.path.isfile(g):
            continue
        if not os.path.isfile(f):
            problems.append("%s %s: no D17 record" % (d, sd))
            continue
        a, b = json.load(open(f)), json.load(open(g))
        cov_a, cov_b = (a.get("covered_s_mm") or [None, None])[1], (b.get("covered_s_mm") or [None, None])[1]
        oe = (a.get("surface_info") or {}).get("open_or_nonmanifold_edges")
        r = {"status": a.get("status"), "covered_end_d15": cov_b, "covered_end_d17": cov_a, "junction_failed_d15": b.get("junction_failed"),
             "junction_failed_d17": a.get("junction_failed"), "open_edges": oe, "d16": bool(a.get("d16_tube")),
             "genus": (a.get("surface_info") or {}).get("genus_largest")}
        if a.get("status") != "ok":
            problems.append("%s %s: status %s" % (d, sd, a.get("status")))
        if oe:
            problems.append("%s %s: %s open / non-manifold edges" % (d, sd, oe))
        if cov_a is None or (cov_b is not None and cov_a < cov_b - 1.0):
            problems.append("%s %s: D17 coverage %s below D15's %s" % (d, sd, cov_a, cov_b))
        rows["%s:%s" % (d, sd)] = r
for k in ("ia172_164:L", "ia172_071:L"):
    if k in rows and rows[k]["junction_failed_d17"]:
        problems.append("%s: still junction_failed after D16 (%s)" % (k, rows[k]["junction_failed_d17"]))
out = {"date": time.strftime("%F %T"), "dry": dry, "sides": len(rows), "problems": problems, "rows": rows}
if not problems and not dry:
    if not os.path.exists(BAK):
        shutil.copytree(DST, BAK)
    for k in rows:
        d, sd = k.split(":")
        src, dst = os.path.join(SRC, d, sd, "junction.json"), os.path.join(DST, d, sd, "junction.json")
        shutil.copy2(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    for f in ("_summary.json", "_paths_check.json"):
        if os.path.exists(os.path.join(SRC, f)):
            shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
    open(os.path.join(DST, "_PUBLISHED_D16_D17.txt"), "w").write(
        "D16 + D17 records (wall_d17/junction_records_d17.py on wall/d17/local) published %s over the D15 records; the D15-only "
        "records are in ../_junction_records_d15only/\n" % time.strftime("%F %T"))
    out["published"] = True
json.dump(out, open("/v6w/d17/publish.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
