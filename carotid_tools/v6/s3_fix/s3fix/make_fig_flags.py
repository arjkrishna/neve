#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\make_fig_flags.py (= /sp/v6/s3fix/make_fig_flags.py in the build containers)
"""s3_fix: the extra-flags file for monitoring/figure_carotid_anatomies_v6.py (--build-json) for the anatomies s3_fix
changed: {"anatomies": {name: {"flags": provenance v6_checks.derived_flags}}} (the flags the build collection derived
outside provenance: left_pose_map_bounded_dev, orient:*), so the re-rendered titles carry the same flag set as the s2 renders.
    python3 make_fig_flags.py <names file> <out.json>"""
import json, sys
names = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]
out = {"anatomies": {}}
for n in names:
    p = json.load(open("/repo/carotid_data/anatomies_v6/%s/provenance.json" % n))
    out["anatomies"][n] = {"flags": list((p.get("v6_checks") or {}).get("derived_flags") or [])}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print({n: v["flags"] for n, v in out["anatomies"].items()})
