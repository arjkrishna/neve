# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_meta_visual.py (= /sp/v5_va/pilot_ia172_meta_visual.py in the build containers)
"""pilot_ia172_meta_visual: merge my visual reading (results/pilot_ia172_visual.json) into each case's meta.json and
qc.json (visual_check + verdict) without recomputing the quick look.  usage: python3 pilot_ia172_meta_visual.py <case> ..."""
import json
import os
import sys

VIS = json.load(open("/sp/v5_va/results/pilot_ia172_visual.json"))
for case in sys.argv[1:]:
    for p in ("/repo/carotid_data/va_sources/thin_slice/ia172/%s/meta.json" % case,
              "/repo/carotid_data/va_sources/derived/thin_slice/ia172/%s/qc.json" % case):
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        v = VIS.get(case)
        if v:
            m["visual_check"] = v
            m["verdict"] = v["verdict"]
        json.dump(m, open(p, "w"), indent=1)
    print(case, m["verdict"])
