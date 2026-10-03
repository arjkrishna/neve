#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_eye172_bridges.py (= /sp/v5_va/v5_eye172_bridges.py in the build containers)
"""v5_eye172_bridges.py -- (v5_eye172) print, per pending side, the qc record items the by-eye verdict needs: bridges
(position / length / segment), landmarks, foramina passed, V3 loop, HU/CCA, rule-6 agreement, gate status. Read-only."""
import json

B = "/repo/carotid_data/va_sources/derived/thin_slice/"
ITEMS = [("ia172", "ia172_058", "L", "qc.json"), ("ia172", "ia172_039", "R", "qc.json"), ("ia172", "ia172_039", "L", "qc.json"),
         ("ia172", "ia172_015", "R", "qc.json"), ("ia172", "ia172_190", "R", "qc.json"), ("ia172", "ia172_190", "L", "qc.json"),
         ("ia172", "ia172_141", "R", "qc.json"), ("ia172", "ia172_172", "L", "qc.json"), ("ia172", "ia172_172", "R", "qc.json"),
         ("song", "AHMU1218138", "R", "qc_tsba_v1r.json"), ("song", "AHMU1218138", "L", "qc.json")]


def short(v, n=300):
    return json.dumps(v, default=str)[:n]


for src, case, side, qf in ITEMS:
    Q = json.load(open(B + "%s/%s/qc/%s" % (src, case, qf)))
    s = Q["sides"][side]
    print("=== %s %s %s (%s)" % (src, case, side, qf))
    for k in ("landmarks_used_mm", "entry_level_ts_qc", "entry_level_Erec", "foramen_levels_passed", "n_C6_to_C1_passed",
              "bridges", "segment_bridges", "length_trace_mm", "s_leave_sa_mm"):
        if k in s:
            print("  %s: %s" % (k, short(s[k])))
    for k, v in s.items():
        if isinstance(v, dict) and any(x in k for x in ("gate", "recal", "verdict", "hu", "HU", "v3", "V3", "label", "rule")):
            print("  %s: %s" % (k, short(v, 400)))
