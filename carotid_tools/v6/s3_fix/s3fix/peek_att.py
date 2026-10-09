# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\peek_att.py (= /sp/v6/s3fix/peek_att.py in the build containers)
import json, sys
L = [("w_052_left__topcow_mr_026","fork"),("w_027_left__topcow_mr_022","fork"),("w_027_left__topcow_mr_026","fork"),
     ("k_011_left__topcow_mr_012_L","fork"),("w_002_right__topcow_mr_013_L","fork"),("w_051_right__topcow_mr_008_L","siphon"),
     ("w_012_left__topcow_mr_003_L","fork"),("w_008_right__topcow_mr_003","fork"),
     ("w_012_left__topcow_mr_001","fork"),("w_017_right__topcow_mr_011","siphon"),("w_035_left__topcow_mr_015_L","siphon"),
     ("w_012_left__topcow_mr_013_L","fork"),("w_025_right__topcow_mr_021_L","fork")]
rows = {r[0]: r for r in json.load(open('/w/verify_final/d22b/cov_rows.json'))}
for n, pc in L:
    full = "case_" + n
    d = json.load(open('/repo/carotid_data/anatomies_v6/%s/mesh_v6.json' % full))
    d5 = json.load(open('/repo/carotid_data/anatomies_v5/%s/mesh_v5.json' % full))
    r = [x for x in json.load(open('/w/verify_final/d22b/cov_rows.json')) if x[0] == n and x[1] == pc]
    print("==", n, pc, "cov", r[0][2] if r else None, "unc", r[0][4] if r else None, "s_runs", r[0][12] if r else None)
    print("  v6 rung", d["ladder_v5b"]["rung"], "sup", d["suppressed_stretches_mm"], "tube_only", d["sections_tube_only"], "farclamp", d["far_clamp"]["used"])
    print("  v5 rung", d5["ladder_v5b"]["rung"], "sup", d5["suppressed_stretches_mm"], "final dropped v5", d5["attempts"][-1]["dropped_volume_mm3"])
    for a in d["attempts"]:
        tag = {k: v for k, v in a.items() if k not in ("genus","nonmanifold_edges","components_before_keep","dropped_fragments","dropped_volume_mm3","seconds")}
        print("   ", json.dumps(tag)[:110], "g", a["genus"], "comps", a["components_before_keep"], "drop", a["dropped_volume_mm3"], a["dropped_fragments"][:6])
