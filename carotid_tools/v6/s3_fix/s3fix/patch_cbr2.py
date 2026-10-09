# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_cbr2.py (= /sp/v6/s3fix/patch_cbr2.py in the build containers)
F = "/sp/v6/records/check_build_record.py"
s = open(F).read()
a = '       "information_vs_build_collection_and_set_checks": {"rows": info_rows, "not_equal": [r for r in info_rows if not r["equal"]]}}'
b = ('       "information_vs_build_collection_and_set_checks": {"rows": info_rows, "not_equal": [r for r in info_rows if not r["equal"]],\n'
     '           "note": "1.1 (s3_fix): results/v6_build.json and v6_final_summary.json are s2 records, written before s3_fix changed 17 "\n'
     '                   "anatomies (2026-10-08: 8 re-baked, 1 fold fix, 8 re-stamped for a flag); every not-equal row on D22 references, "\n'
     '                   "D26 / D27 lists, bake rungs and check flags is that change (BUILD_v6.json -> s3_fix); the boundary case "\n'
     '                   "va_target 19.999 is explained in its row"}}')
assert s.count(a) == 1
s = s.replace(a, b)
open(F, "w").write(s)
print("ok")
