# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_bvr2.py (= /sp/v6/s3fix/patch_bvr2.py in the build containers)
F = "/sp/v6/records/build_v6_record.py"
s = open(F).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)
rep('ap.add_argument("--checks4", default="/repo/carotid_data/anatomies_v6/_records/per_anatomy/checks4")',
    'ap.add_argument("--checks4", default="/repo/carotid_data/anatomies_v6/_records/per_anatomy/checks4")\n'
    'ap.add_argument("--s3set", default="/sp/v6/results/v6_s3_set.json")   # s3_fix set-level checks after the fix (labelled copy)')
rep('                 "changed": L["s3_fix"],\n',
    '                 "changed": L["s3_fix"],\n'
    '                 "set_checks_after_fix": dict(jl(a.s3set), file="_records/results/s3_fix/ (split_s3.json, loader_s3.json, sofa_rcca_summary.json); set-level copy"),\n')
open(F, "w").write(s)
print("ok")
