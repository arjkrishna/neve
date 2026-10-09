# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_tools2.py (= /sp/v6/s3fix/patch_tools2.py in the build containers)
F = "/sp/v6/records/make_tools_v6.py"
s = open(F).read()
a = ('S3X = ["v6/s3fix/" + f for f in ("cov_results.py", "sofa_summary.py", "make_fig_flags.py", "copy_records_s3.sh", "patch_bvr.py",\n'
     '                                 "patch_cbr.py", "patch_tools.py", "diag_ringclose.py", "final_rebake.txt", "final_other.txt",\n'
     '                                 "final_sofa.txt", "stamp_items.tsv", "names8.txt", "names13.txt")]')
b = ('import glob as _glob   # 1.1: every file of the s3_fix folder (scripts, item lists, the records README it wrote)\n'
     'S3X = sorted("v6/s3fix/" + os.path.basename(p) for p in _glob.glob(os.path.join(SP, "v6/s3fix/*"))\n'
     '             if os.path.isfile(p) and not p.endswith((".pyc",)))')
assert s.count(a) == 1
s = s.replace(a, b)
open(F, "w").write(s)
print("ok")
