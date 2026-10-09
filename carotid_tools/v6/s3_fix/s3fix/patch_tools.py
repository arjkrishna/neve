# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_tools.py (= /sp/v6/s3fix/patch_tools.py in the build containers)
import sys
R = "/sp/v6/records/"
def patch(F, pairs):
    s = open(F).read()
    for a, b in pairs:
        assert s.count(a) == 1, (F, a[:70], s.count(a))
        s = s.replace(a, b)
    open(F, "w").write(s)
patch(R + "make_tools_v6.py", [
 ('VERSION = "make_tools_v6 1.0 (s2_records, 2026-10-07)"', 'VERSION = "make_tools_v6 1.1 (s3_fix, 2026-10-08; 1.0 s2_records 2026-10-07)"'),
 ('             "v6/full/TOOLCHAIN_s2build_supplement.md5"]',
  '             "v6/full/TOOLCHAIN_s2build_supplement.md5", "v6/build/TOOLCHAIN_s3fix.md5"]   # 1.1: + the s3_fix manifest'),
 ('    if rel.startswith(("v5_va/", "v4_build/")):\n        return "v5_deps"\n',
  '    if rel.startswith(("v5_va/", "v4_build/")):\n        return "v5_deps"\n    if rel.startswith("v6/s3fix/"):\n        return "s3_fix"\n'),
 ('RECS = ["v6/records/" + f for f in ("make_tools_v6.py", "build_v6_record.py", "check_build_record.py", "copy_records.sh", "dr_rec.sh",\n                                    "finalize_tools.sh", "imports_check.py", "restore_layout.sh", "MANIFEST_head.md")]',
  'RECS = ["v6/records/" + f for f in ("make_tools_v6.py", "build_v6_record.py", "check_build_record.py", "copy_records.sh", "dr_rec.sh",\n                                    "finalize_tools.sh", "imports_check.py", "restore_layout.sh", "MANIFEST_head.md",\n'
  '                                    # 1.1 (s3_fix): the s2_records versions, frozen before s3_fix changed them\n'
  '                                    "make_tools_v6_v1_0.py", "build_v6_record_v1_0.py", "check_build_record_v1_0.py",\n'
  '                                    "copy_records_v1_0.sh", "finalize_tools_v1_0.sh", "MANIFEST_head_v1_0.md")]\n'
  '# 1.1 (s3_fix): the s3_fix scripts outside its manifest (records, summaries, the patch scripts that made the records 1.1\n'
  '# versions) and the item lists that defined the final round\n'
  'S3X = ["v6/s3fix/" + f for f in ("cov_results.py", "sofa_summary.py", "make_fig_flags.py", "copy_records_s3.sh", "patch_bvr.py",\n'
  '                                 "patch_cbr.py", "patch_tools.py", "diag_ringclose.py", "final_rebake.txt", "final_other.txt",\n'
  '                                 "final_sofa.txt", "stamp_items.tsv", "names8.txt", "names13.txt")]'),
 ('for lst, st in ((BUILDX, "build"), (DONOR, "donor"), (GATES, "gates"), (ITEMB, "item_b"), (SETC, "set_checks"), (RECS, "records")):',
  'for lst, st in ((BUILDX, "build"), (DONOR, "donor"), (GATES, "gates"), (ITEMB, "item_b"), (SETC, "set_checks"), (RECS, "records"),\n                (S3X, "s3_fix")):'),
])
patch(R + "finalize_tools.sh", [
 ('# s2_records: (re)stage carotid_tools/v6 and install it. Host side (Git Bash).',
  '# finalize_tools 1.1 (s3_fix, 2026-10-08; 1.0 s2_records): (re)stage carotid_tools/v6 and install it. Host side (Git Bash).\n'
  '# 1.1: the restore test runs md5sum -c on the FINAL manifest TOOLCHAIN_s3fix.md5 and checks that every entry of every\n'
  '# earlier manifest resolves (TOOLCHAINS.tsv: current file / frozen copy / repo module, md5-equal).'),
 ('  ( cd $RT && sed -n \'s#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p\' v6/build/TOOLCHAIN_s2build.md5 | md5sum -c --quiet && echo "final manifest d65b7209: every /sp entry OK on the restored layout ($(sed -n \'s#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p\' v6/build/TOOLCHAIN_s2build.md5 | wc -l) entries)" )',
  '  ( cd $RT && sed -n \'s#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p\' v6/build/TOOLCHAIN_s3fix.md5 | md5sum -c --quiet && echo "final manifest $(md5sum < v6/build/TOOLCHAIN_s3fix.md5 | cut -c1-8): every /sp entry OK on the restored layout ($(sed -n \'s#^\([0-9a-f]\{32\}\) \*/sp/#\1 *#p\' v6/build/TOOLCHAIN_s3fix.md5 | wc -l) entries)" )\n'
  '  echo "earlier manifests: $(awk -F\'\t\' \'NR>1 && $5 ~ /^(current file|frozen copy|repo module, md5 matches)/\' $T/TOOLCHAINS.tsv | wc -l) rows resolved, $(awk -F\'\t\' \'NR>1 && $5 == "NOT FOUND"\' $T/TOOLCHAINS.tsv | wc -l) not preserved (listed in MANIFEST.md)"'),
 ('grep -q "sha256 mismatches: 0" $R/out/restore_test.txt && grep -q ", 0 differ" $R/out/restore_test.txt && grep -q "every /sp entry OK" $R/out/restore_test.txt',
  'grep -q "sha256 mismatches: 0" $R/out/restore_test.txt && grep -q ", 0 differ" $R/out/restore_test.txt && grep -q "every /sp entry OK" $R/out/restore_test.txt && grep -q "^final manifest 2288f778" $R/out/restore_test.txt'),
])
print("patched")
