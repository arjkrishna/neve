# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_mh.py (= /sp/v6/s3fix/patch_mh.py in the build containers)
F = "/sp/v6/records/MANIFEST_head.md"
s = open(F).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)
rep("- Copied 2026-10-07 by the `s2_records` agent of the v6 stage-2 workflow (wf_87bbef23-7e4), with\n  `records/records/make_tools_v6.py`.",
    "- Copied 2026-10-07 by the `s2_records` agent of the v6 stage-2 workflow (wf_87bbef23-7e4), with\n  `records/records/make_tools_v6.py`; re-staged 2026-10-08 by `s3_fix` (the final verifier's must-fix list) with\n  make_tools_v6 1.1, which adds the `s3_fix/` folder, the s3_fix manifest and the frozen s2_records versions.")
rep("| `records/` | this stage: `BUILD_v6.json` generator and checker, the `_records` copy, this copy (`make_tools_v6.py`, `finalize_tools.sh`, `imports_check.py`, the restore script and this file's head) |",
    "| `records/` | this stage: `BUILD_v6.json` generator and checker, the `_records` copy, this copy (`make_tools_v6.py`, `finalize_tools.sh`, `imports_check.py`, the restore script and this file's head); s3_fix's versions 1.1 (`build_v6_record`, `check_build_record`, `make_tools_v6`, `finalize_tools`) with the s2_records 1.0 versions frozen as `*_v1_0.*` |\n"
    "| `s3_fix/` | the final verifier's must-fix list (2026-10-08): the lost-wall diagnoses (`diag_*.py`), the re-bake / check runners (`batch_rebake.sh`, `final_other.sh`, `batch_sofa.sh`, `batch_cov.sh`, `dr_s3.sh`), the OBJ fold-over fix (`fold_fix.py` 1.0), the provenance stamp (`stamp_prov_s3.py` = stamp_prov 1.1), the copy of the changed anatomies (`copy_v6_s3.sh` = copy_v6 1.2), the coverage / SOFA summaries, the install script and the item lists of the final round. `bake_v6` 1.2 and `checks_v6` 1.6 themselves are in `build/` |")
rep("| `TOOLCHAIN_s2build.md5` (**d65b7209**) | 72 | + checks_v6 1.5 (D27), stamp_prov, copy_v6 1.1, v6_build_collect 1.3 | route_d22 on case_k_011_left__topcow_mr_012_L; the provenance stamp (`v6_checks.toolchain_manifest_md5`), the copy into anatomies_v6 |",
    "| `TOOLCHAIN_s2build.md5` (d65b7209) | 72 | + checks_v6 1.5 (D27), stamp_prov, copy_v6 1.1, v6_build_collect 1.3 | route_d22 on case_k_011_left__topcow_mr_012_L; the provenance stamp (`v6_checks.toolchain_manifest_md5`) and the copy of the 206 anatomies s3_fix did not change |\n"
    "| `TOOLCHAIN_s3fix.md5` (**2288f778**, final) | 97 | d65b7209's files with **bake_v6 1.2** (L1 lattice shift, L2 route-island bridge; both opt-in, defaults = 1.1) and **checks_v6 1.6** (parts coverage + folds), the frozen 1.1 / 1.5 copies, and the s3_fix scripts | the 8 re-bakes, the OBJ fold fix, checks_v6 1.6 on the 17 changed anatomies (and its coverage / folds parts on all 223), their provenance stamp and copy |")
rep("`_v1_4.py`, `full/check4_helper_v1_0.py`, `full/copy_v6_v1_0.sh`, `full/v6_build_collect_v1_2.py`.",
    "`_v1_4.py`, `_v1_5.py`, `bake/bake_v6_v1_1.py`, `full/check4_helper_v1_0.py`, `full/copy_v6_v1_0.sh`,\n`full/v6_build_collect_v1_2.py`.")
rep("graft_v6_va 1.4.2, bake_v6 1.1, checks_v6 parts 1.3 with route_d22 1.4 (222) / 1.5 (1).",
    "graft_v6_va 1.4.2, bake_v6 1.1, checks_v6 parts 1.3 with route_d22 1.4 (222) / 1.5 (1); on the 17 anatomies s3_fix\nchanged: bake_v6 1.2 (the 8 re-baked), checks_v6 1.6 (every part on the 9 whose geometry changed; coverage and folds on\nthe 8 re-stamped for a flag).")
rep("- **records**: BUILD_v6.json, _records, this copy, CAROTID_V6.md (s2_records, 2026-10-07)",
    "- **records**: BUILD_v6.json, _records, this copy, CAROTID_V6.md (s2_records, 2026-10-07)\n- **verify**: the final verifier (verify_v6_final, 2026-10-07): NO-GO on lost real wall (8 pieces), records, one fold-over\n- **s3_fix**: the must-fix list under D27 (2026-10-08): coverage check, 8 re-bakes, 8 flags, the fold fix, records 1.1")
rep("| records | `records/` build_v6_record, check_build_record, copy_records, make_tools_v6, dr_rec | records |",
    "| records | `records/` build_v6_record, check_build_record, copy_records, make_tools_v6, dr_rec | records, s3_fix (1.1) |\n"
    "| build | `bake/bake_v6.py` 1.2, `checks/checks_v6.py` 1.6 (+ `bake_v6_v1_1.py`, `checks_v6_v1_5.py`) | s3_fix |\n"
    "| s3_fix | `s3fix/` diagnoses, runners, fold_fix, stamp_prov_s3, copy_v6_s3, copy_records_s3, cov_results, sofa_summary, install_s3fix_tools | s3_fix |")
open(F, "w").write(s)
print("ok")
