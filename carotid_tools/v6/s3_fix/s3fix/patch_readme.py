# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\patch_readme.py (= /sp/v6/s3fix/patch_readme.py in the build containers)
F = "/sp/v6/s3fix/README_records_new.md"
s = open("/sp/v6/s3fix/README_records_v1_0.md").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)
rep("`RECORDS.sha256` holds the sha256 of every file in this folder except itself and this README (1,156 files; check with\n`sha256sum -c RECORDS.sha256` from inside `_records/`).",
    "`RECORDS.sha256` holds the sha256 of every file in this folder except itself and this README (check with\n`sha256sum -c RECORDS.sha256` from inside `_records/`).\n\n"
    "**2026-10-08, s3_fix** (the final verifier's must-fix list; `notes/s3_fix.md`): added `results/v6_coverage_s3.json`,\n"
    "`results/s3_fix/`, `per_anatomy/checks_s3/`, `per_anatomy/bake_s3/`, `per_anatomy/fold_fix/`, `sofa/rcca_v6_s3/`,\n"
    "`checks/coverage_v1p6_prefix/`, `diag/s3_fix/`, `notes/s3_fix.md`, `notes/verify_v6_final.md`, `notes/s2_images.md`;\n"
    "replaced the grown originals `notes/DECISIONS_ADDENDUM.md` (logged in `results/s3_fix/records_replaced.txt`) and\n"
    "rebuilt `RECORDS.sha256` and `checks/build_record_check.json`. The files of the 17 anatomies s3_fix changed are\n"
    "superseded, not removed: their s2 checks records stay in `per_anatomy/checks4/`, their s2 copy manifest in\n"
    "`results/build/anatomies_v6.sha256`; the current ones are in `per_anatomy/checks_s3/` and `results/s3_fix/anatomies_v6.sha256`.")
rep("| `build/license_check.json` | the generated LICENSE_v6_donors.txt files: 223, 0 problems |",
    "| `build/license_check.json` | the generated LICENSE_v6_donors.txt files: 223, 0 problems |\n"
    "| `v6_coverage_s3.json` | **s3_fix**: checks_v6 1.6 parts coverage (against the v4 / v5 SHIPPED wall) + folds on all 223 of the final anatomies_v6: per anatomy and piece the real-wall deficit, lost area and runs; the pieces >= 1 % before / after the fix; the 0.1-1 % pieces accepted with a flag |\n"
    "| `s3_fix/anatomies_v6.sha256`, `s3_fix/copy_verify_s3.json` | the folder manifest after s3_fix (5,136 files; the 17 changed anatomies' lines replaced) and the verified copy |\n"
    "| `s3_fix/stamp_prov_s3.json` | the `v6_checks` block re-stamped into the 17 changed provenance files (stamp_prov 1.1) |\n"
    "| `s3_fix/sofa_rcca_summary.json` | SOFA on the RCCA of the 9 anatomies whose geometry changed, before / after; the all-223 and 25-anatomy totals after |\n"
    "| `s3_fix/split_s3.json`, `s3_fix/loader_s3.json` | split and the training loader on the final folder |\n"
    "| `s3_fix/final_rebake.txt`, `final_other.txt`, `final*.batch.log`, `coverage_before_fix_summary.txt`, `records_replaced.txt` | the final round's item lists and logs, the coverage on the pre-fix folder, the replaced notes |")
rep("| `diag_cov12/` | the coverage diagnosis of the 4 pieces whose v4 wall does not reappear (diag_cov 1.2) |",
    "| `diag_cov12/` | the coverage diagnosis of the 4 pieces whose v4 wall does not reappear (diag_cov 1.2) |\n"
    "| `coverage_v1p6_prefix/<name>.json` | **s3_fix**: checks_v6 1.6 parts coverage + folds on the PRE-fix anatomies_v6 (all 223): exactly the final verifier's 8 pieces >= 1 % |")
rep("| `tube_genus_prebake/<name>.json` | the pre-bake scan of the bake's tube-only union (genus) on the final composed trees |",
    "| `tube_genus_prebake/<name>.json` | the pre-bake scan of the bake's tube-only union (genus) on the final composed trees |\n"
    "| `checks_s3/<name>.json` | **s3_fix**: the final checks record (checks_v6 1.6) of the 17 changed anatomies: every part on the 9 whose geometry changed, coverage + folds on the 8 re-stamped for a flag (the other parts carried from `checks4/`) |\n"
    "| `bake_s3/<name>.bake.json` | the done markers of the 8 re-bakes (bake_v6 1.2: lattice shift / route-island bridge, timings, md5) |\n"
    "| `fold_fix/<name>.json` | the OBJ fold-over fix of case_m_022_left__topcow_mr_021_L (fold_fix 1.0) |")
rep("`bif_angles_w042.json` (why the scripted wire takes w_042_left's ECA).",
    "`bif_angles_w042.json` (why the scripted wire takes w_042_left's ECA). `rcca_v6_s3/` (s3_fix): the RCCA runs of the 9\n"
    "anatomies whose geometry s3_fix changed (they replace those anatomies' `rcca_v6/` records).")
rep("(`d22_w008_mr001_fork.json`).",
    "(`d22_w008_mr001_fork.json`). `s3_fix/` (s3_fix): the lost-wall diagnosis (`frag_*.json`: the dropped islands;\n"
    "`sign_w052.json`), the lattice-shift screen (`shift/`), and the dev re-bake variants (`dev_rebake/<tag>/`: A, B, C0-C3,\n"
    "D1-D3 as `notes/s3_fix.md` section 3.1 lists them).")
rep("  `verify_s2_gate`, `s2_integrate2`, `verify_s2_gate2`, `s2_build`, `s2_checks`, `s2_records`.",
    "  `verify_s2_gate`, `s2_integrate2`, `verify_s2_gate2`, `s2_build`, `s2_checks`, `s2_records`, `s2_images` (the figures).\n"
    "- The final verifier and the fix: `verify_v6_final` (NO-GO: real wall lost on 8 right-route pieces, the records, one\n"
    "  fold-over), `s3_fix` (the must-fix list under D27).")
open(F, "w").write(s)
print("ok")
