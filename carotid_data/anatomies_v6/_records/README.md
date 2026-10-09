# anatomies_v6/_records: the results and notes behind CAROTID_V6.md

Byte copies, made on 2026-10-07 by the `s2_records` agent (`carotid_tools/v6/records/records/copy_records.sh`), of the
files `CAROTID_V6.md` cites. The originals are in the build session's scratchpad (`...\scratchpad\v6\`,
`...\scratchpad\v5_va\`), which is temporary, and in the git-ignored work folder `carotid_data/v6_work/`.
`carotid_tools/v6/MANIFEST.md` covers the code. Nothing here was edited. This README, `checks/build_record_check.json`
and `RECORDS.sha256` are the only files written here; `notes/s2_records.md` is the records agent's own notes.
`RECORDS.sha256` holds the sha256 of every file in this folder except itself and this README (check with
`sha256sum -c RECORDS.sha256` from inside `_records/`).

**2026-10-08, s3_fix** (the final verifier's must-fix list; `notes/s3_fix.md`): added `results/v6_coverage_s3.json`,
`results/s3_fix/`, `per_anatomy/checks_s3/`, `per_anatomy/bake_s3/`, `per_anatomy/fold_fix/`, `sofa/rcca_v6_s3/`,
`checks/coverage_v1p6_prefix/`, `diag/s3_fix/`, `notes/s3_fix.md`, `notes/verify_v6_final.md`, `notes/s2_images.md`;
replaced `checks/build_record_check.json` (check_build_record 1.1 on the regenerated BUILD_v6.json) and this README
(logged in `results/s3_fix/records_replaced.txt`), and rebuilt `RECORDS.sha256`. The files of the 17 anatomies s3_fix changed are
superseded, not removed: their s2 checks records stay in `per_anatomy/checks4/`, their s2 copy manifest in
`results/build/anatomies_v6.sha256`; the current ones are in `per_anatomy/checks_s3/` and `results/s3_fix/anatomies_v6.sha256`.

**2026-10-08, s4_records** (the second final verifier's record items, record only; `notes/s4_records.md`): 10 provenance
files re-stamped for a `d27_coverage_*` flag (the pieces with a real-wall loss > 0 and < 0.1 %), no other anatomy file
changed. Added `results/s4_records/`, `notes/s4_records.md`, `notes/verify_v6_final2.md`; replaced
`checks/build_record_check.json` (check_build_record 1.2 on the regenerated BUILD_v6.json) and this README (logged in
`results/s4_records/records_replaced.txt`), and rebuilt `RECORDS.sha256`. The current folder manifest is
`results/s4_records/anatomies_v6.sha256`; s3_fix's (`results/s3_fix/anatomies_v6.sha256`) stays as the record of the
folder after s3_fix.

## results/

| file | what |
|---|---|
| `v6_final_summary.json` | **the set-level summary**: every invariant with its verdict, SOFA, D14, right route, orientation, ICA AP, fork heading, wall detail, VA lumen, split, loader, flags (s2_checks, collect_s2c 1.0) |
| `v6_checks.json` | the set-level checks in full, per anatomy (s2_checks) |
| `v6_build.json` | the build's collection: per anatomy every compose / bake / check flag, gate value, D22 piece record, timing; set counts, the copy verification and the bake reuse (s2_build, v6_build_collect 1.3) |
| `stage2_dry2.json`, `stage2_dry2_auto.json` | the all-223 dry run under the frozen toolchain bedf172b, whose carotid plan the final compose replays (s2_integrate2) |
| `stage2_dry.json`, `stage2_dry_auto.json` | the first all-223 dry run (s2_integrate), before D22–D25 |
| `pilot_summary.json`, `repilot_summary.json`, `repilot2_summary.json` | the stage-1 pilot (14 anatomies) and the two stage-2 re-pilots |
| `s2_harden_wall_bake.json` | the wall and bake hardening: D15 local walls, D16 / D17 junction records, wall detail, the bake regression |
| `build/copy_verify_final.json` | the copy into anatomies_v6 (staged, then final; sha256sum -c, file sets) and the deterministic load of every anatomy |
| `build/copy_verify.json`, `build/copy_verify_staged.json` | the same before the load; the staged copy |
| `build/anatomies_v6.sha256`, `build/anatomies_v6_staged.sha256` | the sha256 manifest of the 5,136 copied files (and of the staged copy) |
| `build/check4_summary.txt` | the final checks round (checks_v6 1.4 route_d22 on all 223): 222 pass, the one D27 case |
| `build/stamp_prov.json` | the `v6_checks` block stamped into each provenance (md5 before / after, flags) |
| `build/tree3_merge.json`, `build/verify_tree3.json` | bake reuse: which bake produced each anatomy (216 first build, 7 VA-1.4 round), and the consistency check |
| `build/cmp_build4.json`, `build/cmp_build4_final2.json` | the final re-compose against the first: 216 unchanged for the bake, 7 changed |
| `build/cmp_dry.json` | the first full compose against the dry run |
| `build/license_check.json` | the generated LICENSE_v6_donors.txt files: 223, 0 problems |
| `v6_coverage_s3.json` | **s3_fix**: checks_v6 1.6 parts coverage (against the v4 / v5 SHIPPED wall) + folds on all 223 of the final anatomies_v6: per anatomy and piece the real-wall deficit, lost area and runs; the pieces >= 1 % before / after the fix; the 0.1-1 % pieces accepted with a flag |
| `s3_fix/anatomies_v6.sha256`, `s3_fix/copy_verify_s3.json` | the folder manifest after s3_fix (5,136 files; the 17 changed anatomies' lines replaced) and the verified copy |
| `s3_fix/stamp_prov_s3.json` | the `v6_checks` block re-stamped into the 17 changed provenance files (stamp_prov 1.1) |
| `s3_fix/sofa_rcca_summary.json` | SOFA on the RCCA of the 9 anatomies whose geometry changed, before / after; the all-223 and 25-anatomy totals after |
| `s3_fix/split_s3.json`, `s3_fix/loader_s3.json` | split and the training loader on the final folder |
| `s3_fix/final_rebake.txt`, `final_other.txt`, `final*.batch.log`, `coverage_before_fix_summary.txt`, `records_replaced.txt` | the final round's item lists and logs, the coverage on the pre-fix folder, the replaced notes |
| `s4_records/coverage_sub01_pieces.json` | **s4_records**: the 10 pieces with a real-wall loss > 0 and < 0.1 % (the second final verifier's own coverage, union and shipped collision_full, v4 and v5 references; checks_v6 1.6's value for the same piece), and the 2 pieces with a loss only on the shipped collision_full (not flagged) |
| `s4_records/verify_final2_coverage_all223.json` | the second final verifier's coverage summary on all 446 pieces of the final folder (vf2_cov.py) |
| `s4_records/stamp_flag_s4.json` | the re-stamp of the 10 provenance files (flag per piece, sha256 before / after; stamp_flag_s4 1.0) |
| `s4_records/anatomies_v6.sha256`, `s4_records/copy_verify_s4.json` | **the current folder manifest** (5,136 files; exactly the 10 provenance lines differ from s3_fix's) and its check (`sha256sum -c` exit 0, file sets equal) |
| `s4_records/build_v6_diff_vs_s3.txt` | every value of BUILD_v6.json that changed against s3_fix's (38 leaves: the 10 flags, their provenance hashes and digest, the text entries) |
| `s4_records/records_replaced.txt` | the files this step replaced in `_records/` |
| `s4_records/scripts/` | this step's scripts: `cov10.py`, `stamp_flag_s4.py`, `copy_prov_s4.sh`, `copy_records_s4.sh`, `diff_build.py`, `patch_bvr_s4.py`, `patch_readme_s4.py`, `dr_s4.sh`, and `build_v6_record.py` / `check_build_record.py` 1.2 (text-only changes from the 1.1 in `carotid_tools/v6/records/records/`) |

## checks/

| folder / file | what |
|---|---|
| `build_record_check.json` | `BUILD_v6.json` recounted from the final provenance with separate code: 123 checks, 0 differences (check_build_record 1.2, after s4_records; 1.1 after s3_fix, also 123; the s2 version had 98); plus a comparison with the s2 build collection (the rows that differ are s3_fix's and s4_records' changes, and one boundary case, explained) |
| `split/` | `v5_split.py anatomies_v6 --baseline split_v4.json`: exit 0, 45 compared, none lowered (`split_v4.json` is in `anatomies_v5/_records/results/`) |
| `loader/` | the training loader over the whole folder: 223 / 223 |
| `orient/` | the orientation gates in tree mode on anatomies_v6 and, with the same code, on anatomies_v5 |
| `lumen/` | the VA lumen on the 0.05 mm polyline, every OBJ and collision_full |
| `entry_geom/` | the geometric VA entry probe, v6 / v5 / v4, the 25-anatomy set and all 223 |
| `ident/` | anatomies_v6 against the checked tree, file for file |
| `route_spot/` | a fresh run of the route check straight from anatomies_v6 on 4 anatomies |
| `diag_cov12/` | the coverage diagnosis of the 4 pieces whose v4 wall does not reappear (diag_cov 1.2) |
| `coverage_v1p6_prefix/<name>.json` | **s3_fix**: checks_v6 1.6 parts coverage + folds on the PRE-fix anatomies_v6 (all 223): exactly the final verifier's 8 pieces >= 1 % |

## gates/

`selftest.txt` (orient_gates self-test against the audit: v5 141 tier A, v4 171, v3 11; every class count), `test_api.txt`
(the API against tree mode), `orient_v{3,4,5}.json` (the gates on the earlier trees), `heading_evidence.txt` / `.json` (the
per-patient Zenodo frames and the anatomical map, with the evidence), `zenodo_frames.json` (the frame table the tools read).

## per_anatomy/

| folder | what |
|---|---|
| `checks4/<name>.json` | the final checks record of every anatomy (checks_v6 1.3 parts carried, route_d22 1.4; 1.5 for case_k_011_left__topcow_mr_012_L), with `_done/` markers, `_v1p4/` (k_011's 1.4 record) and `_d27/` |
| `tube_genus_prebake/<name>.json` | the pre-bake scan of the bake's tube-only union (genus) on the final composed trees |
| `checks_s3/<name>.json` | **s3_fix**: the final checks record (checks_v6 1.6) of the 17 changed anatomies: every part on the 9 whose geometry changed, coverage + folds on the 8 re-stamped for a flag (the other parts carried from `checks4/`) |
| `bake_s3/<name>.bake.json` | the done markers of the 8 re-bakes (bake_v6 1.2: lattice shift / route-island bridge, timings, md5) |
| `fold_fix/<name>.json` | the OBJ fold-over fix of case_m_022_left__topcow_mr_021_L (fold_fix 1.0) |

## sofa/

`rcca_v6/` (the RCCA training route, all 223), `rcca_v5/`, `rcca_v4/` (references: the 25-anatomy set and the w_042_left
cases), `d14_v6/`, `d14_v5/`, `d14_v4/` (VA entry on the 25-anatomy set, same tool and seeds), `set25.json`, `items.txt`,
`bif_angles_w042.json` (why the scripted wire takes w_042_left's ECA). `rcca_v6_s3/` (s3_fix): the RCCA runs of the 9
anatomies whose geometry s3_fix changed (they replace those anatomies' `rcca_v6/` records).

## diag/

The diagnoses CAROTID_V6.md cites: D26 (`d22alt_*`, `d22dir_*`, `d26test/`, `d26eq_cmp*.txt`: the R-type fork failures lie on
the fork's own scanned surface; the route_d22-only re-run equals the full part route), D27 (`d27test/`, `d27eq_cmp.txt`,
`offboth_k011_mr012L_siphon.json`, `own2_k011_mr012L_siphon.json`: k_011_left mr_012_L's capsule-clipped cells), the bake
fusions (`tube_m030_mr013L.json`, `tube_w014R_mr010.json`, `scan_self_final.json`), and w_008_right mr_001's RECA tip
(`d22_w008_mr001_fork.json`). `s3_fix/` (s3_fix): the lost-wall diagnosis (`frag_*.json`: the dropped islands;
`sign_w052.json`), the lattice-shift screen (`shift/`), and the dev re-bake variants (`dev_rebake/<tag>/`: A, B, C0-C3,
D1-D3 as `notes/s3_fix.md` section 3.1 lists them).

## notes/

- `V6_ITEMS.md` (the to-do list after the audit), `V6_BUILD_PLAN.md` (the plan, decisions D1–D7), `DECISIONS_ADDENDUM.md`
  (D1's amendments, D8–D27), `V5_BUILD_LOG.md` (the build log; the v6 entries start 2026-10-03 17:25).
- Stage 1: `v6_plan`, `v6_carotid`, `v6_va`, `v6_wall`, `v6_pilot`, `verify_v6_pilot`.
- Stage 2: `s2_harden_carotid`, `s2_harden_carotid2`, `s2_harden_va`, `s2_harden_wall_bake`, `s2_integrate`,
  `verify_s2_gate`, `s2_integrate2`, `verify_s2_gate2`, `s2_build`, `s2_checks`, `s2_records`, `s2_images` (the figures).
- The final verifier and the fix: `verify_v6_final` (NO-GO: real wall lost on 8 right-route pieces, the records, one
  fold-over), `s3_fix` (the must-fix list under D27).
- The second final verifier and the record fix: `verify_v6_final2` (s3_fix holds; record items: flags on the 10 pieces
  with a loss below 0.1 %, stale numbers in CAROTID_V6.md), `s4_records` (those items, record only).
- `audit/`: the orientation audit's agent notes (`tbo_history`, `tbo_geom`, `tbo_cause`, `va_narrow`, `verify_tbo`,
  `tbo_report`); the report itself is `reports/TopBrain piece orientation audit (v5).md`.
