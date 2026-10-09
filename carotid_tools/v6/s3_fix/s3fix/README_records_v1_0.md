# anatomies_v6/_records: the results and notes behind CAROTID_V6.md

Byte copies, made on 2026-10-07 by the `s2_records` agent (`carotid_tools/v6/records/records/copy_records.sh`), of the
files `CAROTID_V6.md` cites. The originals are in the build session's scratchpad (`...\scratchpad\v6\`,
`...\scratchpad\v5_va\`), which is temporary, and in the git-ignored work folder `carotid_data/v6_work/`.
`carotid_tools/v6/MANIFEST.md` covers the code. Nothing here was edited. This README, `checks/build_record_check.json`
and `RECORDS.sha256` are the only files written here; `notes/s2_records.md` is the records agent's own notes.
`RECORDS.sha256` holds the sha256 of every file in this folder except itself and this README (1,156 files; check with
`sha256sum -c RECORDS.sha256` from inside `_records/`).

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

## checks/

| folder / file | what |
|---|---|
| `build_record_check.json` | `BUILD_v6.json` recounted from the final provenance with separate code: 98 checks, 0 differences; plus a comparison with the build collection (one boundary case explained) |
| `split/` | `v5_split.py anatomies_v6 --baseline split_v4.json`: exit 0, 45 compared, none lowered (`split_v4.json` is in `anatomies_v5/_records/results/`) |
| `loader/` | the training loader over the whole folder: 223 / 223 |
| `orient/` | the orientation gates in tree mode on anatomies_v6 and, with the same code, on anatomies_v5 |
| `lumen/` | the VA lumen on the 0.05 mm polyline, every OBJ and collision_full |
| `entry_geom/` | the geometric VA entry probe, v6 / v5 / v4, the 25-anatomy set and all 223 |
| `ident/` | anatomies_v6 against the checked tree, file for file |
| `route_spot/` | a fresh run of the route check straight from anatomies_v6 on 4 anatomies |
| `diag_cov12/` | the coverage diagnosis of the 4 pieces whose v4 wall does not reappear (diag_cov 1.2) |

## gates/

`selftest.txt` (orient_gates self-test against the audit: v5 141 tier A, v4 171, v3 11; every class count), `test_api.txt`
(the API against tree mode), `orient_v{3,4,5}.json` (the gates on the earlier trees), `heading_evidence.txt` / `.json` (the
per-patient Zenodo frames and the anatomical map, with the evidence), `zenodo_frames.json` (the frame table the tools read).

## per_anatomy/

| folder | what |
|---|---|
| `checks4/<name>.json` | the final checks record of every anatomy (checks_v6 1.3 parts carried, route_d22 1.4; 1.5 for case_k_011_left__topcow_mr_012_L), with `_done/` markers, `_v1p4/` (k_011's 1.4 record) and `_d27/` |
| `tube_genus_prebake/<name>.json` | the pre-bake scan of the bake's tube-only union (genus) on the final composed trees |

## sofa/

`rcca_v6/` (the RCCA training route, all 223), `rcca_v5/`, `rcca_v4/` (references: the 25-anatomy set and the w_042_left
cases), `d14_v6/`, `d14_v5/`, `d14_v4/` (VA entry on the 25-anatomy set, same tool and seeds), `set25.json`, `items.txt`,
`bif_angles_w042.json` (why the scripted wire takes w_042_left's ECA).

## diag/

The diagnoses CAROTID_V6.md cites: D26 (`d22alt_*`, `d22dir_*`, `d26test/`, `d26eq_cmp*.txt`: the R-type fork failures lie on
the fork's own scanned surface; the route_d22-only re-run equals the full part route), D27 (`d27test/`, `d27eq_cmp.txt`,
`offboth_k011_mr012L_siphon.json`, `own2_k011_mr012L_siphon.json`: k_011_left mr_012_L's capsule-clipped cells), the bake
fusions (`tube_m030_mr013L.json`, `tube_w014R_mr010.json`, `scan_self_final.json`), and w_008_right mr_001's RECA tip
(`d22_w008_mr001_fork.json`).

## notes/

- `V6_ITEMS.md` (the to-do list after the audit), `V6_BUILD_PLAN.md` (the plan, decisions D1–D7), `DECISIONS_ADDENDUM.md`
  (D1's amendments, D8–D27), `V5_BUILD_LOG.md` (the build log; the v6 entries start 2026-10-03 17:25).
- Stage 1: `v6_plan`, `v6_carotid`, `v6_va`, `v6_wall`, `v6_pilot`, `verify_v6_pilot`.
- Stage 2: `s2_harden_carotid`, `s2_harden_carotid2`, `s2_harden_va`, `s2_harden_wall_bake`, `s2_integrate`,
  `verify_s2_gate`, `s2_integrate2`, `verify_s2_gate2`, `s2_build`, `s2_checks`, `s2_records`.
- `audit/`: the orientation audit's agent notes (`tbo_history`, `tbo_geom`, `tbo_cause`, `va_narrow`, `verify_tbo`,
  `tbo_report`); the report itself is `reports/TopBrain piece orientation audit (v5).md`.
