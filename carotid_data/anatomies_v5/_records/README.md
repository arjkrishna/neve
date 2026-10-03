# anatomies_v5/_records: the results and notes behind CAROTID_V5.md

Byte copies, made on 2026-10-03 by the `v5_records` agent, of the files `CAROTID_V5.md` cites. The originals were in
the build session's scratchpad (`...\scratchpad\v5_va\`), which is temporary; `carotid_tools/v5/MANIFEST.md` covers
the code. Nothing here was edited. `checks/va_entry_geom.txt`, `checks/route13_tool_v1_0/` and this README are the only
new files; `notes/verify_v5_final.md` and the refreshed `notes/V5_BUILD_LOG.md` were added after the final verifier.

## results/

| file | what |
|---|---|
| `v5_final_summary.json` | **the final record**: every number of CAROTID_V5.md, after the 13-anatomy move (swap13/final_collect.py, 2026-10-03 15:09) |
| `v5_bake_full_final.json` | the bake / measure collection over the final 223 (bake_full/collect.py) |
| `v5_bake_full.json` | the same collection before the 13-anatomy move |
| `v5_bake_probe.json` | the probe bakes of the high-genus donors that chose bake_v5b |
| `v5_checks.json` | the post-bake check stage (before the move) |
| `v5_route13.json` | the 13-anatomy move: diagnosis, variants, picks |
| `v5_assignment.json` | the assignment record: patient → donor table, totals, flags, dry-run summary, split |
| `graft_reg_assign_v5.json` | graft_v5_assign.py's own summary (score matrix, optima, verify log) |
| `plan_v5.json` | the per-anatomy plan written to anatomies_v5 (from `va_sources/v5_reg/`) |
| `v5_assign_records.json`, `v5_assign_dry_stats.json`, `v5_assign_written_check.json` | donor-record checks; dry-run statistics per unit, side and gate; written-anatomy check |
| `v5_assign_alt_orders.json`, `v5_assign_alt_all_served.json` | objective-order sensitivity; the reserve variant not taken |
| `v5_eligible_final.json` | the final eligible list: 25 primary + 2 reserve pairs, with every side's wall / QC / trace file |
| `scaleup_tally.json` | the final tally (2026-10-02 14:25): eligibility, coarseness, stop rule |
| `scaleup_tally_*_<hhmm>.json` | tally snapshots: pilot baseline 04:05, 172 lane batch 1 11:30, after 22 12:32, final 13:03, Song final 13:55 |
| `scaleup_tally_work/` | the grades the tally computed from the v2 records |
| `scaleup_ia172.json`, `scaleup_song.json` (+ `_grade`) | the lanes' per-case records and grades |
| `scaleup_fixes_cases_*.json`, `scaleup_fixes_before_after.json` | pilot cases re-run under pipeline v2; before / after the two fixes |
| `scaleup_eye_rejections.json` | every by-eye verdict (kept or rejected) |
| `pilot_decide.json`, `pilot_decide_tables.json`, `pilot_donor_ranking.json`, `pilot_next_cases.json` | the pilot decision, its tables, the ranking and the ordered next-case list |
| `pilot_proc_ia172_grade.json`, `pilot_proc_song_grade.json` | the pilot grades the tally reads |
| `vmr0248_metrics.json` | VMR 0248's landmarks and wall measures (its donor record reads them) |
| `standard.json` | the natural-anatomy standard (V, G, W rules); `va_standard.py` reads it |
| `split_v4.json`, `split_v5.json`, `split_v5_baked.json`, `split_v5_final.json` | v5_split.py: v4 baseline; after assignment; after the bake; final |

## checks/, swap13/, route13/

| file | what |
|---|---|
| `checks/loader_v5.json` | training loader over all 223 (check stage) |
| `checks/va_lumen.json` | VA lumen in the OBJ, v5 against v4 |
| `checks/entry_geom.json`, `checks/ostium_lumen.json` | the RVA entry geometry and the lumen at the ostium (SOFA set) |
| `checks/sofa_selection.json` | the 25-anatomy SOFA set and its rule |
| `checks/tb_geom.json` | the TopBrain piece unchanged (geometry) |
| `checks/item5_v4_untouched*.json` | anatomies_v4 against git HEAD, before and after the checks |
| `checks/va_entry_geom.txt` | the VA-entry comparison with the host (va_entry_geom.py re-run read-only on 2026-10-03) |
| `checks/route13_tool_v1_0/` | dry re-run of graft_v5_reg_sep 1.0 on the pre-move inputs: 13 of 13 decision records identical to the move's except run time (`cmp.txt`; added after the final verifier) |
| `swap13/loader_v5_after_swap.json` | the loader over all 223 after the move |
| `swap13/lumen13.json`, `swap13/va_lumen13.json` | route lumen and OBJ VA lumen of the 13 moved anatomies |
| `swap13/picks.tsv` | the 13 picks (variant per anatomy) |
| `swap13/diag_case_w_018b_left__topcow_mr_002.json` | why that anatomy reads 0.956 of v4 under exact ownership |
| `route13/lumen_iso.json` | undecimated route lumen, v5 against the chosen variant |

## per_anatomy/

| folder | what |
|---|---|
| `route_vs_v4_exact_final/` | right route against v4, exact ownership with k24 beside, all 223 after the move (swap13/rrdx.py) |
| `checks_out_preswap/`, `checks_out_swapped13/` | strict audit + route and VA detail (k24) per anatomy: the check stage's 223, and the 13 re-checked after the move |
| `va_wall_exact_preswap/`, `va_wall_exact_swapped13/` | VA wall detail per VA, exact ownership (vvw/measure.py): all 223 before the move, and the 13 after |

For the 13 moved anatomies the `*_swapped13` file supersedes the `*_preswap` one.

## sofa/

`check_stage_run4/{v5,v4}/` holds the 25-anatomy SOFA set (v5 and its v4 control); `swap13/{v5,v4}/` the 13 moved
anatomies after the move and the v4 controls not run before. For case_m_030_right__topcow_mr_013_L and
case_w_050_left__topcow_mr_004_L the `swap13/v5/` run supersedes the check-stage one.

## notes/

The build log (`V5_BUILD_LOG.md`) and plan, then every stage's notes and every verifier's notes, in build order:
arterial set-up (`arterial_infra`), VMR 0248 (`vmr`), the standard (`ref`), the QC calibration (`tag_calibration`,
`verify_tag_calibration`), the pilot (`pilot_fetch_*`, `pilot_proc_*`, `pilot_decide`, `verify_pilot_qc`,
`verify_pilot_wall`), the graft tool and split checker (`graft_reg`, `split`, `verify_graft_reg`), the scale-up
(`SCALEUP_PIPELINE`, `scaleup_fixes`, `scaleup_lane_ia172`, `scaleup_lane_song`, `verify_scaleup`, `v5_eye172`), the
assignment (`v5_assign`, `verify_v5_assign`), the bake (`v5_bake_probe`, `v5_bake_full`, `v5_checks`,
`verify_v5_mesh`, `verify_v5_wall`), the move (`v5_route13`, `verify_v5_route13`) and the swap (`v5_swap13`).
`v5_records.md` is the notes of the agent that wrote this folder and CAROTID_V5.md.
`verify_v5_final.md` is the final verifier's notes (all claims reproduced except the missing route13 tool 1.0, since
added to carotid_tools/v5).
