# s2_build -- the full v6 build (compose 223, bake, checks_v6, anatomies_v6)

Agent s2_build, 2026-10-07. State: `state/s2_build.md`. Code: `scratchpad/v6/full/` (runner `dr_full.sh`, containers
`s2b_*`). Work outputs: `carotid_data/v6_work/full/` (/full). Mounts: /repo read-only, /bld (carotid_data/v6_work/build)
read-only (the DRY's plan), /full read-write. No git writes; no repo-tracked tree touched except as stated in section 7.

## Answer so far

**Complete (17:31 EDT).** All 223 anatomies composed (VA 1.4.2 / compose 3.2.0, toolchain a80644e0), baked (216 first-build bakes + 7 VA-1.4-round bakes, reused only where the bake inputs are identical) and checked: 223/223 pass every gate under checks_v6 1.4 (D26), with D27 applied to the one residual (k_011_left mr_012_L siphon: bake repair window 17.79 mm2 reported, not gated, flagged). Copied into carotid_data/anatomies_v6 (223 anatomies, 5136 files, sha256-verified, LICENSE per anatomy, provenance stamped with v6_checks); every anatomy loads (deterministic, 223/223). Flags: D26 16 anatomies, D27 1, D23 1, coverage < 0.99 reported 4. Records: results/v6_build.json; final manifest d65b7209.

## 1. Method and toolchain
- Compose: `build_v6.py <stage> --root R --plan /bld/dry223b/car_dry [--manifest M]` stage by stage (va1, carwrite,
  carverify, va2, recheck, fix, route13, final, finalcheck, split, summary) in 4 shard slots (`chain_compose.sh`,
  `chain_compose2.sh`); cardry is not re-run -- its decisions are the DRY's gated plan, replayed by `--plan` as the
  re-pilot did. Every re-compose covers ALL 223 so that every final record carries one toolchain.
- Bake + checks: `chain_bake*.sh <wid>` (neve-build-meshlab): checks the manifest with `md5sum -c` first, claims anatomies
  by atomic mkdir (claim dir + owner file), runs bake_v6 1.1 (marker bake*/_done/<n>.bake.json) and checks_v6 1.3 (all 7
  parts; marker checks*/_done/<n>.json only when every part is present without error). 2 workers per container (both
  are single-threaded, ~1-2 GB each); a relaunched worker drops its own stale claims.
- Reuse of bakes across re-composes (`merge_tree2.py`, `merge_tree3.py`): a baked anatomy is reused only when its bake
  inputs are identical -- every curve byte-identical and the provenance different only in version / date / timing /
  toolchain / record-only fields (`cmp_build2.py` + `cmp_build2_classify.py`); the baked provenance minus the bake's keys
  (bake_v6, c4_v6) must equal its own pre-bake provenance; the new provenance + those keys is written. The bake reads
  only the curves and provenance xform / cca / ica / host_cut, checks_v6 the curves, xform and v6_build.d23_accepted.
- Toolchains: first build `build/TOOLCHAIN_s2i2.md5` (bedf172b, unchanged); final `build/TOOLCHAIN_s2build.md5`
  (a80644e0, 59 files = the 46 + graft_v6_va 1.4.2 / compose_v6 3.2.0 + the s2_build runners and assembly scripts +
  build/eca_c3_feedback.json + carotid_tools/extend_ica.py + v5_va/sources.py, the two modules the gate verifier found
  imported but unlisted). Superseded intermediate manifests kept: TOOLCHAIN_s2build_80b29bc3.md5 (VA 1.4),
  TOOLCHAIN_s2build_11875b8d.md5 (VA 1.4.1). Frozen previous versions: va/graft_v6_va_1p3.py, _1p4.py, _1p4p1.py;
  carotid/compose_v6_v310.py.

## 2. The first full build (toolchain bedf172b)
- Compose (/full/build): 223/223, 0 hard failures, finalcheck 223/223, split exit 0 (none of 45 lowered), donors 1:1
  25/25. Identical to the DRY on 223/223: curves and licence files byte-identical, provenance different only in
  v6_va.date and v6_va.record.seconds (`/full/cmp_dry.json`).
- Bake + checks_v6 1.3 on all 223 (/full/tree, bake, checks; 06:30-11:35 EDT host clock): 219 baked, 4 bake failures;
  209 passed every gate, 10 failed one gate (D22 fork).

## 3. Failures, diagnosis and fixes
### 3.1 Bake: "even the tube-only mesh has a handle: a centerline-level fusion" (4)
- A pre-bake scan of the bake's own precondition on all 223 composed trees (`scan_tube_genus.py`: sdf_mesher.tube_field
  at 0.45 mm on the world lattice, keep_largest, genus) finds exactly these 4 (`/full/diag/tubegenus_final`).
- m_030_right mr_013_L: the RVA's new seam-1 stretch (radius 4.5-5 mm) turns back and its tube touches the RVA's own
  host prefix (RVA s 6 vs s 21, gap -0.067 mm). graft_v6_va 1.3 gated the stretch against its seam parents and every
  other branch, never against its own prefix (`/full/diag/tube_m030_mr013L.json`). A self-fold scan of all curves
  (`scan_self.py`) finds 2 below 0.6 mm (m_030 -0.067, m_024_left mr_013_L 0.596; same donor stretch), none in v5.
- w_014_right mr_010 / 011_L / 027 (genus 2 / 1 / 2): the fork's own ECA runs 0.37 mm (wall to wall) from its ICA at
  RECA s 16-19 / RCCA s 89 -- the same in v4 / v5, whose tube union stayed genus 0 at v4's pose -- and at the re-posed
  R2 pose the 0.45 mm lattice bridges it (`/full/diag/tube_w014R_mr010.json`).
### 3.2 checks_v6 D22 right fork (10)
- Diagnosis per anatomy (`d22_alt.py`: the same D22 cells, also counting cells within 0.3 mm of the piece's OWN lumen
  surface; `diag_d22dir.py`: inside / outside v4's union, protrusion off the RCCA tube).
- L-type (1), w_008_right mr_001 (0.964): the re-posed RECA TopBrain tip (wall gap 2.07 mm) leaves 22 mm2 of the ICA
  wall on neither v4's surface (D13 baseline v4_no_tip) nor the fork's own lumen. With the tip on its last rung (C3:
  v5 pose only, else skipped) D22 fork = 1.0, siphon 1.0 (`/full/diag/c3test`).
- R-type (9; all MIRRORED contralateral forks, D20, at R1 map): the failing cells lie OUTSIDE v4's union, ON the fork's
  own lumen (median 0.11 mm), 2.5-3.9 mm out of the RCCA tube: v6 shows real bulb wall that v4's union (the unmirrored
  fork in v4) never showed. value_own >= 0.996 on all 9. No ladder rung fixes it: R5 for a contralateral fork is still
  mirrored (frame match, D20) -- tested on w_033_left mr_007: 0.943 (`/full/diag/r5test`).
  w_012_left mr_020 0.9835 / mr_021_L 0.9743; w_033_left mr_007 0.9385, mr_010 0.9384, mr_011_L 0.9411, mr_016 0.9380,
  mr_024 0.9381; w_040_left mr_021 0.9721; w_048_left mr_008 0.9859.
- Tested and dropped: a pre-bake D22 estimate with the RECA tip as a competitor (`carotid/compose_v6_d22tip_UNUSED_dev.py`)
  -- it reads 0.998 on w_008 (the transition zone is not modelled), so it could not have caught the failure.
### 3.3 Fixes (tools versioned, previous versions frozen; gates added, none relaxed)
- compose_v6 3.2.0: U-TG -- the bake's tube-only union of the right fork group (RCCA + RECA) must be genus 0 in the
  final check; a failing variant gives way to the next (a variant refused ONLY by this gate does not spend
  RIGHT_LADDER_TRY). U-C3F -- post-bake D22 feedback (`build/eca_c3_feedback.json`, part of the manifest): listed
  anatomies take C3 for the right ECA tip (w_008_right mr_001). Result: w_014_right x3 -> R5 (v5 pose; all 8 R2 shifts
  5-12 mm fuse), w_008 -> tip skipped.
- graft_v6_va U3 (the fold: d - r_i - r_j >= 0.6 over pairs > r_i + r_j + 6 mm apart, at new points), in three steps:
  * 1.4: pose filter + final side gate. Re-compose /full/build2_va14 (manifest 80b29bc3), 9 anatomies re-baked
    (/full/tree2_va14, bake2, checks2): all 9 PASS every gate. BUT for ia172_069 L (m_006_left / w_051_right mr_025_L,
    the sides D25 names as direction-gate failures -> D18) the filter refused the pose the search had ranked first and
    the search landed on a pose whose D17 branch runs 91 mm ANTERIOR and 11 mm lateral (angle 71.9 deg: passes the
    letter of D25's gate) -- the contorted kind the integrator kept the direction gate a side gate to avoid, and
    m_006_left mr_025_L is one of the user's 8 named anatomies.
  * 1.4.1: final side gate only (pose search = 1.3's). Re-compose /full/build2_va141 (manifest 11875b8d): ia172_069 L
    back to rung 2, but m_030_right mr_013_L R went rung 1 (fold) -> rung 2 (D10 corridor -8.4 mm: fails) -> rung 3
    (v5's top as shipped) -> orientation blocking 'heading' on RVA_top: an invariant failure, while its rung-1 search
    holds a passing pose.
  * 1.4.2 (final): side gate + a side whose ONLY final-gate failure is the fold re-runs its rung-1 pose search once with
    folding poses refused; any other failure (D25 direction included) demotes as in 1.3. Test (`/full/diag/va142`):
    m_030 R rung 1 (tgt 28.8, = 1.4), m_024 R rung 2 (= 1.4), ia172_069 L rung 2 (= 1.3), w_002_right mr_013_L W1 = 1.4.

### 3.4 D26 (user decision "A", 2026-10-07 15:00): the piece's own scanned surface as the D22 reference where v4 hid it
- checks_v6 1.4 (`checks/checks_v6.py`; 1.3 frozen as `checks/checks_v6_v1_3.py`; dev copy `checks_v6_1p4_dev.py`). ONLY
  part route changed. The gated area is now EVERY RCCA-owned cell of the piece beyond its seam zone (2 mm end trim, minus
  the handle windows): 1.3's v4-occupied exclusion (the integrator's reference-validity rule, which the gate verifier said
  needed confirming) is replaced by D26 and kept as information (value_1p3). A cell is preserved when it lies within
  0.3 mm of v4's (baseline) union carried by the piece's own transform (D22) or, where v4 did not expose the piece's own
  scanned surface there, within 0.3 mm of that OWN scanned surface carried by the v6 transform (the bake's rigid
  placement: sdf_union.load_surface with mirror, R, origin, anchor). Per piece: value_v4 (all cells, v4 only), value_own
  (v4 or own surface); d22_reference = v4 when value_v4 >= 0.99, else own_surface (FLAGGED d26) when value_own >= 0.99
  AND no real wall is lost (coverage: v4's exposed RCCA-owned cells of the same stretch, carried, within 0.3 mm of the
  v6 union for >= 99 % of their area), else none (fails). Records carry part_versions.
- The compose toolchain is unchanged (a80644e0, carried by every composed record, kept as
  `build/TOOLCHAIN_s2build_a80644e0.md5`); the final manifest `build/TOOLCHAIN_s2build.md5` = the same files with only
  checks_v6.py changed + the final checks round (`full/chain_check4.sh`, `full/check4_helper.py`), `copy_v6.sh`,
  `verify_tree3.py`, `v6_build_collect.py` 1.2 and `cmp_build4_classify.py`. Installed by `full/install_s2build_tools4.sh`
  after the VA 1.4.2 compose exited (15:47 EDT): manifest **88ab481d**, 65 files; `diff` against a80644e0 shows only
  checks_v6.py (eb6921a9 -> 6ca3c595); the installed checks_v6.py is byte-identical to the tested dev copy.

### 3.5 The final re-compose (VA 1.4.2) and the bake reuse -- no re-bake needed
- `/full/build2` (manifest a80644e0; 19:54-21:46 container clock = 13:54-15:46 EDT): 223 composed, 0 hard failures,
  finalcheck 223/223 ok, split exit 0 (none of 45 lowered), summary written (`build2/logs/chain_A.out`).
- Compared with the first compose (`cmp_build2.py` -> `/full/cmp_build4.json`; `cmp_build2_classify.py` ->
  `cmp_build4_final.json`): 214 unchanged for the bake, 9 changed = exactly the VA-1.4 round's 9.
- Two of the 9 (m_006_left mr_025_L, w_051_right mr_025_L: donor ia172_069 L, D9 rung 2 in both) have NO file difference;
  the only provenance difference is the VA demotion-REASON text (1.4.2 also prints the self-fold reading of the rung-1
  pose, `diag`: `peek_vadiff.py`). New `cmp_build4_classify.py` (in the manifest) marks such an anatomy unchanged only when
  no file differs, every remaining difference is one of the three demotion-reason record paths, and the bake inputs
  (xform, cca, ica, host cut, left cca / ica, d23) and every side's VA rung are equal -> `cmp_build4_final2.json`: 216
  unchanged, 7 changed.
- `merge_tree3.py` -> `/full/tree3` + `/full/tree3_merge.json`: **bake 216** (first build's bake, provenance = the final
  compose's + the bake keys), **bake2 7** (the VA-1.4 round's bakes: curves byte-identical to `/full/build2_va14/final` and
  bake inputs equal), **bake3 0**. Unions: the first-build unions of the 2 record-only anatomies restored from
  `/full/superseded/iso` (the VA-1.4 re-bake's moved to `superseded/iso_va14`). `verify_tree3.py`: 223/223 consistent
  (passed bake record, OBJ + collision_full md5 = the bake record's, union written by that bake, bake keys present).
- Licences: `gen_license.py /full/tree3 /full/license` -> 223 files, 0 problems (donor set = v5's on every anatomy, body =
  v5's licence text), 216 non-commercial (Arterial-derived donor VA), 7 VMR_0248.

### 3.6 The final checks round: ONLY the changed route-check part (user, 15:15: "re-run ONLY the changed route-check part")
- checks_v6 1.4 has a part `route_d22` that re-runs ONLY `route.d22` (D22 / D26) with the same function and the same
  inputs part route gives it (exact ownership of the v6 union and of the D13 baseline union), carrying the rest of the
  route record (RCCA prefix, D13 baseline decision, relief information) from the record's earlier pass; it re-derives the
  provenance inputs of the carried D13 decision (RECA tip status, declared baseline) and errors if they differ, and the
  runner then runs the full part route. Records: `part_versions.route` = the carried pass's version (1.3),
  `part_versions.route_d22` = 1.4, `route.d22_only_rerun` says so; the checks4 done marker lists `carried_from_source`.
- Equivalence (test on the first tree, `/full/diag/d26eq_full` = full part route 1.4 vs `/full/diag/d26test` =
  route_d22): the d22 record (every key) and all gates identical on k_011_left mr_012_L (v4_no_tip baseline) and
  w_033_left mr_007 (v4 baseline) (`/full/diag/d26eq_cmp.txt`); 137-158 s against 182-248 s. Repeated on the FINAL
  round: check4 route_d22 on tree3 vs the full part route on the first tree for k_011_left mr_012_L, w_033_left mr_007 and
  w_040_left mr_021 (v4_no_tip): EQUIVALENT, d22 + gates identical (`/full/diag/d26eq_cmp_checks4.txt`).
- `chain_check4.sh` (2 workers x 4 containers, manifest-checked): every anatomy's record = the record of the bake that
  produced it (/full/checks or /full/checks2), copied to /full/checks4, then route_d22 with 1.4. The other six parts
  (strict audit, lumen, VA, orientation, loader, geom) and the rest of route are carried: geometry and bake are identical.

### 3.7 D27 (user rule 2026-10-07 16:25, "just do the recommended and then report/flag it"): the bake repair window
- Residual after the checks4 round (checks_v6 1.4): k_011_left mr_012_L SIPHON, d22_reference none (value_v4 0.9776,
  value_own 0.9898; no wall lost: coverage 0.9915). Its 244 cells off BOTH references (17.79 mm2, s 206.75-207.5) lie on
  the bake's section CAPSULE (sdf_union 1.8 r + 1 mm round the siphon's own centreline): the scanned lumen reaches
  0.3-2.3 mm beyond it and the bake's field min(f_real, capsule) shows the capsule there. v4's bake clips the same cells
  (carried into v4's world: 100 % capsule-clipped; v4's RVA top occupied them) -- the capsule is carried rigidly with the
  piece, so the clip does not come from v6's placement (`/full/diag/offboth_k011_mr012L_siphon.json`, `own2_*`).
- D27 as implemented, checks_v6 **1.5** (`checks/checks_v6.py`; 1.4 frozen as `checks_v6_v1_4.py`; dev copy
  `checks_v6_1p5_dev.py`): ONLY route_d22() changed and ONLY for a piece whose 1.4 reference is none. Its off-both cells
  where the bake's own section field is the capsule (capsule < f_real; bake_v6's sections as baked, sections_for_morph;
  capsule evaluated as sdf_mesher.tube_field does: 0.25 mm resampling, 8-NN) are the bake's repair window -- reported
  (`d22_repair_window`: cells, area, arclength range, lumen beyond the capsule, the v4-world test) and left out of the
  gated area, like D22's seam zone and handle windows; the reference is then re-decided exactly as 1.4 does on the
  remaining cells. value_before_d27 keeps 1.4's value; d27_flag marks the piece. Every piece that passes under 1.4 takes
  1.4's code path and values (only the key d27_flag = false is added).
- Tests: k_011_left mr_012_L (`/full/diag/d27test`): siphon -> own_surface + D27, repair window 244 cells, 17.79 mm2,
  s 206.75-207.5, scanned lumen 0.28-2.37 mm beyond the capsule, 100 % of the cells clipped by v4's own bake section too
  (clipped_in_v4_world_frac 1.0); value_own without the window 1.0; coverage 0.9915 (no wall lost). Equivalence on a
  passing anatomy (w_033_left mr_007, D26 fork): 1.5's d22 record == 1.4's except d27_flag = false, gates identical
  (`/full/diag/d27eq_cmp.txt`: EQUIVALENT).
- Installed 17:09 EDT after the checks4 round had finished (its workers re-import the tools per item) by
  `full/install_s2build_tools5.sh`: final manifest **`build/TOOLCHAIN_s2build.md5` = d65b7209** (72 files: 88ab481d's 65
  with checks_v6 1.5, check4_helper 1.1, copy_v6 1.1, v6_build_collect 1.3 + chain_check5.sh, stamp_prov.py, the
  installer and the 4 frozen previous versions); 88ab481d kept as `build/TOOLCHAIN_s2build_88ab481d.md5`.
- `full/chain_check5.sh` (route_d22 with 1.5 ONLY on the anatomies that failed the checks4 round = k_011_left mr_012_L;
  1.4 record kept as `/full/checks4/_v1p4/`): PASS, siphon own_surface + d27, d22 record identical to the dev test.

## 4. Final result (all 223)
- checks4 round (checks_v6 1.4 route_d22 on all 223, 15:48-17:09 EDT, 8 workers; `/full/check4_summary.txt`): 222 pass;
  the only failure k_011_left mr_012_L siphon (above) -> D27 -> pass. **Final: 223/223 pass every gate** (strict audit,
  D22 fork + siphon, RCCA prefix, VA lumen OBJ >= 0.65, orientation tier A 0 (D23 accepted), blocking 0, loader,
  geom clearance >= 0.6 and 1 mm turns <= 60) -- `results/v6_build.json` (v6_build_collect 1.3).
- D22 reference: fork v4 209 / own_surface 14; siphon v4 221 / own_surface 2 (16 anatomies flagged D26: the 9 R-type +
  w_046_left mr_012 / mr_020_L / mr_026, w_047_left mr_023, w_048_left mr_010_L, w_051_right mr_008_L (fork),
  k_011_left mr_012_L (siphon) -- w_046 x3 / w_047 / w_048 mr_010_L / w_051 / k_011 are the ones 1.3 passed only by its
  v4-occupied exclusion). D27: 1 (k_011_left mr_012_L siphon, 17.79 mm2).
- Values: D22 fork min 0.9915, siphon min 0.9923 (under the reference used); strict v4-only fork min 0.929, siphon
  0.9776. VA lumen OBJ min 0.664 mm; new clearance min 0.614 mm; bake 136-851 s, peak RSS max 2.11 GB.
- Reported, not gated (D22 coverage, v4's exposed RCCA wall carried onto v6): 4 pieces < 0.99, all on reference v4 (their
  v6 surface is 100 % within 0.3 mm of v4): w_027_left mr_022 fork 0.979 (26.7 mm2), w_027_left mr_026 fork 0.979
  (26.9 mm2), w_052_left mr_026 fork 0.950 (65.0 mm2), w_051_right mr_008_L siphon 0.983 (24.1 mm2).
- Compose flags (unchanged from the final compose): right fork R3 3 (w_002_right x3), R4 3 (w_018b_left mr_002_L / mr_007,
  w_027_right mr_020: fix loop VA contacts), R5 5 (w_003_right mr_001, w_036_right mr_018_L, w_014_right mr_010 / mr_011_L /
  mr_027: U-TG); seam-1 bend > 45: 5; ICA terminus > 40 mm posterior: 3 (w_003_right mr_001, w_018_right mr_007_L,
  w_027_right mr_016); D23 w_017_left mr_002_L RVA rung 3 (accepted); left forks substituted 22, mirrored 101, map
  bounded deviation 29 (5-43 deg), not on map 8; VA rungs right 198 / 12 / 1 + 12 host, left 215 / 3 + 5 host.

## 5. anatomies_v6 (the copy)
- `stamp_prov.py`: each anatomy's provenance = tree3's + ONE key `v6_checks` (checks verdict, part versions, gates, D22
  per piece, check-level flags: D26 / D27 / coverage / D23, manifest md5); asserted that nothing else changed
  (`/full/prov_final/stamp_prov.json`: 223 pass).
- `copy_v6.sh` 1.1 (tree3 + stamped provenance + LICENSE_v6_donors.txt; LICENSE_VMR_0248.txt kept on the 7 VMR anatomies;
  the inherited LICENSE_v5_donors.txt dropped): staged copy (`/full/anatomies_v6_staged`, `copy_verify_staged.json`) ok, then
  **carotid_data/anatomies_v6** 17:21-17:23 EDT: 223 anatomies, 5136 files, `sha256sum -c` rc 0, file sets equal, manifest
  `v6_work/full/anatomies_v6.sha256` (sha256 977388c1..., identical to the staged one) -> `/full/copy_verify.json`.
  Per-anatomy layout = anatomies_v5's (Centrelines_comb 18 curves, OBJ, collision_full.vtp, mesh_v6.json in place of
  mesh_v5.json, provenance.json, licence). Undecimated unions stay in `v6_work/full/iso/` (223, written by the bake that
  produced each tree3 anatomy: `verify_tree3.json` 223/223). The folder-level BUILD record / _records (as anatomies_v5
  has) are left to the records stage.
- Deterministic load of EVERY anatomy in the final folder (`full/load_v6_all.sh`, checks/loader_one.py 1.1, 2 containers,
  17:24-17:30 EDT): 223/223 loaded ok, 0 problems, identical to the checks' loader results on tree3
  (`/full/copy_verify_final.json`, carried into v6_build.json as copy_verify.load_v6).

## 6. Records
- `results/v6_build.json` (v6_build_collect 1.3, final run 17:30 EDT): per anatomy every compose / bake / check flag, gate
  value, D22 piece record (value, value_v4, value_own, value_1p3, value_before_d27, reference, d26 / d27, repair window,
  coverage), part versions, carried parts, toolchain md5s and timings; set counts, stats, lists (not_pass empty,
  d26_own_surface 16, d27_repair_window 1, coverage < 0.99 4, left_pose_map_bounded_dev 29), licence check, bake reuse,
  provenance stamp, copy verification + load.

## 7. Repo-tracked locations touched
- Created ONLY `carotid_data/anatomies_v6/` (this agent's). Read-only trees unchanged (git status clean on anatomies_v2..v5,
  topbrain_tools, carotid_tools, eve, eve_bench; nothing there newer than the build). No git writes. Note: anatomies_v6 falls
  under `.gitignore` `carotid_data/*` (anatomies_v5 was force-added), so tracking it needs `git add -f` at commit time.
- Container note: from 16:20 another agent's container (archcov `ac_*`) ran alongside; my 4 checks4 containers were already
  mid-round (started 15:48 when no other agent ran) and were left to finish (RAM 10.8 of 26 GB); every container started after
  that was at most 2 at a time.
