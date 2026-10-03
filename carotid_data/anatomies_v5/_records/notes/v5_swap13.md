# v5_swap13: the 13 route-fixed anatomies into the release, and the final build record

Agent prefix `v5_swap13`. State `v5_va/state/v5_swap13.md`. Work folder `v5_va/swap13/`. No git writes; no tracked
repo file changed; `carotid_data/anatomies_v4` untouched (0 files newer than 2026-10-01, `git status` clean).

Status: complete (2026-10-03 15:15 EDT; resumed after the ~05:20 restart from the state file).

## Answer so far

- **Step 1 (done 02:12).** The 13 folders were backed up and swapped; every hash check passed.
  - Backup: `v5_va/backup_pre_route13/<name>/` (byte copies, `cp -a`), plus `_iso_v5/<name>.vtp`,
    `_bake_full_done/<name>.{bake,measure}.json` and `BUILD_v5.json`. sha256 of every file equals the original.
  - Swap: 12 `moved_ext` + 1 `moved_pin` (w_012_left mr_021_L), as results/v5_route13.json picks. Each pick folder has
    the same 23-file list as the v5 folder; every file was copied over with `cp -p`; sha256 of every file now equals
    the pick's, and the undecimated union (`iso_v5`) and the bake / measure markers equal the pick's too.
  - Only these 13 anatomy folders have files newer than the bake (2026-10-02 20:30).
  - The picks' `provenance.json` already records the move (`v5_va.route13`: tool, targets, separation objective,
    course shift); it was copied byte for byte, not edited.
- **Step 2: the 13 re-checked with the check stage's own tools** (`swap13/chk13.py` / `vvw13.py` / `lumen13.py` /
  `valumen13.py` are the check-stage scripts with only the output folder changed; outputs `swap13/{chk_out,vvw_out}`,
  `swap13/lumen13.json`, `swap13/va_lumen13.json`, `swap13/sofa/`).
  - Strict mesh audit (chk_one item 1, v4's topo_audit unchanged): 13/13 strict pass, 0 self-intersecting faces.
  - right_route_detail (k24, chk_one item 2): recomputed = the mesh_v5.json record on 13/13; v4 rebuild max diff 0.
    Exact ownership (swap13/rrdx.py): every fork and siphon ratio of the 13 is >= 0.9994 of v4
    (w_007 siphon 0.9994; w_012_left mr_001 fork 1.0024, w_035 fork 1.0068). Classes (tolerant cell match):
    identical everywhere 7, beyond 4 mm 4, beyond 6 mm 1 (w_035), not even beyond 8 mm 1 (w_012_left mr_001:
    238 + 52 cells differ, a net GAIN: fork 1.0024). k24 beside it: min 0.9925 (w_012_left mr_021_L fork; exact
    1.0000); classes 7 / 3 / 1 / 2.
  - VA wall detail of the moved VAs (vvw/measure.py, exact, seam = 0): donor stretch > 0.3 mm 0.067-0.278, within
    -0.015 / +0.006 of before the move (largest change w_050 RVA 0.293 -> 0.278); windows unchanged (3 VAs with 1).
  - Route lumen on the undecimated unions: now = before the swap = v4 on 13/13 (RCCA min 0.959-1.886 mm).
  - OBJ VA lumen: m_022_left mr_021_L's two narrow points fixed (0.551 / 0.589 -> 0.713 / 0.704 mm); all 26 VAs of the
    13 >= 0.65 mm. Over all 223, 3 VAs stay below 0.65 mm (not a v4 gate; known from the check stage):
    w_025_right mr_021_L LVA 0.578 / RVA 0.632 and w_027_right mr_022 RVA 0.646.
  - SOFA (checks/sofa_rva.py unchanged) on all 13: 13/13 PASSED (0 exceptions, 0 non-finite steps). The wire went
    >= 20 mm into the moved RVA on 4 of 12 (w_050 74.7 mm, w_007 144.3, k_008 136.5, w_035 107.5; m_022 moved only its
    LVA). **m_030_right mr_013_L lost its RVA entry with the move**: before the swap the wire went 55.7 mm into the
    donor RVA, now 0 mm in 3 episodes (still PASSED). This is the known VA-entry issue (donor ostium past the
    subclavian branch point), awaiting the user. v4 controls for the 11 not in the check stage: 11/11 PASSED, the
    wire entered the RVA in all 13 v4 controls.
  - Loader (checks/loader_v5.py) over all 223 after the swap: 223 load, 18 branches each, insertion inside,
    0 centreline points outside, no problem that v4's audit does not have.
- **Step 3: split.** `v5_split.py anatomies_v5 --baseline results/split_v4.json`: EXIT 0, none of the 45 quantities
  lowered; VA donors one-to-one and nested (results/split_v5_final.json, swap13/logs/split.log).
- **Step 4: the final record** (`swap13/final_collect.py`, run 2026-10-03 15:10 EDT, log `swap13/logs/final_collect.log`).
  - `carotid_data/anatomies_v5/BUILD_v5.json` (sha256 907b7717...; the pre-swap file is in backup_pre_route13/):
    bake_full/collect.py's `build_v5()` run UNCHANGED on the final bake/measure markers, plus `route13` (the 13
    moves: variant, targets, route before/after), `v4_va_kept_sides` (17 = 15 at assignment + 2 at bake:
    w_033_left mr_007 LVA, w_050_left mr_026_L RVA), `donor_flags`, `licences` extended (attributions with anatomy
    counts), `checks_final`, route13 timings and records. Collection over the final 223:
    `results/v5_bake_full_final.json` (the pre-swap one, v5_bake_full.json, is kept).
  - `results/v5_final_summary.json`: every number below, with the per-anatomy detail.
- **Final numbers (all 223 after the swap).**
  - Counts: 223 anatomies (none dropped), 223 baked and passing; 429 donor VAs (RVA 211, LVA 218; 206 anatomies
    both sides, 17 one side); 17 v4 VAs kept (15 assignment, 2 bake); 25 donor units (ia172 13, Song 11, VMR 1)
    one-to-one with the 25 TopBrain patients, no reserve used; donor VAs by source ia172 232, Song 184, VMR 13.
  - Rungs: 193 + 3 pinned VA whole (margin 0.5 / 1.0), 25 VA-local windows, 2 carotid-including windows (both
    flagged carotid_changed: w_016_left / w_016_right mr_011_L); m_030 left rung C with the move.
  - Strict mesh audit 223/223 (every sub-gate 223), 0 self-intersecting faces in OBJ / full / union, 0 centreline
    points outside.
  - **Right route vs v4, EXACT ownership** (the study's convention; tolerant cell match): identical everywhere 106,
    beyond 4 / 6 / 8 mm 80 / 23 / 8, not even beyond 8 mm 6 (w_012_left mr_001, mr_020; w_033_left mr_010, mr_011_L,
    mr_024; w_048_left mr_008). Fork ratio > 0.3 mm median 1.000 (min 0.956, p90 1.0066), siphon min 0.9979;
    222 / 223 >= 0.99 on both. RCCA medians v5 / v4: fork 0.1968 / 0.1961, siphon 0.3218 / 0.3218.
    **k24 beside it:** identical 104, 78 / 26 / 8, none 7; all 223 >= 0.99 (min 0.9909, w_003_left mr_024_L fork).
    Validation: v4 rebuild = mesh_v4.json record 223/223; v5 k24 = mesh_v5.json record 223/223. The bake record's
    class ("identical" = equal at 4 dp) agrees with the cell-level k24 class on 201; of the 22 others, 21 are
    4-dp-equal in the record but have cells differing within 4 mm (18) or 6 mm (3) of a VA, and one
    (w_017_left mr_002_L) is "beyond 4 mm" in the record and "beyond 6 mm" by cells.
  - **One anatomy below 0.99 under exact ownership: w_018b_left mr_002, fork 0.956** (not among the 13: its k24
    route equals v4 at 4 dp, so the k24-based selection did not catch it; verify_v5_wall had v4 exact numbers for
    27 anatomies only). `swap13/diag_lost.py` (measure.py / rrd_v4 unchanged; `swap13/diag/<name>.json`): the loss is
    121 v4 wall cells (7.85 mm2) near the bifurcation (RCCA s 92-103 mm) that k24 gives to the RECA and exact
    ownership to the RCCA, deeper than 0.3 mm (up to 3.9 mm) off its tube. None of them exists in v5 (nearest v5
    cell 0-1.6 mm away, median 0.48): the wall is reshaped where a donor VA (both sides are ia172_136; the side was
    not separated) passes 1.6-3.9 mm from it (v4's host VAs were 13.5-16 mm away). The RCCA's own (k24) route wall is identical to v4. Not moved here; it
    is a candidate for the route13 move if the exact-ownership rule is to hold for all 223.
  - VA wall detail (vvw/measure.py, exact, seam = 0; 429 donor VAs, 49 donor sides): donor stretch > 0.3 mm median
    0.134 (0.029-0.336); whole VA ostium -> end 0.128 (0.028-0.267) vs v4's whole VA 0.018 (k24); 429/429 above
    v4 on the same side; OBJ keeps 0.81 of the union's relief (p10 0.63); 25 VAs with one tube window (1.15-9.14 mm).
  - Lumen: route lumen of the 13 on the undecimated unions = before = v4; OBJ VA lumen < 0.65 mm on 3 VAs
    (w_025_right mr_021_L LVA 0.578 / RVA 0.632, w_027_right mr_022 RVA 0.646; not a v4 gate; 0 points outside).
  - Split: EXIT 0, none of the 45 quantities lowered; V donors 25 in 25 patients, one-to-one nested (all rosters).
  - Loader: 223 load, 18 branches, insertion inside on 223, RCCA byte-identical to v3 and non-VA files to v4 on 223,
    VA files replaced RVA 211 / LVA 218, no problem that v4's audit lacks.
  - SOFA (36 distinct anatomies: the check stage's 25, with m_030 and w_050 re-run after the swap, plus the other
    11 of the 13): 36/36 PASSED, 0 exceptions, 0 non-finite steps; v4 controls 36/36. Wire >= 20 mm into the RVA:
    v5 13/36 (11/34 donor RVAs) vs v4 35/36. Set of 25: 10/25 (8/23 donor) vs v4 24/25 (before the swap 11/25;
    m_030 lost its entry with the move).
  - Licences: 216 anatomies non-commercial (Arterial-derived donor VA), 7 VMR 0248 (research and development use;
    LICENSE_VMR file in all 7); LICENSE_v5_donors.txt in 223/223; attributions Head-CTA 172-series 124, Song CMHA
    92, VMR 7, TotalSegmentator 216.
- **Open for the user / the next stage.**
  1. w_018b_left mr_002 is below 0.99 of v4 under exact ownership (0.956, fork), above it under k24. Move its VA
     (route13 tool) or accept it as an ownership effect on ECA-side wall.
  2. VA entry in SOFA (pending question): 10/25 vs v4 24/25; the move cost m_030 its entry.
  3. Records / images / verifier stages: not this agent's task.
