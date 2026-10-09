# s2_integrate -- the v6 joint-pass driver, the all-223 DRY, the split check and the re-pilot

Agent s2_integrate, 2026-10-06 (after the weekly-limit reset). State: `state/s2_integrate.md`. Code: `scratchpad/v6/build/`
(driver, runner `dr_int.sh`, chains), `sofa/sofa_rcca_v6.py`, `figs/fig_v6_repilot.py`; tool fixes in `carotid/compose_v6.py`
(3.0.3), `va/graft_v6_va.py` (1.2.3) and `carotid/verify_s3.py` (1.2); diagnostics `build/diag/`. Outputs: `carotid_data/v6_work/build/` (/bld: `dry223/` the DRY, `smoke/`,
`test_*`) and `carotid_data/v6_work/repilot/` (/rp). Containers `s2int_*`, /repo read-only. No git writes; no repo-tracked
tree touched.

## Answer so far

- Driver `build/build_v6.py` 1.0 written and smoke-tested end to end on 3 anatomies (every stage incl. the fix loop and
  route13). Integration bugs found and fixed so far (versioned, stated diffs, no gate relaxed):
  - compose_v6 3.0.3 `--va-hard-root`: the carotid re-check with the VA tool's final VAs as HARD neighbours (fix loop).
  - graft_v6_va 1.2: 1 mm turns gated on the curves AS WRITTEN (T1 fillet, T2 D17 branch from F with corner spreading,
    T3 final gates with geom_v6's rule).
  - route13_table_v6 1.0: diag13's v4 fork table carried onto the D19/D20-moved fork (self-check exact).
- **DRY (all 223) DONE**: 223 / 223 composed, 0 hard failures, final re-check 223 / 223 (VA recheck, geom_v6, verify_s3);
  ICA AP > 40 mm: 3 (v5: 108); donor's own junction 416 / 429 (stage 1 dry10: 359 / 429); D10 corridor 421 / 429;
  donors one to one 25 / 25 -> `results/stage2_dry.json`.
- **Split check DONE**: exit 0, none of 45 quantities lowered (`/bld/dry223/split/`).
- **Re-pilot (14)**: build 14 / 14, bake 14 / 14, load test passed, SOFA D14 14 / 14 passed, SOFA RCCA 14 / 14 passed
  (v5 14 / 14), figures 14. **checks_v6: 7 / 14 pass; the other 7 fail ONLY route_fork_ge_0.99** (0.944-0.9898).
  Diagnosis (section 5.3): sampling noise alone is small (v4's whole anatomy moved rigidly: 0.990-1.007 of itself over
  14 poses); the v6 fork centreline is v4's rigid image and the baked surface is v4's carried surface to ~0.02 mm, yet
  the relief share (cells > 0.3 mm off the tube) moves by a few % with the pose: threshold flips on sub-0.05 mm
  differences and on how the union is triangulated around the moved fork, plus the D19 seam-1 blend removing v4's
  seam ledge (material on w_034 only), the R2 host-cut shift and bake handle windows on right-fork pieces (w_035,
  w_025). The composer's C0 (0.997-1.014 on all 14) does not track the baked value (0.944-1.051). No tool bug found;
  the gate was NOT touched -> decision for the orchestrator (section 7, issue 0).
- Tool versions: compose_v6 3.0.3, graft_v6_va 1.2.3 (T4-T6 added after the DRY's orientation pre-check; 2 anatomies
  redone), verify_s3 1.2, build_v6 1.1, repilot_summary 1.1; manifest `build/TOOLS_md5_final.txt`.

## 1. Driver (`build/build_v6.py` 1.1; 1.0 frozen)

`python3 build_v6.py <stage> --root R [--names-file N] [--shard i/n] [--workers k] [--plan DIR]` (eve-training-fixed), or
`all` (every stage in order, one process; with `--plan` the sequential carotid dry is replaced by the given plan -- used
for subset runs such as the re-pilot). Stages and what they run:

| stage | tool call | per-anatomy output (= done marker) |
|---|---|---|
| va1 | `graft_v6_va --write R/w1` (base anatomies_v5) | `w1/<n>`, `w1_rec/<n>.json` |
| cardry | `compose_v6 dry --base R/w1 --out R/car_dry --c0-cache` (sequential: the split guard) | `car_dry/<n>.json` |
| carwrite | `compose_v6 write --base R/w1 --plan <car_dry or --plan> --out R/car` (only unwritten names) | `car/<n>`, `car/<n>.json` |
| carverify | `verify_s3 R/car R/w1` | `checks/car_verify*.json` |
| va2 | `graft_v6_va --write R/w2 --carotid-root R/car` | `w2/<n>`, `w2_rec/<n>.json` |
| recheck | `graft_v6_va --recheck R/w2` + `geom_v6 R/w2` + `verify_s3 R/w2 R/w1` | `recheck/va/<n>.json`, `recheck/verdicts*.json` |
| fix | for VA-vs-carotid contacts: `compose_v6 write --va-hard-root <VA tree>` -> `graft_v6_va --write --carotid-root` -> recheck; <= 2 iterations | `fix/fix.json`, `fix/<k>/...` |
| route13 | `route13_table_v6` (diag13 v4 table onto the moved fork) -> `graft_v6_va --route13-diag ... --route13-side right --route13-other-tree` -> recheck; adopted if both sides pass, rechecks pass, LOST does not rise | `route13/route13.json` |
| final | copy of the adopted source (route13 > fix > W2) + provenance `v6_build` + c4 planned (bake_v6.apply_c4 write=False) | `final/<n>`, `final/_done/<n>.json` |
| finalcheck | recheck on `final` + `orient_gates tree --fork-gates-placed` + `va_corridor tree` | `finalcheck/...` |
| split | `v5_split.py R/final --baseline split_v4.json` | `split/split_check.json`, `split/exit.json` |
| summary | counts per rung / flag | `summary.json` (or `--out`) |

Provenance `v6_build` per anatomy: source, tool versions; carotid: right fork (pose source, rung, flag, placement,
mirrored, contralateral, D20 applied, seam-1 bend, blend, host-cut shift, host cut, deviation, yaw, C0 fork / siphon,
siphon flag, VA gap), left fork (rung, decision, lower, pose, pose flag, mirrored, D20, seam-1 bend, tubed ICA mm), ICA
AP offsets + flags, ECA right decision, skipped ECA tips, route baseline (D13); VA per side (rung, junction type, donor,
pass_all, flags, va_target + band, donor-stretch course RMS, D10 corridor, take-off error, D16 tube mm, D17 cut end /
past ostium / label / tube / trimmed / host distal dropped / turns, route excess, pose); fix loop; route13 (LOST before
/ after); c4 plan; the pre-final recheck verdict; a flat flag list.

Runner `build/dr_int.sh` (containers `s2int_*`, mounts /repo ro, /sp, /bld, /rp). All-223 chain `build/chain_dry.sh A|B|C`
(three containers, marker waits inside the containers). Re-pilot chain `build/chain_repilot.sh build|bake|sofa|sofa5|post`.

## 2. Integration bugs found and fixed (versioned; stated diffs in each file; no gate relaxed)

1. **compose_v6 3.0.2 -> 3.0.3** (`carotid/compose_v6.py`; 3.0.2 frozen `carotid/compose_v6_v302.py`, md5 36016616):
   `--va-hard-root TREE`. The VA tool cannot clear a fork-VA contact when the contact is on v5's donor stretch (its
   rung 3) -- J223's w_018b_left mr_007 R (-0.71 mm vs RECA) and mr_002_L R (-2.30 mm vs RCCA); the carotid tool treats
   VAs as soft. With the option, RVA / LVA of TREE (the VA tool's final placement) replace the base's as neighbours and
   every VA contact < 0.6 mm is HARD (right variants touching a VA refused in emit() / at R5; final check fails on a soft
   failure). Without the option 3.0.3 == 3.0.2. Used only by the driver's fix loop. Smoke: w_018b_left mr_007 -> right
   fork R4 map_dev_bend60 (45.4 deg, flagged), VA gap 3.8 mm; VA R rung 2 / L rung 1; every recheck passes.
2. **graft_v6_va 1.1 -> 1.2** (`va/graft_v6_va.py`; 1.1 frozen `va/graft_v6_va_1p1.py`, md5 2801d1d5): checks_v6 1.2's
   geom part (1 mm turns <= 60 at new points of the WRITTEN curves) failed 1.1's joint trees on 2 of 65 checked
   (FYI wall-geom-1, unanswered): the LVA of the ia172_136 L tube-bridged junction turns 61.0 deg at its fillet (s 15;
   every mr_002 anatomy), and song_AHMU1218095's (19) / (17) reverses 152 deg at the ostium foot (its Arterial path
   climbs to the VA node and back -- the 1.1 flag d17_branch_kink, 8 sides). Both passed 1.1's own gates, which
   measured on a different sampling (fillet) or skipped F .. foot + 20 (D17). Fixes:
   T1 the fillet turn measured on any sampling the written curve can have (offsets 0.05 mm, steps 0.98-1.02 -- the VA
      part is resampled over its whole length, so the phase at the fillet depends on the top) and the junction corners
      spread with the same Gaussian at a wider sigma until <= 58 (construction target); gate = max(1.1 measure, written).
   T2 the D17 branch's 1 mm turn gated from F (was foot + 19); a corner in F .. foot + 20 is spread by the same Gaussian
      first, accepted only if no point moves >= 0.75 x its radius (song_095: 152 -> 58.7 deg, max 1.59 mm in a 2.83 mm
      lumen); recorded `d17.gate.corner_spread`.
   T3 final gates: geom_v6's turn rule on each side's own curves as written (demotes like any final-gate failure).
   Tests (`build/test_va12`, carotid root = the carotid tool's w223 as in J223): geom 5 / 5 pass; vs 1.1's J223 records
   9 / 10 sides identical (rung, junction, pose, target, pass); w_018b mr_002 L same pose, va_target 25.451 -> 25.461;
   VA-side curve files byte-identical except the three fixed curves (m_006 mr_002 LVA, w_018b mr_002 LVA, k_004 mr_014_L
   (19)); all three keep rung 1 (1.1 first-draft of 1.2 had demoted them to rung 2 -- refined); `--recheck` 5 / 5 ok.
   Scope on 1.1's whole J223 joint tree (geom_v6 on all 223, `build/geom_j223/geom_j223_va11.log`): 1 mm turn failures on
   11 anatomies -- the 8 song_095 mr_014_L (19) reversals (T2), the 2 ia172_136 L fillets (T1), w_018_right mr_022 LVA
   70.8 deg at s 147 (near its top; T3 now demotes it inside the tool) -- and the 2 known fork-VA clearance contacts
   (w_018b mr_002_L, mr_007: the fix loop). Nothing else fails geom on J223.
3. **route13_table_v6 1.0** (new, `build/route13_table_v6.py`): item (b)'s diag13 table is v4's RCCA real-wall voxels in
   v4's WORLD pose; under D19 / D20 w_018b_left mr_002's right fork is map-posed and x-mirrored, so the table is carried
   with the fork's own xforms (+ the seam-1 blend displacement); self-check: distance of each carried voxel to the v6
   RCCA polyline == to the v4 one (median 4e-15 mm, max 0.019 mm in the blend). Smoke: the plain joint placement's RVA
   takes 3574 v4 fork voxels (RCCA s 58-124) -> 0 after the route13 move.
4. **Driver bugs** (build_v6 during the smoke): verify_s3 output dir not created (stage marked done) -> fixed + the
   stage now raises when verify writes nothing; summary flag counter / ECA flag text.
5. **verify_s3 1.1 -> 1.2** (`carotid/verify_s3.py`; 1.1 frozen `carotid/verify_s3_v1_1.py`): on the DRY's carotid
   write, 222 / 223 verified; w_042_left mr_023 failed only the record match of the INFORMATIONAL full-segment siphon C0
   (seam-2 zone included): record 1.15432 (the same on base v5 and W1) vs 1.15393 from the files (4e-4 > 1e-4), while
   its gated value (seam-2 zone skipped) matched exactly (1.09823) and every gate passed. 1.1 had exercised that check on
   17 anatomies only. 1.2 reports the full-segment match (`c0_siphon_full_match`, `c0_siphon_full_diff`) without letting
   it decide ok; the gated-value match (<= 1e-4) and the C0 >= 0.99 gate are unchanged.
6. **New check tool sofa/sofa_rcca_v6.py 1.0**: SOFA on the RCCA TRAINING route (= sofa_va_entry 1.1 with the path (11)
   -> RCCA -> ICA, fork 2 = the RECA take-off with the same hook-back), soundness verdict as D14 (0 exceptions,
   0 non-finite, tip never > 0.5 mm outside) + route reach per seed (past seam 1 + 20, past the bifurcation, siphon,
   target). Test on v5 w_034 (1 seed, 150 steps): PASSED, max RCCA s 98.9 (bifurcation 71.9), 0.22 s / step.
7. **graft_v6_va 1.2 -> 1.2.1 / 1.2.2 / 1.2.3** (`va/graft_v6_va.py` md5 18bdc01d; frozen `_1p2.py` 5366fcd3,
   `_1p2p1.py`, `_1p2p2.py`), found by the DRY's final orientation pre-check (`build/precheck_orient.py` = checks_v6's
   part orient on the composed tree):
   T4 (1.2.1) splice join of top-only placements gated AS WRITTEN (1 mm resampled from the splice) as well as natively;
   T5 (1.2.2) host-VA extension join: the eve_bench host VA joined past v4's cut only where both 1 mm turns are <= 60
      (else the first later host point within 6 mm that joins within 60, else no extension) -- the real cause of
      w_018_right mr_022 L's 70.8 deg turn (a 0.68 mm zig-zag between two near-coincident polylines), which T3 had
      demoted to host_v5 (v5's tier-A top);
   T6 (1.2.3) D8 on max(own top-to-target distance, orient_gates' tip_move_anat = checks_v6's post-bake measure), pose
      level at 30 - 0.5 mm (construction margin; the gate stays 30): w_013_right mr_013_L R read 29.93 here vs 30.013
      in checks_v6.
   Redone with build_v6 1.1 `redo` (VA second pass and everything after it): w_018_right mr_022 (L now host rung with
   an anatomical top, target 9.4 mm) and w_013_right mr_013_L (R rung 1, new pose, target 29.25 / 29.33). Scope:
   1.2.3 reproduces 1.2 on the other 221 (every other non-terminal va_target <= 28.33 mm < 29.5; geom_v6 finds no join
   > 60 in the final tree). Split re-run after the redo: exit 0.
8. **build_v6 1.0 -> 1.1** (1.0 frozen `build/build_v6_v1_0.py`): stage `redo` (re-run va2 -> final -> finalcheck for
   named anatomies after a tool fix, old outputs kept under `redo/<time>/`).
9. **repilot_summary 1.0 -> 1.1** (frozen `build/repilot_summary_v1_0.py`): + the route-gate diagnosis from /rp/diag.
   `stage2_dry_compose.py`: the tools block now states the versions that actually ran (manifest
   `build/TOOLS_md5_final.txt`); the rest of stage2_dry.json byte-identical.
No tool's GATE was changed by this agent: bake_v6 (b3731df4) and checks_v6 (6eb7fe1f) are the hardening agent's files,
untouched.

## 3. The all-223 DRY (`carotid_data/v6_work/build/dry223/`; `results/stage2_dry.json`)
Run 2026-10-06 ~11:30-14:07 EDT, build_v6 chain in 3 containers (compose_v6 3.0.3, graft_v6_va 1.2 / 1.2.3 for the 2
redone, verify_s3 1.1 for carverify and 1.2 for the rechecks).
- **Composed 223 / 223, 0 hard failures** (not composed 0, carotid not ok 0, final check not ok 0). Final re-check
  223 / 223: VA recheck, geom_v6 (min new clearance 0.615 mm), verify_s3.
- **Right fork** (D19 / D20): R1 map 118, R2 host-cut shift 83 (87 shifted, median 2.0, max 11.96 mm), R3 bend 60 13,
  R4 deviation 4, R5 v5 5; 22 flagged; 113 contralateral, all mirrored. Seam-1 bend median 39.1, max 59.8 deg (17 > 45).
  vs the carotid tool's base-v5 run: R1 121 / R4 1 -- the 3 fix-loop anatomies moved R1 -> R4.
- **Left fork** (D1 / D11 / D20): own fork 201, substitute 22, no ICA extension; map-posed 215, 8 pose flags; 101
  mirrored.
- **ICA AP > 40 mm: 3** (w_003_right mr_001 R 46.0, w_018_right mr_007_L R 53.3, w_027_right mr_016 R 45.3); v5: 108.
  RICA median 6.0, LICA median 4.5 mm.
- **ECA tips**: right carried 6 / re-posed 28 / v5 1 / skipped 17 / none 171; left skipped 36; 52 anatomies skip >= 1.
- **D13 baseline**: v4 190, v4_no_tip 33.
- **VA**: donor's own junction **416 / 429** donor sides (stage 1 dry10 359 / 429; VA agent J223 414 / 429): plain 396,
  + D16 tube 18, + tube bridge 2; v5 junction 12 (D9 rung 2); v5 top as shipped 1 (rung 3); host (D21) 17.
  pass_all 444 / 446: the 2 fails are w_017_left mr_002_L (R rung 3 va_target 55.3, L rung 2 pair flag).
  va_target <= 20: 436, 20-30: 9, > 30: 1. D16 18 sides (ia172_164 L 16 x 9.71 mm, ia172_071 L 2 x 10.01 mm). D17: branch
  29.2-119.7 mm past the ostium (median 70.1), 55 sides tubed (median 0.66, max 12.1 mm), host distal dropped 102-115 mm.
  **D10 corridor 421 / 429** (8 fails: 6 native-offset excursions incl. w_017 x2, w_018b mr_002_L; 13 host-VA fallbacks
  flagged); take-off error median 0.09 deg; donor-stretch course RMS median 12.8, max 21.3 mm (2 > 20 inherited v5).
- **Fix loop** (VA-vs-carotid contacts the VA tool could not dodge): 3 anatomies (w_018b_left mr_002_L, mr_007,
  w_027_right mr_020), all adopted at iteration 1 (right fork R4 map_dev_bend60, VA gap 7.5 / 3.8 / 19.8 mm).
- **route13** (item b, w_018b_left mr_002): adopted; RVA takes 3574 v4 fork voxels -> 0; both VA sides rung 1.
- **Orientation (orient_gates tree on the final)**: 1 anatomy tier A = w_017_left mr_002_L RVA_top (D9 rung 3 "v5's
  top as shipped") -- a DECISION CONFLICT between D9's terminal rung and the "0 tier-A" invariant (open issue 1).
- **Donors one to one**: 25 donor units onto 25 TopBrain patients (no donor to many patients, no patient to many).

## 4. Split check on the planned set
`v5_split.py /bld/dry223/final --baseline split_v4.json`: **exit 0, 45 quantities compared, none lowered** (re-run
after the redo: exit 0). Extra donors span one TopBrain patient each under v3 / v4side / strict (25 / 25 / 25).

## 5. Re-pilot (`carotid_data/v6_work/repilot/`)

**Build** (`build_v6.py all --plan /bld/dry223/car_dry`, 14 pilot names, 14 min; `/rp/build/summary.json`): 14 / 14
composed, 0 hard failures; final rechecks 14 / 14 (VA recheck, geom_v6, verify_s3); right fork R1 map 10, R2 host-cut
shift 4, 10 mirrored (D20); left rung 1: 12, rung 2 (substitute): 2, 6 mirrored; ICA AP > 40 mm: 0; VA: donor own
junction 25 of 26 donor sides (24 + 1 tube-bridged), v5 junction 1, host (D21) 2; D10 corridor 26 / 26 donor sides
(2 medial excursions flagged); orientation 0 tier A, 2 LVA tops in the D8 20-30 mm band; donors one-to-one.
**route13 (item b) ADOPTED**: on the D19 / D20-moved fork the plain joint placement's RVA takes 3574 v4 fork voxels
(RCCA s 58-124); the route13 move takes 0 (both VA sides rung 1, rechecks pass). No fix loop was needed among the 14.
Tree copied to `/rp/tree` for the bake. The re-pilot's composed tree equals the DRY's final tree on 14 / 14 (centreline
files byte for byte, xforms equal).

### 5.1 Bake, checks, SOFA, figures (`results/repilot_summary.json`, table_md there)
- **bake_v6 1.1: 14 / 14 pass** (rung 1 on 7, rung 2 VA-local windows on 7; windows on the RIGHT carotid pieces: w_035 lower_R [109.1, 111.1] +
  siphon_R [132.5, 133.6], w_025 eca_R [7.9, 9.1] (the ECA stub at the bifurcation), w_050 lower_R [105.9, 107.5] (v4 had
  the same window), w_051 siphon_R [200.1, 202.1]; the rest on left / VA sections). c4 applied on w_025 mr_021_L.
- **checks_v6 1.2: 7 / 14 pass.** Every gate passes on 14 / 14 EXCEPT route_fork_ge_0.99 (7 / 14): strict audit,
  route siphon (0.992-1.026), RCCA prefix unchanged, VA lumen OBJ >= 0.65 (min 0.664), orientation tier A 0, D8
  blocking 0, loader, geom clearance / kinks.
- **load_test_v5 (14, 150 steps): LOAD TEST PASSED.**
- **SOFA D14** (sofa_va_entry 1.1, seeds 777-779, 300 steps): 14 / 14 passed; seeds entering the VA >= 20 mm: v6 33 / 42
  (13 / 14 anatomies) vs stage-1 pilot 30 / 42 (12 / 14) vs v5 22 / 42 (12 / 14). Geometric entry probe: v6 R 14 / L 12,
  stage 1 13 / 11, v5 13 / 12.
- **SOFA on the RCCA training route** (sofa_rcca_v6 1.0, same seeds): v6 14 / 14 passed (seeds past bifurcation 41 / 42,
  siphon 34, target 22) vs v5 14 / 14 (41, 33, 19).
- ICA AP > 40 mm: 0 / 14 (v5 8 / 14). Figures: `/rp/figs/v6_repilot_<name>.png` (v5 | stage 1 | v6, front + side).

### 5.2 The route-fork failures
| anatomy | R rung / pose | mirrored | gate fork | blend zone | beyond blend | C0 (pre-bake) |
|---|---|---|---|---|---|---|
| w_035_left mr_024_L | R2 shift 9.0 mm | yes | 0.944 | 1.096 | 0.926 | 0.997 |
| w_034_right mr_017_L | R2 shift 1.0 | no | 0.974 | 0.513 | 1.006 | 1.008 |
| w_003_left mr_007_L | R1 | yes | 0.978 | 0.813 | 0.983 | 1.001 |
| w_025_right mr_021_L | R1 | no | 0.979 | 1.003 | 0.978 | 0.998 |
| w_003_left mr_010 | R1 | yes | 0.981 | 0.774 | 0.987 | 0.998 |
| w_018b_left mr_002 | R1 | yes | 0.9897 | 0.862 | 0.990 | 0.999 |
| m_006_left mr_025_L | R1 | yes | 0.9898 | 0.966 | 0.992 | 0.999 |
| (passing 7) | | | 0.990-1.051 | | 0.993-1.045 | 0.999-1.014 |
(`/rp/diag/<name>.fork_zones.json`; blend zone = [cut, cut + blend_mm + 2], same trims and ownership as checks_v6.)

### 5.3 Diagnosis (no tool bug found; gate untouched)
1. **C0 does not track the baked value**: C0 0.997-1.014 on all 14, baked 0.944-1.051, uncorrelated.
2. **Sampling noise alone is small**: v4's own union rebuilt with the WHOLE anatomy moved rigidly (5 sub-voxel shifts,
   rotations 5 / 11 / 23 deg; `build/diag/voxphase.py` 1.1) reads 0.990-1.007 of itself (14 poses, 2 anatomies;
   `/rp/diag/voxphase_*.json`) -- one shift of w_034 already reads 0.9899 < 0.99.
3. **Centreline identical**: the v6 RCCA / RECA of the fork are the rigid image of v4's (<= 0.025 mm on 11 / 14, <= 0.15
   on 3; radii within 0.01 mm; `fork_cl.json`).
4. **Surface nearly identical, relief share not**: on w_003 mr_007_L the baked v6 union is v4's union carried by the
   fork map to 0.02 mm (p95; `fork_surfdiff.json`), yet its fork relief share is 2.2 % lower: the frac_0.3mm measure
   flips cells at the 0.3 mm threshold on sub-0.05 mm differences and on how the union is triangulated around the moved
   section (`fork_depth.json`: v6 surface sampled at v4's cells -0.5 %, v6's own cells -2.2 %).
5. **Seam-1 blend zone**: the D19 blend + E6 morph removes v4's unblended seam-kink "relief" (v4 has a ledge where the
   rigid inlet meets the host tube at an angle) -- large only on w_034 (blend-zone relief 17.7 -> 9.1 mm2 = 3.1 % of the
   fork's relief; beyond the zone w_034 is 1.006); elsewhere <= 0.7 %.
6. **Host-cut shift + bake handle windows**: w_035 (R2, 9 mm of donor CCA replaced by host tube; 1.95 mm of the fork and
   1.15 mm of the siphon tubed by the bake's handle search) 0.926 beyond the blend; w_025's handle window on the ECA stub
   (eca_R [7.9, 9.1]) sits at its largest loss bin (RCCA s 71, the bifurcation: relief 14.6 -> 7.9 mm2, fork_bins.json).
7. Counts under alternative definitions (information only, NOT applied): fork beyond the blend zone >= 0.99: 10 / 14;
   C0 form ([cut + 10, end]) >= 0.99: 9 / 14; whole fork >= 0.97: 13 / 14.
Conclusion: the 0.99 fork gate was calibrated when the fork never moved (v5: identical union -> 1.000); with D19 / D20
re-posed forks the baked ratio scatters ~+-3 % from composition and triangulation effects that no pre-bake estimate
here predicts. Not fixable inside the tools without changing which forks ship or how the gate is measured -> decision
(section 7, issue 0).

## 6. Decisions taken here (reversible; how)

- **Joint-pass order = the hardening agents' recipe** (W1 = VA on v5 -> carotid dry on W1 -> write with the plan -> W2 = VA on
  C -> rechecks), not a per-anatomy loop: the carotid dry must run in name order (split guard), the VA steps parallelise.
  Undo: none needed (the per-anatomy `all` mode with `--plan` reproduces an anatomy of the DRY).
- **Fix loop (bounded, 2 iterations)** only for VA-vs-CAROTID contacts (recheck / geom names a carotid curve): compose
  with the VA tool's final VAs HARD, then re-place the VA side. Adopted only when every recheck passes; otherwise the
  anatomy keeps its W2 result and is listed. Undo: skip stage `fix`.
- **route13 adoption rule**: both VA sides pass_all, every recheck passes, LOST (v4 fork voxels taken) does not rise.
  Otherwise the plain joint placement is kept (recorded). Undo: drop stage `route13`.
- **c4 stays inside bake_v6** (designed there with its revert-on-bake-failure); the composed provenance records the
  plan (apply_c4 write=False).
- **graft_v6_va 1.2 construction parameters** (not gates): FILLET_MARGIN_DEG 2 (target 58 on the written sampling; the gate
  stays 60), JUNC_SIGMA_MAX_MM 3, D17_SPREAD_MAX_FRAC 0.75 (a spread corner keeps the skeleton >= 0.25 r inside the old
  tube; post-bake "centrelines inside" still applies). Undo: graft_v6_va_1p1.py.
- **SOFA on the RCCA route** with a closed-loop tool built from the D14 tool (no RCCA navigation test existed;
  load_test_v5 only loads + steps). Reported beside load_test_v5 on the re-pilot; s2_checks can run it on all 223.
- **The 1.1-based partial DRY (112 W1 anatomies) was discarded** (`build/void_dry223_va11/`) when 1.2 was needed, so
  that the whole DRY ran one toolchain and the re-pilot can equal it.
- **Route-fork failures: diagnosed, not "fixed"** (2026-10-06 afternoon). No tool bug was found; making the 7 pass would
  need either a different measurement (a user decision) or giving up D19 on those forks via a post-bake ladder (changes
  which forks ship at scale). Neither is an integration fix, so both are left to the orchestrator with the evidence
  (section 7 issue 0); checks_v6 / bake_v6 untouched. Undo: n/a.
- **Re-pilot post part**: the session cut-off left a partial `post/load_test.log` that the chain would have taken as
  done; it was deleted and `post` re-run (LOAD TEST PASSED).

## 7. Open issues (for the orchestrator)

0. **BLOCKING for stage 3: the right-route fork gate (>= 0.99 of v4, checks_v6 1.2) fails on 7 / 14 re-piloted
   anatomies** (0.944-0.9898); every other gate passes on 14 / 14. Expect about half of the 223 to fail it after the bake.
   Evidence and mechanism in 5.2 / 5.3: not a tool bug, not sampling noise; the gate was calibrated on unmoved forks.
   Options (none applied; the gate is untouched):
   (a) KEEP THE GATE, post-bake right-fork ladder: a baked fork < 0.99 refuses that variant -> next composer variant
       (compose_v6 needs a variant-level refusal list, today only --right-skip-rungs) -> VA pass -> rechecks -> bake ->
       checks; terminal = R5 (v5's pose, reproduces v4's fork: stage-1 pilot 1.000) or, for contralateral forks, R6
       (v5's UNmirrored pose: D20 not applied). R2 shifts keep R1's rotation, so they move the baked ratio little (whole-
       anatomy shifts of v4: +-0.6 %); R4 deviations change it more. Cost ~10-15 min per extra variant per anatomy; an
       unknown share of the ~110 failing anatomies would end at R5 / R6 (losing D19, and D20 at R6).
   (b) USER DECISION, re-define the measured zone: fork beyond the seam-1 blend zone (D19 mandates the blend there;
       v4's "relief" there is its unblended seam ledge): 10 / 14 pass; the C0 form [cut + 10, end]: 9 / 14.
   (c) USER DECISION, tolerance for re-posed forks: whole fork >= 0.97 -> 13 / 14 (w_035, the 9 mm host-cut shift with
       bake windows, still fails -> (a) for it).
   (d) Possible bake refinement (not tried): E3 searches changed carotid sections together with the VAs when localising
       a handle; preferring VA-only windows first would have kept w_035's fork / siphon and w_025's ECA stub real (2 of
       the 7 failures only).
   My recommendation: (b) or (c) by user decision for the measurement, with (a) as the gate-keeping fallback for what
   still fails -- (a) alone would keep the gate but give up D19 on an unknown, probably large, share of right forks.
1. **Decision conflict (DRY)**: w_017_left mr_002_L R ends at D9 rung 3 (v5's top as shipped, flagged, va_target 55.3)
   whose v5 top is ORIENTATION TIER A -- the invariant "orientation gates 0 tier-A" fails on this 1 anatomy (orient_tree
   on the DRY final: 1 / 223). D9 says never drop a donor VA; D18 allows no other relaxation. Needs a rule (e.g. drop the
   donor on this side = D21 host VA with an anatomical top, or accept tier A here).
2. w_017_left mr_002_L L (D9 rung 2, pair flag) and the 8 D10 corridor failures (6 native-offset excursions, inherited
   v5 stretches on rungs 2 / 3) are flagged, not gated.
3. VA agent's watch items now closed or carried: fork-VA contacts (w_018b x2 + w_027_right mr_020) cleared by the fix
   loop (R4 map_dev_bend60, flagged); route13 adopted (LOST 3574 -> 0; its fork 0.9897 in the re-pilot, v5 0.956);
   w_013_right mr_021_L route excess -- check its route after the bake in stage 3.
4. Siphon C0 flags (12 between 0.990 and 0.995 in the DRY) -- the re-pilot siphons all passed (0.992-1.026), but the same
   scatter applies to siphons on the 223.
5. D21 (17 no-donor sides) out of scope as instructed.
6. Stage 3 runs: SOFA RCCA on all 223 after the bake (sofa_rcca_v6 1.0, 3 seeds x 300 steps, ~4 min per anatomy: ~15
   container-hours); D14 on all 223 with sofa_va_entry 1.1 (~4 min per anatomy). bake + checks ~8-10 min per anatomy.
   The DRY tree `/bld/dry223/final` is the composed input (bake_v6 --tree; c4 runs inside the bake).

Status: complete
