# v6_pilot notes -- the 14-anatomy pilot built end to end

Agent v6_pilot, 2026-10-04. State: `state/v6_pilot.md`. Code: `scratchpad/v6/pilot_build/` (+ `figs/fig_v6_pilot.py`).
Outputs: `D:\Arjun\neve\carotid_data\v6_work\pilot\` (`/pw` in the containers): **`tree/<name>/` = the pilot anatomies**
(Centrelines_comb, provenance.json, vessel_architecture_collision.obj, collision_full.vtp, mesh_v6.json, LICENSE),
`iso/` (undecimated unions), `bake/`, `checks/`, `route13/`, `sofa/`, `figs/`, `records/` (`pilot_table.txt`).
Intermediate trees: `car2/` + `base3/` (unguarded joint pass), `car3/` + `base4/` (guarded joint pass, the final
one), `pass1/` (VA on the carotid pilot), `round1/` and `round2/` (earlier bakes + checks: the evidence for the
fixes), `void/` (invalid or superseded runs, kept for the record). Summary: `scratchpad/v6/results/pilot_summary.json`. Read-only on the repo, anatomies_v2..v5, carotid_tools, topbrain_tools; every
container ran with /repo read-only (runner `pilot_build/dr.sh`, containers `v6pil_*`).

## Answer so far (final)

The 14 pilot anatomies are built end to end in `carotid_data/v6_work/pilot/tree/` and checked. **Recommendation:
NO-GO for stage 2 as the plan's GO rules stand (rules 1, 2, 5, 8 fail), GO once three things are settled**: a D2
va_target rule; the route baseline for anatomies whose v4 RECA tip ran along the ICA; and the pilot's tool fixes
adopted in the stage-2 tools with the all-223 dry runs re-run (section 9, `results/pilot_summary.json` go_no_go).
- Mesh: bake 14 / 14, strict audit 14 / 14 (genus 0, closed, 0 self-intersections, centrelines inside, routes
  navigable). Orientation: 0 tier-A pieces; 2 va_target flags (21.6 / 21.9 mm).
- Right route (exact ownership, v6 / v4): >= 0.99 on 13 / 14; k_011 fork 0.932 is v4's RECA-tip surface counted as
  RCCA relief (0.9997 against v4 rebuilt without its tip). w_018b (item b): fork 0.956 (v5) -> 0.994.
- VA: 20 / 28 sides with the donor's own subclavian junction (take-off vs native median 0.04 deg), 26 / 28 pass every
  VA gate; lumen on the polyline >= 0.769 mm (v5: 4 VAs < 0.65, min 0.547); walls coarse (donor 0.135, subclavian
  0.70). Loader 14 / 14, load test passed, SOFA 14 / 14 passed with RVA entry >= 20 mm in 7 / 14 (v5 5 / 14).
- Fixes found by the pilot (no gate relaxed): the joint pass order; VA G4 (distal subclavian vs seam-1 parents, a real
  collision in w_025); route guard in both tools; bake E2 (crash), E3/E3b, E6 (blend morph), E7 (VA-aware OBJ cut).

## 1. Merging the carotid and VA sides (build order carotid -> VA)

- **pass 1** (`pass1/`): the VA tool (graft_v6_va 0.8, frozen writer dry10w, imported unchanged; driver
  `va_on_carotid.py`) run ON the carotid tool's pilot tree. 27 / 28 sides placed; **w_050_left mr_004_L RVA: no top
  passes** (all 30 ranked tops hit the v6 RCCA siphon / LCCA / LECA; `dbg_top.py`). Cause: each tool gated against
  the OTHER side's v5 geometry (the carotid tool treated the v5 VAs as soft; the VA tool placed against v5's carotid
  side), so their outputs can collide.
- **joint pass** (no tool changed): (1) the carotid tool re-run with the VA pilot tree as its base (`car_on_va.py`;
  strict soft mode now prefers poses clear of the v6 VAs); (2) its carotid side with the VA side reset to v5's
  (`make_base3.py`: the VA tool assumes a v5 VA side in its base -- a run on the carotid output directly used the v6
  subclavian as "host" and was discarded, `void/`); (3) the VA tool on that base. Unguarded: `car2/` 14 / 14 ok,
  0 soft / 0 hard contacts, the same 3 substitute forks as the dry plan (split unchanged), siphon poses changed in 9;
  26 / 28 VA sides pass, w_050's RVA places (`records/va_round1`, `records/va_cmp.txt`).
- The 2 failing VA sides are the VA tool's own D2 cases (failing on v5's carotid side too): k_011_left mr_012_L RVA
  (v5-junction fallback, va_target 21.6 mm) and w_018b_left mr_002 LVA (va_target 28.3 mm; 21.9 after the route13 move).

## 2. Round 1 (unguarded joint pass, `round1/`): what broke

Baked 10, bake failed 1, 3 not reached (stopped once the causes were clear).
- **w_025_right mr_021_L bake FAILED** ("even the tube-only mesh has a handle"): `tube_handle.py` -> the new (17)
  (donor subclavian distal stretch) overlaps the host RCCA by 2.8 mm (RCCA s 27-29). **VA tool bug G4**: the distal
  new subclavian part (F -> E -> bridge -> H) was gated only against `others_nb`, which leaves out seam 1's parents
  (BT + RCCA on the right, the arch on the left) -- they are not its parents. Gate made complete (not relaxed).
- **Right route under exact ownership below 0.99 of v4** in 2 of the first 7 checked: w_051_right mr_008 siphon 0.989,
  k_011_left mr_012_L fork 0.981 (`round1/checks`, `checks_v6.py`; v5: 1.000 / 1.027). Located with `diag_siphon.py`
  (per-2-mm RCCA bins, v6 vs v4) and `route_guard.py` (calibration `checks/rg_cal.json`): w_051's anatomical LVA top
  (aimed at the D2 target) passes 6.4 mm from the distal right siphon and takes 1.8 % of the siphon's wall relief
  (v4's LVA: 0); k_011's re-posed RECA tip (D4 kept it, grid pose 20 deg off anatomical) passes 5.3 mm from the ICA
  and takes 9.1 % of the fork relief vs 7.6 % in v4. Neither tool had a route-protection term (v5 fixed 13 such
  anatomies after its bake by moving the VA, route13).

## 3. The route guard (fix; no gate changed)

`route_guard.py`: relief points = vertices of the RCCA real sections (lower_R, siphon_R) placed as the bake places
them, inside the section capsule, owned by the RCCA among the fixed branches, standing > 0.3 mm off the RCCA tube;
a candidate piece takes a point when its tube value exceeds the RCCA's minus the ownership margin (0.5 mm; 0 for the
fork's own ECA, which only re-attributes). Excess = share taken in v6 minus the share the same branch took in v4.
- `car_guarded.py` (C1, right ECA tip): the first passing tip pose (tool order) with excess <= 0.3 % of the fork
  relief, else skip (D4 spirit). k_011: a different grid pose, excess 0.
- `va_guarded.py` (G1 place_top: first passing top with excess <= 0.3 %, else the least; G2 place_side: excess in
  the pose score + early stop; G3 guard per anatomy; G4 the distal-gate bug fix above).
- `route13_v6.py` v2 runs on va_guarded (ROUTE13 objective slot in G2).
- Final tree = car3 (guarded carotid) -> base4 -> va_guarded -> route13 (w_018b) -> c4 (w_025).

### 3b. Round 2 (guarded tree, `round2/`) and what it taught

- The guard fixed w_051 (siphon 0.989 -> 1.061) and kept w_050 / w_025 / w_027 / w_018b >= 0.99; G4 fixed w_025's bake.
- **k_011 fork stayed at 0.933** with every RECA choice: car2's tip (1.31 mm from the ICA, round 1: 0.981), the
  relief-guarded tip (0.87 mm), the C2 tip (>= v4's 2.84 mm clearance: 3.78 mm) and no tip at all. `v4_no_tip.py`
  rebuilt v4's own union WITHOUT its RECA tip: fork 0.2034 -- exactly v6's (ratio 1.0002). So the deficit is the v4
  baseline: v4's tip ran along the cervical ICA (2.84 mm, RCCA s 110-118) and the exact-ownership measure counted its
  label surface as RCCA wall relief (`owner_zone.py`: v4 has 64.5 mm2 of RECA-owned cells in that zone, v6 none).
  The ICA's own wall (lower_R, same xform and points as v4) is unchanged. v4's tip pose fails the v6 ECA tilt gate
  (a5), so any v6 that obeys a5 loses this pseudo-relief. Kept: the C2 tip (route-neutral against v4-without-tip);
  C3 ("v5 pose or skip") written, tested, turned off.
- **w_034_right mr_017_L siphon 0.986** (control anatomy): no neighbour within 28 mm; the loss sits at RCCA s 130-134 =
  the siphon seam + its 6 mm blend. **Bake bug E6**: compose_v6 blends the first blend_mm of each re-posed piece's
  centreline but the bake carried the piece's real surface rigidly, so the capsule clipped it there (the carotid
  agent's open issue 5). `bake_v6p` 1.2 morphs the surface in the blend by the blend's own displacement (rigid piece
  rebuilt from the v5 back-map, checked <= 0.01 mm beyond the blend). All 14 re-baked with 1.2 (final round).
- **E3b** (bake_v6p 1.1): the VA margin ladder applied to the changed carotid sections too (k_011's first round-2 bake
  went to margin 1.0 for its siphons); the margin is now chosen by branch (VA branches only).

## 4. Item (b): w_018b route13 move

`route13_v6.py`: the route13 tool's objective (graft_v5_reg_sep 1.1 `SepObjective`, imported unchanged, on diag13's
v4 table `route13/diag/<name>.rcca_v4.npz`, sections lower_R only -- the fork is unchanged in v6, v4's siphon voxels
do not apply to the re-posed siphon) applied to graft_v6_va's own candidate enumeration (score (target ok, guard
excess, net deficit, LOST, RCCA-sep soft, cost); early stop only at LOST 0; every gate graft_v6_va's).
diag13 on v5: the RVA's donor lower VA took 15 532 of the 15 588 lost fork voxels (wall-to-wall 1.03 mm at RCCA
s 99.8). On the v6 tree before the move the v6 RVA still took **1525** v4-kept fork voxels (RCCA s 51.8-65.2); after
the move (pose aim2_20:turn+6y, donor junction kept, target 12.6 mm, RCCA separation 10.7 mm) it takes **0**. The left
VA was re-placed against the moved RVA: va_target 28.3 -> 21.9 mm (still > 20). Records `route13/<name>.json`
(unguarded) and `route13/<name>.guarded.json` (final).

## 5. c4

`c4_apply.py` on w_025_right mr_021_L RVA: applied (19 points, RVA s 199.9-217.8 mm, r_min 1.00 -> 1.65 mm), the
widened stretch clears every other branch by >= 6.4 mm -> `tree/<name>/provenance.json` v6_pilot.c4.

## 6. Bake and checks (final tree, bake_v6p 1.2)

Per anatomy: `records/pilot_table.txt`, `results/pilot_summary.json` -> anatomies.
- **Bake**: 14 / 14 pass (rung 1 on 11, VA-local windows on 3: w_051 siphon_R 2.3 mm, w_035, w_018b); 0 whole
  sections as tube, no flags; 115-364 s each, peak RSS ~1.9 GB. Morph (E6) applied on every blended piece (largest
  displacement per piece 0.10-0.99 mm; rigid rebuild error <= 1e-4 mm).
- **Strict audit** (v4's topo_audit through checks/chk_one.item1): 14 / 14 (genus 0, closed, two-manifold, 0
  self-intersecting faces on OBJ / 60k / union, budgets, both routes navigable, every centreline point inside,
  orientation as v4). k_004 re-checked after its OBJ re-cut (E7).
- **Right route, exact ownership, v6 / v4** (fork, siphon): >= 0.99 on 13 / 14 (fork 0.994-1.027, siphon
  0.991-1.062; w_018b fork 0.956 in v5 -> 0.994). **k_011_left mr_012_L fork 0.932** -- against v4 rebuilt without
  its RECA tip 0.9997 (`checks/v4_no_tip_*.json`): the v4 baseline artefact of section 3b.
- **Orientation (orient_gates tree mode)**: 0 tier-A pieces, 0 primary; v6-blocking: va_target on 2 tops (k_011 RVA
  21.6 mm, w_018b LVA 21.9 mm). With `--fork-gates`: + fork_heading on 22 forks (informational: v5 fork headings kept).
- **Carotid file check** (`carotid/verify_pilot.py` on the final tree): 14 / 14 ok, 0 hard / 0 soft contacts (the v6
  VAs now count as the soft neighbours), 1 mm joins <= 30.9 deg, right route identical to v5 up to seam 2.
- **VA side** (VA tool records as written): 26 / 28 sides pass every VA gate (the 2 va_target sides above); 20 donor
  junctions (ia172, Song and VMR 0248), 6 v5-junction fallbacks, 2 host; take-off vs native |median| 0.04 deg (max
  0.36) on the donor junctions; exact clearance >= 0.683 mm; joins <= 57.6 deg; course RMS <= 19.8 mm.
- **VA lumen on the polyline** (wall/va_lumen_poly.py, OBJ): min 0.769 mm, 0 VAs < 0.65 (v5: 4, min 0.547). The four
  narrow VAs: w_025 RVA 0.547 -> 1.087 (c1 + c4), w_025 LVA 0.567 -> 0.804, w_027 RVA 0.630 -> 0.772, w_047 RVA
  0.637 -> 0.845. **E7**: k_004_left mr_010 LVA read 0.561 on the OBJ (60k 0.887, union 0.927: a decimation
  artefact on the TopBrain top) -> `obj_fix_va.py` re-cut the OBJ from the union at 23 500 tris with the VA lumen as an
  extra acceptance criterion: 0.769.
- **D5 wall detail** (exact ownership, undecimated unions, > 0.3 mm off the tube): donor VA stretch (minus the 25 mm
  seam zone and tube windows) median 0.156 (v5, 26 sides) -> 0.135 (v6) -- still coarse (> 0.10; the c1 plateau rule
  removes the bone-side cut artefact, as v6_wall predicted 0.134 -> 0.112 on its proxy); the new donor subclavian
  stretches median 0.70 (v5: host tube, ~0).
- **Loader**: 14 / 14 load with 18 branches, insertion inside, every centreline point inside the OBJ, RCCA targets
  >= 83 (the only "problems" are the host's sparse point gaps, identical in v5's loader record for all 223).
  `load_test_v5.py` on the tree: LOAD TEST PASSED (14 x 150 SOFA steps, all finite).
- **SOFA** (sofa_rva v5 tool on anatomies_v5; sofa_rva_v6 on the tree): see section 8 / the summary.
- **Figures** `figs/v6_pilot_<name>.png`: v5 and v6, FRONT (from anterior) and SIDE (from the patient's right) views,
  same window for all; the k_011 L_PITCH siphon now rises; the VA tops end near the host confluence.

## 8. SOFA

Same schedule as v5 (sofa_rva, closed loop into the RVA, up to 3 x 300 steps), v5 on anatomies_v5 and v6 (sofa_rva_v6:
ostium on the v6 junction) on the tree (`sofa/v5`, `sofa/v6`): **passed 14 / 14 both** (0 exceptions, 0 non-finite,
tip never > 0.5 mm outside). RVA entered >= 20 mm past the ostium: **v6 7 / 14, v5 5 / 14**. v6 entries: w_003 mr_007_L
(VMR 0248 junction, 71.6 mm), w_003 mr_010, m_006 mr_025_L (host VA), m_006 mr_027_L, w_027, w_035, w_047 (55-137 mm).
Not entered: 4 v5-junction sides (w_051, k_011, w_034, w_050 -- the v5 stub kept), w_025 (13 mm), k_004 and w_018b
(the wire stays at the subclavian, max RVA s 14-15 mm). As in v5 (user, 2026-10-03), entry is geometric diversity under
one scripted schedule, not a pass criterion. RCCA route run: `load_test_v5.py` on the tree, LOAD TEST PASSED.

## 9. GO / NO-GO (plan section 8; `results/pilot_summary.json` go_no_go)

| rule | status | why |
|---|---|---|
| 1 gates | FAIL | 0 tier A; 2 va_target flags (k_011 RVA 21.6, w_018b LVA 21.9 mm) |
| 2 invariants | FAIL as worded | all hold except k_011's exact fork 0.932 vs v4 (0.9997 vs v4 without its RECA tip) |
| 3 D1 | PASS | k_011, w_047 (+ m_006 mr_027_L) get split-keeping substitutes; dry run 223 / 223; split exit 0 |
| 4 D3-fork | NO-GO for the fork re-pose only | pure map: 31 % of right forks need a bend > 45; v5 fork headings kept |
| 5 D2 | FAIL | 2 pilot tops > 20 mm; dry10 94.4 % of 446 tops <= 20 mm (< 95 %) |
| 6 e | PASS | own junctions from ia172, Song and VMR 0248; take-off 0.04 deg; SOFA entry 7 vs 5 |
| 7 c | PASS | VA lumen >= 0.769; c4 applied on w_025; donor wall 0.135, subclavian 0.70 |
| 8 dry runs | FAIL | 36 VA sides of dry10 without a passing configuration (needs the D2 rule) |
| 9 resources | PASS | ~2-6 min bake + ~5 min checks + ~3-5 min SOFA per anatomy |

**Recommendation: NO-GO now; GO after** (1) a D2 va_target rule (the plan's 30 mm fallback passes both pilot tops;
the 5 dry-run sides without any placement need "keep v5's top, flagged" or similar), (2) a decision on the route
baseline where v4's RECA tip ran along the ICA (accept v4-without-tip per anatomy, or keep v4's tip pose against a5),
(3) the pilot's fixes adopted in the stage-2 tools and the all-223 dry runs repeated with them (counts of ECA tips,
VA placements and the split re-checked).

## 7. Stage-2 recipe validated by the pilot (order of the tools)

1. VA tool write on v5's carotid side for all 223 (dry10 exists; the write is needed as the carotid base).
2. Carotid tool (guarded, `car_guarded.py`) with that VA tree as base -> carotid side VA-clean, RECA tips route-safe.
3. Base = that carotid side + v5's VA side (`make_base3.py`); VA tool (guarded + G4, `va_guarded.py`) on it.
4. route13 move for w_018b (`route13_v6.py`); c4 on the 8 mr_021_L anatomies (`c4_apply.py`).
5. Tube-only genus check before baking (`tube_handle.py` first line) to catch centreline fusions early.
6. Bake (`bake_one_v6.py` = bake_v6p E1-E5), checks (`checks_v6.py`), orient tree, lumen, loader, SOFA, figures, records.

## Open issues

1. **D2 va_target** (VA tool): 2 pilot tops end 21.6 / 21.9 mm from the D2 target (k_011 mr_012_L RVA on a
   v5-junction fallback; w_018b mr_002 LVA, 28.3 before the route13 move). The all-223 dry run (dry10) has 36 failing
   sides (6 pass at 30 mm, 6 course-RMS only, 5 with no placement). A rule is needed before stage 2 (threshold
   25 / 30 mm, the siphon-termini reference, or keep-v5-top-flagged for the 5).
2. **k_011-type route baseline**: where v4's RECA tip ran along the ICA, v4's exact-ownership fork counts the tip's
   label surface as RCCA relief; any v6 that obeys a5 (tilt <= 40 on ECA tips) loses it (k_011: 0.932 vs v4, 0.9997
   vs v4 without its tip). Decide: accept the v4-without-tip baseline for these (documented per anatomy), or keep the
   v4 tip pose despite a5. How many of the 223: anatomies whose v5 RECA tip is re-posed or skipped (dry: 18 re-posed +
   28 skipped right/left; the right ones need `v4_no_tip.py`).
3. **Tool changes the pilot needed** (all written as wrappers / source patches in `pilot_build/`, no gate relaxed;
   stage 2 must adopt them): joint pass order (carotid with the VA output as base, then VA on carotid + v5 VA side);
   VA G1-G2 route guard, G4 distal-subclavian gate; carotid C1-C2 RECA-tip guard; bake E2 branch lookup, E3/E3b pin and
   margin by branch, E6 blend morph, E7 VA-aware OBJ re-cut; SOFA ostium on v6 junctions. Re-run the all-223 dry runs
   with them (the guards can change ECA-tip and VA counts).
4. **Route margin**: four anatomies sit within 1 % of the 0.99 line after E6 (w_034 siphon 0.991, w_025 0.997,
   m_006 mr_025 0.998, w_018b 0.994/0.997); at 223 a few will fall below -- the route guard tolerance (0.3 %) and the
   v4 pinned windows on re-posed siphons are the levers.
5. **Left fork order**: the pilot's 3 substitutes equal the dry plan's (split unchanged). The joint pass re-runs the
   carotid tool with the v6 VAs as soft neighbours; at 223 a substitute could change -> re-run the split check.
6. **D3-fork**: fork headings stay v5's (GO rule 4 not met by the pure map: 31 % > 10 %); fork_heading flags on 22
   pilot forks are informational.
7. Not done here: item (d) BUILD_v6.json (stage 2), item (f) donor colour (later).
