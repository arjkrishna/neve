# va_narrow: the three v5 VAs whose OBJ lumen is below 0.65 mm

Task (user, via the v6 audit): check the three VAs with an OBJ wall clearance under the 0.65 mm wire criterion
(0.35 mm wire radius + 0.3 mm SOFA contact distance, v4's navigable rule):
case_w_025_right__topcow_mr_021_L LVA 0.578 mm and RVA 0.632 mm; case_w_027_right__topcow_mr_022 RVA 0.646 mm.
Real narrowing in the donor's anatomy, or an artefact (which step)? Does a guidewire pass? Did v4 have it?

All work is in `/sp/v5_va/tbo/va_narrow/` (read-only on the repo). State: `state/va_narrow.md`.

## Answer so far

**None of the three is a real narrowing in the donor's anatomy that the CT supports, and none blocks a wire.**
Each value below 0.65 mm comes from a chain of steps:
1. The source surface is narrow at that point:
   - the two donor points: a ct_seg4 segmentation artefact where the VA lies against bone in a V2 transverse foramen;
   - the TopBrain point: a narrowing in the TopBrain label itself, also present in v4.
2. The graft floors the centreline radius at 1.0 mm (`VA_MIN_R`), so the union's tube floor holds the lumen at about
   0.93–0.95 mm (iso_v5), and the 60 k mesh at 0.87–0.92 mm.
3. Decimation to the 23.5 k OBJ pulls the wall about 0.3 mm in on one side of this small, about 1 mm radius section.
   The clearance measured at the centreline point drops to 0.578, 0.632 and 0.646 mm.
4. A wire is not tied to the centreline. In every section the OBJ still has a point with 0.75–0.85 mm clearance, only
   0.14–0.26 mm off the centreline.

| VA | where (anatomy s) | segment | source of the narrow surface | union / full / OBJ at centreline | OBJ best wire path | v4 |
|---|---|---|---|---|---|---|
| w_025_right mr_021_L LVA | 92.7 mm (narrow 87–96) | donor Song AHMU1218113 R (x-mirrored), donor s 86, V2 ≈ C4 | ct_seg4 cut the bone side of the lumen (CT: lumen HU to 2.0–2.3 mm, surface at 0.2–0.4 mm) | 0.934 / 0.872 / 0.578 | 0.753 mm | host VA tube there; whole-LVA min 0.863 (TopBrain piece) |
| w_025_right mr_021_L RVA | 219.7 mm (narrow 215–224) | TopBrain piece topcow_mr_021 lVA (x-mirrored), 34 mm past the splice | the TopBrain label narrows (label 24 r_eq 0.70–1.04 against 2.1–2.4); V3 horizontal loop, 4.5 mm above the label's lower end | 0.935 / 0.898 / 0.632 | 0.829 mm | **same TopBrain point**, v4 OBJ 0.850 / full 0.864 |
| w_027_right mr_022 RVA | 93.6 mm (narrow 91.6–94.6) | donor ia172_192 R (native), donor s 61, V2 ≈ C5 | ct_seg4 cut the bone side (CT: lumen HU to 1.5–2.0 mm, surface at 0.3–0.7 mm) | 0.946 / 0.866 / 0.646 | 0.854 mm | host VA tube there; whole-RVA min 1.029 |

**SOFA could not reach these points.** No SOFA run existed for these anatomies, so new runs used `checks/sofa_rva.py`
unchanged:
- 3 episodes on each anatomy, plus 8 more on w_027 (seed 900);
- every run PASSED as a simulation: 0 exceptions, 0 non-finite steps, the tip never outside;
- in all 14 episodes the wire stopped at or before the donor ostium (s 32.9; max RVA s 28–32.9 mm), on the junction
  stub / subclavian side. That is the known v5 entry limitation (run4: 9 of 23 donor RVAs entered), not the narrow
  point;
- the LVA cannot be reached from the training insertion;
- the TopBrain point (187 mm past the ostium) lies beyond anything run4 reached.

Passage therefore rests on the geometry: a path with 0.75–0.85 mm clearance exists in the shipped OBJ.

**The donor artefact is systematic** (`floor_scan.json`, `bone_scan.json`):
- 377 of the 429 donor VAs use some donor stretch whose MISR is below 1.0 mm (floored); median 11.2 mm per VA, p90
  35.9, max 72.1.
- 140 such stretches of at least 2 mm in the used range were checked on the CT (26 donors; VMR 0248 skipped, no NIfTI).
  At each stretch's narrowest point the independent CT lumen r_eq is a median 1.78 mm, against 1.04 for the stage-A
  surface and 0.76 for the MISR:
  - 131 of 140 have a CT r_eq of at least 1.0 mm;
  - 83 of 140 (489 of 755 mm) are at least 1.5 times the surface, against 7 of 136 reference points on the same sides;
  - a median 28 % of rays end at a brighter neighbour, against 0 % at the reference points.
- So in most floored stretches the v5 lumen is the 1.0 mm tube floor where the donor's CT lumen is about 1.8 mm. This
  explains the lower median VA lumen (0.834 mm in v5 against 0.978 in v4).
- 8 points read a CT r_eq under 1.0 with almost no bright contact. These are candidate real narrowings or trace
  problems, not inspected.

## 1. Where the narrowest point is (profile.py → profile_<name>_<side>_<v5|v4>.json)

Per centreline point:
- the exact distance to the OBJ, collision_full and the undecimated union (`va_sources/v5_reg/iso_v5`);
- the signed distance to each section's own real surface, transformed exactly as `sdf_union.load_surface` does;
- the cross-section (plane normal to the centreline) of every surface: area-equivalent radius, in-plane clearance and
  minimum width.

**w_025_right LVA.** Donor stretch vaC_L runs over [6.98, 215.99] mm; its only handle window is at [53.81, 54.96], far
from the narrow point.
- The donor surface narrows over 87–96 mm: r_eq 1.10–1.13 mm and minimum width 1.66 mm at 91.7–92.7, against r_eq
  about 1.95 and width 3.65 on both sides.
- The centreline radius sits at the 1.0 floor from 89.7 to 94.7.
- At s 92.7:

  | surface | r_eq | clearance at the centreline |
  |---|---|---|
  | iso_v5 union | 1.22 | 0.95 |
  | collision_full | 1.21 | 0.93 |
  | OBJ | 1.12 | 0.578 (in-plane 0.63) |

**w_025_right RVA.** The minimum (s 219.7) is NOT on the donor stretch, which is [32.89, 185.94] mm and whose own minimum
is 0.665 at s 158.2. It is in the TopBrain piece va_R (s ≥ 185.94).
- The TopBrain surface section narrows to r_eq 0.67–0.86 (width 0.99–1.29) at 216.7–218.7.
- The centreline radius sits at the 1.0 floor from 215.7 to 221.7.
- iso 0.934, full 0.923, OBJ 0.632.
- CAROTID_V5.md's "where the donor radius sits at its 1.0 mm floor (Song 113 L/R)" is therefore wrong for this RVA: the
  floored radius there is the TopBrain piece's.

**w_027_right RVA.** Donor stretch vaC_R runs over [32.93, 178.21] mm.
- A short focal narrowing at 91.6–94.6: donor surface r_eq 0.99–1.09 and width 1.61–1.79, against 1.3–1.5 and 2.6–2.9
  around it.
- iso 0.94, full 0.86, OBJ 0.646.

None of the three points lies on the junction stub, the origin seam, the splice, the 25 mm calibre ramp or a handle-rule
tube window.

## 2. The donor CT (ct_check.py → ct_<key>.json / .png; ray_dump.py → ray_<key>.txt / .png)

**Mapping.** The anatomy centreline is mapped back into the donor's native RAS frame with the inverse of the bake
transform. The mapped points fall 0.04 mm from the donor's own centreline.

**Per mm, compared:**
- the ct_seg4 station: its 64 rays' r_e, the ray rule kinds, I_c / I_bg, and its open-bright / fix-8 flags;
- the donor MISR and CT calibre;
- the section r_eq of the ct_seg4 label, Arterial's tsba VA mask, Arterial's full segmentation and the stage-A surface;
- an independent half-max read of the shipped ct.nii.gz;
- HU along 16 rays at the narrow and at reference stations.

**w_025_right LVA → Song AHMU1218113 R.** Siemens, 1.0 mm slices on a 0.49 × 0.49 × 0.70 mm grid.
- Narrowest at donor s 86.2. With C6 at 39 and C2 at 137, that is V2, around the C4 foramen.
- Stations 81–89: 18–26 of the 64 rays end by the "bright neighbour upper" rule (kind 3), 5–16 by "dip before
  bright", and r_e min is 0.35–0.6 mm. The core HU holds at 484–508, the same as elsewhere.
- **HU along the bone-side rays (45–157°)** at the narrowest station: 523 525 526 526 525 522 517 511 500 489 507 537
  645 778 … HU (r = 0–3.25 mm, 0.25 mm steps). The lumen level runs to about 2.0–2.3 mm, then rises into bone
  (900–1500 HU from about 3 mm). The stage-A surface on those rays sits at 0.20–0.41 mm.
- On the free side the half-max falls at 1.25–1.45 mm. The station centre sits off the lumen centre, toward the bone.
- CT lumen width: about 1.3 + 2.2 ≈ 3.5 mm, against 3.6–4.0 mm at the reference stations 12 mm away, so at most a
  10 % narrowing. The surface shows r_eq 1.1 against 1.9 (42 %).
- Arterial's mask narrows only to r_eq 1.53–1.60 (neighbours about 1.75).
- The segmenter's own QC marked this stretch. Stations 78–93 are the `keep_connected_fix8` run "not opened": the 0.8 mm
  opening would have cut the label there, because it was already pinched on the bone side.
- The record lists the 40–70 mm stretch as "bright neighbour".

**w_027_right RVA → ia172_192 R.** Toshiba, 0.507 × 0.507 × 0.4 mm.
- Narrowest at donor s 61. With C6 at 39 and C2 at 115, that is V2, around the C5 foramen.
- The station is open-bright flagged ([56.5, 64.25]); 17 of its 64 rays are kind 1 or kind 3.
- **HU along the bone-side rays** at −1 mm: 498 477 462 455 454 457 454 460 495 568 705 850 … The lumen level runs to
  1.75–2.0 mm, then bone. The stage-A surface sits at 0.30–0.47 mm on those rays.
- The free side's half-max is 1.3–1.45 mm, against 1.45–1.70 mm 8 mm downstream.
- CT width is about 3.2 mm at the narrow point and about 3.2 mm 8 mm downstream: no narrowing.
- The Arterial mask shows r_eq 1.36 here, against 1.2–1.5 around: no narrowing.

**Verdict for both donor points: a segmentation artefact, not stenosis.** Where the VA lies directly against the
transverse-foramen bone with no soft-tissue dip, ct_seg4's ray rules end the lumen 0.2–0.7 mm from the centre on the
bone side:
- "bright neighbour upper" (T_hi) and "dip before bright";
- the threshold field min-filtered over 5 stations × 3 rays.

The CT keeps lumen-level HU for another 1.3–2.0 mm. The stage-A surface and its VMTK MISR (0.75–0.86 mm) inherit the
cut. In v5's union the 1.0 mm floor hides it.

Caveat: the bone-side plateau could in principle include iso-dense venous plexus, which cannot be separated at this
resolution. Its width matches the artery's own calibre elsewhere, so the simple reading is the artery.

## 3. The TopBrain point (tb_check.py → tb_w025_RVA.json / .png)

No MRA is on disk: the TopBrain release here holds labels only. The check is therefore against the label.
- `topcow_mr_021.nii.gz` label 24 is L-VA. The RAS convention holds: 293 / 293 surface samples lie on or next to label
  24, and 0 under LPS.
- The label itself narrows: r_eq 0.86 / 0.70 / 0.93 / 1.04 at −3 / −2 / −1 / 0 mm, against 2.1–2.4 mm 10–15 mm away.
  The TopBrain surface follows it (0.67–1.03).
- Where it sits:
  - 4.4–5 mm above the lower end of label 24 (label z −87.1 to −48.8; the point is at z about −82.6);
  - on a horizontal stretch: z changes by under 1 mm over 11 mm, which is the V3 loop over the atlas;
  - this is the classic place for in-plane flow saturation in an axial TOF slab.
- So the narrowing is real in the TopBrain annotation, but whether the artery is narrow cannot be checked here.
- **v4 has the same narrowing at the same TopBrain point.** TopBrain-piece arclength is 219.7 − 185.94 + 5.0 (v5 start
  trim) = 38.8 mm, and in v4 179.4 − 140.6 = 38.8 mm.
  - v4 OBJ 0.850, full 0.864: v4 passed the 0.65 rule there by its decimation, not by geometry.
  - v5 OBJ 0.632, full 0.923.

## 4. Wire path (passage.py → passage.json)

v4's rule measures clearance at the centreline point. A wire can sit anywhere in the section.
- Method:
  - sections every 0.25 mm over s0 ± 6 mm;
  - inside each section's lumen loop, the exact distance to the mesh on a 0.04 mm grid;
  - the best point's clearance is taken, then the minimum over the sections.

| VA | OBJ at centreline | OBJ best path (offset from the centreline) | full best | iso best | v4 OBJ best |
|---|---|---|---|---|---|
| w_025 LVA | 0.578 | **0.753** (0.18 mm) | 0.916 | 0.935 | 0.884 (TopBrain piece) |
| w_025 RVA | 0.632 | **0.829** (0.14 mm) | 0.914 | 0.930 | 0.893 (same TopBrain point) |
| w_027 RVA | 0.637–0.646 | **0.854** (0.26 mm) | 0.926 | 0.952 | 1.083 |

A path with at least 0.65 mm clearance exists through all three on the shipped OBJ.

## Guidewire in SOFA

- No SOFA run of these two anatomies existed: `checks/sofa/run4` holds 25 others.
- The runs used `checks/sofa_rva.py` unchanged: run4's final schedule from the training insertion into the RVA,
  300 steps per episode, one container at a time.

| run | anatomy | episodes | result | max RVA s | donor ostium |
|---|---|---|---|---|---|
| vanarrow_sofa, seed 777 | w_027_right mr_022 | 3 | PASSED, not exercised | 29.0 | 32.9 |
| vanarrow_sofa, seed 777 | w_025_right mr_021_L | 3 | PASSED, not exercised | 30.0 | 32.9 |
| vanarrow_sofa2, seed 900 | w_027_right mr_022 | 8 | PASSED, not exercised | 32.9 | 32.9 |

- Every episode: 0 exceptions, 0 non-finite steps, the tip never outside the wall.
- The wire runs on into the subclavian continuation (17) or stops at the 5.9 mm junction stub.
- entry_geom: the donor VA leaves at 39.6° (w_025) and 71.2° (w_027) to (17), against 88° at v4's host T-junction.
- **So SOFA cannot test the narrow points under the training schedule.** The blocker is the donor ostium (v5's known
  entry limitation; the v6 subclavian-junction follow-up), not the narrowing.
- The LVA (w_025) cannot be tested: the left side is not reachable from the training insertion.
- The w_025 RVA TopBrain point (s 219.7) lies past anything run4 reached (max 168.5 mm).
- Outputs:
  - `tbo/va_narrow/sofa/v5/*.json` and `sofa/v5_seed900/*.json`;
  - logs in `sofa/logs/`.

## 5. Is the donor artefact systematic? (floor_scan.py → floor_scan.json; bone_scan.py → bone_scan.json, bone_scan/)

**floor_scan.** Stretches of each donor side where the VMTK MISR is below 1.0 mm. The graft floors these radii to 1.0,
so the union's tube floor, not the donor wall, sets the lumen there.
- 52 of 54 donor sides have such stretches.
- 377 of the 429 v5 donor VAs use at least one inside their used range [0, splice]: median 11.2 mm per VA, p90 35.9,
  max 72.1; 52 VAs use none.
- Most of the floored length sits at ct_seg4 stations where 10 or more of the 64 rays end at a bright neighbour.

**bone_scan.** The independent CT read at the narrowest point of every floored stretch of at least 2 mm in the used
range: 140 points on 26 donors (VMR 0248 skipped: no thin_slice NIfTI). Each is compared with a reference point on the
same side (MISR ≥ 1.3, at least 5 mm from any floored stretch).
- The method: 64 rays; a ray ends where HU falls below the soft-tissue half max or rises above I_c + 150.
- Calibration at the reference points: CT / stage-A r_eq ratio p10 / median / p90 = 1.00 / 1.04 / 1.33; bright-ray
  share median 0.

Results at the 140 floored points (p10 / median / p90 where three values are given):

| quantity | floored points |
|---|---|
| MISR | 0.56 / 0.76 / 0.94 |
| stage-A r_eq | 0.73 / 1.04 / 1.47 |
| CT r_eq | 1.14 / 1.78 / 2.37 |
| bright-ray share | 0.09 / 0.28 / 0.47 |
| CT r_eq ≥ 1.0 | 131 of 140 |
| CT / stage-A ≥ 1.25 | 119 of 140 (654 of 755 mm); reference 21 of 136 |
| CT / stage-A ≥ 1.5 | 83 of 140 (489 of 755 mm); reference 7 of 136 |
| CT / stage-A ≥ 2.0 | 32 of 140; reference 0 of 136 |

- The 8 points with a CT r_eq under 1.0 and almost no bright contact are at:
  - ia172_015 R s 151; ia172_039 R 126 and L 125; ia172_058 R 48;
  - ia172_071 R 87 and 89, L 110; ia172_190 L 120.
  - At five of them the read failed (r_eq 0.05: the brightest point within 1 mm is not lumen).
  - They are candidate trace problems or real narrowings, not inspected.
- Song 094 L s 37 is also under 1.0, but bright contact is high there (HU 2170).
- **Conclusion:** the w_025 / w_027 mechanism is the rule, not the exception. Where a donor VA touches the foramen bone
  (or a dense vein), the ct_seg4 surface and its MISR are too narrow. v5's 1.0 mm floor then sets the lumen in a
  stretch the CT shows at about 1.8 mm. The donor's real wall is lost there, and the lumen is under-sized but never
  closed.
- This explains why v5's median VA lumen minimum (0.834 mm) is below v4's (0.978).

## v6 implications (for the orchestrator)

- No anatomy needs a geometry fix for wire passage. The three OBJ values are decimation of a floored 1 mm tube; the
  best path keeps 0.75–0.85 mm.
- **The real issue is systematic** (section 5). The ct_seg4 donor surfaces are cut on the bone side wherever a V2
  segment touches the foramen. 83 of 140 checked floored stretches are at least 1.5 times wider in CT than the surface.
- The 1.0 mm floor keeps the lumen open but under-sized there (about 1.0 against about 1.8 mm), and it replaces the
  donor's real wall with a plain tube.
- Possible v6 fixes in the donor segmentation, then re-bake:
  - end bone-side rays at the end of the lumen plateau, or at the half-way crossing between the lumen and the bone peak,
    instead of the min-filtered T_hi;
  - or take the bone-side edge from Arterial's mask;
  - or floor the graft radius at the CT calibre measured with that rule, rather than at a fixed 1.0.
- After the fix, re-check `floor_scan` / `bone_scan`, the VA lumen and the wall-detail figures.
- The w_025 RVA point is a v4 heritage of the TopBrain label (topcow_mr_021 L-VA, V3 loop). Leave it, or tube it at the
  neighbouring calibre. The same would apply to any other TopBrain VA piece with a label pinch.
- Record correction: CAROTID_V5.md says the three points sit "where the donor radius sits at its 1.0 mm floor
  (Song 113 L/R, ia172_192 R)". That holds for w_025 LVA and w_027 RVA. For w_025 RVA the floored radius is the
  TopBrain piece's: the point is 34 mm above the splice, and the donor stretch minimum is 0.665 mm.

## Files

- Scripts, all in `/sp/v5_va/tbo/va_narrow/`:
  - `profile.py`, `ct_check.py`, `ray_dump.py`, `tb_check.py`, `passage.py`;
  - `floor_scan.py`, `bone_scan.py`;
  - `sofa_vn.sh` and `sofa_vn2.sh`, which run `checks/sofa_rva.py` unchanged;
  - `dr.sh`;
  - `q*.py`, small readers.
- Outputs:
  - `profile_*.json`;
  - `ct_w025_LVA.json/.png`, `ct_w027_RVA.json/.png`;
  - `ray_w025_LVA.txt/.png`, `ray_w027_RVA.txt/.png`;
  - `tb_w025_RVA.json/.png`;
  - `passage.json`;
  - `floor_scan.json`;
  - `bone_scan.json`, `bone_scan/<unit>.json`, `bone_scan.log`;
  - `sofa/v5/`, `sofa/v5_seed900/`, `sofa/logs/`.
- All containers have exited: vanarrow_sofa, vanarrow_sofa2, vanarrow_bone, and the foreground `--rm` runs.
