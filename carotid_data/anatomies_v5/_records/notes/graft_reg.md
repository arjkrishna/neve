# graft_reg: the v5 registry graft tool

Agent prefix `graft_reg`. State file `v5_va/state/graft_reg.md`. All scripts are NEW files in `/sp/v5_va/`; the
prototype (`graft_v5_va.py`, `bake_v5.py`) and every tracked repo module (`graft_v4`, `graft_three`, `graft_siphon`,
`bake_meshes_v4`, `sdf_union_v4`) are imported or called unchanged. Scratch data is in
`D:\Arjun\neve\carotid_data\va_sources\v5_reg\` (gitignored). No git, no GPU, one container at a time (`--cpus 6`).
`anatomies_v4` was only read.

## Answer so far

1. **Donor records work for all three donor classes.** `graft_v5_reg_donor.py` builds a per-side JSON record (plus a
   copy of the lumen surface) from a thin-slice pilot case, from VMR 0248, or from a 1.5 mm TotalSegmentator trace.
   - Built: VMR 0248 R / L, and the five pilot sides passing so far (ia172_058 R, ia172_061 R, ia172_141 L,
     ia172_015 L, Dissec_01 L).
   - Tube class: the TotalSegmentator pairs that pass the recalibrated gate (s0610, s0777, s1786, s0960). s0960 has no
     Arterial label, so its subclavian comes from the TS mask, built in neve-build-stagea.
   - VMR's record reproduces the prototype exactly: take-off 53.5° / 58.7°, subclavian elevation 63.9° / 36.3°.
   - Every pilot stage-A surface has handles (genus 1–6). The one baked (061 R, genus 6) bakes clean.
2. **The graft generalises without changing what v4 and the prototype do.** `graft_v5_reg.py --compat` (prototype
   settings, VMR record) reproduces the prototype's final 223-anatomy dry run leaf for leaf: 9366 leaves,
   **0 differences**. Its origin-turn measure reproduces the verifier's 22 / 223 failures exactly.
3. **The report's fixes are in, and they bite.** Over 10 donor units × 223 anatomies (3345 unit-anatomy-sides whose
   follow-rule donor side exists), **66.5%** end with a placement passing every gate.
   - New gates, on the base candidate (before any repair): origin turn 79.5%, course RMS 81.7%, take-off band 96.4%,
     rise 99.9%, calibre 79.6%, tip move 99.5%.
   - v4's own gates on the same candidate: join 84.1%, clearance 55.7%.
   - The repair ladder (pose, splice, TopBrain trim, ostium advance, calibre pass 2) recovers most failures. Course
     RMS and calibre are what it cannot fix.
4. **The one-to-one assignment runs.** It uses a Hungarian solve (strict, default) or an ILP that lets two
   half-pairs share a patient (composite), and then verifies each assembled anatomy.
   - With today's 10 units for 25 TopBrain patients, strict replaces **89 RVAs and 102 LVAs** (191 of 446 VAs). That
     makes 53 anatomies v5 on both sides and 85 on one side; 85 keep v4's VAs on both sides (flagged).
   - Composite replaces 92 / 102 and completes 80 anatomies on both sides. Real-wall units alone (strict, 6 units)
     replace 53 / 54.
   - Verification changed nothing: the per-unit dry runs predict the assembled anatomies.
   - Each donor is used with exactly one TopBrain patient, so v4's overlap-free split is untouched.
5. **Registry anatomies bake.** I wrote two anatomies from the plan and baked them with `bake_v5.py`:
   - `case_m_022_left__topcow_mr_016`: ia172_061 R, a real-wall pilot donor whose stage-A surface has genus 6;
   - `case_k_008_right__topcow_mr_010_L`: ts_s0610, tube on both sides, mirrored.
   - Both passed at the first rung (margin 0.5): 1 component, genus 0, 0 centreline points outside, in 106 s and 203 s.
   - `right_route_detail` is identical to v4's on both (ratio 1.000).
6. **Take-off.** Preservation (|Δ| ≤ 15° at 10 mm) follows the donor–host subclavian elevation gap only loosely:
   64% of VAs are preserved when the gap is ≤ 15°, against 38% when it is larger (Spearman 0.38 over 30 donor-side ×
   host-side groups).
   - VMR is the worst donor here (|Δ| median 22–44°).
   - ia172_058 and ia172_015 keep their take-off (|Δ| median 2–11°).

## 1. Files

| file | what |
|---|---|
| `/sp/v5_va/graft_v5_reg_donor.py` | donor record builder (`--source ia172/song/aortaseg/vmr0248/ts --case X`, `--testing`) |
| `/sp/v5_va/graft_v5_reg.py` | the registry graft: `--unit U` dry run of one donor unit over all anatomies, `--plan P` dry run of a plan, `--plan P --out D` WRITE anatomies for `bake_v5.py`, `--compat` prototype settings |
| `/sp/v5_va/graft_v5_assign.py` | one-to-one assignment (`--solver strict` Hungarian, `composite` ILP), plan + verification |
| `/sp/v5_va/graft_v5_reg_dryall.sh` | harness: every unit in `v5_reg/donors` × 223, detached, idempotent (per-anatomy JSON; existing skipped; `FORCE=1`, `UNITS=`) |
| `/sp/v5_va/graft_v5_reg_report.py` | the test: per unit / gate statistics, examples, figure → `results/graft_reg_test.json`, `notes/graft_reg_rejected.png` |
| `/sp/v5_va/graft_reg_work/` | `regress_compat.py/.json` (the regression), `takeoff_table.py/.json`, `quick_stats.py`, `bake_test.sh`, `start_worker.sh`, `finish_seq.sh/.log`, `patch_reg1/2.py` and `patch_report1.py` (the edit record) |
| `/sp/v5_va/graft_reg_peek.py` | small JSON structure printer |
| `v5_reg/donors/<unit>/` | `record.json` (unit), `R.json` / `L.json` (sides), `R_surface.vtp` / `L_surface.vtp` (copies, md5 recorded) |
| `v5_reg/dry/<unit>/<anatomy>.json` | per-anatomy dry-run outputs (`_done.json` per unit; `_dryall.log`); `compat_vmr0248/` is the regression run |
| `v5_reg/plan_<tag>.json` | per-anatomy plans (`test_strict`, `test_composite`, `test_strict_realwall`) + `_verify_rows.json` |
| `v5_reg/write_test/` | three anatomies written from the 9-unit `plan_test_strict.json`; two baked; `_bake_*.log`, `_iso/` |
| `results/graft_reg_assign_<tag>.json` | assignment tables, score matrix, flags per anatomy |
| `results/graft_reg_test.json` | the test result (per unit, aggregate, examples, figure panels, regression, assignments) |

## 2. Donor record (`graft_v5_reg_donor.py`)

One **unit** = one donor patient (`record.json`), with up to two **side** records (`R.json`, `L.json`, schema
`v5_reg_donor_side/1`). Fields:

- **Identity.** source, source reference, licence text (and licence file for VMR), case id, patient id,
  `patient_key` for de-duplication across sets.
- **Wall class.** `real_surface` (CT-edge `ct_seg4` stage-A surface, copied into the unit folder) or `tube`
  (TotalSegmentator, `real_surface: false`, `why_tube`).
  - `real_from_mm`: the length past the ostium baked as tube. VMR L uses 40 mm (vein contact); every other donor uses 0.
- **Centreline.** The VMTK MISR line of the stage-A surface, ostium → distal, oriented by the subclavian.
  - `radii_misr` are the declared radii of real-wall donors: the inscribed radius, as for every v4 section.
  - `radii_calibre` are the declared radii of tube donors: the CT half-max equal-volume radius (`seg_qc
    r_volume_equiv_per_mm`, mapped by normalised arclength) where the pilot measured it, else MISR × 1.10 (flagged:
    the TS-skull protocol factor).
  - `cut_at_dura_mm` is set when the pilot cut the seed at the dura.
- **Ostium.** O_v is the foot of the ostium on the donor's OWN subclavian axis (projection onto the nearest segments,
  the prototype's rule). `ostium_to_axis_mm` is recorded.
- **Subclavian.** The Arterial RSA / LSA label segment nearest the ostium (≤ 6 mm), oriented medial → lateral, ±40 mm
  about O_v. It carries `t_fwd_5mm` (5 mm forward chord, lateral), `t_central`, and the elevation.
  - The ct_seg4 station path is only the fallback: its 0.25 mm stations wiggle at the branch. On 058 R the fwd-5
    elevation reads −10.6° on the stations against 17.6° on the label; the pilot's gross.json reads 17.8°.
  - For TotalSegmentator subjects without labels (s0960) the fallback is the TS subclavian mask (2 mm principal-axis
    bins, the pilot's `_sa_path` rule). It needs nibabel, so it runs in neve-build-stagea.
  - VMR uses the expert SimVascular path, as the prototype did.
- **Landmarks.** C6 / C2 / C1 / dura / end along the centreline.
  - Thin-slice donors use the pilot's `landmarks_rec` on the ct_seg4 line: Erec C6, ts_qc C2, C1f, ts_qc dura.
    AortaSeg falls back to the `ct_seg4_stageA/Erec` set from the pilot record.
  - VMR uses `landmarks_ct`, with the verifier's note on the left C6.
  - TotalSegmentator uses ts_qc on the trace, mapped onto the VMTK line (C2 / C1 flagged as uncertain).
- **Take-off (native).** Measured on the prep_siphon 1 mm line the graft places: the VA chord from the ostium to 5 / 10
  mm, against `t_fwd_5mm`. Also recorded: the elevation of the 10 mm chord and the subclavian's elevation.
- **QC.** verdict pass / review / fail from the pilot record (`status` donor / donor_review / fail; for AortaSeg its
  `cervical_verdict`), with failed rules, review flags, the Song `select.json` choice, and any later `pilot_decide.json`
  entry. VMR counts as a reference pass. TotalSegmentator sides use the recalibrated Stage-2 list in
  `tag_calibration.json`.
  - "usable" = verdict pass or review, and chirality +1.
- **Coarseness.** Whole-VA, C6 → C1 and origin → C1 union > 0.3 mm (k64), the plain-tube 10 mm bin fraction (< 0.01),
  the edge FWHM, and `clean_union_0.3`: the area-weighted 10 mm bins with contaminated bins removed.
  - `score` = clean union, else origin → C1. It is the assignment's coarseness term; tube donors score 0.
- **Contamination zones.** 10 mm bins of the pilot's `edge.json` with ≥ 15 non-bone contact rays (other artery / vein
  / unlabelled bright) or ≥ 5% dip-before-bright rays, merged. Also AortaSeg `attrib.json` stretches where
  bright-neighbour contacts carry ≥ 30% of the relief, and VMR L s 10–40 mm (vein).
  - Bone contacts are not counted: every V2 touches bone.
- **Frame.** RAS, with a chirality sign from the case's two traces (R − L at a common height × anterior · up). All
  records read +1.
- **Builder.** Version, time, and the mtime of every input, so a record made stale by a later pilot change can be
  detected.

Built records (test set):

| unit | side | QC | wall | length mm | C2 / dura mm | take-off 10 mm native | subclavian elev. | coarseness score (whole union) | surface genus |
|---|---|---|---|---|---|---|---|---|---|
| vmr0248 | R | pass | real | 220 | 112 / 183 | 53.5° | 63.9° | 0.041 (0.045) | 0 (union) |
| vmr0248 | L | pass | real (V1 tube 40 mm) | 225 | 115 / 189 | 58.7° | 36.3° | 0.078 (0.138) | 0 (union) |
| ia172_058 | R | pass | real | 232 | 131 / 212 | 81.0° | 17.6° | 0.049 (0.075) | 2 |
| ia172_061 | R | pass | real | 215 | 113 / 190 | 60.7° | −0.6° | 0.148 (0.245) | 6 |
| ia172_141 | L | pass | real | 212 | 122 / 192 | 50.6° | 37.8° | 0.011 (0.038) | 1 |
| ia172_015 | L | pass | real | 242 | 136 / 216 | 75.5° | 32.3° | 0.084 (0.171) | 5 |
| aortaseg_Dissec_01 | L | review | real | 244 | 128 / 208 | 65.1° | 25.1° | 0.260 (0.387) | 4 |
| ts_s0610 | R / L | pass / review | tube | 221 / 206 | 120 / 203, 108 / 188 | 64.1° / 103.7° | 38.3° / −28.7° | 0 | – |
| ts_s0777 | R / L | review / review | tube | 224 / 252 | 119 / 190, 133 / 219 | 49.8° / 63.6° | 52.3° / 57.7° | 0 | – |
| ts_s1786 | R / L | review / pass | tube | 247 / 245 | 142 / 231, 143 / 231 | 45.2° / 82.7° | 46.1° / 39.3° | 0 | – |
| ts_s0960 | R / L | pass / review | tube (TS-mask subclavian) | 224 / 224 | 128 / 207, 128 / 205 | 85.3° / 65.0° | 25.1° / 25.1° | 0 | – |

Notes on the records:
- ia172_058 was built from the fix-8 `ct_seg4` (files 01:12–01:18, before the build). Rebuild every record once the
  pilot's decide step has run: `graft_v5_reg_donor.py --testing` takes about 10 s (s0960 in neve-build-stagea).
- Dissec_01 L is a dissection patient with a dominant left VA (r 2.4 mm MISR). 18 of its 24 bins are flagged as
  contaminated (bright neighbours and other arteries), and its clean union is still 0.26. Its relief needs a visual
  review before it counts as nooks.
- ts_s0960's subclavians come from 2 mm principal-axis bins of the TS mask (nearly straight). Its subclavian
  elevations (25.1° on both sides; equal to 0.1° by coincidence, the chords differ in x / y) and its native take-offs
  are therefore coarse.
- The contamination rule is conservative (dip-before-bright also fires at bone), so the clean scores are lower bounds.
  It is a ranking term only and changes no geometry.

## 3. The graft (`graft_v5_reg.py`)

The prototype's pieces are imported unchanged and used as they are:
- `base_pose`, `poses`, `compose`, `graft_splice` / `tb_rotation` (TB fitted, v4 ramp);
- `side_setup`, `others_for`, `gate_va`, `Neighbours`, `corridor_misfit`, `forward_chord`, `d_chord`;
- `graft_v4.join_kink` and `graft_v4.tb_part`.

New code is limited to: the donor loader, the handedness rule, the new gates, the search loop that applies them, the
records, and `write`. `--compat` switches the new code off; the regression above shows nothing else changed.

**Handedness: follow (default).** A native anatomy takes the donor's R VA as its RVA and the donor's L VA as its LVA.
A mirrored `_L` anatomy takes the donor's L VA, x-mirrored, as its RVA, and the donor's R VA, mirrored, as its LVA. The
TopBrain piece keeps v4's own rule. Every rotation is checked proper (det +1); all are.

**Gates.** Every v4 gate is kept unchanged:
- 0.6 mm clearance from the parent exit on, against every other branch (including the stub and the ramps), the parent
  after exit, the other VA (as finally written) and itself; exact all-pairs re-check in the final pass;
- exit from the parent ≤ 12 mm with no re-entry;
- splice join ≤ 60°;
- TopBrain blob cap and floor.

New gates and repairs:

| gate | rule | why / evidence |
|---|---|---|
| origin turn | largest 1 mm step turn within ±3 mm of O_h AND of the placed ostium ≤ 60° | v4's join rule; ungated in the prototype (22 / 223 > 60°). The ostium turn is the donor's own and no pose changes it (Dissec_01 L: 68–70° in every pose), hence the repair below |
| course RMS | course RMS to the v4 VA corridor (21 matched normalised-arclength points, origins pinned) ≤ 20 mm | report: corridor fits median 9.6 / 9.4 mm (max 30), absurd frame placements ≥ 21.5 mm |
| tip move | VA blind end moved vs v4's ≤ 60 mm | prototype corridor fits max 53 / 58; frame placements median 118 |
| take-off band | take-off at 10 mm in [20°, 120°] | prototype 38–107°; donors 45–104° |
| rise | 10 mm chord elevation ≥ 15° | prototype ≥ 43°; "never dips below its branch point" |
| splice in V3 | splice on the donor within [C2, dura] (landmarks), else the prototype's [100, L − 10] | brief: "donor VA from its origin to a splice in V3" |
| splice calibre | q = TopBrain join radius / donor radius 4 mm before the splice (as graft_splice floors and caps them). Pass 1: max(q, 1/q) ≤ 1.25. Pass 2, only if pass 1 finds nothing: ≤ 1.6, flagged (`splice_calibre_preferred: false`) | report: ratio up to 2.3; the ramp widened the donor's last 25 mm up to 2× |
| host parent tangent | 5 mm forward chord of the host subclavian from O_h | report: the prototype's 5-point chord spanned 8.7 / 7.1 mm |
| take-off record | native and placed at 5 / 10 mm, Δ, `takeoff_not_preserved` (\|Δ\| > 15°), parent-tangent mismatch, both elevations | report |
| repair: ostium advance | only while the origin turn fails AT THE OSTIUM: start the donor 1–5 mm later (v4's own "a donor that opens on a turn: advance its start", `DONOR_TRIM_STEPS_MM`) | recorded as `ostium_advance_mm`; used only by Dissec_01 (1 mm, in every placement it has) |

**Search order (per VA).**
1. Pose: base Kabsch, then ±3…15° about x / y / z.
2. Pose-level gates: rotation, course RMS, take-off band, rise.
3. Splice: v4's ±5…30 mm moves inside the window.
4. TopBrain trim 0–5 mm, for the calibre and join rules.
5. Origin turn (on failure, next pose).
6. Join.
7. Tip move.
8. Clearance.

Calibre pass 2 and the ostium advance wrap this loop. The right VA is placed first, against every branch but both VAs
(plus v4's LVA if the left stays v4). The left is then placed against the placed right, or v4's RVA. A final pass
re-gates both VAs as written.

**Records per side** (`v5_va` in provenance, and the dry JSON):
- donor id / side / mirror, wall class, QC verdict, coarseness;
- pose, splice, trim, advance, window, calibre ratio and pass, origin turns, join, ramp;
- take-off block, course RMS, tip moves, rotation det;
- first-fail counts with the first rejections;
- the BASE candidate with every gate evaluated (pass rates per gate), plus its geometry (every 2nd point) for figures.

**Write mode** (`--plan P --out D`) works like the prototype:
- copies the v4 anatomy and writes new RVA / LVA curves; a side the plan leaves null keeps v4's curve and xform;
- fills `xform.vaC_R` / `vaC_L`: the donor surface copy and its md5, the record path, `lo_mm` = ostium + `real_from`,
  `hi_mm` = splice, and `real_surface` (false for tube donors, with `why_tube`, so `bake_v5.sections_for_v5` bakes them
  as tube);
- updates `xform.va_R` / `va_L` with v4's cut kept;
- writes `provenance.v5_va` (gates, licences) and `LICENSE_v5_donors.txt`, and copies the VMR licence file when VMR is
  used.

A bake driver therefore needs only: `graft_v5_reg.py --plan v5_reg/plan_<tag>.json --out <dir>`, then
`bake_v5.py --anatomies <dir>`.

## 4. Assignment (`graft_v5_assign.py`)

- **Score.** W(u, P) = 10⁴ · VAs passing + 100 · Σ coarseness of the donor sides used + 0.1 · take-offs kept.
  - The sum runs over every anatomy of TopBrain patient P and both host sides, with the donor side taken by the
    follow rule.
  - Gate passes therefore come strictly first. Coarseness comes second (tube 0), with take-off as a weak tie-break.
- **Strict solver.** Hungarian (`linear_sum_assignment`, maximise) on units × patients; pairs with 0 passes are dropped.
- **Composite solver.** `milp`: each unit to at most one patient, and each patient takes at most one donor-R supplier
  and one donor-L supplier. This lets a right-only donor and a left-only donor share a patient. It is still
  split-safe; scores are per-side additive.
- **Plan.** Per anatomy and host side: the assigned donor side if its dry run passed, else null, flagged with the reason
  ("no donor X VA assigned to P" or "donor U fails (first fails …)").
- **Verification.** Every anatomy is re-run with its whole plan; a failing side is dropped to v4 and the anatomy re-run,
  until stable (≤ 3 rounds).
- **Split safety.** Asserted: each unit goes to exactly one patient, with at most one donor-R and one donor-L supplier
  per patient.

Results with the 10 test units:

| variant | units → patients | RVAs / LVAs replaced | anatomies v5 on both sides / one side / none | after verify |
|---|---|---|---|---|
| strict (default) | 10 → 10 | 89 / 102 | 53 / 85 / 85 | unchanged |
| composite | 10 → 8 (two shared patients) | 92 / 102 | 80 / 34 / 109 | unchanged |
| strict, real-wall only (6 units) | 6 → 6 | 53 / 54 | 16 / 75 / 132 | unchanged |

(With 9 units, before ts_s0960: strict 76 / 86, composite 77 / 87.)

Strict assignment (unit → patient, anatomies, VAs passing / candidates):
- vmr0248 → topcow_mr_020: 16 anatomies, 32 / 32
- ts_s0960 → topcow_mr_016: 16, 32 / 32
- ts_s0610 → topcow_mr_010: 15, 26 / 30
- ts_s0777 → topcow_mr_023: 15, 24 / 30
- ia172_015 → topcow_mr_011: 16, 16 / 16
- ia172_058 → topcow_mr_005: 16, 15 / 16
- ia172_061 → topcow_mr_024: 15, 15 / 15
- ia172_141 → topcow_mr_026: 15, 15 / 15
- aortaseg_Dissec_01 → topcow_mr_001: 12, 11 / 12
- ts_s1786 → topcow_mr_027: 9, 5 / 18

The objective pushes donors onto the large patients (15–16 anatomies). Adding one unit reshuffles the optimum: with
s0960, 061 moved from topcow_mr_016 to _024 and Dissec_01 from _005 to _001.

The 15 patients left without a unit, and the sides a half-pair cannot supply, keep v4's VA (flagged in `plan_*.json`
and in `flags_per_anatomy`).

## 5. The test (harness over 223 × 10 units)

**Run time.**
- Dry run: 9–34 s per unit for 223 anatomies on 6 CPUs (about 3 min for all 10). Per anatomy 0.1–1 s; failing sides
  cost the most (up to ~700 candidates).
- Assignment with verification: about 20–24 s.
- Bake: 106 s (real-wall donor) and 203 s (tube donors; v4's bake of that anatomy took 137 s).

**Per unit** (`results/graft_reg_test.json` → `per_unit`). "Base-all" counts the anatomies whose base candidate passed
everything before any repair; "dominant failing gate" is the gate with the most first-fails among the sides that end
with no placement.

| unit | side | pass / candidates | base-all | dominant failing gate | \|Δ take-off\| median |
|---|---|---|---|---|---|
| vmr0248 | R | 199 / 223 | 83 | course RMS 11, calibre 6, origin 5, clearance 2 | 37.6° |
| vmr0248 | L | 217 / 223 | 98 | course RMS 3, calibre 2, join 1 | 30.8° |
| ia172_058 | R | 79 / 103 | 40 | course RMS 24 | 4.4° |
| ia172_058 | L | 88 / 120 | 46 | course RMS 27, calibre 5 | 5.0° |
| ia172_061 | R | 46 / 103 | 15 | calibre 36, course 8, clearance 8, origin 5 | 22.5° |
| ia172_061 | L | 106 / 120 | 30 | calibre 10, clearance 4 | 7.8° |
| ia172_141 | R | 94 / 120 | 26 | course 15, calibre 9, join 2 | 19.2° |
| ia172_141 | L | 98 / 103 | 41 | course 2, calibre 2, join 1 | 22.2° |
| ia172_015 | R | 87 / 120 | 24 | course 32, calibre 1 | 2.0° |
| ia172_015 | L | 83 / 103 | 37 | course 11, join 6, calibre 3 | 10.8° |
| aortaseg_Dissec_01 | R | 35 / 120 | 0 | origin turn 53, calibre 30 | 19.0° |
| aortaseg_Dissec_01 | L | 101 / 103 | 0 | calibre 2 | 7.7° |
| ts_s0610 | R | 128 / 223 | 48 | origin turn 54, calibre 25, course 7, join 6 | 4.8° |
| ts_s0610 | L | 208 / 223 | 117 | calibre 8, course 4, join 3 | 13.9° |
| ts_s0777 | R | 51 / 223 | 3 | calibre 77, course 74, take-off band 13, origin 8 | 10.0° |
| ts_s0777 | L | 173 / 223 | 29 | calibre 28, course 16, take-off band 3 | 49.5° |
| ts_s1786 | R | 12 / 223 | 1 | course RMS 168, calibre 25, origin 13 | 14.3° |
| ts_s1786 | L | 50 / 223 | 31 | course RMS 168 | 7.3° |
| ts_s0960 | R | 173 / 223 | 75 | course RMS 42, origin 6 | 4.9–9.4° |
| ts_s0960 | L | 196 / 223 | 93 | course RMS 18, clearance 7 | 16.8–18.3° |

The candidate counts differ by side because of the follow rule: a one-sided donor serves the native anatomies on that
side and the mirrored anatomies on the other (103 or 120 of 223). Where a range is given, it is over the native and
mirrored donor sides.

**Per gate** (aggregate over 3345 unit-anatomy-sides; the base candidate before repairs, then final):

| gate | base pass rate |
|---|---|
| rotation proper | 1.000 |
| origin turn ≤ 60° | 0.795 |
| course RMS ≤ 20 mm | 0.817 |
| take-off band 20–120° | 0.964 |
| rise ≥ 15° | 0.999 |
| splice in V3 | 1.000 |
| calibre ≤ 1.6 | 0.796 |
| join ≤ 60° (v4) | 0.841 |
| tip move ≤ 60 mm | 0.995 |
| clearance 0.6 mm incl. exit (v4) | 0.557 |
| **final, after repairs** | **0.665** |

What the repairs could and could not fix:
- **Clearance and join** (v4's gates) fail on the base candidate in 44% / 16% of cases. The ladder almost always fixes
  them: they dominate only 1–8 failing sides per unit.
- **Course RMS** is the main blocker for donors whose proportions differ from the host corridor. ts_s1786's VAs are
  245 mm long, against 208–238 mm host corridors, and fail in 168 / 223 on each side.
  - The RMS is taken at normalised arclength, so part of it is length mismatch rather than course. An arclength-matched
    variant was not tested (open item).
- **Calibre** blocks donors whose MISR is far from the TopBrain piece's at the splice: 061 R (36 failing sides) and
  s0777 (77 R).
- **The origin turn at O_h** (the host side) is mostly fixed by the pose search. At the ostium (the donor side) only the
  advance can fix it: Dissec_01 needs 1 mm, which takes its R / L from 0 / 0 to 35 / 101.

**Take-off** (`graft_reg_work/takeoff_table.json`). Host subclavian elevation on the 5 mm chord is 23.4–24.0° (R) and
13.2° (L).
- Kept (median \|Δ\| ≤ 5.5°): ia172_058 (4.4 / 5.0°, gap 4–6°); ia172_015 as RVA (2.0°, gap 8°); ts_s0960 R as RVA
  (4.9°, gap 1.5°).
- ts_s0610 keeps it too (4.8–5.5° in three of its four groups) despite gaps of 15–53°: one reason the gap predicts
  only weakly.
- Lost: VMR (\|Δ\| 22–44°, gaps 12–51°), ia172_141 (19–22°, gaps 14–25°), ts_s0777 L (45–50°, gaps 39–45°).
- Across the 30 donor-side × host-side groups: Spearman(gap, median \|Δ\|) = 0.38. 64% of VAs are preserved when the gap
  is ≤ 15°, against 38% above.
- The elevation gap is a useful but weak selection feature, as the report expected. Every placement records
  `takeoff_not_preserved`.

**Rejected placements (figure).** `notes/graft_reg_rejected.png` shows, for each gate, the most extreme rejected base
placement that no repair rescued (red), over v4's VA corridor (blue) and the host (grey), in coronal and sagittal views:
- origin turn 91.9° (ts_s0777 R in `case_w_012_left__topcow_mr_021_L`);
- course RMS 45.6 mm (ts_s1786 R in `case_w_048_left__topcow_mr_001`: the VA runs ~60 mm past v4's blind end);
- calibre 2.4 (ts_s0777 L);
- take-off 136.8° (ts_s0777 L);
- join 82.1° (ts_s0960 R);
- clearance −5.4 mm (ts_s0960 L, through a neighbour).

Every repaired placement's base candidate is also in the dry JSON (`sides.<side>.base`) if a before / after panel is
wanted.

![rejected placements](graft_reg_rejected.png)

## 6. Write and bake test

`graft_v5_reg.py --plan plan_test_strict.json --out v5_reg/write_test` (the 9-unit plan) wrote three anatomies:
- `case_m_022_left__topcow_mr_016`: ia172_061 R as the RVA, a real wall with a genus-6 surface; the LVA is v4's;
- `case_k_008_right__topcow_mr_010_L`: ts_s0610 L mirrored as the RVA and R mirrored as the LVA, both tube;
- `case_w_008_right__topcow_mr_020`: the VMR pair (written, not baked).

`bake_v5.py` (bake_meshes_v4 unchanged) on `case_m_022_left__topcow_mr_016`:
- pass at the first rung (margin 0.5): obj 23,498 triangles, 1 component, genus 0;
- 0 centreline points outside, nothing suppressed, 106 s;
- vaC_R section: 205k voxels, 7.0k refused (not owned);
- `right_route_detail` identical to v4's (fork 0.268 / 0.1445, siphon 0.3176 / 0.1577: ratio 1.000);
- RCCA / LCCA lumen 0.896 / 0.841 mm (v4 0.887 / 0.838).

The handles in the pilot's stage-A surface did not reach the owned union.

`case_k_008_right__topcow_mr_010_L` (ts_s0610 tube, mirrored, both VAs):
- pass at the first rung: 1 component, genus 0, 0 outside, nothing suppressed, 203 s (v4 137 s);
- both vaC spans read "tube only" (no real section), as `bake_v5.sections_for_v5` intends for `real_surface: false`;
- `right_route_detail` identical to v4's; RCCA / LCCA lumen 0.767 / 1.364 mm, equal to v4's.

## 7. Open issues

- **Pool size.** Today's 10 units cover 10 of 25 TopBrain patients; 25 pairs are needed. 5 of the 6 real-wall units
  are half-pairs. The composite solver uses two of them on one patient; whether that is acceptable under "one donor PAIR
  per patient" is the user's call. Strict is the default.
- **Course RMS.** Normalised-arclength RMS penalises long or short donors (ts_s1786). Consider an arclength-matched RMS
  or a length-scaled bar before scaling.
- **Contamination and wall.** The coarseness score's contamination rule is a heuristic. Dissec_01 L's relief (0.39
  whole, 0.26 clean) sits largely on bright-neighbour contacts in a dissection patient and needs a visual review before
  it ranks first.
- **Surface genus.** Pilot stage-A surfaces carry handles (genus 1–6). Two bakes were clean, but every assigned donor
  should be baked before trusting the plan; the bake ladder may suppress stretches.
- **Splice calibre.** The 1.25 / 1.6 bars are my reading of the report's "prefer within about 1.25 / cap the ratio";
  they were not tuned. Pass-2 placements are flagged.
- **Ostium advance.** It replaces up to 5 mm of real VA with the straight stub. Only Dissec_01 uses it (1 mm).
- **Inputs may change.** The pilot is still grading. Records carry their inputs' mtimes; rebuild them (`--testing` or
  per case) and rerun `graft_v5_reg_dryall.sh` with `FORCE=1` (about 3 min), then the assignment.
- **TS-mask subclavian (s0960).** It is a straight principal-axis path, so the take-off and elevation are approximate.
- **Not done here.** No SOFA load test of the written anatomies; no wall-detail measure (`va_detail_v5.py`) of the
  baked registry anatomies. Both are bake-driver steps.

## Session record

- 01:1x–01:30: read the report sections, the prototype, graft_v4 and the pilot outputs; built the donor builder and
  records.
- 01:30: graft_v5_reg.py; compat regression (0 differences).
- 01:31–01:34: first dry run of all units.
- ~01:40: added the ostium-advance repair (compat still 0 differences); rerun 01:36–01:39 (log times).
- Then: assignment (3 variants), the report with its figure, write mode, the first bake test (106 s, pass).
- 01:46–01:52: closing sequence `graft_reg_work/finish_seq.sh`: ts_s0960 record in stagea, tube bake, dry ts_s0960,
  rerun of the assignments and the report.
